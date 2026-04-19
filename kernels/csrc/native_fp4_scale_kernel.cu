// SPDX-License-Identifier: Apache-2.0
//
// Native FP4 scale quantization kernel for MoE path.
//
// Isolates the per-block scale factor computation from CUTLASS's
// `scaled_fp4_experts_quant` so we can benchmark it independently and
// experiment with PTX-level optimizations to the FP8 round-trip.
//
// Input:  [M, K] BF16/FP16 activations + [1] FP32 global scale
// Output: [M, K/2] packed FP4-E2M1 + [rounded_m, rounded_n/4] swizzled FP8-E4M3
//         scale factors (same layout as CUTLASS output_scales.view(fp8_e4m3)).
//
// Uses on SM100+/SM120a:
//   - cvt.rn.satfinite.e2m1x2.f32  (FP4 data, 2 fp32 → 1 byte)
//   - cvt.rn.satfinite.e4m3x2.f32  (FP8 scale, 2 fp32 → 1 .b16)
//   - cvt.rn.f16x2.e4m3x2          (FP8 → FP16 readback)
//
// Single-expert (degenerate) shape: we process a dense [M, K] tensor,
// equivalent to scaled_fp4_experts_quant with expert_offsets=[0, M].

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace vllm_nfs {

static constexpr int FP4_BLOCK_SIZE = 16;          // elements per SF
static constexpr int ELTS_PER_THREAD = 8;          // one STG.32 of packed FP4
static constexpr int THREADS_PER_SF = 2;           // two threads cooperate per SF (16 elts / 8)
static constexpr float FP4_E2M1_MAX = 6.0f;

// -------- PTX-level native conversions --------

// 2 fp32 -> 1 byte packed E2M1 (as a .b8). Stored into a byte of a .b32 by caller.
__device__ __forceinline__ uint32_t cvt_f32x2_to_e2m1(float a, float b) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
      "cvt.u32.u8 %0, byte0;\n"
      "}\n"
      : "=r"(val) : "f"(a), "f"(b));
  return val & 0xFFu;
}

// 8 fp32 -> 1 uint32 (4 bytes of packed FP4-E2M1).
__device__ __forceinline__ uint32_t cvt_f32x8_to_e2m1(const float v[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 b0;\n"
      ".reg .b8 b1;\n"
      ".reg .b8 b2;\n"
      ".reg .b8 b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}\n"
      : "=r"(val)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
        "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return val;
}

// Native fp32 -> fp8 E4M3 (single value). Uses the packed x2 PTX with the
// second operand = 0, then keeps the low byte. Saturating to [0, 448].
__device__ __forceinline__ uint8_t cvt_f32_to_e4m3_sat(float x) {
  unsigned short storage;
  asm volatile(
      "{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n"
      : "=h"(storage) : "f"(x), "f"(0.0f));
  return static_cast<uint8_t>(storage & 0xFFu);
}

// fp8 E4M3 byte -> float (exact via fp8->fp16->fp32 path, native PTX).
__device__ __forceinline__ float cvt_e4m3_to_f32(uint8_t byte) {
  unsigned short fp8pair = static_cast<unsigned short>(byte);
  unsigned int half2_storage;
  asm volatile(
      "{cvt.rn.f16x2.e4m3x2 %0, %1;}\n"
      : "=r"(half2_storage) : "h"(fp8pair));
  __half h = __short_as_half(static_cast<short>(half2_storage & 0xFFFFu));
  return __half2float(h);
}

__device__ __forceinline__ float rcp_approx(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// -------- swizzled SF layout [numMTiles, numKTiles, 32, 4, 4] --------
__device__ __forceinline__ int64_t swizzled_sf_offset(
    int row, int sf_col, int numKTiles) {
  int mTileIdx  = row >> 7;           // row / 128
  int outerM    = row & 31;           // row % 32
  int innerM    = (row >> 5) & 3;     // (row / 32) % 4
  int kTileIdx  = sf_col >> 2;        // sf_col / 4
  int innerK    = sf_col & 3;         // sf_col % 4

  return ((static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9)
       | (outerM << 4) | (innerM << 2) | innerK;
}

// -------- main kernel --------
//
// One thread converts 8 elements → 1 packed uint32 of FP4.
// Two adjacent threads (THREADS_PER_SF=2) cooperate on 16 elements → 1 SF byte.
// Grid is launched so that threadIdx.x iterates over (k / 8) blocks within a row,
// and blockIdx iterates over rows.
template <typename scalar_t>
__global__ void native_fp4_scale_quant_kernel(
    uint32_t* __restrict__ out_fp4,       // [M, K/8] as uint32 (K/2 bytes)
    uint8_t*  __restrict__ out_sf,        // swizzled FP8 scale factors
    const scalar_t* __restrict__ input,   // [M, K] bf16/fp16
    const float* __restrict__ global_scale_ptr,
    int M, int K, int numKTiles) {

  const int row = blockIdx.y;
  if (row >= M) return;

  const int col8 = blockIdx.x * blockDim.x + threadIdx.x;    // element pack index (8 elts)
  const int colsPer8 = K / ELTS_PER_THREAD;
  if (col8 >= colsPer8) return;

  const float global_scale = *global_scale_ptr;

  // Each thread loads 8 elements.
  const int col_start = col8 * ELTS_PER_THREAD;
  float x_f[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    scalar_t v = input[row * K + col_start + i];
    if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
      x_f[i] = __bfloat162float(v);
    } else {
      x_f[i] = __half2float(v);
    }
  }

  // Compute per-thread abs max over 8 elements.
  float local_amax = 0.f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    local_amax = fmaxf(local_amax, fabsf(x_f[i]));
  }

  // Pair across threads for 16-element SF block: XOR lane 1 with lane 0.
  // All threads within a warp; pairs are (even, odd) along threadIdx.x.
  float pair_amax = fmaxf(local_amax, __shfl_xor_sync(0xFFFFFFFFu, local_amax, 1));

  // SF = global_scale * (amax / 6.0)
  float sf_val = global_scale * (pair_amax * rcp_approx(FP4_E2M1_MAX));

  // Native fp32 -> fp8 E4M3 saturating. Match CUTLASS semantics (__nv_fp8_e4m3).
  uint8_t sf_byte = cvt_f32_to_e4m3_sat(sf_val);
  // Readback for exact-consistent output scale.
  float sf_readback = cvt_e4m3_to_f32(sf_byte);

  // Output scale: rcp(sf_readback / global_scale) = global_scale / sf_readback.
  float out_scale = (sf_readback != 0.f)
      ? rcp_approx(sf_readback * rcp_approx(global_scale))
      : 0.f;

  // Quantize 8 elts to FP4 (one uint32).
  float scaled[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) scaled[i] = x_f[i] * out_scale;
  uint32_t packed = cvt_f32x8_to_e2m1(scaled);

  // Store 8 FP4 values (4 bytes) at [row, col8].
  out_fp4[row * colsPer8 + col8] = packed;

  // Only the even thread in each pair writes the SF byte (matches CUTLASS).
  if ((threadIdx.x & 1) == 0) {
    int sf_col = col8 / THREADS_PER_SF;   // one SF per 16 elts (two col8's)
    int64_t off = swizzled_sf_offset(row, sf_col, numKTiles);
    out_sf[off] = sf_byte;
  }
}

}  // namespace vllm_nfs

// -------- entry point --------
void native_fp4_scale_quant(
    torch::Tensor& out_fp4,           // [M, K/2] uint8 packed
    torch::Tensor& out_sf,            // [rounded_m, rounded_n_div4] int32 (viewed as fp8_e4m3)
    torch::Tensor const& input,       // [M, K] bf16 or fp16
    torch::Tensor const& input_global_scale /* [1] float */) {

  TORCH_CHECK(out_fp4.is_contiguous() && out_sf.is_contiguous());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kHalf);

  const int M = input.size(0);
  const int K = input.size(1);
  TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");

  const int numKTiles = (K + 63) / 64;

  // Threads per block: tune to have enough to fill a warp for shfl_xor.
  // Use 256 threads, launch (ceil(K/8/256), M) blocks.
  const int threads = 256;
  const int blocks_x = (K / vllm_nfs::ELTS_PER_THREAD + threads - 1) / threads;
  dim3 grid(blocks_x, M);
  dim3 block(threads);

  const at::cuda::OptionalCUDAGuard dg(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::kBFloat16) {
    vllm_nfs::native_fp4_scale_quant_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        reinterpret_cast<uint32_t*>(out_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(out_sf.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const float*>(input_global_scale.data_ptr()),
        M, K, numKTiles);
  } else {
    vllm_nfs::native_fp4_scale_quant_kernel<__half><<<grid, block, 0, stream>>>(
        reinterpret_cast<uint32_t*>(out_fp4.data_ptr()),
        reinterpret_cast<uint8_t*>(out_sf.data_ptr()),
        reinterpret_cast<const __half*>(input.data_ptr()),
        reinterpret_cast<const float*>(input_global_scale.data_ptr()),
        M, K, numKTiles);
  }
}
