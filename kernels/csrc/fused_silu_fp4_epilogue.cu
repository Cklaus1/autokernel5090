// SPDX-License-Identifier: Apache-2.0
//
// ============================================================
// T2-N v3: Fused SiLU+Mul + NVFP4 quant epilogue
// ============================================================
// Tag: W6_T2N_silu_epilogue
//
// This kernel is the drop-in replacement for vLLM's
//   ops.silu_and_mul_scaled_fp4_experts_quant(c1, a2_gscale,
//       expert_offsets, blockscale_offsets, num_topk)
// in the CUTLASS MoE FP4 path.
//
// Plan ref: plans/t2n_ceiling_analysis.md §Rank 1 (+4% projected, ~930
// tok/s on top of banked 23,254 peak).
//
// ------------------------------------------------------------
// KILL_PATTERNS entries that apply:
//   §1 calibration constants — SM120 barrier cost = ~3µs; native PTX
//     cvt.rn.satfinite.e2m1x2.f32 is encode-only (fp32→fp4×2).
//   §2 P1  (silent-None dispatch) — python caller asserts kernel loaded,
//     and logs fallback class name.
//   §2 P2  (plugin banner vs fusion active) — wrapper logs
//     [T2N-SILU-EPILOGUE] active expert_count=N topk=K on first call.
//   §2 P7  (single-shape KILL) — tested at Qwen3 (K=2048, M=4096) AND
//     Gemma4 (K=7168, M=256) regimes before any KILL claim.
//   §2 P11 (PROCEED category) — this is Category 1 (specific code hook
//     bug-fix-like) + Category 2 (recalibration).  Cross-apply is
//     regime-matched because the fused kernel uses identical PTX path
//     for both K values.
//
// ------------------------------------------------------------
// Silicon assumptions:
//   * SM120a (RTX PRO 6000 / 5090).  Uses cvt.rn.satfinite.e2m1x2.f32
//     for FP4 conversion (available on CUDA >= 12.8, __CUDA_ARCH__>=1000).
//   * BF16 activation path is the hot one for Qwen3-30B-A3B.  FP16 path
//     kept for symmetry with sister kernel.
//   * No WGMMA / no FP8 MMA / no native FP4 MMA — conversion happens via
//     scalar PTX inside the warp (encoder only).
//   * Scale layout matches CUTLASS 128x4 swizzle used by
//     cutlass_fp4_moe_mm for GEMM2 input.  Same formula as
//     fused_shuffle_quant.cu:compute_cutlass_sf_offset (line 90-102).
//
// ------------------------------------------------------------
// What it fuses:
//   c1 = GEMM1 output, BF16 [m*topk, 2N], gate || up layout.
//   For each row r and each 16-element column block b:
//     gate[j] = c1[r, b*16 + j]                       (0 <= j < 16)
//     up  [j] = c1[r, N + b*16 + j]
//     act [j] = silu(gate[j]) * up[j] = (gate/(1+exp(-gate))) * up[j]
//     max_abs = max(|act[j]|)
//     sf  = a2_gscale[expert] * (max_abs / 6.0)          (FP8-E4M3)
//     out_scale = 1 / (sf_readback / a2_gscale)
//     code[j] = cvt.rn.satfinite.e2m1x2.f32(act[j] * out_scale)
//     write packed FP4 bytes + swizzled scale byte.
//
// Drop-in semantics: shape of `output_scales` matches vLLM's int32
// buffer shape `[MAX_TOKENS_PER_EXPERT * topk, padded_k_int32]` so the
// downstream CUTLASS GEMM2 blockscale_offsets are valid unchanged.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// ============================================================
// FP4-E2M1 constants
// ============================================================
static constexpr float FP4_E2M1_MAX = 6.0f;
static constexpr int   FP4_BLOCK_SIZE = 16;

// ============================================================
// Hardware FP4 conversion (PTX) with software fallback
// ============================================================
__device__ __forceinline__ uint32_t fp32x8_to_e2m1(const float v[8]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
        "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  return val;
#else
  // Boundary-based software fallback (matches fused_shuffle_quant.cu).
  uint32_t packed = 0;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    float x = v[i];
    int sign = (x < 0.0f) ? 1 : 0;
    float ax = fabsf(x);
    ax = fminf(ax, FP4_E2M1_MAX);
    int code = (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) +
               (ax > 1.75f) + (ax > 2.5f)  + (ax > 3.5f)  + (ax > 5.0f);
    int nibble = (sign << 3) | code;
    int byte_idx = i / 2;
    int nibble_pos = (i % 2) * 4;
    packed |= ((nibble & 0xF) << (byte_idx * 8 + nibble_pos));
  }
  return packed;
#endif
}

// Fast approximate reciprocal
__device__ __forceinline__ float rcp_approx(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

// SiLU = x / (1 + exp(-x)) = x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
  return x * rcp_approx(1.0f + __expf(-x));
}

// ============================================================
// CUTLASS 128x4 swizzle (same formula as fused_shuffle_quant.cu:72)
// ============================================================
__device__ __forceinline__ int64_t compute_cutlass_sf_offset(
    int row_in_buf, int block_idx, int numKTiles) {
  int mTileIdx  = row_in_buf >> 7;
  int outerMIdx = row_in_buf & 31;
  int innerMIdx = (row_in_buf >> 5) & 3;
  int kTileIdx  = block_idx >> 2;
  int innerKIdx = block_idx & 3;
  return ((static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9)
         | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;
}

// Explicit conversion helpers — the build flags
// -D__CUDA_NO_BFLOAT16_CONVERSIONS__ / -D__CUDA_NO_HALF_CONVERSIONS__
// disable implicit static_cast so we route through the intrinsics.
__device__ __forceinline__ float _to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ float _to_float(__half x) {
  return __half2float(x);
}

// ============================================================
// Main kernel: one thread block per output row.
// Each thread strides over the K/16 output blocks.
//
// The kernel handles per-expert scaling by linear-scanning the
// expert_offsets (E<=128), identical to fused_shuffle_quant.cu.
// This runs once per block (one row) and uses a single shared-memory
// slot to broadcast the expert_id to all threads in the block.
// ============================================================
template <typename scalar_t>
__global__ void fused_silu_fp4_quant_kernel(
    uint8_t* __restrict__ out_fp4,           // [m*topk, k/2]
    uint8_t* __restrict__ out_sf,            // swizzled scale bytes
    const scalar_t* __restrict__ c1,         // [m*topk, 2*k], gate||up
    const float*    __restrict__ a2_gscale,  // [E]
    const int32_t*  __restrict__ expert_offsets,     // [E+1]
    const int32_t*  __restrict__ blockscale_offsets, // [E+1]
    int m_topk,
    int k,
    int num_experts,
    int numKTiles) {

  const int row = blockIdx.x;
  if (row >= m_topk) return;

  const int n_blocks = k / FP4_BLOCK_SIZE;

  // --- Expert lookup: broadcast to whole block ---
  __shared__ int s_expert_id;
  __shared__ int s_row_in_buf;
  __shared__ float s_gscale;
  if (threadIdx.x == 0) {
    int e = 0;
    #pragma unroll 1
    for (int i = 0; i < num_experts; i++) {
      if (row < expert_offsets[i + 1]) { e = i; break; }
    }
    int token_in_expert = row - expert_offsets[e];
    s_expert_id  = e;
    s_row_in_buf = blockscale_offsets[e] + token_in_expert;
    s_gscale     = a2_gscale[e];
  }
  __syncthreads();

  const int   row_in_buf = s_row_in_buf;
  const float gscale     = s_gscale;
  const float inv_gscale = rcp_approx(gscale);

  const scalar_t* gate_row = c1 + row * (2 * k);       // first k elements
  const scalar_t* up_row   = gate_row + k;             // next k elements

  // --- Each thread processes one or more 16-element output blocks ---
  for (int blk = threadIdx.x; blk < n_blocks; blk += blockDim.x) {
    const int base = blk * FP4_BLOCK_SIZE;

    float act[FP4_BLOCK_SIZE];
    float abs_max = 0.0f;

    #pragma unroll
    for (int j = 0; j < FP4_BLOCK_SIZE; j++) {
      float g = _to_float(gate_row[base + j]);
      float u = _to_float(up_row  [base + j]);
      float a = silu(g) * u;
      act[j] = a;
      abs_max = fmaxf(abs_max, fabsf(a));
    }

    // Per-block scale (matches rms_norm_dynamic_fp4_quant.cu math).
    float sf_val = gscale * (abs_max * rcp_approx(FP4_E2M1_MAX));
    sf_val = fminf(sf_val, 448.0f);

    __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_val);
    uint8_t sf_byte      = reinterpret_cast<uint8_t&>(sf_fp8);
    float   sf_readback  = static_cast<float>(sf_fp8);

    float output_scale = (sf_readback != 0.0f)
        ? rcp_approx(sf_readback * inv_gscale) : 0.0f;

    // Write scale in CUTLASS 128x4 swizzled layout.
    int64_t sf_off = compute_cutlass_sf_offset(row_in_buf, blk, numKTiles);
    out_sf[sf_off] = sf_byte;

    // Scale values for quant.
    float scaled[FP4_BLOCK_SIZE];
    #pragma unroll
    for (int j = 0; j < FP4_BLOCK_SIZE; j++) {
      scaled[j] = act[j] * output_scale;
    }

    // Pack 16 FP4 nibbles into 8 bytes (2 x uint32).
    uint32_t lo = fp32x8_to_e2m1(scaled);
    uint32_t hi = fp32x8_to_e2m1(scaled + 8);

    int byte_off = row * (k / 2) + blk * (FP4_BLOCK_SIZE / 2);
    reinterpret_cast<uint32_t*>(out_fp4 + byte_off)[0] = lo;
    reinterpret_cast<uint32_t*>(out_fp4 + byte_off)[1] = hi;
  }
}

// ============================================================
// Host entry point
// ============================================================
std::vector<torch::Tensor> fused_silu_fp4_quant(
    torch::Tensor c1,                  // [m*topk, 2*k] BF16 or FP16
    torch::Tensor a2_gscale,           // [E] float32
    torch::Tensor expert_offsets,      // [E+1] int32
    torch::Tensor blockscale_offsets,  // [E+1] int32
    int64_t topk) {

  TORCH_CHECK(c1.is_cuda(), "c1 must be CUDA");
  TORCH_CHECK(a2_gscale.is_cuda(), "a2_gscale must be CUDA");
  TORCH_CHECK(expert_offsets.is_cuda(), "expert_offsets must be CUDA");
  TORCH_CHECK(blockscale_offsets.is_cuda(), "blockscale_offsets must be CUDA");
  TORCH_CHECK(c1.dim() == 2, "c1 must be 2D");
  TORCH_CHECK(c1.size(1) % 2 == 0, "c1.shape[1] must be even (gate||up)");
  TORCH_CHECK(expert_offsets.dtype() == torch::kInt32, "expert_offsets int32");
  TORCH_CHECK(blockscale_offsets.dtype() == torch::kInt32,
              "blockscale_offsets int32");
  TORCH_CHECK(a2_gscale.dtype() == torch::kFloat32, "a2_gscale float32");
  TORCH_CHECK(topk > 0, "topk positive");

  const at::cuda::CUDAGuard device_guard(c1.device());

  const int m_topk = c1.size(0);
  const int k      = c1.size(1) / 2;
  const int num_experts = expert_offsets.size(0) - 1;
  TORCH_CHECK(k % FP4_BLOCK_SIZE == 0, "k must be divisible by 16");

  const int n_blocks       = k / FP4_BLOCK_SIZE;
  const int padded_k_int32 = (n_blocks + 3) / 4;
  const int numKTiles      = (k + 63) / 64;

  // Match vLLM's buffer shape so blockscale_offsets remain valid for the
  // downstream cutlass_fp4_moe_mm call.  We allocate num_experts*320 rows
  // as the upper bound (same as fused_shuffle_quant.cu L304).  For
  // typical Qwen3-30B-A3B (E=128, topk=8, C<=512) this is 40960 rows
  // (== MAX_TOKENS_PER_EXPERT * topk for the default env).
  int max_rows = num_experts * 320;
  if (max_rows < m_topk + 128) {
    max_rows = ((m_topk + 127) / 128) * 128 + 128;
  }

  auto output_fp4 = torch::empty({m_topk, k / 2},
                                 c1.options().dtype(torch::kUInt8));
  auto output_sf  = torch::empty({max_rows, padded_k_int32},
                                 c1.options().dtype(torch::kInt32));

  dim3 grid(m_topk);
  int  threads = n_blocks;
  if (threads > 1024) threads = 1024;
  if (threads < 32)   threads = 32;
  dim3 block(threads);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (c1.dtype() == torch::kBFloat16) {
    fused_silu_fp4_quant_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        output_fp4.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(output_sf.data_ptr<int32_t>()),
        reinterpret_cast<const __nv_bfloat16*>(c1.data_ptr<at::BFloat16>()),
        a2_gscale.data_ptr<float>(),
        expert_offsets.data_ptr<int32_t>(),
        blockscale_offsets.data_ptr<int32_t>(),
        m_topk, k, num_experts, numKTiles);
  } else if (c1.dtype() == torch::kFloat16) {
    fused_silu_fp4_quant_kernel<__half><<<grid, block, 0, stream>>>(
        output_fp4.data_ptr<uint8_t>(),
        reinterpret_cast<uint8_t*>(output_sf.data_ptr<int32_t>()),
        reinterpret_cast<const __half*>(c1.data_ptr<at::Half>()),
        a2_gscale.data_ptr<float>(),
        expert_offsets.data_ptr<int32_t>(),
        blockscale_offsets.data_ptr<int32_t>(),
        m_topk, k, num_experts, numKTiles);
  } else {
    TORCH_CHECK(false, "c1 must be BF16 or FP16, got ", c1.dtype());
  }

  return {output_fp4, output_sf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_silu_fp4_quant", &fused_silu_fp4_quant,
        "Fused SiLU+mul + NVFP4 block-quantization (post-GEMM1 epilogue).\n"
        "Drop-in replacement for silu_and_mul_scaled_fp4_experts_quant.\n"
        "Writes FP4 output + CUTLASS 128x4 swizzled FP8-E4M3 scales.\n\n"
        "Args:\n"
        "  c1: [m*topk, 2*k] BF16/FP16 GEMM1 output (gate||up)\n"
        "  a2_gscale: [E] float32 per-expert global scale\n"
        "  expert_offsets: [E+1] int32\n"
        "  blockscale_offsets: [E+1] int32\n"
        "  topk: int, top-k experts per token\n\n"
        "Returns: [output_fp4, output_sf_int32_swizzled]",
        py::arg("c1"),
        py::arg("a2_gscale"),
        py::arg("expert_offsets"),
        py::arg("blockscale_offsets"),
        py::arg("topk"));
}
