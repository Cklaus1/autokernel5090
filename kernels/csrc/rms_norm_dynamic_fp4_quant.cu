// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused RMSNorm + Dynamic FP4-E2M1 Block Quantization kernel.
//
// Eliminates the BF16 intermediate between RMSNorm and scaled_fp4_quant
// by computing norm + quantize in a single pass through registers.
//
// For each row:
//   1. Compute variance = sum(x[i]^2) / N  (using CUB block reduce)
//   2. rms = rsqrt(variance + epsilon)
//   3. For each block of 16 elements:
//      a. x_norm[j] = x[j] * rms * weight[j]
//      b. block_max = max(|x_norm[j]|) for j in block
//      c. sf = global_scale * (block_max / 6.0)  -- scale factor for FP4
//      d. sf_fp8 = fp8_e4m3fn(sf)                -- store as FP8
//      e. output_scale = 1.0 / (fp32(sf_fp8) / global_scale)
//      f. code[j] = nearest_e2m1(x_norm[j] * output_scale)
//      g. Pack pairs into uint8: out[j/2] = (code[j+1] << 4) | code[j]
//
// Uses SM120 (Blackwell) native `cvt.rn.satfinite.e2m1x2.f32` PTX instruction
// for hardware FP4 conversion when available.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cub/cub.cuh>

// CUB add op
#if CUB_VERSION >= 200800
  #include <cuda/std/functional>
using CubAddOp = cuda::std::plus<>;
#else
using CubAddOp = cub::Sum;
#endif

namespace vllm {

// ============================================================
// FP4-E2M1 constants
// ============================================================
static constexpr float FP4_E2M1_MAX = 6.0f;
static constexpr int FP4_BLOCK_SIZE = 16;  // elements per scale factor

// ============================================================
// FP4 conversion utilities
// ============================================================

// Convert 8 float32 values to packed FP4-E2M1 (8 values -> 1 uint32_t)
// Uses native PTX on SM120+, software fallback otherwise.
__device__ __forceinline__ uint32_t fp32x8_to_e2m1(float v[8]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  // Hardware FP4 conversion via PTX
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
  // Software fallback: FP4-E2M1 lookup table approach
  // E2M1 values: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
  // Boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
  uint32_t packed = 0;
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    float x = v[i];
    int sign = (x < 0.0f) ? 1 : 0;
    float ax = fabsf(x);
    // Clamp to FP4 max
    ax = fminf(ax, FP4_E2M1_MAX);

    int code = 0;
    if (ax > 5.0f) code = 7;
    else if (ax > 3.5f) code = 6;
    else if (ax > 2.5f) code = 5;
    else if (ax > 1.75f) code = 4;
    else if (ax > 1.25f) code = 3;
    else if (ax > 0.75f) code = 2;
    else if (ax > 0.25f) code = 1;
    else code = 0;

    int nibble = (sign << 3) | code;
    // Pack: even indices in low nibble, odd in high nibble of each byte
    // Byte i/2, nibble position depends on i%2
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

// ============================================================
// Swizzled SF (scale factor) layout for CUTLASS compatibility
// ============================================================
// SF layout: [numMTiles, numKTiles, 32, 4, 4]
// Where numMTiles = ceil(M/128), numKTiles = ceil(N/64)
__device__ __forceinline__ int64_t swizzled_sf_offset(
    int row, int sf_col, int numKTiles) {
  int mTileIdx = row >> 7;         // row / 128
  int outerMIdx = row & 31;        // row % 32
  int innerMIdx = (row >> 5) & 3;  // (row / 32) % 4
  int kTileIdx = sf_col >> 2;      // sf_col / 4
  int innerKIdx = sf_col & 3;      // sf_col % 4

  return ((static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9) |
         (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;
}

// ============================================================
// Main fused kernel: RMSNorm + FP4-E2M1 block quantization
// ============================================================
// One thread block per row. Each thread processes multiple elements.
// Two passes:
//   Pass 1: compute sum-of-squares for RMS normalization
//   Pass 2: normalize, compute per-block scale, quantize to FP4, pack
//
// Template parameters:
//   scalar_t: input dtype (half or __nv_bfloat16)
//   BLOCK_THREADS: threads per block (must be <= 1024)
template <typename scalar_t, int BLOCK_THREADS>
__global__ void rms_norm_dynamic_fp4_quant_kernel(
    uint8_t* __restrict__ out_fp4,       // [M, N/2] packed FP4 output
    uint8_t* __restrict__ out_sf,        // scale factors (swizzled or row-major)
    const scalar_t* __restrict__ input,  // [M, N] input
    const scalar_t* __restrict__ weight, // [N] RMS norm weight
    const float* __restrict__ global_scale_ptr, // [1] FP4 global scale
    const float epsilon,
    const int num_tokens,
    const int hidden_size,
    const int input_stride,
    const bool is_sf_swizzled) {

  const int row = blockIdx.x;
  if (row >= num_tokens) return;

  const scalar_t* input_row = input + row * input_stride;
  const float global_scale = *global_scale_ptr;

  // Number of SF blocks = hidden_size / 16
  const int num_sf_blocks = hidden_size / FP4_BLOCK_SIZE;
  // For swizzled layout
  const int numKTiles = (hidden_size + 63) / 64;

  // ---- Pass 1: Compute sum of squares for RMS ----
  float thread_sum_sq = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += BLOCK_THREADS) {
    float x = static_cast<float>(input_row[idx]);
    thread_sum_sq += x * x;
  }

  // Block-level reduction
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  float sum_sq = BlockReduce(reduce_storage).Reduce(thread_sum_sq, CubAddOp{},
                                                     BLOCK_THREADS);

  __shared__ float s_rrms;
  if (threadIdx.x == 0) {
    s_rrms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();
  float rrms = s_rrms;

  // ---- Pass 2: Normalize + per-block scale + quantize + pack ----
  // Each thread processes blocks of 16 elements.
  // A block of 16 elements produces 1 SF + 8 packed bytes (16 FP4 values).
  for (int blk = threadIdx.x; blk < num_sf_blocks; blk += BLOCK_THREADS) {
    int base = blk * FP4_BLOCK_SIZE;

    // Load, normalize, and apply weight for 16 elements
    float normed[FP4_BLOCK_SIZE];
    float abs_max = 0.0f;

    #pragma unroll
    for (int j = 0; j < FP4_BLOCK_SIZE; j++) {
      int col = base + j;
      float x = static_cast<float>(input_row[col]);
      float w = static_cast<float>(weight[col]);
      float xn = x * rrms * w;
      normed[j] = xn;
      abs_max = fmaxf(abs_max, fabsf(xn));
    }

    // Compute scale factor:
    //   sf = global_scale * (abs_max / FP4_MAX)
    //   Store as FP8-E4M3, read back for exact consistency
    float sf_val = global_scale * (abs_max * rcp_approx(FP4_E2M1_MAX));

    // Clamp to FP8-E4M3 range (max 448.0)
    sf_val = fminf(sf_val, 448.0f);

    // Convert to FP8-E4M3
    __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_val);
    uint8_t sf_byte = reinterpret_cast<uint8_t&>(sf_fp8);

    // Read back to float for exact match
    float sf_readback = static_cast<float>(sf_fp8);

    // Compute output scale: values need to be divided by (sf_readback / global_scale)
    // i.e., multiplied by global_scale / sf_readback
    float output_scale = (sf_readback != 0.0f)
        ? rcp_approx(sf_readback * rcp_approx(global_scale))
        : 0.0f;

    // Store scale factor
    if (is_sf_swizzled) {
      int sf_col = blk;  // one SF per block of 16 elements
      int64_t sf_offset = swizzled_sf_offset(row, sf_col, numKTiles);
      out_sf[sf_offset] = sf_byte;
    } else {
      // Row-major: [M, num_sf_blocks]
      int sf_n = num_sf_blocks;
      out_sf[row * sf_n + blk] = sf_byte;
    }

    // Scale all normed values and convert to FP4
    float scaled[FP4_BLOCK_SIZE];
    #pragma unroll
    for (int j = 0; j < FP4_BLOCK_SIZE; j++) {
      scaled[j] = normed[j] * output_scale;
    }

    // Convert 16 values to packed FP4 (2 uint32_t = 8 bytes)
    // First 8 values
    uint32_t packed_lo = fp32x8_to_e2m1(scaled);
    // Second 8 values
    uint32_t packed_hi = fp32x8_to_e2m1(scaled + 8);

    // Write 8 packed bytes (16 FP4 values = 8 bytes)
    int byte_offset = row * (hidden_size / 2) + blk * (FP4_BLOCK_SIZE / 2);
    reinterpret_cast<uint32_t*>(out_fp4 + byte_offset)[0] = packed_lo;
    reinterpret_cast<uint32_t*>(out_fp4 + byte_offset)[1] = packed_hi;
  }
}

// ============================================================
// Fused residual add + RMSNorm + FP4 quantization variant
// ============================================================
template <typename scalar_t, int BLOCK_THREADS>
__global__ void fused_add_rms_norm_dynamic_fp4_quant_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf,
    scalar_t* __restrict__ input,         // [M, N] -- modified in-place (input += residual)
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ residual,      // [M, N] -- updated with input + residual
    const float* __restrict__ global_scale_ptr,
    const float epsilon,
    const int num_tokens,
    const int hidden_size,
    const int input_stride,
    const bool is_sf_swizzled) {

  const int row = blockIdx.x;
  if (row >= num_tokens) return;

  scalar_t* input_row = input + row * input_stride;
  scalar_t* residual_row = residual + row * hidden_size;
  const float global_scale = *global_scale_ptr;
  const int num_sf_blocks = hidden_size / FP4_BLOCK_SIZE;
  const int numKTiles = (hidden_size + 63) / 64;

  // ---- Pass 1: Add residual, store back, compute sum of squares ----
  float thread_sum_sq = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += BLOCK_THREADS) {
    float x_val = static_cast<float>(input_row[idx]);
    float r_val = static_cast<float>(residual_row[idx]);
    float hidden = x_val + r_val;

    // Store to residual (for skip connection downstream)
    residual_row[idx] = static_cast<scalar_t>(hidden);

    thread_sum_sq += hidden * hidden;
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  float sum_sq = BlockReduce(reduce_storage).Reduce(thread_sum_sq, CubAddOp{},
                                                     BLOCK_THREADS);

  __shared__ float s_rrms;
  if (threadIdx.x == 0) {
    s_rrms = rsqrtf(sum_sq / static_cast<float>(hidden_size) + epsilon);
  }
  __syncthreads();
  float rrms = s_rrms;

  // ---- Pass 2: Normalize from residual + quantize ----
  for (int blk = threadIdx.x; blk < num_sf_blocks; blk += BLOCK_THREADS) {
    int base = blk * FP4_BLOCK_SIZE;

    float normed[FP4_BLOCK_SIZE];
    float abs_max = 0.0f;

    #pragma unroll
    for (int j = 0; j < FP4_BLOCK_SIZE; j++) {
      int col = base + j;
      float x = static_cast<float>(residual_row[col]);
      float w = static_cast<float>(weight[col]);
      float xn = x * rrms * w;
      normed[j] = xn;
      abs_max = fmaxf(abs_max, fabsf(xn));
    }

    float sf_val = global_scale * (abs_max * rcp_approx(FP4_E2M1_MAX));
    sf_val = fminf(sf_val, 448.0f);
    __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_val);
    uint8_t sf_byte = reinterpret_cast<uint8_t&>(sf_fp8);
    float sf_readback = static_cast<float>(sf_fp8);

    float output_scale = (sf_readback != 0.0f)
        ? rcp_approx(sf_readback * rcp_approx(global_scale))
        : 0.0f;

    if (is_sf_swizzled) {
      int64_t sf_offset = swizzled_sf_offset(row, blk, numKTiles);
      out_sf[sf_offset] = sf_byte;
    } else {
      out_sf[row * num_sf_blocks + blk] = sf_byte;
    }

    float scaled[FP4_BLOCK_SIZE];
    #pragma unroll
    for (int j = 0; j < FP4_BLOCK_SIZE; j++) {
      scaled[j] = normed[j] * output_scale;
    }

    uint32_t packed_lo = fp32x8_to_e2m1(scaled);
    uint32_t packed_hi = fp32x8_to_e2m1(scaled + 8);

    int byte_offset = row * (hidden_size / 2) + blk * (FP4_BLOCK_SIZE / 2);
    reinterpret_cast<uint32_t*>(out_fp4 + byte_offset)[0] = packed_lo;
    reinterpret_cast<uint32_t*>(out_fp4 + byte_offset)[1] = packed_hi;
  }
}

}  // namespace vllm

// ============================================================
// C++ entry points (called from torch_bindings.cpp)
// ============================================================

void rms_norm_dynamic_fp4_quant(
    torch::Tensor& result,           // [M, N/2] uint8 packed FP4
    torch::Tensor& result_scale,     // scale factors (swizzled int32 or row-major fp8)
    torch::Tensor const& input,      // [M, N] bf16/fp16
    torch::Tensor const& weight,     // [N] bf16/fp16
    torch::Tensor const& input_global_scale,  // [1] float32
    double epsilon,
    bool is_sf_swizzled_layout) {

  TORCH_CHECK(result.is_contiguous());
  TORCH_CHECK(input.stride(-1) == 1, "Input must be contiguous in last dim");
  TORCH_CHECK(weight.dtype() == input.dtype());

  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  TORCH_CHECK(hidden_size % 16 == 0,
              "Hidden size must be divisible by 16 for FP4 block quant");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_tokens);

  // Choose block size based on token count for better occupancy
  const int block_size = (num_tokens < 256) ? 1024 : 256;

  // Dispatch on input dtype
  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "rms_norm_dynamic_fp4_quant",
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          if (block_size == 1024) {
            vllm::rms_norm_dynamic_fp4_quant_kernel<scalar_t, 1024>
                <<<grid, 1024, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          } else {
            vllm::rms_norm_dynamic_fp4_quant_kernel<scalar_t, 256>
                <<<grid, 256, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          }
        })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          if (block_size == 1024) {
            vllm::rms_norm_dynamic_fp4_quant_kernel<scalar_t, 1024>
                <<<grid, 1024, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          } else {
            vllm::rms_norm_dynamic_fp4_quant_kernel<scalar_t, 256>
                <<<grid, 256, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          }
        })
  );
}

void fused_add_rms_norm_dynamic_fp4_quant(
    torch::Tensor& result,           // [M, N/2] uint8 packed FP4
    torch::Tensor& result_scale,     // scale factors
    torch::Tensor& input,            // [M, N] bf16/fp16 -- modified in place
    torch::Tensor const& weight,     // [N] bf16/fp16
    torch::Tensor& residual,         // [M, N] -- updated to input + residual
    torch::Tensor const& input_global_scale,  // [1] float32
    double epsilon,
    bool is_sf_swizzled_layout) {

  TORCH_CHECK(result.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  TORCH_CHECK(input.stride(-1) == 1, "Input must be contiguous in last dim");
  TORCH_CHECK(weight.dtype() == input.dtype());
  TORCH_CHECK(residual.scalar_type() == input.scalar_type());

  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  TORCH_CHECK(hidden_size % 16 == 0,
              "Hidden size must be divisible by 16 for FP4 block quant");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_tokens);
  const int block_size = (num_tokens < 256) ? 1024 : 256;

  AT_DISPATCH_SWITCH(
      input.scalar_type(),
      "fused_add_rms_norm_dynamic_fp4_quant",
      AT_DISPATCH_CASE(at::ScalarType::Half,
        [&] {
          if (block_size == 1024) {
            vllm::fused_add_rms_norm_dynamic_fp4_quant_kernel<scalar_t, 1024>
                <<<grid, 1024, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    residual.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          } else {
            vllm::fused_add_rms_norm_dynamic_fp4_quant_kernel<scalar_t, 256>
                <<<grid, 256, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    residual.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          }
        })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16,
        [&] {
          if (block_size == 1024) {
            vllm::fused_add_rms_norm_dynamic_fp4_quant_kernel<scalar_t, 1024>
                <<<grid, 1024, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    residual.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          } else {
            vllm::fused_add_rms_norm_dynamic_fp4_quant_kernel<scalar_t, 256>
                <<<grid, 256, 0, stream>>>(
                    result.data_ptr<uint8_t>(),
                    reinterpret_cast<uint8_t*>(result_scale.data_ptr()),
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    residual.data_ptr<scalar_t>(),
                    input_global_scale.data_ptr<float>(),
                    static_cast<float>(epsilon),
                    num_tokens, hidden_size, input_stride,
                    is_sf_swizzled_layout);
          }
        })
  );
}
