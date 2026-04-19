// SPDX-License-Identifier: Apache-2.0
//
// Fused shuffle_rows + quantize_to_nvfp4 kernel for MoE dispatch.
//
// Eliminates the BF16 intermediate between shuffle_rows and scaled_fp4_quant
// by gathering from unsorted positions during quantization (gather-read pattern).
//
// Instead of:
//   sorted_bf16 = shuffle_rows(normalized, a_map)    // write M*H bf16
//   fp4, scales = scaled_fp4_quant(sorted_bf16, ...)  // read M*H bf16
//
// We do:
//   fp4, scales = fused_shuffle_quant(normalized, a_map, ...)
//                 // gather-read from unsorted, write fp4 directly
//
// The dst2src_map (a_map) is a [num_tokens_permuted] int32 tensor where
// dst2src_map[dst_row] = src_row in the original (unsorted) tensor.
//
// Key constraint: norm CANNOT be fused here (data dependency on routing).
// Only shuffle + quant are fused.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

// ============================================================
// FP4-E2M1 constants
// ============================================================
static constexpr float FP4_E2M1_MAX = 6.0f;
static constexpr int FP4_BLOCK_SIZE = 16;  // elements per scale factor

// ============================================================
// FP4 conversion: software fallback
// ============================================================
// Quantize a single float to FP4-E2M1 nibble (sign + 3-bit magnitude)
// E2M1 values: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
__device__ __forceinline__ uint8_t float_to_e2m1_nibble(float x) {
    int sign = (x < 0.0f) ? 1 : 0;
    float ax = fabsf(x);
    ax = fminf(ax, FP4_E2M1_MAX);

    // Boundary-based quantization
    int code = (ax > 0.25f) + (ax > 0.75f) + (ax > 1.25f) +
               (ax > 1.75f) + (ax > 2.5f) + (ax > 3.5f) + (ax > 5.0f);

    return (uint8_t)((sign << 3) | code);
}

// Convert FP32 scale to FP8-E4M3 byte (manual bit conversion)
// FP32 -> FP16 -> FP8 e4m3 (s1 e4 m3, bias=7)
__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float scale) {
    __half sh = __float2half(scale);
    unsigned short hbits = *reinterpret_cast<unsigned short*>(&sh);
    // FP16: s(1) e(5) m(10), bias=15
    // FP8 e4m3: s(1) e(4) m(3), bias=7
    int exp16 = (hbits >> 10) & 0x1F;
    int exp8 = exp16 - 15 + 7;  // rebias
    uint8_t fp8;
    if (exp8 <= 0) {
        fp8 = 0;  // underflow to zero
    } else if (exp8 >= 15) {
        fp8 = 0x7E;  // max normal (0 1111 110 = 448.0)
    } else {
        fp8 = ((uint8_t)exp8 << 3) | ((hbits >> 7) & 0x7);
    }
    // Scale is always positive, no sign bit needed
    return fp8;
}

// ============================================================
// CUTLASS 128x4 blockscale swizzle helper
// ============================================================
// CUTLASS NVFP4 MoE GEMM expects scale factors in a 5D swizzled layout
// [numMTiles, numKTiles, 32, 4, 4] where numMTiles = ceil(M/128),
// numKTiles = ceil(K/64). Bytes live at:
//   row_in_buf = blockscale_offsets[e] + (dst_row - expert_offsets[e])
//   mTileIdx   = row_in_buf >> 7
//   outerMIdx  = row_in_buf & 31
//   innerMIdx  = (row_in_buf >> 5) & 3
//   kTileIdx   = block_idx >> 2
//   innerKIdx  = block_idx & 3
//   byte_off   = ((mTileIdx * numKTiles + kTileIdx) << 9)
//                | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx
//
// This matches the Triton reference in patches/fused_shuffle_quant_swizzle.py
// and the sibling CUDA kernel in rms_norm_dynamic_fp4_quant.cu.
__device__ __forceinline__ int64_t compute_cutlass_sf_offset(
    int row_in_buf,   // blockscale_offsets[expert] + token_in_expert
    int block_idx,    // 0..n_blocks-1
    int numKTiles     // ceil(K / 64)
) {
    int mTileIdx  = row_in_buf >> 7;         // /128
    int outerMIdx = row_in_buf & 31;         // %32
    int innerMIdx = (row_in_buf >> 5) & 3;   // (/32)%4
    int kTileIdx  = block_idx >> 2;          // /4
    int innerKIdx = block_idx & 3;           // %4
    return ((static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx) << 9)
           | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;
}

// ============================================================
// Fused shuffle + quantize kernel (with CUTLASS swizzled scales)
// ============================================================
// Each block processes one output row. Each thread processes one
// 16-element FP4 block within that row using vectorized half2 loads.
//
// The key fusion: instead of reading from sorted (contiguous) memory,
// we use dst2src_map to gather from the original unsorted positions.
//
// Scale output is written in CUTLASS 128x4 swizzled format, compatible
// with cutlass_fp4_moe_mm.
__global__ void fused_shuffle_quant_kernel(
    const half2* __restrict__ input,       // [M_unsorted, K/2] as half2 (unsorted/original)
    const int32_t* __restrict__ dst2src_map, // [M_sorted] mapping sorted->original row
    const int32_t* __restrict__ expert_offsets, // [E+1] token start per expert
    const int32_t* __restrict__ blockscale_offsets, // [E+1] scale row start per expert
    uint8_t* __restrict__ output,          // [M_sorted, K/2] packed FP4
    uint8_t* __restrict__ scales_out,      // CUTLASS-swizzled scale buffer (uint8)
    int M_sorted,                          // number of output rows (permuted tokens)
    int K,                                 // hidden dimension
    int n_blocks,                          // K / 16 (number of FP4 blocks per row)
    int num_experts,                       // E
    int numKTiles                          // ceil(K / 64) for CUTLASS swizzle
) {
    int dst_row = blockIdx.x;
    int block_idx = threadIdx.x;

    if (dst_row >= M_sorted || block_idx >= n_blocks) return;

    // --- Find which expert this row belongs to ---
    // Linear scan (E <= 128, done once per block = once per row)
    int expert_id = 0;
    for (int e = 0; e < num_experts; e++) {
        if (dst_row < expert_offsets[e + 1]) {
            expert_id = e;
            break;
        }
    }
    int token_in_expert = dst_row - expert_offsets[expert_id];
    int row_in_buf = blockscale_offsets[expert_id] + token_in_expert;

    // --- FUSION POINT: gather from unsorted position ---
    int src_row = dst2src_map[dst_row];

    // Load 8x half2 = 16 FP16 values via vectorized loads from SOURCE row
    const half2* in_ptr = input + src_row * (K / 2) + block_idx * 8;
    float vals[16];
    float max_abs = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        half2 h2 = in_ptr[i];
        vals[i * 2]     = __half2float(h2.x);
        vals[i * 2 + 1] = __half2float(h2.y);
        float a0 = fabsf(vals[i * 2]);
        float a1 = fabsf(vals[i * 2 + 1]);
        if (a0 > max_abs) max_abs = a0;
        if (a1 > max_abs) max_abs = a1;
    }

    // Compute per-block scale factor
    float scale = fmaxf(max_abs / FP4_E2M1_MAX, 1e-12f);
    float inv_scale = 1.0f / scale;

    // Store scale as FP8-E4M3 in CUTLASS 128x4 swizzled layout
    int64_t scale_offset = compute_cutlass_sf_offset(row_in_buf, block_idx, numKTiles);
    scales_out[scale_offset] = float_to_fp8_e4m3(scale);

    // Quantize to FP4-E2M1 and pack pairs into bytes
    // Output is written to DESTINATION (sorted) row
    uint8_t* out_ptr = output + dst_row * (K / 2) + block_idx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v0 = vals[i * 2] * inv_scale;
        float v1 = vals[i * 2 + 1] * inv_scale;

        uint8_t c0 = float_to_e2m1_nibble(v0);
        uint8_t c1 = float_to_e2m1_nibble(v1);

        out_ptr[i] = c0 | (c1 << 4);
    }
}

// ============================================================
// BF16 variant — same kernel but reads nv_bfloat162 instead of half2
// (Gemma-4 MoE path typically uses BF16 after RMSNorm)
// ============================================================
__global__ void fused_shuffle_quant_bf16_kernel(
    const nv_bfloat162* __restrict__ input,  // [M_unsorted, K/2] as bf162
    const int32_t* __restrict__ dst2src_map, // [M_sorted]
    const int32_t* __restrict__ expert_offsets, // [E+1]
    const int32_t* __restrict__ blockscale_offsets, // [E+1]
    uint8_t* __restrict__ output,            // [M_sorted, K/2] packed FP4
    uint8_t* __restrict__ scales_out,        // CUTLASS-swizzled scale buffer (uint8)
    int M_sorted, int K,
    int n_blocks, int num_experts,
    int numKTiles
) {
    int dst_row = blockIdx.x;
    int block_idx = threadIdx.x;

    if (dst_row >= M_sorted || block_idx >= n_blocks) return;

    // Find expert for this row
    int expert_id = 0;
    for (int e = 0; e < num_experts; e++) {
        if (dst_row < expert_offsets[e + 1]) {
            expert_id = e;
            break;
        }
    }
    int token_in_expert = dst_row - expert_offsets[expert_id];
    int row_in_buf = blockscale_offsets[expert_id] + token_in_expert;

    int src_row = dst2src_map[dst_row];

    const nv_bfloat162* in_ptr = input + src_row * (K / 2) + block_idx * 8;
    float vals[16];
    float max_abs = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        nv_bfloat162 b2 = in_ptr[i];
        vals[i * 2]     = __bfloat162float(b2.x);
        vals[i * 2 + 1] = __bfloat162float(b2.y);
        float a0 = fabsf(vals[i * 2]);
        float a1 = fabsf(vals[i * 2 + 1]);
        if (a0 > max_abs) max_abs = a0;
        if (a1 > max_abs) max_abs = a1;
    }

    float scale = fmaxf(max_abs / FP4_E2M1_MAX, 1e-12f);
    float inv_scale = 1.0f / scale;

    // Store scale in CUTLASS 128x4 swizzled layout
    int64_t scale_offset = compute_cutlass_sf_offset(row_in_buf, block_idx, numKTiles);
    scales_out[scale_offset] = float_to_fp8_e4m3(scale);

    uint8_t* out_ptr = output + dst_row * (K / 2) + block_idx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v0 = vals[i * 2] * inv_scale;
        float v1 = vals[i * 2 + 1] * inv_scale;

        uint8_t c0 = float_to_e2m1_nibble(v0);
        uint8_t c1 = float_to_e2m1_nibble(v1);

        out_ptr[i] = c0 | (c1 << 4);
    }
}

// ============================================================
// Host entry points
// ============================================================

std::vector<torch::Tensor> fused_shuffle_quant(
    torch::Tensor input,              // [M_unsorted, K] FP16 or BF16 (unsorted)
    torch::Tensor dst2src_map,        // [M_sorted] int32
    torch::Tensor expert_offsets,     // [E+1] int32
    torch::Tensor blockscale_offsets, // [E+1] int32
    int num_topk                      // topk (for MAX_TOKENS_PER_EXPERT scaling)
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(dst2src_map.is_cuda(), "dst2src_map must be CUDA");
    TORCH_CHECK(expert_offsets.is_cuda(), "expert_offsets must be CUDA");
    TORCH_CHECK(blockscale_offsets.is_cuda(), "blockscale_offsets must be CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [M, K]");
    TORCH_CHECK(dst2src_map.dim() == 1, "dst2src_map must be 1D [M_sorted]");
    TORCH_CHECK(dst2src_map.dtype() == torch::kInt32, "dst2src_map must be int32");
    TORCH_CHECK(expert_offsets.dtype() == torch::kInt32, "expert_offsets must be int32");
    TORCH_CHECK(blockscale_offsets.dtype() == torch::kInt32, "blockscale_offsets must be int32");

    int M_sorted = dst2src_map.size(0);
    int K = input.size(1);
    int num_experts = expert_offsets.size(0) - 1;
    TORCH_CHECK(K % 16 == 0, "K must be divisible by 16 (FP4 block size)");
    TORCH_CHECK(num_topk > 0, "num_topk must be positive");

    const at::cuda::CUDAGuard device_guard(input.device());

    int n_blocks = K / 16;
    int padded_k_int32 = (n_blocks + 3) / 4;   // int32 columns per scale row
    int numKTiles = (K + 63) / 64;              // CUTLASS K tile count

    // Allocate output FP4 tensor
    auto output = torch::empty({M_sorted, K / 2}, input.options().dtype(torch::kUInt8));

    // Allocate CUTLASS 128x4 swizzled scale buffer, matching the shape of
    // vLLM's scaled_fp4_experts_quant: int32 [MAX_TPE*topk, padded_k_int32].
    // The maximum row index written is blockscale_offsets[num_experts-1] + (tokens_in_last_expert).
    // MAX_TPE is a compile-time vLLM constant (VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE, typically
    // 163840 / topk). Since we can't read the env-var from CUDA, mirror vLLM's buffer sizing:
    // the largest valid row is bounded by `num_experts * MAX_TPE_per_expert`, and the swizzle
    // rounds rows up to multiples of 128. We allocate enough to cover all offsets:
    // padded_M = 128 * ceil((blockscale_offsets_max)/128). To avoid a device->host sync,
    // allocate num_experts * 320 rows (each expert can hold up to ~320 tokens = MAX_TPE/topk
    // for topk<=8) which matches the vLLM default VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=163840
    // split across experts. The caller may provide a pre-sized buffer via future overloads.
    int max_rows = num_experts * 320;  // 320 = common MAX_TPE per expert upper bound
    if (max_rows < M_sorted + 128) {
        max_rows = ((M_sorted + 127) / 128) * 128 + 128;
    }
    auto scales = torch::empty(
        {max_rows, padded_k_int32},
        input.options().dtype(torch::kInt32)
    );

    dim3 grid(M_sorted);
    dim3 block(min(n_blocks, 1024));

    auto stream = at::cuda::getCurrentCUDAStream();

    if (input.dtype() == torch::kFloat16) {
        fused_shuffle_quant_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const half2*>(input.data_ptr<at::Half>()),
            dst2src_map.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            blockscale_offsets.data_ptr<int32_t>(),
            output.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(scales.data_ptr<int32_t>()),
            M_sorted, K, n_blocks, num_experts, numKTiles
        );
    } else if (input.dtype() == torch::kBFloat16) {
        fused_shuffle_quant_bf16_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const nv_bfloat162*>(input.data_ptr<at::BFloat16>()),
            dst2src_map.data_ptr<int32_t>(),
            expert_offsets.data_ptr<int32_t>(),
            blockscale_offsets.data_ptr<int32_t>(),
            output.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(scales.data_ptr<int32_t>()),
            M_sorted, K, n_blocks, num_experts, numKTiles
        );
    } else {
        TORCH_CHECK(false, "Input must be FP16 or BF16, got ", input.dtype());
    }

    return {output, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_shuffle_quant", &fused_shuffle_quant,
          "Fused shuffle_rows + NVFP4 quantization for MoE dispatch.\n"
          "Eliminates the BF16 intermediate by gathering from unsorted positions\n"
          "during quantization via dst2src_map.\n"
          "Writes scales in CUTLASS 128x4 swizzled blockscale format.\n\n"
          "Args:\n"
          "  input: [M_unsorted, K] FP16/BF16 tensor (pre-norm, unsorted)\n"
          "  dst2src_map: [M_sorted] int32 tensor (shuffled->original row mapping)\n"
          "  expert_offsets: [E+1] int32 tensor (token start per expert)\n"
          "  blockscale_offsets: [E+1] int32 tensor (scale row start per expert)\n"
          "  num_topk: number of top-k experts per token\n\n"
          "Returns: [output_fp4, scales_int32_swizzled]",
          py::arg("input"),
          py::arg("dst2src_map"),
          py::arg("expert_offsets"),
          py::arg("blockscale_offsets"),
          py::arg("num_topk"));
}
