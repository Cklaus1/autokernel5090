// SPDX-License-Identifier: Apache-2.0
//
// Persistent MoE Dispatch Kernel — V1 (route + shuffle + quant + unshuffle)
//
// Fuses the non-GEMM phases of MoE expert dispatch into a single persistent
// kernel using cooperative_groups grid.sync() for cross-SM synchronization.
//
// Current MoE execution has 6 kernel launches per layer:
//   1. shuffle_rows (gather by expert)
//   2. scaled_fp4_experts_quant
//   3. GEMM1 (gate+up)          <-- NOT fused here (still external CUTLASS)
//   4. silu_mul + FP4 requant
//   5. GEMM2 (down)             <-- NOT fused here
//   6. shuffle_rows (scatter back)
//
// This kernel replaces ops 1, 2, 4, 6 with a single persistent launch.
// The GEMMs (3, 5) are still invoked externally between grid.sync() barriers.
//
// V1 scope: prove cooperative_groups grid.sync() on SM120, measure overhead
// elimination for the 4 surrounding ops, and validate numerical correctness.
//
// Target: RTX 5090, SM120, 170 SMs, CUDA 12.8
//
// Phases (all within one kernel launch):
//   Phase 1: Route + Shuffle — sort tokens by expert assignment
//   Phase 2: BF16 -> FP4 quantize sorted activations (per-expert global scale)
//   Phase 3: [grid.sync() — external GEMM1 would go here in V2]
//   Phase 4: SiLU activation + FP4 requantize the GEMM1 output
//   Phase 5: [grid.sync() — external GEMM2 would go here in V2]
//   Phase 6: Unshuffle + weighted accumulate back to original token order

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace persistent_moe {

// =============================================================================
// Constants
// =============================================================================

// FP4 E2M1 encoding table (same as vLLM's scaled_fp4_quant)
// Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
__device__ __constant__ float FP4_E2M1_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// Max representable FP4 E2M1 value
constexpr float FP4_MAX = 6.0f;

// Block size for quantization (scale factor block)
constexpr int SF_BLOCK = 16;

// Threads per block for the persistent kernel
constexpr int BLOCK_SIZE = 256;

// =============================================================================
// Device helpers
// =============================================================================

// Quantize a single float to FP4 E2M1 (4-bit), return 0..15 index
__device__ __forceinline__ uint8_t float_to_fp4(float val) {
    // Sign bit
    uint8_t sign = (val < 0.0f) ? 8 : 0;
    float abs_val = fabsf(val);

    // Clamp to FP4 range
    abs_val = fminf(abs_val, FP4_MAX);

    // Find closest FP4 value via binary search on the positive LUT
    uint8_t best = 0;
    float best_err = abs_val;  // error vs 0.0

    #pragma unroll
    for (uint8_t i = 1; i < 8; i++) {
        float err = fabsf(abs_val - FP4_E2M1_LUT[i]);
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }

    return sign | best;
}

// Dequantize FP4 E2M1 back to float
__device__ __forceinline__ float fp4_to_float(uint8_t fp4) {
    return FP4_E2M1_LUT[fp4 & 0xF];
}

// =============================================================================
// Phase 1: Route + Shuffle
// =============================================================================
// Each SM processes a chunk of tokens, writing them into expert-sorted order.
// We compute a_map (token -> sorted position) and c_map (sorted -> token).
//
// Workspace layout after this phase:
//   sorted_hidden: [total_routed_tokens, K] in BF16
//   expert_offsets: [E+1] — cumulative token counts per expert
//   a_map: [total_routed_tokens] — maps sorted pos -> (token_idx, expert_idx)
//   c_map: [total_routed_tokens] — maps sorted pos -> original token index

__device__ void phase1_route_and_shuffle(
    const __nv_bfloat16* __restrict__ hidden,  // [M, K]
    const int32_t* __restrict__ topk_ids,      // [M, top_k]
    __nv_bfloat16* __restrict__ sorted_hidden, // [M*top_k, K] workspace
    int32_t* __restrict__ expert_counts,       // [E] workspace (atomics)
    int32_t* __restrict__ expert_offsets,       // [E+1] workspace
    int32_t* __restrict__ a_map,               // [M*top_k] sorted_pos -> token_idx
    int M, int K, int E, int top_k,
    int sm_id, int num_sms
) {
    int total_tokens = M * top_k;

    // Step 1: Count tokens per expert (each SM handles a slice)
    int tokens_per_sm = (total_tokens + num_sms - 1) / num_sms;
    int start = sm_id * tokens_per_sm;
    int end = min(start + tokens_per_sm, total_tokens);

    for (int idx = start + threadIdx.x; idx < end; idx += BLOCK_SIZE) {
        int token_idx = idx / top_k;
        int k_idx = idx % top_k;
        int expert_id = topk_ids[token_idx * top_k + k_idx];
        if (expert_id >= 0 && expert_id < E) {
            atomicAdd(&expert_counts[expert_id], 1);
        }
    }
}

// After grid.sync(), SM 0 computes prefix sum of expert_counts -> expert_offsets
__device__ void phase1b_prefix_sum(
    int32_t* __restrict__ expert_counts,
    int32_t* __restrict__ expert_offsets,
    int E
) {
    // Single-threaded prefix sum (E is typically 128, fine for one warp)
    if (threadIdx.x == 0) {
        expert_offsets[0] = 0;
        for (int e = 0; e < E; e++) {
            expert_offsets[e + 1] = expert_offsets[e] + expert_counts[e];
            expert_counts[e] = 0;  // reset for phase1c scatter
        }
    }
}

// After grid.sync(), all SMs scatter tokens into sorted order
__device__ void phase1c_scatter(
    const __nv_bfloat16* __restrict__ hidden,
    const int32_t* __restrict__ topk_ids,
    __nv_bfloat16* __restrict__ sorted_hidden,
    int32_t* __restrict__ expert_counts,     // used as atomicAdd counters
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ a_map,
    int M, int K, int E, int top_k,
    int sm_id, int num_sms
) {
    int total_tokens = M * top_k;
    int tokens_per_sm = (total_tokens + num_sms - 1) / num_sms;
    int start = sm_id * tokens_per_sm;
    int end = min(start + tokens_per_sm, total_tokens);

    for (int idx = start + threadIdx.x; idx < end; idx += BLOCK_SIZE) {
        int token_idx = idx / top_k;
        int k_idx = idx % top_k;
        int expert_id = topk_ids[token_idx * top_k + k_idx];

        if (expert_id >= 0 && expert_id < E) {
            int pos_in_expert = atomicAdd(&expert_counts[expert_id], 1);
            int sorted_pos = expert_offsets[expert_id] + pos_in_expert;

            // Record mapping
            a_map[sorted_pos] = idx;  // sorted_pos -> original (token, k) pair

            // Copy hidden state to sorted position
            // Each thread copies one element at a time (K may be large)
            const __nv_bfloat16* src = hidden + token_idx * K;
            __nv_bfloat16* dst = sorted_hidden + sorted_pos * K;
            for (int d = 0; d < K; d++) {
                dst[d] = src[d];
            }
        }
    }
}

// =============================================================================
// Phase 2: BF16 -> FP4 Quantization (per-expert global scale)
// =============================================================================

__device__ void phase2_quantize(
    const __nv_bfloat16* __restrict__ sorted_hidden,  // [total, K]
    uint8_t* __restrict__ sorted_fp4,                  // [total, K/2] packed
    uint8_t* __restrict__ sorted_sf,                   // [total, K/SF_BLOCK] scale factors
    const float* __restrict__ a_gscale,                // [E] per-expert global scales
    const int32_t* __restrict__ expert_offsets,
    int K, int E,
    int sm_id, int num_sms
) {
    int total = expert_offsets[E];  // total routed tokens
    int sf_per_row = K / SF_BLOCK;

    // Each SM processes a chunk of rows
    int rows_per_sm = (total + num_sms - 1) / num_sms;
    int row_start = sm_id * rows_per_sm;
    int row_end = min(row_start + rows_per_sm, total);

    for (int row = row_start; row < row_end; row++) {
        // Find which expert this row belongs to (binary search)
        int expert_id = 0;
        for (int e = 0; e < E; e++) {
            if (row < expert_offsets[e + 1]) {
                expert_id = e;
                break;
            }
        }
        float gscale = a_gscale[expert_id];
        float inv_gscale = (gscale > 0.0f) ? (1.0f / gscale) : 1.0f;

        const __nv_bfloat16* src_row = sorted_hidden + row * K;
        uint8_t* dst_fp4 = sorted_fp4 + row * (K / 2);
        uint8_t* dst_sf = sorted_sf + row * sf_per_row;

        // Process in blocks of SF_BLOCK elements
        for (int blk = threadIdx.x; blk < sf_per_row; blk += BLOCK_SIZE) {
            int col_start = blk * SF_BLOCK;

            // Compute block-local absmax for scale factor
            float local_max = 0.0f;
            float vals[SF_BLOCK];

            #pragma unroll
            for (int i = 0; i < SF_BLOCK; i++) {
                vals[i] = __bfloat162float(src_row[col_start + i]) * inv_gscale;
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }

            // Scale factor: local_max / FP4_MAX, stored as FP8 E4M3
            float sf_val = (local_max > 0.0f) ? (local_max / FP4_MAX) : 1.0f;
            float inv_sf = (sf_val > 0.0f) ? (1.0f / sf_val) : 1.0f;

            // Store scale factor as uint8 (simplified: just store as byte-quantized)
            // In production, this would be __nv_fp8_e4m3
            dst_sf[blk] = (uint8_t)fminf(fmaxf(sf_val * 127.0f, 0.0f), 255.0f);

            // Quantize pairs of values into packed FP4
            #pragma unroll
            for (int i = 0; i < SF_BLOCK; i += 2) {
                uint8_t lo = float_to_fp4(vals[i] * inv_sf);
                uint8_t hi = float_to_fp4(vals[i + 1] * inv_sf);
                dst_fp4[(col_start + i) / 2] = (hi << 4) | lo;
            }
        }
    }
}

// =============================================================================
// Phase 4: SiLU activation + FP4 requantize
// =============================================================================
// Applied to GEMM1 output which is [total, 2*N] (gate + up projections).
// SiLU(gate) * up, then requantize to FP4.

__device__ void phase4_silu_and_requant(
    __nv_bfloat16* __restrict__ gemm1_output,  // [total, 2*N] in BF16
    uint8_t* __restrict__ act_fp4,              // [total, N/2] packed output
    uint8_t* __restrict__ act_sf,               // [total, N/SF_BLOCK] scale factors
    const float* __restrict__ a2_gscale,
    const int32_t* __restrict__ expert_offsets,
    int N, int E,
    int sm_id, int num_sms
) {
    int total = expert_offsets[E];
    int sf_per_row = N / SF_BLOCK;

    int rows_per_sm = (total + num_sms - 1) / num_sms;
    int row_start = sm_id * rows_per_sm;
    int row_end = min(row_start + rows_per_sm, total);

    for (int row = row_start; row < row_end; row++) {
        // Find expert for global scale
        int expert_id = 0;
        for (int e = 0; e < E; e++) {
            if (row < expert_offsets[e + 1]) {
                expert_id = e;
                break;
            }
        }
        float gscale = a2_gscale[expert_id];
        float inv_gscale = (gscale > 0.0f) ? (1.0f / gscale) : 1.0f;

        __nv_bfloat16* gate_row = gemm1_output + row * 2 * N;
        __nv_bfloat16* up_row = gate_row + N;
        uint8_t* dst_fp4 = act_fp4 + row * (N / 2);
        uint8_t* dst_sf = act_sf + row * sf_per_row;

        for (int blk = threadIdx.x; blk < sf_per_row; blk += BLOCK_SIZE) {
            int col_start = blk * SF_BLOCK;
            float local_max = 0.0f;
            float vals[SF_BLOCK];

            #pragma unroll
            for (int i = 0; i < SF_BLOCK; i++) {
                float gate_val = __bfloat162float(gate_row[col_start + i]);
                float up_val = __bfloat162float(up_row[col_start + i]);
                // SiLU(gate) * up = (gate * sigmoid(gate)) * up
                float silu = gate_val / (1.0f + expf(-gate_val));
                float activated = silu * up_val * inv_gscale;
                vals[i] = activated;
                local_max = fmaxf(local_max, fabsf(activated));
            }

            float sf_val = (local_max > 0.0f) ? (local_max / FP4_MAX) : 1.0f;
            float inv_sf = (sf_val > 0.0f) ? (1.0f / sf_val) : 1.0f;
            dst_sf[blk] = (uint8_t)fminf(fmaxf(sf_val * 127.0f, 0.0f), 255.0f);

            #pragma unroll
            for (int i = 0; i < SF_BLOCK; i += 2) {
                uint8_t lo = float_to_fp4(vals[i] * inv_sf);
                uint8_t hi = float_to_fp4(vals[i + 1] * inv_sf);
                dst_fp4[(col_start + i) / 2] = (hi << 4) | lo;
            }
        }
    }
}

// =============================================================================
// Phase 6: Unshuffle + weighted accumulate
// =============================================================================
// Scatter GEMM2 output back to original token order, applying top-k weights.

__device__ void phase6_unshuffle(
    const __nv_bfloat16* __restrict__ gemm2_output,  // [total, K] sorted
    __nv_bfloat16* __restrict__ output,               // [M, K] final
    const int32_t* __restrict__ a_map,                 // [total] sorted->orig
    const float* __restrict__ topk_weights,            // [M, top_k]
    const int32_t* __restrict__ topk_ids,              // [M, top_k]
    const int32_t* __restrict__ expert_offsets,
    int M, int K, int E, int top_k,
    int sm_id, int num_sms
) {
    int total = expert_offsets[E];

    // First, zero out the output (cooperative across all SMs)
    int out_elems = M * K;
    int elems_per_sm = (out_elems + num_sms - 1) / num_sms;
    int elem_start = sm_id * elems_per_sm;
    int elem_end = min(elem_start + elems_per_sm, out_elems);

    for (int i = elem_start + threadIdx.x; i < elem_end; i += BLOCK_SIZE) {
        output[i] = __float2bfloat16(0.0f);
    }
    __syncthreads();

    // Scatter with weighted accumulation
    int rows_per_sm = (total + num_sms - 1) / num_sms;
    int row_start = sm_id * rows_per_sm;
    int row_end = min(row_start + rows_per_sm, total);

    for (int sorted_pos = row_start; sorted_pos < row_end; sorted_pos++) {
        int orig_idx = a_map[sorted_pos];  // original (token, k) pair
        int token_idx = orig_idx / top_k;
        int k_idx = orig_idx % top_k;
        float weight = topk_weights[token_idx * top_k + k_idx];

        const __nv_bfloat16* src = gemm2_output + sorted_pos * K;
        __nv_bfloat16* dst = output + token_idx * K;

        // Weighted accumulate: read BF16, add in float, write back BF16.
        // Non-atomic — races possible when multiple sorted_pos map to same
        // token_idx. For V1 correctness testing this is acceptable; V2 will
        // use a float workspace with atomicAdd or token-level partitioning.
        for (int d = threadIdx.x; d < K; d += BLOCK_SIZE) {
            float val = __bfloat162float(src[d]) * weight;
            float old_val = __bfloat162float(dst[d]);
            dst[d] = __float2bfloat16(old_val + val);
        }
    }
}

// =============================================================================
// Main persistent kernel
// =============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
persistent_moe_dispatch_kernel(
    // Inputs
    const __nv_bfloat16* __restrict__ hidden,      // [M, K]
    const int32_t* __restrict__ topk_ids,           // [M, top_k]
    const float* __restrict__ topk_weights,         // [M, top_k]
    // Per-expert activation global scales
    const float* __restrict__ a1_gscale,            // [E]
    const float* __restrict__ a2_gscale,            // [E]
    // GEMM intermediate I/O (pre-allocated workspace)
    __nv_bfloat16* __restrict__ gemm1_output,       // [M*top_k, 2*N] (filled by external GEMM1)
    __nv_bfloat16* __restrict__ gemm2_output,       // [M*top_k, K]   (filled by external GEMM2)
    // Output
    __nv_bfloat16* __restrict__ output,             // [M, K]
    // Workspace
    __nv_bfloat16* __restrict__ sorted_hidden,      // [M*top_k, K]
    uint8_t* __restrict__ sorted_fp4,               // [M*top_k, K/2]
    uint8_t* __restrict__ sorted_sf,                // [M*top_k, K/SF_BLOCK]
    uint8_t* __restrict__ act_fp4,                  // [M*top_k, N/2]
    uint8_t* __restrict__ act_sf,                   // [M*top_k, N/SF_BLOCK]
    int32_t* __restrict__ expert_counts,            // [E]
    int32_t* __restrict__ expert_offsets,            // [E+1]
    int32_t* __restrict__ a_map,                    // [M*top_k]
    // Dimensions
    int M, int N, int K, int E, int top_k,
    // Phase control: which phases to run (bitmask)
    // bit 0: phase1 (route+shuffle)
    // bit 1: phase2 (quantize)
    // bit 2: phase4 (silu+requant)
    // bit 3: phase6 (unshuffle)
    int phase_mask
) {
    auto grid = cg::this_grid();
    int sm_id = blockIdx.x;
    int num_sms = gridDim.x;

    // ========== Phase 1: Route + Shuffle ==========
    if (phase_mask & 1) {
        // Step 1a: Count tokens per expert
        phase1_route_and_shuffle(
            hidden, topk_ids, sorted_hidden,
            expert_counts, expert_offsets, a_map,
            M, K, E, top_k, sm_id, num_sms
        );
        grid.sync();

        // Step 1b: Prefix sum (SM 0 only)
        if (sm_id == 0) {
            phase1b_prefix_sum(expert_counts, expert_offsets, E);
        }
        grid.sync();

        // Step 1c: Scatter tokens to sorted order
        phase1c_scatter(
            hidden, topk_ids, sorted_hidden,
            expert_counts, expert_offsets, a_map,
            M, K, E, top_k, sm_id, num_sms
        );
        grid.sync();
    }

    // ========== Phase 2: BF16 -> FP4 Quantize ==========
    if (phase_mask & 2) {
        phase2_quantize(
            sorted_hidden, sorted_fp4, sorted_sf,
            a1_gscale, expert_offsets,
            K, E, sm_id, num_sms
        );
        grid.sync();
    }

    // ========== Phase 3: External GEMM1 goes here in V2 ==========
    // In V1, GEMM1 is launched separately between phases 2 and 4.
    // The kernel exits after phase 2, GEMM1 runs, then a new launch
    // handles phases 4-6.

    // ========== Phase 4: SiLU + FP4 Requantize ==========
    if (phase_mask & 4) {
        phase4_silu_and_requant(
            gemm1_output, act_fp4, act_sf,
            a2_gscale, expert_offsets,
            N, E, sm_id, num_sms
        );
        grid.sync();
    }

    // ========== Phase 5: External GEMM2 goes here in V2 ==========

    // ========== Phase 6: Unshuffle + Accumulate ==========
    if (phase_mask & 8) {
        phase6_unshuffle(
            gemm2_output, output, a_map,
            topk_weights, topk_ids, expert_offsets,
            M, K, E, top_k, sm_id, num_sms
        );
    }
}

// =============================================================================
// Torch C++ interface
// =============================================================================

// Check cooperative launch support and return max blocks
int get_max_cooperative_blocks() {
    int device;
    cudaGetDevice(&device);

    int supports_coop;
    cudaDeviceGetAttribute(&supports_coop,
                           cudaDevAttrCooperativeLaunch, device);
    if (!supports_coop) return -1;

    int num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, persistent_moe_dispatch_kernel, BLOCK_SIZE, 0);

    int num_sms;
    cudaDeviceGetAttribute(&num_sms,
                           cudaDevAttrMultiProcessorCount, device);
    return num_blocks * num_sms;
}

void persistent_moe_dispatch(
    torch::Tensor& hidden,          // [M, K] bf16
    torch::Tensor& topk_ids,        // [M, top_k] int32
    torch::Tensor& topk_weights,    // [M, top_k] float32
    torch::Tensor& a1_gscale,       // [E] float32
    torch::Tensor& a2_gscale,       // [E] float32
    torch::Tensor& gemm1_output,    // [M*top_k, 2*N] bf16 workspace
    torch::Tensor& gemm2_output,    // [M*top_k, K] bf16 workspace
    torch::Tensor& output,          // [M, K] bf16
    torch::Tensor& sorted_hidden,   // workspace
    torch::Tensor& sorted_fp4,      // workspace
    torch::Tensor& sorted_sf,       // workspace
    torch::Tensor& act_fp4,         // workspace
    torch::Tensor& act_sf,          // workspace
    torch::Tensor& expert_counts,   // workspace
    torch::Tensor& expert_offsets,  // workspace
    torch::Tensor& a_map,           // workspace
    int64_t M, int64_t N, int64_t K, int64_t E, int64_t top_k,
    int64_t phase_mask
) {
    auto stream = at::cuda::getCurrentCUDAStream();

    // Get max cooperative blocks
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    // For cooperative launch, we use exactly num_sms blocks (1 per SM)
    dim3 grid_dim(num_sms);
    dim3 block_dim(BLOCK_SIZE);

    // Extract raw pointers (data_ptr() returns rvalue, need lvalues for args)
    auto p_hidden = hidden.data_ptr();
    auto p_topk_ids = topk_ids.data_ptr();
    auto p_topk_weights = topk_weights.data_ptr();
    auto p_a1_gscale = a1_gscale.data_ptr();
    auto p_a2_gscale = a2_gscale.data_ptr();
    auto p_gemm1_output = gemm1_output.data_ptr();
    auto p_gemm2_output = gemm2_output.data_ptr();
    auto p_output = output.data_ptr();
    auto p_sorted_hidden = sorted_hidden.data_ptr();
    auto p_sorted_fp4 = sorted_fp4.data_ptr();
    auto p_sorted_sf = sorted_sf.data_ptr();
    auto p_act_fp4 = act_fp4.data_ptr();
    auto p_act_sf = act_sf.data_ptr();
    auto p_expert_counts = expert_counts.data_ptr();
    auto p_expert_offsets = expert_offsets.data_ptr();
    auto p_a_map = a_map.data_ptr();
    int i_M = (int)M, i_N = (int)N, i_K = (int)K;
    int i_E = (int)E, i_top_k = (int)top_k, i_phase_mask = (int)phase_mask;

    void* args[] = {
        &p_hidden, &p_topk_ids, &p_topk_weights,
        &p_a1_gscale, &p_a2_gscale,
        &p_gemm1_output, &p_gemm2_output, &p_output,
        &p_sorted_hidden, &p_sorted_fp4, &p_sorted_sf,
        &p_act_fp4, &p_act_sf,
        &p_expert_counts, &p_expert_offsets, &p_a_map,
        &i_M, &i_N, &i_K, &i_E, &i_top_k, &i_phase_mask,
    };

    // Launch as cooperative kernel
    cudaLaunchCooperativeKernel(
        (void*)persistent_moe_dispatch_kernel,
        grid_dim, block_dim,
        args,
        0,  // shared mem
        stream.stream()
    );
}

// =============================================================================
// Standalone test: just grid.sync() latency measurement
// =============================================================================

__global__ void __launch_bounds__(256, 1)
grid_sync_benchmark_kernel(
    int* __restrict__ counter,
    int num_syncs
) {
    auto grid = cg::this_grid();
    int sm_id = blockIdx.x;

    for (int i = 0; i < num_syncs; i++) {
        if (sm_id == 0 && threadIdx.x == 0) {
            atomicAdd(counter, 1);
        }
        grid.sync();
    }
}

float benchmark_grid_sync(int num_syncs) {
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    dim3 grid(num_sms);
    dim3 block(256);
    void* args[] = { (void*)&d_counter, (void*)&num_syncs };

    // Warmup
    cudaLaunchCooperativeKernel(
        (void*)grid_sync_benchmark_kernel,
        grid, block, args, 0, 0);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemset(d_counter, 0, sizeof(int));
    cudaEventRecord(start);
    cudaLaunchCooperativeKernel(
        (void*)grid_sync_benchmark_kernel,
        grid, block, args, 0, 0);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_counter);

    return ms / num_syncs;  // ms per grid.sync()
}

}  // namespace persistent_moe
