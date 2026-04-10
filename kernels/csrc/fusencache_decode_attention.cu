// SPDX-License-Identifier: Apache-2.0
//
// FusenCache Decode Attention — CUDA C++ kernel for CUDA graph capture.
//
// Translates the Triton _universal_decode_stage1 + _universal_decode_stage2
// kernels into a single fused CUDA kernel that can be captured in CUDA graphs,
// eliminating the ~1000x Triton JIT overhead during graph replay.
//
// Target config: k4v4b64 (4-bit K, 4-bit V, scale_block=64)
// GQA: 16 query heads, 8 KV heads (group_size=2)
// Head dims: 256 (sliding) or 512 (global), template-specialized.
//
// Algorithm:
//   Stage 1 (split-K): For each (batch, head_group, split):
//     - Page table lookup -> load packed K/V uint8 + FP16 scales
//     - Dequant 4-bit: unpack nibbles, subtract offset (7.5), multiply by scale
//     - QK^T via split even/odd (packed 4-bit needs two half-dim dots)
//     - Optional logits_soft_cap: cap * tanh(score / cap)
//     - Online softmax accumulation
//     - V weighted sum (also split even/odd for 4-bit V)
//     -> Store partial result + log-sum-exp to mid_out
//
//   Stage 2 (reduce): For each (batch, head):
//     - Merge partial results from all splits via log-sum-exp
//     -> Store final output

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace fusencache {

// ============================================================
// Stage 1: Split-KV decode with 4-bit dequant
// ============================================================
//
// Grid: (batch, num_head_groups, num_kv_splits)
// Block: BLOCK_THREADS threads
//
// Each thread block processes one (batch, head_group, split) tuple.
// For 4-bit K: Q is split into even/odd halves, K nibbles unpacked.
// For 4-bit V: V nibbles unpacked into even/odd, accumulated separately.
//
// Template params:
//   HEAD_DIM: 256 or 512
//   BLOCK_KV: KV tokens per iteration (16)
//   BLOCK_THREADS: threads per block (128 or 256)
//   LOGITS_CAP: 0 = disabled, otherwise soft cap value

template <int HEAD_DIM, int BLOCK_KV = 16, int BLOCK_THREADS = 128>
__global__ void decode_stage1_k4v4(
    const __nv_bfloat16* __restrict__ query,      // [B, Hq, D]
    const uint8_t* __restrict__ kv_cache,          // [num_blocks, page_size, Hk, slot_bytes]
    const __half* __restrict__ scales,              // [max_slots, Hk, num_scale_blocks, 2] fp16
    const int32_t* __restrict__ block_table,       // [B, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,          // [B]
    float* __restrict__ mid_out,                   // [B, Hq, num_splits, D+1]
    // Strides (in elements, not bytes)
    int64_t stride_qb, int64_t stride_qh,
    int64_t stride_cache_block, int64_t stride_cache_pos, int64_t stride_cache_head,
    int64_t stride_bt_b,
    int64_t stride_mid_b, int64_t stride_mid_h, int64_t stride_mid_s,
    int64_t stride_sc_slot, int64_t stride_sc_head, int64_t stride_sc_block,
    int64_t stride_sc_kv,
    // Params
    float sm_scale,
    float logits_soft_cap,
    int num_kv_splits,
    int page_size,
    int kv_group_size,
    int q_head_num,
    int k_region_bytes,   // = HEAD_DIM / 2 for 4-bit K
    int v_region_start,   // byte offset to V region in slot
    float k_offset,       // 7.5 for 4-bit
    float v_offset,       // 7.5 for 4-bit
    int scale_block_store // min(k_scale_block, v_scale_block)
) {
    const int cur_batch = blockIdx.x;
    const int cur_head_group = blockIdx.y;
    const int split_kv_id = blockIdx.z;

    // Map head group to KV head and query heads
    const int cur_kv_head = cur_head_group / ((kv_group_size + 1) / 2);  // integer ceil
    // For simplicity with group_size=2, each head_group maps to 2 query heads
    const int valid_block_h = min(2, kv_group_size);  // typically 2
    const int first_q_head = cur_head_group * valid_block_h;

    // Sequence length and split range
    const int seq_len = seq_lens[cur_batch];
    const int kv_len = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = kv_len * split_kv_id;
    const int split_end = min(split_start + kv_len, seq_len);

    if (split_start >= split_end) {
        // Write sentinel: lse = -inf so this split is ignored in stage2
        for (int h = 0; h < valid_block_h; h++) {
            int head_idx = first_q_head + h;
            if (head_idx >= q_head_num) break;
            int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                             + (int64_t)head_idx * stride_mid_h
                             + (int64_t)split_kv_id * stride_mid_s;
            mid_out[mid_base + HEAD_DIM] = -1e30f;  // lse = -inf
        }
        return;
    }

    constexpr int HALF_D = HEAD_DIM / 2;
    const int tid = threadIdx.x;

    // Shared memory for query vectors (even and odd halves for both heads)
    // Layout: [valid_block_h][2][HALF_D] where [2] = {even, odd}
    constexpr int Q_SMEM_SIZE = 2 * 2 * HALF_D;  // 2 heads * 2 halves * HALF_D
    __shared__ float s_q[Q_SMEM_SIZE];

    // Load query into shared memory
    // q_even[h][i] = query[batch, first_q_head + h, 2*i]
    // q_odd[h][i]  = query[batch, first_q_head + h, 2*i + 1]
    for (int h = 0; h < valid_block_h; h++) {
        int head_idx = first_q_head + h;
        if (head_idx >= q_head_num) break;
        const __nv_bfloat16* q_ptr = query + (int64_t)cur_batch * stride_qb
                                           + (int64_t)head_idx * stride_qh;
        float* q_even = s_q + h * 2 * HALF_D;
        float* q_odd  = s_q + h * 2 * HALF_D + HALF_D;
        for (int i = tid; i < HALF_D; i += BLOCK_THREADS) {
            if (i < HALF_D) {
                q_even[i] = __bfloat162float(q_ptr[2 * i]);
                q_odd[i]  = __bfloat162float(q_ptr[2 * i + 1]);
            }
        }
    }
    __syncthreads();

    // Per-head accumulators (in registers, each thread maintains partial sums)
    // For 4-bit V: accumulate even and odd halves separately
    // Each thread handles a subset of the HALF_D output dimensions
    // Strategy: thread tid handles dims [tid, tid + BLOCK_THREADS, tid + 2*BLOCK_THREADS, ...]

    // Per-head softmax state
    float e_max[2];
    float e_sum[2];
    e_max[0] = e_max[1] = -1e30f;
    e_sum[0] = e_sum[1] = 0.0f;

    // Per-head, per-dimension accumulators
    // Each thread owns HALF_D / BLOCK_THREADS dimensions (rounded up)
    constexpr int DIMS_PER_THREAD = (HALF_D + BLOCK_THREADS - 1) / BLOCK_THREADS;
    float acc_even[2][DIMS_PER_THREAD];
    float acc_odd[2][DIMS_PER_THREAD];

    #pragma unroll
    for (int h = 0; h < 2; h++) {
        #pragma unroll
        for (int d = 0; d < DIMS_PER_THREAD; d++) {
            acc_even[h][d] = 0.0f;
            acc_odd[h][d] = 0.0f;
        }
    }

    // Iterate over KV tokens in this split
    for (int kv_start = split_start; kv_start < split_end; kv_start += BLOCK_KV) {
        int kv_count = min(BLOCK_KV, split_end - kv_start);

        // For each KV token in this block, compute QK^T scores
        // All threads cooperate to compute scores for each token

        // Process each KV token
        for (int kv_idx = 0; kv_idx < kv_count; kv_idx++) {
            int kv_pos = kv_start + kv_idx;

            // Page table lookup
            int page_idx = kv_pos / page_size;
            int page_off = kv_pos % page_size;
            int block_num = block_table[cur_batch * stride_bt_b + page_idx];

            int64_t slot_base = (int64_t)block_num * stride_cache_block
                              + (int64_t)page_off * stride_cache_pos
                              + (int64_t)cur_kv_head * stride_cache_head;

            // Flat slot for scale lookup
            int64_t flat_slot = (int64_t)block_num * page_size + page_off;
            int64_t sc_base = flat_slot * stride_sc_slot + (int64_t)cur_kv_head * stride_sc_head;

            // Compute QK^T for each query head using parallel reduction
            // 4-bit K: each byte has 2 values (lo nibble, hi nibble)
            // QK = sum_i(q_even[i] * k_lo[i]) + sum_i(q_odd[i] * k_hi[i])
            float qk_partial[2] = {0.0f, 0.0f};

            for (int i = tid; i < HALF_D; i += BLOCK_THREADS) {
                // Load packed K byte
                uint8_t k_packed = kv_cache[slot_base + i];
                float k_lo = (float)(k_packed & 0xF) - k_offset;
                float k_hi = (float)((k_packed >> 4) & 0xF) - k_offset;

                // Load K scale
                int sc_idx = i / (scale_block_store / 2);
                float k_sc = __half2float(scales[sc_base + sc_idx * stride_sc_block]);
                k_lo *= k_sc;
                k_hi *= k_sc;

                // Dot product contributions
                for (int h = 0; h < valid_block_h; h++) {
                    float* q_even = s_q + h * 2 * HALF_D;
                    float* q_odd  = s_q + h * 2 * HALF_D + HALF_D;
                    qk_partial[h] += q_even[i] * k_lo + q_odd[i] * k_hi;
                }
            }

            // Warp-level reduction for QK scores
            // Use CUB or manual warp reduce
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                // Warp shuffle reduction
                for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                    qk_partial[h] += __shfl_xor_sync(0xffffffff, qk_partial[h], offset);
                }
            }

            // Now lane 0 of each warp has partial sum. Need block-level reduction.
            // Use shared memory for cross-warp reduction
            __shared__ float s_qk_warp[2][32];  // max 32 warps

            int warp_id = tid / 32;
            int lane_id = tid % 32;

            for (int h = 0; h < valid_block_h; h++) {
                if (lane_id == 0) {
                    s_qk_warp[h][warp_id] = qk_partial[h];
                }
            }
            __syncthreads();

            float qk[2] = {0.0f, 0.0f};
            int num_warps = BLOCK_THREADS / 32;
            if (tid < 32) {
                for (int h = 0; h < valid_block_h; h++) {
                    float val = (tid < num_warps) ? s_qk_warp[h][tid] : 0.0f;
                    for (int offset = 16; offset > 0; offset /= 2) {
                        val += __shfl_xor_sync(0xffffffff, val, offset);
                    }
                    qk[h] = val;
                }
            }

            // Broadcast final QK scores to all threads
            __shared__ float s_qk_final[2];
            if (tid == 0) {
                for (int h = 0; h < valid_block_h; h++) {
                    float score = qk[h] * sm_scale;

                    // Logits soft cap
                    if (logits_soft_cap > 0.0f) {
                        float x = score / logits_soft_cap;
                        float e2x = expf(2.0f * x);
                        float th = 1.0f - 2.0f / (e2x + 1.0f);
                        score = logits_soft_cap * th;
                    }

                    s_qk_final[h] = score;
                }
            }
            __syncthreads();

            // Online softmax update + V accumulation
            for (int h = 0; h < valid_block_h; h++) {
                int head_idx = first_q_head + h;
                if (head_idx >= q_head_num) break;

                float score = s_qk_final[h];
                float new_max = fmaxf(e_max[h], score);
                float rescale = expf(e_max[h] - new_max);
                float p = expf(score - new_max);

                // Rescale existing accumulators
                for (int d = 0; d < DIMS_PER_THREAD; d++) {
                    acc_even[h][d] *= rescale;
                    acc_odd[h][d]  *= rescale;
                }

                // Load V and accumulate (4-bit V: nibble unpack)
                for (int d = 0; d < DIMS_PER_THREAD; d++) {
                    int dim_idx = tid + d * BLOCK_THREADS;
                    if (dim_idx < HALF_D) {
                        uint8_t v_packed = kv_cache[slot_base + v_region_start + dim_idx];
                        float v_lo = (float)(v_packed & 0xF) - v_offset;
                        float v_hi = (float)((v_packed >> 4) & 0xF) - v_offset;

                        // V scale
                        int v_sc_idx = dim_idx / (scale_block_store / 2);
                        float v_sc = __half2float(scales[sc_base + v_sc_idx * stride_sc_block + stride_sc_kv]);
                        v_lo *= v_sc;
                        v_hi *= v_sc;

                        acc_even[h][d] += p * v_lo;
                        acc_odd[h][d]  += p * v_hi;
                    }
                }

                e_sum[h] = e_sum[h] * rescale + p;
                e_max[h] = new_max;
            }
            __syncthreads();  // sync before next KV token uses shared mem
        }
    }

    // Store results to mid_out
    for (int h = 0; h < valid_block_h; h++) {
        int head_idx = first_q_head + h;
        if (head_idx >= q_head_num) break;

        int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                         + (int64_t)head_idx * stride_mid_h
                         + (int64_t)split_kv_id * stride_mid_s;

        // Normalize by e_sum
        float safe_sum = (e_sum[h] > 0.0f) ? e_sum[h] : 1.0f;
        float inv_sum = 1.0f / safe_sum;

        // Store interleaved: even dims at 2*i, odd dims at 2*i+1
        for (int d = 0; d < DIMS_PER_THREAD; d++) {
            int dim_idx = tid + d * BLOCK_THREADS;
            if (dim_idx < HALF_D) {
                mid_out[mid_base + dim_idx * 2]     = acc_even[h][d] * inv_sum;
                mid_out[mid_base + dim_idx * 2 + 1] = acc_odd[h][d]  * inv_sum;
            }
        }

        // Store log-sum-exp (one thread per head)
        if (tid == 0) {
            float lse = e_max[h] + logf(fmaxf(e_sum[h], 1e-30f));
            mid_out[mid_base + HEAD_DIM] = lse;
        }
    }
}


// ============================================================
// Stage 2: Reduce across splits
// ============================================================
// Grid: (batch, num_q_heads)
// Block: 256 threads

template <int HEAD_DIM, int BLOCK_THREADS = 256>
__global__ void decode_stage2(
    const float* __restrict__ mid_out,   // [B, Hq, num_splits, D+1]
    __nv_bfloat16* __restrict__ output,  // [B, Hq, D]
    const int32_t* __restrict__ seq_lens,
    int64_t stride_mid_b, int64_t stride_mid_h, int64_t stride_mid_s,
    int64_t stride_out_b, int64_t stride_out_h,
    int num_kv_splits
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;
    const int seq_len = seq_lens[bid];

    int64_t mid_base = (int64_t)bid * stride_mid_b + (int64_t)hid * stride_mid_h;

    // Each thread processes a subset of dimensions
    constexpr int DIMS_PER_THREAD = (HEAD_DIM + BLOCK_THREADS - 1) / BLOCK_THREADS;

    float acc[DIMS_PER_THREAD];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; d++) acc[d] = 0.0f;

    float e_max = -1e30f;
    float e_sum = 0.0f;

    for (int s = 0; s < num_kv_splits; s++) {
        int sl = (seq_len + num_kv_splits - 1) / num_kv_splits;
        int split_start = sl * s;
        int split_end_v = min(split_start + sl, seq_len);
        if (split_start >= split_end_v) continue;

        int64_t off = mid_base + (int64_t)s * stride_mid_s;
        float lse = mid_out[off + HEAD_DIM];

        float new_max = fmaxf(lse, e_max);
        float r = expf(e_max - new_max);
        float w = expf(lse - new_max);

        for (int d = 0; d < DIMS_PER_THREAD; d++) {
            int dim_idx = tid + d * BLOCK_THREADS;
            if (dim_idx < HEAD_DIM) {
                float tv = mid_out[off + dim_idx];
                acc[d] = acc[d] * r + w * tv;
            }
        }
        e_sum = e_sum * r + w;
        e_max = new_max;
    }

    // Normalize and store
    float safe_sum = (e_sum > 0.0f) ? e_sum : 1.0f;
    float inv_sum = 1.0f / safe_sum;

    int64_t out_base = (int64_t)bid * stride_out_b + (int64_t)hid * stride_out_h;
    for (int d = 0; d < DIMS_PER_THREAD; d++) {
        int dim_idx = tid + d * BLOCK_THREADS;
        if (dim_idx < HEAD_DIM) {
            output[out_base + dim_idx] = __float2bfloat16(acc[d] * inv_sum);
        }
    }
}

}  // namespace fusencache


// ============================================================
// C++ entry point
// ============================================================

void fusencache_decode_attention(
    torch::Tensor& output,           // [B, Hq, D] bf16
    torch::Tensor const& query,      // [B, Hq, D] bf16
    torch::Tensor const& kv_cache,   // [num_blocks, page_size, Hk, slot_bytes] uint8
    torch::Tensor const& scales,     // [max_slots, Hk, num_scale_blocks, 2] fp16
    torch::Tensor const& block_table,// [B, max_blocks_per_seq] int32
    torch::Tensor const& seq_lens,   // [B] int32
    torch::Tensor& mid_out,          // [B, Hq, num_splits, D+1] float32
    double sm_scale,
    double logits_soft_cap,
    int64_t num_kv_splits,
    int64_t head_dim,
    int64_t num_kv_heads,
    int64_t kv_group_size,
    int64_t page_size,
    int64_t k_bits,
    int64_t v_bits,
    int64_t scale_block_k,
    int64_t scale_block_v,
    double k_offset,
    double v_offset
) {
    const int B = query.size(0);
    const int Hq = query.size(1);
    const int D = query.size(2);

    TORCH_CHECK(k_bits == 4 && v_bits == 4,
                "Only k4v4 (4-bit K, 4-bit V) is currently supported");
    TORCH_CHECK(D == 256 || D == 512,
                "Only head_dim=256 or 512 is supported, got ", D);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int k_region_bytes = D / 2;  // 4-bit: half byte per dim
    int v_region_start = k_region_bytes;
    int scale_block_store = min((int)scale_block_k, (int)scale_block_v);

    // Stage 1
    int num_head_groups = (Hq + min(2, (int)kv_group_size) - 1) / min(2, (int)kv_group_size);
    dim3 grid1(B, num_head_groups, num_kv_splits);

    // Choose block threads based on head_dim
    constexpr int BLOCK_THREADS_256 = 128;
    constexpr int BLOCK_THREADS_512 = 256;

    if (D == 256) {
        fusencache::decode_stage1_k4v4<256, 16, BLOCK_THREADS_256>
            <<<grid1, BLOCK_THREADS_256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
                kv_cache.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(scales.data_ptr()),
                block_table.data_ptr<int32_t>(),
                seq_lens.data_ptr<int32_t>(),
                mid_out.data_ptr<float>(),
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                block_table.stride(0),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                scales.stride(0), scales.stride(1), scales.stride(2), scales.stride(3),
                (float)sm_scale,
                (float)logits_soft_cap,
                (int)num_kv_splits,
                (int)page_size,
                (int)kv_group_size,
                Hq,
                k_region_bytes,
                v_region_start,
                (float)k_offset,
                (float)v_offset,
                scale_block_store
            );
    } else {
        fusencache::decode_stage1_k4v4<512, 16, BLOCK_THREADS_512>
            <<<grid1, BLOCK_THREADS_512, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
                kv_cache.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(scales.data_ptr()),
                block_table.data_ptr<int32_t>(),
                seq_lens.data_ptr<int32_t>(),
                mid_out.data_ptr<float>(),
                query.stride(0), query.stride(1),
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
                block_table.stride(0),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                scales.stride(0), scales.stride(1), scales.stride(2), scales.stride(3),
                (float)sm_scale,
                (float)logits_soft_cap,
                (int)num_kv_splits,
                (int)page_size,
                (int)kv_group_size,
                Hq,
                k_region_bytes,
                v_region_start,
                (float)k_offset,
                (float)v_offset,
                scale_block_store
            );
    }

    // Stage 2
    dim3 grid2(B, Hq);
    if (D == 256) {
        fusencache::decode_stage2<256, 256>
            <<<grid2, 256, 0, stream>>>(
                mid_out.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                seq_lens.data_ptr<int32_t>(),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                output.stride(0), output.stride(1),
                (int)num_kv_splits
            );
    } else {
        fusencache::decode_stage2<512, 256>
            <<<grid2, 256, 0, stream>>>(
                mid_out.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                seq_lens.data_ptr<int32_t>(),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                output.stride(0), output.stride(1),
                (int)num_kv_splits
            );
    }
}
