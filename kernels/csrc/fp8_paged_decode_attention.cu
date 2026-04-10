// SPDX-License-Identifier: Apache-2.0
//
// FP8 Paged Decode Attention -- high-bandwidth CUDA C++ kernel for SM120.
//
// Two-stage split-K paged decode attention with FP8 E4M3 KV cache.
// Uses vectorized uint4 (16-byte) loads for maximum HBM throughput.
//
// KV cache layout: [num_blocks, block_size, num_kv_heads, head_dim] FP8 E4M3
// Query layout: [batch, num_q_heads, head_dim] BF16

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace fp8_decode {

// ============================================================
// FP8 E4M3 -> float
// ============================================================
__device__ __forceinline__ float fp8_to_float(uint8_t val) {
    __nv_fp8_e4m3 fp8_val;
    *reinterpret_cast<uint8_t*>(&fp8_val) = val;
    return static_cast<float>(fp8_val);
}

// ============================================================
// Warp reduction
// ============================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================
// Stage 1: Split-KV paged FP8 decode
// ============================================================
//
// Strategy:
//   - Load K+V page into SMEM via vectorized uint4 loads
//   - QK: 8 warps, each handles 2 positions (16 total per page)
//     Each warp's 32 lanes do the full dot product over HEAD_DIM
//   - V accumulation: all 256 threads, each handles HEAD_DIM/256 dims
//
// Grid: (batch, num_q_heads, num_kv_splits)
// Block: 256 threads = 8 warps

template <int HEAD_DIM, int PAGE_SIZE = 16, int BLOCK_THREADS = 256>
__global__ void fp8_paged_decode_stage1(
    const __nv_bfloat16* __restrict__ query,
    const uint8_t* __restrict__ kv_cache_k,
    const uint8_t* __restrict__ kv_cache_v,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ mid_out,
    int64_t stride_qb, int64_t stride_qh,
    int64_t stride_kb, int64_t stride_kp, int64_t stride_kh,
    int64_t stride_vb, int64_t stride_vp, int64_t stride_vh,
    int64_t stride_bt_b,
    int64_t stride_mid_b, int64_t stride_mid_h, int64_t stride_mid_s,
    float sm_scale,
    float logits_soft_cap,
    int num_kv_splits,
    int kv_group_size,
    int num_q_heads,
    int per_head_scale
) {
    const int cur_batch = blockIdx.x;
    const int cur_q_head = blockIdx.y;
    const int split_kv_id = blockIdx.z;
    const int tid = threadIdx.x;

    if (cur_q_head >= num_q_heads) return;
    const int cur_kv_head = cur_q_head / kv_group_size;

    const int seq_len = seq_lens[cur_batch];
    const int num_pages = (seq_len + PAGE_SIZE - 1) / PAGE_SIZE;
    const int pages_per_split = (num_pages + num_kv_splits - 1) / num_kv_splits;
    const int page_start = pages_per_split * split_kv_id;
    const int page_end = min(page_start + pages_per_split, num_pages);

    if (page_start >= page_end) {
        if (tid == 0) {
            int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                             + (int64_t)cur_q_head * stride_mid_h
                             + (int64_t)split_kv_id * stride_mid_s;
            mid_out[mid_base + HEAD_DIM] = -1e30f;
        }
        return;
    }

    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    constexpr int POSITIONS_PER_WARP = (PAGE_SIZE + NUM_WARPS - 1) / NUM_WARPS;
    constexpr int QK_ITERS = (HEAD_DIM + 31) / 32;
    constexpr int DIMS_PER_THREAD = (HEAD_DIM + BLOCK_THREADS - 1) / BLOCK_THREADS;
    constexpr int PAGE_BYTES = PAGE_SIZE * HEAD_DIM;
    constexpr int UINT4_PER_PAGE = PAGE_BYTES / 16;
    constexpr int LOADS_PER_THREAD = (UINT4_PER_PAGE + BLOCK_THREADS - 1) / BLOCK_THREADS;

    // Load query for QK dot product (warp-parallel layout)
    float q_qk[QK_ITERS];
    {
        const __nv_bfloat16* q_ptr = query + (int64_t)cur_batch * stride_qb
                                           + (int64_t)cur_q_head * stride_qh;
        #pragma unroll
        for (int i = 0; i < QK_ITERS; i++) {
            int dim = lane_id + i * 32;
            q_qk[i] = (dim < HEAD_DIM) ? __bfloat162float(q_ptr[dim]) : 0.0f;
        }
    }

    float ks = per_head_scale ? k_scale[cur_kv_head] : k_scale[0];
    float vs = per_head_scale ? v_scale[cur_kv_head] : v_scale[0];
    float qk_scale = sm_scale * ks;

    // ---- Shared memory ----
    extern __shared__ char smem_raw[];
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw);
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + PAGE_BYTES);
    float* smem_scores = reinterpret_cast<float*>(smem_raw + 2 * PAGE_BYTES);

    // ---- Accumulators ----
    float e_max = -1e30f;
    float e_sum = 0.0f;
    float acc[DIMS_PER_THREAD];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; d++) acc[d] = 0.0f;

    // ---- Main loop over pages ----
    for (int page = page_start; page < page_end; page++) {
        int phys_block = block_table[cur_batch * stride_bt_b + page];
        int abs_start = page * PAGE_SIZE;
        int valid_count = min(PAGE_SIZE, seq_len - abs_start);

        // ---- Load K+V pages via vectorized uint4 ----
        {
            int64_t k_head_base = (int64_t)phys_block * stride_kb
                                + (int64_t)cur_kv_head * stride_kh;
            int64_t v_head_base = (int64_t)phys_block * stride_vb
                                + (int64_t)cur_kv_head * stride_vh;

            #pragma unroll
            for (int load_idx = 0; load_idx < LOADS_PER_THREAD; load_idx++) {
                int flat_uint4 = tid + load_idx * BLOCK_THREADS;
                if (flat_uint4 < UINT4_PER_PAGE) {
                    int flat_byte = flat_uint4 * 16;
                    int pos = flat_byte / HEAD_DIM;
                    int byte_off = flat_byte % HEAD_DIM;

                    uint4* dst_k = reinterpret_cast<uint4*>(smem_k + flat_byte);
                    uint4* dst_v = reinterpret_cast<uint4*>(smem_v + flat_byte);

                    if (pos < valid_count) {
                        const uint4* src_k = reinterpret_cast<const uint4*>(
                            kv_cache_k + k_head_base + (int64_t)pos * stride_kp + byte_off);
                        const uint4* src_v = reinterpret_cast<const uint4*>(
                            kv_cache_v + v_head_base + (int64_t)pos * stride_vp + byte_off);
                        *dst_k = *src_k;
                        *dst_v = *src_v;
                    } else {
                        uint4 zero = {0, 0, 0, 0};
                        *dst_k = zero;
                        *dst_v = zero;
                    }
                }
            }
        }
        __syncthreads();

        // ---- QK^T scores: warp-parallel ----
        {
            #pragma unroll
            for (int p = 0; p < POSITIONS_PER_WARP; p++) {
                int pos = warp_id * POSITIONS_PER_WARP + p;
                float dot = 0.0f;

                if (pos < valid_count) {
                    const uint8_t* k_row = smem_k + pos * HEAD_DIM;
                    #pragma unroll
                    for (int i = 0; i < QK_ITERS; i++) {
                        int dim = lane_id + i * 32;
                        if (dim < HEAD_DIM) {
                            dot += q_qk[i] * fp8_to_float(k_row[dim]);
                        }
                    }
                    dot = warp_reduce_sum(dot);
                }

                if (lane_id == 0 && pos < PAGE_SIZE) {
                    float score;
                    if (pos < valid_count) {
                        score = dot * qk_scale;
                        if (logits_soft_cap > 0.0f) {
                            score = logits_soft_cap * tanhf(score / logits_soft_cap);
                        }
                    } else {
                        score = -1e30f;
                    }
                    smem_scores[pos] = score;
                }
            }
        }
        __syncthreads();

        // ---- Online softmax + V accumulation ----
        for (int pos = 0; pos < valid_count; pos++) {
            float score = smem_scores[pos];
            float new_max = fmaxf(e_max, score);
            float rescale = expf(e_max - new_max);
            float p = expf(score - new_max);

            const uint8_t* v_row = smem_v + pos * HEAD_DIM;
            #pragma unroll
            for (int d = 0; d < DIMS_PER_THREAD; d++) {
                int dim_idx = tid + d * BLOCK_THREADS;
                if (dim_idx < HEAD_DIM) {
                    acc[d] = acc[d] * rescale + p * fp8_to_float(v_row[dim_idx]) * vs;
                }
            }
            e_sum = e_sum * rescale + p;
            e_max = new_max;
        }

        __syncthreads();
    }

    // ---- Store results ----
    {
        int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                         + (int64_t)cur_q_head * stride_mid_h
                         + (int64_t)split_kv_id * stride_mid_s;

        float safe_sum = (e_sum > 0.0f) ? e_sum : 1.0f;
        float inv_sum = 1.0f / safe_sum;

        #pragma unroll
        for (int d = 0; d < DIMS_PER_THREAD; d++) {
            int dim_idx = tid + d * BLOCK_THREADS;
            if (dim_idx < HEAD_DIM) {
                mid_out[mid_base + dim_idx] = acc[d] * inv_sum;
            }
        }
        if (tid == 0) {
            mid_out[mid_base + HEAD_DIM] = e_max + logf(fmaxf(e_sum, 1e-30f));
        }
    }
}


// ============================================================
// Stage 2: Reduce across KV splits
// ============================================================
template <int HEAD_DIM, int BLOCK_THREADS = 256>
__global__ void fp8_paged_decode_stage2(
    const float* __restrict__ mid_out,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ seq_lens,
    int64_t stride_mid_b, int64_t stride_mid_h, int64_t stride_mid_s,
    int64_t stride_out_b, int64_t stride_out_h,
    int num_kv_splits,
    int page_size
) {
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;
    const int seq_len = seq_lens[bid];

    int64_t mid_base = (int64_t)bid * stride_mid_b + (int64_t)hid * stride_mid_h;
    constexpr int DIMS_PER_THREAD = (HEAD_DIM + BLOCK_THREADS - 1) / BLOCK_THREADS;

    float acc[DIMS_PER_THREAD];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; d++) acc[d] = 0.0f;

    float e_max = -1e30f;
    float e_sum = 0.0f;
    int num_pages = (seq_len + page_size - 1) / page_size;

    for (int s = 0; s < num_kv_splits; s++) {
        int pps = (num_pages + num_kv_splits - 1) / num_kv_splits;
        int ps = pps * s;
        int pe = min(ps + pps, num_pages);
        if (ps >= pe) continue;

        int64_t off = mid_base + (int64_t)s * stride_mid_s;
        float lse = mid_out[off + HEAD_DIM];

        float new_max = fmaxf(lse, e_max);
        float r = expf(e_max - new_max);
        float w = expf(lse - new_max);

        #pragma unroll
        for (int d = 0; d < DIMS_PER_THREAD; d++) {
            int dim_idx = tid + d * BLOCK_THREADS;
            if (dim_idx < HEAD_DIM) {
                acc[d] = acc[d] * r + w * mid_out[off + dim_idx];
            }
        }
        e_sum = e_sum * r + w;
        e_max = new_max;
    }

    float safe_sum = (e_sum > 0.0f) ? e_sum : 1.0f;
    float inv_sum = 1.0f / safe_sum;

    int64_t out_base = (int64_t)bid * stride_out_b + (int64_t)hid * stride_out_h;
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; d++) {
        int dim_idx = tid + d * BLOCK_THREADS;
        if (dim_idx < HEAD_DIM) {
            output[out_base + dim_idx] = __float2bfloat16(acc[d] * inv_sum);
        }
    }
}

}  // namespace fp8_decode


// ============================================================
// C++ entry point
// ============================================================

void fp8_paged_decode_attention(
    torch::Tensor& output,
    torch::Tensor const& query,
    torch::Tensor const& kv_cache_k,
    torch::Tensor const& kv_cache_v,
    torch::Tensor const& k_scale,
    torch::Tensor const& v_scale,
    torch::Tensor const& block_table,
    torch::Tensor const& seq_lens,
    torch::Tensor& mid_out,
    double sm_scale,
    double logits_soft_cap,
    int64_t num_kv_splits,
    int64_t head_dim,
    int64_t num_kv_heads,
    int64_t kv_group_size,
    int64_t page_size,
    int64_t per_head_scale
) {
    const int B = query.size(0);
    const int Hq = query.size(1);
    const int D = head_dim;

    TORCH_CHECK(D == 128 || D == 256,
                "Only head_dim=128 or 256 supported, got ", D);
    TORCH_CHECK(page_size == 16,
                "Only page_size=16 supported, got ", page_size);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 grid1(B, Hq, num_kv_splits);
    constexpr int BLOCK_THREADS = 256;
    constexpr int PS = 16;

    auto launch = [&]<int HD>() {
        int smem_bytes = 2 * PS * HD + PS * sizeof(float);
        fp8_decode::fp8_paged_decode_stage1<HD, PS, BLOCK_THREADS>
            <<<grid1, BLOCK_THREADS, smem_bytes, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
                reinterpret_cast<const uint8_t*>(kv_cache_k.data_ptr()),
                reinterpret_cast<const uint8_t*>(kv_cache_v.data_ptr()),
                k_scale.data_ptr<float>(),
                v_scale.data_ptr<float>(),
                block_table.data_ptr<int32_t>(),
                seq_lens.data_ptr<int32_t>(),
                mid_out.data_ptr<float>(),
                query.stride(0), query.stride(1),
                kv_cache_k.stride(0), kv_cache_k.stride(1), kv_cache_k.stride(2),
                kv_cache_v.stride(0), kv_cache_v.stride(1), kv_cache_v.stride(2),
                block_table.stride(0),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                (float)sm_scale,
                (float)logits_soft_cap,
                (int)num_kv_splits,
                (int)kv_group_size,
                Hq,
                (int)per_head_scale
            );
    };

    if (D == 256) {
        launch.template operator()<256>();
    } else {
        launch.template operator()<128>();
    }

    // Stage 2
    dim3 grid2(B, Hq);
    if (D == 256) {
        fp8_decode::fp8_paged_decode_stage2<256, 256>
            <<<grid2, 256, 0, stream>>>(
                mid_out.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                seq_lens.data_ptr<int32_t>(),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                output.stride(0), output.stride(1),
                (int)num_kv_splits,
                (int)page_size
            );
    } else {
        fp8_decode::fp8_paged_decode_stage2<128, 256>
            <<<grid2, 256, 0, stream>>>(
                mid_out.data_ptr<float>(),
                reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
                seq_lens.data_ptr<int32_t>(),
                mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
                output.stride(0), output.stride(1),
                (int)num_kv_splits,
                (int)page_size
            );
    }
}
