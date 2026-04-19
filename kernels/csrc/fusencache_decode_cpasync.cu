// SPDX-License-Identifier: Apache-2.0
//
// FusenCache decode attention — cp.async pipelined prototype (SM120a).
//
// Goal: replace the per-byte scalar loads of `fusencache_decode_attention.cu`
// (35 % HBM BW measured) with 16-B cp.async bulk loads, double-buffered
// across a 3-stage pipeline, to approach FA2's 93 % BW on BF16 KV.
//
// Restricted scope (prototype):
//   - k_bits = v_bits = 4 (FusenCache k4v4b64)
//   - head_dim = 256
//   - scale_block = 64 (→ D/scale_block/2 = 2 FP16 scales per K region,
//                        2 for V region, per KV token per head)
//   - kv_group_size = 2 (Gemma-4 GQA)
//   - page_size arbitrary (we use block_table indirection)
//   - single-token decode, batch up to 32
//
// NOTE on correctness: this is a speed-first prototype. It runs online
// softmax + V accumulation in the same style as the baseline kernel but
// uses the SMEM tile instead of scalar global loads. Numeric parity with
// the Triton reference is a follow-up; for now we only require
// non-NaN output at small shapes.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace fusencache_cpasync {

// ------------------------------------------------------------
// PTX wrappers for cp.async (SM80+)
// ------------------------------------------------------------

// 16-byte cp.async (cache-global) from global to shared.
// `smem_ptr`: generic-to-shared converted pointer (via __cvta_generic_to_shared).
__device__ __forceinline__ void cp_async_16B(uint32_t smem_ptr,
                                             const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// ------------------------------------------------------------
// Kernel — decode stage1 with cp.async pipelining, k4v4, D=256
// ------------------------------------------------------------

// Tile shape:
//   BLOCK_KV  = 16 KV tokens per tile
//   HEAD_DIM  = 256  (fixed)
//   HALF_D    = 128  (4-bit nibbles: 128 bytes per K-region per token)
//   STAGES    = 3    (3-stage pipeline)
//   THREADS   = 128

template <int HEAD_DIM   = 256,
          int BLOCK_KV   = 16,
          int BLOCK_THREADS = 128,
          int STAGES     = 3,
          int SCALE_BLOCK = 64>
__global__ __launch_bounds__(BLOCK_THREADS)
void decode_stage1_k4v4_cpasync(
    const __nv_bfloat16* __restrict__ query,
    const uint8_t*        __restrict__ kv_cache,
    const __half*         __restrict__ scales,
    const int32_t*        __restrict__ block_table,
    const int32_t*        __restrict__ seq_lens,
    float*                __restrict__ mid_out,
    int64_t stride_qb, int64_t stride_qh,
    int64_t stride_cache_block, int64_t stride_cache_pos, int64_t stride_cache_head,
    int64_t stride_bt_b,
    int64_t stride_mid_b, int64_t stride_mid_h, int64_t stride_mid_s,
    int64_t stride_sc_slot, int64_t stride_sc_head, int64_t stride_sc_block,
    int64_t stride_sc_kv,
    float  sm_scale,
    float  logits_soft_cap,
    int    num_kv_splits,
    int    page_size,
    int    kv_group_size,
    int    q_head_num,
    int    k_region_bytes,     // = HEAD_DIM / 2 (= 128)
    int    v_region_start,     // = HEAD_DIM / 2 (V bytes start after K bytes)
    float  k_offset,           // 7.5
    float  v_offset)           // 7.5
{
    static_assert(HEAD_DIM == 256, "prototype: HEAD_DIM=256 only");
    static_assert(BLOCK_THREADS == 128, "prototype: BLOCK_THREADS=128 only");
    static_assert(BLOCK_KV == 16, "prototype: BLOCK_KV=16 only");
    static_assert(STAGES >= 2 && STAGES <= 4, "STAGES in [2,4]");
    static_assert(SCALE_BLOCK == 64, "prototype: SCALE_BLOCK=64 only");

    constexpr int HALF_D        = HEAD_DIM / 2;          // 128
    constexpr int K_BYTES_TILE  = HALF_D * BLOCK_KV;     // 2048 B per stage per K
    constexpr int V_BYTES_TILE  = HALF_D * BLOCK_KV;     // 2048 B per stage per V
    constexpr int SCALES_PER_TOK = 2 * (HEAD_DIM / SCALE_BLOCK); // 2 K-sb + 2 V-sb (stored interleaved)
    // We pack scales into a flat per-tile smem array; size in __half elements.
    constexpr int SC_HALF_PER_TILE = SCALES_PER_TOK * BLOCK_KV; // 64

    // Align everything to 16 B for cp.async.
    extern __shared__ __align__(16) uint8_t s_raw[];
    uint8_t* s_q_bytes   = s_raw;                                    // query smem
    // query: 2 heads * HEAD_DIM floats = 2 * 256 * 4 = 2048 B
    constexpr int Q_BYTES = 2 * HEAD_DIM * sizeof(float);
    float* s_q = reinterpret_cast<float*>(s_q_bytes);

    uint8_t* s_ring = s_q_bytes + Q_BYTES;
    // Per stage: K tile (2048 B), V tile (2048 B), scales (64 * 2 B = 128 B).
    constexpr int STAGE_BYTES = K_BYTES_TILE + V_BYTES_TILE + SC_HALF_PER_TILE * (int)sizeof(__half);
    // Pad stage to multiple of 16 B (already is: 2048+2048+128 = 4224, mult of 16).

    // Ring-buffer pointers per stage
    auto stage_k    = [&](int s) -> uint8_t* { return s_ring + s * STAGE_BYTES; };
    auto stage_v    = [&](int s) -> uint8_t* { return stage_k(s) + K_BYTES_TILE; };
    auto stage_sc   = [&](int s) -> __half*  { return reinterpret_cast<__half*>(stage_v(s) + V_BYTES_TILE); };

    const int cur_batch      = blockIdx.x;
    const int cur_head_group = blockIdx.y;
    const int split_kv_id    = blockIdx.z;
    const int tid            = threadIdx.x;

    const int cur_kv_head    = cur_head_group / ((kv_group_size + 1) / 2);
    const int valid_block_h  = min(2, kv_group_size);
    const int first_q_head   = cur_head_group * valid_block_h;

    const int seq_len   = seq_lens[cur_batch];
    const int kv_len    = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = kv_len * split_kv_id;
    const int split_end   = min(split_start + kv_len, seq_len);

    if (split_start >= split_end) {
        for (int h = 0; h < valid_block_h; ++h) {
            int head_idx = first_q_head + h;
            if (head_idx >= q_head_num) break;
            int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                             + (int64_t)head_idx * stride_mid_h
                             + (int64_t)split_kv_id * stride_mid_s;
            mid_out[mid_base + HEAD_DIM] = -1e30f;
        }
        return;
    }

    // --- Load query into SMEM (FP32) ---
    // Same layout as baseline kernel: s_q[h * 2 * HALF_D + 0..HALF_D-1] = even,
    //                                 s_q[h * 2 * HALF_D + HALF_D..]   = odd.
    for (int h = 0; h < valid_block_h; ++h) {
        int head_idx = first_q_head + h;
        if (head_idx >= q_head_num) break;
        const __nv_bfloat16* q_ptr = query
            + (int64_t)cur_batch * stride_qb
            + (int64_t)head_idx * stride_qh;
        float* q_even = s_q + h * 2 * HALF_D;
        float* q_odd  = q_even + HALF_D;
        #pragma unroll
        for (int i = tid; i < HALF_D; i += BLOCK_THREADS) {
            q_even[i] = __bfloat162float(q_ptr[2 * i]);
            q_odd[i]  = __bfloat162float(q_ptr[2 * i + 1]);
        }
    }
    __syncthreads();

    // Per-head online softmax state
    float e_max[2] = { -1e30f, -1e30f };
    float e_sum[2] = { 0.0f, 0.0f };

    // Each thread owns HALF_D / BLOCK_THREADS = 1 dim for both heads.
    constexpr int DIMS_PER_THREAD = (HALF_D + BLOCK_THREADS - 1) / BLOCK_THREADS; // = 1
    float acc_even[2][DIMS_PER_THREAD] = {{0}};
    float acc_odd [2][DIMS_PER_THREAD] = {{0}};

    // --- Pipeline prologue: enqueue up to STAGES-1 tiles ahead ---
    const int num_tiles = (split_end - split_start + BLOCK_KV - 1) / BLOCK_KV;

    auto issue_tile = [&](int tile_idx, int stage_slot) {
        // Enqueue cp.async loads for KV tile `tile_idx` into shared ring slot `stage_slot`.
        if (tile_idx >= num_tiles) return;

        int kv_start = split_start + tile_idx * BLOCK_KV;
        int kv_count = min(BLOCK_KV, split_end - kv_start);

        uint8_t*  k_dst = stage_k(stage_slot);
        uint8_t*  v_dst = stage_v(stage_slot);
        __half*   s_dst = stage_sc(stage_slot);

        // Each thread loads 16 B of K (or V, or scales).
        // 128 threads * 16 B = 2048 B per group → exactly one tile worth.
        // We'll dispatch threads:
        //   threads  0..127 → K region (HALF_D * BLOCK_KV = 2048 B)
        //   threads  0..127 → V region (separate group)
        //   threads  0..7   → scales (128 B total)

        // K load: tid covers K_BYTES_TILE / 16 = 128 chunks
        {
            int chunk = tid;                    // 0..127
            int kv_idx_in_tile = chunk / (HALF_D / 16);   // 0..15
            int dim_chunk      = chunk % (HALF_D / 16);   // 0..7  (each = 16 B)
            uint8_t* dst_ptr = k_dst + chunk * 16;

            // Source byte address in paged KV.
            const uint8_t* src_ptr;
            if (kv_idx_in_tile < kv_count) {
                int kv_pos = kv_start + kv_idx_in_tile;
                int page_idx = kv_pos / page_size;
                int page_off = kv_pos % page_size;
                int block_num = block_table[cur_batch * stride_bt_b + page_idx];
                int64_t slot_base = (int64_t)block_num * stride_cache_block
                                  + (int64_t)page_off * stride_cache_pos
                                  + (int64_t)cur_kv_head * stride_cache_head;
                src_ptr = kv_cache + slot_base + dim_chunk * 16;
            } else {
                // Out-of-range: load from self (zero would be nicer but cp.async
                // requires a valid gmem address; we'll just reuse slot 0).
                src_ptr = kv_cache;
            }

            uint32_t smem_u = __cvta_generic_to_shared(dst_ptr);
            cp_async_16B(smem_u, src_ptr);
        }

        // V load
        {
            int chunk = tid;
            int kv_idx_in_tile = chunk / (HALF_D / 16);
            int dim_chunk      = chunk % (HALF_D / 16);
            uint8_t* dst_ptr = v_dst + chunk * 16;

            const uint8_t* src_ptr;
            if (kv_idx_in_tile < kv_count) {
                int kv_pos = kv_start + kv_idx_in_tile;
                int page_idx = kv_pos / page_size;
                int page_off = kv_pos % page_size;
                int block_num = block_table[cur_batch * stride_bt_b + page_idx];
                int64_t slot_base = (int64_t)block_num * stride_cache_block
                                  + (int64_t)page_off * stride_cache_pos
                                  + (int64_t)cur_kv_head * stride_cache_head;
                src_ptr = kv_cache + slot_base + v_region_start + dim_chunk * 16;
            } else {
                src_ptr = kv_cache;
            }

            uint32_t smem_u = __cvta_generic_to_shared(dst_ptr);
            cp_async_16B(smem_u, src_ptr);
        }

        // Scales load: 4 halves per KV token × 16 tokens = 64 halves = 128 B.
        // 8 threads × 16 B covers it exactly.
        if (tid < 8) {
            int kv_idx_in_tile = tid * 2;  // two tokens per 16-B chunk
            uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(s_dst) + tid * 16;

            const uint8_t* src_ptr;
            if (kv_idx_in_tile < kv_count) {
                int kv_pos = kv_start + kv_idx_in_tile;
                int page_idx = kv_pos / page_size;
                int page_off = kv_pos % page_size;
                int block_num = block_table[cur_batch * stride_bt_b + page_idx];
                int64_t flat_slot = (int64_t)block_num * page_size + page_off;
                int64_t sc_base = flat_slot * stride_sc_slot
                                + (int64_t)cur_kv_head * stride_sc_head;
                // We pack: [sb0_K, sb1_K, sb0_V, sb1_V, sb0_K_tok+1, ...]
                // But scales tensor has stride_sc_block for sb dim and stride_sc_kv for K/V.
                // For simplicity: load 4 consecutive halves for token `kv_idx_in_tile` (8 B),
                // then same for token `kv_idx_in_tile + 1` (8 B) — still 16 B total from
                // a contiguous-ish region, but source might not be 16 B contiguous.
                // Fallback: do per-half scalar loads below after the cp.async group.
                // For the prototype, we emit the cp.async from a dummy pointer and
                // replace with scalar loads (scales are tiny — 0.2 % of traffic).
                src_ptr = reinterpret_cast<const uint8_t*>(scales + sc_base);
            } else {
                src_ptr = reinterpret_cast<const uint8_t*>(scales);
            }
            uint32_t smem_u = __cvta_generic_to_shared(dst_ptr);
            cp_async_16B(smem_u, src_ptr);
        }

        cp_async_commit();
    };

    // Prologue: issue STAGES-1 tiles ahead
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        issue_tile(s, s % STAGES);
    }

    // --- Main loop ---
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Issue next tile to keep the pipeline full
        issue_tile(tile + (STAGES - 1), (tile + STAGES - 1) % STAGES);

        // Wait for tile `tile` to become ready: we want STAGES-1 outstanding.
        cp_async_wait_group<STAGES - 1>();
        __syncthreads();

        int stage_slot = tile % STAGES;
        const uint8_t* k_tile = stage_k(stage_slot);
        const uint8_t* v_tile = stage_v(stage_slot);

        int kv_start = split_start + tile * BLOCK_KV;
        int kv_count = min(BLOCK_KV, split_end - kv_start);

        // For each KV token in the tile, compute QK^T + online softmax + PV.
        for (int kv_idx = 0; kv_idx < kv_count; ++kv_idx) {
            const uint8_t* k_row = k_tile + kv_idx * HALF_D;
            const uint8_t* v_row = v_tile + kv_idx * HALF_D;

            // Scales: fall back to global-memory scalar loads for prototype
            // (scales are 0.2 % of traffic; not worth an aligned cp.async path).
            int kv_pos = kv_start + kv_idx;
            int page_idx = kv_pos / page_size;
            int page_off = kv_pos % page_size;
            int block_num = block_table[cur_batch * stride_bt_b + page_idx];
            int64_t flat_slot = (int64_t)block_num * page_size + page_off;
            int64_t sc_base = flat_slot * stride_sc_slot
                            + (int64_t)cur_kv_head * stride_sc_head;

            // Partial QK^T
            float qk_partial[2] = {0.0f, 0.0f};
            for (int i = tid; i < HALF_D; i += BLOCK_THREADS) {
                uint8_t k_packed = k_row[i];
                float k_lo = (float)(k_packed & 0xF) - k_offset;
                float k_hi = (float)((k_packed >> 4) & 0xF) - k_offset;
                int sc_idx = i / (SCALE_BLOCK / 2);
                float k_sc = __half2float(scales[sc_base + sc_idx * stride_sc_block]);
                k_lo *= k_sc; k_hi *= k_sc;
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (h < valid_block_h) {
                        float* q_even = s_q + h * 2 * HALF_D;
                        float* q_odd  = q_even + HALF_D;
                        qk_partial[h] += q_even[i] * k_lo + q_odd[i] * k_hi;
                    }
                }
            }

            // Warp reduce
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                for (int off = warpSize / 2; off > 0; off >>= 1) {
                    qk_partial[h] += __shfl_xor_sync(0xffffffff, qk_partial[h], off);
                }
            }

            __shared__ float s_qk_warp[2][4];  // BLOCK_THREADS / 32 = 4
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0) {
                s_qk_warp[0][warp_id] = qk_partial[0];
                s_qk_warp[1][warp_id] = qk_partial[1];
            }
            __syncthreads();

            __shared__ float s_qk_final[2];
            if (tid < 32) {
                for (int h = 0; h < valid_block_h; ++h) {
                    float val = (tid < 4) ? s_qk_warp[h][tid] : 0.0f;
                    for (int off = 16; off > 0; off >>= 1) {
                        val += __shfl_xor_sync(0xffffffff, val, off);
                    }
                    if (tid == 0) {
                        float score = val * sm_scale;
                        if (logits_soft_cap > 0.0f) {
                            float x = score / logits_soft_cap;
                            float e2x = expf(2.0f * x);
                            float th = 1.0f - 2.0f / (e2x + 1.0f);
                            score = logits_soft_cap * th;
                        }
                        s_qk_final[h] = score;
                    }
                }
            }
            __syncthreads();

            // Online softmax + PV accumulate
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (h >= valid_block_h) break;
                float score = s_qk_final[h];
                float new_max = fmaxf(e_max[h], score);
                float rescale = expf(e_max[h] - new_max);
                float p = expf(score - new_max);

                #pragma unroll
                for (int d = 0; d < DIMS_PER_THREAD; ++d) {
                    acc_even[h][d] *= rescale;
                    acc_odd [h][d] *= rescale;
                }

                #pragma unroll
                for (int d = 0; d < DIMS_PER_THREAD; ++d) {
                    int dim_idx = tid + d * BLOCK_THREADS;
                    if (dim_idx < HALF_D) {
                        uint8_t v_packed = v_row[dim_idx];
                        float v_lo = (float)(v_packed & 0xF) - v_offset;
                        float v_hi = (float)((v_packed >> 4) & 0xF) - v_offset;
                        int v_sc_idx = dim_idx / (SCALE_BLOCK / 2);
                        float v_sc = __half2float(scales[sc_base + v_sc_idx * stride_sc_block + stride_sc_kv]);
                        v_lo *= v_sc; v_hi *= v_sc;
                        acc_even[h][d] += p * v_lo;
                        acc_odd [h][d] += p * v_hi;
                    }
                }

                e_sum[h] = e_sum[h] * rescale + p;
                e_max[h] = new_max;
            }
            __syncthreads();
        }
    }

    // Drain pipeline
    cp_async_wait_all();
    __syncthreads();

    // --- Store results ---
    for (int h = 0; h < valid_block_h; ++h) {
        int head_idx = first_q_head + h;
        if (head_idx >= q_head_num) break;

        int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                         + (int64_t)head_idx * stride_mid_h
                         + (int64_t)split_kv_id * stride_mid_s;

        float safe_sum = (e_sum[h] > 0.0f) ? e_sum[h] : 1.0f;
        float inv_sum  = 1.0f / safe_sum;

        #pragma unroll
        for (int d = 0; d < DIMS_PER_THREAD; ++d) {
            int dim_idx = tid + d * BLOCK_THREADS;
            if (dim_idx < HALF_D) {
                mid_out[mid_base + dim_idx * 2    ] = acc_even[h][d] * inv_sum;
                mid_out[mid_base + dim_idx * 2 + 1] = acc_odd [h][d] * inv_sum;
            }
        }
        if (tid == 0) {
            float lse = e_max[h] + logf(fmaxf(e_sum[h], 1e-30f));
            mid_out[mid_base + HEAD_DIM] = lse;
        }
    }
}

// ------------------------------------------------------------
// Reuse baseline stage-2 reduction (inline copy — kept tiny).
// ------------------------------------------------------------

template <int HEAD_DIM, int BLOCK_THREADS = 256>
__global__ void decode_stage2(
    const float*        __restrict__ mid_out,
    __nv_bfloat16*      __restrict__ output,
    const int32_t*      __restrict__ seq_lens,
    int64_t stride_mid_b, int64_t stride_mid_h, int64_t stride_mid_s,
    int64_t stride_out_b, int64_t stride_out_h,
    int num_kv_splits)
{
    const int bid = blockIdx.x;
    const int hid = blockIdx.y;
    const int tid = threadIdx.x;
    const int seq_len = seq_lens[bid];

    int64_t mid_base = (int64_t)bid * stride_mid_b + (int64_t)hid * stride_mid_h;

    constexpr int DIMS_PER_THREAD = (HEAD_DIM + BLOCK_THREADS - 1) / BLOCK_THREADS;
    float acc[DIMS_PER_THREAD];
    #pragma unroll
    for (int d = 0; d < DIMS_PER_THREAD; ++d) acc[d] = 0.0f;

    float e_max = -1e30f;
    float e_sum = 0.0f;

    for (int s = 0; s < num_kv_splits; ++s) {
        int sl = (seq_len + num_kv_splits - 1) / num_kv_splits;
        int split_start = sl * s;
        int split_end_v = min(split_start + sl, seq_len);
        if (split_start >= split_end_v) continue;

        int64_t off = mid_base + (int64_t)s * stride_mid_s;
        float lse = mid_out[off + HEAD_DIM];
        float new_max = fmaxf(lse, e_max);
        float r = expf(e_max - new_max);
        float w = expf(lse - new_max);

        for (int d = 0; d < DIMS_PER_THREAD; ++d) {
            int dim_idx = tid + d * BLOCK_THREADS;
            if (dim_idx < HEAD_DIM) {
                float tv = mid_out[off + dim_idx];
                acc[d] = acc[d] * r + w * tv;
            }
        }
        e_sum = e_sum * r + w;
        e_max = new_max;
    }

    float safe_sum = (e_sum > 0.0f) ? e_sum : 1.0f;
    float inv_sum = 1.0f / safe_sum;

    int64_t out_base = (int64_t)bid * stride_out_b + (int64_t)hid * stride_out_h;
    for (int d = 0; d < DIMS_PER_THREAD; ++d) {
        int dim_idx = tid + d * BLOCK_THREADS;
        if (dim_idx < HEAD_DIM) {
            output[out_base + dim_idx] = __float2bfloat16(acc[d] * inv_sum);
        }
    }
}

} // namespace fusencache_cpasync


// ------------------------------------------------------------
// Entry point
// ------------------------------------------------------------

void fusencache_decode_cpasync(
    torch::Tensor& output,
    torch::Tensor const& query,
    torch::Tensor const& kv_cache,
    torch::Tensor const& scales,
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
    int64_t k_bits,
    int64_t v_bits,
    int64_t scale_block_k,
    int64_t scale_block_v,
    double k_offset,
    double v_offset)
{
    const int B  = query.size(0);
    const int Hq = query.size(1);
    const int D  = query.size(2);

    TORCH_CHECK(k_bits == 4 && v_bits == 4, "cpasync prototype: k4v4 only");
    TORCH_CHECK(D == 256, "cpasync prototype: head_dim=256 only, got ", D);
    TORCH_CHECK(scale_block_k == 64 && scale_block_v == 64,
                "cpasync prototype: scale_block=64 only");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK_KV      = 16;
    constexpr int BLOCK_THREADS = 128;
    constexpr int STAGES        = 3;
    constexpr int HEAD_DIM      = 256;
    constexpr int HALF_D        = HEAD_DIM / 2;
    constexpr int SCALE_BLOCK   = 64;

    // SMEM size: Q (2 KiB) + STAGES * (K 2KiB + V 2KiB + scales 0.128KiB)
    constexpr int STAGE_BYTES = 2 * HALF_D * BLOCK_KV
                              + 2 * (HEAD_DIM / SCALE_BLOCK) * BLOCK_KV * (int)sizeof(__half);
    constexpr int SMEM_BYTES  = 2 * HEAD_DIM * (int)sizeof(float) + STAGES * STAGE_BYTES;

    // Enable large smem (optin)
    cudaFuncSetAttribute(
        &fusencache_cpasync::decode_stage1_k4v4_cpasync<HEAD_DIM, BLOCK_KV, BLOCK_THREADS, STAGES, SCALE_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES
    );

    const int v_region_start     = D / 2;
    const int k_region_bytes     = D / 2;
    const int num_head_groups    = (Hq + min(2, (int)kv_group_size) - 1) / min(2, (int)kv_group_size);

    dim3 grid1(B, num_head_groups, num_kv_splits);

    fusencache_cpasync::decode_stage1_k4v4_cpasync
        <HEAD_DIM, BLOCK_KV, BLOCK_THREADS, STAGES, SCALE_BLOCK>
        <<<grid1, BLOCK_THREADS, SMEM_BYTES, stream>>>(
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
            (float)v_offset
        );

    // Stage 2 reduce (same as baseline)
    dim3 grid2(B, Hq);
    fusencache_cpasync::decode_stage2<HEAD_DIM, 256>
        <<<grid2, 256, 0, stream>>>(
            mid_out.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            seq_lens.data_ptr<int32_t>(),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
            output.stride(0), output.stride(1),
            (int)num_kv_splits
        );
}
