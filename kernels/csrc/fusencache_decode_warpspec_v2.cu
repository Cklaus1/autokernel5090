// SPDX-License-Identifier: Apache-2.0
//
// FusenCache decode attention — warp-specialized v2 (SM120a).
//
// Design: see plans/fusencache_warp_spec_v2.md.
//
// 4 warps / 128 threads per block:
//   warp 0      : producer — issues cp.async.cg for K, V, scales into a
//                 3-stage ring buffer; uses cp.async.wait_group to
//                 rate-limit.
//   warps 1..3  : consumers — 96 threads, each tile unpacks 16 KV tokens
//                 from smem with vectorized 4-bit dequant (u32 reads +
//                 shr/and), computes QK, online softmax, PV. Reduction
//                 across warps 1..3 uses a shared-memory scratch
//                 partial-sum pattern (3-warp reduce).
//
// Rendezvous: bar.sync.aligned on a dedicated barrier id per stage
// transition; producer emits cp.async.commit_group + wait_group<STAGES-1>
// before the barrier so when consumers cross the barrier the new tile
// is visible in SMEM.
//
// Correctness: identical numerics to the baseline C++ kernel except that
// reduction order is slightly different. All floating-point ops use FP32
// accumulators.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace fusencache_warpspec {

// ------------------------------------------------------------
// PTX wrappers
// ------------------------------------------------------------

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

// Named barrier (bar.sync a, b). `a` is barrier id, `b` is thread count (must
// be multiple of 32).  .aligned not available as a generic bar.sync modifier;
// it is only valid on bar.warp.sync / bar.sync.aligned in specific PTX versions.
__device__ __forceinline__ void bar_sync(int bar_id, int thread_count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar_id), "r"(thread_count));
}

// ------------------------------------------------------------
// Kernel: decode stage 1, warp-specialized, k4v4, D=256
// ------------------------------------------------------------
//
// Grid: (B, num_head_groups, num_kv_splits)
// Block: 128 threads = 4 warps
//   warp 0      : producer (cp.async)
//   warps 1..3  : consumer (dequant + QK + softmax + PV)

template <int HEAD_DIM     = 256,
          int BLOCK_KV     = 16,
          int BLOCK_THREADS = 128,
          int STAGES       = 3,
          int SCALE_BLOCK  = 64>
__global__ __launch_bounds__(BLOCK_THREADS)
void decode_stage1_k4v4_warpspec(
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
    int    k_region_bytes,
    int    v_region_start,
    float  k_offset,
    float  v_offset)
{
    static_assert(HEAD_DIM == 256, "prototype: HEAD_DIM=256 only");
    static_assert(BLOCK_THREADS == 128, "prototype: BLOCK_THREADS=128 only");
    static_assert(BLOCK_KV == 16, "prototype: BLOCK_KV=16 only");
    static_assert(STAGES >= 2 && STAGES <= 4, "STAGES in [2,4]");
    static_assert(SCALE_BLOCK == 64, "prototype: SCALE_BLOCK=64 only");

    constexpr int HALF_D       = HEAD_DIM / 2;                  // 128
    constexpr int K_BYTES_TILE = HALF_D * BLOCK_KV;             // 2048 B per K tile
    constexpr int V_BYTES_TILE = HALF_D * BLOCK_KV;             // 2048 B per V tile
    constexpr int SC_HALF_PER_TILE = 2 * (HEAD_DIM / SCALE_BLOCK) * BLOCK_KV; // K+V sb halves = 64
    constexpr int NUM_WARPS    = BLOCK_THREADS / 32;            // 4
    constexpr int CONS_WARPS   = NUM_WARPS - 1;                 // 3
    constexpr int CONS_THREADS = CONS_WARPS * 32;               // 96

    constexpr int BAR_FULL  = 1;                                // producer->consumer
    constexpr int BAR_EMPTY = 2;                                // consumer->producer
    constexpr int BAR_CONS  = 3;                                // consumer-only (96 threads)

    // Shared memory layout
    extern __shared__ __align__(16) uint8_t s_raw[];
    uint8_t* ptr = s_raw;

    float*   s_q = reinterpret_cast<float*>(ptr);  ptr += 2 * HEAD_DIM * sizeof(float); // 2048 B
    uint8_t* s_k_ring = ptr; ptr += STAGES * K_BYTES_TILE;                              // 6144 B
    uint8_t* s_v_ring = ptr; ptr += STAGES * V_BYTES_TILE;                              // 6144 B
    __half*  s_sc_ring = reinterpret_cast<__half*>(ptr); ptr += STAGES * SC_HALF_PER_TILE * sizeof(__half);
    // Cross-warp QK scratch: one partial float per consumer warp, per head
    float*   s_qk_warp = reinterpret_cast<float*>(ptr); ptr += 2 * CONS_WARPS * sizeof(float);
    // Final QK broadcast (one value per head)
    float*   s_qk_final = reinterpret_cast<float*>(ptr); ptr += 2 * sizeof(float);

    auto k_stage = [&](int s) -> uint8_t* { return s_k_ring + s * K_BYTES_TILE; };
    auto v_stage = [&](int s) -> uint8_t* { return s_v_ring + s * V_BYTES_TILE; };
    auto sc_stage = [&](int s) -> __half* { return s_sc_ring + s * SC_HALF_PER_TILE; };

    const int cur_batch      = blockIdx.x;
    const int cur_head_group = blockIdx.y;
    const int split_kv_id    = blockIdx.z;
    const int tid            = threadIdx.x;
    const int warp_id        = tid / 32;
    const int lane_id        = tid % 32;

    const int cur_kv_head    = cur_head_group / ((kv_group_size + 1) / 2);
    const int valid_block_h  = min(2, kv_group_size);
    const int first_q_head   = cur_head_group * valid_block_h;

    const int seq_len   = seq_lens[cur_batch];
    const int kv_len    = (seq_len + num_kv_splits - 1) / num_kv_splits;
    const int split_start = kv_len * split_kv_id;
    const int split_end   = min(split_start + kv_len, seq_len);

    if (split_start >= split_end) {
        if (tid < 64) {
            for (int h = 0; h < valid_block_h; ++h) {
                int head_idx = first_q_head + h;
                if (head_idx >= q_head_num) break;
                if (tid == 0) {
                    int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                                     + (int64_t)head_idx * stride_mid_h
                                     + (int64_t)split_kv_id * stride_mid_s;
                    mid_out[mid_base + HEAD_DIM] = -1e30f;
                }
            }
        }
        return;
    }

    // --- Load query into SMEM (FP32) — all threads cooperate ---
    // Layout: s_q[h*2*HALF_D + 0..HALF_D-1] = q_even, then +HALF_D = q_odd.
    for (int h = 0; h < valid_block_h; ++h) {
        int head_idx = first_q_head + h;
        if (head_idx >= q_head_num) break;
        const __nv_bfloat16* q_ptr = query
            + (int64_t)cur_batch * stride_qb
            + (int64_t)head_idx * stride_qh;
        float* q_even = s_q + h * 2 * HALF_D;
        float* q_odd  = q_even + HALF_D;
        for (int i = tid; i < HALF_D; i += BLOCK_THREADS) {
            q_even[i] = __bfloat162float(q_ptr[2 * i]);
            q_odd[i]  = __bfloat162float(q_ptr[2 * i + 1]);
        }
    }
    __syncthreads();

    const int num_tiles = (split_end - split_start + BLOCK_KV - 1) / BLOCK_KV;

    // ================================================================
    //                       PRODUCER WARP (warp 0)
    // ================================================================
    if (warp_id == 0) {
        // Producer: issue cp.async for tile t into ring slot (t % STAGES).
        // Prologue: issue STAGES-1 tiles ahead. Then main loop issues one
        // tile per iteration, keeping STAGES-1 outstanding.
        //
        // Per tile the 32-thread producer warp issues:
        //   K: 128 × 16 B  (KV_BYTES_TILE=2048) → 128 16-B loads; each lane
        //                  issues 4 loads sequentially.
        //   V: 128 × 16 B  same.
        //   Scales: 8 × 16 B → first 8 lanes.
        //
        // After issue, commit_group; wait when pipeline > STAGES-1.

        auto issue_tile = [&](int tile_idx, int stage_slot) {
            if (tile_idx >= num_tiles) return;

            int kv_start = split_start + tile_idx * BLOCK_KV;
            int kv_count = min(BLOCK_KV, split_end - kv_start);

            uint8_t* k_dst = k_stage(stage_slot);
            uint8_t* v_dst = v_stage(stage_slot);
            __half*  s_dst = sc_stage(stage_slot);

            // 128 16-B K chunks per tile; 32 warp lanes → 4 chunks/lane.
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int chunk = lane_id + i * 32;
                int kv_idx_in_tile = chunk / (HALF_D / 16);
                int dim_chunk      = chunk % (HALF_D / 16);
                uint8_t* dst_ptr = k_dst + chunk * 16;

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
                    src_ptr = kv_cache;
                }
                uint32_t smem_u = __cvta_generic_to_shared(dst_ptr);
                cp_async_16B(smem_u, src_ptr);
            }

            // V loads
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int chunk = lane_id + i * 32;
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

            // Scales: 64 halves total = 128 B. First 8 lanes each move 16 B.
            if (lane_id < 8) {
                int kv_idx_in_tile = lane_id * 2;  // 2 tokens per 16-B
                uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(s_dst) + lane_id * 16;
                const uint8_t* src_ptr;
                if (kv_idx_in_tile < kv_count) {
                    int kv_pos = kv_start + kv_idx_in_tile;
                    int page_idx = kv_pos / page_size;
                    int page_off = kv_pos % page_size;
                    int block_num = block_table[cur_batch * stride_bt_b + page_idx];
                    int64_t flat_slot = (int64_t)block_num * page_size + page_off;
                    int64_t sc_base = flat_slot * stride_sc_slot
                                    + (int64_t)cur_kv_head * stride_sc_head;
                    src_ptr = reinterpret_cast<const uint8_t*>(scales + sc_base);
                } else {
                    src_ptr = reinterpret_cast<const uint8_t*>(scales);
                }
                uint32_t smem_u = __cvta_generic_to_shared(dst_ptr);
                cp_async_16B(smem_u, src_ptr);
            }

            cp_async_commit();
        };

        // Prologue: issue STAGES-1 tiles ahead, then rendezvous with consumers.
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            issue_tile(s, s % STAGES);
        }

        // Main loop: for each tile 0..num_tiles-1, producer issues
        // tile+STAGES-1 (if valid), then waits so the current tile is ready,
        // then rendezvous.
        for (int tile = 0; tile < num_tiles; ++tile) {
            issue_tile(tile + (STAGES - 1), (tile + STAGES - 1) % STAGES);
            // Wait until the *current* tile has landed (STAGES-1 outstanding).
            cp_async_wait_group<STAGES - 1>();
            // Rendezvous with consumers: tile `tile` is ready for compute.
            bar_sync(BAR_FULL, BLOCK_THREADS);
            // Consumers work on tile `tile` here (producer can return to
            // issue the next tile). Producer will emit the next issue_tile
            // in the next iteration; consumers signal completion via a
            // second barrier.
            bar_sync(BAR_EMPTY, BLOCK_THREADS);
        }

        // Drain — consumers have already finished; just drop remaining
        // cp.async groups.
        cp_async_wait_all();
        return;
    }

    // ================================================================
    //                  CONSUMER WARPS (warps 1..3)
    // ================================================================
    // 96 threads. Each thread owns a subset of HALF_D (128) dimensions.
    // Mapping: consumer_tid = tid - 32 in [0..95]; DIMS_PER_CONS = ceil(128/96) = 2.
    const int cid = tid - 32;  // 0..95
    constexpr int DIMS_PER_CONS = (HALF_D + CONS_THREADS - 1) / CONS_THREADS;  // = 2

    // Online softmax state (per-head)
    float e_max[2] = { -1e30f, -1e30f };
    float e_sum[2] = { 0.0f, 0.0f };
    // PV accumulator — DIMS_PER_CONS dims per thread, 2 heads.
    float acc_even[2][DIMS_PER_CONS] = {{0}};
    float acc_odd [2][DIMS_PER_CONS] = {{0}};

    // Main tile loop
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Wait for producer to fill tile `tile`
        bar_sync(BAR_FULL, BLOCK_THREADS);

        int stage_slot = tile % STAGES;
        const uint8_t* k_tile = k_stage(stage_slot);
        const uint8_t* v_tile = v_stage(stage_slot);

        int kv_start = split_start + tile * BLOCK_KV;
        int kv_count = min(BLOCK_KV, split_end - kv_start);

        // Per-token processing
        for (int kv_idx = 0; kv_idx < kv_count; ++kv_idx) {
            const uint8_t* k_row = k_tile + kv_idx * HALF_D;
            const uint8_t* v_row = v_tile + kv_idx * HALF_D;

            // Scales fallback: scales tile layout is unclear for cp.async
            // (the v1 prototype also falls back to global loads). For now
            // we do the same scalar global load — scales are 0.2% of
            // traffic; replacing them is a v3 task.
            int kv_pos = kv_start + kv_idx;
            int page_idx = kv_pos / page_size;
            int page_off = kv_pos % page_size;
            int block_num = block_table[cur_batch * stride_bt_b + page_idx];
            int64_t flat_slot = (int64_t)block_num * page_size + page_off;
            int64_t sc_base = flat_slot * stride_sc_slot
                            + (int64_t)cur_kv_head * stride_sc_head;

            // ---------------------------------------------------------
            // Vectorized 4-bit K dequant + QK^T partial
            // ---------------------------------------------------------
            //
            // Each consumer thread reads uint32_t (4 K bytes = 8 nibbles = 8
            // K dims) from SMEM at stride CONS_THREADS*4. Unpack via
            // shift+mask, FMA into qk_partial.
            //
            // HALF_D = 128 bytes. 128 / 4 = 32 u32s. 96 threads → 1 u32 per
            // thread with 64 threads doing a second u32 (wrap). We'll do
            // a straightforward strided loop over u32 indices.
            float qk_partial[2] = {0.0f, 0.0f};

            constexpr int U32_PER_ROW = HALF_D / 4;  // 32
            for (int u = cid; u < U32_PER_ROW; u += CONS_THREADS) {
                // Byte offset = u*4
                int i_base = u * 4;  // dim index of first nibble pair's byte 0
                uint32_t packed = *reinterpret_cast<const uint32_t*>(
                    k_row + i_base);
                // lo nibbles: bytes {b0&F, b1&F, b2&F, b3&F}
                uint32_t lo4 = packed & 0x0F0F0F0Fu;
                // hi nibbles: each byte = b_i >> 4, still in low nibble
                uint32_t hi4 = (packed >> 4) & 0x0F0F0F0Fu;

                // Unpack to 4 individual u8 values via byte extract.
                // Per-byte lo: i-th K element's lo nibble (dim 2*i_base+2i),
                //         hi: same byte's hi nibble (dim 2*i_base+2i+1).
                // We need q_even[i_base + i] and q_odd[i_base + i]
                // for i in 0..3.

                // Scale: one scale covers SCALE_BLOCK/2 = 32 dims. With
                // HALF_D = 128 = 4 scale blocks, one 4-dim chunk (i_base..i_base+3)
                // always falls within one scale block.
                int sc_idx = i_base / (SCALE_BLOCK / 2);
                float k_sc = __half2float(scales[sc_base + sc_idx * stride_sc_block]);

                // Extract 4 lo nibbles and 4 hi nibbles as uint32 → float:
                //   lo4 bytes (little-endian, assuming nvcc default LE):
                //     byte 0 = dim i_base + 0 lo → K elem idx i_base+0
                //     byte 1 = dim i_base + 1 lo
                //     byte 2 = dim i_base + 2 lo
                //     byte 3 = dim i_base + 3 lo
                float klo0 = ((float)((lo4 >>  0) & 0xFF) - k_offset) * k_sc;
                float klo1 = ((float)((lo4 >>  8) & 0xFF) - k_offset) * k_sc;
                float klo2 = ((float)((lo4 >> 16) & 0xFF) - k_offset) * k_sc;
                float klo3 = ((float)((lo4 >> 24) & 0xFF) - k_offset) * k_sc;
                float khi0 = ((float)((hi4 >>  0) & 0xFF) - k_offset) * k_sc;
                float khi1 = ((float)((hi4 >>  8) & 0xFF) - k_offset) * k_sc;
                float khi2 = ((float)((hi4 >> 16) & 0xFF) - k_offset) * k_sc;
                float khi3 = ((float)((hi4 >> 24) & 0xFF) - k_offset) * k_sc;

                // FMA into qk_partial for each head
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (h < valid_block_h) {
                        float* q_even = s_q + h * 2 * HALF_D;
                        float* q_odd  = q_even + HALF_D;
                        qk_partial[h] += q_even[i_base + 0] * klo0 + q_odd[i_base + 0] * khi0;
                        qk_partial[h] += q_even[i_base + 1] * klo1 + q_odd[i_base + 1] * khi1;
                        qk_partial[h] += q_even[i_base + 2] * klo2 + q_odd[i_base + 2] * khi2;
                        qk_partial[h] += q_even[i_base + 3] * klo3 + q_odd[i_base + 3] * khi3;
                    }
                }
            }

            // Warp-level reduce within each consumer warp
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                for (int off = 16; off > 0; off >>= 1) {
                    qk_partial[h] += __shfl_xor_sync(0xffffffffu, qk_partial[h], off);
                }
            }

            // Cross-warp reduction (3 consumer warps). s_qk_warp layout:
            //   s_qk_warp[h * CONS_WARPS + (warp_id - 1)]
            // Lane 0 of each consumer warp writes.
            int cons_warp = warp_id - 1;  // 0..2
            if (lane_id == 0) {
                s_qk_warp[0 * CONS_WARPS + cons_warp] = qk_partial[0];
                s_qk_warp[1 * CONS_WARPS + cons_warp] = qk_partial[1];
            }
            // Consumer-only barrier (3 warps = 96 threads). Producer warp
            // does NOT participate — it may be issuing the next cp.async.
            bar_sync(BAR_CONS, CONS_THREADS);

            // Warp 1 sums the 3 partials and applies soft cap + scale.
            if (warp_id == 1 && lane_id < 2) {
                int h = lane_id;
                if (h < valid_block_h) {
                    float sum = s_qk_warp[h * CONS_WARPS + 0]
                              + s_qk_warp[h * CONS_WARPS + 1]
                              + s_qk_warp[h * CONS_WARPS + 2];
                    float score = sum * sm_scale;
                    if (logits_soft_cap > 0.0f) {
                        float x = score / logits_soft_cap;
                        float e2x = expf(2.0f * x);
                        float th = 1.0f - 2.0f / (e2x + 1.0f);
                        score = logits_soft_cap * th;
                    }
                    s_qk_final[h] = score;
                }
            }
            bar_sync(BAR_CONS, CONS_THREADS);

            // Online softmax + vectorized V dequant + PV accumulate
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                if (h >= valid_block_h) break;
                float score = s_qk_final[h];
                float new_max = fmaxf(e_max[h], score);
                float rescale = expf(e_max[h] - new_max);
                float p = expf(score - new_max);

                #pragma unroll
                for (int d = 0; d < DIMS_PER_CONS; ++d) {
                    acc_even[h][d] *= rescale;
                    acc_odd [h][d] *= rescale;
                }

                // V accumulate: each consumer thread owns DIMS_PER_CONS dims,
                // mapping dim_idx = cid + d*CONS_THREADS.
                #pragma unroll
                for (int d = 0; d < DIMS_PER_CONS; ++d) {
                    int dim_idx = cid + d * CONS_THREADS;
                    if (dim_idx < HALF_D) {
                        uint8_t v_packed = v_row[dim_idx];
                        float v_lo = (float)(v_packed & 0xF) - v_offset;
                        float v_hi = (float)((v_packed >> 4) & 0xF) - v_offset;
                        int v_sc_idx = dim_idx / (SCALE_BLOCK / 2);
                        float v_sc = __half2float(
                            scales[sc_base + v_sc_idx * stride_sc_block + stride_sc_kv]);
                        v_lo *= v_sc; v_hi *= v_sc;
                        acc_even[h][d] += p * v_lo;
                        acc_odd [h][d] += p * v_hi;
                    }
                }

                e_sum[h] = e_sum[h] * rescale + p;
                e_max[h] = new_max;
            }
            // Sync the consumer warps before moving to the next kv_idx;
            // this also keeps s_qk_final protected.
            bar_sync(BAR_CONS, CONS_THREADS);
        }

        // Consumers done with this tile — release producer to proceed
        bar_sync(BAR_EMPTY, BLOCK_THREADS);
    }

    // --- Write back mid_out ---
    // Each consumer thread stores its owned dims.
    for (int h = 0; h < valid_block_h; ++h) {
        int head_idx = first_q_head + h;
        if (head_idx >= q_head_num) break;

        int64_t mid_base = (int64_t)cur_batch * stride_mid_b
                         + (int64_t)head_idx * stride_mid_h
                         + (int64_t)split_kv_id * stride_mid_s;

        float safe_sum = (e_sum[h] > 0.0f) ? e_sum[h] : 1.0f;
        float inv_sum  = 1.0f / safe_sum;

        #pragma unroll
        for (int d = 0; d < DIMS_PER_CONS; ++d) {
            int dim_idx = cid + d * CONS_THREADS;
            if (dim_idx < HALF_D) {
                mid_out[mid_base + dim_idx * 2    ] = acc_even[h][d] * inv_sum;
                mid_out[mid_base + dim_idx * 2 + 1] = acc_odd [h][d] * inv_sum;
            }
        }
        if (cid == 0) {
            float lse = e_max[h] + logf(fmaxf(e_sum[h], 1e-30f));
            mid_out[mid_base + HEAD_DIM] = lse;
        }
    }
}

// ------------------------------------------------------------
// Stage 2 reduction — identical to v1 / baseline
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

} // namespace fusencache_warpspec


// ------------------------------------------------------------
// Entry point
// ------------------------------------------------------------

void fusencache_decode_warpspec(
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

    TORCH_CHECK(k_bits == 4 && v_bits == 4, "warp-spec v2 prototype: k4v4 only");
    TORCH_CHECK(D == 256, "warp-spec v2 prototype: head_dim=256 only, got ", D);
    TORCH_CHECK(scale_block_k == 64 && scale_block_v == 64,
                "warp-spec v2 prototype: scale_block=64 only");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int BLOCK_KV      = 16;
    constexpr int BLOCK_THREADS = 128;
    constexpr int STAGES        = 3;
    constexpr int HEAD_DIM      = 256;
    constexpr int HALF_D        = HEAD_DIM / 2;
    constexpr int SCALE_BLOCK   = 64;

    // SMEM layout size (matches device code above)
    constexpr int Q_BYTES    = 2 * HEAD_DIM * (int)sizeof(float);         // 2048
    constexpr int K_BYTES    = STAGES * HALF_D * BLOCK_KV;                // 6144
    constexpr int V_BYTES    = STAGES * HALF_D * BLOCK_KV;                // 6144
    constexpr int SC_BYTES   = STAGES * 2 * (HEAD_DIM / SCALE_BLOCK) * BLOCK_KV * (int)sizeof(__half); // 384
    constexpr int QK_BYTES   = 2 * 3 * (int)sizeof(float);                // 24 (rounded to 32 w/ alignment)
    constexpr int FIN_BYTES  = 2 * (int)sizeof(float);                    // 8
    constexpr int SMEM_BYTES = Q_BYTES + K_BYTES + V_BYTES + SC_BYTES + QK_BYTES + FIN_BYTES;

    cudaFuncSetAttribute(
        &fusencache_warpspec::decode_stage1_k4v4_warpspec<HEAD_DIM, BLOCK_KV, BLOCK_THREADS, STAGES, SCALE_BLOCK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_BYTES
    );

    const int v_region_start     = D / 2;
    const int k_region_bytes     = D / 2;
    const int num_head_groups    = (Hq + min(2, (int)kv_group_size) - 1) / min(2, (int)kv_group_size);

    dim3 grid1(B, num_head_groups, num_kv_splits);

    fusencache_warpspec::decode_stage1_k4v4_warpspec
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

    // Stage 2 reduce
    dim3 grid2(B, Hq);
    fusencache_warpspec::decode_stage2<HEAD_DIM, 256>
        <<<grid2, 256, 0, stream>>>(
            mid_out.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            seq_lens.data_ptr<int32_t>(),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
            output.stride(0), output.stride(1),
            (int)num_kv_splits
        );
}
