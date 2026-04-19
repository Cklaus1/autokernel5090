// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — 30-LAYER TENSOR-CORE v4a.
//
// Derived from v3 (mega_graph_gemma4_30layer_v3.cu). Pure async-prefetch
// addition:
//   * Adds a double-buffered cp.async.ca.shared prefetch pipe for B-tiles
//     in all four GEMM stages (QKV, O-proj, MLP gate/up, MLP down).
//   * v3 agent diagnosed B-fragment global-load latency (~4000 cycles
//     stall per stage per warp due to 30 cycles/BF16 cache line at ~150
//     lines per tile loop) as the dominant bottleneck. cp.async overlaps
//     the next k-step's B load with the current k-step's WMMA compute.
//
// Keep v3 invariants:
//   * N_SMALL=16 everywhere (MMAS_SMALL=1, MMAS_LARGE=1).
//   * 5 barriers/layer.
//   * Fused in-place RMSNorm into QKV and MLP gate/up.
//   * Same layer topology.
//
// Prefetch design (depth=2 ping-pong for each B-tile stream):
//   * For stages with ONE B-matrix (QKV pre-dispatch per-warp, O, down):
//     one ping-pong pair (PREFETCH_B0/PREFETCH_B1).
//   * For gate/up (TWO B-matrices per k-step): two pairs sharing the k-index.
//   * Each slot is 16 k-rows × N_SMALL columns = 16x16 bf16 = 512 B.
//
// Per-warp smem layout (appended after the v3 SMEM_C end):
//   Stage-singleton B-pingpong:  8 warps * 2 slots * (WMMA_K * WMMA_N * 2 B) = 8*2*512 = 8192
//   Stage-gate/up extra pingpong: 8 warps * 2 slots * 512                    = 8192
// Total additional smem: 16 KB. v3 used ~37 KB, we have 228 KB — plenty.
//
// cp.async mechanism: cp.async.ca.shared.global with 16-byte line size.
//   * 16 bytes = 8 bf16 elements.
//   * One B-tile row (N_SMALL=16 bf16 = 32 B) = 2 cp.async transactions.
//   * Entire 16x16 tile = 16 rows × 2 = 32 cp.async transactions per warp
//     per prefetch. Lanes 0-31 can each issue one — exactly one per lane
//     covers 16 rows × 2 cols fully when we arrange lane→(row,col_hi) map.
//
// Inline PTX is required since CUDA 12.x exposes cp.async via
// __pipeline_memcpy_async / cooperative_groups::memcpy_async with a more
// awkward interface; we hand-roll it for full control.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

namespace mega_graph_gemma4_30_v4a {

// =============================================================================
// Config — identical to v3.
// =============================================================================
static constexpr int BLOCK_SIZE      = 256;
static constexpr int BLOCKS_PER_SM   = 1;
static constexpr int WARP_SIZE       = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 8

static constexpr int HIDDEN     = 2048;
static constexpr int NUM_HEADS  = 16;
static constexpr int HEAD_DIM   = 128;
static constexpr int INTER_DIM  = 8192;
static constexpr int MAX_SEQ    = 256;
static constexpr int NUM_LAYERS = 30;

static_assert(NUM_HEADS * HEAD_DIM == HIDDEN, "HIDDEN must equal NUM_HEADS*HEAD_DIM");

static constexpr float RMS_EPS = 1e-6f;

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static constexpr int N_SMALL    = 16;
static constexpr int MMAS_SMALL = N_SMALL / WMMA_N;
static constexpr int N_LARGE    = 16;
static constexpr int MMAS_LARGE = N_LARGE / WMMA_N;
static_assert(HIDDEN    % N_SMALL == 0, "HIDDEN divisible by N_SMALL");
static_assert(INTER_DIM % N_LARGE == 0, "INTER_DIM divisible by N_LARGE");

// =============================================================================
// SMEM layout. v3 base + appended B-prefetch ping-pong buffers.
// Each B-tile slot: WMMA_K * WMMA_N * sizeof(bf16) = 16*16*2 = 512 B.
// One pair (depth=2) per warp = 1024 B. 8 warps = 8192 B per pair.
// We need TWO pairs (one for singleton B-stages, one extra for gate/up
// second B — named BU for the up-projection).
// =============================================================================
static constexpr int SMEM_X       = 0;
static constexpr int SMEM_X_SZ    = INTER_DIM * 2;                         // 16384

static constexpr int SMEM_A       = SMEM_X + SMEM_X_SZ;
static constexpr int SMEM_A_SZ    = WARPS_PER_BLOCK * WMMA_M * WMMA_K * 2; // 4096

static constexpr int SMEM_C       = SMEM_A + SMEM_A_SZ;
static constexpr int SMEM_C_SZ    = WARPS_PER_BLOCK * WMMA_M * WMMA_N * 4; // 8192

static constexpr int SMEM_INV     = SMEM_C + SMEM_C_SZ;
static constexpr int SMEM_INV_SZ  = 64;

static constexpr int SMEM_RED     = SMEM_INV + SMEM_INV_SZ;
static constexpr int SMEM_RED_SZ  = 32 * 4;                                 // 128

// Single-B prefetch pool (ping-pong, used by QKV / O / down and as "gate"
// buffer in gate/up).
static constexpr int SMEM_B_SLOT_SZ = WMMA_K * WMMA_N * 2;                  // 512
static constexpr int SMEM_BP        = SMEM_RED + SMEM_RED_SZ;
static constexpr int SMEM_BP_SZ     = WARPS_PER_BLOCK * 2 * SMEM_B_SLOT_SZ; // 8192

// Second prefetch pool for the gate/up stage's up-projection.
static constexpr int SMEM_BP2       = SMEM_BP + SMEM_BP_SZ;
static constexpr int SMEM_BP2_SZ    = WARPS_PER_BLOCK * 2 * SMEM_B_SLOT_SZ; // 8192

// Attention scratch aliases SMEM_A (unused during attention).
static constexpr int SMEM_SCORES    = SMEM_A;
static constexpr int SMEM_SCORES_SZ = MAX_SEQ * 4;                          // 1024
static constexpr int SMEM_QH        = SMEM_SCORES + SMEM_SCORES_SZ;
static constexpr int SMEM_QH_SZ     = HEAD_DIM * 2;                         // 256

static constexpr int SMEM_POOL_BYTES         = SMEM_BP2 + SMEM_BP2_SZ;       // ~45 KB
static constexpr int SMEM_POOL_BYTES_ALIGNED = (SMEM_POOL_BYTES + 15) & ~15;

// =============================================================================
// Block-wide reductions (identical to v3).
// =============================================================================
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int tid = threadIdx.x;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    int wid = tid >> 5;
    int lid = tid & 31;
    if (lid == 0) smem[wid] = v;
    __syncthreads();
    float out = 0.f;
    if (wid == 0) {
        int nwarps = BLOCK_SIZE / 32;
        out = (lid < nwarps) ? smem[lid] : 0.f;
        for (int off = 16; off > 0; off >>= 1) out += __shfl_xor_sync(0xffffffff, out, off);
        if (lid == 0) smem[0] = out;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ float block_reduce_max(float v, float* smem) {
    int tid = threadIdx.x;
    for (int off = 16; off > 0; off >>= 1) {
        float o = __shfl_xor_sync(0xffffffff, v, off);
        v = fmaxf(v, o);
    }
    int wid = tid >> 5;
    int lid = tid & 31;
    if (lid == 0) smem[wid] = v;
    __syncthreads();
    float out = -INFINITY;
    if (wid == 0) {
        int nwarps = BLOCK_SIZE / 32;
        out = (lid < nwarps) ? smem[lid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) {
            float o = __shfl_xor_sync(0xffffffff, out, off);
            out = fmaxf(out, o);
        }
        if (lid == 0) smem[0] = out;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ void stage_vector_to_smem(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16*                    dst_smem,
    int K,
    int tid)
{
    const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(src);
    __nv_bfloat162*       dst2 = reinterpret_cast<__nv_bfloat162*>(dst_smem);
    int n = K >> 1;
    for (int i = tid; i < n; i += BLOCK_SIZE) {
        dst2[i] = src2[i];
    }
}

__device__ __forceinline__ float compute_inv_rms(
    const __nv_bfloat16* __restrict__ src,
    int K,
    int tid,
    float* red_smem)
{
    float local_ss = 0.f;
    for (int d = tid; d < K; d += BLOCK_SIZE) {
        float x = __bfloat162float(src[d]);
        local_ss += x * x;
    }
    float block_ss = block_reduce_sum(local_ss, red_smem);
    return rsqrtf(block_ss / (float)K + RMS_EPS);
}

__device__ __forceinline__ void rmsnorm_smem_inplace(
    __nv_bfloat16*              x_smem,
    const __nv_bfloat16* __restrict__ rms_w,
    float                       inv_rms,
    int                         K,
    int                         tid)
{
    for (int d = tid; d < K; d += BLOCK_SIZE) {
        float x = __bfloat162float(x_smem[d]);
        float w = __bfloat162float(rms_w[d]);
        x_smem[d] = __float2bfloat16(x * inv_rms * w);
    }
}

__device__ __forceinline__ void load_a_tile_plain(
    const __nv_bfloat16* __restrict__ x,
    int                 k0,
    __nv_bfloat16*      a_smem,
    int                 lane)
{
    __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(a_smem);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int pair_idx = lane * 4 + i;
        int col_pair = pair_idx & 7;
        int kc = k0 + (col_pair << 1);
        __nv_bfloat16 a0 = x[kc + 0];
        __nv_bfloat16 a1 = x[kc + 1];
        a2[pair_idx] = __halves2bfloat162(a0, a1);
    }
}

// =============================================================================
// cp.async helpers.
//
// We prefetch a 16x16 bf16 B-tile (row-major in global, ld=stride) into a
// contiguous 16x16 smem slot (ld=16). Each slot is 32 x 16 bytes = 16 rows
// × 32 B/row = 512 B. Using 16-byte cp.async transactions, each row is 2
// transactions (2 × 16 B = 32 B). 16 rows × 2 = 32 total transactions per
// tile. With 32 lanes per warp, each lane issues exactly one cp.async.
//
// Lane → (row, col_half) map:
//   row       = lane >> 1        (0..15)
//   col_half  = lane & 1         (0: cols 0..7, 1: cols 8..15)
// =============================================================================

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Issue a 16B cp.async.ca.shared.global copy.
__device__ __forceinline__ void cp_async_16B(
    uint32_t smem_dst, const void* gmem_src)
{
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Prefetch one 16x16 bf16 B-tile from (W + k0*ld + n0) of stride `ld` into
// `slot_smem` (contiguous, 16x16 ld=16). Commits a new async group.
__device__ __forceinline__ void prefetch_b_tile(
    const __nv_bfloat16* __restrict__ W,
    int k0, int n0, int ld,
    __nv_bfloat16* slot_smem,
    int lane)
{
    // Each lane loads 16 B (8 bf16) from row=(lane>>1), col=(lane&1)*8.
    int row      = lane >> 1;
    int col_half = lane & 1;
    const void* src = W + (k0 + row) * ld + n0 + col_half * 8;
    uint32_t   dst = smem_ptr_u32(slot_smem + row * WMMA_N + col_half * 8);
    cp_async_16B(dst, src);
}

// =============================================================================
// QKV (fused rmsnorm) with cp.async double-buffered B-prefetch.
//
// Tile schedule identical to v3: each warp iterates over its assigned
// (proj, sub) tile and runs HIDDEN/WMMA_K k-steps, each producing one
// 16x16 BF16 B-fragment. Difference: B-fragment for step k is already
// sitting in smem having been issued at step k-1 (or the prologue).
// =============================================================================
__device__ void qkv_proj_fused_rmsnorm_stage(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ rms_w_in,
    const __nv_bfloat16* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ Wk,
    const __nv_bfloat16* __restrict__ Wv,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool,
    float*         red_smem,
    __nv_bfloat16* bp_pool)
{
    constexpr int TILES_PER_PROJ = HIDDEN / N_SMALL;
    constexpr int TOTAL_TILES    = 3 * TILES_PER_PROJ;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);
    // Per-warp ping-pong B slots: [slot0, slot1], each 16*16 bf16.
    __nv_bfloat16* bp_warp = bp_pool + warp_id * 2 * (WMMA_K * WMMA_N);
    __nv_bfloat16* b_slot[2] = { bp_warp, bp_warp + WMMA_K * WMMA_N };

    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();
    float inv_rms = compute_inv_rms(x_smem, HIDDEN, tid, red_smem);
    rmsnorm_smem_inplace(x_smem, rms_w_in, inv_rms, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int proj = tile / TILES_PER_PROJ;
        int sub  = tile - proj * TILES_PER_PROJ;
        int n0   = sub * N_SMALL;

        const __nv_bfloat16* W;
        __nv_bfloat16* y;
        if      (proj == 0) { W = Wq; y = q_out; }
        else if (proj == 1) { W = Wk; y = k_out; }
        else                { W = Wv; y = v_out; }

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // Prologue: prefetch B-tile at k=0 into slot 0.
        prefetch_b_tile(W, 0, n0, HIDDEN, b_slot[0], lane);
        cp_async_commit();

        int cur = 0;
        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            // Prefetch NEXT B-tile into the other slot (if any).
            int k_next = k0 + WMMA_K;
            int nxt = cur ^ 1;
            if (k_next < HIDDEN) {
                prefetch_b_tile(W, k_next, n0, HIDDEN, b_slot[nxt], lane);
                cp_async_commit();
                // Wait for CURRENT (oldest outstanding) group to complete.
                cp_async_wait_group<1>();
            } else {
                // Last k-step: only current is in flight.
                cp_async_wait_group<0>();
            }
            __syncwarp();

            // Load A from x_smem (cheap, smem-resident).
            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);
            wmma::load_matrix_sync(b_frag, b_slot[cur], WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            cur = nxt;
        }

        wmma::store_matrix_sync(c_scr, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            y[n0 + lane] = __float2bfloat16(c_scr[lane]);
        }
        __syncwarp();
    }
    // Drain any leftover outstanding groups issued by this warp.
    cp_async_wait_group<0>();
}

// =============================================================================
// Attention — unchanged from v3.
// =============================================================================
__device__ void attention_core_stage(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ attn_out,
    int seq_len,
    int sm, int num_sms, int tid,
    float* scores_smem,
    __nv_bfloat16* q_h_smem,
    float* red_smem)
{
    if (sm >= NUM_HEADS) return;

    int h = sm;
    constexpr float inv_sqrt_d = 1.0f / 11.3137084989848f;

    if (tid < HEAD_DIM) q_h_smem[tid] = q[h * HEAD_DIM + tid];
    __syncthreads();

    for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
        float s = 0.f;
        const __nv_bfloat16* kt = K + t * HIDDEN + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            s += __bfloat162float(q_h_smem[d]) * __bfloat162float(kt[d]);
        }
        scores_smem[t] = s * inv_sqrt_d;
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < seq_len; t += BLOCK_SIZE) local_max = fmaxf(local_max, scores_smem[t]);
    float smax = block_reduce_max(local_max, red_smem);

    float local_sum = 0.f;
    for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
        float e = __expf(scores_smem[t] - smax);
        scores_smem[t] = e;
        local_sum += e;
    }
    float ssum = block_reduce_sum(local_sum, red_smem);
    float inv_sum = 1.0f / (ssum + 1e-20f);

    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            acc += scores_smem[t] * __bfloat162float(V[t * HIDDEN + h * HEAD_DIM + d]);
        }
        attn_out[h * HEAD_DIM + d] = __float2bfloat16(acc * inv_sum);
    }
}

// =============================================================================
// O-projection + residual with cp.async B-prefetch.
// =============================================================================
__device__ void attn_oproj_residual_stage(
    const __nv_bfloat16* __restrict__ attn_in,
    const __nv_bfloat16* __restrict__ Wo,
    __nv_bfloat16* __restrict__ residual,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool,
    __nv_bfloat16* bp_pool)
{
    constexpr int TOTAL_TILES = HIDDEN / N_SMALL;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);
    __nv_bfloat16* bp_warp = bp_pool + warp_id * 2 * (WMMA_K * WMMA_N);
    __nv_bfloat16* b_slot[2] = { bp_warp, bp_warp + WMMA_K * WMMA_N };

    stage_vector_to_smem(attn_in, x_smem, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0 = tile * N_SMALL;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        prefetch_b_tile(Wo, 0, n0, HIDDEN, b_slot[0], lane);
        cp_async_commit();

        int cur = 0;
        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            int k_next = k0 + WMMA_K;
            int nxt = cur ^ 1;
            if (k_next < HIDDEN) {
                prefetch_b_tile(Wo, k_next, n0, HIDDEN, b_slot[nxt], lane);
                cp_async_commit();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncwarp();

            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);
            wmma::load_matrix_sync(b_frag, b_slot[cur], WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            cur = nxt;
        }

        wmma::store_matrix_sync(c_scr, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            float acc = c_scr[lane];
            float r   = __bfloat162float(residual[n0 + lane]);
            residual[n0 + lane] = __float2bfloat16(r + acc);
        }
        __syncwarp();
    }
    cp_async_wait_group<0>();
}

// =============================================================================
// MLP gate/up (fused rmsnorm) — two parallel B streams (gate + up), each
// with its own ping-pong pair. Four cp.async commits per k-step, paired
// wait_group<3> to keep depth 2 across the combined stream.
// =============================================================================
__device__ void mlp_gate_up_fused_rmsnorm_stage(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ rms_w_post,
    const __nv_bfloat16* __restrict__ W_gate,
    const __nv_bfloat16* __restrict__ W_up,
    __nv_bfloat16* __restrict__ mlp_scratch,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool,
    float*         red_smem,
    __nv_bfloat16* bp_pool_gate,
    __nv_bfloat16* bp_pool_up)
{
    constexpr int TOTAL_TILES = INTER_DIM / N_LARGE;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);
    __nv_bfloat16* bpg_warp = bp_pool_gate + warp_id * 2 * (WMMA_K * WMMA_N);
    __nv_bfloat16* bpu_warp = bp_pool_up   + warp_id * 2 * (WMMA_K * WMMA_N);
    __nv_bfloat16* bg_slot[2] = { bpg_warp, bpg_warp + WMMA_K * WMMA_N };
    __nv_bfloat16* bu_slot[2] = { bpu_warp, bpu_warp + WMMA_K * WMMA_N };

    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();
    float inv_rms = compute_inv_rms(x_smem, HIDDEN, tid, red_smem);
    rmsnorm_smem_inplace(x_smem, rms_w_post, inv_rms, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0 = tile * N_LARGE;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cg;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cu;
        wmma::fill_fragment(cg, 0.0f);
        wmma::fill_fragment(cu, 0.0f);

        // Prologue: prefetch k=0 for both gate and up into slot 0. Two
        // commits → group stack depth 2.
        prefetch_b_tile(W_gate, 0, n0, INTER_DIM, bg_slot[0], lane);
        cp_async_commit();
        prefetch_b_tile(W_up,   0, n0, INTER_DIM, bu_slot[0], lane);
        cp_async_commit();

        int cur = 0;
        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            int k_next = k0 + WMMA_K;
            int nxt = cur ^ 1;
            if (k_next < HIDDEN) {
                prefetch_b_tile(W_gate, k_next, n0, INTER_DIM, bg_slot[nxt], lane);
                cp_async_commit();
                prefetch_b_tile(W_up,   k_next, n0, INTER_DIM, bu_slot[nxt], lane);
                cp_async_commit();
                // Keep exactly 2 groups (the NEXT pair) outstanding.
                cp_async_wait_group<2>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncwarp();

            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bg_frag;
            wmma::load_matrix_sync(bg_frag, bg_slot[cur], WMMA_N);
            wmma::mma_sync(cg, a_frag, bg_frag, cg);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bu_frag;
            wmma::load_matrix_sync(bu_frag, bu_slot[cur], WMMA_N);
            wmma::mma_sync(cu, a_frag, bu_frag, cu);

            cur = nxt;
        }

        wmma::store_matrix_sync(c_scr, cg, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        float g = (lane < WMMA_N) ? c_scr[lane] : 0.f;
        __syncwarp();
        wmma::store_matrix_sync(c_scr, cu, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        float u = (lane < WMMA_N) ? c_scr[lane] : 0.f;
        if (lane < WMMA_N) {
            float silu_g = g / (1.0f + expf(-g));
            mlp_scratch[n0 + lane] = __float2bfloat16(silu_g * u);
        }
        __syncwarp();
    }
    cp_async_wait_group<0>();
}

// =============================================================================
// MLP down + residual with cp.async B-prefetch.
// =============================================================================
__device__ void mlp_down_residual_stage(
    const __nv_bfloat16* __restrict__ mlp_scratch,
    const __nv_bfloat16* __restrict__ W_down,
    __nv_bfloat16* __restrict__ residual,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool,
    __nv_bfloat16* bp_pool)
{
    constexpr int TOTAL_TILES = HIDDEN / N_SMALL;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);
    __nv_bfloat16* bp_warp = bp_pool + warp_id * 2 * (WMMA_K * WMMA_N);
    __nv_bfloat16* b_slot[2] = { bp_warp, bp_warp + WMMA_K * WMMA_N };

    stage_vector_to_smem(mlp_scratch, x_smem, INTER_DIM, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0 = tile * N_SMALL;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        prefetch_b_tile(W_down, 0, n0, HIDDEN, b_slot[0], lane);
        cp_async_commit();

        int cur = 0;
        for (int k0 = 0; k0 < INTER_DIM; k0 += WMMA_K) {
            int k_next = k0 + WMMA_K;
            int nxt = cur ^ 1;
            if (k_next < INTER_DIM) {
                prefetch_b_tile(W_down, k_next, n0, HIDDEN, b_slot[nxt], lane);
                cp_async_commit();
                cp_async_wait_group<1>();
            } else {
                cp_async_wait_group<0>();
            }
            __syncwarp();

            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);
            wmma::load_matrix_sync(b_frag, b_slot[cur], WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            cur = nxt;
        }

        wmma::store_matrix_sync(c_scr, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            float acc = c_scr[lane];
            float r   = __bfloat162float(residual[n0 + lane]);
            residual[n0 + lane] = __float2bfloat16(r + acc);
        }
        __syncwarp();
    }
    cp_async_wait_group<0>();
}

// =============================================================================
// Per-layer weight bundle (identical to v3).
// =============================================================================
struct LayerWeights {
    const __nv_bfloat16* input_norm;
    const __nv_bfloat16* Wq;
    const __nv_bfloat16* Wk;
    const __nv_bfloat16* Wv;
    const __nv_bfloat16* Wo;
    const __nv_bfloat16* post_attn_norm;
    const __nv_bfloat16* W_gate;
    const __nv_bfloat16* W_up;
    const __nv_bfloat16* W_down;
    __nv_bfloat16* K_cache;
    __nv_bfloat16* V_cache;
};

__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_30layer_v4a_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const LayerWeights* __restrict__ layers,
    __nv_bfloat16* __restrict__ q_scratch,
    __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ mlp_scratch,
    int seq_len,
    int num_layers_run)
{
    auto grid = cg::this_grid();
    const int sm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int num_sms = gridDim.x;

    extern __shared__ __align__(16) unsigned char smem_pool[];
    __nv_bfloat16* pool_x      = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_X);
    __nv_bfloat16* pool_a      = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_A);
    float*         pool_c      = reinterpret_cast<float*>        (smem_pool + SMEM_C);
    float*         pool_red    = reinterpret_cast<float*>        (smem_pool + SMEM_RED);
    float*         pool_scores = reinterpret_cast<float*>        (smem_pool + SMEM_SCORES);
    __nv_bfloat16* pool_qh     = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_QH);
    __nv_bfloat16* pool_bp     = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_BP);
    __nv_bfloat16* pool_bp2    = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_BP2);

    for (int L = 0; L < num_layers_run; L++) {
        const LayerWeights& W = layers[L];

        __nv_bfloat16* k_slot = W.K_cache + (seq_len - 1) * HIDDEN;
        __nv_bfloat16* v_slot = W.V_cache + (seq_len - 1) * HIDDEN;
        qkv_proj_fused_rmsnorm_stage(
            hidden, W.input_norm,
            W.Wq, W.Wk, W.Wv,
            q_scratch, k_slot, v_slot,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_red,
            pool_bp);
        grid.sync();  // [1/5]

        attention_core_stage(q_scratch, W.K_cache, W.V_cache, attn_out,
                             seq_len, sm, num_sms, tid,
                             pool_scores, pool_qh, pool_red);
        grid.sync();  // [2/5]

        attn_oproj_residual_stage(attn_out, W.Wo, hidden,
                                  sm, num_sms,
                                  pool_x, pool_a, pool_c,
                                  pool_bp);
        grid.sync();  // [3/5]

        mlp_gate_up_fused_rmsnorm_stage(
            hidden, W.post_attn_norm,
            W.W_gate, W.W_up, mlp_scratch,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_red,
            pool_bp, pool_bp2);
        grid.sync();  // [4/5]

        mlp_down_residual_stage(mlp_scratch, W.W_down, hidden,
                                sm, num_sms,
                                pool_x, pool_a, pool_c,
                                pool_bp);
        grid.sync();  // [5/5]
    }
}

// =============================================================================
// Host launchers (C ABI for ctypes) — v4a suffix.
// =============================================================================
extern "C" int mgg4_30_v4a_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_30_v4a_hidden()     { return HIDDEN; }
extern "C" int mgg4_30_v4a_inter_dim()  { return INTER_DIM; }
extern "C" int mgg4_30_v4a_num_heads()  { return NUM_HEADS; }
extern "C" int mgg4_30_v4a_head_dim()   { return HEAD_DIM; }
extern "C" int mgg4_30_v4a_max_seq()    { return MAX_SEQ; }
extern "C" int mgg4_30_v4a_num_layers() { return NUM_LAYERS; }
extern "C" size_t mgg4_30_v4a_layer_weights_size() { return sizeof(LayerWeights); }
extern "C" int mgg4_30_v4a_smem_bytes() { return SMEM_POOL_BYTES_ALIGNED; }

extern "C" cudaError_t mgg4_30_v4a_launch(
    __nv_bfloat16* hidden,
    const LayerWeights* layers_device,
    __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* mlp_scratch,
    int seq_len,
    int num_layers_run,
    cudaStream_t stream)
{
    int num_sms = mgg4_30_v4a_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden,
        &layers_device,
        &q_scratch, &attn_out, &mlp_scratch,
        &seq_len,
        &num_layers_run,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_30layer_v4a_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_30layer_v4a_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_gemma4_30_v4a
