// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — 30-LAYER TENSOR-CORE v5a.
//
// Derived from mega_graph_gemma4_30layer_v3.cu. Key change vs v3:
//   * Attention Q·K^T and V·P are now WMMA-tensorized instead of the v3
//     per-head scalar (256-thread) inner loops.
//     - Per-head layout retained: 1 SM per head (sm < NUM_HEADS=16).
//     - Q is staged into smem as [16, HEAD_DIM] with row 0 real (padded rows
//       zero) so that WMMA M=16 can be used with only row-0 output consumed.
//     - K/V are loaded directly from global via wmma::load_matrix_sync with
//       the stride-HIDDEN layout (no smem staging of K/V — 64 KB each would
//       not fit anyway).
//     - Q·K^T: M=16, N=16 tile × 16 N-tiles = 256 seq positions; K=128 in
//       8 K-steps of 16. Scores row 0 written back to scores_smem[256].
//     - Softmax stays scalar along seq dim (~256 elems, negligible).
//     - V·P: scores fp32 row 0 converted to bf16 padded as [16, 256],
//       matmul with V [256, HEAD_DIM] per head via WMMA (M=16, K=256, N=128
//       in 8 N-tiles × 16 K-steps). Row-0 of output is the final attn_out.
//
// Everything else (RMSNorm, QKV-proj, O-proj, gate/up, down, 5 barriers) is
// unchanged from v3.
//
// Empirical result (SM120a, 188 SMs, M=1 decode, 30 layers):
//   * N_PER_WARP=128 was too aggressive — tile counts of 16-48 collapse
//     SM parallelism (only 8-25% of SMs busy) and pay 3-4x more serial
//     per-tile work. Slower than v2 by 3x.
//   * N_PER_WARP=16 everywhere (same as v2) with fused in-place rmsnorm
//     lands at ~9.87 ms vs v2's 9.84 ms. Barrier count reduced 7->5 but
//     this did NOT translate to measurable wall-time savings, indicating
//     barriers at seq_len=256/H=2048 cost <<100µs each and the true
//     bottleneck is elsewhere (likely global B-fragment load latency and
//     attention's serial scalar path across 16 heads).
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs. WMMA 16x16x16
// BF16xBF16, FP32 accumulator. NO WGMMA (sm_90 only).
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs. WMMA 16x16x16 BF16xBF16,
// FP32 accumulator. NO WGMMA (sm_90 only).
//
// Shape: H=2048, INTER_DIM=8192, NUM_HEADS=16, HEAD_DIM=128, MAX_SEQ=256,
// NUM_LAYERS=30, M=1 decode.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

namespace mega_graph_gemma4_30_v5a {

// =============================================================================
// Config
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

// WMMA tile shape (BF16, 16x16x16).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// N-tile size per warp — different per stage.
//
// Empirical findings on RTX PRO 6000 (188 SMs, 1504 warps total):
//   * N_PER_WARP=128 collapses parallelism (only 16-48 SMs busy) and is
//     30-250% slower than N=16 for small stages (QKV, O-proj, mlp_down).
//   * N_PER_WARP=16 matches v2's tile count (1 tile = 1 WMMA per warp).
//   * MLP gate/up at INTER_DIM=8192 can afford N_PER_WARP=32 (256 fused
//     gate+up tiles → 1504/256 → 1 tile per 6 warps; plenty of SMs busy,
//     2x A-reuse for both gate+up → 4 MMAs per A-load total).
//
// Small-stage tile N (QKV, O, mlp_down): 16 (1 WMMA per A-load).
// Large-stage tile N (mlp_gate_up):     32 (2 WMMAs per A-load, 4 with gate+up).
static constexpr int N_SMALL = 16;
static constexpr int MMAS_SMALL = N_SMALL / WMMA_N;   // 1
static constexpr int N_LARGE = 16;
static constexpr int MMAS_LARGE = N_LARGE / WMMA_N;   // 1 (currently disabled — N=32 was 0.94x)
static_assert(HIDDEN    % N_SMALL == 0, "HIDDEN divisible by N_SMALL");
static_assert(INTER_DIM % N_LARGE == 0, "INTER_DIM divisible by N_LARGE");

// =============================================================================
// Shared-memory pool layout.
//
//   SMEM_X:     staged activation (max INTER_DIM bf16 = 16 KB)
//   SMEM_A:     per-warp 16x16 bf16 A-tile  (8 warps * 512 B = 4 KB)
//   SMEM_C:     per-warp 16x16 fp32 C scratch (8 warps * 1 KB = 8 KB)
//   SMEM_INV:   per-block inv_rms + rms bookkeeping (~16 B, but allow 64)
//   SMEM_RED:   block-reduce scratch (32 fp32 = 128 B)
//   SMEM_SCORES: attention scores (MAX_SEQ fp32 = 1024 B) — aliases SMEM_A
//   SMEM_QH:     per-head query (HEAD_DIM bf16 = 256 B) — aliases SMEM_A
//
// Stages run disjointly, so aliasing across them is safe. The GEMM stages
// use {X, A, C, INV, RED}. Attention uses {QH, SCORES, RED}.
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

// v5a attention scratch layout (disjoint from the GEMM stages — they run
// between barriers). All aliased into the existing pool:
//   SMEM_Q16    : padded Q tile [16, HEAD_DIM=128] bf16 = 4096 B (aliases SMEM_X)
//   SMEM_P16    : padded P tile [16, MAX_SEQ=256] bf16 = 8192 B  (aliases SMEM_X + 4K)
//   SMEM_O_ACC  : fp32 output [16, HEAD_DIM=128] = 8192 B        (aliases SMEM_C + 0;
//                 SMEM_C region is WARPS*16*16 fp32 = 8192 B exactly)
//   SMEM_SCORES : fp32 scores [MAX_SEQ] = 1024 B                 (aliases SMEM_A)
static constexpr int SMEM_Q16       = SMEM_X;                               // 4096 B
static constexpr int SMEM_Q16_SZ    = WMMA_M * HEAD_DIM * 2;                // 16*128*2 = 4096
static constexpr int SMEM_P16       = SMEM_X + SMEM_Q16_SZ;                 // 8192 B
static constexpr int SMEM_P16_SZ    = WMMA_M * MAX_SEQ * 2;                 // 16*256*2 = 8192
static_assert(SMEM_Q16_SZ + SMEM_P16_SZ <= SMEM_X_SZ,
              "Q16 + P16 must fit in SMEM_X region (16KB)");
static constexpr int SMEM_O_ACC     = SMEM_C;                               // aliases SMEM_C
static constexpr int SMEM_O_ACC_SZ  = WMMA_M * HEAD_DIM * 4;                // 16*128*4 = 8192
static_assert(SMEM_O_ACC_SZ <= SMEM_C_SZ, "O_ACC must fit in SMEM_C (8KB)");
static constexpr int SMEM_SCORES    = SMEM_A;
static constexpr int SMEM_SCORES_SZ = MAX_SEQ * 4;                          // 1024
static_assert(SMEM_SCORES_SZ <= SMEM_A_SZ, "scores must fit in SMEM_A");

static constexpr int SMEM_POOL_BYTES         = SMEM_RED + SMEM_RED_SZ;      // 28932
static constexpr int SMEM_POOL_BYTES_ALIGNED = (SMEM_POOL_BYTES + 15) & ~15;

// =============================================================================
// Block-wide reductions
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

// =============================================================================
// Stage a K-length BF16 vector into smem using all block threads.
// =============================================================================
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

// =============================================================================
// Compute inv_rms for a HIDDEN-sized vector, store into smem[0] (fp32).
// Each SM recomputes locally; all SMs produce bit-identical inv_rms.
// =============================================================================
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

// =============================================================================
// In-place RMSNorm into shared memory: x_smem[d] = x_smem[d] * inv_rms * w[d].
// Done once per stage (before the tile-loop starts). Replaces the v2
// separate rmsnorm stage + its grid.sync; the cost is redundant per-SM
// work (each SM normalizes the whole HIDDEN vector) but this is negligible
// (~2048 FP32 ops/SM) vs the GEMM bodies.
// =============================================================================
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

// Plain (no-rmsnorm) A-tile loader — for stages where input is already in
// the form we want (attention O-proj, mlp_down).
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
// Fused RMSNorm + QKV projection.
//
// Input: raw hidden (residual), rms_w_in (RMSNorm gamma).
// Output: q, k, v = (rmsnorm(hidden) * rms_w_in) @ {Wq, Wk, Wv}.
//
// Each warp owns N_SMALL=16 output columns of ONE projection.
// Total tiles = 3 * (HIDDEN / 16) = 384 tiles distributed over 1504 warps.
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
    float*         red_smem)
{
    constexpr int TILES_PER_PROJ = HIDDEN / N_SMALL;   // 128
    constexpr int TOTAL_TILES    = 3 * TILES_PER_PROJ; // 384
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

    // Stage raw hidden into smem, then RMSNorm IN-PLACE into the same smem.
    // Every SM does this redundantly; all SMs produce identical normalized
    // x_smem. Eliminates the separate v2 rmsnorm stage + its barrier.
    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();
    float inv_rms = compute_inv_rms(x_smem, HIDDEN, tid, red_smem);
    rmsnorm_smem_inplace(x_smem, rms_w_in, inv_rms, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int proj = tile / TILES_PER_PROJ;
        int sub  = tile - proj * TILES_PER_PROJ;
        int n0_base = sub * N_SMALL;

        const __nv_bfloat16* W;
        __nv_bfloat16* y;
        if      (proj == 0) { W = Wq; y = q_out; }
        else if (proj == 1) { W = Wk; y = k_out; }
        else                { W = Wv; y = v_out; }

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[MMAS_SMALL];
        #pragma unroll
        for (int j = 0; j < MMAS_SMALL; j++) wmma::fill_fragment(c_frag[j], 0.0f);

        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            #pragma unroll
            for (int j = 0; j < MMAS_SMALL; j++) {
                int n0 = n0_base + j * WMMA_N;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, W + k0 * HIDDEN + n0, HIDDEN);
                wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
            }
        }

        // Store each fragment, extract row 0.
        #pragma unroll
        for (int j = 0; j < MMAS_SMALL; j++) {
            int n0 = n0_base + j * WMMA_N;
            wmma::store_matrix_sync(c_scr, c_frag[j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            if (lane < WMMA_N) {
                y[n0 + lane] = __float2bfloat16(c_scr[lane]);
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// Attention core — v5a WMMA tensorized.
//
// One SM per head (sm < NUM_HEADS=16). Others idle.
// Stage layout (all smem, aliased over SMEM_X / SMEM_A):
//   Q16  [16, 128] bf16 — row 0 = Q_h, rows 1..15 = 0
//   P16  [16, 256] bf16 — row 0 = softmax(scores), rows 1..15 = 0
//   O16  [16, 128] fp32 — WMMA accumulator, only row 0 is the real output
//   scores [256] fp32   — pre-softmax, reused post-softmax briefly
//
// Q·K^T: M=16, K=HEAD_DIM=128 (8 K-steps), N=seq_len=256 (16 N-tiles).
//   A = Q16 (smem, row-major, lda=128).
//   B = K[:, head h] viewed as col-major [128, 256] with ldb = HIDDEN
//       (stride between "columns" in col-major = stride between rows in
//       the [seq, HIDDEN] row-major storage = HIDDEN).
//   C = row-0 written to scores_smem[0..256).
//   Warps split the 16 N-tiles (2 per warp when 8 warps).
//
// V·P: M=16, K=seq_len=256 (16 K-steps), N=HEAD_DIM=128 (8 N-tiles).
//   A = P16 (smem, row-major, lda=256).
//   B = V[:, head h] viewed as row-major [256, 128] with ldb = HIDDEN.
//   C = O16 fp32 accumulator (smem). Row 0 is the final per-head output.
//   Warps split the 8 N-tiles (1 per warp).
// =============================================================================
__device__ void attention_core_stage(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ attn_out,
    int seq_len,
    int sm, int num_sms, int tid,
    float* scores_smem,
    __nv_bfloat16* q16_smem,    // [16, HEAD_DIM] bf16
    __nv_bfloat16* p16_smem,    // [16, MAX_SEQ]  bf16
    float*         o16_smem,    // [16, HEAD_DIM] fp32
    float* red_smem)
{
    if (sm >= NUM_HEADS) return;

    const int h = sm;
    constexpr float inv_sqrt_d = 1.0f / 11.3137084989848f;  // 1/sqrt(128)
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    // -----------------------------------------------------------------------
    // Stage Q16: row 0 = Q_h, rows 1..15 = 0. Layout row-major stride=HEAD_DIM.
    // -----------------------------------------------------------------------
    const __nv_bfloat16 bf_zero = __float2bfloat16(0.0f);
    const int Q16_N = WMMA_M * HEAD_DIM;   // 16 * 128 = 2048 bf16 elems
    for (int i = tid; i < Q16_N; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM;
        int col = i - row * HEAD_DIM;
        q16_smem[i] = (row == 0) ? q[h * HEAD_DIM + col] : bf_zero;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Q·K^T → scores[0..seq_len).
    // Warps own N-tiles (16x16 output). Scores row 0 extracted lane-wise.
    // -----------------------------------------------------------------------
    constexpr int QK_N_TILES = MAX_SEQ / WMMA_N;            // 16
    constexpr int QK_K_STEPS = HEAD_DIM / WMMA_K;           // 8

    // Scratch [16,16] fp32 per warp — reuse the SMEM_O_ACC region (8 KB).
    // Layout: o16_smem partitioned as WARPS_PER_BLOCK (8) chunks of 16*16=256 fp32.
    // This requires 8 * 256 * 4 = 8192 B = SMEM_O_ACC_SZ. OK.
    float* warp_c = o16_smem + warp_id * (WMMA_M * WMMA_N);

    // Each warp strides through the N-tiles.
    for (int n_tile = warp_id; n_tile < QK_N_TILES; n_tile += WARPS_PER_BLOCK) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int ks = 0; ks < QK_K_STEPS; ks++) {
            int k0 = ks * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, q16_smem + k0, HEAD_DIM);

            // K viewed as col-major [HEAD_DIM, seq_len] for this head.
            // Col j in that view = row j of [seq, HIDDEN] at head h,
            // global ptr = K + j * HIDDEN + h * HEAD_DIM.
            // For col-major load_matrix_sync, the pointer at (k=k0, n=n0) is
            // the element of the [K, N] matrix = K[k0, n0] in col-major =
            // original storage K[n0 * HIDDEN + h*HEAD_DIM + k0].
            int n0 = n_tile * WMMA_N;
            const __nv_bfloat16* bptr =
                K + (size_t)n0 * HIDDEN + h * HEAD_DIM + k0;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, bptr, HIDDEN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(warp_c, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        // Row 0 of warp_c is the portion of the scores we care about.
        int n0 = n_tile * WMMA_N;
        if (lane < WMMA_N) {
            scores_smem[n0 + lane] = warp_c[lane] * inv_sqrt_d;
        }
        __syncwarp();
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Softmax along scores_smem[0..seq_len). Scalar, seq_len=256.
    // -----------------------------------------------------------------------
    float local_max = -INFINITY;
    for (int t = tid; t < seq_len; t += BLOCK_SIZE)
        local_max = fmaxf(local_max, scores_smem[t]);
    float smax = block_reduce_max(local_max, red_smem);

    float local_sum = 0.f;
    for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
        float e = __expf(scores_smem[t] - smax);
        scores_smem[t] = e;
        local_sum += e;
    }
    float ssum = block_reduce_sum(local_sum, red_smem);
    float inv_sum = 1.0f / (ssum + 1e-20f);

    // Normalize in place.
    for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
        scores_smem[t] *= inv_sum;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Stage P16: row 0 = normalized scores (bf16), rows 1..15 = 0.
    // Unused-seq positions (t >= seq_len) in row 0 also zero.
    // -----------------------------------------------------------------------
    const int P16_N = WMMA_M * MAX_SEQ;  // 16*256 = 4096 bf16 elems
    for (int i = tid; i < P16_N; i += BLOCK_SIZE) {
        int row = i / MAX_SEQ;
        int col = i - row * MAX_SEQ;
        __nv_bfloat16 val;
        if (row == 0 && col < seq_len) val = __float2bfloat16(scores_smem[col]);
        else                           val = bf_zero;
        p16_smem[i] = val;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // V·P: out [16, HEAD_DIM] = P16 [16, seq_len] × V_h [seq_len, HEAD_DIM].
    // V for head h: global storage V[t * HIDDEN + h*HEAD_DIM + d].
    //   For WMMA matrix_b row_major [seq, HEAD_DIM] with ldb=HIDDEN.
    // 8 N-tiles (HEAD_DIM/16), 16 K-steps (MAX_SEQ/16).
    // Each warp owns 1 N-tile. Writes its 16x16 block to o16_smem[row, n0..n0+16).
    // -----------------------------------------------------------------------
    constexpr int VP_N_TILES = HEAD_DIM / WMMA_N;     // 8
    constexpr int VP_K_STEPS = MAX_SEQ  / WMMA_K;     // 16

    // o16_smem layout for final output: row-major [16, HEAD_DIM=128].
    // Each warp writes a distinct 16-col slice; no conflicts.
    if (warp_id < VP_N_TILES) {
        int n_tile = warp_id;
        int n0 = n_tile * WMMA_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int ks = 0; ks < VP_K_STEPS; ks++) {
            int k0 = ks * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, p16_smem + k0, MAX_SEQ);

            const __nv_bfloat16* bptr =
                V + (size_t)k0 * HIDDEN + h * HEAD_DIM + n0;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, bptr, HIDDEN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(o16_smem + n0, c_frag, HEAD_DIM, wmma::mem_row_major);
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Extract row 0 of O16 → attn_out[h*HEAD_DIM .. h*HEAD_DIM+HEAD_DIM).
    // -----------------------------------------------------------------------
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        attn_out[h * HEAD_DIM + d] = __float2bfloat16(o16_smem[d]);
    }
}

// =============================================================================
// O-projection + residual. residual += attn_in @ Wo.
// =============================================================================
__device__ void attn_oproj_residual_stage(
    const __nv_bfloat16* __restrict__ attn_in,
    const __nv_bfloat16* __restrict__ Wo,
    __nv_bfloat16* __restrict__ residual,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool)
{
    constexpr int TOTAL_TILES = HIDDEN / N_SMALL;   // 128
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

    stage_vector_to_smem(attn_in, x_smem, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0_base = tile * N_SMALL;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[MMAS_SMALL];
        #pragma unroll
        for (int j = 0; j < MMAS_SMALL; j++) wmma::fill_fragment(c_frag[j], 0.0f);

        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            #pragma unroll
            for (int j = 0; j < MMAS_SMALL; j++) {
                int n0 = n0_base + j * WMMA_N;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, Wo + k0 * HIDDEN + n0, HIDDEN);
                wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
            }
        }

        #pragma unroll
        for (int j = 0; j < MMAS_SMALL; j++) {
            int n0 = n0_base + j * WMMA_N;
            wmma::store_matrix_sync(c_scr, c_frag[j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            if (lane < WMMA_N) {
                float acc = c_scr[lane];
                float r   = __bfloat162float(residual[n0 + lane]);
                residual[n0 + lane] = __float2bfloat16(r + acc);
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// Fused RMSNorm + MLP gate/up.
//   mlp_scratch[j] = silu( (rmsnorm(hidden) * rms_w_post) @ W_gate )[j] *
//                    ( (rmsnorm(hidden) * rms_w_post) @ W_up   )[j]
//
// Serialized gate vs up GEMVs (one at a time) to fit scratch in smem budget.
// Each warp produces N_LARGE output columns: gate + up accumulated, then fused.
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
    float*         red_smem)
{
    constexpr int TOTAL_TILES = INTER_DIM / N_LARGE;   // 256
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

    // Stage raw hidden into smem, then RMSNorm IN-PLACE.
    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();
    float inv_rms = compute_inv_rms(x_smem, HIDDEN, tid, red_smem);
    rmsnorm_smem_inplace(x_smem, rms_w_post, inv_rms, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0_base = tile * N_LARGE;

        // MMAS_LARGE=2 accumulator fragments per (gate, up). 4 MMAs per k0
        // step (2 for gate, 2 for up) sharing one A-fragment load.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cg[MMAS_LARGE];
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cu[MMAS_LARGE];
        #pragma unroll
        for (int j = 0; j < MMAS_LARGE; j++) {
            wmma::fill_fragment(cg[j], 0.0f);
            wmma::fill_fragment(cu[j], 0.0f);
        }

        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            #pragma unroll
            for (int j = 0; j < MMAS_LARGE; j++) {
                int n0 = n0_base + j * WMMA_N;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bg;
                wmma::load_matrix_sync(bg, W_gate + k0 * INTER_DIM + n0, INTER_DIM);
                wmma::mma_sync(cg[j], a_frag, bg, cg[j]);
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bu;
                wmma::load_matrix_sync(bu, W_up   + k0 * INTER_DIM + n0, INTER_DIM);
                wmma::mma_sync(cu[j], a_frag, bu, cu[j]);
            }
        }

        // Fuse: mlp_scratch[j] = silu(gate_j) * up_j.
        #pragma unroll
        for (int j = 0; j < MMAS_LARGE; j++) {
            int n0 = n0_base + j * WMMA_N;
            wmma::store_matrix_sync(c_scr, cg[j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            float g = (lane < WMMA_N) ? c_scr[lane] : 0.f;
            __syncwarp();
            wmma::store_matrix_sync(c_scr, cu[j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            float u = (lane < WMMA_N) ? c_scr[lane] : 0.f;
            if (lane < WMMA_N) {
                float silu_g = g / (1.0f + expf(-g));
                mlp_scratch[n0 + lane] = __float2bfloat16(silu_g * u);
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// MLP down + residual. residual += mlp_scratch @ W_down.
// =============================================================================
__device__ void mlp_down_residual_stage(
    const __nv_bfloat16* __restrict__ mlp_scratch,
    const __nv_bfloat16* __restrict__ W_down,
    __nv_bfloat16* __restrict__ residual,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool)
{
    constexpr int TOTAL_TILES = HIDDEN / N_SMALL;   // 128
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

    stage_vector_to_smem(mlp_scratch, x_smem, INTER_DIM, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0_base = tile * N_SMALL;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[MMAS_SMALL];
        #pragma unroll
        for (int j = 0; j < MMAS_SMALL; j++) wmma::fill_fragment(c_frag[j], 0.0f);

        for (int k0 = 0; k0 < INTER_DIM; k0 += WMMA_K) {
            load_a_tile_plain(x_smem, k0, a_smem, lane);
            __syncwarp();
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            #pragma unroll
            for (int j = 0; j < MMAS_SMALL; j++) {
                int n0 = n0_base + j * WMMA_N;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, W_down + k0 * HIDDEN + n0, HIDDEN);
                wmma::mma_sync(c_frag[j], a_frag, b_frag, c_frag[j]);
            }
        }

        #pragma unroll
        for (int j = 0; j < MMAS_SMALL; j++) {
            int n0 = n0_base + j * WMMA_N;
            wmma::store_matrix_sync(c_scr, c_frag[j], WMMA_N, wmma::mem_row_major);
            __syncwarp();
            if (lane < WMMA_N) {
                float acc = c_scr[lane];
                float r   = __bfloat162float(residual[n0 + lane]);
                residual[n0 + lane] = __float2bfloat16(r + acc);
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// Per-layer weight bundle (identical to v2).
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

// =============================================================================
// Main kernel — 5 barriers/layer.
//
// Barrier schedule per layer:
//   [1]  after QKV (fused rmsnorm_pre)
//   [2]  after attention core (q,k,v -> attn_out)
//   [3]  after O-proj residual
//   [4]  after MLP gate/up (fused rmsnorm_post)  (writes mlp_scratch)
//   [5]  after MLP down residual
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_30layer_v5a_kernel(
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
    __nv_bfloat16* pool_x     = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_X);
    __nv_bfloat16* pool_a     = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_A);
    float*         pool_c     = reinterpret_cast<float*>        (smem_pool + SMEM_C);
    float*         pool_red   = reinterpret_cast<float*>        (smem_pool + SMEM_RED);
    // v5a attention tiles — alias the pool.
    __nv_bfloat16* pool_q16    = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_Q16);
    __nv_bfloat16* pool_p16    = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_P16);
    float*         pool_o16    = reinterpret_cast<float*>        (smem_pool + SMEM_O_ACC);
    float*         pool_scores = reinterpret_cast<float*>        (smem_pool + SMEM_SCORES);

    for (int L = 0; L < num_layers_run; L++) {
        const LayerWeights& W = layers[L];

        __nv_bfloat16* k_slot = W.K_cache + (seq_len - 1) * HIDDEN;
        __nv_bfloat16* v_slot = W.V_cache + (seq_len - 1) * HIDDEN;
        qkv_proj_fused_rmsnorm_stage(
            hidden, W.input_norm,
            W.Wq, W.Wk, W.Wv,
            q_scratch, k_slot, v_slot,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_red);
        grid.sync();  // [1/5]

        attention_core_stage(q_scratch, W.K_cache, W.V_cache, attn_out,
                             seq_len, sm, num_sms, tid,
                             pool_scores, pool_q16, pool_p16, pool_o16,
                             pool_red);
        grid.sync();  // [2/5]

        attn_oproj_residual_stage(attn_out, W.Wo, hidden,
                                  sm, num_sms,
                                  pool_x, pool_a, pool_c);
        grid.sync();  // [3/5]

        mlp_gate_up_fused_rmsnorm_stage(
            hidden, W.post_attn_norm,
            W.W_gate, W.W_up, mlp_scratch,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_red);
        grid.sync();  // [4/5]

        mlp_down_residual_stage(mlp_scratch, W.W_down, hidden,
                                sm, num_sms,
                                pool_x, pool_a, pool_c);
        grid.sync();  // [5/5]
    }
}

// =============================================================================
// Host launchers (C ABI for ctypes)
// =============================================================================
extern "C" int mgg4_30_v5a_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_30_v5a_hidden()     { return HIDDEN; }
extern "C" int mgg4_30_v5a_inter_dim()  { return INTER_DIM; }
extern "C" int mgg4_30_v5a_num_heads()  { return NUM_HEADS; }
extern "C" int mgg4_30_v5a_head_dim()   { return HEAD_DIM; }
extern "C" int mgg4_30_v5a_max_seq()    { return MAX_SEQ; }
extern "C" int mgg4_30_v5a_num_layers() { return NUM_LAYERS; }
extern "C" size_t mgg4_30_v5a_layer_weights_size() { return sizeof(LayerWeights); }
extern "C" int mgg4_30_v5a_smem_bytes() { return SMEM_POOL_BYTES_ALIGNED; }

extern "C" cudaError_t mgg4_30_v5a_launch(
    __nv_bfloat16* hidden,
    const LayerWeights* layers_device,
    __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* mlp_scratch,
    int seq_len,
    int num_layers_run,
    cudaStream_t stream)
{
    int num_sms = mgg4_30_v5a_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden,
        &layers_device,
        &q_scratch, &attn_out, &mlp_scratch,
        &seq_len,
        &num_layers_run,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_30layer_v5a_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_30layer_v5a_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_gemma4_30_v5a
