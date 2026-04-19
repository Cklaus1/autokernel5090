// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — 30-LAYER TENSOR-CORE v5a.1.
//
// Derived from mega_graph_gemma4_30layer_v5a.cu. The ONLY change vs v5a is
// the attention stage: the per-head "1 SM per head" pattern (which left
// 172/188 SMs idle during attention — the Amdahl cliff) is replaced by
// (head, tile) packing distributed across ALL 188 SMs.
//
// Attention restructure:
//
//  Phase A: Q·K^T  — distributed
//    Work units = NUM_HEADS * (MAX_SEQ/16) = 16 * 16 = 256 units.
//    Each unit = one (head, n_tile) pair => one 16×16 score tile.
//    Scheduled across 188 SMs (~1.36 units per SM on average).
//    WMMA uses v5a's padded-Q-row-0 trick:
//      * A = Q16 [16, 128] bf16 (row 0 = Q_h, rows 1..15 = 0)
//      * B = K   [HEAD_DIM, seq_16] col-major with lda=HIDDEN
//    Each SM owns a private Q16 smem tile and reshuffles it each time it
//    changes to a new head (head usually stays constant over the 1-2 units
//    on that SM thanks to (n_tile-major, head-inner) ordering — see the
//    UNIT_FOR(s) macro).
//    Output scores are written to a GLOBAL scratch:
//      scores_global[head][n_tile*16 .. n_tile*16 + 16)  fp32.
//
//  grid.sync()   -- NEW intra-attention barrier (6/layer total).
//
//  Phase B: softmax + V·P  — distributed
//    Work units = NUM_HEADS * (HEAD_DIM/16) = 16 * 8 = 128 units.
//    Each unit = one (head, out_n_tile) pair => computes 16 out columns.
//    Each SM:
//      * reads full scores_global[head][:] (256 fp32)
//      * scalar softmax into P16 smem [16, 256] bf16 (row 0 real, rest 0)
//      * WMMA V·P for its 16-col slice (N-tile of HEAD_DIM), writing to
//        attn_out[head*HEAD_DIM + out_n_tile*16 .. +16).
//    When an SM has multiple units for the SAME head it caches the softmax
//    in smem (P16 reused) so softmax runs once per distinct head on an SM.
//
// Barrier impact:
//   * v5a: 5 grid.syncs / layer × 30 layers = 150 barriers.
//   * v5a.1: 6 grid.syncs / layer × 30 layers = 180 barriers.
//   * Added cost ~30 × ~50µs = +1.5ms. Must be > offset by attention win.
//
// Barrier schedule per layer (v5a.1):
//   [1/6]  after QKV (fused rmsnorm_pre)
//   [2a/6] after attention QK phase (new)      <-- v5a.1 addition
//   [2b/6] after attention softmax+VP phase    (replaces v5a's single [2/5])
//   [3/6]  after O-proj residual
//   [4/6]  after MLP gate/up
//   [5/6]  after MLP down residual
//
// K-load coalescing (bonus fix): K's col-major layout with ldb=HIDDEN
// already produces decent cache-line hit patterns because within a
// 16-col-major load the same head's K row is accessed contiguously along
// head_dim (bf16 head_dim=128 = 256 B contiguous). The real gain vs v5a
// comes from distributing work across 188 SMs (up from 16).
//
// Shape: H=2048, INTER_DIM=8192, NUM_HEADS=16, HEAD_DIM=128, MAX_SEQ=256,
// NUM_LAYERS=30, M=1 decode. Target: RTX PRO 6000 (SM120a), 188 SMs.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

namespace mega_graph_gemma4_30_v5a1 {

// =============================================================================
// Config (all identical to v5a unless annotated)
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

static constexpr int N_SMALL = 16;
static constexpr int MMAS_SMALL = N_SMALL / WMMA_N;   // 1
static constexpr int N_LARGE = 16;
static constexpr int MMAS_LARGE = N_LARGE / WMMA_N;   // 1

// Attention work unit counts.
static constexpr int QK_N_TILES     = MAX_SEQ / WMMA_N;               // 16
static constexpr int QK_K_STEPS     = HEAD_DIM / WMMA_K;              // 8
static constexpr int VP_N_TILES     = HEAD_DIM / WMMA_N;              // 8
static constexpr int VP_K_STEPS     = MAX_SEQ  / WMMA_K;              // 16

static constexpr int QK_UNITS_TOTAL = NUM_HEADS * QK_N_TILES;         // 256
static constexpr int VP_UNITS_TOTAL = NUM_HEADS * VP_N_TILES;         // 128

// =============================================================================
// Shared-memory pool layout — same as v5a.
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

// Attention tiles — alias the pool.
//   SMEM_Q16    : padded Q tile [16, HEAD_DIM=128] bf16 = 4096 B  (aliases SMEM_X)
//   SMEM_P16    : padded P tile [16, MAX_SEQ=256] bf16 = 8192 B   (aliases SMEM_X + 4K)
//   SMEM_SC     : fp32 scores scratch [MAX_SEQ] = 1024 B          (aliases SMEM_X + 12K)
//   SMEM_O_ACC  : fp32 output [16, HEAD_DIM=128] = 8192 B         (aliases SMEM_C)
static constexpr int SMEM_Q16       = SMEM_X;                               // 4096 B
static constexpr int SMEM_Q16_SZ    = WMMA_M * HEAD_DIM * 2;                // 16*128*2 = 4096
static constexpr int SMEM_P16       = SMEM_X + SMEM_Q16_SZ;                 // 8192 B
static constexpr int SMEM_P16_SZ    = WMMA_M * MAX_SEQ * 2;                 // 16*256*2 = 8192
static constexpr int SMEM_SC        = SMEM_X + SMEM_Q16_SZ + SMEM_P16_SZ;   // 1024 B
static constexpr int SMEM_SC_SZ     = MAX_SEQ * 4;                          // 1024
static_assert(SMEM_Q16_SZ + SMEM_P16_SZ + SMEM_SC_SZ <= SMEM_X_SZ,
              "Q16 + P16 + SC must fit in SMEM_X region (16KB)");
static constexpr int SMEM_O_ACC     = SMEM_C;
static constexpr int SMEM_O_ACC_SZ  = WMMA_M * HEAD_DIM * 4;                // 8192
static_assert(SMEM_O_ACC_SZ <= SMEM_C_SZ, "O_ACC must fit in SMEM_C (8KB)");

// Cached (head, inv_sum) for phase B so softmax is not repeated per unit.
// Track the head currently loaded in P16 smem.
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
// Fused RMSNorm + QKV projection — identical to v5a.
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
    constexpr int TILES_PER_PROJ = HIDDEN / N_SMALL;
    constexpr int TOTAL_TILES    = 3 * TILES_PER_PROJ;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

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
// ATTENTION PHASE A (distributed): Q·K^T → scores_global[NUM_HEADS][MAX_SEQ].
//
// Work units = NUM_HEADS * QK_N_TILES = 256 total, striped across num_sms.
// Each SM processes ceil(256/num_sms) ≈ 1-2 units using warp-parallel N-tile
// sub-splits (8 warps per unit, each warp owns 2 K-steps → 4 WMMAs per warp).
//
// For unit = (head h, n_tile nt):
//   * Stage Q16 smem if the head changed since last unit on this SM
//     (row 0 = Q_h, rows 1..15 = 0). Padded-Q trick per v5a.
//   * 8 warps split the 8 K-steps (1 per warp) of one 16x16 output tile,
//     each warp WMMAs its k-step into its own accumulator; warps reduce
//     into warp 0's accumulator via smem (o16_smem scratch).
//   * Row-0 of the final 16x16 fp32 tile = 16 scores for positions
//     (nt*16 .. nt*16+16). Write back to global scores[h][nt*16..+16).
// =============================================================================
__device__ void attention_qk_phase(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ K,
    float* __restrict__  scores_global,     // [NUM_HEADS * MAX_SEQ] fp32
    int sm, int num_sms, int tid,
    __nv_bfloat16* q16_smem,
    float*         o16_smem)                // reused as per-warp scratch
{
    constexpr float inv_sqrt_d = 1.0f / 11.3137084989848f;
    const __nv_bfloat16 bf_zero = __float2bfloat16(0.0f);
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    // Per-warp 16x16 fp32 scratch — SMEM_O_ACC region is WARPS*16*16 fp32 = 8192B.
    float* warp_c = o16_smem + warp_id * (WMMA_M * WMMA_N);

    // Track head currently staged in Q16 smem. Units may share a head; avoid
    // restaging whenever possible.
    int current_head = -1;

    // Stripe units across SMs. To maximize head reuse in smem, order so that
    // SMs processing consecutive units process the same head when possible:
    //   unit_id = h * QK_N_TILES + nt.
    // With num_sms=188 and total=256, each SM processes 1-2 consecutive unit
    // IDs; these share a head when both fall within the same 16-block.
    for (int unit = sm; unit < QK_UNITS_TOTAL; unit += num_sms) {
        int h  = unit / QK_N_TILES;
        int nt = unit - h * QK_N_TILES;

        if (h != current_head) {
            // Restage Q16: row 0 = Q_h, rows 1..15 = 0.
            const int Q16_N = WMMA_M * HEAD_DIM;
            for (int i = tid; i < Q16_N; i += BLOCK_SIZE) {
                int row = i / HEAD_DIM;
                int col = i - row * HEAD_DIM;
                q16_smem[i] = (row == 0) ? q[h * HEAD_DIM + col] : bf_zero;
            }
            __syncthreads();
            current_head = h;
        }

        // 8 warps split the 8 K-steps. Each warp computes partial 16x16 tile
        // for k-step ks = warp_id, writes to its own warp_c, then we reduce
        // across warps in smem. This keeps each warp busy and leverages all
        // 256 threads in the block.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        int ks = warp_id;  // 0..7
        int k0 = ks * WMMA_K;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, q16_smem + k0, HEAD_DIM);

        int n0 = nt * WMMA_N;
        const __nv_bfloat16* bptr =
            K + (size_t)n0 * HIDDEN + h * HEAD_DIM + k0;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, bptr, HIDDEN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        wmma::store_matrix_sync(warp_c, c_frag, WMMA_N, wmma::mem_row_major);
        __syncthreads();

        // Reduce row 0 across the 8 warps.
        // Only lane < 16 participates; each lane sums warp_c[w*256 + 0*16 + lane]
        // for w in [0..7]. (warp_c strided by WMMA_M*WMMA_N = 256 fp32.)
        if (warp_id == 0) {
            if (lane < WMMA_N) {
                float s = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                    // row 0 of warp w's tile = o16_smem[w*256 + 0*16 + lane]
                    s += o16_smem[w * (WMMA_M * WMMA_N) + lane];
                }
                scores_global[h * MAX_SEQ + n0 + lane] = s * inv_sqrt_d;
            }
        }
        __syncthreads();
    }
}

// =============================================================================
// ATTENTION PHASE B (distributed): softmax + V·P → attn_out.
//
// Work units = NUM_HEADS * VP_N_TILES = 128, striped across num_sms.
// Each SM: for its units grouped by head, softmax once, then VP once per
// (head, out_n_tile) unit.
//
// Softmax is done scalar over 256 elements, block-parallel. Each SM reads
// its head's scores from scores_global.
//
// V·P tile (one unit): M=16, K=MAX_SEQ=256 (16 K-steps), N=16 (single N-tile).
//   * A = P16 smem [16, 256] row-major, stride 256.
//   * B = V[:, head, head_dim_slice] row-major [256, HEAD_DIM] with ldb=HIDDEN.
//   * C = 16x16 fp32, row 0 is the final per-head output slice.
//   * 8 warps split 16 K-steps (2 each); each warp accumulates partial c_frag
//     then warps reduce across o16_smem into a final 16x16.
// =============================================================================
__device__ void attention_vp_phase(
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__          scores_global,
    __nv_bfloat16* __restrict__        attn_out,
    int seq_len,
    int sm, int num_sms, int tid,
    __nv_bfloat16* p16_smem,
    float*         o16_smem,
    float*         sc_smem,
    float*         red_smem)
{
    const __nv_bfloat16 bf_zero = __float2bfloat16(0.0f);
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    // Iterate work units. Group units by head so softmax runs once per head
    // per SM. Stripe order: unit = h * VP_N_TILES + out_nt.
    int current_head = -1;

    float* warp_c = o16_smem + warp_id * (WMMA_M * WMMA_N);

    for (int unit = sm; unit < VP_UNITS_TOTAL; unit += num_sms) {
        int h      = unit / VP_N_TILES;
        int out_nt = unit - h * VP_N_TILES;

        if (h != current_head) {
            // Load scores_global[h][0..seq_len) into sc_smem, softmax in place,
            // broadcast into P16 smem (row 0 = P, rows 1..15 = 0).
            for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
                sc_smem[t] = scores_global[h * MAX_SEQ + t];
            }
            __syncthreads();

            float local_max = -INFINITY;
            for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
                local_max = fmaxf(local_max, sc_smem[t]);
            }
            float smax = block_reduce_max(local_max, red_smem);

            float local_sum = 0.f;
            for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
                float e = __expf(sc_smem[t] - smax);
                sc_smem[t] = e;
                local_sum += e;
            }
            float ssum = block_reduce_sum(local_sum, red_smem);
            float inv_sum = 1.0f / (ssum + 1e-20f);

            // Broadcast into P16 smem. row 0 = normalized scores, rest 0.
            const int P16_N = WMMA_M * MAX_SEQ;
            for (int i = tid; i < P16_N; i += BLOCK_SIZE) {
                int row = i / MAX_SEQ;
                int col = i - row * MAX_SEQ;
                __nv_bfloat16 val;
                if (row == 0 && col < seq_len) {
                    val = __float2bfloat16(sc_smem[col] * inv_sum);
                } else {
                    val = bf_zero;
                }
                p16_smem[i] = val;
            }
            __syncthreads();
            current_head = h;
        }

        // V·P for this (head, out_nt). 16 K-steps split 2-per-warp across 8 warps.
        int n0 = out_nt * WMMA_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        // This warp processes K-steps {warp_id*2, warp_id*2+1} (2 per warp * 8 warps = 16).
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            int ks = warp_id * 2 + sub;
            int k0 = ks * WMMA_K;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, p16_smem + k0, MAX_SEQ);

            const __nv_bfloat16* bptr =
                V + (size_t)k0 * HIDDEN + h * HEAD_DIM + n0;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, bptr, HIDDEN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Store partial tile, reduce across warps, extract row 0.
        wmma::store_matrix_sync(warp_c, c_frag, WMMA_N, wmma::mem_row_major);
        __syncthreads();

        if (warp_id == 0) {
            if (lane < WMMA_N) {
                float s = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                    s += o16_smem[w * (WMMA_M * WMMA_N) + lane];
                }
                attn_out[h * HEAD_DIM + n0 + lane] = __float2bfloat16(s);
            }
        }
        __syncthreads();
    }
}

// =============================================================================
// O-projection + residual — identical to v5a.
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
    constexpr int TOTAL_TILES = HIDDEN / N_SMALL;
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
// Fused RMSNorm + MLP gate/up — identical to v5a.
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
    constexpr int TOTAL_TILES = INTER_DIM / N_LARGE;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();
    float inv_rms = compute_inv_rms(x_smem, HIDDEN, tid, red_smem);
    rmsnorm_smem_inplace(x_smem, rms_w_post, inv_rms, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0_base = tile * N_LARGE;

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
// MLP down + residual — identical to v5a.
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
    constexpr int TOTAL_TILES = HIDDEN / N_SMALL;
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
// Layer weight bundle (identical to v5a)
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
// Main kernel — 6 barriers/layer.
//
// Per-layer barrier schedule:
//   [1]  after QKV                (v5a's [1/5])
//   [2a] after attention QK phase (NEW)
//   [2b] after attention VP phase (v5a's [2/5])
//   [3]  after O-proj residual    (v5a's [3/5])
//   [4]  after MLP gate/up        (v5a's [4/5])
//   [5]  after MLP down residual  (v5a's [5/5])
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_30layer_v5a1_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const LayerWeights* __restrict__ layers,
    __nv_bfloat16* __restrict__ q_scratch,
    __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ mlp_scratch,
    float*         __restrict__ scores_global,
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
    __nv_bfloat16* pool_q16   = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_Q16);
    __nv_bfloat16* pool_p16   = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_P16);
    float*         pool_sc    = reinterpret_cast<float*>        (smem_pool + SMEM_SC);
    float*         pool_o16   = reinterpret_cast<float*>        (smem_pool + SMEM_O_ACC);

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
        grid.sync();  // [1/6]

        // -------------------- Attention QK phase (distributed) --------------------
        attention_qk_phase(
            q_scratch, W.K_cache, scores_global,
            sm, num_sms, tid,
            pool_q16, pool_o16);
        grid.sync();  // [2a/6]

        // -------------------- Attention VP phase (distributed) --------------------
        attention_vp_phase(
            W.V_cache, scores_global, attn_out,
            seq_len, sm, num_sms, tid,
            pool_p16, pool_o16, pool_sc, pool_red);
        grid.sync();  // [2b/6]

        attn_oproj_residual_stage(attn_out, W.Wo, hidden,
                                  sm, num_sms,
                                  pool_x, pool_a, pool_c);
        grid.sync();  // [3/6]

        mlp_gate_up_fused_rmsnorm_stage(
            hidden, W.post_attn_norm,
            W.W_gate, W.W_up, mlp_scratch,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_red);
        grid.sync();  // [4/6]

        mlp_down_residual_stage(mlp_scratch, W.W_down, hidden,
                                sm, num_sms,
                                pool_x, pool_a, pool_c);
        grid.sync();  // [5/6]
    }
}

// =============================================================================
// Host launchers (C ABI)
// =============================================================================
extern "C" int mgg4_30_v5a1_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_30_v5a1_hidden()     { return HIDDEN; }
extern "C" int mgg4_30_v5a1_inter_dim()  { return INTER_DIM; }
extern "C" int mgg4_30_v5a1_num_heads()  { return NUM_HEADS; }
extern "C" int mgg4_30_v5a1_head_dim()   { return HEAD_DIM; }
extern "C" int mgg4_30_v5a1_max_seq()    { return MAX_SEQ; }
extern "C" int mgg4_30_v5a1_num_layers() { return NUM_LAYERS; }
extern "C" size_t mgg4_30_v5a1_layer_weights_size() { return sizeof(LayerWeights); }
extern "C" int mgg4_30_v5a1_smem_bytes() { return SMEM_POOL_BYTES_ALIGNED; }
extern "C" size_t mgg4_30_v5a1_scores_bytes() { return (size_t)NUM_HEADS * MAX_SEQ * sizeof(float); }

extern "C" cudaError_t mgg4_30_v5a1_launch(
    __nv_bfloat16* hidden,
    const LayerWeights* layers_device,
    __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* mlp_scratch,
    float*         scores_global,
    int seq_len,
    int num_layers_run,
    cudaStream_t stream)
{
    int num_sms = mgg4_30_v5a1_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden,
        &layers_device,
        &q_scratch, &attn_out, &mlp_scratch,
        &scores_global,
        &seq_len,
        &num_layers_run,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_30layer_v5a1_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_30layer_v5a1_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_gemma4_30_v5a1
