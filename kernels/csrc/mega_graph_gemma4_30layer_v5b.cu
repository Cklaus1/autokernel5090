// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — 30-LAYER TENSOR-CORE v5b.
//
// Derived from v5a. Key change: BARRIER FUSION.
//   * Barrier count 5/layer -> 4/layer via fusing attention + O-proj.
//   * After attention core on head-SM h, the SAME SM continues (no
//     grid.sync) to compute its head's PARTIAL contribution to the
//     O-projection: o_partial[h][m] = sum_{d in h's slice}
//     attn_out_h[d] * Wo[h*HEAD_DIM+d, m] for all m in [0, HIDDEN).
//   * The partial GEMV [HEAD_DIM=128] × [HEAD_DIM, HIDDEN] = [HIDDEN]
//     runs via WMMA with only 8 K-steps (HEAD_DIM/WMMA_K) per output
//     tile. Distribution: 8 warps * 1 head = 8 warps handle HIDDEN/N
//     tiles; each warp owns HIDDEN / WARPS_PER_BLOCK / N_SMALL = 16 tiles.
//   * Output partial buffer: [NUM_HEADS, HIDDEN] bf16 = 16 * 2048 * 2 = 64 KB.
//     Lives in global (new arg o_partials).
//
// Then in the NEW combined "oproj-reduce + mlp-gate-up" stage (replacing
// v5a's separate O-proj and MLP-gate-up stages):
//   * Every SM reads all 16 partials (64 KB) + the current hidden (residual),
//     sums them to produce hidden_post_oproj in smem.
//   * Runs rmsnorm on that smem copy (uses post_attn_norm gamma).
//   * Computes its MLP gate/up tile via WMMA and writes mlp_scratch.
//
// Barrier schedule per layer (v5b):
//   [1/4] after QKV (fused rmsnorm_pre)
//   [2/4] after attention + partial-O-proj (head-SMs only)  <-- FUSED
//   [3/4] after O-proj-reduce + MLP gate/up (fused rmsnorm_post + reduce)
//   [4/4] after MLP down residual
//
// Expected gain: -30 barriers × ~15 µs = -450 µs. Target ~8,370 µs from v5a
// 8,821 µs baseline (PARTIAL -> close-to-PASS).
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

namespace mega_graph_gemma4_30_v5b {

// =============================================================================
// Config (identical to v5a unless annotated)
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
static_assert(HIDDEN    % N_SMALL == 0, "HIDDEN divisible by N_SMALL");
static_assert(INTER_DIM % N_LARGE == 0, "INTER_DIM divisible by N_LARGE");

// =============================================================================
// Shared-memory pool layout — identical to v5a.
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

// Attention tiles aliasing the pool (same as v5a).
static constexpr int SMEM_Q16       = SMEM_X;                               // 4096 B
static constexpr int SMEM_Q16_SZ    = WMMA_M * HEAD_DIM * 2;                // 4096
static constexpr int SMEM_P16       = SMEM_X + SMEM_Q16_SZ;                 // 8192 B
static constexpr int SMEM_P16_SZ    = WMMA_M * MAX_SEQ * 2;                 // 8192
static_assert(SMEM_Q16_SZ + SMEM_P16_SZ <= SMEM_X_SZ,
              "Q16 + P16 must fit in SMEM_X region (16KB)");
static constexpr int SMEM_O_ACC     = SMEM_C;                               // aliases SMEM_C
static constexpr int SMEM_O_ACC_SZ  = WMMA_M * HEAD_DIM * 4;                // 8192
static_assert(SMEM_O_ACC_SZ <= SMEM_C_SZ, "O_ACC must fit in SMEM_C (8KB)");
static constexpr int SMEM_SCORES    = SMEM_A;
static constexpr int SMEM_SCORES_SZ = MAX_SEQ * 4;                          // 1024
static_assert(SMEM_SCORES_SZ <= SMEM_A_SZ, "scores must fit in SMEM_A");

// In partial-O-proj sub-stage, we need an additional smem tile for the
// row-0 attn_out slice [HEAD_DIM=128] bf16 = 256 B. Alias over SMEM_P16
// area since P16 is done after softmax use, but we'll actually reuse
// O_ACC's row-0 values directly via o16_smem smem. The "attn row" for
// partial O-proj reads from attn_out in global (after final store).
// We use attn_out (global) directly — no extra smem needed.

static constexpr int SMEM_POOL_BYTES         = SMEM_RED + SMEM_RED_SZ;
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
// FUSED attention + partial-O-proj stage.
//
// Phase A: v5a attention (one SM per head, 16 head-SMs active).
//          Computes attn_out[h*HEAD_DIM..(h+1)*HEAD_DIM] into smem o16_smem
//          (stage) and also writes to the global attn_out buffer (for debug
//          / correctness checks — optional).
//
// Phase B (NEW, same kernel stage, no grid.sync): each head-SM computes its
// partial contribution to the O-projection:
//
//     o_partial[h][m] = sum_{d=0..HEAD_DIM-1}
//         attn_out_h[d] * Wo[h*HEAD_DIM + d, m]    for m in [0, HIDDEN).
//
// This is a (1 x HEAD_DIM=128) × (HEAD_DIM=128, HIDDEN=2048) GEMV. Using
// the v5a WMMA trick (padded 16-row A where only row 0 is real), we do
// HIDDEN/WMMA_N = 128 output tiles each M=16 x N=16 x K=128 = 8 K-steps.
// With 8 warps per block, each warp handles 16 tiles.
//
// A = attn_out_h staged to smem with row 0 real and rows 1..15 zero
//     (reuses p16_smem / o16_smem region — safe because softmax is done).
// B = Wo[h*HEAD_DIM .. (h+1)*HEAD_DIM, :] row-major slab [HEAD_DIM, HIDDEN]
//     with ldb = HIDDEN.
// C = o_partial[h][:] stored bf16, row 0 of the WMMA output tile.
//
// Head-SMs writer produces 64KB of partials (16 heads x 2048 cols x 2 bytes).
// =============================================================================
__device__ void attention_partial_oproj_stage(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ Wo,
    __nv_bfloat16* __restrict__ attn_out,       // [HIDDEN] (per-head slice written)
    __nv_bfloat16* __restrict__ o_partials,     // [NUM_HEADS, HIDDEN]
    int seq_len,
    int sm, int num_sms, int tid,
    float* scores_smem,
    __nv_bfloat16* q16_smem,    // [16, HEAD_DIM] bf16
    __nv_bfloat16* p16_smem,    // [16, MAX_SEQ]  bf16 — later reused as attn-row smem
    float*         o16_smem,    // [16, HEAD_DIM] fp32
    __nv_bfloat16* a_smem_pool, // per-warp A tile pool
    float*         c_scratch_pool,
    float*         red_smem)
{
    // ---- Phase A: v5a attention (only 16 head-SMs) ------------------------
    const bool is_head_sm = (sm < NUM_HEADS);
    const int h = sm;
    constexpr float inv_sqrt_d = 1.0f / 11.3137084989848f;  // 1/sqrt(128)
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    if (is_head_sm) {
        const __nv_bfloat16 bf_zero = __float2bfloat16(0.0f);
        const int Q16_N = WMMA_M * HEAD_DIM;
        for (int i = tid; i < Q16_N; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i - row * HEAD_DIM;
            q16_smem[i] = (row == 0) ? q[h * HEAD_DIM + col] : bf_zero;
        }
        __syncthreads();

        constexpr int QK_N_TILES = MAX_SEQ / WMMA_N;
        constexpr int QK_K_STEPS = HEAD_DIM / WMMA_K;

        float* warp_c = o16_smem + warp_id * (WMMA_M * WMMA_N);
        for (int n_tile = warp_id; n_tile < QK_N_TILES; n_tile += WARPS_PER_BLOCK) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            #pragma unroll
            for (int ks = 0; ks < QK_K_STEPS; ks++) {
                int k0 = ks * WMMA_K;
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, q16_smem + k0, HEAD_DIM);

                int n0 = n_tile * WMMA_N;
                const __nv_bfloat16* bptr = K + (size_t)n0 * HIDDEN + h * HEAD_DIM + k0;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, bptr, HIDDEN);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            wmma::store_matrix_sync(warp_c, c_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();
            int n0 = n_tile * WMMA_N;
            if (lane < WMMA_N) {
                scores_smem[n0 + lane] = warp_c[lane] * inv_sqrt_d;
            }
            __syncwarp();
        }
        __syncthreads();

        // Softmax
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
        for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
            scores_smem[t] *= inv_sum;
        }
        __syncthreads();

        const int P16_N = WMMA_M * MAX_SEQ;
        for (int i = tid; i < P16_N; i += BLOCK_SIZE) {
            int row = i / MAX_SEQ;
            int col = i - row * MAX_SEQ;
            __nv_bfloat16 val;
            if (row == 0 && col < seq_len) val = __float2bfloat16(scores_smem[col]);
            else                           val = bf_zero;
            p16_smem[i] = val;
        }
        __syncthreads();

        constexpr int VP_N_TILES = HEAD_DIM / WMMA_N;     // 8
        constexpr int VP_K_STEPS = MAX_SEQ  / WMMA_K;     // 16

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
                const __nv_bfloat16* bptr = V + (size_t)k0 * HIDDEN + h * HEAD_DIM + n0;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, bptr, HIDDEN);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(o16_smem + n0, c_frag, HEAD_DIM, wmma::mem_row_major);
        }
        __syncthreads();

        // Row 0 of o16_smem[0 .. HEAD_DIM) is this head's attn_out slice in fp32.
        // Write it to attn_out global (for correctness / downstream use) and
        // also stash the bf16 version into p16_smem[0..HEAD_DIM) for reuse
        // as the A-row in partial-O-proj below.
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            float f = o16_smem[d];
            __nv_bfloat16 bf = __float2bfloat16(f);
            attn_out[h * HEAD_DIM + d] = bf;
        }
        __syncthreads();

        // ---- Phase B: partial-O-proj (SAME head-SM continues, no barrier) --
        //
        // Construct A tile = [16, HEAD_DIM=128] with row 0 = attn_out_h,
        // rows 1..15 = 0. Reuse q16_smem (overwritten, fine — attention done).
        const int AP_N = WMMA_M * HEAD_DIM;
        for (int i = tid; i < AP_N; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM;
            int col = i - row * HEAD_DIM;
            q16_smem[i] = (row == 0) ? __float2bfloat16(o16_smem[col])
                                     : bf_zero;
        }
        __syncthreads();

        // Compute o_partials[h][m] = A_row0 · Wo_h[d, m] for m in [0, HIDDEN).
        // Tiles: HIDDEN/WMMA_N = 128. Distribute across 8 warps => 16 tiles/warp.
        constexpr int OP_N_TILES = HIDDEN / WMMA_N;
        constexpr int OP_K_STEPS = HEAD_DIM / WMMA_K;

        float* warp_acc = o16_smem + warp_id * (WMMA_M * WMMA_N);
        for (int n_tile = warp_id; n_tile < OP_N_TILES; n_tile += WARPS_PER_BLOCK) {
            int n0 = n_tile * WMMA_N;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            #pragma unroll
            for (int ks = 0; ks < OP_K_STEPS; ks++) {
                int k0 = ks * WMMA_K;
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, q16_smem + k0, HEAD_DIM);

                // Wo[h*HEAD_DIM + k0 .. +k0+16, n0 .. n0+16], stride HIDDEN.
                const __nv_bfloat16* bptr =
                    Wo + (size_t)(h * HEAD_DIM + k0) * HIDDEN + n0;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, bptr, HIDDEN);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            wmma::store_matrix_sync(warp_acc, c_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();
            if (lane < WMMA_N) {
                o_partials[(size_t)h * HIDDEN + n0 + lane] =
                    __float2bfloat16(warp_acc[lane]);
            }
            __syncwarp();
        }
    }
    // Non-head SMs idle (same as v5a attention).
}

// =============================================================================
// FUSED O-proj-reduce + RMSNorm_post + MLP gate/up.
//
// Replaces v5a's separate O-proj residual stage and MLP gate/up stage.
// Each SM:
//   1. Reads current hidden (residual) AND all 16 o_partials into smem.
//      Sums them: hidden_smem[d] = hidden[d] + sum_h o_partials[h][d].
//   2. Runs rmsnorm on hidden_smem (uses post_attn_norm gamma).
//   3. Also writes back the residual-updated hidden to GLOBAL (for the next
//      MLP down residual to add into) — this is needed because MLP down
//      also does hidden += ...
//   4. Produces its share of mlp_scratch tiles via WMMA on the normed smem.
// =============================================================================
__device__ void oproj_reduce_mlp_gate_up_stage(
    __nv_bfloat16* __restrict__ hidden,              // [HIDDEN] — residual, updated in place
    const __nv_bfloat16* __restrict__ o_partials,    // [NUM_HEADS, HIDDEN]
    const __nv_bfloat16* __restrict__ rms_w_post,
    const __nv_bfloat16* __restrict__ W_gate,
    const __nv_bfloat16* __restrict__ W_up,
    __nv_bfloat16* __restrict__ mlp_scratch,         // [INTER_DIM]
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool,
    float*         red_smem)
{
    constexpr int TOTAL_TILES = INTER_DIM / N_LARGE;   // 512
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

    // Step 1: hidden_smem[d] = hidden[d] + sum_h o_partials[h][d].
    // Each thread processes a subset of d in [0, HIDDEN). Read-only on
    // 'hidden' in this stage — the final post-O-proj residual update is
    // folded into mlp_down_residual_stage to avoid a cross-SM race on
    // hidden writes.
    for (int d = tid; d < HIDDEN; d += BLOCK_SIZE) {
        float acc = __bfloat162float(hidden[d]);
        #pragma unroll
        for (int h = 0; h < NUM_HEADS; h++) {
            acc += __bfloat162float(o_partials[(size_t)h * HIDDEN + d]);
        }
        x_smem[d] = __float2bfloat16(acc);
    }
    __syncthreads();

    // Step 2: rmsnorm in-place on x_smem using post_attn_norm.
    float inv_rms = compute_inv_rms(x_smem, HIDDEN, tid, red_smem);
    rmsnorm_smem_inplace(x_smem, rms_w_post, inv_rms, HIDDEN, tid);
    __syncthreads();

    // Step 3: MLP gate/up — per-tile WMMA, writes mlp_scratch[n] = silu(g)*u.
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
                wmma::load_matrix_sync(bu, W_up + k0 * INTER_DIM + n0, INTER_DIM);
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
// MLP down + residual + O-proj-partials-sum (v5b-specific fusion).
//
// Folds the "post-attention residual add" into mlp_down's residual update
// so that the O-proj-reduce stage can leave 'hidden' untouched (avoiding
// a cross-SM race on hidden writes).
//
// Computation:
//   hidden_new[m] = hidden_old[m] + sum_h o_partials[h][m] + mlp_out[m]
// Since each 16-col output tile is owned by exactly one warp, and each
// warp's lane writes a distinct m, there is NO race on hidden[m].
// =============================================================================
__device__ void mlp_down_plus_oproj_residual_stage(
    const __nv_bfloat16* __restrict__ mlp_scratch,
    const __nv_bfloat16* __restrict__ W_down,
    const __nv_bfloat16* __restrict__ o_partials,  // [NUM_HEADS, HIDDEN]
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
                int m = n0 + lane;
                float mlp_acc = c_scr[lane];
                float o_acc = 0.f;
                #pragma unroll
                for (int h = 0; h < NUM_HEADS; h++) {
                    o_acc += __bfloat162float(o_partials[(size_t)h * HIDDEN + m]);
                }
                float r = __bfloat162float(residual[m]);
                residual[m] = __float2bfloat16(r + o_acc + mlp_acc);
            }
            __syncwarp();
        }
    }
}

// =============================================================================
// Per-layer weight bundle (identical to v5a).
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
// Main kernel — 4 barriers/layer.
//
// Barrier schedule per layer:
//   [1/4]  after QKV (fused rmsnorm_pre)
//   [2/4]  after attention + partial-O-proj (FUSED, saves 1 barrier vs v5a)
//   [3/4]  after O-proj-reduce + MLP gate/up (replaces v5a's two stages)
//   [4/4]  after MLP down residual
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_30layer_v5b_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const LayerWeights* __restrict__ layers,
    __nv_bfloat16* __restrict__ q_scratch,
    __nv_bfloat16* __restrict__ attn_out,
    __nv_bfloat16* __restrict__ mlp_scratch,
    __nv_bfloat16* __restrict__ o_partials,  // NEW: [NUM_HEADS, HIDDEN]
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
        grid.sync();  // [1/4]

        attention_partial_oproj_stage(
            q_scratch, W.K_cache, W.V_cache, W.Wo,
            attn_out, o_partials,
            seq_len, sm, num_sms, tid,
            pool_scores, pool_q16, pool_p16, pool_o16,
            pool_a, pool_c, pool_red);
        grid.sync();  // [2/4]

        oproj_reduce_mlp_gate_up_stage(
            hidden, o_partials, W.post_attn_norm,
            W.W_gate, W.W_up, mlp_scratch,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_red);
        grid.sync();  // [3/4]

        mlp_down_plus_oproj_residual_stage(
            mlp_scratch, W.W_down, o_partials, hidden,
            sm, num_sms,
            pool_x, pool_a, pool_c);
        grid.sync();  // [4/4]
    }
}

// =============================================================================
// Host launchers (C ABI for ctypes)
// =============================================================================
extern "C" int mgg4_30_v5b_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_30_v5b_hidden()     { return HIDDEN; }
extern "C" int mgg4_30_v5b_inter_dim()  { return INTER_DIM; }
extern "C" int mgg4_30_v5b_num_heads()  { return NUM_HEADS; }
extern "C" int mgg4_30_v5b_head_dim()   { return HEAD_DIM; }
extern "C" int mgg4_30_v5b_max_seq()    { return MAX_SEQ; }
extern "C" int mgg4_30_v5b_num_layers() { return NUM_LAYERS; }
extern "C" size_t mgg4_30_v5b_layer_weights_size() { return sizeof(LayerWeights); }
extern "C" int mgg4_30_v5b_smem_bytes() { return SMEM_POOL_BYTES_ALIGNED; }

extern "C" cudaError_t mgg4_30_v5b_launch(
    __nv_bfloat16* hidden,
    const LayerWeights* layers_device,
    __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* mlp_scratch,
    __nv_bfloat16* o_partials,
    int seq_len,
    int num_layers_run,
    cudaStream_t stream)
{
    int num_sms = mgg4_30_v5b_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden,
        &layers_device,
        &q_scratch, &attn_out, &mlp_scratch,
        &o_partials,
        &seq_len,
        &num_layers_run,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_30layer_v5b_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_30layer_v5b_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_gemma4_30_v5b
