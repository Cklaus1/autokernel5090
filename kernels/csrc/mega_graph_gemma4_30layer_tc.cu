// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — 30-LAYER TENSOR-CORE VERSION.
//
// Derived from mega_graph_gemma4_30layer.cu; the 4 per-layer matmuls
// (QKV proj, O proj, MLP gate+up, MLP down) are rewritten from scalar
// BF16 FMA loops to nvcuda::wmma fragment APIs targeting
// mma.sync.aligned.m16n16k16 tensor-core instructions on SM120a.
//
// Key design choices for M=1 decode (GEMV):
//   * WMMA fragments require M>=16, so we replicate the activation row into
//     a 16xK A-fragment. Only row 0 of the output fragment is written back.
//     Cost: 16x compute overhead, but tensor cores are ~32x faster than
//     scalar BF16 cores -> net ~2x compute speedup vs scalar FMA.
//   * Each WARP owns a 16-column output tile [n0, n0+16). Work is
//     distributed across global (sm*warps_per_sm + warp) index; with
//     num_sms=188 and 8 warps/SM = 1504 warps, enough to cover all
//     N-tiles even for INTER_DIM=8192 (512 tiles).
//   * BF16 weights + BF16 activations; FP32 accumulator inside the
//     fragment. Converted back to BF16 on store.
//   * RMSNorm, attention softmax, SwiGLU remain scalar (elementwise).
//   * Cooperative launch + grid.sync barrier layout is preserved exactly
//     from v1 (7 barriers/layer).
//
// Shared memory strategy: to stay under the 48 KB/SM static smem cap (the
// launch-time dynamic-smem attribute can bump this, but we prefer the
// cooperative-launch default), we carve ONE dynamic shared-memory pool and
// partition it manually per-stage. The pool holds:
//   * 8 warps * 16*16 bf16 = 4096 B  (a_smem_all)
//   * 8 warps * 16*16 fp32 = 8192 B  (c_scratch_0)
//   * 8 warps * 16*16 fp32 = 8192 B  (c_scratch_1, used by mlp_gate_up)
//   * 32 fp32                = 128 B (block-reduce red_smem)
//   * MAX_SEQ fp32 + HEAD_DIM bf16 = 1280 B (attention scores + q_h)
// Since these regions are used DISJOINTLY across stages, we union-overlay
// them starting from a single base pointer. Total footprint = max stage
// requirement = 4096 + 16384 + 128 = 20608 B in mlp_gate_up (the peak).
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

namespace mega_graph_gemma4_30_tc {

// =============================================================================
// Config
// =============================================================================
static constexpr int BLOCK_SIZE     = 256;
static constexpr int BLOCKS_PER_SM  = 1;
static constexpr int WARP_SIZE      = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;   // 8

static constexpr int HIDDEN     = 2048;
static constexpr int NUM_HEADS  = 16;
static constexpr int HEAD_DIM   = 128;
static constexpr int INTER_DIM  = 8192;
static constexpr int MAX_SEQ    = 256;
static constexpr int NUM_LAYERS = 30;

static_assert(NUM_HEADS * HEAD_DIM == HIDDEN, "HIDDEN must equal NUM_HEADS*HEAD_DIM");
static_assert(HIDDEN    % 16 == 0, "HIDDEN must be multiple of 16 for WMMA");
static_assert(INTER_DIM % 16 == 0, "INTER_DIM must be multiple of 16 for WMMA");

static constexpr float RMS_EPS = 1e-6f;

// WMMA tile shape (BF16, 16x16x16).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Shared-memory pool layout (bytes, relative to pool base).
// All stages alias the SAME pool; only one stage runs at a time.
// Layout:
//   SMEM_X (INTER_DIM bf16 = 16 KB)  — activation cache (hoisted x)
//   SMEM_A (8 warps * 16*16 bf16 = 4 KB) — per-warp A-tile
//   SMEM_C0 (8 warps * 16*16 fp32 = 8 KB) — per-warp C accumulator
//   SMEM_C1 (8 warps * 16*16 fp32 = 8 KB) — per-warp C accumulator (mlp_gate_up)
//   SMEM_RED (32 fp32 = 128 B) — block-reduce buffer
// Total peak = 16+4+8+8+0.128 ≈ 36 KB (with C1 only live in mlp_gate_up; in
// other stages that region is unused).
static constexpr int SMEM_X      = 0;
static constexpr int SMEM_X_SZ   = INTER_DIM * 2;                           // 16384
static constexpr int SMEM_A      = SMEM_X + SMEM_X_SZ;
static constexpr int SMEM_A_SZ   = WARPS_PER_BLOCK * WMMA_M * WMMA_K * 2;   // 4096
static constexpr int SMEM_C0     = SMEM_A + SMEM_A_SZ;
static constexpr int SMEM_C0_SZ  = WARPS_PER_BLOCK * WMMA_M * WMMA_N * 4;   // 8192
static constexpr int SMEM_C1     = SMEM_C0 + SMEM_C0_SZ;
static constexpr int SMEM_C1_SZ  = WARPS_PER_BLOCK * WMMA_M * WMMA_N * 4;   // 8192
static constexpr int SMEM_RED    = SMEM_C1 + SMEM_C1_SZ;
static constexpr int SMEM_RED_SZ = 32 * 4;                                  // 128
// Attention scratch overlays SMEM_A (unused in attention stage). scores[MAX_SEQ]
// then q_h[HEAD_DIM] -- 1024 + 256 = 1280 B, fits in SMEM_A's 4096 B.
static constexpr int SMEM_SCORES    = SMEM_A;
static constexpr int SMEM_SCORES_SZ = MAX_SEQ * 4;                          // 1024
static constexpr int SMEM_QH        = SMEM_SCORES + SMEM_SCORES_SZ;
static constexpr int SMEM_QH_SZ     = HEAD_DIM * 2;                         // 256

static constexpr int SMEM_POOL_BYTES = SMEM_RED + SMEM_RED_SZ;              // 36992
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
// RMSNorm — local per-SM (scalar; unchanged from v1 math)
// =============================================================================
__device__ void rmsnorm_local_stage(
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ dst,
    int sm, int num_sms, int tid,
    float* red_smem)
{
    float local_ss = 0.f;
    for (int d = tid; d < HIDDEN; d += BLOCK_SIZE) {
        float x = __bfloat162float(src[d]);
        local_ss += x * x;
    }
    float block_ss = block_reduce_sum(local_ss, red_smem);
    float inv_rms = rsqrtf(block_ss / (float)HIDDEN + RMS_EPS);

    int dims_per_sm = (HIDDEN + num_sms - 1) / num_sms;
    int d0 = sm * dims_per_sm;
    int d1 = min(d0 + dims_per_sm, HIDDEN);
    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float x = __bfloat162float(src[d]) * inv_rms;
        float w = __bfloat162float(weight[d]);
        dst[d] = __float2bfloat16(x * w);
    }
}

// =============================================================================
// Tensor-core GEMV helpers — A-tile loader.
//
// Materialize a 16x16 BF16 A-tile in shared memory where every row == x[k0:k0+16].
// Called per-warp (each warp has its own a_smem[256] bf16 region).
// =============================================================================
__device__ __forceinline__ void load_a_tile_warp(
    const __nv_bfloat16* __restrict__ x,
    int k0,
    __nv_bfloat16* a_smem,   // [256] bf16 for this warp
    int lane)
{
    __nv_bfloat162* a_smem_2 = reinterpret_cast<__nv_bfloat162*>(a_smem);
    // 128 pair slots (=256 bf16s). 32 lanes * 4 pairs/lane = 128.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int pair_idx = lane * 4 + i;       // 0..127
        int col_pair = pair_idx & 7;        // 0..7  (col = 2*col_pair)
        int kc = k0 + (col_pair << 1);
        __nv_bfloat16 a0 = x[kc + 0];
        __nv_bfloat16 a1 = x[kc + 1];
        a_smem_2[pair_idx] = __halves2bfloat162(a0, a1);
    }
}

// Stage a whole vector (length K) into shared memory using the block's
// BLOCK_SIZE threads. K must be a multiple of 2 (we vector-copy bf162).
__device__ __forceinline__ void stage_vector_to_smem(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16*                     dst_smem,
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
// Tensor-core QKV projection: q, k, v = x @ {Wq, Wk, Wv}.
// Distributed across (sm, warp) over 3 * (HIDDEN/16) = 384 tiles.
// =============================================================================
__device__ void qkv_proj_stage_tc(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ Wk,
    const __nv_bfloat16* __restrict__ Wv,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,        // [HIDDEN] bf16 (block-shared cache of x)
    __nv_bfloat16* a_smem_pool,   // [WARPS_PER_BLOCK * 256] bf16
    float*         c_scratch_pool)// [WARPS_PER_BLOCK * 256] fp32
{
    constexpr int TILES_PER_PROJ = HIDDEN / WMMA_N;    // 128
    constexpr int TOTAL_TILES    = 3 * TILES_PER_PROJ; // 384
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool  + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr  = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);

    // Stage x into shared memory once; all warps reuse.
    stage_vector_to_smem(x, x_smem, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int proj = tile / TILES_PER_PROJ;
        int sub  = tile - proj * TILES_PER_PROJ;
        int n0   = sub * WMMA_N;

        const __nv_bfloat16* W;
        __nv_bfloat16* y;
        if      (proj == 0) { W = Wq; y = q_out; }
        else if (proj == 1) { W = Wk; y = k_out; }
        else                { W = Wv; y = v_out; }

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            load_a_tile_warp(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, W + k0 * HIDDEN + n0, HIDDEN);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(c_scr, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            y[n0 + lane] = __float2bfloat16(c_scr[lane]);
        }
        __syncwarp();
    }
}

// =============================================================================
// Attention core — unchanged (elementwise softmax, scalar dot products).
// =============================================================================
__device__ void attention_core_stage(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ attn_out,
    int seq_len,
    int sm, int num_sms, int tid,
    float* scores_smem,             // [MAX_SEQ] fp32
    __nv_bfloat16* q_h_smem,        // [HEAD_DIM] bf16
    float* red_smem)                // [32] fp32
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
// Tensor-core O-projection + residual: residual += attn_in @ Wo.
// =============================================================================
__device__ void attn_oproj_residual_stage_tc(
    const __nv_bfloat16* __restrict__ attn_in,
    const __nv_bfloat16* __restrict__ Wo,
    __nv_bfloat16* __restrict__ residual,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool)
{
    constexpr int TOTAL_TILES = HIDDEN / WMMA_N;
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
        int n0 = tile * WMMA_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            load_a_tile_warp(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, Wo + k0 * HIDDEN + n0, HIDDEN);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
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
}

// =============================================================================
// Tensor-core MLP gate+up: mlp_scratch[j] = silu(gate_j) * up_j.
// =============================================================================
__device__ void mlp_gate_up_stage_tc(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ W_gate,
    const __nv_bfloat16* __restrict__ W_up,
    __nv_bfloat16* __restrict__ mlp_scratch,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_g_pool,
    float*         c_scratch_u_pool)
{
    constexpr int TOTAL_TILES = INTER_DIM / WMMA_N;   // 512
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem = a_smem_pool      + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr_g = c_scratch_g_pool + warp_id * (WMMA_M * WMMA_N);
    float*         c_scr_u = c_scratch_u_pool + warp_id * (WMMA_M * WMMA_N);

    stage_vector_to_smem(x, x_smem, HIDDEN, tid);
    __syncthreads();

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int n0 = tile * WMMA_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cg_frag, cu_frag;
        wmma::fill_fragment(cg_frag, 0.0f);
        wmma::fill_fragment(cu_frag, 0.0f);

        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            load_a_tile_warp(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bg_frag, bu_frag;
            wmma::load_matrix_sync(bg_frag, W_gate + k0 * INTER_DIM + n0, INTER_DIM);
            wmma::load_matrix_sync(bu_frag, W_up   + k0 * INTER_DIM + n0, INTER_DIM);

            wmma::mma_sync(cg_frag, a_frag, bg_frag, cg_frag);
            wmma::mma_sync(cu_frag, a_frag, bu_frag, cu_frag);
        }

        wmma::store_matrix_sync(c_scr_g, cg_frag, WMMA_N, wmma::mem_row_major);
        wmma::store_matrix_sync(c_scr_u, cu_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();

        if (lane < WMMA_N) {
            float g = c_scr_g[lane];
            float u = c_scr_u[lane];
            float silu_g = g / (1.0f + expf(-g));
            mlp_scratch[n0 + lane] = __float2bfloat16(silu_g * u);
        }
        __syncwarp();
    }
}

// =============================================================================
// Tensor-core MLP down + residual: residual += mlp_scratch @ W_down.
// =============================================================================
__device__ void mlp_down_residual_stage_tc(
    const __nv_bfloat16* __restrict__ mlp_scratch,
    const __nv_bfloat16* __restrict__ W_down,
    __nv_bfloat16* __restrict__ residual,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool)
{
    constexpr int TOTAL_TILES = HIDDEN / WMMA_N;  // 128
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
        int n0 = tile * WMMA_N;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k0 = 0; k0 < INTER_DIM; k0 += WMMA_K) {
            load_a_tile_warp(x_smem, k0, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, W_down + k0 * HIDDEN + n0, HIDDEN);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
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
}

// =============================================================================
// Per-layer weight bundle
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
// Main kernel
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_30layer_tc_kernel(
    __nv_bfloat16* __restrict__ hidden,
    const LayerWeights* __restrict__ layers,
    __nv_bfloat16* __restrict__ normed,
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

    // Dynamic smem pool, partitioned per-stage. Each stage uses a subset of
    // the pool; stages do not execute concurrently, so aliasing is safe.
    extern __shared__ __align__(16) unsigned char smem_pool[];
    __nv_bfloat16* pool_x_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_X);
    __nv_bfloat16* pool_a_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_A);
    float*         pool_c0     = reinterpret_cast<float*>        (smem_pool + SMEM_C0);
    float*         pool_c1     = reinterpret_cast<float*>        (smem_pool + SMEM_C1);
    float*         pool_red    = reinterpret_cast<float*>        (smem_pool + SMEM_RED);
    float*         pool_scores = reinterpret_cast<float*>        (smem_pool + SMEM_SCORES);
    __nv_bfloat16* pool_qh     = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_QH);

    for (int L = 0; L < num_layers_run; L++) {
        const LayerWeights& W = layers[L];

        rmsnorm_local_stage(hidden, W.input_norm, normed, sm, num_sms, tid, pool_red);
        grid.sync();  // [1/7]

        __nv_bfloat16* k_slot = W.K_cache + (seq_len - 1) * HIDDEN;
        __nv_bfloat16* v_slot = W.V_cache + (seq_len - 1) * HIDDEN;
        qkv_proj_stage_tc(normed, W.Wq, W.Wk, W.Wv,
                          q_scratch, k_slot, v_slot,
                          sm, num_sms,
                          pool_x_bf16, pool_a_bf16, pool_c0);
        grid.sync();  // [2/7]

        attention_core_stage(q_scratch, W.K_cache, W.V_cache, attn_out,
                             seq_len, sm, num_sms, tid,
                             pool_scores, pool_qh, pool_red);
        grid.sync();  // [3/7]

        attn_oproj_residual_stage_tc(attn_out, W.Wo, hidden,
                                     sm, num_sms,
                                     pool_x_bf16, pool_a_bf16, pool_c0);
        grid.sync();  // [4/7]

        rmsnorm_local_stage(hidden, W.post_attn_norm, normed, sm, num_sms, tid, pool_red);
        grid.sync();  // [5/7]

        mlp_gate_up_stage_tc(normed, W.W_gate, W.W_up, mlp_scratch,
                             sm, num_sms,
                             pool_x_bf16, pool_a_bf16, pool_c0, pool_c1);
        grid.sync();  // [6/7]

        mlp_down_residual_stage_tc(mlp_scratch, W.W_down, hidden,
                                   sm, num_sms,
                                   pool_x_bf16, pool_a_bf16, pool_c0);
        grid.sync();  // [7/7]
    }
}

// =============================================================================
// Host launchers
// =============================================================================
extern "C" int mgg4_30_tc_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_30_tc_hidden()     { return HIDDEN; }
extern "C" int mgg4_30_tc_inter_dim()  { return INTER_DIM; }
extern "C" int mgg4_30_tc_num_heads()  { return NUM_HEADS; }
extern "C" int mgg4_30_tc_head_dim()   { return HEAD_DIM; }
extern "C" int mgg4_30_tc_max_seq()    { return MAX_SEQ; }
extern "C" int mgg4_30_tc_num_layers() { return NUM_LAYERS; }
extern "C" size_t mgg4_30_tc_layer_weights_size() { return sizeof(LayerWeights); }

extern "C" int mgg4_30_tc_smem_bytes() { return SMEM_POOL_BYTES_ALIGNED; }

extern "C" cudaError_t mgg4_30_tc_launch(
    __nv_bfloat16* hidden,
    const LayerWeights* layers_device,
    __nv_bfloat16* normed,
    __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* mlp_scratch,
    int seq_len,
    int num_layers_run,
    cudaStream_t stream)
{
    int num_sms = mgg4_30_tc_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden,
        &layers_device,
        &normed, &q_scratch, &attn_out, &mlp_scratch,
        &seq_len,
        &num_layers_run,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_30layer_tc_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_30layer_tc_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_gemma4_30_tc
