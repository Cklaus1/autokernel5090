// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — 30-LAYER v6 (FP4 MLP WEIGHTS).
//
// Derived from mega_graph_gemma4_30layer_v5a.cu. Key change:
//   * MLP stages (gate, up, down) use FP4-E2M1 packed weights + FP8-E4M3
//     per-block scales + fp32 global scale. Attention (QKV/O) and RMSNorm
//     remain BF16.
//   * Dequantization path (Option A): FP4 nibble unpack -> fp32 lookup ->
//     multiply by fp8 scale * fp32 global scale -> BF16 in smem ->
//     wmma::load_matrix_sync from BF16 smem.
//
// FP4 weight layout (per GEMM):
//   Packed weights:  uint8 [K, N/2] row-major. Low nibble = even N, high
//                    nibble = odd N. One K-row = N/2 bytes.
//   Block scales:    fp8_e4m3fn [K/16, N]. One scale per 16 K-elements per
//                    N-column.
//   Global scale:    fp32 scalar (single value for the full W tensor).
//
// Applied to:
//   W_gate: [HIDDEN=2048, INTER_DIM=8192] BF16 -> uint8 [2048, 4096] + fp8 [128, 8192]
//   W_up:   [HIDDEN=2048, INTER_DIM=8192] BF16 -> uint8 [2048, 4096] + fp8 [128, 8192]
//   W_down: [INTER_DIM=8192, HIDDEN=2048] BF16 -> uint8 [8192, 1024] + fp8 [512, 2048]
//
// WMMA B-tile load now goes through a per-warp BF16 staging area in smem:
//   For each K-step (k0 += WMMA_K=16):
//     Lanes cooperatively load 128 bytes of packed FP4 (16 rows × 8 bytes).
//     Lanes cooperatively load 16 fp8 scales (one per N column in the tile).
//     Dequant to BF16 into warp-local smem [16, 16] (512 bytes).
//     wmma::load_matrix_sync(b_frag, b_dq_smem, 16).
//
// Dequant cost estimate per tile:
//   * 4 × uint32_t loads (via 4-byte lane reads) = 4 instr for 128 B of FP4
//   * 16 fp8 -> fp32 converts (broadcast-loaded scales)
//   * 32 lanes × 8 elements × (1 table lookup + 2 fp32 muls + 1 bf16 cast)
//     = ~8 instr/elem, inlined in ~8 clock-equivalent cycles per lane
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs. WMMA 16x16x16
// BF16xBF16, FP32 accumulator.
//
// Shape: H=2048, INTER_DIM=8192, NUM_HEADS=16, HEAD_DIM=128, MAX_SEQ=256,
// NUM_LAYERS=30, M=1 decode.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

namespace mega_graph_gemma4_30_v6 {

// =============================================================================
// Config (identical to v5a).
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
static constexpr int   FP4_BLOCK = 16;  // elements per scale block (K direction)

// WMMA tile shape (BF16, 16x16x16).
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// N tile size (one WMMA per warp per K-step).
static constexpr int N_SMALL = 16;
static constexpr int MMAS_SMALL = N_SMALL / WMMA_N;   // 1
static constexpr int N_LARGE = 16;
static constexpr int MMAS_LARGE = N_LARGE / WMMA_N;   // 1
static_assert(HIDDEN    % N_SMALL == 0, "HIDDEN divisible by N_SMALL");
static_assert(INTER_DIM % N_LARGE == 0, "INTER_DIM divisible by N_LARGE");

// =============================================================================
// Shared-memory pool layout (v5a with B-dequant addition).
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

// B-dequant staging: TWO 16x16 BF16 tiles per warp (gate+up double buffer)
// to allow gate/up MMAs to overlap dequants. 1024 B/warp.
// Must be 16-byte aligned for ldmatrix.
static constexpr int SMEM_BDQ_RAW = SMEM_RED + SMEM_RED_SZ;
static constexpr int SMEM_BDQ     = (SMEM_BDQ_RAW + 15) & ~15;
static constexpr int SMEM_BDQ_SZ  = 2 * WARPS_PER_BLOCK * WMMA_K * WMMA_N * 2;  // 8192

// Attention aliases (unchanged from v5a).
static constexpr int SMEM_Q16       = SMEM_X;
static constexpr int SMEM_Q16_SZ    = WMMA_M * HEAD_DIM * 2;                // 4096
static constexpr int SMEM_P16       = SMEM_X + SMEM_Q16_SZ;
static constexpr int SMEM_P16_SZ    = WMMA_M * MAX_SEQ * 2;                 // 8192
static_assert(SMEM_Q16_SZ + SMEM_P16_SZ <= SMEM_X_SZ, "Q16+P16 fit in SMEM_X");
static constexpr int SMEM_O_ACC     = SMEM_C;
static constexpr int SMEM_O_ACC_SZ  = WMMA_M * HEAD_DIM * 4;                // 8192
static_assert(SMEM_O_ACC_SZ <= SMEM_C_SZ, "O_ACC fits in SMEM_C");
static constexpr int SMEM_SCORES    = SMEM_A;
static constexpr int SMEM_SCORES_SZ = MAX_SEQ * 4;                          // 1024
static_assert(SMEM_SCORES_SZ <= SMEM_A_SZ, "scores fits in SMEM_A");

static constexpr int SMEM_POOL_BYTES         = SMEM_BDQ + SMEM_BDQ_SZ;      // ~33 KB
static constexpr int SMEM_POOL_BYTES_ALIGNED = (SMEM_POOL_BYTES + 15) & ~15;

// =============================================================================
// FP4-E2M1 decode — 16-entry LUT in constant memory.
//
// Constant memory broadcasts a scalar to all threads; on divergent reads it
// SERIALIZES (one cycle per unique address in the warp). With 16 possible
// nibbles and 32 lanes, worst-case 16 cycles per load, but in practice the
// FP4 checkpoint shows nonzero-biased distributions that converge to ~4-6
// unique codes per warp-wide read. Still slower than register-backed LUT
// but more register-economical.
// =============================================================================
__device__ __constant__ float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// =============================================================================
// Block-wide reductions (unchanged from v5a).
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
    const __nv_bfloat16* __restrict__ src, int K, int tid, float* red_smem)
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
    __nv_bfloat16* x_smem, const __nv_bfloat16* __restrict__ rms_w,
    float inv_rms, int K, int tid)
{
    for (int d = tid; d < K; d += BLOCK_SIZE) {
        float x = __bfloat162float(x_smem[d]);
        float w = __bfloat162float(rms_w[d]);
        x_smem[d] = __float2bfloat16(x * inv_rms * w);
    }
}

__device__ __forceinline__ void load_a_tile_plain(
    const __nv_bfloat16* __restrict__ x, int k0,
    __nv_bfloat16* a_smem, int lane)
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
// Dequantize a single FP4 B-tile (16x16) into warp-local BF16 smem.
//
// Inputs:
//   fp4_ptr   : pointer to row k0, col-byte (n0/2) in the [K, N/2] FP4 tensor
//   ldb_bytes : row stride in BYTES of the FP4 tensor (= N/2)
//   sf_ptr    : pointer to scale row (k0/16), col n0 in the [K/16, N] FP8 scales
//   global_sc : fp32 global scale
//   b_dq_smem : per-warp output buffer [16, 16] BF16 (256 elems = 512 B)
//   lane      : threadIdx in warp (0..31)
//
// Strategy (2 lanes per row, 8 elements/lane):
//   Lane l handles row (l >> 1), and the 8 N-columns (l & 1) * 8 .. (l & 1)*8 + 7.
//   Lane reads 4 packed bytes = 8 FP4 values from [row, col_start/2 .. col_start/2+3].
//   16 scales (one per N-column) are loaded cooperatively then broadcast via shuffles.
// =============================================================================
__device__ __forceinline__ void dequant_fp4_tile_bf16(
    const uint8_t* __restrict__ fp4_ptr,
    int ldb_bytes,
    const __nv_fp8_e4m3* __restrict__ sf_ptr,
    float global_sc,
    __nv_bfloat16* b_dq_smem,
    int lane)
{
    // Per-lane tile mapping: lane l handles row (l>>1), and 8 N-columns
    // starting at col0 = (l&1)*8 .. col0+7.
    int row  = lane >> 1;           // 0..15
    int half = lane & 1;            // 0 => cols 0..7, 1 => cols 8..15
    int col0 = half * 8;            // first N-column this lane owns

    // Step 1: load 8 FP8 scales for this lane's N-columns directly.
    // This lane will re-read cols 0..7 (half=0) or cols 8..15 (half=1).
    // All 32 lanes touch the same 16 cache-line bytes → single 64-byte
    // L1 fill.
    const uint8_t* sf_bytes = reinterpret_cast<const uint8_t*>(sf_ptr);
    uint64_t scales_u64 = *reinterpret_cast<const uint64_t*>(sf_bytes + col0);
    float sc[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        uint8_t b = (scales_u64 >> (k * 8)) & 0xFFu;
        __nv_fp8_e4m3 fp8 = reinterpret_cast<__nv_fp8_e4m3&>(b);
        sc[k] = static_cast<float>(fp8) * global_sc;
    }

    // Step 2: Read 4 packed bytes = 8 FP4 nibbles for [row, col0..col0+7].
    const uint8_t* src = fp4_ptr + row * ldb_bytes + (col0 >> 1);
    uint32_t packed32 = *reinterpret_cast<const uint32_t*>(src);

    // Decode 8 FP4 nibbles to fp32 — bit-math decode.
    //
    // Encoding: sign=bit3, exp_bits=bits[2:1] (unbiased), mant_bit=bit0.
    //   if exp_bits==0: |v| = mant_bit * 0.5  (subnormal: 0 or 0.5)
    //   else:           |v| = (1 + 0.5*mant_bit) * 2^(exp_bits-1)
    //
    // Branch-free float construction via direct fp32 bit assembly:
    //   exp_field = exp_bits ? (127 + exp_bits - 1) : ((mant_bit) ? 126 : 0)
    //   mant_field = exp_bits ? (mant_bit << 22) : 0
    //   (subnormal: we use exp=126 (0.5) * mantissa=0 when mant_bit=1 → 0.5;
    //    when mant_bit=0, exp=0 mant=0 → 0.0. OK.)
    //   result_bits = (sign << 31) | (exp_field << 23) | mant_field
    auto dec = [](uint32_t c) -> float {
        uint32_t sign_bit = (c >> 3) & 1u;
        uint32_t exp_bits = (c >> 1) & 3u;
        uint32_t mant_bit =  c       & 1u;
        // exp_bits == 0 path: subnormal. exp_field = mant_bit ? 126 : 0.
        // exp_bits  > 0 path: exp_field = 127 + exp_bits - 1 = 126 + exp_bits.
        //                     mant_field = mant_bit << 22.
        uint32_t is_normal = exp_bits != 0u;
        uint32_t exp_field = is_normal
            ? (126u + exp_bits)
            : (mant_bit * 126u);
        uint32_t mant_field = is_normal ? (mant_bit << 22) : 0u;
        uint32_t bits = (sign_bit << 31) | (exp_field << 23) | mant_field;
        return __int_as_float((int)bits);
    };

    float f[8];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t byte = (packed32 >> (j * 8)) & 0xFFu;
        f[j * 2 + 0] = dec(byte & 0xFu);
        f[j * 2 + 1] = dec((byte >> 4) & 0xFu);
    }

    // Step 3: multiply and write BF16 out.
    __nv_bfloat16* dst = b_dq_smem + row * 16 + col0;
    #pragma unroll
    for (int k = 0; k < 8; k += 2) {
        __nv_bfloat162 v2 = __halves2bfloat162(
            __float2bfloat16(f[k    ] * sc[k    ]),
            __float2bfloat16(f[k + 1] * sc[k + 1]));
        *reinterpret_cast<__nv_bfloat162*>(dst + k) = v2;
    }
}

// =============================================================================
// QKV + RMSNorm (BF16, unchanged from v5a).
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
// Attention (BF16, unchanged from v5a).
// =============================================================================
__device__ void attention_core_stage(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ attn_out,
    int seq_len,
    int sm, int num_sms, int tid,
    float* scores_smem,
    __nv_bfloat16* q16_smem,
    __nv_bfloat16* p16_smem,
    float*         o16_smem,
    float* red_smem)
{
    if (sm >= NUM_HEADS) return;

    const int h = sm;
    constexpr float inv_sqrt_d = 1.0f / 11.3137084989848f;  // 1/sqrt(128)
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

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
            const __nv_bfloat16* bptr =
                K + (size_t)n0 * HIDDEN + h * HEAD_DIM + k0;
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

    constexpr int VP_N_TILES = HEAD_DIM / WMMA_N;
    constexpr int VP_K_STEPS = MAX_SEQ  / WMMA_K;

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

    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        attn_out[h * HEAD_DIM + d] = __float2bfloat16(o16_smem[d]);
    }
}

// =============================================================================
// O-proj + residual (BF16, unchanged from v5a).
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
// FP4 MLP gate/up + RMSNorm (v6 NEW).
//
// gate/up weights are FP4 packed [HIDDEN, INTER_DIM/2] + fp8 scales
// [HIDDEN/16, INTER_DIM] + fp32 global scales.
// =============================================================================
__device__ void mlp_gate_up_fp4_fused_rmsnorm_stage(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ rms_w_post,
    const uint8_t*        __restrict__ W_gate_fp4,
    const __nv_fp8_e4m3*  __restrict__ W_gate_sf,
    float                              W_gate_gs,
    const uint8_t*        __restrict__ W_up_fp4,
    const __nv_fp8_e4m3*  __restrict__ W_up_sf,
    float                              W_up_gs,
    __nv_bfloat16* __restrict__ mlp_scratch,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool,
    __nv_bfloat16* b_dq_smem_pool,
    float*         red_smem)
{
    constexpr int TOTAL_TILES = INTER_DIM / N_LARGE;   // 512
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem     = a_smem_pool     + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr      = c_scratch_pool  + warp_id * (WMMA_M * WMMA_N);
    // Two 16x16 BF16 buffers per warp: [gate, up].
    __nv_bfloat16* b_dq_gate  = b_dq_smem_pool  + (2 * warp_id + 0) * (WMMA_K * WMMA_N);
    __nv_bfloat16* b_dq_up    = b_dq_smem_pool  + (2 * warp_id + 1) * (WMMA_K * WMMA_N);

    // RMSNorm into x_smem.
    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();
    float inv_rms = compute_inv_rms(x_smem, HIDDEN, tid, red_smem);
    rmsnorm_smem_inplace(x_smem, rms_w_post, inv_rms, HIDDEN, tid);
    __syncthreads();

    // Row stride of FP4 weights in bytes: INTER_DIM / 2.
    const int ldb_fp4 = INTER_DIM / 2;

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

            int sf_row = k0 / FP4_BLOCK;  // one scale row per 16 K-elements

            #pragma unroll
            for (int j = 0; j < MMAS_LARGE; j++) {
                int n0 = n0_base + j * WMMA_N;

                // Dequant gate AND up into separate smem buffers. The compiler
                // can now interleave/overlap the gate and up dequants.
                {
                    const uint8_t*       fp4_g = W_gate_fp4 + k0 * ldb_fp4 + (n0 >> 1);
                    const __nv_fp8_e4m3* sf_g  = W_gate_sf  + sf_row * INTER_DIM + n0;
                    dequant_fp4_tile_bf16(fp4_g, ldb_fp4, sf_g, W_gate_gs, b_dq_gate, lane);

                    const uint8_t*       fp4_u = W_up_fp4 + k0 * ldb_fp4 + (n0 >> 1);
                    const __nv_fp8_e4m3* sf_u  = W_up_sf  + sf_row * INTER_DIM + n0;
                    dequant_fp4_tile_bf16(fp4_u, ldb_fp4, sf_u, W_up_gs, b_dq_up, lane);
                }
                __syncwarp();
                {
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bg;
                    wmma::load_matrix_sync(bg, b_dq_gate, WMMA_N);
                    wmma::mma_sync(cg[j], a_frag, bg, cg[j]);

                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> bu;
                    wmma::load_matrix_sync(bu, b_dq_up, WMMA_N);
                    wmma::mma_sync(cu[j], a_frag, bu, cu[j]);
                }
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
// FP4 MLP down + residual (v6 NEW).
//
// W_down is FP4 packed [INTER_DIM, HIDDEN/2] + fp8 scales [INTER_DIM/16, HIDDEN]
// + fp32 global scale.
// =============================================================================
__device__ void mlp_down_fp4_residual_stage(
    const __nv_bfloat16* __restrict__ mlp_scratch,
    const uint8_t*       __restrict__ W_down_fp4,
    const __nv_fp8_e4m3* __restrict__ W_down_sf,
    float                             W_down_gs,
    __nv_bfloat16* __restrict__ residual,
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float*         c_scratch_pool,
    __nv_bfloat16* b_dq_smem_pool)
{
    constexpr int TOTAL_TILES = HIDDEN / N_SMALL;   // 128
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem    = a_smem_pool     + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr     = c_scratch_pool  + warp_id * (WMMA_M * WMMA_N);
    // mlp_down only needs one B-dequant buffer per warp; reuse first slot.
    __nv_bfloat16* b_dq_smem = b_dq_smem_pool  + (2 * warp_id + 0) * (WMMA_K * WMMA_N);

    stage_vector_to_smem(mlp_scratch, x_smem, INTER_DIM, tid);
    __syncthreads();

    // Row stride of FP4 weights in bytes: HIDDEN / 2.
    const int ldb_fp4 = HIDDEN / 2;

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

            int sf_row = k0 / FP4_BLOCK;

            #pragma unroll
            for (int j = 0; j < MMAS_SMALL; j++) {
                int n0 = n0_base + j * WMMA_N;
                const uint8_t*       fp4_ptr = W_down_fp4 + k0 * ldb_fp4 + (n0 >> 1);
                const __nv_fp8_e4m3* sf_ptr  = W_down_sf  + sf_row * HIDDEN + n0;
                dequant_fp4_tile_bf16(fp4_ptr, ldb_fp4, sf_ptr, W_down_gs, b_dq_smem, lane);
                __syncwarp();
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, b_dq_smem, WMMA_N);
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
// Per-layer weight bundle (v6: adds FP4 MLP tensors).
// =============================================================================
struct LayerWeights {
    // BF16 path (unchanged).
    const __nv_bfloat16* input_norm;
    const __nv_bfloat16* Wq;
    const __nv_bfloat16* Wk;
    const __nv_bfloat16* Wv;
    const __nv_bfloat16* Wo;
    const __nv_bfloat16* post_attn_norm;
    // FP4 MLP weights + fp8 scales.
    const uint8_t*       W_gate_fp4;
    const __nv_fp8_e4m3* W_gate_sf;
    float                W_gate_gs;
    const uint8_t*       W_up_fp4;
    const __nv_fp8_e4m3* W_up_sf;
    float                W_up_gs;
    const uint8_t*       W_down_fp4;
    const __nv_fp8_e4m3* W_down_sf;
    float                W_down_gs;
    // KV cache.
    __nv_bfloat16* K_cache;
    __nv_bfloat16* V_cache;
};

// =============================================================================
// Main kernel — 5 barriers/layer (same as v5a).
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_30layer_v6_kernel(
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
    __nv_bfloat16* pool_bdq    = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_BDQ);
    __nv_bfloat16* pool_q16    = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_Q16);
    __nv_bfloat16* pool_p16    = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_P16);
    float*         pool_o16    = reinterpret_cast<float*>        (smem_pool + SMEM_O_ACC);
    float*         pool_scores = reinterpret_cast<float*>        (smem_pool + SMEM_SCORES);

    for (int L = 0; L < num_layers_run; L++) {
        const LayerWeights& W = layers[L];

        __nv_bfloat16* k_slot = W.K_cache + (seq_len - 1) * HIDDEN;
        __nv_bfloat16* v_slot = W.V_cache + (seq_len - 1) * HIDDEN;
        qkv_proj_fused_rmsnorm_stage(
            hidden, W.input_norm, W.Wq, W.Wk, W.Wv,
            q_scratch, k_slot, v_slot,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_red);
        grid.sync();  // [1/5]

        attention_core_stage(q_scratch, W.K_cache, W.V_cache, attn_out,
                             seq_len, sm, num_sms, tid,
                             pool_scores, pool_q16, pool_p16, pool_o16, pool_red);
        grid.sync();  // [2/5]

        attn_oproj_residual_stage(attn_out, W.Wo, hidden,
                                  sm, num_sms,
                                  pool_x, pool_a, pool_c);
        grid.sync();  // [3/5]

        mlp_gate_up_fp4_fused_rmsnorm_stage(
            hidden, W.post_attn_norm,
            W.W_gate_fp4, W.W_gate_sf, W.W_gate_gs,
            W.W_up_fp4,   W.W_up_sf,   W.W_up_gs,
            mlp_scratch,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_bdq, pool_red);
        grid.sync();  // [4/5]

        mlp_down_fp4_residual_stage(
            mlp_scratch,
            W.W_down_fp4, W.W_down_sf, W.W_down_gs,
            hidden,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_bdq);
        grid.sync();  // [5/5]
    }
}

// =============================================================================
// Host launchers (C ABI for ctypes)
// =============================================================================
extern "C" int mgg4_30_v6_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_30_v6_hidden()     { return HIDDEN; }
extern "C" int mgg4_30_v6_inter_dim()  { return INTER_DIM; }
extern "C" int mgg4_30_v6_num_heads()  { return NUM_HEADS; }
extern "C" int mgg4_30_v6_head_dim()   { return HEAD_DIM; }
extern "C" int mgg4_30_v6_max_seq()    { return MAX_SEQ; }
extern "C" int mgg4_30_v6_num_layers() { return NUM_LAYERS; }
extern "C" int mgg4_30_v6_fp4_block()  { return FP4_BLOCK; }
extern "C" size_t mgg4_30_v6_layer_weights_size() { return sizeof(LayerWeights); }
extern "C" int mgg4_30_v6_smem_bytes() { return SMEM_POOL_BYTES_ALIGNED; }

extern "C" cudaError_t mgg4_30_v6_launch(
    __nv_bfloat16* hidden,
    const LayerWeights* layers_device,
    __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* mlp_scratch,
    int seq_len,
    int num_layers_run,
    cudaStream_t stream)
{
    int num_sms = mgg4_30_v6_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden, &layers_device,
        &q_scratch, &attn_out, &mlp_scratch,
        &seq_len, &num_layers_run,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_30layer_v6_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_30layer_v6_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_gemma4_30_v6
