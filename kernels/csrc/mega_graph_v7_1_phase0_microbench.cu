// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph v7.1 Phase 0 — K-gang real-MoE FP4 microbench.
//
// Goal: break the v7 Phase 0 layout ceiling (4.7% HBM on Qwen3 [N, K/2] layout).
//
// v7 Phase 0 root cause (from plans/mega_graph_v7_phase0_results.md §3):
//   Each warp dequanted a [16 K, 16 N] WMMA tile by reading 16 N-rows that
//   were 1024 B apart. This generated 16 L1 cache line loads (~128 B each)
//   with only 8 B consumed per line → 6% cache-line utilization → ceiling
//   ~6% × 1792 GB/s = ~112 GB/s. Measured 83.6 GB/s = 75% of ceiling.
//
// v7.1 K-gang design:
//   Replace per-warp [16 K, 16 N] tile dequant with per-BLOCK [16 N, 128 K]
//   cooperative dequant across all 8 warps (256 threads). Since K is the
//   CONTIGUOUS axis in the Qwen3 `[N, K/2]` layout:
//
//     - 128 K elements per N-row = 64 packed bytes (= ½ of a 128 B cache line).
//     - 16 N-rows × 64 B per row = 1024 B of FP4 = 16 rows × ½-line each.
//     - Each cache line carries the full row's 128 B worth ≥ 50% used (and
//       fully amortized once we chain two adjacent K-blocks inside the loop
//       structure — the L1 retains the line).
//
//   Amortization expectation: ~50-100% cache-line utilization (vs 6%),
//   projecting ≥ 28% HBM on the Phase 0 workload (PASS gate). ≥ 50% =
//   BIG_WIN, unlocks v7 full-layer.
//
// Block / work mapping (differs from v7 where a warp owned an N-tile):
//   - One CUDA block owns ONE (expert, N-tile) work item.
//   - Grid stride across blocks covers the 384 (TOP_K × 48) GEMM1 tiles and
//     1024 (TOP_K × 128) GEMM2 tiles.
//   - Inside a block: 8 warps cooperate on each K-gang load, then each of
//     the 8 warps executes one 16-K sub-WMMA (a K-gang = 8 × WMMA_K = 128 K).
//     Accumulators are reduced across warps at the end into a single
//     per-tile output.
//
// The MMA math is unchanged (BF16 WMMA 16×16×16). Only the LOAD phase is
// restructured. Correctness target: cos ≥ 0.999, max_abs ≤ 1e-3 vs the
// same PyTorch BF16 reference used by v7 Phase 0 (which reached 1.0000).
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs.
// =============================================================================

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

namespace mega_graph_v7_1_phase0 {

// =============================================================================
// Config (same Qwen3-30B-A3B shapes as v7)
// =============================================================================
static constexpr int BLOCK_SIZE      = 256;
static constexpr int BLOCKS_PER_SM   = 1;
static constexpr int WARP_SIZE       = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;   // 8

static constexpr int HIDDEN        = 2048;
static constexpr int INTER_PER_EXP = 768;
static constexpr int TOP_K         = 8;
static constexpr int FP4_BLOCK     = 16;

// WMMA 16x16x16 BF16
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// K-gang: per-block, all 8 warps cooperatively load WMMA_K * WARPS_PER_BLOCK
// contiguous K elements at once (= 128 K).
static constexpr int KGANG_K = WMMA_K * WARPS_PER_BLOCK;         // 128

static_assert(HIDDEN        % KGANG_K == 0, "HIDDEN must divide KGANG_K");
static_assert(INTER_PER_EXP % WMMA_K  == 0, "INTER must divide WMMA_K");
static_assert(HIDDEN        % WMMA_N  == 0, "HIDDEN must divide WMMA_N");
static_assert(INTER_PER_EXP % WMMA_N  == 0, "INTER must divide WMMA_N");

// down: K = INTER_PER_EXP = 768. 768 / 128 = 6 gangs. OK.
static_assert(INTER_PER_EXP % KGANG_K == 0 || (INTER_PER_EXP % WMMA_K == 0),
              "down path supports WMMA_K-granular remainder");

// Per-expert weight sizes (same as v7).
static constexpr size_t GATE_UP_FP4_BYTES = (size_t)INTER_PER_EXP * (HIDDEN / 2);
static constexpr size_t GATE_UP_SF_BYTES  = (size_t)INTER_PER_EXP * (HIDDEN / FP4_BLOCK);
static constexpr size_t DOWN_FP4_BYTES    = (size_t)HIDDEN * (INTER_PER_EXP / 2);
static constexpr size_t DOWN_SF_BYTES     = (size_t)HIDDEN * (INTER_PER_EXP / FP4_BLOCK);

// FP4-E2M1 16-entry LUT (constant memory).
__device__ __constant__ float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// =============================================================================
// Smem layout (per block):
//   SMEM_X        : HIDDEN BF16 (4 KB) — activation staging.
//   SMEM_A        : 16×16 BF16 (512 B) — A-tile shared by all 8 warps (M=1,
//                   each warp uses the SAME A since they all compute the same
//                   output row of the [1, INTER] × [INTER, N_TILE] sub-GEMM).
//                   Actually with K-gang, warp w reads A-rows k0+w*16..k0+w*16+15
//                   — different K sub-slabs — so each warp needs its own A-tile.
//                   Allocate per-warp A-tile (8 × 512 B = 4 KB).
//   SMEM_BG       : 16 N × 128 K BF16 (4 KB) — K-gang dequant for gate.
//   SMEM_BU       : 16 N × 128 K BF16 (4 KB) — K-gang dequant for up.
//   SMEM_RED_G    : 8 warps × 16 outputs × 4 B fp32 = 512 B — cross-warp
//                   accumulator reduction for gate.
//   SMEM_RED_U    : 512 B — same for up.
// Total: 4K + 4K + 4K + 4K + 0.5K + 0.5K = ~17 KB
// For the down kernel (K=INTER=768), one SMEM_BG is used (no up). Subset of
// same pool.
// =============================================================================
static constexpr int SMEM_X_SZ     = HIDDEN * 2;                                // 4096
static constexpr int SMEM_A_SZ     = WARPS_PER_BLOCK * WMMA_M * WMMA_K * 2;     // 4096
static constexpr int SMEM_BG_SZ    = WMMA_N * KGANG_K * 2;                      // 4096
static constexpr int SMEM_BU_SZ    = WMMA_N * KGANG_K * 2;                      // 4096
static constexpr int SMEM_RED_G_SZ = WARPS_PER_BLOCK * WMMA_N * 4;              // 512
static constexpr int SMEM_RED_U_SZ = WARPS_PER_BLOCK * WMMA_N * 4;              // 512

static constexpr int SMEM_X     = 0;
static constexpr int SMEM_A     = SMEM_X     + SMEM_X_SZ;
static constexpr int SMEM_BG    = SMEM_A     + SMEM_A_SZ;
static constexpr int SMEM_BU    = SMEM_BG    + SMEM_BG_SZ;
static constexpr int SMEM_RED_G = SMEM_BU    + SMEM_BU_SZ;
static constexpr int SMEM_RED_U = SMEM_RED_G + SMEM_RED_G_SZ;

static constexpr int SMEM_POOL_BYTES         = SMEM_RED_U + SMEM_RED_U_SZ;       // ~17.5 KB
static constexpr int SMEM_POOL_BYTES_ALIGNED = (SMEM_POOL_BYTES + 15) & ~15;

// =============================================================================
// Stage HIDDEN BF16 vector into smem using all block threads.
// =============================================================================
__device__ __forceinline__ void stage_vector_to_smem(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* dst_smem, int K, int tid)
{
    const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(src);
    __nv_bfloat162* dst2 = reinterpret_cast<__nv_bfloat162*>(dst_smem);
    int n = K >> 1;
    for (int i = tid; i < n; i += BLOCK_SIZE) dst2[i] = src2[i];
}

// =============================================================================
// K-gang cooperative dequant: load & dequant a [16 N, KGANG_K] block.
//
// Qwen3 FP4 layout: weight [N_out, K_in/2] uint8, row-major. K is contiguous.
// Scale: [N_out, K_in/16] fp8_e4m3fn, row-major.
//
// For a tile at (n_base, k_base) with n_span=16 and k_span=KGANG_K=128:
//   Each thread handles ONE 4-byte chunk = 8 FP4 elements = 8 K values for
//   ONE N-row. Thread decomposition:
//     tid in [0, 256):
//       n_row   = tid & 15          (0..15, N-row within the tile)
//       k_chunk = tid >> 4          (0..15, each covers 8 K-elements)
//   K-range per thread: [k_chunk*8 .. k_chunk*8 + 8).
//   FP4 byte offset:    k_chunk * 4 (since 2 nibbles/byte → 8 K = 4 bytes).
//
// Each K-chunk (8 K-elements) always lies within ONE FP4 scale block (16 K),
// so exactly one scale per thread.
//
// Output: b_dq_smem[n_row, k_chunk*8..k_chunk*8+8] as row-major BF16
//         (ld = KGANG_K). This is the shape required for the subsequent
//         wmma::load_matrix_sync on 16×16 sub-tiles at (n_base, k_base + w*16).
//
// HBM access pattern per warp (32 threads, 2 K-chunks × 16 N-rows):
//   Each warp's 32 threads cover 16 N-rows × 2 K-chunks. The 16 N-rows access
//   16 different cache lines (one per row, same 4 B offset). Across all 8
//   warps, threads 0-31 of warp 0 load [n=0..15, chunk=0..1], threads 0-31 of
//   warp 1 load [n=0..15, chunk=2..3], …
//
// The KEY WIN vs v7: each N-row ends up reading 16 × 4 B = 64 B consecutively
// (all 16 K-chunks for that row), which is half of a 128 B cache line. Paired
// with neighboring warps' loads on the SAME row, the L1 sees one 128-B line
// per row per gang, with ~64 B / 128 B = 50% utilization (vs 6% in v7).
// =============================================================================
__device__ __forceinline__ void kgang_dequant(
    const uint8_t* __restrict__ fp4_base,   // pointer to weight[0, 0]
    int n_base,                             // N starting row
    int k_base,                             // K starting column
    int ldb_fp4,                            // K / 2 in bytes
    const uint8_t* __restrict__ sf_base,    // pointer to scale[0, 0] as bytes
    int ldsf,                               // K / 16
    float global_sc,
    __nv_bfloat16* __restrict__ b_smem,     // [16 N, KGANG_K] row-major, ld = KGANG_K
    int tid)
{
    int n_row   = tid & 15;                 // 0..15
    int k_chunk = tid >> 4;                 // 0..15 (= KGANG_K / 8 - 1 = 15)

    int n_global = n_base + n_row;
    int k_start  = k_base + k_chunk * 8;    // 8 K-elements per chunk
    int k_byte   = k_start >> 1;            // packed FP4 byte offset (4 B chunks)

    // Fetch 4 packed bytes (8 nibbles) from fp4[n_global, k_byte..k_byte+4).
    const uint8_t* src = fp4_base + (size_t)n_global * ldb_fp4 + k_byte;
    uint32_t packed32 = *reinterpret_cast<const uint32_t*>(src);

    // Fetch scale for this chunk. k_start is always 8-aligned; scale block is
    // 16 K elements → block index = k_start / 16 = (k_base + k_chunk * 8) / 16.
    int k_block = k_start >> 4;            // scale block index
    uint8_t sf_byte = sf_base[(size_t)n_global * ldsf + k_block];
    __nv_fp8_e4m3 sf_fp8 = reinterpret_cast<__nv_fp8_e4m3&>(sf_byte);
    float sc = static_cast<float>(sf_fp8) * global_sc;

    // Decode 8 nibbles → 8 BF16. Write to b_smem[n_row, k_chunk*8..+8].
    // Output is row-major [16 N, KGANG_K] with ld = KGANG_K.
    int out_base = n_row * KGANG_K + k_chunk * 8;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t byte = (packed32 >> (j * 8)) & 0xFFu;
        float v_lo = FP4_LUT[byte & 0xFu]        * sc;
        float v_hi = FP4_LUT[(byte >> 4) & 0xFu] * sc;
        b_smem[out_base + j * 2 + 0] = __float2bfloat16(v_lo);
        b_smem[out_base + j * 2 + 1] = __float2bfloat16(v_hi);
    }
}

// =============================================================================
// Load A-tile: 16×16 BF16 tile with row 0 = x_smem[k0..k0+15], rest zero.
// Per-warp; lane 0..15 writes row 0, others zero.
// Called after clearing the tile memory.
// =============================================================================
__device__ __forceinline__ void load_a_tile_m1_padded(
    const __nv_bfloat16* __restrict__ x_smem, int k0,
    __nv_bfloat16* a_smem, int lane)
{
    __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(a_smem);
    const __nv_bfloat16 bf_zero = __float2bfloat16(0.0f);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane * 4 + i;   // 0..127 pairs = 256 bf16
        a2[idx] = __halves2bfloat162(bf_zero, bf_zero);
    }
    __syncwarp();
    if (lane < WMMA_K) {
        a_smem[lane] = x_smem[k0 + lane];
    }
}

// =============================================================================
// Expert GEMM1 (gate+up) — one block per (expert, N-tile) work item.
//
// Per block:
//   block-owned (e_slot, n_tile): compute 16 output elements [e_slot, n0..n0+16).
//   for k_gang = 0, 1, ..., HIDDEN/KGANG_K - 1 (i.e. HIDDEN/128 = 16 gangs):
//     k_base = k_gang * KGANG_K
//     cooperatively K-gang-dequant gate[n_base, k_base..+128] into smem_bg
//     cooperatively K-gang-dequant up  [n_base, k_base..+128] into smem_bu
//     __syncthreads()
//     // each warp w processes 16-K-wide sub-block: k_sub = k_base + w*WMMA_K
//     load A-tile from x_smem[k_sub..+16]
//     WMMA bg_frag from smem_bg[:, w*16..+16]
//     WMMA bu_frag from smem_bu[:, w*16..+16]
//     accumulate into c_g_frag[w], c_u_frag[w]
//   cross-warp reduce c_g, c_u into 16-element vectors
//   apply silu(g) * u and store
// =============================================================================
__device__ void kgang_expert_gemm1_stage(
    const __nv_bfloat16* __restrict__ hidden,
    const uint8_t* __restrict__ gate_fp4_all,
    const uint8_t* __restrict__ gate_sf_all,
    const float* __restrict__ gate_gs_all,
    const uint8_t* __restrict__ up_fp4_all,
    const uint8_t* __restrict__ up_sf_all,
    const float* __restrict__ up_gs_all,
    __nv_bfloat16* __restrict__ mlp_scratch,
    int bid, int num_blocks,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    __nv_bfloat16* bg_smem,
    __nv_bfloat16* bu_smem,
    float* red_g_smem,
    float* red_u_smem)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    __nv_bfloat16* a_smem = a_smem_pool + warp_id * (WMMA_M * WMMA_K);

    // Stage hidden into smem (all block threads).
    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();

    constexpr int TILES_PER_EXPERT = INTER_PER_EXP / WMMA_N;   // 48
    constexpr int TOTAL_TILES      = TOP_K * TILES_PER_EXPERT; // 384

    const int ldb_fp4 = HIDDEN / 2;         // 1024
    const int ldsf    = HIDDEN / FP4_BLOCK; // 128
    const size_t per_expert_fp4 = (size_t)INTER_PER_EXP * ldb_fp4;
    const size_t per_expert_sf  = (size_t)INTER_PER_EXP * ldsf;

    for (int tile = bid; tile < TOTAL_TILES; tile += num_blocks) {
        int e_slot  = tile / TILES_PER_EXPERT;
        int n_tile  = tile - e_slot * TILES_PER_EXPERT;
        int n_base  = n_tile * WMMA_N;

        float gate_gs = gate_gs_all[e_slot];
        float up_gs   = up_gs_all[e_slot];

        const uint8_t* gate_fp4 = gate_fp4_all + e_slot * per_expert_fp4;
        const uint8_t* gate_sf  = gate_sf_all  + e_slot * per_expert_sf;
        const uint8_t* up_fp4   = up_fp4_all   + e_slot * per_expert_fp4;
        const uint8_t* up_sf    = up_sf_all    + e_slot * per_expert_sf;

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cg_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cu_frag;
        wmma::fill_fragment(cg_frag, 0.0f);
        wmma::fill_fragment(cu_frag, 0.0f);

        // Outer loop over K-gangs.
        constexpr int NUM_GANGS = HIDDEN / KGANG_K;   // 16
        #pragma unroll 1
        for (int g = 0; g < NUM_GANGS; g++) {
            int k_base = g * KGANG_K;

            // K-gang cooperative dequant of gate and up.
            kgang_dequant(gate_fp4, n_base, k_base, ldb_fp4,
                          gate_sf, ldsf, gate_gs, bg_smem, tid);
            kgang_dequant(up_fp4, n_base, k_base, ldb_fp4,
                          up_sf, ldsf, up_gs, bu_smem, tid);
            __syncthreads();

            // Each warp handles a 16-K sub-block at offset w*WMMA_K.
            int k_sub = k_base + warp_id * WMMA_K;
            load_a_tile_m1_padded(x_smem, k_sub, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            // B-tile is at b_smem[:, warp_id*16 : warp_id*16 + 16].
            // Layout is row-major [16 N, KGANG_K]; wmma::matrix_b row_major
            // with ldb = KGANG_K reads a [WMMA_K, WMMA_N] tile starting at
            // b_smem + warp_id * WMMA_K. But wait: wmma::matrix_b row_major
            // expects [K, N] — our smem is [N, K]. We must use col_major.
            //
            // Effective GEMM: a_frag is [M=1(+15 zeros), K=16] row_major.
            //                 B needs to be [K=16, N=16] so output is [M, N].
            // Our b_smem[n_row, k_col] with k starting at warp_id*16 and
            // spanning 16 K-columns, 16 N-columns (= rows of our [N, K]
            // matrix). wmma col_major expects pointer to B[0,0] and ld =
            // column-stride = KGANG_K (since consecutive elements in the same
            // row are contiguous — column step is 1 row = KGANG_K elements).
            // So: col_major with ld = KGANG_K, pointer = b_smem + warp_id*WMMA_K.
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> bg_frag;
            wmma::load_matrix_sync(bg_frag,
                                   bg_smem + warp_id * WMMA_K,
                                   KGANG_K);
            wmma::mma_sync(cg_frag, a_frag, bg_frag, cg_frag);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> bu_frag;
            wmma::load_matrix_sync(bu_frag,
                                   bu_smem + warp_id * WMMA_K,
                                   KGANG_K);
            wmma::mma_sync(cu_frag, a_frag, bu_frag, cu_frag);

            __syncthreads();   // before next gang overwrites bg/bu smem
        }

        // Each warp's c_g_frag / c_u_frag holds row-0 result for the tile
        // (plus 15 zero rows from padding). wmma::store_matrix_sync needs a
        // 256-fp32 scratch per warp. 8 warps × 256 fp32 = 8 KB. Reuse bg_smem
        // and bu_smem (4 KB each, 8 KB combined) as scratch — safe since their
        // B-dequant contents are no longer needed after the final mma_sync and
        // we've __syncthreaded.
        float* c_scratch = reinterpret_cast<float*>(bg_smem);  // 8 KB total
        // Warp w uses c_scratch[w * 256 .. (w+1) * 256)
        float* c_slab = c_scratch + warp_id * (WMMA_M * WMMA_N);

        wmma::store_matrix_sync(c_slab, cg_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            red_g_smem[warp_id * WMMA_N + lane] = c_slab[lane];
        }
        __syncwarp();

        wmma::store_matrix_sync(c_slab, cu_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            red_u_smem[warp_id * WMMA_N + lane] = c_slab[lane];
        }
        __syncthreads();

        // Cross-warp reduce: sum 8 warps' 16-element vectors, then silu*up.
        // Use lanes 0..15 of warp 0.
        if (warp_id == 0 && lane < WMMA_N) {
            float g_sum = 0.0f;
            float u_sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                g_sum += red_g_smem[w * WMMA_N + lane];
                u_sum += red_u_smem[w * WMMA_N + lane];
            }
            float silu_g = g_sum / (1.0f + expf(-g_sum));
            mlp_scratch[e_slot * INTER_PER_EXP + n_base + lane] =
                __float2bfloat16(silu_g * u_sum);
        }
        __syncthreads();
    }
}

// =============================================================================
// Expert GEMM2 (down) — same K-gang pattern. K = INTER_PER_EXP = 768.
// 768 / KGANG_K = 768 / 128 = 6 K-gangs per tile.
// Tiles: TOP_K × (HIDDEN/WMMA_N) = 8 × 128 = 1024.
// =============================================================================
__device__ void kgang_expert_gemm2_stage(
    const __nv_bfloat16* __restrict__ mlp_scratch,
    const uint8_t* __restrict__ down_fp4_all,
    const uint8_t* __restrict__ down_sf_all,
    const float* __restrict__ down_gs_all,
    __nv_bfloat16* __restrict__ expert_out,
    int bid, int num_blocks,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    __nv_bfloat16* bg_smem,   // used as B-dq buffer
    float* red_g_smem)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    __nv_bfloat16* a_smem = a_smem_pool + warp_id * (WMMA_M * WMMA_K);

    constexpr int TILES_PER_EXPERT = HIDDEN / WMMA_N;         // 128
    constexpr int TOTAL_TILES      = TOP_K * TILES_PER_EXPERT; // 1024

    const int ldb_fp4 = INTER_PER_EXP / 2;         // 384
    const int ldsf    = INTER_PER_EXP / FP4_BLOCK; // 48
    const size_t per_expert_fp4 = (size_t)HIDDEN * ldb_fp4;
    const size_t per_expert_sf  = (size_t)HIDDEN * ldsf;

    constexpr int NUM_GANGS = INTER_PER_EXP / KGANG_K;   // 6

    for (int tile = bid; tile < TOTAL_TILES; tile += num_blocks) {
        int e_slot  = tile / TILES_PER_EXPERT;
        int n_tile  = tile - e_slot * TILES_PER_EXPERT;
        int n_base  = n_tile * WMMA_N;

        float gs = down_gs_all[e_slot];
        const uint8_t* fp4 = down_fp4_all + e_slot * per_expert_fp4;
        const uint8_t* sf  = down_sf_all  + e_slot * per_expert_sf;

        // For the first tile (or when e_slot changes), stage mlp_scratch[e_slot]
        // into x_smem. To keep the block simple, always re-stage: INTER=768 BF16
        // = 1.5 KB per stage, 1024 tiles / 188 SMs ≈ 5.4 tiles/SM, so ~6 stages
        // per SM across the whole phase. Negligible compared to weight traffic.
        const __nv_bfloat16* mlp_ptr_e = mlp_scratch + e_slot * INTER_PER_EXP;
        stage_vector_to_smem(mlp_ptr_e, x_smem, INTER_PER_EXP, tid);
        __syncthreads();

        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll 1
        for (int g = 0; g < NUM_GANGS; g++) {
            int k_base = g * KGANG_K;

            kgang_dequant(fp4, n_base, k_base, ldb_fp4,
                          sf, ldsf, gs, bg_smem, tid);
            __syncthreads();

            int k_sub = k_base + warp_id * WMMA_K;
            load_a_tile_m1_padded(x_smem, k_sub, a_smem, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag,
                                   bg_smem + warp_id * WMMA_K,
                                   KGANG_K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

            __syncthreads();  // before next gang
        }

        float* c_slab = reinterpret_cast<float*>(bg_smem) + warp_id * (WMMA_M * WMMA_N);
        wmma::store_matrix_sync(c_slab, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            red_g_smem[warp_id * WMMA_N + lane] = c_slab[lane];
        }
        __syncthreads();

        if (warp_id == 0 && lane < WMMA_N) {
            float s = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) {
                s += red_g_smem[w * WMMA_N + lane];
            }
            expert_out[e_slot * HIDDEN + n_base + lane] = __float2bfloat16(s);
        }
        __syncthreads();
    }
}

// =============================================================================
// Main cooperative kernel — 2 grid barriers between 3 phases, matching v7.
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_v7_1_phase0_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const uint8_t* __restrict__ gate_fp4_all,
    const uint8_t* __restrict__ gate_sf_all,
    const float* __restrict__ gate_gs_all,
    const uint8_t* __restrict__ up_fp4_all,
    const uint8_t* __restrict__ up_sf_all,
    const float* __restrict__ up_gs_all,
    const uint8_t* __restrict__ down_fp4_all,
    const uint8_t* __restrict__ down_sf_all,
    const float* __restrict__ down_gs_all,
    __nv_bfloat16* __restrict__ mlp_scratch,
    __nv_bfloat16* __restrict__ expert_out,
    int num_iters)
{
    auto grid = cg::this_grid();
    const int bid        = blockIdx.x;
    const int num_blocks = gridDim.x;

    extern __shared__ __align__(16) unsigned char smem_pool[];
    __nv_bfloat16* s_x      = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_X);
    __nv_bfloat16* s_a      = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_A);
    __nv_bfloat16* s_bg     = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_BG);
    __nv_bfloat16* s_bu     = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_BU);
    float*         s_red_g  = reinterpret_cast<float*>        (smem_pool + SMEM_RED_G);
    float*         s_red_u  = reinterpret_cast<float*>        (smem_pool + SMEM_RED_U);

    for (int it = 0; it < num_iters; it++) {
        kgang_expert_gemm1_stage(
            hidden,
            gate_fp4_all, gate_sf_all, gate_gs_all,
            up_fp4_all,   up_sf_all,   up_gs_all,
            mlp_scratch,
            bid, num_blocks,
            s_x, s_a, s_bg, s_bu, s_red_g, s_red_u);
        grid.sync();

        kgang_expert_gemm2_stage(
            mlp_scratch,
            down_fp4_all, down_sf_all, down_gs_all,
            expert_out,
            bid, num_blocks,
            s_x, s_a, s_bg, s_red_g);
        grid.sync();
    }
}

// =============================================================================
// Host launcher (C ABI)
// =============================================================================
extern "C" int v71p0_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int v71p0_hidden()     { return HIDDEN; }
extern "C" int v71p0_inter()      { return INTER_PER_EXP; }
extern "C" int v71p0_top_k()      { return TOP_K; }
extern "C" int v71p0_fp4_block()  { return FP4_BLOCK; }
extern "C" int v71p0_kgang_k()    { return KGANG_K; }
extern "C" int v71p0_smem_bytes() { return SMEM_POOL_BYTES_ALIGNED; }
extern "C" size_t v71p0_gate_up_fp4_bytes_per_exp() { return GATE_UP_FP4_BYTES; }
extern "C" size_t v71p0_gate_up_sf_bytes_per_exp()  { return GATE_UP_SF_BYTES; }
extern "C" size_t v71p0_down_fp4_bytes_per_exp()    { return DOWN_FP4_BYTES; }
extern "C" size_t v71p0_down_sf_bytes_per_exp()     { return DOWN_SF_BYTES; }

extern "C" cudaError_t v71p0_launch(
    const __nv_bfloat16* hidden,
    const uint8_t* gate_fp4_all, const uint8_t* gate_sf_all, const float* gate_gs_all,
    const uint8_t* up_fp4_all,   const uint8_t* up_sf_all,   const float* up_gs_all,
    const uint8_t* down_fp4_all, const uint8_t* down_sf_all, const float* down_gs_all,
    __nv_bfloat16* mlp_scratch,
    __nv_bfloat16* expert_out,
    int num_iters,
    cudaStream_t stream)
{
    int num_sms = v71p0_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);

    void* args[] = {
        (void*)&hidden,
        (void*)&gate_fp4_all, (void*)&gate_sf_all, (void*)&gate_gs_all,
        (void*)&up_fp4_all,   (void*)&up_sf_all,   (void*)&up_gs_all,
        (void*)&down_fp4_all, (void*)&down_sf_all, (void*)&down_gs_all,
        (void*)&mlp_scratch,  (void*)&expert_out,
        (void*)&num_iters,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_v7_1_phase0_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_v7_1_phase0_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_v7_1_phase0
