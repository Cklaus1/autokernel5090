// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph v7 Phase 0 — real-MoE FP4 microbench (decision gate).
//
// Purpose: Before committing to v7 cooperative kernel, run a STANDALONE
// microbench of ONE real Qwen3 MoE layer with:
//   * Real FP4 expert weights loaded from /root/models/Qwen3-30B-A3B-NVFP4/
//   * top_k=8 routing (fixed selection for isolation; real router in Phase 1)
//   * Inline FP4->BF16 dequant (Option A per plans/mega_graph_v4b_fp4_spec.md)
//   * WMMA 16x16x16 BF16 matmul (sm_120a; no WGMMA/FP8/native-FP4 MMA)
//   * Cooperative kernel with grid.sync() barriers — mirrors v7 cost model
//
// Gate:
//   dram__throughput / peak_HBM >= 0.50 -> PASS (proceed to Phase 1)
//   0.30 <= ratio < 0.50 -> MARGINAL (stop, flag v7.1 redesign)
//   ratio < 0.30 -> v6 pathology repeating (STOP, do not proceed)
//
// Qwen3-30B-A3B-NVFP4 shapes (from config.json + safetensors):
//   hidden_size = 2048
//   moe_intermediate_size = 768 (per-expert INTER_DIM)
//   num_experts = 128
//   num_experts_per_tok = 8
//   Per expert per projection:
//     gate_proj.weight: uint8 [768, 1024]  (N=768 out, K=2048 in, K/2=1024 packed)
//     gate_proj.weight_scale: fp8_e4m3fn [768, 128]  (K/16=128 blocks)
//     gate_proj.weight_scale_2: fp32 scalar
//     up_proj   : same shape as gate_proj
//     down_proj.weight: uint8 [2048, 384]  (N=2048 out, K=768 in, K/2=384 packed)
//     down_proj.weight_scale: fp8_e4m3fn [2048, 48]  (K/16=48 blocks)
//     down_proj.weight_scale_2: fp32 scalar
//
// Layout note (IMPORTANT — differs from v6):
//   Qwen3 FP4 weights on disk are [N_out, K_in/2] row-major (N = output-dim).
//   Scale tensor is     [N_out, K_in/16] row-major.
//   For wmma::matrix_b[row_major] with ldb = N_out, we view the weight as
//   a [K_in, N_out] matrix where pointer (k0, n0) -> W_nibble_at[n0, k0]:
//   the dequant path has to transpose indexing implicitly — we dequant a
//   [16 (K), 16 (N)] tile by gathering nibbles from 16 N rows of the FP4
//   tensor at K columns k0..k0+15.
//
// Per-tile HBM traffic (per expert GEMM1 gate+up = 2 projections):
//   FP4 bytes:   2 * 768 * 1024 = 1,572,864 B (1.5 MB) per expert
//   FP8 scales:  2 * 768 * 128  = 196,608 B (192 KB) per expert
//   Per-expert gate+up total: ~1.73 MB
//   down: 2048 * 384 = 786,432 B + 2048 * 48 = 98,304 B = 884,736 B (864 KB)
//   Per-expert FULL MoE weight traffic: 1.73 + 0.87 = 2.60 MB
//   8 active experts: 20.8 MB
//   At 896 GB/s (50% of 1792 peak): 23.2 µs ideal
//
// Cooperative kernel structure (mirrors v7):
//   Phase A: stage hidden (BF16) into each SM's smem, compute router (faked
//            via pre-supplied topk_ids[0..7]); this stage is negligible.
//   grid.sync() [1/3]
//   Phase B: Expert GEMM1 (gate+up, FP4->BF16 dequant -> WMMA)
//            Work = (expert e in 8 active) x (N-tile in 768/16=48) = 384 tiles.
//            Distributed across 188 SMs -> ~2 tiles/SM avg.
//            Per tile: K-loop 2048/16 = 128 K-steps. Each step dequants a
//            [16, 16] B-tile from FP4 and mma_sync.
//   grid.sync() [2/3]
//   Phase C: Expert GEMM2 (down, FP4->BF16 dequant -> WMMA). silu fused.
//            Work = 8 experts x HIDDEN/16 = 8*128 = 1024 tiles.
//            ~5 tiles/SM. K-loop = 768/16 = 48 K-steps.
//   grid.sync() [3/3]
//   (Combine happens in eager PyTorch in the driver; we care about kernel
//    time only.)
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

namespace mega_graph_v7_phase0 {

// =============================================================================
// Config (Qwen3-30B-A3B shapes)
// =============================================================================
static constexpr int BLOCK_SIZE      = 256;
static constexpr int BLOCKS_PER_SM   = 1;
static constexpr int WARP_SIZE       = 32;
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 8

static constexpr int HIDDEN        = 2048;
static constexpr int INTER_PER_EXP = 768;
static constexpr int NUM_EXPERTS   = 128;
static constexpr int TOP_K         = 8;
static constexpr int FP4_BLOCK     = 16;   // elements per scale block

// WMMA 16x16x16 BF16 tile shape
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

static_assert(HIDDEN        % WMMA_K == 0, "K must divide WMMA_K for gate/up");
static_assert(INTER_PER_EXP % WMMA_N == 0, "INTER_PER_EXP must divide WMMA_N");
static_assert(HIDDEN        % WMMA_N == 0, "HIDDEN must divide WMMA_N for down");
static_assert(INTER_PER_EXP % WMMA_K == 0, "INTER_PER_EXP must divide WMMA_K for down");

// Per-expert weight sizes
// gate/up: FP4 [768, 1024] = 786432 B, SF [768, 128] = 98304 B
// down   : FP4 [2048, 384] = 786432 B, SF [2048, 48] = 98304 B
static constexpr size_t GATE_UP_FP4_BYTES = (size_t)INTER_PER_EXP * (HIDDEN / 2);
static constexpr size_t GATE_UP_SF_BYTES  = (size_t)INTER_PER_EXP * (HIDDEN / FP4_BLOCK);
static constexpr size_t DOWN_FP4_BYTES    = (size_t)HIDDEN * (INTER_PER_EXP / 2);
static constexpr size_t DOWN_SF_BYTES     = (size_t)HIDDEN * (INTER_PER_EXP / FP4_BLOCK);

// =============================================================================
// FP4-E2M1 16-entry LUT (constant memory)
// =============================================================================
__device__ __constant__ float FP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// =============================================================================
// Shared memory layout (per-SM, 188 SMs total)
// =============================================================================
// SMEM_X   : hidden activation staging (HIDDEN BF16 = 4 KB) OR
//            intermediate_per_expert staging (768 BF16 = 1.5 KB)
static constexpr int SMEM_X_SZ   = HIDDEN * 2;                               // 4096
// SMEM_A   : per-warp 16x16 BF16 A-tile   (8 * 512 = 4 KB)
static constexpr int SMEM_A_SZ   = WARPS_PER_BLOCK * WMMA_M * WMMA_K * 2;    // 4096
// SMEM_C   : per-warp 16x16 FP32 C-scratch (8 * 1024 = 8 KB)
static constexpr int SMEM_C_SZ   = WARPS_PER_BLOCK * WMMA_M * WMMA_N * 4;    // 8192
// SMEM_BDQ : per-warp 16x16 BF16 dequant B-tile (8 * 512 = 4 KB); two bufs for gate+up
static constexpr int SMEM_BDQ_SZ = 2 * WARPS_PER_BLOCK * WMMA_K * WMMA_N * 2; // 8192

static constexpr int SMEM_X   = 0;
static constexpr int SMEM_A   = SMEM_X + SMEM_X_SZ;
static constexpr int SMEM_C   = SMEM_A + SMEM_A_SZ;
static constexpr int SMEM_BDQ_RAW = SMEM_C + SMEM_C_SZ;
static constexpr int SMEM_BDQ = (SMEM_BDQ_RAW + 15) & ~15;

static constexpr int SMEM_POOL_BYTES         = SMEM_BDQ + SMEM_BDQ_SZ;   // ~24 KB
static constexpr int SMEM_POOL_BYTES_ALIGNED = (SMEM_POOL_BYTES + 15) & ~15;

// =============================================================================
// Stage a K-length BF16 vector into smem using all block threads.
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

// Load A-tile [1, 16] -> padded [16, 16] BF16 tile (only row 0 real, rest zero).
__device__ __forceinline__ void load_a_tile_m1_padded(
    const __nv_bfloat16* __restrict__ x_smem, int k0,
    __nv_bfloat16* a_smem, int lane)
{
    // Zero-fill the 16x16 A-tile (256 bf16 elems / 32 lanes = 8 per lane).
    __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(a_smem);
    const __nv_bfloat16 bf_zero = __float2bfloat16(0.0f);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = lane * 4 + i;    // 0..127 covers 128 pairs = 256 elems
        a2[idx] = __halves2bfloat162(bf_zero, bf_zero);
    }
    __syncwarp();
    // Write row 0 (16 elements) from x_smem[k0..k0+15].
    if (lane < WMMA_K) {
        a_smem[lane] = x_smem[k0 + lane];
    }
}

// =============================================================================
// Dequantize a 16x16 BF16 B-tile from FP4 with Qwen3 layout.
//
// Qwen3 layout: weight is [N_out, K/2] uint8, scale is [N_out, K/16] fp8.
// For wmma::matrix_b row_major with ldb = N (so effectively the matrix
// used in the MMA is [K, N] = A @ B), element (k_row, n_col) of the matmul
// B corresponds to weight[n_col, k_row] in the on-disk layout. So to
// dequant a 16x16 tile at (k_row_base, n_col_base), we read from
// fp4[n_col_base + n, (k_row_base + k)/2] and scale[n_col_base + n, (k_row_base + k)/16].
//
// Tile mapping (per lane):
//   We have 32 lanes, producing 256 BF16 outputs = 16 k-rows x 16 n-cols.
//   Lane layout: lane l covers 8 elements in the tile.
//     lane l  ->  k = l & 15 (k-row within tile), n-slot = l >> 4 (0=cols 0..7, 1=cols 8..15)
//   Each lane reads: 1 FP4 byte (holds cols k_col/2 low, k_col/2 high? — no,
//   depends on packing).
//
// Simpler mapping:
//   Lane l in [0, 32).
//     row_idx   = l >> 1        (0..15) => this is the N-column within the tile.
//     half      = l & 1         => determines which 8-K-row slab
//     k_start   = half * 8      (0 or 8)
//   So lane (row_idx, half=0) outputs tile[0..7, row_idx], and
//       lane (row_idx, half=1) outputs tile[8..15, row_idx].
//
// Inputs:
//   fp4_ptr   : pointer into FP4 tensor; we'll read 4 packed bytes =
//               8 FP4 nibbles from fp4[n_col = row_idx, (k_start)/2 .. +4 bytes].
//   ldb_fp4   : row stride in bytes of the FP4 tensor = K / 2
//   sf_ptr    : pointer into scale tensor [N, K/16]; one scale per 16 k-elements.
//               Since half=0 or 1 both fall within the same 16-element block
//               (K_tile is exactly 16), we need ONE scale per lane per row.
//   ldsf      : row stride in elements of SF tensor = K / 16
//   global_sc : fp32 global scale
//   b_dq_smem : per-warp output buffer [16 (k), 16 (n)] BF16 (256 elems) row-major, lda = 16
//
// Output layout in b_dq_smem: row-major [K=16, N=16]. Element b_dq_smem[k*16+n].
// This is the layout required by wmma::matrix_b row_major with ldb=16.
// =============================================================================
__device__ __forceinline__ void dequant_fp4_tile_qwen3(
    const uint8_t* __restrict__ fp4_base,   // pointer to weight[0, 0]
    int n_col_base,                         // base N-column in the tile
    int k_row_base,                         // base K-row in the tile
    int ldb_fp4,                            // K/2
    const uint8_t* __restrict__ sf_base,    // pointer to scale[0, 0] as bytes
    int ldsf,                               // K/16
    float global_sc,
    __nv_bfloat16* b_dq_smem,               // [16 k, 16 n] row-major, ld = 16
    int lane)
{
    // lane decomposition:
    int row_idx = lane >> 1;     // 0..15, N-column index within tile
    int half    = lane & 1;      // 0 => k_rows 0..7, 1 => k_rows 8..15
    int k_start = half * 8;

    // Load per-lane scale: one scale per (N-row, K-block). Since WMMA_K=16
    // matches FP4_BLOCK=16, the whole tile's K-range [k_row_base .. k_row_base+15]
    // lies in ONE scale block at k_row_base / 16.
    int n = n_col_base + row_idx;
    int k_block = k_row_base / FP4_BLOCK;
    uint8_t sf_byte = sf_base[n * ldsf + k_block];
    __nv_fp8_e4m3 sf_fp8 = reinterpret_cast<__nv_fp8_e4m3&>(sf_byte);
    float sc = static_cast<float>(sf_fp8) * global_sc;

    // Load 4 packed bytes = 8 FP4 nibbles from fp4[n, (k_row_base + k_start)/2 .. +4].
    int fp4_col_byte = (k_row_base + k_start) >> 1;
    const uint8_t* src = fp4_base + (size_t)n * ldb_fp4 + fp4_col_byte;
    // Vector 32-bit load (aligned to 4B when k_start is 0 or 8 and k_row_base is multiple of 16).
    uint32_t packed32 = *reinterpret_cast<const uint32_t*>(src);

    // Decode 8 nibbles -> 8 fp32 (via LUT in constant memory).
    float f[8];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint32_t byte = (packed32 >> (j * 8)) & 0xFFu;
        f[j * 2 + 0] = FP4_LUT[byte & 0xFu] * sc;
        f[j * 2 + 1] = FP4_LUT[(byte >> 4) & 0xFu] * sc;
    }

    // Write 8 BF16 values to b_dq_smem at [k_start..k_start+7, row_idx].
    // Row-major indexing: b_dq_smem[k * 16 + n_in_tile].
    // n_in_tile = row_idx (0..15).
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        int k_tile = k_start + k;      // 0..15
        b_dq_smem[k_tile * 16 + row_idx] = __float2bfloat16(f[k]);
    }
}

// =============================================================================
// Expert GEMM1: gate + up, FP4 -> BF16 dequant -> WMMA.
// Output: mlp_scratch[e_slot, 0..INTER_PER_EXP) = silu(x @ Wg) * (x @ Wu)
//   for each active expert. e_slot in [0, TOP_K).
//
// Weights laid out contiguously across all active experts:
//   gate_fp4[e] at offset e * (INTER_PER_EXP * HIDDEN/2) bytes
//   gate_sf[e]  at offset e * (INTER_PER_EXP * HIDDEN/16) bytes
//   Same layout for up. (gate and up are in separate contiguous arrays.)
// =============================================================================
__device__ void expert_gemm1_stage(
    const __nv_bfloat16* __restrict__ hidden,     // [HIDDEN]
    const uint8_t* __restrict__ gate_fp4_all,     // [TOP_K, INTER_PER_EXP, HIDDEN/2]
    const uint8_t* __restrict__ gate_sf_all,      // [TOP_K, INTER_PER_EXP, HIDDEN/16]
    const float* __restrict__ gate_gs_all,        // [TOP_K]
    const uint8_t* __restrict__ up_fp4_all,
    const uint8_t* __restrict__ up_sf_all,
    const float* __restrict__ up_gs_all,
    __nv_bfloat16* __restrict__ mlp_scratch,      // [TOP_K, INTER_PER_EXP]
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float* c_scratch_pool,
    __nv_bfloat16* b_dq_smem_pool)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem    = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr     = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);
    __nv_bfloat16* b_dq_g    = b_dq_smem_pool + (2 * warp_id + 0) * (WMMA_K * WMMA_N);
    __nv_bfloat16* b_dq_u    = b_dq_smem_pool + (2 * warp_id + 1) * (WMMA_K * WMMA_N);

    // Stage hidden into smem (shared across all warps in this SM).
    stage_vector_to_smem(hidden, x_smem, HIDDEN, tid);
    __syncthreads();

    // Work list: (expert e_slot in 0..TOP_K) x (n_tile in 0..INTER_PER_EXP/16).
    constexpr int TILES_PER_EXPERT = INTER_PER_EXP / WMMA_N;   // 48
    constexpr int TOTAL_TILES      = TOP_K * TILES_PER_EXPERT; // 384

    const int ldb_fp4 = HIDDEN / 2;          // 1024
    const int ldsf    = HIDDEN / FP4_BLOCK;  // 128
    const size_t per_expert_fp4 = (size_t)INTER_PER_EXP * ldb_fp4;
    const size_t per_expert_sf  = (size_t)INTER_PER_EXP * ldsf;

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int e_slot  = tile / TILES_PER_EXPERT;
        int n_tile  = tile - e_slot * TILES_PER_EXPERT;
        int n0_base = n_tile * WMMA_N;   // 0..767

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

        for (int k0 = 0; k0 < HIDDEN; k0 += WMMA_K) {
            load_a_tile_m1_padded(x_smem, k0, a_smem, lane);
            __syncwarp();
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            // Dequant gate and up tiles.
            dequant_fp4_tile_qwen3(gate_fp4, n0_base, k0, ldb_fp4,
                                   gate_sf, ldsf, gate_gs, b_dq_g, lane);
            dequant_fp4_tile_qwen3(up_fp4, n0_base, k0, ldb_fp4,
                                   up_sf, ldsf, up_gs, b_dq_u, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> bg_frag;
            wmma::load_matrix_sync(bg_frag, b_dq_g, WMMA_N);
            wmma::mma_sync(cg_frag, a_frag, bg_frag, cg_frag);

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> bu_frag;
            wmma::load_matrix_sync(bu_frag, b_dq_u, WMMA_N);
            wmma::mma_sync(cu_frag, a_frag, bu_frag, cu_frag);
        }

        // Store tile row 0 (actual output) -> mlp_scratch[e_slot, n0_base..+16)
        wmma::store_matrix_sync(c_scr, cg_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        float g = (lane < WMMA_N) ? c_scr[lane] : 0.f;
        __syncwarp();
        wmma::store_matrix_sync(c_scr, cu_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        float u = (lane < WMMA_N) ? c_scr[lane] : 0.f;
        if (lane < WMMA_N) {
            float silu_g = g / (1.0f + expf(-g));
            mlp_scratch[e_slot * INTER_PER_EXP + n0_base + lane] =
                __float2bfloat16(silu_g * u);
        }
        __syncwarp();
    }
}

// =============================================================================
// Expert GEMM2: down, FP4 -> BF16 dequant -> WMMA.
// Output: expert_out[e_slot, 0..HIDDEN) = mlp_scratch[e_slot, :] @ W_down[e_slot, :, :]
//
// down weight: FP4 [HIDDEN=2048, INTER_PER_EXP/2=384], scale [HIDDEN, INTER_PER_EXP/16=48]
// So N-axis of GEMM is HIDDEN (2048), K-axis is INTER_PER_EXP (768).
// =============================================================================
__device__ void expert_gemm2_stage(
    const __nv_bfloat16* __restrict__ mlp_scratch,  // [TOP_K, INTER_PER_EXP]
    const uint8_t* __restrict__ down_fp4_all,       // [TOP_K, HIDDEN, INTER_PER_EXP/2]
    const uint8_t* __restrict__ down_sf_all,        // [TOP_K, HIDDEN, INTER_PER_EXP/16]
    const float* __restrict__ down_gs_all,          // [TOP_K]
    __nv_bfloat16* __restrict__ expert_out,         // [TOP_K, HIDDEN]
    int sm, int num_sms,
    __nv_bfloat16* x_smem,
    __nv_bfloat16* a_smem_pool,
    float* c_scratch_pool,
    __nv_bfloat16* b_dq_smem_pool)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int total_warps    = num_sms * WARPS_PER_BLOCK;
    const int global_warp_id = sm * WARPS_PER_BLOCK + warp_id;

    __nv_bfloat16* a_smem    = a_smem_pool    + warp_id * (WMMA_M * WMMA_K);
    float*         c_scr     = c_scratch_pool + warp_id * (WMMA_M * WMMA_N);
    __nv_bfloat16* b_dq      = b_dq_smem_pool + (2 * warp_id + 0) * (WMMA_K * WMMA_N);

    // One SM can only stage ONE expert's mlp_scratch into smem at a time. To
    // keep the microbench simple, each SM binds to a (e_slot, n_tile) pair
    // and re-stages the corresponding mlp_scratch slice on demand. Given 8
    // experts x 128 n-tiles = 1024 tiles and 188 SMs, expert re-staging per
    // warp is cheap (done once per tile via the smem broadcast).

    // Work list: (e_slot) x (n_tile in HIDDEN/16). Stage mlp_scratch[e_slot]
    // into smem at the FIRST tile of that expert encountered per SM; carry
    // across tiles if e_slot is the same.
    constexpr int TILES_PER_EXPERT = HIDDEN / WMMA_N;        // 128
    constexpr int TOTAL_TILES      = TOP_K * TILES_PER_EXPERT; // 1024

    const int ldb_fp4 = INTER_PER_EXP / 2;         // 384
    const int ldsf    = INTER_PER_EXP / FP4_BLOCK; // 48
    const size_t per_expert_fp4 = (size_t)HIDDEN * ldb_fp4;
    const size_t per_expert_sf  = (size_t)HIDDEN * ldsf;

    int last_e_slot = -1;

    for (int tile = global_warp_id; tile < TOTAL_TILES; tile += total_warps) {
        int e_slot  = tile / TILES_PER_EXPERT;
        int n_tile  = tile - e_slot * TILES_PER_EXPERT;
        int n0_base = n_tile * WMMA_N;   // 0..2032

        // Re-stage mlp_scratch[e_slot] if needed (all warps on the SM
        // need the same slice; gate on SM-level via __syncthreads).
        // We use a simple strategy: every warp re-stages its own expert
        // on SM-level when e_slot changes on any warp.
        // For correctness simplicity: re-stage cooperatively for each
        // distinct e_slot per warp-block iteration.
        // Actually simpler: stage_vector_to_smem runs on all threads and
        // only triggers when tid==0's warp has a different e_slot — but
        // that would require cross-warp coordination. Easiest: every SM
        // only binds to ONE expert's tiles (i.e., warps within an SM
        // share e_slot). Distribute by: sm % TOP_K gives expert, warps
        // within SM stride over tiles for that expert.
        //
        // See reassignment below using a different work partitioning.
        (void)last_e_slot;

        float gs = down_gs_all[e_slot];
        const uint8_t* fp4 = down_fp4_all + e_slot * per_expert_fp4;
        const uint8_t* sf  = down_sf_all  + e_slot * per_expert_sf;

        // We avoid the smem-staging complication by reading mlp_scratch
        // directly for the M=1 case. A-tile row 0 = mlp_scratch[e_slot, k0..k0+15].
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        const __nv_bfloat16* mlp_ptr_e = mlp_scratch + e_slot * INTER_PER_EXP;

        for (int k0 = 0; k0 < INTER_PER_EXP; k0 += WMMA_K) {
            // Load A-tile: row 0 from mlp_ptr_e[k0..k0+15], pad with zeros.
            __nv_bfloat162* a2 = reinterpret_cast<__nv_bfloat162*>(a_smem);
            const __nv_bfloat16 bf_zero = __float2bfloat16(0.0f);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int idx = lane * 4 + i;
                a2[idx] = __halves2bfloat162(bf_zero, bf_zero);
            }
            __syncwarp();
            if (lane < WMMA_K) {
                a_smem[lane] = mlp_ptr_e[k0 + lane];
            }
            __syncwarp();

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, a_smem, 16);

            dequant_fp4_tile_qwen3(fp4, n0_base, k0, ldb_fp4,
                                   sf, ldsf, gs, b_dq, lane);
            __syncwarp();

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> b_frag;
            wmma::load_matrix_sync(b_frag, b_dq, WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        wmma::store_matrix_sync(c_scr, c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        if (lane < WMMA_N) {
            expert_out[e_slot * HIDDEN + n0_base + lane] = __float2bfloat16(c_scr[lane]);
        }
        __syncwarp();
    }
}

// =============================================================================
// Main kernel — 2 barriers between 3 phases.
//
// Phase A: (implicit in host — staging hidden and providing topk_ids)
//   Host provides: hidden [HIDDEN] bf16, packed expert-sorted FP4 weights.
// Phase B: expert GEMM1 (gate+up -> silu*up -> mlp_scratch)
//   grid.sync()
// Phase C: expert GEMM2 (down -> expert_out)
//   grid.sync() (for barrier-cost parity with v7)
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_v7_phase0_kernel(
    const __nv_bfloat16* __restrict__ hidden,        // [HIDDEN]
    const uint8_t* __restrict__ gate_fp4_all,
    const uint8_t* __restrict__ gate_sf_all,
    const float* __restrict__ gate_gs_all,
    const uint8_t* __restrict__ up_fp4_all,
    const uint8_t* __restrict__ up_sf_all,
    const float* __restrict__ up_gs_all,
    const uint8_t* __restrict__ down_fp4_all,
    const uint8_t* __restrict__ down_sf_all,
    const float* __restrict__ down_gs_all,
    __nv_bfloat16* __restrict__ mlp_scratch,         // [TOP_K, INTER_PER_EXP]
    __nv_bfloat16* __restrict__ expert_out,          // [TOP_K, HIDDEN]
    int num_iters)  // iterate the MoE block multiple times inside one kernel
                    // launch (for steady-state BW measurement without grid
                    // launch overhead).
{
    auto grid = cg::this_grid();
    const int sm      = blockIdx.x;
    const int tid     = threadIdx.x;
    const int num_sms = gridDim.x;

    extern __shared__ __align__(16) unsigned char smem_pool[];
    __nv_bfloat16* pool_x   = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_X);
    __nv_bfloat16* pool_a   = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_A);
    float*         pool_c   = reinterpret_cast<float*>        (smem_pool + SMEM_C);
    __nv_bfloat16* pool_bdq = reinterpret_cast<__nv_bfloat16*>(smem_pool + SMEM_BDQ);

    (void)tid;

    for (int it = 0; it < num_iters; it++) {
        expert_gemm1_stage(
            hidden,
            gate_fp4_all, gate_sf_all, gate_gs_all,
            up_fp4_all,   up_sf_all,   up_gs_all,
            mlp_scratch,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_bdq);
        grid.sync();

        expert_gemm2_stage(
            mlp_scratch,
            down_fp4_all, down_sf_all, down_gs_all,
            expert_out,
            sm, num_sms,
            pool_x, pool_a, pool_c, pool_bdq);
        grid.sync();
    }
}

// =============================================================================
// Host launcher (C ABI)
// =============================================================================
extern "C" int v7p0_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int v7p0_hidden()       { return HIDDEN; }
extern "C" int v7p0_inter()        { return INTER_PER_EXP; }
extern "C" int v7p0_top_k()        { return TOP_K; }
extern "C" int v7p0_fp4_block()    { return FP4_BLOCK; }
extern "C" int v7p0_smem_bytes()   { return SMEM_POOL_BYTES_ALIGNED; }
extern "C" size_t v7p0_gate_up_fp4_bytes_per_exp() { return GATE_UP_FP4_BYTES; }
extern "C" size_t v7p0_gate_up_sf_bytes_per_exp()  { return GATE_UP_SF_BYTES; }
extern "C" size_t v7p0_down_fp4_bytes_per_exp()    { return DOWN_FP4_BYTES; }
extern "C" size_t v7p0_down_sf_bytes_per_exp()     { return DOWN_SF_BYTES; }

extern "C" cudaError_t v7p0_launch(
    const __nv_bfloat16* hidden,
    const uint8_t* gate_fp4_all, const uint8_t* gate_sf_all, const float* gate_gs_all,
    const uint8_t* up_fp4_all,   const uint8_t* up_sf_all,   const float* up_gs_all,
    const uint8_t* down_fp4_all, const uint8_t* down_sf_all, const float* down_gs_all,
    __nv_bfloat16* mlp_scratch,
    __nv_bfloat16* expert_out,
    int num_iters,
    cudaStream_t stream)
{
    int num_sms = v7p0_num_sms();
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
        (void*)mega_graph_v7_phase0_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        SMEM_POOL_BYTES_ALIGNED);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_v7_phase0_kernel,
        grid, block, args, SMEM_POOL_BYTES_ALIGNED, stream);
}

}  // namespace mega_graph_v7_phase0
