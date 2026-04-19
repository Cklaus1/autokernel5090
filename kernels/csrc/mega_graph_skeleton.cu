// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Cooperative Kernel — SKELETON for Gemma4 26B NVFP4 decode on SM120a.
//
// Goal:
//   Replace the full 30-layer model forward with ONE cooperative kernel launch,
//   bypassing the SM120 CUDA-graph-size replay bug (Discoveries #56-58) and
//   freeing the tok/s left on the table by piecewise / FULL_DECODE_ONLY.
//
// What this file is:
//   - A compilable skeleton (-arch=sm_120a, CUDA 12.8+) showing the scaffolding:
//     cooperative launch, per-layer loop, grid.sync() barriers, shared-memory
//     layout, and weight-pointer table plumbing.
//   - Placeholder `decoder_layer_step(...)` with no real attention/MoE bodies.
//     Hot paths are marked with `TODO(megagraph)` so a follow-up PR can slot in
//     real kernels.
//
// What this file is NOT:
//   - A working inference kernel. It does not compute correct logits.
//   - A full MoE dispatch. persistent_moe_dispatch.cu remains the MoE reference.
//
// See /home/cklaus/projects/autokernel/plans/mega_graph_cooperative_kernel.md
// for design decisions, feasibility math, and the go/no-go gate.
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs, 228 KB smem/SM, 64 K regs/SM.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace mega_graph {

// =============================================================================
// Launch config
// =============================================================================

// 1 block per SM; 256 threads/block; 1 resident block → 256 regs/thread max.
// __launch_bounds__ ensures nvcc budgets registers for this occupancy.
static constexpr int BLOCK_SIZE = 256;
static constexpr int BLOCKS_PER_SM = 1;

// Hidden dim and tile shape for Gemma4 26B (kept as constants; real integration
// would template on these).
static constexpr int HIDDEN = 4096;         // Gemma4 26B hidden size
static constexpr int D_HEAD = 128;          // GQA head dim
static constexpr int H_Q = 16;              // Q heads
static constexpr int H_KV = 8;              // KV heads (GQA)
static constexpr int TOP_K = 8;             // MoE top-k
static constexpr int NUM_EXPERTS = 128;     // Gemma4 26B-A4B

// Tokens staged per SM per step. C ≤ 16 keeps activation tile ≤ 128 KB of smem
// (C * HIDDEN * sizeof(bf16) = 16 * 4096 * 2 = 128 KB). See §2.4 of the plan.
static constexpr int MAX_TILE_TOKENS = 16;

// =============================================================================
// Shared-memory layout (static; sized for sm_120a 228 KB cap with margin)
// =============================================================================
//
// Budget audit:
//   activation tile    :  MAX_TILE_TOKENS * HIDDEN * 2  = 16 * 4096 * 2 = 131 072 B
//   router scores      :  MAX_TILE_TOKENS * NUM_EXPERTS * 4 = 16 * 128 * 4 = 8 192 B
//   topk scratch       :  MAX_TILE_TOKENS * TOP_K * 8        = 1 024 B
//   attention qk scratch (per-block):  8 192 B
//   moe expert meta    :  2 048 B
//   ----------------------------------------------------------
//   TOTAL             ≈  150 528 B  (< 228 KB cap, ≈ 78 KB headroom)
//
// All arrays declared __shared__ in the kernel body below.

// =============================================================================
// Per-layer weight pointer table
// =============================================================================
//
// Pre-resolved once at model load. Flat so we can index by layer_idx * STRIDE
// + slot from device code without pointer-chase. Each layer contributes:
//   0: input layernorm weight           (bf16 [HIDDEN])
//   1: q_proj weight (fp4)              + 2: q_proj scales (fp8)
//   3: k_proj weight (fp4)              + 4: k_proj scales (fp8)
//   5: v_proj weight (fp4)              + 6: v_proj scales (fp8)
//   7: o_proj weight (fp4)              + 8: o_proj scales (fp8)
//   9: post-attn layernorm              (bf16 [HIDDEN])
//  10: router weight                    (bf16 [HIDDEN, NUM_EXPERTS])
//  11: expert bank base (fp4)           ← per-expert offset math in kernel
//  12: expert scale bank base (fp8)
//
// Slots 11/12 are bases; per-expert offset is
//   base + expert_idx * (expert_weight_bytes_per_expert).
// A follow-up PR may split these into a per-expert pointer table for NUMA-style
// access patterns, but for now a base + stride suffices.
static constexpr int PTRS_PER_LAYER = 13;

// =============================================================================
// Placeholders (to be filled in by follow-up PRs)
// =============================================================================
//
// The real attention/MoE/GEMM work goes here. In the skeleton these are no-ops
// so we can measure cooperative-launch round-trip latency in isolation.

// --- Attention stage --------------------------------------------------------
// TODO(megagraph): implement GQA attention reading the per-layer KV cache.
//   - Use the FusenCache decode path (see fusencache_decode_attention.cu) as a
//     starting point; inline its inner loop body.
//   - Keep QK softmax scratch in `attn_qk_scratch` smem.
//   - Output: writes hidden += attn_out into the activation tile.
//
// Current behavior: touches the tile just enough that the compiler can't DCE
// the whole call, so we're actually measuring grid.sync() overhead and not
// a kernel that the optimizer collapsed to a no-op.
__device__ __forceinline__ void attention_stage_stub(
    __nv_bfloat16* __restrict__ tile,           // [MAX_TILE_TOKENS, HIDDEN] smem
    const void** weight_ptrs,
    int layer_idx,
    int tokens_in_tile,
    int thread_id)
{
    // Single DRAM-free fence: read-modify-write one element to prevent DCE.
    // This costs ~1 ns and lets us isolate barrier overhead in the microbench.
    if (thread_id == 0 && tokens_in_tile > 0) {
        // No-op visible side effect.
        tile[0] = __hadd(tile[0], __float2bfloat16(0.0f));
    }
}

// --- MoE stage --------------------------------------------------------------
// TODO(megagraph): implement router + top-k + shuffle + expert GEMMs + scatter.
//   - Reuse the 6 phases from persistent_moe_dispatch.cu (and fold in the
//     phase-6 atomicAdd fix — see §7 of the plan).
//   - The two expert GEMMs (gate+up, down) are the compute floor. A follow-up
//     PR may cp.async-prefetch expert weights while the router is running.
//
// Current behavior: no-op stub. Structure mirrors attention_stage_stub.
__device__ __forceinline__ void moe_stage_stub(
    __nv_bfloat16* __restrict__ tile,
    const void** weight_ptrs,
    int layer_idx,
    int tokens_in_tile,
    int thread_id)
{
    if (thread_id == 0 && tokens_in_tile > 0) {
        tile[0] = __hadd(tile[0], __float2bfloat16(0.0f));
    }
}

// --- Residual / norm stage --------------------------------------------------
// In Gemma4, residual is applied twice per layer (pre-attn and pre-mlp). These
// are register-only ops; no barrier required. Left as an explicit stub so the
// follow-up PR has a spot to hang the rmsnorm call.
__device__ __forceinline__ void residual_norm_stub(
    __nv_bfloat16* __restrict__ tile,
    const void** weight_ptrs,
    int layer_idx,
    int tokens_in_tile,
    int thread_id)
{
    // TODO(megagraph): fused rmsnorm. Reuse rms_norm_dynamic_fp4_quant.cu body.
}

// =============================================================================
// Per-layer step
// =============================================================================
//
// Called once per decoder layer from the main loop. `num_barriers_per_layer`
// is a compile-time constant so nvcc can unroll / avoid extra branches; the
// feasibility bench uses 3 (to sweep 1..4 barriers) while the production
// target is 2 (or 1 with producer/consumer SM partition).
template <int BARRIERS_PER_LAYER>
__device__ __forceinline__ void decoder_layer_step(
    cg::grid_group& grid,
    __nv_bfloat16* __restrict__ tile,
    const void** weight_ptrs,
    int layer_idx,
    int tokens_in_tile,
    int thread_id)
{
    residual_norm_stub(tile, weight_ptrs, layer_idx, tokens_in_tile, thread_id);
    attention_stage_stub(tile, weight_ptrs, layer_idx, tokens_in_tile, thread_id);

    if constexpr (BARRIERS_PER_LAYER >= 3) grid.sync();  // post-attn fence

    residual_norm_stub(tile, weight_ptrs, layer_idx, tokens_in_tile, thread_id);
    moe_stage_stub(tile, weight_ptrs, layer_idx, tokens_in_tile, thread_id);

    if constexpr (BARRIERS_PER_LAYER >= 2) grid.sync();  // post-MoE fence

    // BARRIERS_PER_LAYER == 1 : single barrier AFTER both stages (below).
    if constexpr (BARRIERS_PER_LAYER == 1) grid.sync();

    // BARRIERS_PER_LAYER == 4 : extra post-residual fence (mostly for testing).
    if constexpr (BARRIERS_PER_LAYER >= 4) grid.sync();
}

// =============================================================================
// Main kernel — one cooperative launch, runs full decode.
// =============================================================================
//
// Grid: one block per SM (188 on PRO 6000).
// Block: BLOCK_SIZE threads.
//
// layer_count is a runtime arg so the same kernel serves Gemma4 26B (L=30) and
// any other decoder; BARRIERS_PER_LAYER is templated for compile-time unroll.

template <int BARRIERS_PER_LAYER>
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_decode_kernel(
    // Activations (in/out)
    __nv_bfloat16* __restrict__ hidden,        // [M, HIDDEN]
    // Weight pointer table (flat: layer_idx * PTRS_PER_LAYER + slot)
    const void** __restrict__ weight_ptrs,
    // KV cache (paged) — unused in skeleton.
    uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    // Dims
    int num_tokens,
    int layer_count)
{
    auto grid    = cg::this_grid();
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_sms = gridDim.x;

    // ------------------------------------------------------------------
    // Shared memory (DYNAMIC — sm_120a supports up to 228 KB but the static
    // limit is only 48 KB). Caller sets smem_bytes at launch; layout is:
    //   [0 .. 131 072)           : activation tile (bf16 [MAX_TILE_TOKENS, HIDDEN])
    //   [131 072 .. 139 264)     : router scores (float [MAX_TILE_TOKENS, NUM_EXPERTS])
    //   [139 264 .. 139 776)     : topk scratch  (int32 [MAX_TILE_TOKENS, TOP_K])
    //   [139 776 .. 147 968)     : attention QK scratch (8 KB)
    //   [147 968 .. 150 016)     : MoE expert metadata (2 KB)
    // TOTAL ≈ 150 016 B (< 228 KB SM cap; ≈ 78 KB headroom).
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* tile          = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float*         router_scores = reinterpret_cast<float*>(
        smem_raw + MAX_TILE_TOKENS * HIDDEN * sizeof(__nv_bfloat16));
    int32_t*       topk_scratch  = reinterpret_cast<int32_t*>(
        reinterpret_cast<unsigned char*>(router_scores)
        + MAX_TILE_TOKENS * NUM_EXPERTS * sizeof(float));
    uint8_t*       attn_qk_scratch = reinterpret_cast<uint8_t*>(
        reinterpret_cast<unsigned char*>(topk_scratch)
        + MAX_TILE_TOKENS * TOP_K * sizeof(int32_t));
    uint8_t*       moe_meta      = attn_qk_scratch + 8192;

    // Silence "unused" warnings on skeleton-only paths.
    (void)router_scores; (void)topk_scratch;
    (void)attn_qk_scratch; (void)moe_meta;
    (void)kv_cache; (void)block_table; (void)seq_lens;

    // ------------------------------------------------------------------
    // Load activation tile into smem (this SM's slice of tokens).
    // ------------------------------------------------------------------
    const int tokens_per_sm = (num_tokens + num_sms - 1) / num_sms;
    const int tile_start    = sm * tokens_per_sm;
    const int tile_end      = min(tile_start + tokens_per_sm, num_tokens);
    const int tokens_in_tile = max(0, tile_end - tile_start);

    // Stage hidden -> tile smem. For the skeleton we stage only the first
    // MAX_TILE_TOKENS rows per SM; real kernel will loop-tile if > MAX.
    const int stage_rows = min(tokens_in_tile, MAX_TILE_TOKENS);
    for (int r = 0; r < stage_rows; r++) {
        const __nv_bfloat16* src = hidden + (tile_start + r) * HIDDEN;
        for (int d = tid; d < HIDDEN; d += BLOCK_SIZE) {
            tile[r * HIDDEN + d] = src[d];
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Main decoder loop — one iteration per layer.
    // ------------------------------------------------------------------
    for (int L = 0; L < layer_count; L++) {
        decoder_layer_step<BARRIERS_PER_LAYER>(
            grid, tile, weight_ptrs, L, stage_rows, tid);
    }

    // ------------------------------------------------------------------
    // Write tile back to HBM hidden.
    // ------------------------------------------------------------------
    __syncthreads();
    for (int r = 0; r < stage_rows; r++) {
        __nv_bfloat16* dst = hidden + (tile_start + r) * HIDDEN;
        for (int d = tid; d < HIDDEN; d += BLOCK_SIZE) {
            dst[d] = tile[r * HIDDEN + d];
        }
    }
}

// =============================================================================
// Microbench entry points — isolate grid.sync() round-trip on SM120.
// =============================================================================
//
// These two kernels have IDENTICAL work per iteration (essentially none) —
// the only difference is how many grid.sync() barriers they issue. The
// feasibility harness launches each under both cooperative launch AND CUDA
// graph (same kernel, many launches). The ratio tells us whether the
// cooperative approach is within 2× of graph-replay overhead.

// Cooperative variant: one launch, N "layers" × K barriers internally.
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
coop_barrier_bench_kernel(int layer_count, int barriers_per_layer,
                          int* __restrict__ side_effect)
{
    auto grid = cg::this_grid();
    int sm = blockIdx.x;
    int tid = threadIdx.x;

    for (int L = 0; L < layer_count; L++) {
        // Keep compiler honest.
        if (sm == 0 && tid == 0) atomicAdd(side_effect, 1);
        for (int b = 0; b < barriers_per_layer; b++) {
            grid.sync();
        }
    }
}

// Non-cooperative single-"layer" step, for comparison under CUDA graph capture.
__global__ void noop_step_kernel(int* __restrict__ side_effect)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(side_effect, 1);
}

// =============================================================================
// Host-side launchers
// =============================================================================

extern "C" int mega_graph_num_sms() {
    int device;
    cudaGetDevice(&device);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    return num_sms;
}

extern "C" int mega_graph_supports_cooperative() {
    int device;
    cudaGetDevice(&device);
    int val = 0;
    cudaDeviceGetAttribute(&val, cudaDevAttrCooperativeLaunch, device);
    return val;
}

// Launch the cooperative barrier microbench.
extern "C" cudaError_t mega_graph_launch_coop_bench(
    int layer_count, int barriers_per_layer, int* d_side_effect,
    cudaStream_t stream)
{
    int num_sms = mega_graph_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);

    void* args[] = {
        (void*)&layer_count,
        (void*)&barriers_per_layer,
        (void*)&d_side_effect,
    };
    return cudaLaunchCooperativeKernel(
        (void*)coop_barrier_bench_kernel,
        grid, block, args, /*smem=*/0, stream);
}

// Launch the ordinary "one no-op kernel per barrier" path (what CUDA graphs
// replay). Caller strings these together inside a graph capture.
extern "C" cudaError_t mega_graph_launch_noop_step(
    int* d_side_effect, cudaStream_t stream)
{
    dim3 grid(mega_graph_num_sms()), block(BLOCK_SIZE);
    noop_step_kernel<<<grid, block, 0, stream>>>(d_side_effect);
    return cudaGetLastError();
}

// Dynamic shared-memory bytes we request from the driver. Must match the
// layout above. 150 016 rounded up to 16 for alignment safety.
static constexpr int MEGA_GRAPH_SMEM_BYTES = 150016;

// Launch the full skeleton decode kernel (weight pointers can be null in the
// skeleton — the stubs don't dereference them).
extern "C" cudaError_t mega_graph_launch_decode(
    __nv_bfloat16* hidden,
    const void** weight_ptrs,
    uint8_t* kv_cache,
    const int32_t* block_table,
    const int32_t* seq_lens,
    int num_tokens,
    int layer_count,
    int barriers_per_layer,
    cudaStream_t stream)
{
    int num_sms = mega_graph_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);

    void* args[] = {
        (void*)&hidden,
        (void*)&weight_ptrs,
        (void*)&kv_cache,
        (void*)&block_table,
        (void*)&seq_lens,
        (void*)&num_tokens,
        (void*)&layer_count,
    };

    // Pick template instantiation by barrier count. Skeleton supports 1..4.
    void* kfn = nullptr;
    switch (barriers_per_layer) {
        case 1: kfn = (void*)mega_graph_decode_kernel<1>; break;
        case 2: kfn = (void*)mega_graph_decode_kernel<2>; break;
        case 3: kfn = (void*)mega_graph_decode_kernel<3>; break;
        case 4: kfn = (void*)mega_graph_decode_kernel<4>; break;
        default: return cudaErrorInvalidValue;
    }

    // Opt into >48 KB shared memory. Idempotent — safe to call every launch.
    cudaError_t e = cudaFuncSetAttribute(
        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,
        MEGA_GRAPH_SMEM_BYTES);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(kfn, grid, block, args,
                                       MEGA_GRAPH_SMEM_BYTES, stream);
}

}  // namespace mega_graph
