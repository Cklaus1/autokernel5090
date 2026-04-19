// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — PROTOTYPE (2 layers, M=1 decode).
//
// Converts the skeleton (mega_graph_skeleton.cu) into an actually-computing
// 2-layer transformer prototype to prove the cooperative-persistent pattern
// end-to-end on SM120a.
//
// Simplifications vs. full Gemma4 26B (intentional — this is a 2-layer probe):
//   - BF16 weights throughout (no FP4). Lets us skip FP4 dequant plumbing this
//     session; the pattern is trivially extensible to FP4 via the existing
//     rms_norm_dynamic_fp4_quant.cu helpers.
//   - Dense single-expert MoE (W_gate, W_up, W_down). Full router/top-k is
//     orthogonal to the barrier scaffolding and out of scope for this session.
//   - Standard MHA (num_heads=NUM_HEADS, head_dim=HEAD_DIM) with in-kernel
//     K/V that has been pre-stored into a dense [SEQ_LEN, num_heads,
//     head_dim] buffer by the caller. No paged KV cache; no FusenCache 4-bit
//     dequant. Again, additive — not load-bearing for the barrier pattern.
//   - M=1 (single token decode).
//
// Barrier topology (2 per layer, matches production target):
//   RMSNorm -> Attention -> grid.sync() (post-attn, hidden += attn_out)
//   RMSNorm -> MoE      -> grid.sync() (post-moe, hidden += moe_out)
//
// Cooperative launch: 1 block per SM. All blocks participate in each stage,
// cooperatively tiling the per-token hidden dim across SMs.
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace mega_graph_gemma4 {

// =============================================================================
// Config (compile-time for cleaner codegen)
// =============================================================================
static constexpr int BLOCK_SIZE = 256;
static constexpr int BLOCKS_PER_SM = 1;

// Prototype dims. Kept modest so the whole working set fits; the barrier
// scaffolding is what we're probing, not peak FLOPS.
static constexpr int HIDDEN     = 512;   // hidden size
static constexpr int NUM_HEADS  = 4;     // attention heads
static constexpr int HEAD_DIM   = 128;   // = HIDDEN / NUM_HEADS
static constexpr int INTER_DIM  = 1024;  // MoE intermediate (2x hidden)
static constexpr int MAX_SEQ    = 128;   // KV cache length cap (tile of prior tokens)

static_assert(NUM_HEADS * HEAD_DIM == HIDDEN, "HIDDEN must equal NUM_HEADS*HEAD_DIM");

// RMSNorm epsilon (matches Gemma4 reference)
static constexpr float RMS_EPS = 1e-6f;

// =============================================================================
// Device helpers
// =============================================================================

// Block-wide sum (via shared memory). `tid` is threadIdx.x, BLOCK_SIZE threads.
__device__ __forceinline__ float block_reduce_sum(float v, float* smem) {
    int tid = threadIdx.x;
    // warp-level
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

// Block-wide max (for softmax stability).
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

// Grid-wide sum across SMs. Each SM contributes a partial (computed in-block),
// writes to gmem slot, grid.sync(), then all read + sum.
__device__ __forceinline__ float grid_reduce_sum(
    float partial, float* __restrict__ scratch, int num_sms,
    cg::grid_group& grid, int tid, int sm)
{
    if (tid == 0) scratch[sm] = partial;
    grid.sync();
    // All blocks read back and sum. Cheap: num_sms is small (188).
    float tot = 0.f;
    for (int i = tid; i < num_sms; i += BLOCK_SIZE) tot += scratch[i];
    // block reduce
    __shared__ float red_smem[32];
    float s = 0.f;
    for (int off = 16; off > 0; off >>= 1) tot += __shfl_xor_sync(0xffffffff, tot, off);
    int wid = tid >> 5;
    int lid = tid & 31;
    if (lid == 0) red_smem[wid] = tot;
    __syncthreads();
    if (wid == 0) {
        int nwarps = BLOCK_SIZE / 32;
        s = (lid < nwarps) ? red_smem[lid] : 0.f;
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
        if (lid == 0) red_smem[0] = s;
    }
    __syncthreads();
    return red_smem[0];
}

// =============================================================================
// Stages — all operate on the shared hidden vector (residual stream) in HBM.
// Tile-by-SM strategy: each SM owns `dims_per_sm = ceil(HIDDEN/num_sms)` dims
// of the output. For output projections that are HIDDEN-wide this is natural.
// For the attention matmul we have all SMs cooperate because the K/V reduction
// crosses head boundaries.
// =============================================================================

// RMSNorm applied to `src` into `dst`, both [HIDDEN]. Uses grid reduction so
// all SMs see a consistent norm factor. Scaled by `weight`.
__device__ void rmsnorm_stage(
    const __nv_bfloat16* __restrict__ src,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ dst,
    float* __restrict__ scratch_f,    // [num_sms] grid reduction scratch
    cg::grid_group& grid,
    int sm, int num_sms, int tid)
{
    // Per-SM partial sum of squares over this SM's stripe of the hidden dim.
    int dims_per_sm = (HIDDEN + num_sms - 1) / num_sms;
    int d0 = sm * dims_per_sm;
    int d1 = min(d0 + dims_per_sm, HIDDEN);

    float local_ss = 0.f;
    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float x = __bfloat162float(src[d]);
        local_ss += x * x;
    }
    // Block reduce to block-wide sum, then grid reduce.
    __shared__ float red_smem[32];
    float block_ss = block_reduce_sum(local_ss, red_smem);
    // Only one partial per SM; tid==0 writes.
    float grid_ss = grid_reduce_sum(block_ss, scratch_f, num_sms, grid, tid, sm);
    float inv_rms = rsqrtf(grid_ss / (float)HIDDEN + RMS_EPS);

    // Apply normalization with per-dim weight.
    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float x = __bfloat162float(src[d]) * inv_rms;
        float w = __bfloat162float(weight[d]);
        dst[d] = __float2bfloat16(x * w);
    }
    // No barrier needed here — next stage reads this SM's own stripe first,
    // then uses grid.sync() when going cross-SM.
}

// Attention Q/K/V projections — produce Q for this token, K/V are already
// written into seq buffer. For M=1 decode, Q is a [HIDDEN] vector; we compute
// it, plus append the new token's K/V to the existing sequence buffer at
// position `seq_len - 1` (caller sets this up).
//
// Q = x @ Wq, K_new = x @ Wk, V_new = x @ Wv, all [HIDDEN].
// Each SM computes its stripe of output.
//
// After this, grid.sync is needed because the attention softmax needs all K/V.
__device__ void qkv_proj_stage(
    const __nv_bfloat16* __restrict__ x,        // [HIDDEN] normed hidden
    const __nv_bfloat16* __restrict__ Wq,       // [HIDDEN, HIDDEN]
    const __nv_bfloat16* __restrict__ Wk,       // [HIDDEN, HIDDEN]
    const __nv_bfloat16* __restrict__ Wv,       // [HIDDEN, HIDDEN]
    __nv_bfloat16* __restrict__ q_out,          // [HIDDEN]
    __nv_bfloat16* __restrict__ k_out,          // [HIDDEN]  (slot in K cache for pos)
    __nv_bfloat16* __restrict__ v_out,          // [HIDDEN]  (slot in V cache for pos)
    int sm, int num_sms, int tid)
{
    int dims_per_sm = (HIDDEN + num_sms - 1) / num_sms;
    int d0 = sm * dims_per_sm;
    int d1 = min(d0 + dims_per_sm, HIDDEN);

    // Each thread handles a subset of output dims in [d0, d1).
    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float qv = 0.f, kv = 0.f, vv = 0.f;
        // Weights are row-major [HIDDEN_in, HIDDEN_out] for a nn.Linear
        // semantic of y = x @ W  (so W has shape [in, out]).
        for (int i = 0; i < HIDDEN; i++) {
            float xi = __bfloat162float(x[i]);
            qv += xi * __bfloat162float(Wq[i * HIDDEN + d]);
            kv += xi * __bfloat162float(Wk[i * HIDDEN + d]);
            vv += xi * __bfloat162float(Wv[i * HIDDEN + d]);
        }
        q_out[d] = __float2bfloat16(qv);
        k_out[d] = __float2bfloat16(kv);
        v_out[d] = __float2bfloat16(vv);
    }
}

// Multi-head attention over stored K/V buffer for a single new token.
// K_cache/V_cache shape: [MAX_SEQ, NUM_HEADS, HEAD_DIM] (dense BF16).
// Q shape: [NUM_HEADS, HEAD_DIM] (this token).
// Output: [NUM_HEADS, HEAD_DIM], contiguous flattened to [HIDDEN].
//
// Simplified: no causal mask (caller only writes up to seq_len). No soft cap.
// Head partition: each head is assigned to a subset of SMs that cooperatively
// compute its attention output. For simplicity here we let EACH SM process
// ALL heads (since HIDDEN is small); the reduction over keys is inside-SM.
// This costs a bit of redundancy but gives a clean pattern that's easy to
// verify and keeps intra-kernel data movement minimal.
//
// We then have each SM write its assigned head-stripe of the output — that's
// the "partition" that matters for correctness.
__device__ void attention_core_stage(
    const __nv_bfloat16* __restrict__ q,       // [HIDDEN]
    const __nv_bfloat16* __restrict__ K,       // [MAX_SEQ, HIDDEN]
    const __nv_bfloat16* __restrict__ V,       // [MAX_SEQ, HIDDEN]
    __nv_bfloat16* __restrict__ attn_out,       // [HIDDEN]
    int seq_len,
    float* __restrict__ /*scratch_f unused*/,
    int sm, int num_sms, int tid)
{
    // Each SM takes a subset of heads.
    int heads_per_sm = (NUM_HEADS + num_sms - 1) / num_sms;
    int h0 = sm * heads_per_sm;
    int h1 = min(h0 + heads_per_sm, NUM_HEADS);

    constexpr float inv_sqrt_d = 1.0f / 11.3137084989848f;  // 1/sqrt(128) for HEAD_DIM=128

    for (int h = h0; h < h1; h++) {
        // Load Q for this head into registers (tid handles HEAD_DIM/BLOCK_SIZE dims).
        // For HEAD_DIM=128, BLOCK_SIZE=256 => most threads idle for Q; that's fine,
        // this is a prototype. A real kernel uses warps per head.
        // Shared scratch for scores and softmax.
        extern __shared__ __align__(16) unsigned char smem_raw[];
        float* scores = reinterpret_cast<float*>(smem_raw);      // [MAX_SEQ]
        __nv_bfloat16* q_h = reinterpret_cast<__nv_bfloat16*>(scores + MAX_SEQ);  // [HEAD_DIM]

        if (tid < HEAD_DIM) q_h[tid] = q[h * HEAD_DIM + tid];
        __syncthreads();

        // Compute scores[t] = dot(Q_h, K[t, h, :]) * inv_sqrt_d
        for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
            float s = 0.f;
            const __nv_bfloat16* kt = K + t * HIDDEN + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                s += __bfloat162float(q_h[d]) * __bfloat162float(kt[d]);
            }
            scores[t] = s * inv_sqrt_d;
        }
        __syncthreads();

        // Softmax (block-wide max, block-wide sum).
        __shared__ float red_smem[32];
        float local_max = -INFINITY;
        for (int t = tid; t < seq_len; t += BLOCK_SIZE) local_max = fmaxf(local_max, scores[t]);
        float smax = block_reduce_max(local_max, red_smem);

        float local_sum = 0.f;
        for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
            float e = __expf(scores[t] - smax);
            scores[t] = e;
            local_sum += e;
        }
        float ssum = block_reduce_sum(local_sum, red_smem);
        float inv_sum = 1.0f / (ssum + 1e-20f);

        // Output = sum_t softmax(scores)_t * V[t, h, :]
        // Each thread handles a subset of HEAD_DIM output dims.
        for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
            float acc = 0.f;
            for (int t = 0; t < seq_len; t++) {
                acc += scores[t] * __bfloat162float(V[t * HIDDEN + h * HEAD_DIM + d]);
            }
            attn_out[h * HEAD_DIM + d] = __float2bfloat16(acc * inv_sum);
        }
        __syncthreads();
    }
}

// Attention output projection: attn_out = attn @ Wo, add into residual.
//   hidden_out[d] = hidden_in[d] + sum_i(attn_in[i] * Wo[i, d])
__device__ void attn_oproj_residual_stage(
    const __nv_bfloat16* __restrict__ attn_in,   // [HIDDEN]
    const __nv_bfloat16* __restrict__ Wo,        // [HIDDEN, HIDDEN]
    __nv_bfloat16* __restrict__ residual,        // [HIDDEN] in/out
    int sm, int num_sms, int tid)
{
    int dims_per_sm = (HIDDEN + num_sms - 1) / num_sms;
    int d0 = sm * dims_per_sm;
    int d1 = min(d0 + dims_per_sm, HIDDEN);
    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float acc = 0.f;
        for (int i = 0; i < HIDDEN; i++) {
            acc += __bfloat162float(attn_in[i]) * __bfloat162float(Wo[i * HIDDEN + d]);
        }
        residual[d] = __hadd(residual[d], __float2bfloat16(acc));
    }
}

// Simplified single-expert MoE (SwiGLU MLP):
//   gate = x @ W_gate    (BF16 [HIDDEN, INTER_DIM])
//   up   = x @ W_up      (BF16 [HIDDEN, INTER_DIM])
//   h    = silu(gate) * up       (BF16 [INTER_DIM])
//   out  = h @ W_down    (BF16 [INTER_DIM, HIDDEN])
//   residual += out
//
// We stage the gate/up intermediate in gmem (mlp_scratch) because INTER_DIM
// would be tight in smem at full Gemma4 sizes. For INTER_DIM=1024 it's only
// 2 KB so smem is fine, but we keep the gmem path for extensibility.
__device__ void moe_mlp_residual_stage(
    const __nv_bfloat16* __restrict__ x,         // [HIDDEN]
    const __nv_bfloat16* __restrict__ W_gate,    // [HIDDEN, INTER_DIM]
    const __nv_bfloat16* __restrict__ W_up,      // [HIDDEN, INTER_DIM]
    const __nv_bfloat16* __restrict__ W_down,    // [INTER_DIM, HIDDEN]
    __nv_bfloat16* __restrict__ mlp_scratch,      // [INTER_DIM] bf16
    __nv_bfloat16* __restrict__ residual,         // [HIDDEN] in/out
    cg::grid_group& grid,
    int sm, int num_sms, int tid)
{
    // Phase A: compute mlp_scratch[j] = silu(gate_j) * up_j for each j.
    // Partition INTER_DIM across SMs.
    int inter_per_sm = (INTER_DIM + num_sms - 1) / num_sms;
    int j0 = sm * inter_per_sm;
    int j1 = min(j0 + inter_per_sm, INTER_DIM);
    for (int j = j0 + tid; j < j1; j += BLOCK_SIZE) {
        float g = 0.f, u = 0.f;
        for (int i = 0; i < HIDDEN; i++) {
            float xi = __bfloat162float(x[i]);
            g += xi * __bfloat162float(W_gate[i * INTER_DIM + j]);
            u += xi * __bfloat162float(W_up[i * INTER_DIM + j]);
        }
        float silu_g = g / (1.0f + __expf(-g));
        mlp_scratch[j] = __float2bfloat16(silu_g * u);
    }

    // Need to synchronize: all SMs must see the full mlp_scratch before
    // the down-projection. This is the INNER fence — NOT one of the two
    // "per-layer" barriers the design doc counts. It's internal to the MoE
    // stage. (A producer/consumer partition could avoid it but is out of
    // scope.)
    grid.sync();

    // Phase B: residual += mlp_scratch @ W_down
    int dims_per_sm = (HIDDEN + num_sms - 1) / num_sms;
    int d0 = sm * dims_per_sm;
    int d1 = min(d0 + dims_per_sm, HIDDEN);
    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float acc = 0.f;
        for (int i = 0; i < INTER_DIM; i++) {
            acc += __bfloat162float(mlp_scratch[i]) * __bfloat162float(W_down[i * HIDDEN + d]);
        }
        residual[d] = __hadd(residual[d], __float2bfloat16(acc));
    }
}

// =============================================================================
// Per-layer weight bundle
// =============================================================================
struct LayerWeights {
    const __nv_bfloat16* input_norm;     // [HIDDEN]
    const __nv_bfloat16* Wq;              // [HIDDEN, HIDDEN]
    const __nv_bfloat16* Wk;              // [HIDDEN, HIDDEN]
    const __nv_bfloat16* Wv;              // [HIDDEN, HIDDEN]
    const __nv_bfloat16* Wo;              // [HIDDEN, HIDDEN]
    const __nv_bfloat16* post_attn_norm; // [HIDDEN]
    const __nv_bfloat16* W_gate;          // [HIDDEN, INTER_DIM]
    const __nv_bfloat16* W_up;            // [HIDDEN, INTER_DIM]
    const __nv_bfloat16* W_down;          // [INTER_DIM, HIDDEN]
};

// =============================================================================
// Main kernel: 2-layer mega-graph decode, 2 barriers per layer.
// =============================================================================
//
// Workspace (allocated by caller, device-resident):
//   hidden         [HIDDEN]         — residual stream, mutated in place.
//   normed         [HIDDEN]         — rmsnorm output scratch (reused).
//   q, k_new,v_new [HIDDEN]         — Q and the current token's K,V.
//   attn_out       [HIDDEN]         — attention output before o_proj.
//   mlp_scratch    [INTER_DIM]      — silu(gate)*up intermediate.
//   reduce_scratch [num_sms]        — grid-reduce intermediate (RMSNorm).
//   K_cache, V_cache [MAX_SEQ,HIDDEN] — KV cache (caller seeds positions 0..seq_len-2).
//
// Layer index `Li` writes its new token's K/V at position seq_len-1, then
// attention reads positions 0..seq_len-1. We don't re-write KV across layers
// because in a real transformer each layer has its own KV cache — the caller
// provides arrays sized [L, MAX_SEQ, HIDDEN] each.
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_2layer_kernel(
    __nv_bfloat16* __restrict__ hidden,       // [HIDDEN] in/out
    const __nv_bfloat16* __restrict__ layer0_norm_in,
    const __nv_bfloat16* __restrict__ layer0_Wq,
    const __nv_bfloat16* __restrict__ layer0_Wk,
    const __nv_bfloat16* __restrict__ layer0_Wv,
    const __nv_bfloat16* __restrict__ layer0_Wo,
    const __nv_bfloat16* __restrict__ layer0_norm_post,
    const __nv_bfloat16* __restrict__ layer0_Wgate,
    const __nv_bfloat16* __restrict__ layer0_Wup,
    const __nv_bfloat16* __restrict__ layer0_Wdown,
    const __nv_bfloat16* __restrict__ layer1_norm_in,
    const __nv_bfloat16* __restrict__ layer1_Wq,
    const __nv_bfloat16* __restrict__ layer1_Wk,
    const __nv_bfloat16* __restrict__ layer1_Wv,
    const __nv_bfloat16* __restrict__ layer1_Wo,
    const __nv_bfloat16* __restrict__ layer1_norm_post,
    const __nv_bfloat16* __restrict__ layer1_Wgate,
    const __nv_bfloat16* __restrict__ layer1_Wup,
    const __nv_bfloat16* __restrict__ layer1_Wdown,
    __nv_bfloat16* __restrict__ K0_cache,     // [MAX_SEQ, HIDDEN]
    __nv_bfloat16* __restrict__ V0_cache,
    __nv_bfloat16* __restrict__ K1_cache,
    __nv_bfloat16* __restrict__ V1_cache,
    __nv_bfloat16* __restrict__ normed,       // [HIDDEN] scratch
    __nv_bfloat16* __restrict__ q_scratch,    // [HIDDEN]
    __nv_bfloat16* __restrict__ attn_out,      // [HIDDEN]
    __nv_bfloat16* __restrict__ mlp_scratch,   // [INTER_DIM]
    float* __restrict__ reduce_scratch,        // [num_sms]
    int seq_len)                              // current seq len (>=1)
{
    auto grid = cg::this_grid();
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_sms = gridDim.x;

    // ===== LAYER 0 =====================================================
    // RMSNorm(hidden) -> normed, using grid-reduce for ss sum.
    rmsnorm_stage(hidden, layer0_norm_in, normed, reduce_scratch, grid,
                  sm, num_sms, tid);
    grid.sync();  // norm result must be visible to all SMs before qkv_proj

    // Q/K/V projections. K/V are written directly into cache at position seq_len-1.
    __nv_bfloat16* k_slot0 = K0_cache + (seq_len - 1) * HIDDEN;
    __nv_bfloat16* v_slot0 = V0_cache + (seq_len - 1) * HIDDEN;
    qkv_proj_stage(normed, layer0_Wq, layer0_Wk, layer0_Wv,
                   q_scratch, k_slot0, v_slot0,
                   sm, num_sms, tid);
    grid.sync();  // KV must be visible to all SMs before attention reads

    // Attention core (reads K0/V0 for positions 0..seq_len-1).
    attention_core_stage(q_scratch, K0_cache, V0_cache, attn_out, seq_len,
                         reduce_scratch, sm, num_sms, tid);
    grid.sync();  // attn_out must be visible before o_proj uses it

    // O projection + residual add into `hidden`.
    attn_oproj_residual_stage(attn_out, layer0_Wo, hidden, sm, num_sms, tid);
    grid.sync();  // BARRIER A (post-attention residual); counted as 1 of 2/layer.

    // RMSNorm(hidden) -> normed for the MLP branch.
    rmsnorm_stage(hidden, layer0_norm_post, normed, reduce_scratch, grid,
                  sm, num_sms, tid);
    grid.sync();  // normed visibility

    // MoE MLP (swiGLU), residual add.
    moe_mlp_residual_stage(normed, layer0_Wgate, layer0_Wup, layer0_Wdown,
                           mlp_scratch, hidden, grid, sm, num_sms, tid);
    grid.sync();  // BARRIER B (post-MoE residual); counted as 2 of 2/layer.

    // ===== LAYER 1 =====================================================
    rmsnorm_stage(hidden, layer1_norm_in, normed, reduce_scratch, grid,
                  sm, num_sms, tid);
    grid.sync();

    __nv_bfloat16* k_slot1 = K1_cache + (seq_len - 1) * HIDDEN;
    __nv_bfloat16* v_slot1 = V1_cache + (seq_len - 1) * HIDDEN;
    qkv_proj_stage(normed, layer1_Wq, layer1_Wk, layer1_Wv,
                   q_scratch, k_slot1, v_slot1,
                   sm, num_sms, tid);
    grid.sync();

    attention_core_stage(q_scratch, K1_cache, V1_cache, attn_out, seq_len,
                         reduce_scratch, sm, num_sms, tid);
    grid.sync();

    attn_oproj_residual_stage(attn_out, layer1_Wo, hidden, sm, num_sms, tid);
    grid.sync();  // BARRIER A layer 1

    rmsnorm_stage(hidden, layer1_norm_post, normed, reduce_scratch, grid,
                  sm, num_sms, tid);
    grid.sync();

    moe_mlp_residual_stage(normed, layer1_Wgate, layer1_Wup, layer1_Wdown,
                           mlp_scratch, hidden, grid, sm, num_sms, tid);
    grid.sync();  // BARRIER B layer 1
}

// =============================================================================
// Host launchers (extern "C" for ctypes + torch wrappers)
// =============================================================================

extern "C" int mgg4_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_hidden() { return HIDDEN; }
extern "C" int mgg4_inter_dim() { return INTER_DIM; }
extern "C" int mgg4_num_heads() { return NUM_HEADS; }
extern "C" int mgg4_head_dim() { return HEAD_DIM; }
extern "C" int mgg4_max_seq() { return MAX_SEQ; }

// Dynamic shared memory: MAX_SEQ * sizeof(float) for scores + HEAD_DIM * bf16.
static constexpr int MGG4_SMEM_BYTES =
    MAX_SEQ * sizeof(float) + HEAD_DIM * sizeof(__nv_bfloat16) + 64;

extern "C" cudaError_t mgg4_launch_2layer(
    __nv_bfloat16* hidden,
    const __nv_bfloat16* l0_norm_in,
    const __nv_bfloat16* l0_Wq, const __nv_bfloat16* l0_Wk,
    const __nv_bfloat16* l0_Wv, const __nv_bfloat16* l0_Wo,
    const __nv_bfloat16* l0_norm_post,
    const __nv_bfloat16* l0_Wgate, const __nv_bfloat16* l0_Wup,
    const __nv_bfloat16* l0_Wdown,
    const __nv_bfloat16* l1_norm_in,
    const __nv_bfloat16* l1_Wq, const __nv_bfloat16* l1_Wk,
    const __nv_bfloat16* l1_Wv, const __nv_bfloat16* l1_Wo,
    const __nv_bfloat16* l1_norm_post,
    const __nv_bfloat16* l1_Wgate, const __nv_bfloat16* l1_Wup,
    const __nv_bfloat16* l1_Wdown,
    __nv_bfloat16* K0, __nv_bfloat16* V0,
    __nv_bfloat16* K1, __nv_bfloat16* V1,
    __nv_bfloat16* normed, __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out, __nv_bfloat16* mlp_scratch,
    float* reduce_scratch,
    int seq_len,
    cudaStream_t stream)
{
    int num_sms = mgg4_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden,
        &l0_norm_in, &l0_Wq, &l0_Wk, &l0_Wv, &l0_Wo,
        &l0_norm_post, &l0_Wgate, &l0_Wup, &l0_Wdown,
        &l1_norm_in, &l1_Wq, &l1_Wk, &l1_Wv, &l1_Wo,
        &l1_norm_post, &l1_Wgate, &l1_Wup, &l1_Wdown,
        &K0, &V0, &K1, &V1,
        &normed, &q_scratch, &attn_out, &mlp_scratch,
        &reduce_scratch,
        &seq_len,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_2layer_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        MGG4_SMEM_BYTES);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_2layer_kernel,
        grid, block, args, MGG4_SMEM_BYTES, stream);
}

}  // namespace mega_graph_gemma4
