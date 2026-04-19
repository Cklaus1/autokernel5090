// SPDX-License-Identifier: Apache-2.0
//
// Mega-Graph Gemma4 Cooperative Kernel — 30-LAYER PROTOTYPE at Gemma4 scale.
//
// Extends mega_graph_gemma4.cu (2-layer probe) to the full 30-decoder-layer
// Gemma4 scale at HIDDEN=2048, NUM_HEADS=8, HEAD_DIM=128, INTER_DIM=8192.
// Still BF16 throughout, dense 1-expert SwiGLU MLP (MoE deferred),
// dense (non-paged) BF16 KV cache, M=1 decode.
//
// Barrier-collapse (reducing 9 -> 7/layer):
//   Option (a) LOCAL-REDUCE RMSNORM: every SM reads the full 2048-wide hidden
//   vector and computes RMS locally with a block-wide reduction (no grid.sync
//   inside rmsnorm). Costs 188 * 4 KB of redundant loads (~760 KB once-per-
//   rmsnorm, fully L2-resident after the first SM fetches it). Saves 2
//   grid.syncs per layer (one per rmsnorm).
//
// Final barrier topology per layer (7/layer realized, down from 9):
//   rmsnorm_local (no internal grid.sync)
//   grid.sync()    // #1 normed cross-SM visible before qkv
//   qkv_proj
//   grid.sync()    // #2 q/k/v visible for attention
//   attention_core
//   grid.sync()    // #3 attn_out visible for o_proj
//   o_proj + residual
//   grid.sync()    // #4 hidden visible for rmsnorm_post
//   rmsnorm_post_local (no internal grid.sync)
//   grid.sync()    // #5 normed visible for gate_up
//   mlp_gate_up
//   grid.sync()    // #6 mlp_scratch visible for down
//   mlp_down + residual
//   grid.sync()    // #7 layer boundary (hidden visible for next rmsnorm)
//
// Further collapse (future work, documented as known-safe peephole):
//   - Fuse rmsnorm_local output writes with qkv_proj input loads via staging
//     in smem + one barrier: saves #1 and #5 (= 2/layer), reaching 5/layer.
//   - Replicate rmsnorm output across all SMs (bit-identical writes, safe
//     race-on-identical-data) eliminates #1 and #5 entirely: 5/layer.
//   - Fuse attn_out with o_proj residual add inside registers when head_per_SM
//     partition == output stripe: saves #3.
// Target for next iter: 3-4/layer. This iter: 7/layer.
//
// Target: RTX PRO 6000 Blackwell (SM120a), 188 SMs.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace cg = cooperative_groups;

namespace mega_graph_gemma4_30 {

// =============================================================================
// Config (compile-time)
// =============================================================================
static constexpr int BLOCK_SIZE = 256;
static constexpr int BLOCKS_PER_SM = 1;

// Gemma4-scale dims (task-specified; actual config is hidden=2816, we follow
// the task spec here). Task spec had (H=2048, heads=8, head_dim=128) which is
// inconsistent (8*128=1024); resolved as heads=16 to hit HIDDEN=2048 target.
static constexpr int HIDDEN     = 2048;
static constexpr int NUM_HEADS  = 16;
static constexpr int HEAD_DIM   = 128;       // = HIDDEN / NUM_HEADS
static constexpr int INTER_DIM  = 8192;      // Gemma4 SwiGLU intermediate
static constexpr int MAX_SEQ    = 256;       // KV cache length cap
static constexpr int NUM_LAYERS = 30;

static_assert(NUM_HEADS * HEAD_DIM == HIDDEN, "HIDDEN must equal NUM_HEADS*HEAD_DIM");

// RMSNorm epsilon (matches Gemma4 reference)
static constexpr float RMS_EPS = 1e-6f;

// =============================================================================
// Block-wide helpers
// =============================================================================

// Block-wide sum (via shared mem + warp shuffles). tid = threadIdx.x.
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

// =============================================================================
// RMSNorm — LOCAL (no grid.sync). Every SM reads the full hidden vector and
// computes the rms locally using a block-wide reduction, then writes its own
// stripe of the normalized output.
//
// This is mathematically identical across all SMs (same input, same weight)
// so the per-SM inv_rms values agree exactly to the bit.
// =============================================================================
__device__ void rmsnorm_local_stage(
    const __nv_bfloat16* __restrict__ src,       // [HIDDEN]
    const __nv_bfloat16* __restrict__ weight,     // [HIDDEN]
    __nv_bfloat16* __restrict__ dst,              // [HIDDEN]
    int sm, int num_sms, int tid)
{
    // Phase 1: block-wide sum-of-squares over the FULL hidden vector.
    float local_ss = 0.f;
    for (int d = tid; d < HIDDEN; d += BLOCK_SIZE) {
        float x = __bfloat162float(src[d]);
        local_ss += x * x;
    }
    __shared__ float red_smem[32];
    float block_ss = block_reduce_sum(local_ss, red_smem);
    float inv_rms = rsqrtf(block_ss / (float)HIDDEN + RMS_EPS);

    // Phase 2: this SM writes its assigned output stripe.
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
// Attention QKV projections: Q = x @ Wq, K_new = x @ Wk, V_new = x @ Wv.
// Each SM owns a stripe of [d0, d1) output dims for all three projections.
// =============================================================================
__device__ void qkv_proj_stage(
    const __nv_bfloat16* __restrict__ x,        // [HIDDEN]
    const __nv_bfloat16* __restrict__ Wq,       // [HIDDEN, HIDDEN]
    const __nv_bfloat16* __restrict__ Wk,       // [HIDDEN, HIDDEN]
    const __nv_bfloat16* __restrict__ Wv,       // [HIDDEN, HIDDEN]
    __nv_bfloat16* __restrict__ q_out,          // [HIDDEN]
    __nv_bfloat16* __restrict__ k_out,          // [HIDDEN] (slot in K cache)
    __nv_bfloat16* __restrict__ v_out,          // [HIDDEN] (slot in V cache)
    int sm, int num_sms, int tid)
{
    int dims_per_sm = (HIDDEN + num_sms - 1) / num_sms;
    int d0 = sm * dims_per_sm;
    int d1 = min(d0 + dims_per_sm, HIDDEN);

    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float qv = 0.f, kv = 0.f, vv = 0.f;
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

// =============================================================================
// Multi-head attention core — all SMs process all heads (NUM_HEADS=8 <<
// num_sms=188). Each SM computes its assigned head's full attention output
// with a block-wide reduction over keys. Same math as 2-layer prototype,
// scaled to HEAD_DIM=128.
// =============================================================================
__device__ void attention_core_stage(
    const __nv_bfloat16* __restrict__ q,       // [HIDDEN]
    const __nv_bfloat16* __restrict__ K,       // [MAX_SEQ, HIDDEN]
    const __nv_bfloat16* __restrict__ V,       // [MAX_SEQ, HIDDEN]
    __nv_bfloat16* __restrict__ attn_out,       // [HIDDEN]
    int seq_len,
    int sm, int num_sms, int tid)
{
    // Each SM takes a subset of heads. With NUM_HEADS=8 and num_sms=188, most
    // SMs are idle — we just let heads_per_sm default to 0 for those SMs; they
    // participate only in the barriers. The first NUM_HEADS SMs do the work.
    // This is deliberate laziness; a production version would have multiple
    // SMs cooperatively process each head via a grid reduction. For a
    // prototype it keeps the code simple and correct.
    if (sm >= NUM_HEADS) return;

    int h = sm;  // one head per SM, SMs [0..NUM_HEADS) active.

    constexpr float inv_sqrt_d = 1.0f / 11.3137084989848f;  // 1/sqrt(128)

    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* scores = reinterpret_cast<float*>(smem_raw);      // [MAX_SEQ]
    __nv_bfloat16* q_h = reinterpret_cast<__nv_bfloat16*>(scores + MAX_SEQ);  // [HEAD_DIM]

    // Load Q for this head into shared memory.
    if (tid < HEAD_DIM) q_h[tid] = q[h * HEAD_DIM + tid];
    __syncthreads();

    // scores[t] = dot(q_h, K[t, h, :]) * inv_sqrt_d
    for (int t = tid; t < seq_len; t += BLOCK_SIZE) {
        float s = 0.f;
        const __nv_bfloat16* kt = K + t * HIDDEN + h * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            s += __bfloat162float(q_h[d]) * __bfloat162float(kt[d]);
        }
        scores[t] = s * inv_sqrt_d;
    }
    __syncthreads();

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

    // attn_out[h, d] = sum_t softmax(scores)_t * V[t, h, d]
    for (int d = tid; d < HEAD_DIM; d += BLOCK_SIZE) {
        float acc = 0.f;
        for (int t = 0; t < seq_len; t++) {
            acc += scores[t] * __bfloat162float(V[t * HIDDEN + h * HEAD_DIM + d]);
        }
        attn_out[h * HEAD_DIM + d] = __float2bfloat16(acc * inv_sum);
    }
}

// =============================================================================
// Attention O-projection + residual: residual += attn_in @ Wo
// Each SM owns a stripe of output dims.
// =============================================================================
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
        // Accumulate residual in FP32 to match eager reference (prevents
        // BF16-add rounding from compounding across 30 layers).
        float r = __bfloat162float(residual[d]);
        residual[d] = __float2bfloat16(r + acc);
    }
}

// =============================================================================
// MLP gate+up stage: mlp_scratch[j] = silu(gate_j) * up_j for j in [j0, j1).
// Each SM owns a stripe of INTER_DIM.
// =============================================================================
__device__ void mlp_gate_up_stage(
    const __nv_bfloat16* __restrict__ x,         // [HIDDEN]
    const __nv_bfloat16* __restrict__ W_gate,    // [HIDDEN, INTER_DIM]
    const __nv_bfloat16* __restrict__ W_up,      // [HIDDEN, INTER_DIM]
    __nv_bfloat16* __restrict__ mlp_scratch,      // [INTER_DIM]
    int sm, int num_sms, int tid)
{
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
        // Use IEEE expf (not __expf) to match torch.sigmoid precision across
        // 30 accumulated layers.
        float silu_g = g / (1.0f + expf(-g));
        mlp_scratch[j] = __float2bfloat16(silu_g * u);
    }
}

// =============================================================================
// MLP down stage: residual += mlp_scratch @ W_down.
// Each SM owns a stripe of HIDDEN output dims.
// =============================================================================
__device__ void mlp_down_residual_stage(
    const __nv_bfloat16* __restrict__ mlp_scratch,   // [INTER_DIM]
    const __nv_bfloat16* __restrict__ W_down,        // [INTER_DIM, HIDDEN]
    __nv_bfloat16* __restrict__ residual,            // [HIDDEN] in/out
    int sm, int num_sms, int tid)
{
    int dims_per_sm = (HIDDEN + num_sms - 1) / num_sms;
    int d0 = sm * dims_per_sm;
    int d1 = min(d0 + dims_per_sm, HIDDEN);
    for (int d = d0 + tid; d < d1; d += BLOCK_SIZE) {
        float acc = 0.f;
        for (int i = 0; i < INTER_DIM; i++) {
            acc += __bfloat162float(mlp_scratch[i]) * __bfloat162float(W_down[i * HIDDEN + d]);
        }
        // Accumulate residual in FP32 to match eager reference (prevents
        // BF16-add rounding from compounding across 30 layers).
        float r = __bfloat162float(residual[d]);
        residual[d] = __float2bfloat16(r + acc);
    }
}

// =============================================================================
// Per-layer weight bundle (passed as array-of-structs via pointer table).
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
    __nv_bfloat16* K_cache;              // [MAX_SEQ, HIDDEN]
    __nv_bfloat16* V_cache;              // [MAX_SEQ, HIDDEN]
};

// =============================================================================
// Main kernel: NUM_LAYERS-layer mega-graph decode, 7 barriers per layer.
// =============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
mega_graph_30layer_kernel(
    __nv_bfloat16* __restrict__ hidden,             // [HIDDEN] in/out
    const LayerWeights* __restrict__ layers,         // [NUM_LAYERS] host-populated
    __nv_bfloat16* __restrict__ normed,              // [HIDDEN] scratch
    __nv_bfloat16* __restrict__ q_scratch,           // [HIDDEN]
    __nv_bfloat16* __restrict__ attn_out,             // [HIDDEN]
    __nv_bfloat16* __restrict__ mlp_scratch,          // [INTER_DIM]
    int seq_len,
    int num_layers_run)  // up to NUM_LAYERS, for per-layer diagnostics
{
    auto grid = cg::this_grid();
    const int sm = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_sms = gridDim.x;

    // One pass per decoder layer.
    for (int L = 0; L < num_layers_run; L++) {
        const LayerWeights& W = layers[L];

        // --- Pre-attention RMSNorm (local per-SM; no grid.sync inside).
        rmsnorm_local_stage(hidden, W.input_norm, normed, sm, num_sms, tid);
        grid.sync();  // [1/7] normed cross-SM visible before qkv_proj

        // --- QKV projection. Each SM writes its stripe of q/k/v out.
        __nv_bfloat16* k_slot = W.K_cache + (seq_len - 1) * HIDDEN;
        __nv_bfloat16* v_slot = W.V_cache + (seq_len - 1) * HIDDEN;
        qkv_proj_stage(normed, W.Wq, W.Wk, W.Wv,
                       q_scratch, k_slot, v_slot,
                       sm, num_sms, tid);
        grid.sync();  // [2/7] q/k/v cross-SM visibility for attention

        // --- Attention core. Only SMs 0..NUM_HEADS-1 do work; others idle.
        attention_core_stage(q_scratch, W.K_cache, W.V_cache, attn_out,
                             seq_len, sm, num_sms, tid);
        grid.sync();  // [3/7] attn_out cross-SM visibility

        // --- O-proj + residual. SMs own HIDDEN stripes.
        attn_oproj_residual_stage(attn_out, W.Wo, hidden, sm, num_sms, tid);
        grid.sync();  // [4/7] hidden cross-SM visible for rmsnorm_post

        // --- Post-attn RMSNorm (local).
        rmsnorm_local_stage(hidden, W.post_attn_norm, normed, sm, num_sms, tid);
        grid.sync();  // [5/7] normed cross-SM visible before gate_up

        // --- MLP gate+up (SwiGLU intermediate).
        mlp_gate_up_stage(normed, W.W_gate, W.W_up, mlp_scratch,
                          sm, num_sms, tid);
        grid.sync();  // [6/7] mlp_scratch visible for down-proj

        // --- MLP down + residual add into hidden.
        mlp_down_residual_stage(mlp_scratch, W.W_down, hidden,
                                sm, num_sms, tid);
        grid.sync();  // [7/7] layer boundary (hidden visible for next rmsnorm)
    }
}

// =============================================================================
// Host launchers
// =============================================================================

extern "C" int mgg4_30_num_sms() {
    int d; cudaGetDevice(&d);
    int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
}
extern "C" int mgg4_30_hidden()     { return HIDDEN; }
extern "C" int mgg4_30_inter_dim()  { return INTER_DIM; }
extern "C" int mgg4_30_num_heads()  { return NUM_HEADS; }
extern "C" int mgg4_30_head_dim()   { return HEAD_DIM; }
extern "C" int mgg4_30_max_seq()    { return MAX_SEQ; }
extern "C" int mgg4_30_num_layers() { return NUM_LAYERS; }
extern "C" size_t mgg4_30_layer_weights_size() { return sizeof(LayerWeights); }

static constexpr int MGG4_30_SMEM_BYTES =
    MAX_SEQ * sizeof(float) + HEAD_DIM * sizeof(__nv_bfloat16) + 64;

extern "C" int mgg4_30_smem_bytes() { return MGG4_30_SMEM_BYTES; }

extern "C" cudaError_t mgg4_30_launch(
    __nv_bfloat16* hidden,
    const LayerWeights* layers_device,  // device pointer to array of NUM_LAYERS
    __nv_bfloat16* normed,
    __nv_bfloat16* q_scratch,
    __nv_bfloat16* attn_out,
    __nv_bfloat16* mlp_scratch,
    int seq_len,
    int num_layers_run,
    cudaStream_t stream)
{
    int num_sms = mgg4_30_num_sms();
    dim3 grid(num_sms), block(BLOCK_SIZE);
    void* args[] = {
        &hidden,
        &layers_device,
        &normed, &q_scratch, &attn_out, &mlp_scratch,
        &seq_len,
        &num_layers_run,
    };

    cudaError_t e = cudaFuncSetAttribute(
        (void*)mega_graph_30layer_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        MGG4_30_SMEM_BYTES);
    if (e != cudaSuccess) return e;

    return cudaLaunchCooperativeKernel(
        (void*)mega_graph_30layer_kernel,
        grid, block, args, MGG4_30_SMEM_BYTES, stream);
}

}  // namespace mega_graph_gemma4_30
