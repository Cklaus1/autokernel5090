// SPDX-License-Identifier: Apache-2.0
//
// Warp-specialized single-head decode attention for SM120a (consumer
// Blackwell). BF16 KV, head_dim=256, single token query.
//
// Design (see plans/warp_specialized_attention.md):
//   4 warps / CTA:
//     warp 0: producer A -- cp.async K tile, commit+wait
//     warp 1: producer B -- cp.async V tile, commit+wait
//     warps 2,3: consumers -- Q.K, online softmax, P.V accumulate
//
// Pipeline sync: we use __syncthreads() at stage boundaries (robust
// on SM120a -- mbarrier.try_wait.parity variants are flaky on
// consumer Blackwell compared to Hopper). We still get the producer
// /consumer decoupling within a stage (different warps overlap load
// and compute within one tile) and 2-stage software pipelining via
// explicit double-buffering.
//
// Correctness: online softmax recurrence identical to FlashInfer
// decode.
//
// Launch: 1 CTA, 128 threads. Single-split first; split-K later.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace warp_spec_decode {

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------
constexpr int HEAD_DIM = 256;
constexpr int BLOCK_N  = 32;        // KV tokens per stage
constexpr int NUM_WARPS = 4;
constexpr int NUM_THREADS = NUM_WARPS * 32;   // 128
constexpr int PRODUCER_WARPS = 2;              // K, V
constexpr int CONSUMER_WARPS = 2;

// Double-buffer: two K/V tile slots so producer can prefetch N+1
// while consumer works on N.
constexpr int STAGES = 2;

// ------------------------------------------------------------------
// PTX wrappers (cp.async -- SM80-class, present on SM120a)
// ------------------------------------------------------------------

__device__ __forceinline__
void cp_async_16B(void* smem_dst, const void* gmem_src) {
    uint32_t smem_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(smem_dst));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__
void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__
void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// ------------------------------------------------------------------
// Warp reduction helpers
// ------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

// ------------------------------------------------------------------
// Producer warps issue cp.async for one K or V tile.
// Each warp of 32 lanes copies 32 * 16B = 512B per pass; tile is
// BLOCK_N * HEAD_DIM * 2B = 16384 B => 32 passes per lane = 1024
// cp.async per warp, but we pipeline via commit_group / wait_group.
// ------------------------------------------------------------------
__device__ __forceinline__
void producer_copy_tile(
    __nv_bfloat16* smem_dst,                  // [BLOCK_N * HEAD_DIM]
    const __nv_bfloat16* gmem_src,            // full [seq, HEAD_DIM]
    int row0, int seq_len, int lane)
{
    constexpr int TILE_BYTES = BLOCK_N * HEAD_DIM * 2;   // bf16
    constexpr int TOTAL_16B  = TILE_BYTES / 16;          // 1024 for BLOCK_N=32, d=256
    constexpr int PER_LANE   = TOTAL_16B / 32;           // 32

    #pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
        int flat16   = lane + i * 32;            // 0..1023
        int flat_byte = flat16 * 16;
        int row = flat_byte / (HEAD_DIM * 2);
        int col = (flat_byte % (HEAD_DIM * 2)) / 2;   // bf16 idx
        int abs_row = row0 + row;
        __nv_bfloat16* d = smem_dst + row * HEAD_DIM + col;
        if (abs_row < seq_len) {
            const __nv_bfloat16* s =
                gmem_src + abs_row * HEAD_DIM + col;
            cp_async_16B(d, s);
        } else {
            *reinterpret_cast<uint4*>(d) = uint4{0,0,0,0};
        }
    }
}

// ------------------------------------------------------------------
// Kernel
// ------------------------------------------------------------------
//
// SMEM layout:
//   K[STAGES][BLOCK_N][HEAD_DIM] bf16
//   V[STAGES][BLOCK_N][HEAD_DIM] bf16
//   Q[HEAD_DIM] bf16                -- shared copy (decode Q small)
//   scores[BLOCK_N] float           -- per-tile logits
//   row_stats[2] float              -- row_max, row_sum
//
__launch_bounds__(NUM_THREADS, 2)
__global__ void warp_spec_decode_kernel(
    const __nv_bfloat16* __restrict__ query,   // [num_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ key,     // [num_heads, seq_len, HEAD_DIM]
    const __nv_bfloat16* __restrict__ value,   // [num_heads, seq_len, HEAD_DIM]
    __nv_bfloat16* __restrict__ out,           // [num_heads, HEAD_DIM]
    int seq_len,
    float sm_scale)
{
    // Each CTA processes one head = grid.x
    const int head_id = blockIdx.x;
    query = query + head_id * HEAD_DIM;
    key   = key   + (int64_t)head_id * seq_len * HEAD_DIM;
    value = value + (int64_t)head_id * seq_len * HEAD_DIM;
    out   = out   + head_id * HEAD_DIM;

    extern __shared__ __align__(16) char smem_raw[];
    auto* smem_K = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    auto* smem_V = smem_K + STAGES * BLOCK_N * HEAD_DIM;
    auto* smem_Q = smem_V + STAGES * BLOCK_N * HEAD_DIM;
    auto* smem_scores = reinterpret_cast<float*>(smem_Q + HEAD_DIM);
    auto* smem_rowstats = smem_scores + BLOCK_N;        // [2]

    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane    = tid & 31;

    // ---- Load Q into SMEM ----
    if (tid < HEAD_DIM / 2) {
        reinterpret_cast<float*>(smem_Q)[tid] =
            reinterpret_cast<const float*>(query)[tid];
    }

    const int num_tiles = (seq_len + BLOCK_N - 1) / BLOCK_N;
    constexpr int TILE_ELEMS = BLOCK_N * HEAD_DIM;

    // ==============================================================
    // Prologue: producer warps prefetch tile 0 into stage 0.
    // ==============================================================
    if (warp_id == 0 && num_tiles > 0) {
        // K producer
        producer_copy_tile(smem_K + 0 * TILE_ELEMS, key, 0, seq_len, lane);
        cp_async_commit_group();
    }
    if (warp_id == 1 && num_tiles > 0) {
        // V producer
        producer_copy_tile(smem_V + 0 * TILE_ELEMS, value, 0, seq_len, lane);
        cp_async_commit_group();
    }
    // All other warps idle through prologue.
    __syncthreads();

    // ---- Consumer state ----
    // Each consumer warp owns HEAD_DIM / CONSUMER_WARPS = 128 dims,
    // each lane handles DIMS_PER_LANE = 128/32 = 4 dims.
    const int cons_id = warp_id - PRODUCER_WARPS;      // 0 or 1 for warps 2,3
    constexpr int DIMS_PER_CONS = HEAD_DIM / CONSUMER_WARPS;   // 128
    constexpr int PER_LANE_OUT  = DIMS_PER_CONS / 32;          // 4

    float acc[PER_LANE_OUT];
    #pragma unroll
    for (int i = 0; i < PER_LANE_OUT; ++i) acc[i] = 0.0f;
    float m_state = -1e30f;
    float l_state = 0.0f;

    // ==============================================================
    // Main loop: at iteration t, stage `s = t % STAGES` holds tile t,
    // producers prefetch t+1 into stage `1 - s`.
    // ==============================================================
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int stage = tile % STAGES;
        const int next_tile = tile + 1;
        const int next_stage = next_tile % STAGES;
        const int row0 = tile * BLOCK_N;
        const int valid = min(BLOCK_N, seq_len - row0);

        // Wait for this tile's cp.async to land.
        // (Each producer committed one group per tile; wait until
        //  all prior groups complete.)
        cp_async_wait_group<0>();
        __syncthreads();

        // Kick off prefetch of next tile while consumers compute.
        if (next_tile < num_tiles) {
            if (warp_id == 0) {
                producer_copy_tile(
                    smem_K + next_stage * TILE_ELEMS,
                    key, next_tile * BLOCK_N, seq_len, lane);
                cp_async_commit_group();
            } else if (warp_id == 1) {
                producer_copy_tile(
                    smem_V + next_stage * TILE_ELEMS,
                    value, next_tile * BLOCK_N, seq_len, lane);
                cp_async_commit_group();
            }
        }

        // ---- Consumer work (only on consumer warps) ----
        if (warp_id >= PRODUCER_WARPS) {
            __nv_bfloat16* K_tile = smem_K + stage * TILE_ELEMS;
            __nv_bfloat16* V_tile = smem_V + stage * TILE_ELEMS;

            // --- Q . K^T : warp splits positions ---
            constexpr int POS_PER_WARP = BLOCK_N / CONSUMER_WARPS;   // 16
            for (int pp = 0; pp < POS_PER_WARP; ++pp) {
                int pos = cons_id * POS_PER_WARP + pp;
                float dot = 0.0f;
                if (pos < valid) {
                    // HEAD_DIM / 32 = 8 dims per lane (full head_dim)
                    constexpr int DIMS_PER_LANE_FULL = HEAD_DIM / 32;
                    #pragma unroll
                    for (int d = 0; d < DIMS_PER_LANE_FULL; ++d) {
                        int dim = lane + d * 32;
                        float q = __bfloat162float(smem_Q[dim]);
                        float k = __bfloat162float(
                            K_tile[pos * HEAD_DIM + dim]);
                        dot += q * k;
                    }
                    dot = warp_reduce_sum(dot);
                }
                if (lane == 0) {
                    smem_scores[pos] =
                        (pos < valid) ? dot * sm_scale : -1e30f;
                }
            }
        }
        // Sync so producers + consumers agree before softmax reduce.
        __syncthreads();

        // ---- Row softmax (consumer warp 0 does it) ----
        if (warp_id == PRODUCER_WARPS) {
            float v = (lane < BLOCK_N) ? smem_scores[lane] : -1e30f;
            float rmax = warp_reduce_max(v);
            float e = (lane < BLOCK_N) ? expf(v - rmax) : 0.0f;
            float rsum = warp_reduce_sum(e);
            if (lane == 0) {
                smem_rowstats[0] = rmax;
                smem_rowstats[1] = rsum;
            }
            // Write unnormalized P back
            if (lane < BLOCK_N) {
                smem_scores[lane] = e;
            }
        }
        __syncthreads();

        // ---- P . V accumulate (both consumer warps) ----
        if (warp_id >= PRODUCER_WARPS) {
            float tile_max = smem_rowstats[0];
            float tile_sum = smem_rowstats[1];

            float m_new = fmaxf(m_state, tile_max);
            float alpha = expf(m_state - m_new);
            float beta  = expf(tile_max - m_new);

            #pragma unroll
            for (int i = 0; i < PER_LANE_OUT; ++i) acc[i] *= alpha;

            __nv_bfloat16* V_tile = smem_V + stage * TILE_ELEMS;
            #pragma unroll
            for (int i = 0; i < PER_LANE_OUT; ++i) {
                int dim = cons_id * DIMS_PER_CONS + lane + i * 32;
                float s = 0.0f;
                #pragma unroll
                for (int p = 0; p < BLOCK_N; ++p) {
                    if (p < valid) {
                        s += smem_scores[p]
                           * __bfloat162float(V_tile[p * HEAD_DIM + dim]);
                    }
                }
                acc[i] += beta * s;
            }

            l_state = l_state * alpha + tile_sum * beta;
            m_state = m_new;
        }

        // End of tile: barrier before next iteration reuses scores/rowstats.
        __syncthreads();
    }

    // ---- Finalize ----
    if (warp_id >= PRODUCER_WARPS) {
        float inv_l = 1.0f / l_state;
        #pragma unroll
        for (int i = 0; i < PER_LANE_OUT; ++i) {
            int dim = cons_id * DIMS_PER_CONS + lane + i * 32;
            out[dim] = __float2bfloat16(acc[i] * inv_l);
        }
    }
}

// ------------------------------------------------------------------
// Launcher
// ------------------------------------------------------------------
void warp_spec_decode_attention(
    torch::Tensor& out,        // [H, HEAD_DIM] or [HEAD_DIM]
    torch::Tensor const& q,    // [H, HEAD_DIM] or [HEAD_DIM]
    torch::Tensor const& k,    // [H, seq_len, HEAD_DIM] or [seq_len, HEAD_DIM]
    torch::Tensor const& v,
    double sm_scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && out.is_cuda());
    TORCH_CHECK(q.scalar_type() == at::kBFloat16);
    TORCH_CHECK(k.size(-1) == HEAD_DIM);
    TORCH_CHECK(v.size(-1) == HEAD_DIM);

    int num_heads, seq_len;
    if (k.dim() == 2) {
        num_heads = 1;
        seq_len = static_cast<int>(k.size(0));
    } else {
        num_heads = static_cast<int>(k.size(0));
        seq_len = static_cast<int>(k.size(1));
    }

    const at::cuda::OptionalCUDAGuard guard(q.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    constexpr int SMEM_KV = 2 * STAGES * BLOCK_N * HEAD_DIM * 2;   // K + V
    constexpr int SMEM_Q  = HEAD_DIM * 2;
    constexpr int SMEM_STATS = (BLOCK_N + 2) * 4;                   // scores + rowstats
    constexpr int SMEM_TOTAL = SMEM_KV + SMEM_Q + SMEM_STATS + 256;

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            (const void*)warp_spec_decode_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            SMEM_TOTAL);
        attr_set = true;
    }

    dim3 grid(num_heads);
    dim3 block(NUM_THREADS);
    warp_spec_decode_kernel<<<grid, block, SMEM_TOTAL, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        seq_len,
        static_cast<float>(sm_scale));
    C10_CUDA_CHECK(cudaGetLastError());
}

}   // namespace warp_spec_decode

// ------------------------------------------------------------------
// Torch bindings
// ------------------------------------------------------------------
TORCH_LIBRARY(warp_spec_decode, m) {
    m.def("decode(Tensor! out, Tensor q, Tensor k, Tensor v, float sm_scale) -> ()");
    m.impl("decode", torch::kCUDA, &warp_spec_decode::warp_spec_decode_attention);
}
