// SPDX-License-Identifier: Apache-2.0
//
// SM120a Calibration Microbenchmark Suite — W6_sm120_calibration_extended
//
// 8 independent calibration benches. Each bench is a standalone cooperative
// kernel or device kernel that emits raw cycle counts + derived µs values.
//
// Build: nvcc --gpu-architecture=sm_120a -O3 -lineinfo \
//             -I/usr/local/cuda/include \
//             -lcuda -lcudart \
//             -o sm120_calib_bench sm120_calibration_bench.cu
//
// Design notes:
//   * All benches use clock64() for cycle counting (SM-local hardware counter).
//   * grid.sync benches use cudaLaunchCooperativeKernel — requires all SMs.
//   * TMA bench (bench8) is compiled conditionally; if ptxas rejects the
//     cp.async.bulk.tensor instruction, a #error is emitted so the parent
//     can distinguish "compile-time illegal" from "run-time illegal instruction".
//   * No result analysis done on GPU — everything is raw numbers, host prints.
//
// Compile-time guards:
//   BENCH_TMA_FORCE   — define to attempt TMA compilation even on <sm_120a
//   SKIP_TMA          — define to skip bench8 entirely (if nvcc refuses to compile)
//
// Tag: W6_sm120_calibration_extended

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

// ---------------------------------------------------------------------------
// Shared result buffer layout (written by GPU, read by host)
// ---------------------------------------------------------------------------
// Each bench writes into a fixed slot of a uint64_t[MAX_RESULTS] array.
// Slot assignment:
//   [0..3]   bench1  grid.sync cost (4 conditions × 2 values: start+end)
//   [8..15]  bench2  cp.async vs WMMA crossover (8 B-tile sizes × 2 values)
//   [16]     bench3  mma.sync throughput (per-warp cycle count)
//   [17]     bench3  iterations done
//   [18]     bench4  cvt throughput (cycle count for N conversions)
//   [19]     bench4  N conversions done
//   [20]     bench5  shfl_sync cycle count
//   [21]     bench5  atomicAdd cycle count
//   [22]     bench6  HBM copy cycle count
//   [23]     bench6  bytes transferred
//   [24..33] bench7  graph node counts (10 steps)
//   [34..43] bench7  crash flags (1=ok, 0=crash) — filled by host
//   [44]     bench8  TMA probe: 0=ok, 1=illegal-instruction caught
//
#define MAX_RESULTS 64

// ---------------------------------------------------------------------------
// Utility: nanoseconds from cycle count
// ---------------------------------------------------------------------------
// Host helper: cycles_to_us(cycles, clock_ghz) = cycles / (clock_ghz * 1e3)

// ---------------------------------------------------------------------------
// BENCH 1 — grid.sync cost under three occupancy conditions
//
// Three sub-kernels share the same harness:
//   1a: WMMA-idle  (no tensor-core ops in loop body)
//   1b: WMMA-active (mma.sync fires between syncs)
//   1c: cp.async-active (cp.async.ca.shared fires between syncs)
//
// Each sub-kernel does BENCH1_ITERS barriers, reports total cycles / iters.
// ---------------------------------------------------------------------------
#define BENCH1_ITERS 200

__global__ void bench1a_gridsync_idle(uint64_t* __restrict__ out, int iters) {
    // Cooperative kernel: launched with one block per SM
    cg::grid_group grid = cg::this_grid();
    // Warmup
    for (int i = 0; i < 5; i++) grid.sync();

    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) {
        grid.sync();
    }
    uint64_t t1 = clock64();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = t0;
        out[1] = t1;
        out[2] = (uint64_t)iters;
        // slot 3: condition tag = 1 (idle)
        out[3] = 1;
    }
}

__global__ void bench1b_gridsync_wmma_active(uint64_t* __restrict__ out, int iters,
                                              const __nv_bfloat16* __restrict__ A,
                                              const __nv_bfloat16* __restrict__ B) {
    cg::grid_group grid = cg::this_grid();
    // WMMA fragments (16x16x16 BF16 -> FP32)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    // Warmup
    for (int i = 0; i < 5; i++) grid.sync();

    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) {
        // Fire a WMMA load + mma between each sync to keep tensor cores warm
        wmma::load_matrix_sync(fa, A, 16);
        wmma::load_matrix_sync(fb, B, 16);
        wmma::mma_sync(fc, fa, fb, fc);
        grid.sync();
    }
    uint64_t t1 = clock64();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Sink accumulator to prevent dead-code elimination
        float sink = 0.0f;
        for (int e = 0; e < (int)fc.num_elements; e++) sink += fc.x[e];
        out[0] = t0;
        out[1] = t1;
        out[2] = (uint64_t)iters;
        out[3] = 2;  // condition tag = 2 (WMMA-active)
        // Prevent DCE of sink
        if (sink == 12345.0f) out[3] = 99;
    }
}

__global__ void bench1c_gridsync_cpasync_active(uint64_t* __restrict__ out, int iters,
                                                 const __nv_bfloat16* __restrict__ src) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __nv_bfloat16 smem[256];

    // Warmup
    for (int i = 0; i < 5; i++) grid.sync();

    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) {
        // Issue cp.async between syncs (128-byte line from global -> shared)
        if (threadIdx.x < 64) {
            uint32_t smem_addr = __cvta_generic_to_shared(&smem[threadIdx.x * 4]);
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 8;\n"
                :
                : "r"(smem_addr),
                  "l"((const __nv_bfloat16*)src + (size_t)threadIdx.x * 4)
            );
        }
        asm volatile("cp.async.wait_all;\n" :::);
        __syncthreads();
        grid.sync();
    }
    uint64_t t1 = clock64();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = t0;
        out[1] = t1;
        out[2] = (uint64_t)iters;
        out[3] = 3;  // condition tag = 3 (cp.async-active)
        // Sink to prevent DCE
        float s = (float)smem[0];
        if (s == 12345.0f) out[3] = 99;
    }
}

// ---------------------------------------------------------------------------
// BENCH 2 — wmma::load_matrix_sync vs cp.async crossover
//
// Sweeps B-tile sizes [16, 32, 64, 128, 256] × K-step counts [4, 8, 16, 32].
// For each combo: runs BENCH2_ITERS iterations of the inner loop, measures
// cycles. Result written to out[8 + combo_idx * 2] = {cycles, combo_tag}.
//
// combo_tag encodes: (b_tile_idx * 4 + k_step_idx)
// B-tile sizes: 16=0, 32=1, 64=2, 128=3, 256=4
// K-step counts: 4=0, 8=1, 16=2, 32=3
//
// Two variants:
//   bench2a: wmma::load_matrix_sync path
//   bench2b: cp.async path
// Each bench sweeps over a single (B, K) combo selected by launch parameter.
// ---------------------------------------------------------------------------
#define BENCH2_ITERS 100

// wmma load path — one warp does K-step mma.sync ops with B-tile load via wmma
__global__ void bench2a_wmma_load(uint64_t* __restrict__ out,
                                   const __nv_bfloat16* __restrict__ B_global,
                                   int b_tile_cols,   // 16..256 (multiple of 16)
                                   int k_steps,       // 4..32
                                   int combo_idx,
                                   int iters) {
    // Only warp 0 in block 0 does the measurement
    if (blockIdx.x != 0 || (threadIdx.x / 32) != 0) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    // Fake A tile in shared memory (16×16 = 256 bf16)
    __shared__ __nv_bfloat16 A_smem[16 * 16];
    if (threadIdx.x < 256) A_smem[threadIdx.x] = __float2bfloat16(1.0f);

    // Warmup
    for (int w = 0; w < 5; w++) {
        wmma::load_matrix_sync(fa, A_smem, 16);
    }

    uint64_t t0 = clock64();
    for (int it = 0; it < iters; it++) {
        // Loop over B-tile: b_tile_cols/16 WMMA N-tiles, each K-step deep
        for (int n = 0; n < b_tile_cols / 16; n++) {
            for (int k = 0; k < k_steps; k++) {
                wmma::load_matrix_sync(fa, A_smem, 16);
                wmma::load_matrix_sync(fb, B_global + (size_t)k * 16 * 16 + n * 16, 16);
                wmma::mma_sync(fc, fa, fb, fc);
            }
        }
    }
    uint64_t t1 = clock64();

    if (threadIdx.x == 0) {
        int slot = 8 + combo_idx * 2;
        out[slot]     = t1 - t0;
        out[slot + 1] = (uint64_t)combo_idx * 100 + 1;  // tag: wmma=1
        // DCE sink
        float s = 0.0f;
        for (int e = 0; e < (int)fc.num_elements; e++) s += fc.x[e];
        if (s == 12345.0f) out[slot + 1] = 99;
    }
}

// cp.async path — loads B-tile into smem via cp.async, then does mma.sync
__global__ void bench2b_cpasync_load(uint64_t* __restrict__ out,
                                      const __nv_bfloat16* __restrict__ B_global,
                                      int b_tile_cols,
                                      int k_steps,
                                      int combo_idx,
                                      int iters) {
    if (blockIdx.x != 0 || (threadIdx.x / 32) != 0) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.0f);

    // Shared memory: enough for 256 B-tile cols × 16 rows × k_steps=32
    // Max: 256 * 16 * 32 * 2 bytes = 262 KB — too large. Use double-buffer of
    // current B-tile only: b_tile_cols * 16 * 2 bytes max = 256*16*2 = 8 KB
    extern __shared__ __nv_bfloat16 smem_b[];  // [k_steps * 16 * 16]

    __shared__ __nv_bfloat16 A_smem[16 * 16];
    if (threadIdx.x < 256) A_smem[threadIdx.x] = __float2bfloat16(1.0f);
    __syncthreads();

    // Warmup
    for (int w = 0; w < 5; w++) {
        wmma::load_matrix_sync(fa, A_smem, 16);
    }

    uint64_t t0 = clock64();
    for (int it = 0; it < iters; it++) {
        for (int n = 0; n < b_tile_cols / 16; n++) {
            // Async-copy one 16×16 B fragment into smem
            if (threadIdx.x < (k_steps <= 4 ? 32 : (k_steps <= 8 ? 64 : 128))) {
                int elems_per_thread = (k_steps * 16 * 16) / 32;
                for (int e = 0; e < elems_per_thread && e < 8; e++) {
                    int idx = threadIdx.x * elems_per_thread + e;
                    if (idx < k_steps * 16 * 16) {
                        uint32_t dst = __cvta_generic_to_shared(&smem_b[idx]);
                        asm volatile(
                            "cp.async.ca.shared.global [%0], [%1], 4;\n"
                            :
                            : "r"(dst),
                              "l"(B_global + (size_t)n * k_steps * 256 + idx)
                        );
                    }
                }
            }
            asm volatile("cp.async.wait_all;\n" :::);
            __syncwarp();

            for (int k = 0; k < k_steps; k++) {
                wmma::load_matrix_sync(fa, A_smem, 16);
                wmma::load_matrix_sync(fb, smem_b + k * 16 * 16, 16);
                wmma::mma_sync(fc, fa, fb, fc);
            }
        }
    }
    uint64_t t1 = clock64();

    if (threadIdx.x == 0) {
        int slot = 8 + combo_idx * 2;
        out[slot]     = t1 - t0;
        out[slot + 1] = (uint64_t)combo_idx * 100 + 2;  // tag: cpasync=2
        float s = 0.0f;
        for (int e = 0; e < (int)fc.num_elements; e++) s += fc.x[e];
        if (s == 12345.0f) out[slot + 1] = 99;
    }
}

// ---------------------------------------------------------------------------
// BENCH 3 — mma.sync.m16n8k16 BF16 throughput
//
// Measures:
//   (a) per-warp sustained rate: warp 0 of block 0, unrolled loop
//   (b) full-SM rate: all 8 warps in block 0
//   (c) full-SM-occupancy rate: all blocks across all SMs (cooperative launch)
//
// Output: out[16] = total_cycles, out[17] = total_mma_ops
// Throughput in TOPS = (total_mma_ops * 2 * 16*8*16) / (cycles / clk_ghz * 1e9)
// ---------------------------------------------------------------------------
#define BENCH3_MMA_ITERS 1000
#define BENCH3_UNROLL    16

__global__ void bench3_mma_throughput(uint64_t* __restrict__ out, int iters) {
    // All warps participate. Use mma.sync m16n8k16 via wmma API.
    // Note: wmma only exposes m16n16k16 at the C++ level; the PTX m16n8k16
    // is the underlying hardware op. We use m16n16k16 as a proxy (2 m16n8k16
    // internally on SM80+). On SM120 this is the same tile.

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fa, __float2bfloat16(1.0f));
    wmma::fill_fragment(fb, __float2bfloat16(1.0f));
    wmma::fill_fragment(fc, 0.0f);

    // Warmup
    for (int w = 0; w < 10; w++) {
        wmma::mma_sync(fc, fa, fb, fc);
    }
    __syncwarp();

    uint64_t t0 = clock64();
    #pragma unroll BENCH3_UNROLL
    for (int i = 0; i < iters; i++) {
        wmma::mma_sync(fc, fa, fb, fc);
    }
    uint64_t t1 = clock64();

    // Only warp 0 thread 0 of block 0 reports
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[16] = t1 - t0;
        out[17] = (uint64_t)iters;
        // DCE sink
        float s = 0.0f;
        for (int e = 0; e < (int)fc.num_elements; e++) s += fc.x[e];
        if (s == 12345.0f) out[17] = 99;
    }
}

// Full-SM-occupancy variant (cooperative): every SM runs, block 0 warp 0 reports
__global__ void bench3b_mma_fullsm(uint64_t* __restrict__ out, int iters) {
    cg::grid_group grid = cg::this_grid();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fa, __float2bfloat16(1.0f));
    wmma::fill_fragment(fb, __float2bfloat16(1.0f));
    wmma::fill_fragment(fc, 0.0f);

    // Warmup + barrier to ensure all SMs are live before timing
    for (int w = 0; w < 5; w++) wmma::mma_sync(fc, fa, fb, fc);
    grid.sync();

    uint64_t t0 = clock64();
    #pragma unroll 8
    for (int i = 0; i < iters; i++) {
        wmma::mma_sync(fc, fa, fb, fc);
    }
    grid.sync();
    uint64_t t1 = clock64();

    // SM 0 warp 0 thread 0 reports (worst-case SM timing after barrier)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[16] = t1 - t0;      // overwrite single-warp result
        out[17] = (uint64_t)iters * gridDim.x * (blockDim.x / 32);
        float s = 0.0f;
        for (int e = 0; e < (int)fc.num_elements; e++) s += fc.x[e];
        if (s == 12345.0f) out[17] = 99;
    }
}

// ---------------------------------------------------------------------------
// BENCH 4 — cvt.rn.satfinite.e2m1x2.f32 throughput
//
// Converts BENCH4_N fp32 values to fp4x2 via inline PTX.
// Reports: out[18] = cycles for N conversions, out[19] = N
// ---------------------------------------------------------------------------
#define BENCH4_N 8192

__global__ void bench4_cvt_fp4_throughput(uint64_t* __restrict__ out,
                                           const float* __restrict__ src,
                                           uint8_t* __restrict__ dst) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Warmup
    uint32_t sink = 0;
    for (int i = 0; i < 32; i++) {
        float v0 = src[i * 2];
        float v1 = src[i * 2 + 1];
        uint32_t packed;
        asm volatile(
            "cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;\n"
            : "=r"(packed)
            : "f"(v1), "f"(v0)
        );
        sink ^= packed;
    }

    uint64_t t0 = clock64();
    for (int i = 0; i < BENCH4_N / 2; i++) {
        float v0 = src[i * 2];
        float v1 = src[i * 2 + 1];
        uint32_t packed;
        asm volatile(
            "cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;\n"
            : "=r"(packed)
            : "f"(v1), "f"(v0)
        );
        dst[i] = (uint8_t)(packed & 0xFF);
        sink ^= packed;
    }
    uint64_t t1 = clock64();

    out[18] = t1 - t0;
    out[19] = (uint64_t)BENCH4_N;
    // DCE sink
    if (sink == 0xDEADBEEF) out[19] = 99;
}

// ---------------------------------------------------------------------------
// BENCH 5 — __shfl_sync vs atomicAdd on smem for cross-warp reduction
//
// Measures the cost of reducing 32 partial sums (one per warp in a 256-thread
// block) into a single sum via two approaches:
//   (a) Warp-level shfl_sync tree → single warp writes result → __syncthreads
//   (b) Each warp leader atomicAdd to smem location → __syncthreads
//
// Runs BENCH5_ITERS reductions, reports cycles per reduction.
// Output: out[20] = shfl_sync total cycles, out[21] = atomicAdd total cycles
// ---------------------------------------------------------------------------
#define BENCH5_ITERS 10000
#define BENCH5_WARPS 8  // 256 threads / 32

__global__ void bench5_reduction_methods(uint64_t* __restrict__ out, int iters) {
    if (blockIdx.x != 0) return;

    __shared__ float smem_atomic[1];
    __shared__ float smem_shfl[BENCH5_WARPS];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    float val = (float)(threadIdx.x + 1);

    // ---- (a) shfl_sync tree ----
    float sink_shfl = 0.0f;
    // Warmup
    for (int w = 0; w < 10; w++) {
        float v = val;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane_id == 0) smem_shfl[warp_id] = v;
        __syncthreads();
        // Warp 0 reduces the per-warp sums
        if (warp_id == 0) {
            float wv = (lane_id < BENCH5_WARPS) ? smem_shfl[lane_id] : 0.0f;
            for (int offset = 4; offset > 0; offset >>= 1)
                wv += __shfl_down_sync(0xFFFFFFFF, wv, offset);
            if (lane_id == 0) sink_shfl += wv;
        }
        __syncthreads();
    }

    __syncthreads();
    uint64_t t0_shfl = clock64();
    for (int it = 0; it < iters; it++) {
        float v = val;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane_id == 0) smem_shfl[warp_id] = v;
        __syncthreads();
        if (warp_id == 0) {
            float wv = (lane_id < BENCH5_WARPS) ? smem_shfl[lane_id] : 0.0f;
            for (int offset = 4; offset > 0; offset >>= 1)
                wv += __shfl_down_sync(0xFFFFFFFF, wv, offset);
            if (lane_id == 0) sink_shfl += wv;
        }
        __syncthreads();
    }
    uint64_t t1_shfl = clock64();

    // ---- (b) atomicAdd to smem ----
    float sink_atomic = 0.0f;
    // Warmup
    for (int w = 0; w < 10; w++) {
        if (threadIdx.x == 0) smem_atomic[0] = 0.0f;
        __syncthreads();
        float v = val;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane_id == 0) atomicAdd(&smem_atomic[0], v);
        __syncthreads();
        if (threadIdx.x == 0) sink_atomic += smem_atomic[0];
    }

    __syncthreads();
    uint64_t t0_atom = clock64();
    for (int it = 0; it < iters; it++) {
        if (threadIdx.x == 0) smem_atomic[0] = 0.0f;
        __syncthreads();
        float v = val;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
        if (lane_id == 0) atomicAdd(&smem_atomic[0], v);
        __syncthreads();
        if (threadIdx.x == 0) sink_atomic += smem_atomic[0];
    }
    uint64_t t1_atom = clock64();

    if (threadIdx.x == 0) {
        out[20] = t1_shfl - t0_shfl;   // shfl cycles
        out[21] = t1_atom - t0_atom;   // atomicAdd cycles
        // DCE sinks
        if (sink_shfl == 12345.0f || sink_atomic == 12345.0f) {
            out[20] = 99; out[21] = 99;
        }
    }
}

// ---------------------------------------------------------------------------
// BENCH 6 — HBM sustained bandwidth at M=1 decode traffic pattern
//
// Issues a pure 128-bit-stride copy mimicking the M=1 weight-streaming pattern:
//   * Reads B bytes from src (non-coalesced stride to mimic weight columns)
//   * Writes B bytes to dst
// Two sub-tests:
//   (a) Sequential (fully coalesced) — ceiling bandwidth
//   (b) Strided (stride = HIDDEN=2048 fp16 elements) — decode pattern
//
// Output: out[22] = sequential cycles, out[23] = bytes_transferred
//         out[24] = strided cycles,    out[25] = bytes_transferred
// ---------------------------------------------------------------------------
#define BENCH6_BYTES (128ULL * 1024 * 1024)  // 128 MB per direction

__global__ void bench6_hbm_bw(uint64_t* __restrict__ out,
                               const float* __restrict__ src,
                               float* __restrict__ dst,
                               long long n_floats) {
    // All threads participate
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    // ---- (a) Sequential coalesced ----
    __syncthreads();
    if (tid == 0) out[22] = clock64();  // approximate: only SM0 timestamps

    float acc = 0.0f;
    for (long long i = tid; i < n_floats; i += stride) {
        float v = src[i];
        dst[i] = v + 1.0f;
        acc += v;
    }
    __syncthreads();
    if (tid == 0) {
        out[22] = clock64() - out[22];
        out[23] = (uint64_t)n_floats * 4ULL;  // bytes read
        if (acc == 12345.0f) out[23] = 99;    // DCE sink
    }
}

// Strided variant: reads every STRIDE-th element (simulates decode column access)
__global__ void bench6b_hbm_strided(uint64_t* __restrict__ out,
                                     const float* __restrict__ src,
                                     float* __restrict__ dst,
                                     long long n_floats,
                                     int col_stride) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    long long n_cols = n_floats / col_stride;

    if (tid == 0) out[24] = clock64();

    float acc = 0.0f;
    for (long long col = tid; col < n_cols; col += stride) {
        float v = src[col * col_stride];
        dst[col * col_stride] = v + 1.0f;
        acc += v;
    }
    __syncthreads();
    if (tid == 0) {
        out[24] = clock64() - out[24];
        out[25] = (uint64_t)n_cols * 4ULL;
        if (acc == 12345.0f) out[25] = 99;
    }
}

// ---------------------------------------------------------------------------
// BENCH 7 — CUDA graph node-count stress test
//
// Builds a chain graph of N no-op kernels (memset to 0), captures it, replays
// 3 times. Uses cudaGraphCapture API — run entirely from host. This bench
// does NOT have a device kernel; the logic lives in run_sm120_calibration.py.
// This stub is a placeholder device kernel that does nothing (the real bench
// is host-side graph construction in the Python runner).
// ---------------------------------------------------------------------------
__global__ void bench7_noop_kernel(int* __restrict__ buf, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) buf[tid] = 0;
}

// ---------------------------------------------------------------------------
// BENCH 8 — TMA cp.async.bulk.tensor availability on SM120a
//
// Attempts to execute a cp.async.bulk.tensor.1d.shared::cluster.global PTX
// instruction. If the hardware supports it, data moves; if not, the thread
// raises an illegal-instruction fault caught by cudaGetLastError.
//
// Compiled with --gpu-architecture=sm_120a. The __CUDA_ARCH__ >= 900 guard
// in upstream code would pass for sm_120a (arch 1200). We check if the
// *instruction* actually executes without fault.
//
// NOTE: cp.async.bulk.tensor requires sm_90+ PTX. We emit it unconditionally
// (the sm_120a arch accepts sm_90 PTX extensions). If it runs without fault,
// TMA is available; if cudaGetLastError returns cudaErrorIllegalInstruction,
// TMA is absent.
//
// Output: out[44] = 0 if successful, 1 if compile-time disabled
// The actual fault detection is done on the host via cudaGetLastError.
// ---------------------------------------------------------------------------

#ifndef SKIP_TMA

// Minimal TMA descriptor type — matches CUDA 12.x cp_async_bulk layout.
// We use a raw 64-byte opaque descriptor (CUDA's CUtensorMap equivalent).
// For this probe we do NOT need a valid descriptor — we just need ptxas to
// accept the instruction. The kernel will trap at runtime if TMA is absent;
// that trap is what we're measuring.
//
// WARNING: This kernel WILL fault on hardware without TMA. That is intentional.
// The host runner catches the fault via cudaGetLastError and reports it.

__global__ void bench8_tma_probe(uint64_t* __restrict__ out,
                                  const void* __restrict__ tma_desc,
                                  void* __restrict__ smem_raw) {
    // Only thread 0 issues the TMA instruction
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    __shared__ uint8_t smem_buf[128];
    // smem_raw is unused — we use smem_buf declared above
    (void)smem_raw;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Attempt cp.async.bulk.tensor.1d.shared::cta.global instruction.
    // This is a minimal 1D bulk copy: copy 16 bytes from global into shared.
    // The mbarrier (synchronization barrier) must be 64-bit aligned in smem.
    __shared__ uint64_t mbar;

    // Initialize mbarrier for 1 arrival
    asm volatile(
        "mbarrier.init.shared.b64 [%0], 1;\n"
        :
        : "r"((uint32_t)__cvta_generic_to_shared(&mbar))
    );

    uint32_t smem_addr = __cvta_generic_to_shared(&smem_buf[0]);
    // Note: tma_desc must be a valid CUtensorMap pointer on real hardware.
    // We pass NULL — this will trap on real HW if TMA is present but we pass
    // a bad descriptor, OR illegal-instruction if TMA is absent entirely.
    asm volatile(
        "cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {0}], 16, [%2];\n"
        :
        : "r"(smem_addr),
          "l"(tma_desc),
          "r"((uint32_t)__cvta_generic_to_shared(&mbar))
    );

    // If we reach here: instruction was accepted by hardware (TMA exists).
    // The NULL descriptor likely causes a separate fault, but the instruction
    // itself was not illegal.
    out[44] = 0;  // TMA instruction accepted
#else
    // Architecture too old for TMA — should not happen with sm_120a but guard.
    out[44] = 2;  // compile-time arch too old
#endif
}

#else
// SKIP_TMA defined — stub that just writes "skipped" sentinel
__global__ void bench8_tma_probe(uint64_t* __restrict__ out,
                                  const void* __restrict__ tma_desc,
                                  void* __restrict__ smem_raw) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[44] = 3;  // skipped
}
#endif  // SKIP_TMA

// ---------------------------------------------------------------------------
// Host-callable launcher stubs (extern "C" for Python ctypes access)
// ---------------------------------------------------------------------------

extern "C" {

// bench1: returns pointer to cooperative kernel functions (by name).
// Actual launch is done from the Python runner via subprocess / cuLaunchCooperativeKernel.
// These are just function pointers for the runner to locate via dlopen.

void* get_bench1a_fn() { return (void*)bench1a_gridsync_idle; }
void* get_bench1b_fn() { return (void*)bench1b_gridsync_wmma_active; }
void* get_bench1c_fn() { return (void*)bench1c_gridsync_cpasync_active; }

// bench7 noop kernel pointer
void* get_bench7_fn() { return (void*)bench7_noop_kernel; }

}  // extern "C"
