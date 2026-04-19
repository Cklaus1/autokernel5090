# Expert Prefetch via L2 Residency — Design Doc

**Date:** 2026-04-17
**Target:** Gemma4 26B NVFP4 MoE, RTX PRO 6000 (SM120a, 128 MB L2)
**Hypothesis:** Prefetch the next MoE layer's top-K likely experts into L2 during the
current layer's MoE GEMM to eliminate cold HBM read latency on expert dispatch.

---

## 1. Why This Is Different From Discovery #6

Discovery #6 (KILL) tried **blanket expert caching**: pin ALL 128 experts × 30 layers
into L2. Working set: 128 × 30 × 2 MB ≈ 7.6 GB — far larger than the 128 MB L2.
Net effect was thrashing: every new expert evicted prior tenants, nothing stuck.

This proposal is **per-step, per-layer, top-K predictive**:

| Quantity                                   | Value       |
|--------------------------------------------|-------------|
| L2 capacity (PRO 6000)                     | 128 MB      |
| Expert weight size (NVFP4, gate_up + down) | ~2.0 MB     |
| Top-8 activated per token per layer        | 16 MB       |
| Top-16 "safety net" (EMA predicted)        | 32 MB       |
| Residual budget for activations + KV       | 96 MB       |

**Working set fits with margin.** Only one layer's active experts live in L2 at a
time. When the layer advances, the old experts are evicted naturally; we explicitly
re-prefetch the new top-K before compute.

## 2. Prefetch Strategy

Two cooperating elements:

### 2a. EMA-based expert predictor
Per-layer running exponential moving average of activation frequency:
```
freq_ema[layer, expert] = 0.9 * freq_ema + 0.1 * (expert in topk_ids this step)
```
Top-16 of the EMA = predicted prefetch set. Jaccard across layers is 0.14
(Discovery #17), so cross-layer prediction is weak — but **within-layer across
steps** (same request, stable token distribution) it's far higher. The 8 extra
slots (16 vs 8) are a safety margin for the 40-60% prediction miss rate.

### 2b. Overlapped prefetch on second stream
- Stream A (default): current layer's MoE compute (shuffle → quant → grouped GEMM).
- Stream B (prefetch): `cudaMemPrefetchAsync` or warmup-kernel issuing
  speculative reads on predicted top-16 of layer N+1.

Synchronization: no explicit sync. We rely on L2 staying warm between Stream B's
warmup kernel and Stream A's subsequent MoE dispatch (a few hundred μs later).

### 2c. WSL2 fallback
`cudaMemPrefetchAsync` requires Unified Memory (`cudaMallocManaged`) and often
silently no-ops on WSL2. Our fallback (and primary on WSL2) is a **warmup kernel**
that issues sparse reads across each predicted expert's weight tensor — enough
sectors to pull into L2 without doing real compute. Cost: ~50 μs for 32 MB at
1.8 TB/s effective HBM BW, well hidden behind the ~200 μs MoE GEMM.

## 3. L2 Residency Verification

Profiler metric (ncu):
```
lts__t_sectors_srcnode_gpc_aperture_device_op_read_hit_rate
```
Directly measures L2 hit rate on device-side reads. Secondary signals:
- `dram__bytes_read.sum` — drop if L2 absorbs reads.
- `smsp__average_warps_issue_stalled_long_scoreboard_per_active_cycle` — stalls
  on memory drop.

Without ncu available, the microbench below uses CUDA events to measure
**apparent HBM bandwidth consumed** by the 2nd GEMM pass: if pre-warmed, the
observed GEMM latency drops toward compute-bound (no HBM weight load), which is
an indirect but quantitative L2 hit-rate proxy.

## 4. Expected Gain

Recompute from `MOE_PROFILING.md`:

| Decode step component           | Per 30 layers |
|---------------------------------|---------------|
| Grouped GEMMs (×2)              | 1.8 ms        |
| FP4 quant (×2)                  | 0.9 ms        |
| Routing + scatter               | 2.3 ms        |
| Total MoE-related               | **5.0 ms / 32% of step** |

Of the 1.8 ms grouped-GEMM time, weight-load bandwidth is:
- Per layer top-8 experts: 8 × 2 MB = 16 MB load
- 30 layers × 16 MB = 480 MB / 1.8 TB/s = **0.27 ms** of pure HBM weight-load.

If prefetch eliminates this on average (pessimistic: 50% of it due to prediction
misses), we save 0.13-0.27 ms / 15.5 ms = **0.9-1.7% e2e**. On the MoE grouped
GEMM alone, the latency drop is 0.27 / 1.8 = **~15% GEMM-local speedup**,
reducing to ~7% after prediction-miss penalty.

**Success gate (microbench):** ≥3% speedup on an isolated 2-layer MoE-GEMM
microbench. Below that, the overhead of the prefetch stream + warmup kernel
isn't worth it and we KILL.

## 5. Prototype Plan

1. Python + `torch.utils.cpp_extension.load_inline` — build a small CUDA module
   with a warmup kernel that strided-reads an expert weight region.
2. Simulate Gemma4 MoE shape:
   - `K_in=1408, K_up=2816, K_down=704` (per-expert shapes from profiling doc)
   - 128 total experts, 8 active per step, 2 layers
   - NVFP4 packed as int8 for simplicity (half-density FP4 is 2 nibbles/byte)
3. Three variants:
   - **A: Baseline** — cold GEMM of active experts.
   - **B: Prefetch** — warmup-read top-16 predicted experts on Stream B,
     concurrently with layer-N's GEMM, before running layer-N+1's GEMM.
   - **C: Oracle prefetch** — warmup exactly the top-8 that WILL be used (upper bound).
4. Measure: per-layer latency, cross-variant speedup.
5. L2 hit rate: use CUDA runtime `cudaDeviceGetAttribute` for reserved L2 size
   plus indirect proxy via observed latency-drop ratio
   (`1 - (B_latency - compute_floor) / (A_latency - compute_floor)`).

## 6. Kill Criteria

- Prefetch gives <3% on the microbench → KILL; HBM is not bottleneck for this shape.
- Warmup kernel regresses due to contention → try `cudaStreamAttachMemAsync` alternative.
- If Oracle variant C also gives <3%, the hypothesis is fundamentally wrong — kill.

---

## 7. Empirical Results (GPU 1, RTX PRO 6000, 2026-04-17)

Microbench: `kernels/csrc/bench_expert_prefetch.py`. 300 iters × 3 runs,
median taken. L2=128 MiB confirmed via `cudaDeviceProp.L2_cache_size`.

| Mode                           | Latency (μs) | vs baseline |
|--------------------------------|--------------|-------------|
| A. Baseline (no prefetch)      | **62.0**     | 1.00× (ref) |
| B. EMA prefetch top-16         | 125.9        | **0.49×** (slower) |
| C. Oracle prefetch exact top-8 | 95.7         | **0.65×** (slower) |

**Verdict: KILL.**

### Why it fails (confirmed empirically)

Stream-overlap microbench (see `kernels/csrc/test_stream_overlap.py`): two
concurrent HBM-bound copies on separate streams run at ratio 1.08× — i.e.
essentially serialized on the HBM bus. The expected "free" second-stream
prefetch doesn't exist because **the first stream's MoE GEMM is already
saturating HBM BW**. The warmup kernel competes for the same HBM, spending
bandwidth to prefetch bytes that would have been fetched on-demand with the
same latency anyway.

Oracle (variant C) is strictly worse than baseline by 34 μs — prefetch just
moves the HBM traffic earlier, adds its own compute/launch cost, and provides
zero end-to-end saving. The L2 hit is real but doesn't convert to speedup
because layer-1's GEMM is **HBM-bound, not HBM-latency-bound** — it would
have reached full HBM throughput either way.

### Core insight for future ideas

**L2 prefetch only helps when the post-prefetch kernel can become
compute-bound after the L2 hit.** MoE grouped GEMM on NVFP4 with 8 activated
experts × 2 MiB = 16 MiB is a streaming load with ~2-3 FLOPs per byte — too
low an arithmetic intensity to ever go compute-bound even with L2 residency.
The CUTLASS grouped GEMM is already running at memory-roofline.

Prefetch-into-L2 will only pay off for:
- Kernels with **re-used weights across queries** (e.g. KV-cache attention on
  common prefixes — but that's distinct from expert weights).
- Much **smaller experts** (sub-128KB) where latency-not-throughput dominates.
- **Compute-bound** kernels where the HBM read is serial overhead hidden
  behind compute (e.g. activations reused across many tokens).

None of these apply to Gemma4 26B MoE grouped GEMM. **Abandon this direction.**
