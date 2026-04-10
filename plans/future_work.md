# Future Work: Data-Driven Kernel System

**Date:** April 9, 2026
**Context:** After building the KV cache kernel system (15 experiments), vLLM plugin, MoE benchmark harness, and profiling the full Gemma4 26B-A4B decode path on RTX 5090.

---

## Current State

**What we control today:** 9% of decode time (attention decode + KV store)
**What we could control:** 78% (attention + MoE dispatch + expert compute)
**Hardware-fixed floor:** 22% (tensor core silicon operations)

**Validated results:**
- KV cache: 3.8x compression, 13x store speedup, 2.16x split-K improvement
- Throughput: 4,500 tok/s at B=240 (up from ~2,900 with BF16 KV)
- Plugin: registered with vLLM, 25/25 tests pass, CUDA graph safe
- MoE baseline: 8,969 tok/s at B=240 (grouped GEMM), Python-level streams 20x slower

**Unvalidated:** Zero real model prompts generated. All quality numbers are on random tensors.

---

## Tier 1: Validated, High Confidence (we have data)

| # | Work Item | Impact | Effort | Status |
|---|-----------|--------|--------|--------|
| 1 | **Real model validation** — load Neural Ice, generate text, run perplexity | Validates or invalidates everything | Minutes | Blocked on vLLM build |
| 2 | **Fused Triton MoE kernel** — single kernel for expert dispatch | 2-5x on 85% of decode (35ms→7-17ms) | Hours | Spec + bench done, kernel TODO |
| 3 | **NVFP4 native tensor cores** — `torch._scaled_mm` with FP4 on SM120 | 1.5-2x on every matmul | Hours | Requires SM120 FP4 path verification |
| 4 | **vLLM end-to-end serving benchmark** — real tok/s under load | Ground truth for all claims | Minutes once build finishes | Blocked on build |
| 5 | **CUDA graph full-model capture** — capture 30-layer decode as one graph | Eliminates ~750μs Python/launch overhead per step | Hours (vLLM integration) | Kernel verified graph-safe |

### Why Tier 1

These items have direct evidence from our experiments. The MoE benchmark (Experiment 14) proved that 85% of time is in expert matmuls with high launch overhead. The CUDA graph test (Experiment 13) proved capture works. The decode simulator (Experiment 3) proved compression enables higher batch. The only missing piece is real model validation.

---

## Tier 2: Strong Thesis, Needs Validation

| # | Work Item | Impact | Effort | Why Uncertain |
|---|-----------|--------|--------|---------------|
| 6 | **Expert weight prefetch** — prefetch expert[i+1] while computing expert[i] | 1.1-1.3x on MoE | Hours | Depends on L2 cache hit rate — 128 experts × 2.4MB each = 307MB, L2 is ~48MB on 5090. Only ~15% of experts fit in L2 simultaneously. |
| 7 | **Per-layer KV spec** — k8v4 for sliding, k4v4 for global, natively in vLLM | +9% throughput (from mixed-spec Experiment 5) | Days (upstream PR) | Requires vLLM core changes to cache allocator |
| 8 | **CUTLASS grouped GEMM** — replace cuBLAS for MoE expert dispatch | Potentially faster than Triton MoE for large batch | Days | CUTLASS 3.x SM120 support is new, may have edge cases |
| 9 | **Multi-GPU tensor parallelism** — verify plugin with TP=2/4 on PRO 6000 | 2x capacity, ~1.8x throughput | Hours to verify, days to fix | Attention kernel is per-head (TP-safe), but store/decode need shard-aware block tables |
| 10 | **FP8 kernel path** — cast-based dequant for fp8_e5m2/fp8_e4m3 KV cache | 2x compression with near-zero quality loss | Hours | Spec system ready, kernel just needs a `tl.float8e5m2` cast branch |

### Why Tier 2

Each has a clear mechanism for improvement but an unresolved variable. Expert prefetch depends on cache geometry. Per-layer spec depends on vLLM accepting the upstream PR. CUTLASS depends on SM120 maturity. These become Tier 1 after one focused experiment each.

---

## Tier 3: Research-Grade, Uncertain Payoff

| # | Work Item | Impact | Effort | Why Uncertain |
|---|-----------|--------|--------|---------------|
| 11 | **2-bit V with outlier protection** — reserve high-precision slots for outlier V vectors | 4.6x compression at 0.97+ quality (vs current 0.91) | Days | Need to study real activation distributions — outlier frequency unknown per model |
| 12 | **Learned quantization offsets** — per-head calibrated offsets instead of fixed 7.5/127.5 | +0.01-0.02 cosine at same bit width | Days | Requires calibration dataset + training loop. Marginal expected gain. |
| 13 | **Mixed-precision within head** — 8-bit for high-variance dims, 4-bit for rest | Better quality at same average compression | Days | Spec system redesign needed. Dim-level variance analysis required per model. |
| 14 | **Persistent kernel for sliding layers** — single kernel resident across 25 layers | Eliminates 25 kernel launches per step | Days | Triton doesn't support persistent kernels natively. Needs raw CUDA or cooperative groups. |
| 15 | **Speculative decoding integration** — multi-token query support in decode kernel | 1.5-2x single-user throughput | Hours (kernel change small) | vLLM orchestrates spec decode — integration complexity is in the scheduler, not the kernel |
| 16 | **Online quality monitoring** — canary cosine check during serving | Catches quality degradation on distribution shift | Hours | Value depends on how much quality varies across real data vs our random-tensor benchmarks |
| 17 | **Adaptive spec at runtime** — switch k8v4↔k4v4 based on current sequence length | Best of both worlds (quality at short, throughput at long) | Days | Requires mid-sequence KV cache format conversion, which is expensive |

### Why Tier 3

Each has an interesting hypothesis but no supporting data. The outlier-protected 2-bit V is compelling in theory but needs real activation profiling. Learned offsets might improve quality by 1% or by 0.01% — we don't know until we calibrate. These are experiments worth running but not worth building infrastructure for until Tier 1-2 are done.

---

## Tier 4: Infrastructure / Ecosystem

| # | Work Item | Impact | Effort | Notes |
|---|-----------|--------|--------|-------|
| 18 | **Upstream vLLM PR: `register_kv_cache_dtype()`** | Eliminates monkey-patching in plugin.py | Hours (small PR) | Benefits all plugin developers. Our monkey-patch is fragile across vLLM versions. |
| 19 | **Upstream vLLM PR: per-layer cache allocator** | Fixes Gemma4 memory waste (allocates D=512 for all 30 layers) | Days (core change) | Enables native mixed-spec. Currently `disable_hybrid_kv_cache_manager=True` wastes ~45% of KV memory. |
| 20 | **RTX PRO 6000 benchmark suite** — re-run all sweeps on 96GB | Updated thresholds for adaptive selector | Hours | Split-K table changes with more SMs. Batch limits change with 3x VRAM. Compression vs quality trade-off shifts. |
| 21 | **SGLang plugin port** — same kernel, different backend ABC | Second serving framework supported | Days | SGLang has different attention API but same concepts. `kv_cache_gen/` is framework-agnostic. |
| 22 | **Model zoo validation** — test on LLaMA, Qwen, Mixtral, not just Gemma4 | Proves generality of spec-driven approach | Days | Different head dims (64, 128, 256, 512), GQA ratios, MoE vs dense, different quant formats |
| 23 | **Precompiled .cubin distribution** — ship compiled Triton kernels in pip package | Zero cold-start for common GPU/spec combos | Hours | Currently 4.9s precompile. Would need builds for SM80/86/89/90/120 × all specs. |
| 24 | **Benchmark dashboard** — automated nightly perf tracking | Catch regressions, track improvements | Days | CI/CD with GPU runner. Store results in SQLite/TSV. Plot trends. |
| 25 | **Documentation + examples** — README, API docs, quickstart guide | Adoption beyond us | Hours | Currently only RESULTS.md. Need: quickstart, API reference, model compatibility table. |

---

## Tier 5: Long-Term / Speculative

| # | Work Item | Impact | Effort | Notes |
|---|-----------|--------|--------|-------|
| 26 | **AutoKernel MoE search** — automated sweep of MoE execution configs | Finds optimal strategy per model per GPU automatically | Days | The 5,184-config search from the vLLM data-driven plan. Now feasible with our sweep infrastructure. |
| 27 | **Distillation-aware KV cache** — train the model to be robust to KV quantization | Near-lossless quality at 4-bit KV | Weeks | Requires fine-tuning loop. Modifies the model, not just the kernel. |
| 28 | **Cross-layer KV sharing** — reuse KV cache across similar layers | Reduces total KV memory by 30-50% | Weeks (research) | Gemma4's 25 sliding layers have similar attention patterns — could share KV across groups of 5. |
| 29 | **Sparse attention kernel** — skip low-attention KV entries during decode | 2-3x decode speedup at long context | Weeks (research) | Needs dynamic sparsity mask. Computing "which KV entries to skip" may cost more than the attention itself. |
| 30 | **Full inference compiler** — generate entire model forward pass as one kernel graph | Eliminates ALL Python overhead | Months | This is what `torch.compile` aspires to. Not our problem to solve — wait for PyTorch/Triton to mature. |

---

## The Priority Order

```
NOW:        #1  Real model validation
            #4  Serving benchmark
            
NEXT:       #2  Fused Triton MoE kernel (the 85%)
            #3  NVFP4 native tensor cores

SOON:       #5  CUDA graph capture (vLLM integration)
            #10 FP8 kernel path
            #9  Multi-GPU for PRO 6000

LATER:      #6-8   Expert prefetch, per-layer spec, CUTLASS grouped GEMM
            #18-19 Upstream vLLM PRs
            #20-25 Infrastructure and ecosystem

SOMEDAY:    #11-17 Research-grade optimizations
            #26-29 Long-term research

NEVER:      #30    Full inference compiler (not our scope)
```

---

## Throughput Projections

Based on measured data from Experiments 1-15 and the MoE benchmark:

### RTX 5090 (32GB)

| Configuration | B=128 tok/s | B=240 tok/s | Confidence |
|--------------|------------|------------|------------|
| Current (BF16 KV, cuBLAS MoE) | ~2,900 | OOM | Measured |
| + k4v4 compression (today) | ~3,300 | ~4,500 | Measured |
| + Fused MoE kernel (#2) | ~6,000-9,000 | ~8,000-12,000 | Estimated from launch overhead analysis |
| + NVFP4 tensor cores (#3) | ~9,000-15,000 | ~12,000-20,000 | Estimated from 1.5-2x matmul speedup |
| + CUDA graphs (#5) | +5-10% on top | +5-10% on top | Measured kernel-level, estimated system-level |

### RTX PRO 6000 (96GB)

| Configuration | B=500 tok/s | B=1000 tok/s | Confidence |
|--------------|------------|-------------|------------|
| Current (BF16 KV, cuBLAS MoE) | ~8,000 | OOM | Extrapolated |
| + k8v8 quality mode | ~8,000 | ~12,000 | Estimated (no compression needed at 96GB) |
| + Fused MoE kernel (#2) | ~16,000-24,000 | ~24,000-36,000 | Estimated |
| + NVFP4 tensor cores (#3) | ~24,000-40,000 | ~36,000-60,000 | Speculative |

### What We Control vs Total

```
Optimization                          Our Code    NVIDIA    Combined
─────────────────────────────────────────────────────────────────────
KV compression (done)                 +55%        —         +55%
Fused MoE kernel (#2)                +100-200%    —         +100-200%
NVFP4 native tensor cores (#3)       (API call)  +50-100%  +50-100%
CUDA graphs (#5)                     (config)    +5-10%    +5-10%
─────────────────────────────────────────────────────────────────────
Combined potential                                          3-6x

Current: ~4,500 tok/s (B=240, RTX 5090)
Target:  ~12,000-20,000 tok/s (B=240, RTX 5090, all optimizations)
```

---

## Decision Framework

Before starting any work item, ask:

1. **Is #1 done?** If we haven't validated on real model output, nothing else matters.
2. **Does it affect the 85% (MoE)?** If not, it's optimizing the minority.
3. **Is it a kernel change or Python change?** Kernel changes compound. Python changes get eliminated by CUDA graphs.
4. **Does it work on the PRO 6000?** The target hardware is 96GB, not 32GB. Optimizations that only matter at 32GB (aggressive compression) lose relevance.
5. **Can we measure the improvement?** If we can't benchmark it, we can't know if it worked. Build the bench harness first.
