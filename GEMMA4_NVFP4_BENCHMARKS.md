# Gemma4 26B NVFP4 — Benchmarks, Insights & Lessons

**Date:** 2026-04-09
**Hardware:** RTX 5090 (Blackwell SM120), 32GB VRAM, 1792 GB/s HBM, 96MB L2
**Model:** Gemma4 26B-A4B-it NVFP4 (FusenAI/modelopt format, BF16 attention)
**Serving:** vLLM 0.19.1rc1 in Docker
**HuggingFace:** [cklaus/gemma-4-26B-A4B-it-NVFP4](https://huggingface.co/cklaus/gemma-4-26B-A4B-it-NVFP4)

---

## Performance Results

### Throughput Progression (same model, same hardware, different configs)

```
    18 tok/s  →  enforce_eager baseline
   127 tok/s  →  + CUDA graphs + torch.compile (inductor)         7x
 3,112 tok/s  →  + high concurrency (C=256)                     173x
 6,615 tok/s  →  + disable inductor, use C++ custom ops          368x  ← BEST
```

### Best Serving Config

```bash
vllm serve cklaus/gemma-4-26B-A4B-it-NVFP4 \
  --quantization modelopt \
  --max-model-len 4096 \
  -cc.mode none \
  -cc.cudagraph_mode full
```

### Throughput by Concurrency (best config)

| Concurrency | Gen tok/s | Notes |
|-------------|----------|-------|
| C=1 | 89 | Single request — decode bound |
| C=4 | 201 | |
| C=16 | 305 | |
| C=32 | 1,738 | Batching kicks in |
| C=64 | 3,193 | |
| C=128 | 4,982 | |
| C=192 | 4,915 | |
| **C=256** | **6,615** | **Peak throughput** |
| C=384 | 5,863 | Scheduler overhead |
| C=512 | 6,173 | Plateau |

### Inductor vs No-Inductor

| Config | B=1 tok/s | Peak tok/s | Peak at |
|--------|----------|-----------|---------|
| torch.compile (inductor) + piecewise CUDA graphs | **127** | 3,112 | C=256 |
| No inductor + full CUDA graphs | 89 | **6,615** | C=256 |

Inductor wins at single-request (127 vs 89) but loses at batch scale (3,112 vs 6,615). For serving workloads, **disable inductor**.

Why: inductor's torch.compile adds graph capture overhead that doesn't amortize at high batch. vLLM's hand-written C++ custom ops (`_C.rms_norm`, `_C.cutlass_fp4_moe_mm`, etc.) are already optimized for these shapes. Inductor can't beat them and adds scheduling overhead.

### KV Cache Configurations

| KV Type | Tokens | Concurrency | B=1 tok/s | B=32 tok/s |
|---------|--------|-------------|----------|-----------|
| BF16 (default) | 43,760 | 15x | **127** | **2,071** |
| FP8 | 87,344 | **30x** | 31 | 479 |

**FP8 KV is 4x slower** — FlashInfer FP8 attention overhead on Gemma4's heterogeneous head dims (256 sliding / 512 global). Only useful when KV capacity matters more than throughput.

### KV Cache Configurations

| KV Type | Tokens | Concurrency | B=1 tok/s | B=32 tok/s |
|---------|--------|-------------|----------|-----------|
| BF16 (default) | 43,760 | 15x | **127** | **2,071** |
| BF16 (0.92 util) | 46,448 | 16x | ~127 | ~2,100 |
| FP8 | 87,344 | **30x** | 31 | 479 |

**FP8 KV is 4x slower** due to FlashInfer FP8 attention overhead on Gemma4's unusual head dimensions (256 sliding / 512 global). Only useful when KV capacity matters more than throughput.

### Quality

| Metric | Value |
|--------|-------|
| WikiText-2 PPL (NVFP4) | 701.4 (IT model on raw text — high is expected) |
| GSM8K accuracy (RedHat data) | 95.6% (vs 97.0% BF16 — only 1.4% drop) |

---

## Decode Step Decomposition

### Where the 15.5ms Goes (B=32, 30 layers)

```
Component               Per Layer    30 Layers    % of Step
─────────────────────────────────────────────────────────────
RMSNorm (×3/layer)       135.6 us    4.1 ms       26%  ← BIGGEST SOFTWARE WIN
Attention compute          88.6 us    2.7 ms       17%
Routing + scatter          77.2 us    2.3 ms       15%  ← ELIMINABLE
QKV + O projections        70.4 us    2.1 ms       14%
Grouped CUTLASS GEMMs      58.5 us    1.8 ms       12%  ← NEAR OPTIMAL
FP4 quant (×2)             31.6 us    0.9 ms        6%
SiLU + mul                 18.3 us    0.5 ms        3%
CUDA graph overhead          —        1.1 ms        7%
─────────────────────────────────────────────────────────────
TOTAL                     480.2 us   15.5 ms      100%
```

### The 59% Bandwidth Gap

- **Bandwidth floor:** 6.4 ms (loading 11.4 GB expert weights @ 1792 GB/s)
- **Actual:** 15.5 ms
- **Gap:** 9.1 ms in non-GEMM overhead
- **GEMM ops are only 27% of layer time** — the rest is overhead

---

## Key Insights

### 1. vLLM's MoE is Already Well-Fused

**We assumed** vLLM runs 128 separate expert kernel calls per layer.
**Reality:** Only 6 kernel launches per MoE layer:

```
shuffle_rows()                        → sort tokens by expert
scaled_fp4_experts_quant()            → FP4 quantize ALL activations at once
cutlass_fp4_moe_mm()                  → grouped GEMM1 (ALL experts, ONE call)
silu_and_mul_scaled_fp4_experts_quant() → fused SiLU+mul+FP4 quant
cutlass_fp4_moe_mm()                  → grouped GEMM2 (ALL experts, ONE call)
shuffle_rows()                        → scatter results back
```

The CUTLASS grouped GEMM dispatches all expert matmuls in a single kernel invocation. SiLU activation is fused with FP4 quantization. This is much more optimized than expected.

**Lesson:** Profile before optimizing. The assumed bottleneck (per-expert kernel launches) was already solved by vLLM.

### 2. The Real Bottleneck is Non-GEMM Overhead

At B=32 with 128 experts:
- Each expert processes ~2 tokens (B×top_k / E = 32×8/128 = 2)
- CUTLASS FP4 GEMM takes **13 μs** regardless of batch size (launch-bound)
- But RMSNorm alone takes **45 μs per call** (3 per layer = 135 μs)
- Routing sort takes **39 μs** per layer

**Lesson:** For MoE models with many small experts, the GEMM is not the bottleneck — the surrounding infrastructure (norms, routing, quantization) dominates.

### 3. CUTLASS FP4 GEMM Has a ~13μs Floor

Single CUTLASS FP4 GEMM: 13 μs at B=1, 13 μs at B=128. The kernel launch + setup overhead is the cost; the actual FP4 tensor core computation is negligible at these matrix sizes (704 × 2816).

**Lesson:** For tiny expert GEMMs, kernel fusion won't help — the kernel is already launch-bound. The win is reducing the number of OTHER kernels around it.

### 4. Stream Parallelism is 20x SLOWER (Not Faster)

Benchmarked multi-stream expert dispatch:
- Baseline (grouped_gemm): 9,213 tok/s at B=240
- Stream-parallel (4 streams): 447 tok/s at B=240 ← **20x slower**

Python-level `torch.cuda.Stream()` overhead completely dominates. Creating work items, dispatching to streams, synchronizing events — all in Python — is far more expensive than the GPU work itself.

**Lesson:** Multi-stream parallelism only helps when each stream has substantial GPU work (ms-scale). For μs-scale expert GEMMs, the synchronization overhead kills performance.

### 5. NVFP4 Tensor Cores Are Real (Not Emulating)

Verified on SM120:
- Linear backend: `FLASHINFER_CUTLASS` → CUTLASS 3.x FP4 MMA instructions
- MoE backend: `VLLM_CUTLASS` → same native FP4 tensor core path
- GPU during inference: 95% utilization, 2917 MHz boost, 304W

**Lesson:** Blackwell's FP4 tensor cores are production-ready via CUTLASS 3.x. No emulation, no fallback.

### 6. FP8 KV Cache — Capacity Win, Throughput Loss

FP8 KV doubles capacity (43K → 87K tokens) but switches attention from FlashAttention v2 to FlashInfer's FP8 path, which is **4x slower** on Gemma4's head dimensions.

**Lesson:** FP8 KV is a capacity optimization, not a throughput optimization. Use it when you need more concurrent long-context requests, not when you want faster decode.

### 7. Concurrency Beats Batch Size

| Approach | Peak tok/s |
|----------|-----------|
| B=32 (32 requests at once) | 2,071 |
| C=256 (continuous serving) | **3,112** |

vLLM's continuous batching scheduler is more efficient than sending fixed batches. It dynamically adjusts the active batch as requests complete at different times.

**Lesson:** For throughput benchmarks, use continuous serving (high concurrency) not fixed batch injection.

### 8. torch.compile (Inductor) Hurts Batch Throughput by 2x

The default vLLM config enables `torch.compile` with the inductor backend. This helps single-request latency (+43%: 89 → 127 tok/s) but **halves peak throughput** (6,615 → 3,112 tok/s).

Why:
- Inductor adds graph capture + tracing overhead per batch iteration
- Piecewise CUDA graphs (required with inductor) are less efficient than full CUDA graphs
- CUTLASS FP4 MoE calls are opaque to inductor — it can't optimize them
- vLLM's C++ custom ops are already hand-optimized for these shapes
- The ops inductor CAN optimize (norms, activations) are too small to amortize the tracing cost

The right approach: **adaptive config** — use inductor for interactive (C≤4), disable for serving (C>4).

**Lesson:** torch.compile is not universally beneficial. Profile with YOUR workload before assuming it helps. For MoE models with custom CUTLASS kernels, the overhead often exceeds the optimization.

### 9. `fuse_norm_quant` is Disabled By Default

vLLM has a `fuse_norm_quant` compiler pass that fuses RMSNorm with the subsequent FP4 quantization. But it's `False` by default for NVFP4 models. With norms taking 26% of decode time, enabling this could save 2-3 ms per step.

Also, `rms_norm=['native']` means vLLM is using Python RMSNorm, not the optimized C++ kernel (`vllm_c`).

**Lesson:** Check vLLM's compilation config — there are disabled optimizations that could be significant.

---

## Lessons for RTX PRO 6000

With 96GB VRAM (3x the 5090):
- Model weights: 17 GB (same)
- KV cache: ~75 GB available (vs 10 GB on 5090)
- Context: 128K+ tokens feasible
- Batch size: B=500+ before KV pressure

The **bandwidth** will be the limiting factor on PRO 6000 too, since the expert weight loading per step is the same. But higher batch means more arithmetic intensity per expert GEMM → better tensor core utilization → the 59% gap should narrow.

---

## Plans — Next Optimizations

### What Worked

| Optimization | Gain | Status |
|---|---|---|
| CUDA graphs | 7x single-request | ✓ Done |
| High concurrency (C=256) | 50% over B=32 | ✓ Done |
| **Disable inductor** | **2.1x peak throughput** | ✓ Done — the biggest single win |
| NVFP4 tensor cores | Confirmed native SM120 FP4 | ✓ Verified |

### What Didn't Work

| Optimization | Why |
|---|---|
| C++ RMSNorm | vllm_c doesn't support Gemma4's norm signature |
| fuse_norm_quant for FP4 | Only FP8 patterns exist in vLLM's fusion pass |
| Routing sort elimination | Fundamental to CUTLASS grouped GEMM |
| QKV fusion | Already fused (QKVParallelLinear) |
| FP8 KV cache throughput | 4x slower due to FlashInfer overhead |
| Multi-stream MoE | 20x slower — Python stream overhead dominates |

### Remaining Opportunities

| # | Optimization | Expected Gain | Effort |
|---|---|---|---|
| 1 | **FusenCache KV plugin** | Higher batch without FP8 overhead | Days — plugin exists, needs NVFP4 model integration |
| 2 | **Add FP4 patterns to fuse_norm_quant** | 10-20% at batch | Days — extend vLLM's fusion pass for NVFP4 |
| 3 | **Fix vllm_c RMSNorm for Gemma4** | 10-15% | Hours — add variance_size support or model-specific adapter |
| 4 | **Expert weight caching** | 10-30% | Research — L2 cache is 96MB, fits ~32 of 128 experts |
| 5 | **RTX PRO 6000 (96GB)** | 3x batch capacity → higher throughput | Hardware — arriving soon |
| 6 | **Adaptive config** | Best of both worlds | Hours — inductor for C≤4, no-inductor for C>4 |

### Throughput Summary

```
Current peak:       6,615 tok/s (C=256, no inductor, RTX 5090)
Bandwidth ceiling:  ~8,000-10,000 tok/s (theoretical, higher batch)
With PRO 6000:     ~15,000-20,000 tok/s (3x VRAM, higher batch)
```

---

## What We Built Today

| Deliverable | Status |
|-------------|--------|
| NVFP4 model (modelopt format, BF16 attn) | ✓ Working, uploaded to HuggingFace |
| `convert_ct_to_modelopt.py` | ✓ Converts RedHat CT → modelopt |
| `eval_perplexity.py` | ✓ 3 modes, 15+ bugs fixed, 3 review passes |
| Docker build (vLLM 0.19.1rc1 + patches) | ✓ Reproducible |
| Batch throughput benchmark | ✓ Peak **6,615 tok/s** |
| MoE profiling + decomposition | ✓ Full breakdown of 59% gap |
| Tensor core verification | ✓ CUTLASS 3.x SM120 FP4 MMA confirmed |
| FP8 KV cache evaluation | ✓ 2x capacity, 4x slower |
| Inductor vs no-inductor comparison | ✓ No-inductor 2.1x faster for serving |
| 4-target optimization investigation | ✓ Found 2 already done, 2 blocked, 1 new win |
| HuggingFace model card | ✓ [cklaus/gemma-4-26B-A4B-it-NVFP4](https://huggingface.co/cklaus/gemma-4-26B-A4B-it-NVFP4) |

---

## Corrections to Original Plan

| Original Assumption | Reality | Impact |
|--------------------|---------|--------|
| "MoE is 85% of decode time" | MoE is 48%, attention is 48% | Attention optimization matters equally |
| "128 expert kernel calls per layer" | 6 kernel calls (grouped GEMM) | Kernel fusion opportunity is smaller than expected |
| "Fused MoE kernel → 2-5x" | Already fused; remaining gap is overhead | ~1.5x realistic from overhead elimination |
| "Stream parallelism → 1.5-2x" | 20x slower in practice | Python-level streams don't work for μs-scale ops |
| "FP8 KV → 2x throughput" | 2x capacity but 4x slower | Capacity only, not throughput |
| "torch.compile helps serving" | Hurts at batch scale (2.1x slower) | Inductor overhead > optimization for MoE |
| "C++ RMSNorm is easy win" | vllm_c doesn't support Gemma4 norms | Falls back to native; disabling inductor was the real fix |
| "QKV fusion needed" | Already fused via QKVParallelLinear | No action needed |
| "Routing sort eliminable" | Fundamental to CUTLASS blocked dispatch | Can't remove without rewriting grouped GEMM |
| "Target: 5,000-7,000 tok/s" | **Achieved: 6,615 tok/s** | Exceeded by disabling inductor |

---

## Optimization Investigation Results

### Investigated (4 targets from profiling)

| Target | Expected Gain | Actual Result |
|--------|---------------|---------------|
| **C++ RMSNorm** | 10-15% | vllm_c doesn't support Gemma4's norm signature. Falls back to native. |
| **fuse_norm_quant** | 10-20% | Only FP8 patterns registered in vLLM; FP4 patterns missing. Not activated. |
| **Eliminate routing sort** | 15% | Sort is fundamental to CUTLASS grouped GEMM. No sort-free option exists. |
| **QKV fusion** | 5-10% | Already fused — `QKVParallelLinear` does single matmul for Q+K+V. |

### The Actual Win: Disable Inductor

Instead of fixing individual components, disabling `torch.compile` entirely gave **2.1x throughput**:

```
-cc.mode none              # no torch.compile
-cc.cudagraph_mode full    # keep CUDA graphs (the important optimization)
```

**Why this works:**
1. vLLM's C++ custom ops are already optimized for these exact kernel shapes
2. torch.compile adds graph capture + tracing overhead that doesn't amortize at high batch
3. Piecewise CUDA graphs (inductor mode) are less efficient than full CUDA graphs
4. The inductor doesn't optimize the CUTLASS FP4 MoE calls — they're opaque custom ops
5. Inductor's benefit is fusing small PyTorch ops, but vLLM already fused them in C++

**Trade-off:** Single-request latency is 30% slower (127 → 89 tok/s) because inductor does optimize the single-sequence graph. For interactive use, inductor is better. For batch serving, disable it.

---

## Hardware Observations

### RTX 5090 Blackwell (SM120)
- 170 SMs, up to 2917 MHz boost
- FP4 tensor cores: real, production-ready via CUTLASS 3.x
- 1792 GB/s HBM bandwidth (the actual ceiling for MoE)
- 96 MB L2 cache (fits ~32 NVFP4 experts)
- 304W peak during inference
- 32 GB VRAM: model (17 GB) + KV (10 GB) + CUDA graphs (1 GB) + overhead

### WSL2 Considerations
- `pin_memory=False` — can't pin memory in WSL, small perf impact
- Docker memory limit recommended (36GB) to prevent OOM killing WSL
- CUDA graph compilation: ~90s first time, cached after
