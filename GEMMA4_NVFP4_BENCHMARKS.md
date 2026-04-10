# Gemma4 26B NVFP4 — Benchmarks, Insights & Lessons

**Date:** 2026-04-09
**Hardware:** RTX 5090 (Blackwell SM120), 32GB VRAM, 1792 GB/s HBM, 96MB L2
**Model:** Gemma4 26B-A4B-it NVFP4 (FusenAI/modelopt format, BF16 attention)
**Serving:** vLLM 0.19.1rc1 in Docker with CUDA graphs + torch.compile

---

## Performance Results

### Throughput Scaling

| Config | Gen tok/s | Notes |
|--------|----------|-------|
| enforce_eager (B=1) | 18 | Baseline — no CUDA graphs, no compile |
| CUDA graphs (B=1) | 127 | **7x** — launch overhead eliminated |
| CUDA graphs (B=4) | 383 | Linear scaling |
| CUDA graphs (B=16) | 1,221 | GPU starting to saturate |
| CUDA graphs (B=32) | 2,071 | Near saturation |
| CUDA graphs (C=64) | 2,046 | Continuous serving |
| CUDA graphs (C=128) | 2,890 | vLLM scheduler batching efficiently |
| **CUDA graphs (C=256)** | **3,112** | **Peak — ceiling reached** |

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

### 8. `fuse_norm_quant` is Disabled By Default

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

## Plans — Closing the 59% Gap

### Tier 1: Quick Wins (hours)

| # | Optimization | Expected Gain | How |
|---|-------------|---------------|-----|
| A | **Enable vLLM C++ RMSNorm** | 10-15% (save ~2ms) | Set `rms_norm=['vllm_c']` in kernel config, or patch the model to use `from vllm._custom_ops import rms_norm` |
| B | **Enable `fuse_norm_quant`** | 10-20% (save ~2ms) | vLLM compiler pass exists but not activated for NVFP4. Need to modify pass_config or patch the compilation pipeline. |
| C | **Increase concurrency** | Already done | C=256 → 3,112 tok/s (50% over B=32) |

### Tier 2: Medium Effort (days)

| # | Optimization | Expected Gain | How |
|---|-------------|---------------|-----|
| D | **Eliminate routing sort** | 15% (save ~2.3ms) | Replace `shuffle_rows` with index_select or persistent expert dispatch that doesn't need global sorting |
| E | **Fuse QKV projection** | 5-10% (save ~1ms) | BF16 Q+K+V as single matmul [B, H] × [H, Q+K+V] instead of 3 separate calls |
| F | **FusenCache KV plugin** | Higher batch capacity | Custom k8v4/k4v4 without FlashInfer FP8 overhead |

### Tier 3: Research (weeks)

| # | Optimization | Expected Gain | How |
|---|-------------|---------------|-----|
| G | **Expert weight caching** | 10-30% at high batch | Hot experts (top-32 by frequency) stay in L2 cache (96MB). Pre-sort expert weights by access frequency. |
| H | **Persistent MoE kernel** | 15-25% | Single kernel that stays resident across all experts, eliminates inter-expert launch overhead |
| I | **Custom CUTLASS MoE with norm fusion** | 20-30% | Write a single CUTLASS kernel: norm → quant → grouped_GEMM → act+quant → grouped_GEMM → scatter. Eliminates 4 of 6 kernel launches per layer. |

### Projected Throughput After Optimizations

```
Current:            3,112 tok/s (C=256, RTX 5090)
+ C++ RMSNorm (A):  ~3,500 tok/s
+ norm_quant (B):   ~4,000 tok/s
+ No sort (D):      ~4,500 tok/s
+ Fused QKV (E):    ~4,800 tok/s

Theoretical ceiling: ~5,500 tok/s (bandwidth-limited)
  11.4 GB weights / 1792 GB/s = 6.4 ms/step
  B=32 / 6.4ms = 5,000 tok/s (pure bandwidth)
  With higher batch from FusenCache: ~6,000-7,000 tok/s
```

---

## What We Built Today

| Deliverable | Status |
|-------------|--------|
| NVFP4 model (modelopt format, BF16 attn) | ✓ Working, uploaded to HuggingFace |
| `convert_ct_to_modelopt.py` | ✓ Converts RedHat CT → modelopt |
| `eval_perplexity.py` | ✓ 3 modes, 15+ bugs fixed, 3 review passes |
| Docker build (vLLM 0.19.1rc1 + patches) | ✓ Reproducible |
| Batch throughput benchmark | ✓ Peak 3,112 tok/s |
| MoE profiling + decomposition | ✓ Full breakdown of 59% gap |
| Tensor core verification | ✓ CUTLASS 3.x SM120 FP4 MMA confirmed |
| FP8 KV cache evaluation | ✓ 2x capacity, 4x slower |
| HuggingFace model card | ✓ cklaus/gemma-4-26B-A4B-it-NVFP4 |

---

## Corrections to Original Plan

| Original Assumption | Reality | Impact |
|--------------------|---------|--------|
| "MoE is 85% of decode time" | MoE is 48%, attention is 48% | Attention optimization matters equally |
| "128 expert kernel calls per layer" | 6 kernel calls (grouped GEMM) | Kernel fusion opportunity is smaller than expected |
| "Fused MoE kernel → 2-5x" | Already fused; remaining gap is overhead | ~1.5x realistic from overhead elimination |
| "Stream parallelism → 1.5-2x" | 20x slower in practice | Python-level streams don't work for μs-scale ops |
| "FP8 KV → 2x throughput" | 2x capacity but 4x slower | Capacity only, not throughput |
| "Target: 12,000-20,000 tok/s" | Revised: 5,000-7,000 tok/s | Bandwidth ceiling is real at 1792 GB/s |

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
