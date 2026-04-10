# Issue/PR: Default to No-Inductor for NVFP4 MoE Models in Serving

## Type

Feature request / performance bug

## Severity

High — affects any user serving Gemma4, Mixtral, or other NVFP4 MoE models.
Default vLLM config cuts throughput by **2.1x** compared to a two-flag change.

---

## Summary

When serving NVFP4 MoE models, vLLM's default `torch.compile` (inductor
backend) cuts peak throughput by 2.1x compared to disabling inductor while
keeping CUDA graphs. The fix is two flags:

```bash
vllm serve <model> \
  -cc.mode none \
  -cc.cudagraph_mode full
```

For serving workloads (concurrency ≥ 4), this is always better. vLLM should
either make this the default for NVFP4/MoE models, or document it prominently
in the serving guide and add a warning when inductor is enabled with these
model classes.

---

## Benchmark Data

**Hardware:** RTX 5090 (Blackwell SM120, 32 GB, 1792 GB/s)
**Model:** Gemma4 26B-A4B-it NVFP4 (128 experts, top-8, 30 layers)
**vLLM:** 0.19.1rc1
**Quantization:** modelopt NVFP4

### Throughput by Concurrency

| Concurrency | Inductor (default) tok/s | No Inductor tok/s | Ratio |
|-------------|--------------------------|-------------------|-------|
| C=1 | **127** | 89 | 0.70x (inductor wins) |
| C=4 | ~201 | ~175 | ~0.87x |
| C=16 | ~400 | ~380 | ~0.95x |
| C=32 | ~900 | ~1,738 | 1.93x |
| C=64 | ~1,600 | ~3,193 | 2.00x |
| C=128 | ~2,400 | ~4,982 | 2.08x |
| C=192 | ~2,500 | ~4,915 | 1.97x |
| **C=256** | **3,112** | **6,615** | **2.12x** |

### Summary Table

| Config | Single-request (B=1) | Peak throughput | Peak concurrency |
|--------|---------------------|-----------------|------------------|
| Default (`-cc.mode reduce-overhead`) | **127 tok/s** | 3,112 tok/s | C=256 |
| No inductor (`-cc.mode none -cc.cudagraph_mode full`) | 89 tok/s | **6,615 tok/s** | C=256 |

Inductor helps single-request latency (+43%) but **halves** peak throughput
for any serving workload with more than ~4 concurrent requests.

The crossover is around C=8-16: above that, no-inductor is always better.

---

## Root Cause Analysis

### Why inductor hurts MoE serving

**1. Piecewise CUDA graphs vs. full CUDA graphs**

Inductor mode requires piecewise CUDA graph capture — the graph is broken at
custom op boundaries. For each decode step, multiple partial graphs are
launched and joined. Full CUDA graphs (possible only without inductor) capture
the entire decode step as one graph, reducing launch overhead significantly.

**2. Custom ops are opaque to inductor**

The dominant compute in NVFP4 MoE is:
- `cutlass_fp4_moe_mm` — CUTLASS grouped GEMM for all experts in one call
- `scaled_fp4_experts_quant` — FP4 quantization of all expert activations
- `silu_and_mul_scaled_fp4_experts_quant` — fused SiLU + FP4 quant

These are hand-written CUDA/CUTLASS C++ ops registered as custom ops. Inductor
cannot inspect or optimize them. It only optimizes the thin Python/ATen layer
around them, which is not where the time is spent.

**3. The ops inductor CAN optimize are not the bottleneck**

From profiling at B=32:

```
Component               Per Layer    30 Layers    % of Step
─────────────────────────────────────────────────────────────
RMSNorm (×3/layer)       135.6 us    4.1 ms       26%
Attention compute          88.6 us    2.7 ms       17%
Routing + scatter          77.2 us    2.3 ms       15%
QKV + O projections        70.4 us    2.1 ms       14%
Grouped CUTLASS GEMMs      58.5 us    1.8 ms       12%
FP4 quant (×2)             31.6 us    0.9 ms        6%
SiLU + mul                 18.3 us    0.5 ms        3%
CUDA graph overhead          —        1.1 ms        7%
─────────────────────────────────────────────────────────────
TOTAL                     480.2 us   15.5 ms      100%
```

Inductor might fuse some of the smaller elementwise ops (SiLU, norms). But
those are 3-6% of step time. The overhead inductor adds (graph tracing,
piecewise graph boundaries) exceeds what it saves.

**4. vLLM already hand-fused the critical ops in C++**

vLLM's MoE path uses only 6 kernel launches per layer (not 128):
```
shuffle_rows()
scaled_fp4_experts_quant()
cutlass_fp4_moe_mm()           # ALL experts, one call
silu_and_mul_scaled_fp4_experts_quant()
cutlass_fp4_moe_mm()           # ALL experts, one call
shuffle_rows()
```

The fusion that inductor would do is already done. There is nothing left for
inductor to optimize in the hot path.

---

## Proposed Fix

### Option A (preferred): Auto-detect and default to no-inductor for NVFP4/MoE

In `vllm/compilation/compile_context.py` or the model runner, detect when:
- quantization is `modelopt` or `compressed-tensors` with NVFP4
- OR the model has MoE layers (`config.num_experts > 1`)

And set:
```python
compilation_config.level = CompilationLevel.PIECEWISE  # CUDA graphs only
compilation_config.cudagraph_mode = CudaGraphMode.FULL
```

This gives peak throughput with zero user-facing config change.

### Option B: Document and warn

Add to vLLM serving docs:

> **MoE and NVFP4 models:** `torch.compile` (inductor backend) hurts
> throughput by up to 2x for MoE models with NVFP4 quantization. Use
> `-cc.mode none -cc.cudagraph_mode full` for serving workloads.

Add a startup warning when an NVFP4/MoE model is detected with
`compilation_config.level == CompilationLevel.OPTIMIZE`:

```
WARNING: torch.compile (inductor) detected with NVFP4/MoE model.
For serving (concurrency > 4), consider: -cc.mode none -cc.cudagraph_mode full
This typically gives 2x higher peak throughput. See: <docs link>
```

### Option C (best of both worlds): Adaptive mode

Use inductor for interactive deployments (the single-request latency is better),
disable for serving. The crossover is at approximately C=8-16 concurrent
requests. A heuristic based on `max_num_seqs` or deployment mode:

```python
if max_num_seqs >= 16 and model_is_moe_or_fp4:
    compilation_config.level = CompilationLevel.PIECEWISE
    compilation_config.cudagraph_mode = CudaGraphMode.FULL
```

---

## Steps to Reproduce

```bash
# With inductor (current default)
vllm serve cklaus/gemma-4-26B-A4B-it-NVFP4 \
  --quantization modelopt \
  --max-model-len 4096

# Benchmark with locust or vllm-benchmark
# Peak: ~3,112 tok/s at C=256

# Without inductor (proposed fix)
vllm serve cklaus/gemma-4-26B-A4B-it-NVFP4 \
  --quantization modelopt \
  --max-model-len 4096 \
  -cc.mode none \
  -cc.cudagraph_mode full

# Peak: ~6,615 tok/s at C=256 (+2.1x)
```

The result is reproducible across 3+ benchmark runs with variance < 2%.

---

## Additional Context

This finding came from profiling Gemma4 26B-A4B-it NVFP4 throughput on
RTX 5090 (SM120 Blackwell). The same principle applies to:

- Any NVFP4 model with CUTLASS custom ops (Gemma4, future NVIDIA-format models)
- Standard FP4/FP8 MoE models where vLLM's grouped GEMM path is active
- Possibly dense NVFP4 models (the CUTLASS FP4 linear path has the same opacity
  to inductor)

The effect is likely hardware-independent. The ratio of inductor benefit
(fusing small elementwise ops) to inductor overhead (piecewise graph boundaries,
tracing) gets worse as the custom op density of the model increases.

**Trade-off to communicate clearly:** Inductor is better for interactive
single-request latency (127 vs 89 tok/s in our tests, +43%). The recommendation
is mode-dependent:
- Interactive / low concurrency (C ≤ 4): keep inductor (default is fine)
- Batch serving / high concurrency (C > 8): disable inductor

---

## Related

- vLLM compilation docs: https://docs.vllm.ai/en/latest/compilation/
- `compilation_config` API: `vllm.config.CompilationConfig`
- `CompilationLevel.PIECEWISE` vs `OPTIMIZE` in `vllm/compilation/levels.py`
- Companion PR: Fused RMSNorm + FP4 quantization kernel (addresses the 26%
  norm overhead that this issue reveals as the next bottleneck)
