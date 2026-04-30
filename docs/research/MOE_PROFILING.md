# MoE Kernel Profiling — Gemma4 26B NVFP4 on RTX 5090

**Date:** 2026-04-09
**Model:** Gemma4 26B-A4B-it NVFP4 (modelopt format, BF16 attention)
**Hardware:** RTX 5090 (Blackwell SM120), 32GB VRAM, 1792 GB/s bandwidth

## Key Finding: vLLM's MoE is Already Well-Fused

The initial plan assumed vLLM runs 128 separate expert kernel calls per layer.
**Reality:** vLLM uses CUTLASS grouped GEMM — only 6 kernel launches per MoE layer:

1. `shuffle_rows` — sort tokens by expert
2. `scaled_fp4_experts_quant` — FP4 quantize all activations at once
3. `cutlass_fp4_moe_mm` — grouped GEMM1 (gate_up) for ALL experts in one call
4. `silu_and_mul_scaled_fp4_experts_quant` — fused SiLU+mul+FP4 quant
5. `cutlass_fp4_moe_mm` — grouped GEMM2 (down) for ALL experts in one call  
6. `shuffle_rows` — scatter results back

## Decode Step Breakdown (B=32, 30 layers)

| Component | Per Layer (μs) | 30 Layers (ms) | % of Step |
|-----------|---------------|----------------|-----------|
| **RMSNorm (×3/layer)** | 135.6 | **4.1** | **26%** |
| **Attention compute** | 88.6 | **2.7** | **17%** |
| **Routing + scatter** | 77.2 | **2.3** | **15%** |
| QKV + O projections | 70.4 | 2.1 | 14% |
| Grouped GEMMs (×2) | 58.5 | 1.8 | 12% |
| FP4 quant (×2) | 31.6 | 0.9 | 6% |
| SiLU + mul | 18.3 | 0.5 | 3% |
| CUDA graph overhead | — | 1.1 | 7% |
| **Total** | **480** | **15.5** | **100%** |

## The 59% Bandwidth Gap

- **Bandwidth floor:** 6.4 ms (loading 11.4 GB weights @ 1792 GB/s)
- **Actual:** 15.5 ms
- **Gap:** 9.1 ms (59%) in non-GEMM overhead

### GEMMs are only 27% of layer time!

The CUTLASS FP4 GEMMs are near-optimal. The overhead is:
- **Norms:** 26% — `fuse_norm_quant` is disabled by default!
- **Attention:** 17% — FlashAttention v2 decode path
- **Sort/scatter:** 15% — token permutation for grouped dispatch

## Optimization Experiments

### Experiment 1: Enable norm_quant fusion
- vLLM has `fuse_norm_quant` pass but it's `False` by default
- Should eliminate separate RMSNorm kernel calls (4.1 ms)
- Expected: save 2-3 ms per step → ~20% throughput gain

### Experiment 2: Increase max-model-len to 8192
- More KV cache headroom → higher effective batch
- Currently B=32 saturates, may push to B=64+

### Experiment 3: gpu-memory-utilization 0.95
- More VRAM for KV cache → more concurrent requests

## Architecture Notes

### CUTLASS FP4 MoE Execution Path
```
run_cutlass_moe_fp4():
  ops.get_cutlass_moe_mm_data()     # Compute expert offsets, problem sizes
  ops.shuffle_rows(a, a_map)         # Sort tokens by expert
  ops.scaled_fp4_experts_quant()     # Quantize activations to FP4
  ops.cutlass_fp4_moe_mm()           # Grouped GEMM1 (ALL experts, one call)
  ops.silu_and_mul_scaled_fp4_experts_quant()  # Fused activation + quant
  ops.cutlass_fp4_moe_mm()           # Grouped GEMM2 (ALL experts, one call)
  ops.shuffle_rows(c3, c_map)        # Scatter results back
  output = (c3 * topk_weights).sum() # Apply routing weights
```

### RTX 5090 Blackwell Specifics
- SM120, 170 SMs, 96 MB L2 cache
- Native FP4 tensor cores via CUTLASS 3.x
- FlashAttention v2 for sliding-window layers
- Triton attention for global (512-dim) layers
- 1792 GB/s HBM bandwidth

### Per-Expert Weight Size
- NVFP4: gate_up [1408, 2816] + down [2816, 704] = 2.97 MB per expert
- 128 experts × 2.97 MB = 380 MB per layer
- L2 cache (96 MB) fits ~32 experts → hot expert caching possible
