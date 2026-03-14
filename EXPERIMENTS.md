# AutoKernel Experiments Summary

## Experiment Log

All experiments for the W4A16 quantized matmul kernel on RTX 5090 (Blackwell, 170 SMs, 419 TFLOPS FP16 peak, 1792 GB/s).

| Exp | Tag | TFLOPS | Speedup | What Was Tried | Result |
|-----|-----|--------|---------|---------------|--------|
| 0 | baseline | 15.1 | 0.23x | 32x32x32 blocks, per-element dequant | KEEP (baseline) |
| 21 | autotune_l2swizzle | 136.8 | 2.10x | 22 autotune configs + L2 cache swizzle tile reordering | KEEP (+121.7 TFLOPS) |
| 29 | two_level_k | 143.9 | 2.20x | Two-level K tiling, hoist scale/zero loads outside inner loop | KEEP (+7.1) |
| 32 | persistent_4x | 153.4 | 2.34x | Persistent kernel with 680 programs (4x SMs), outer tile loop | KEEP (+9.5) |
| 34 | persistent_stages | 155.7 | 2.38x | Persistent kernel + num_stages=4 + expanded configs | KEEP (+2.3) |
| 35 | revert_k128 | 128.8 | 1.92x | BLOCK_SIZE_K=128 to align with group_size | REVERT (-26.9) |
| 36 | flat_k_loop | 170.4 | 2.63x | Flatten K loop, simplify masks, remove nested group loop | KEEP (+14.7) |
| 39 | constexpr_group | 177.5 | 2.66x | Make group_size a tl.constexpr (QUANT_GROUP_SIZE) | KEEP (+7.1) |
| 60 | dequant_cublas | 188.2 | 2.88x | Split: Triton dequant kernel + cuBLAS FP16 matmul | KEEP (+10.7) |
| 61 | dequant_flinear | 196.1 | 3.04x | Transposed dequant + F.linear (NT cuBLAS gemm) | KEEP (+7.9) |
| 62 | aligned_blocks | 196.6 | 2.98x | Optimize for BLOCK_SIZE_K==group_size alignment | KEEP (+0.5) |
| 63 | more_configs | 197.9 | 2.95x | Expanded autotune configs with larger blocks | KEEP (+1.3) |

Fused GEMM kernel (dequantize_fused_gemm) experiments:

| Exp | Tag | TFLOPS | Speedup | What Was Tried | Result |
|-----|-----|--------|---------|---------------|--------|
| 0 | baseline | 16.3 | 0.25x | Baseline 32x32x32 fused dequant+SwiGLU kernel | FAIL (correctness) |
| 1 | autotune | 24.4 | 0.38x | Added autotune configs for fused kernel | FAIL (correctness) |

## Top 5 Biggest Impact Changes

1. **Autotune + L2 swizzle** (exp 0 to 21): +121.7 TFLOPS (15.1 to 136.8). Replacing naive 32x32x32 with 22 autotuned configs and L2 cache-friendly tile ordering delivered a 9x throughput jump -- the single largest gain.

2. **Flat K loop** (exp 34 to 36): +14.7 TFLOPS (155.7 to 170.4). Removing the nested group loop and flattening to a single K-dimension loop let Triton's software pipelining overlap loads across iterations.

3. **Split dequant + cuBLAS** (exp 39 to 60): +10.7 TFLOPS (177.5 to 188.2). The paradigm shift: stop trying to fuse dequant with matmul. Separate Triton dequant (53 us) plus cuBLAS FP16 matmul (500 us) beat the best fused kernel.

4. **Persistent kernel** (exp 29 to 32): +9.5 TFLOPS (143.9 to 153.4). Launching exactly 680 thread blocks (4x SM count) and looping over tiles amortized launch overhead and improved L2 reuse.

5. **Transposed dequant + F.linear** (exp 60 to 61): +7.9 TFLOPS (188.2 to 196.1). Storing dequanted weights as [N,K] and using F.linear (NT GEMM) instead of torch.mm (NN GEMM) gave a 4% boost from better cuBLAS layout preference.

## Top 3 Most Interesting Insights

1. **Don't compete with cuBLAS.** The fused Triton kernel plateaued at 177.5 TFLOPS. Splitting dequant and matmul, letting cuBLAS handle the dense FP16 multiply, immediately jumped to 188.2 TFLOPS. cuBLAS uses SASS-level optimization, TMA loads, and warp specialization that Triton cannot match for pure dense matmul.

2. **Flat loops beat "smarter" nested loops.** A two-level K loop that hoisted invariant scale/zero loads was algorithmically superior, but the flat K loop won because Triton's compiler can only pipeline a single loop. Compiler-friendliness trumps algorithmic cleverness.

3. **W4A16 quantization costs ~50% of peak even when optimized.** Pure FP16 cuBLAS achieves ~215 TFLOPS (51% of peak). The best W4A16 kernel achieves 197.9 TFLOPS (47.2% of peak). The INT4 unpacking overhead (shift, mask, cast, subtract, multiply = 5 ALU ops per weight element) creates pipeline bubbles before tensor cores can fire.

## What Failed and Why

- **BLOCK_SIZE_K=128**: Aligned with group_size but caused register spill, dropping from 155.7 to 128.8 TFLOPS.
- **FP8 matmul**: 2x faster compute but correctness failed -- FP8 E4M3 (3 mantissa bits) cannot represent dequanted weights accurately enough.
- **Fused dequant+matmul kernel**: Always slower than split approach because register pressure and ALU overhead compete with tensor core throughput.
- **CUDA graphs**: Copy overhead for input buffers negated any launch overhead savings.
- **Higher num_stages (5-6)**: Shared memory overflow at the required tile sizes.
- **Eviction policy hints**: `evict_first`/`evict_last` had no measurable effect on Blackwell.
- **Non-persistent 2D grid**: Always slower than persistent kernel pattern.
- **Fused GEMM kernel**: Correctness failures in 9 of 12 shape configurations, likely group boundary handling issues.
