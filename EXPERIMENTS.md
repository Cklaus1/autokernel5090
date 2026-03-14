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
| 74 | dequant_caching | 215.0 | 3.26x | Dequant weight caching by tensor identity | KEEP (+17.1) |
| 75 | fp16_accum | 290.0 | 4.40x | FP16-accumulate Triton matmul (out_dtype=fp16) | KEEP (+75) |
| 76 | trim_configs | 290.0 | 4.40x | Trim autotune configs to top performers | KEEP |
| 77 | persistent_matmul | 278.0 | 4.24x | Persistent matmul kernel 680 programs | REVERT (-12) |
| 78 | pure_fp16_acc | FAIL | — | Pure FP16 accumulator (compile error) | REVERT |
| 79 | expanded_configs | 285.0 | 4.33x | Expanded autotune for non-square shapes | REVERT (-5) |
| 80 | imprecise_acc | 218.7 | 3.34x | max_num_imprecise_acc=BK | REVERT (-71) |
| 81 | cublas_all_m | 218.2 | 3.27x | cuBLAS F.linear for all M sizes | REVERT (-72) |
| 82 | bk128_triton36 | 327.0 | 4.89x | BK=128 configs + Triton 3.6.0 upgrade | KEEP (+37) |
| 83-88 | various | ~322 | ~4.9x | stages=3, dot_scaled, warps=16, ieee precision | REVERT (noise) |
| 89 | aligned_nocheck | 328.9 | 4.93x | ALIGNED flag skips boundary checks | KEEP |
| 90-97 | various | ~322 | ~4.9x | cuBLAS thresholds, hardcoded configs, BK=64 stages=4 | REVERT |
| 96 | clean_4config | 328.0 | 4.93x | Clean 4-config autotune, BK=128 focus (final) | KEEP |

## Other Kernel Types

| Kernel | TFLOPS | vs PyTorch | Status |
|--------|--------|-----------|--------|
| Flash attention (long seq) | 399 | 22.6x | Correct, autotuned |
| Fused SwiGLU MLP (3 matmuls) | 213 | 3.3x | First correct implementation |
| FP8 scaled_mm (fails correctness) | 385 | 5.9x | FP8 precision limit |
| NVFP4 native (torch._scaled_mm_v2) | 1,271 | 5.8x vs cuBLAS | Correct, production-viable |

## Top 5 Biggest Impact Changes

1. **Autotune + L2 swizzle** (exp 0 to 21): +121.7 TFLOPS (15.1 to 136.8). Replacing naive 32x32x32 with 22 autotuned configs and L2 cache-friendly tile ordering delivered a 9x throughput jump -- the single largest gain.

2. **FP16 accumulation** (exp 74 to 75): +75 TFLOPS (215 to 290). Using `tl.dot(a, b, out_dtype=tl.float16)` doubled tensor core throughput on Blackwell SM120.

3. **BK=128 + Triton 3.6.0** (exp 76 to 82): +37 TFLOPS (290 to 327). Triton 3.6.0's improved shared memory allocation allowed BK=128 tiles that were blocked by SM120's 101KB limit on 3.5.1.

4. **Flat K loop** (exp 34 to 36): +14.7 TFLOPS (155.7 to 170.4). Removing the nested group loop let Triton's software pipelining overlap loads across iterations.

5. **Split dequant + cuBLAS** (exp 39 to 60): +10.7 TFLOPS (177.5 to 188.2). Stop trying to fuse dequant with matmul. Separate Triton dequant + cuBLAS FP16 matmul beat the best fused kernel.

## What Failed and Why

- **FP8 matmul (385 TFLOPS)**: 2x faster but correctness fails -- FP8 E4M3 has 3 mantissa bits, fundamentally cannot achieve atol=0.05 at K=5120. Per-row scaling, split-K, and e5m2 all tried and failed. It's a precision limit, not a scaling problem.
- **Hardcoded configs (exps 94-95)**: Dropping from 328 to 250 TFLOPS. Autotune is essential -- no single config is universally optimal due to cache state and thermal variance.
- **Persistent matmul (exp 77)**: -12 TFLOPS. The persistent pattern overhead outweighed L2 benefits at these sizes.
- **Pure FP16 accumulator (exp 78)**: Compilation error -- Triton doesn't support FP16 accumulators in tl.dot directly.
- **dot_scaled fp16 (exp 84)**: Falls back to BF16 software emulation on SM120. Not a native hardware path.
- **2:4 structured sparsity**: cuSPARSELt matmul not supported on SM120 consumer Blackwell.
- **torch.compile**: profile.py in project directory shadows stdlib profile module, breaking Dynamo.
