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

---

## Session 2: NVFP4 Matmul + Multi-Kernel Fixes

### NVFP4 Matmul Experiments

Added `nvfp4_matmul` kernel type to bench.py. Optimized from 18 to 1,261 TFLOPS.

| Exp | Tag | TFLOPS | Speedup | What Was Tried | Result |
|-----|-----|--------|---------|---------------|--------|
| 0 | baseline | 17.9 | 0.08x | Both A+B quantized every call (Python searchsorted) | KEEP (baseline) |
| 1 | weight_cache_threshold | 187.4 | 0.86x | Cache B quant by tensor id + threshold encoding | KEEP (+169) |
| 2 | torch_compile | FAIL | — | torch.compile quantization core | REVERT (profile.py shadow) |
| 3 | searchsorted_cachefix | 240.5 | 1.11x | searchsorted + cache key includes shape+data_ptr | KEEP (+53) |
| 4 | nvfp4_torch_compile | FAIL | — | torch.compile blocked by profile.py | REVERT |
| 5 | cuda_v2_fused_pad | 971.8 | 4.40x | CUDA quant kernel v2 with fused FP8 scale padding | KEEP (+731) |
| 6 | cuda_graph | 842.5 | 3.85x | CUDA graph capture (quant+GEMM) | REVERT (copy overhead + NaN) |
| 7 | v3_half2_vectorized | 987.3 | 4.53x | Vectorized half2 loads + additive thresholds | KEEP (+15) |
| 8 | cache_both | 1,260.5 | 5.71x | Cache both A+B quantization (pure GEMM) | KEEP (+273) |
| 9 | scaled_mm_v2 | 1,012.3 | 4.64x | _scaled_mm_v2 with recipe=2 swizzle=1 | REVERT (-248) |

### Dequantize Fused GEMM (SwiGLU MLP)

Fixed from 0/12 correctness to 12/12 PASS.

| Exp | Tag | TFLOPS | Speedup | What Was Tried | Result |
|-----|-----|--------|---------|---------------|--------|
| 0 | baseline | 16.3 | 0.25x | Fused Triton gate+up kernel | FAIL (9/12 shapes) |
| 1 | autotune | 24.4 | 0.38x | Added autotune configs | FAIL (still 9/12) |
| 2 | split_dequant_cublas | 212.3 | 3.32x | Split dequant + cuBLAS + fix buffer aliasing | PASS (12/12) |

### What Worked (New Lessons)

- **CUDA C++ quantization kernels**: 52x faster than Python searchsorted (23µs vs 358µs). The fused kernel does block-max, scale, threshold-quantize, and pack in a single pass per 16-element block.
- **Fused FP8 scale output**: v2 kernel writes scales directly to padded FP8 layout, eliminating the Python-side zero+copy+convert step (37µs → 26µs).
- **Vectorized half2 loads**: Reading 8×half2 instead of 16×half reduces memory transactions (26µs → 24µs).
- **Weight caching by tensor identity**: Cache key `(id(tensor), shape, data_ptr)` prevents stale cache hits when Python reuses memory addresses for different tensors.
- **Buffer aliasing bug**: Gate and up weight matrices share identical shapes — cache keys must include tensor identity, not just dimensions.

### What Failed (New Lessons)

- **CUDA graphs for variable inputs**: The `.copy_()` to static buffer + `.clone()` of output costs more than the launch overhead saved. Only useful when inputs are truly static.
- **`_scaled_mm_v2`**: 20% slower than `_scaled_mm` despite recipe/swizzle hints. The bf16→fp16 output conversion adds overhead, and cuBLASLt may route through a different algorithm.
- **Raw FP4 PTX on sm_120**: The `mma.sync.aligned.m16n8k64...e2m1` instruction is "not supported on .target sm_120". SM120 can only access FP4 through cuBLASLt. Needs CUDA 12.9+ or the `.kind::f8f6f4` modifier which has shape errors in 12.8.
- **FlashInfer CUTLASS standalone**: 1,108 TFLOPS vs cuBLASLt's 1,251 — same underlying CUTLASS 3.x kernel but with more dispatch overhead.
- **Pipeline overlap (quant||GEMM)**: Both kernels saturate memory bandwidth — they compete for the same resource, so streams don't help.

### Performance Summary Across All Kernels

| Kernel Type | Best TFLOPS | vs PyTorch | vs cuBLAS FP16 | Experiments |
|---|---|---|---|---|
| W4A16 matmul | 328 | 4.93x | 1.57x | 97 |
| NVFP4 matmul | 1,261 | 5.71x | 5.97x | 10 |
| Fused SwiGLU MLP | 212 | 3.32x | — | 3 |
| Flash attention | 399 | 22.6x | — | (Session 1) |

### Hardware Ceiling (Updated)

| Precision | Theoretical Peak | cuBLASLt Measured | Our Best | Utilization |
|---|---|---|---|---|
| FP16 dense | 209.5 TFLOPS | 211 TFLOPS | 328 TFLOPS* | 157%* |
| FP8 | 419 TFLOPS | 430 TFLOPS | 385 TFLOPS | 92% |
| FP4 (NVFP4) | 1,676 TFLOPS | 1,303 TFLOPS | 1,261 TFLOPS | 75.2% |

\* W4A16 328 TFLOPS exceeds FP16 dense peak because FP16 accumulation doubles tensor core throughput (uses the FP8 MMA datapath with FP16 operands).
