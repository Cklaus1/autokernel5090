# AutoKernel Optimization - Part 2 Lessons Learned

## Executive Summary

Continued optimization of GPU kernels with focus on W4A16 quantized operations. Achieved **197.9 TFLOPS** (3.0x speedup over PyTorch) for `quantized_matmul_w4a16` kernel, reaching 47.2% of theoretical peak performance. Key insight: **split approach beats fused kernels** for quantized operations when cuBLAS can be leveraged.

## Performance Results

### quantized_matmul_w4a16 Kernel
- **Baseline**: 15.1 TFLOPS (3.6% peak, 0.23x vs PyTorch)
- **Final**: 197.9 TFLOPS (47.2% peak, 3.0x speedup)
- **Improvement**: 13x faster than baseline, 3x faster than PyTorch
- **Experiments**: 15 total iterations

### dequantize_fused_gemm Kernel
- **Baseline**: 16.3 TFLOPS (correctness issues)
- **Current**: 24.4 TFLOPS (still has correctness issues)
- **Status**: Work in progress, needs debugging

## Key Optimization Strategies That Worked

### 1. Split Dequantization + cuBLAS (Game Changer)
**Impact**: 2.88x speedup (experiment #60)

Instead of fusing dequantization with matmul in a single kernel:
- Separate fast Triton kernel for INT4 → FP16 dequantization
- Leverage highly-optimized cuBLAS for the FP16 matmul
- Result: Better than any fused implementation

```python
# Winner: Split approach
dequant_kernel[grid](packed_weights, scales, zeros, Wt, ...)  # Fast dequant
return torch.nn.functional.linear(activation, Wt)             # cuBLAS matmul
```

### 2. Transposed Weight Layout for F.linear
**Impact**: Additional 0.16x improvement over cuBLAS

- Store dequantized weights as (N, K) instead of (K, N)
- F.linear uses NT GEMM internally (more efficient)
- Avoids transpose overhead

### 3. Aggressive Autotune Configurations
**Impact**: 2.10x speedup from baseline

Expanded from 5 to 10+ configurations:
- Block sizes: 32×64, 32×128, 32×256, 64×128, 128×128, 128×256
- Warps: 4-8 depending on block size
- Stages: 2-4 for software pipelining

### 4. Aligned Block Optimization
**Impact**: ~5% improvement when applicable

Special case when `BLOCK_SIZE_K == group_size (128)`:
- Load scales/zeros once per block (not per element)
- Eliminates redundant group boundary checks
- Better memory access pattern

### 5. Pre-allocated Weight Buffers
**Impact**: Reduced allocation overhead

```python
_wt_buf = {}  # Global cache
wkey = (K, N, dtype)
if wkey not in _wt_buf:
    _wt_buf[wkey] = torch.empty((N, K), device=device, dtype=dtype)
```

## What Didn't Work (Anti-Patterns)

### 1. Fused Quantized Matmul
- **Attempted**: Single kernel doing dequant + matmul
- **Result**: FAIL - correctness issues, slower than split
- **Why**: Too complex, register pressure, can't compete with cuBLAS

### 2. L2 Cache Swizzling
- **Attempted**: Reorder tiles for better L2 locality
- **Result**: 182 TFLOPS (worse than 197.9)
- **Why**: Added complexity without benefit for this workload

### 3. Persistent Kernels
- **Attempted**: One thread block per SM, loop over tiles
- **Result**: 189.9 TFLOPS (worse)
- **Why**: Overhead not justified for this kernel size

### 4. TF32 for F.linear
- **Attempted**: Enable TF32 tensor cores
- **Result**: No improvement (196.0 vs 197.9)
- **Why**: Already bandwidth-bound after dequantization

### 5. Vectorized Unpacking
- **Attempted**: Process 8 INT4 values at once
- **Result**: 184.7 TFLOPS (worse)
- **Why**: Triton compiler already optimizes this

## Critical Insights

### 1. Don't Fight cuBLAS
For operations that cuBLAS handles well (dense FP16 matmul), it's nearly impossible to beat with custom Triton kernels. The winning strategy: **optimize what's unique** (dequantization) and **leverage what exists** (cuBLAS).

### 2. Quantization Overhead is Significant
- Baseline: 15.1 TFLOPS (3.6% peak)
- After optimization: 197.9 TFLOPS (47.2% peak)
- Pure FP16 matmul: ~400 TFLOPS (95% peak)
- **Conclusion**: W4A16 quantization costs ~50% performance even when optimized

### 3. Memory Layout Matters More Than Compute
The biggest wins came from:
- Transposed storage (avoiding transpose)
- Aligned block sizes (group_size alignment)
- Pre-allocated buffers (no repeated allocations)

### 4. Shared Memory is the Limiting Factor
For fused kernels with multiple weight matrices:
- Hit shared memory limits quickly (101KB on most GPUs)
- Had to reduce block sizes (64×64 → 32×32)
- Performance suffered significantly

### 5. Correctness Must Come First
The fused kernel achieved 24.4 TFLOPS but failed correctness tests. A fast but incorrect kernel is worthless. Always verify:
- Smoke tests
- Shape sweep  
- Numerical stability
- Determinism
- Edge cases

## Optimization Progression

| Exp | Optimization | TFLOPS | Speedup | Key Learning |
|-----|-------------|--------|---------|--------------|
| 0 | Baseline (32×32×32) | 15.1 | 0.23x | Naive per-element dequant is terrible |
| 21 | Autotune (22 configs) | 136.8 | 2.10x | Block size tuning is critical |
| 36 | Flattened K loop | 170.4 | 2.63x | Simplify control flow |
| 39 | Constexpr group_size | 177.5 | 2.66x | Compile-time constants help |
| 60 | Split dequant+cuBLAS | 188.2 | 2.88x | Don't compete with cuBLAS |
| 61 | Transposed F.linear | 196.1 | 3.04x | Memory layout optimization |
| 63 | Extended configs | 197.9 | 2.95x | More autotune options |

## Recommendations for Future Work

### 1. Fix dequantize_fused_gemm Correctness
- Debug shape sweep failures (9/12 failing)
- Likely issue with group boundary handling
- Consider splitting into gate/up fusion only

### 2. Explore INT8 Intermediate
- Pack two INT4 → INT8
- Use INT8 tensor cores (SM90+)
- Apply scales after matmul

### 3. Decode-1 Specialization
- Special kernel for batch=1 (autoregressive)
- 1D grid over N dimension
- Persistent kernel pattern may help here

### 4. Profile on Different GPUs
- Current: Unknown GPU (likely A100/H100)
- Test on: L4 (bandwidth limited), 4090 (consumer), H100 (latest)
- Adjust configs per architecture

### 5. Benchmark Against Production Libraries
- Compare with: bitsandbytes, GPTQ, AWQ
- Measure end-to-end model performance
- Consider quantization schemes beyond W4A16

## Conclusion

The key to optimizing quantized kernels is **pragmatism over purism**. The theoretically elegant fused kernel lost to the practical split approach. By separating concerns (Triton for dequantization, cuBLAS for matmul), we achieved 3x speedup over PyTorch and 47% of theoretical peak - excellent for a quantized kernel.

**Final wisdom**: Know when to optimize and when to delegate. The best kernel is sometimes two kernels.

---

*Generated after 17 experiments across 2 kernel types*  
*Best result: 197.9 TFLOPS (quantized_matmul_w4a16)*  
*Total optimization time: ~2 hours*