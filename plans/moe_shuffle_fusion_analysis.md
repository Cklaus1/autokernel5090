# MoE Shuffle-Rows Barrier: Can We Bypass It for Fused Norm+Quant?

## Executive Summary

The previous conclusion that "MoE shuffle_rows is fundamental to CUTLASS grouped GEMM" is **correct but incomplete**. The shuffle IS required -- but it can be **fused into the quantization kernel** with a ~5-line CUDA change, saving one full HBM round-trip of the sorted bf16 intermediate. The norm+quant fusion across the shuffle barrier is NOT viable.

## Current MoE FP4 Execution Path

```
1. RMSNorm(hidden_states)                              -> bf16 [B, H]
2. Router(normalized) -> topk_ids, topk_weights         -> int32 [B, topK]
3. get_cutlass_moe_mm_data(topk_ids) -> a_map, c_map, expert_offsets, etc.
4. shuffle_rows(normalized, a_map)                      -> bf16 [M, H]  (M = B*topK)
5. scaled_fp4_experts_quant(sorted, gscale, offsets)    -> fp4 [M, H/2] + blockscales
6. cutlass_fp4_moe_mm(fp4, w1, scales) -> c1            -> bf16 [M, 2N]
7. silu_and_mul_scaled_fp4_experts_quant(c1, ...)        -> fp4 (already fused activation+quant)
8. cutlass_fp4_moe_mm(fp4_2, w2, scales) -> c3          -> bf16 [M, K]
9. shuffle_rows(c3, c_map)                              -> bf16 unsorted output
10. weighted sum over topK experts
```

## Three Approaches Investigated

### Approach 1: Route on Raw Hidden States (REJECTED)

**Idea**: Move RMSNorm after shuffle, enabling fused norm+shuffle+quant.
```
Router(x) -> topk_ids          # route on un-normalized x
shuffle_rows(x, a_map)         # sort raw tokens
fused_norm_quant(sorted, ...)  # norm + quant in one kernel
```

**Why it fails**:
- Mathematically, `argsort(W @ x)` == `argsort(W @ norm(x))` since `rms(x)` is a positive per-token scalar. Top-k SELECTION should be identical.
- However, in bf16 precision, `W @ x` and `W @ norm(x) * rms(x)` differ due to matmul rounding. Empirically only 75% of expert selections match (tested B=32, H=2816, E=128, topK=8).
- The model was trained with routing on normalized inputs. Changing this is a **model modification**, not an optimization.
- Top-k weights (softmax values) would also change due to different effective temperature.

**Verdict**: Invalid. Changes model semantics.

### Approach 2: Fuse Shuffle + Quant (VIABLE -- Recommended)

**Idea**: Merge steps 4+5 into a single kernel that gathers from unsorted input during quantization.
```
# Before (two kernels, one intermediate):
sorted_bf16 = shuffle_rows(normalized, a_map)          # write M*H bf16
fp4, scales = scaled_fp4_experts_quant(sorted_bf16, ...) # read M*H bf16, write fp4

# After (one kernel, no intermediate):
fp4, scales = shuffle_and_scaled_fp4_experts_quant(normalized, a_map, gscale, offsets)
                                                        # gather-read from unsorted, write fp4
```

**Why it works**:
- `shuffle_rows` is a pure gather: `output[dst] = input[dst2src_map[dst]]`
- `scaled_fp4_experts_quant` reads contiguous rows and quantizes with per-expert scales
- A fused kernel simply gathers (non-contiguous read) instead of reading contiguous rows
- The expert_offsets, blockscale_offsets, and per-expert global scales work identically because the output is still sorted by expert
- Zero semantic change -- bit-identical output

**CUDA implementation** (in `nvfp4_experts_quant.cu`):
```cuda
// Current:
int64_t inOffset = rowIdx * inColsPerRow + colIdx;
PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];

// Fused (add dst2src_map parameter):
int64_t srcRowIdx = dst2src_map ? dst2src_map[rowIdx] : rowIdx;
int64_t inOffset = srcRowIdx * inColsPerRow + colIdx;
PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
```

This is a ~5 line change in each of the two kernel variants (low-latency and large-M).

### Approach 3: Fuse Norm + Shuffle + Quant (NOT VIABLE)

**Why**: The norm must happen before routing (step 2 depends on normalized output). Since routing produces a_map, and a_map is needed for shuffle, the norm is fundamentally separated from shuffle+quant by the routing computation.

## Measured Performance Impact

### Microbenchmarks (RTX 5090)

| Batch | M (B*topK) | shuffle_rows (us) | scaled_fp4_quant (us) | Total (us) | Data (KB) |
|-------|-----------|-------------------|----------------------|-----------|----------|
| 1     | 8         | 7.7               | 23.2                 | 30.9      | 44       |
| 8     | 64        | 7.6               | 10.5                 | 18.1      | 352      |
| 32    | 256       | 7.6               | 16.6                 | 24.2      | 1408     |
| 128   | 1024      | 9.3               | 10.7                 | 20.0      | 5632     |
| 512   | 4096      | 12.6              | 12.9                 | 25.6      | 22528    |

### Memory Traffic Savings (B=32, H=2816, M=256)

| Component | Separate (KB) | Fused (KB) |
|-----------|--------------|-----------|
| shuffle read+write | 2816 | 0 (eliminated) |
| quant read+write | 1936 | 1936 |
| **Total** | **4752** | **1936** |
| **Savings** | -- | **2816 KB (59%)** |

### Estimated End-to-End Impact (Gemma 4 27B, 23 MoE layers)

| Scenario | Per-layer save | Total save | Forward pass | Improvement |
|----------|---------------|-----------|-------------|------------|
| B=32 decode | ~8 us | ~184 us | ~8 ms | **~2.3%** |
| B=128 | ~10 us | ~230 us | ~12 ms | **~1.9%** |
| B=512 prefill | ~26 us | ~598 us | ~25 ms | **~2.4%** |

The savings are dominated by kernel launch overhead at small batch sizes, and by eliminated memory traffic at large batch sizes.

## Why Shuffle IS Fundamentally Required

CUTLASS grouped GEMM (`cutlass_fp4_moe_mm`) requires:
1. **Contiguous expert blocks**: All tokens for expert E must be at rows `[expert_offsets[E], expert_offsets[E+1])`.
2. **Per-expert problem sizes**: Each expert has its own M (token count) in the grouped GEMM.
3. **Block-scale alignment**: FP4 block-scales are indexed by `blockscale_offsets[E]` per expert.

The input tokens arrive in arbitrary order (each token selects topK experts). Without sorting, CUTLASS cannot dispatch per-expert sub-problems. This is NOT bypassable.

However, the sorting can be done AS PART OF the quantization kernel (gather-read pattern) instead of as a separate kernel.

## Implementation Plan

**Difficulty**: Low (CUDA kernel change + Python plumbing)
**Risk**: Low (bit-identical output, no semantic change)
**Files to modify**:
1. `csrc/libtorch_stable/quantization/fp4/nvfp4_experts_quant.cu` -- add `dst2src_map` parameter to both kernel variants
2. `csrc/libtorch_stable/quantization/fp4/nvfp4_quant_entry.cu` -- new entry point
3. `csrc/libtorch_stable/torch_bindings.cpp` -- register new op
4. `csrc/libtorch_stable/ops.h` -- declare new function
5. `vllm/_custom_ops.py` -- Python wrapper
6. `vllm/model_executor/layers/fused_moe/cutlass_moe.py` -- replace `shuffle_rows + scaled_fp4_experts_quant` with fused call

**Note**: The output shuffle (`shuffle_rows(c3, c_map)` at step 9) remains separate -- there is nothing to fuse it with since it feeds directly into the weighted sum.

## Conclusion

The previous agent's conclusion was correct: shuffle_rows IS fundamental to CUTLASS grouped GEMM. But the conclusion was too pessimistic -- while the SORT operation cannot be eliminated, it can be FUSED into the quantization kernel by changing contiguous reads to gathered reads. This eliminates one bf16 intermediate tensor and one kernel launch per MoE layer, saving ~2% of forward pass time with minimal implementation effort.

The norm+quant fusion across the shuffle barrier is genuinely impossible because the routing decision depends on the normalized output, creating a true data dependency: norm -> route -> shuffle -> quant.
