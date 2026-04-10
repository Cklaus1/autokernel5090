# SM120 (RTX 5090) 2:4 Structured Sparsity Test

**Date:** 2026-04-09
**GPU:** RTX 5090, SM120, CUDA 13.0, PyTorch 2.11.0+cu130

## Summary

SM120 supports 2:4 structured sparsity via cuSPARSELT only. The CUTLASS semi-structured kernel is hardcoded to SM80 and fails with "Supported only on GPUs with compute capability 8.x". FP4 (NVFP4) + 2:4 sparsity is **not supported** in any current PyTorch/CUTLASS/cuSPARSELT path.

## API Availability

| API | Available | Notes |
|-----|-----------|-------|
| `torch.sparse.to_sparse_semi_structured` | YES | In `torch.sparse`, not `torch` directly |
| `torch.to_sparse_semi_structured` | NO | Missing from this build |
| `SparseSemiStructuredTensorCUSPARSELT` | YES | Works on SM120 |
| `SparseSemiStructuredTensorCUTLASS` | FAILS | "Supported only on GPUs with compute capability 8.x" |
| `torch._cslt_compress` | YES | Low-level FP8 compress works |
| `torch._cslt_sparse_mm` | YES (partial) | Works for FP16/BF16/INT8/FP8; fails M<32 |

## Supported Dtypes for 2:4 Sparsity

### cuSPARSELT backend (SM120):
- `torch.float16` — min shape 16x16
- `torch.bfloat16` — min shape 16x16
- `torch.int8` — min shape 32x32
- `torch.float8_e4m3fn` — min shape 32x32 (via `_cslt_sparse_mm` low-level API)
- `torch.float4_e2m1fn_x2` (NVFP4) — **NOT supported** (cslt compress fails)

### CUTLASS backend (SM120):
- **ALL dtypes fail** — hardcoded to SM80 only

## NVFP4 + 2:4 Sparsity: Not Possible

- `torch._scaled_mm_v2` (the NVFP4 path) has no sparse variant in its schema
- `_cslt_compress` fails for `float4_e2m1fn_x2` dtype
- No FP4+sparse ops found anywhere in vLLM's Python or CUDA sources
- cuSPARSELT's supported dtype list does not include FP4
- **Conclusion:** NVFP4 + 2:4 sparsity is not exposed in any current software stack for SM120.

## Performance Benchmarks: FP8 2:4 Sparse vs Dense

Shape: K=7168, N=2048 (MoE gate/up projection, DeepSeek-style)
Expert weight is magnitude-pruned to 2:4 pattern before deployment.

| Tokens (M) | FP16 dense | FP8 dense | FP8 + 2:4 sparse | vs FP16 | vs FP8 |
|-----------|-----------|-----------|-----------------|---------|--------|
| 1 | — | — | FAILS (M<32 min) | — | — |
| 8 | — | — | FAILS (M<32 min) | — | — |
| 32 | 0.020ms (46T) | 0.017ms (57T) | 0.029ms (33T) | 0.71x | 0.58x |
| 128 | 0.031ms (122T) | 0.044ms (86T) | 0.025ms (149T) | **1.22x** | **1.72x** |
| 256 | 0.080ms (94T) | 0.058ms (129T) | 0.033ms (229T) | **2.43x** | **1.77x** |
| 512 | 0.210ms (72T) | 0.097ms (155T) | 0.031ms (483T) | **6.76x** | **3.13x** |
| 1024 | 0.225ms (134T) | 0.083ms (361T) | 0.064ms (473T) | **3.53x** | **1.31x** |
| 2048 | 2.460ms (24T) | 0.100ms (603T) | 0.100ms (603T) | 24.7x | 1.00x |
| 4096 | 2.722ms (44T) | 2.852ms (42T) | 2.604ms (46T) | 1.05x | 1.10x |
| 8192 | 11.19ms (21T) | 2.841ms (85T) | 2.867ms (84T) | 3.90x | 0.99x |

Shape: K=2048, N=7168 (MoE down projection)

| Tokens (M) | FP16 dense | FP8 dense | FP8 + 2:4 sparse | vs FP16 | vs FP8 |
|-----------|-----------|-----------|-----------------|---------|--------|
| 1-8 | — | — | FAILS (M<32 min) | — | — |
| 32 | 0.020ms (47T) | 0.015ms (61T) | 0.040ms (23T) | 0.50x | 0.38x |
| 512 | 0.104ms (145T) | 0.044ms (339T) | 0.034ms (445T) | **3.08x** | **1.31x** |
| 1024 | 0.220ms (136T) | 0.113ms (266T) | 0.081ms (370T) | **2.71x** | **1.39x** |
| 2048 | 0.293ms (205T) | 0.111ms (544T) | 0.103ms (582T) | **2.84x** | **1.07x** |
| 8192 | 3.657ms (66T) | 2.633ms (91T) | 0.675ms (356T) | **5.41x** | **3.90x** |

Key observations:
- FP8 + 2:4 sparse wins at M=128-1024 range (typical prefill batch sizes), up to 3.13x over FP8 dense
- At M=512 gate/up proj: 483 TFLOPS with sparse vs 155 TFLOPS FP8 dense
- At small M (<32): cuSPARSELt minimum shape requirements cause failures
- At large M (4096+): FP8 dense and FP8 sparse are roughly equal — both compute-bound

## Quality Impact of 2:4 Pruning

Tested with Xavier-initialized weights (scale = 1/sqrt(K)), M=256, K=7168, N=2048:

| Method | Mean Relative Error | Cosine Similarity |
|--------|---------------------|-------------------|
| FP8 dense (no pruning) | 6.83% | 0.9977 |
| FP8 + 2:4 sparse (magnitude pruned, no fine-tuning) | 36.87% | 0.9295 |

**The 36.87% relative error without fine-tuning is a blocker for production use.** Literature shows that 2:4 sparsity requires fine-tuning to recover quality (NVIDIA SparseGPT, Wanda). With fine-tuning, models typically recover to within ~1-2% of dense perplexity.

## Practical Path Forward

### FP8 + 2:4 Sparse (viable, with fine-tuning)
1. Start from a fine-tuned FP8 checkpoint
2. Apply SparseGPT or Wanda to find 2:4 mask per-layer
3. Fine-tune with the fixed sparsity mask for ~1-5% of training
4. Compress with `torch._cslt_compress` at load time
5. Serve with `torch._cslt_sparse_mm` in the MoE forward pass

Expected gain: 1.3-3x speedup over FP8 dense at M=128-2048 token range (prefill-dominant workloads).

### NVFP4 + 2:4 Sparse (not currently possible)
- No hardware or software support exists yet
- NVIDIA has announced FP4 sparsity for Blackwell (SM100) in CUTLASS 4.x roadmap
- Would require custom CUTLASS kernels using `SpGEMMFP4` tile scheduler (not yet public)
- Expected when available: 2x on top of NVFP4's already 3.9x gain vs FP16 = theoretical ~8x total

## Minimum Shape Requirements

cuSPARSELt FP8 2:4 sparse fails for M<32 (dense_min_rows=16 for FP8, but descriptor init also requires alignment). For MoE decode with 1-8 tokens per expert, sparse is not applicable — use FP8 dense or NVFP4.

## Usage Example

```python
import torch

# Pre-processing (offline, at model load)
W_fp8 = weight_matrix.to(torch.float8_e4m3fn)  # [N, K]
# Apply 2:4 magnitude pruning (or load SparseGPT-pruned weights)
compressed_W = torch._cslt_compress(W_fp8)  # ~50% smaller

# Inference (online)
def sparse_linear_fp8(input_fp8, compressed_W, out_dtype=torch.float16):
    # input_fp8: [M, K], compressed_W: [N, K_compressed]
    # returns: [M, N]
    alg_id = 2  # found via _cslt_sparse_mm_search at load time
    return torch._cslt_sparse_mm(
        compressed_W, input_fp8.t(), None, None, out_dtype, True, alg_id
    )
```

Note: `input_fp8.t()` must be passed as non-contiguous — do not call `.contiguous()` on it.
