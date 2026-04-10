# FusenCache + NVFP4 Integration Plan for Gemma4 26B on RTX 5090

**Date:** April 9, 2026
**Target:** Replace BF16 KV cache with FusenCache k4v4 quantized KV alongside NVFP4 weight quantization
**Goal:** ~2x KV cache capacity (86K tokens vs 43K), minimal throughput regression, no quality loss

---

## Architecture Overview

### Current Setup (Baseline: 6,615 tok/s)

```
Gemma4 26B-A4B (30 layers)
├── 25 sliding-window layers (head_dim=256, Hq=16, Hk=8, window=1024)
│   ├── Weights: NVFP4 (CUTLASS FP4 GEMM)
│   ├── Attention: FlashAttention v2
│   └── KV cache: BF16 (2 bytes/dim K + 2 bytes/dim V = 1024 bytes/token/head)
└── 5 global attention layers (head_dim=512, Hq=16, Hk=2, unlimited ctx)
    ├── Weights: NVFP4 (CUTLASS FP4 GEMM)
    ├── Attention: Triton unified attention
    └── KV cache: BF16 (2 bytes/dim K + 2 bytes/dim V = 2048 bytes/token/head)
```

### Target Setup

```
Gemma4 26B-A4B (30 layers)
├── 25 sliding-window layers (head_dim=256, Hq=16, Hk=8, window=1024)
│   ├── Weights: NVFP4 (unchanged)
│   ├── Attention: FusenKV backend (Triton decode + SDPA prefill)
│   └── KV cache: k4v4b64 (0.5 bytes/dim K + 0.5 bytes/dim V = 256 bytes/token/head + scales)
└── 5 global attention layers (head_dim=512, Hq=16, Hk=2, unlimited ctx)
    ├── Weights: NVFP4 (unchanged)
    ├── Attention: FusenKV backend (Triton decode + SDPA prefill)
    └── KV cache: k4v4b64 (0.5 bytes/dim K + 0.5 bytes/dim V = 512 bytes/token/head + scales)
```

---

## Key Findings from Code Analysis

### 1. FusenCache Backend Needs v1 API Update (CRITICAL)

The current `FusenKVMetadataBuilder.__init__` takes `(self, runner)` but vLLM v1's
`AttentionMetadataBuilder.__init__` requires `(self, kv_cache_spec, layer_names, vllm_config, device)`.

The `FusenKVMetadata` dataclass and `build()` method also use the old API pattern
(with `common_prefix_metadata` and `common_attn_metadata` args). The v1 builder has
a different `build()` signature that receives `CommonAttentionMetadata`.

**Fix:** Rewrite `FusenKVMetadataBuilder` to match the v1 API.

### 2. Per-Layer Backend Selection Works via kv_cache_dtype_skip_layers

vLLM's `Attention.__init__` (in `model_executor/layers/attention/attention.py`)
supports `kv_cache_dtype_skip_layers` in CacheConfig:

```python
# If sliding_window layers should skip quantized KV:
--kv-cache-dtype-skip-layers sliding_window  # Skip all sliding layers
--kv-cache-dtype-skip-layers 0,1,2          # Skip specific layer indices
```

However, for FusenCache we want the OPPOSITE: we want ALL layers to use FusenCache,
both sliding and global. The skip mechanism can be used if we need to exclude certain
layers later, but the default (no skipping) applies FusenCache everywhere.

The backend selection is per-model, not per-layer. vLLM selects ONE attention backend
for all layers that share the same kv_cache_dtype. Gemma4 already works with this:
FlashAttention handles both sliding (D=256) and global (D=512) layers with the same
backend class. FusenCache needs to do the same.

### 3. NVFP4 + FusenCache is Orthogonal (CONFIRMED)

The compatibility matrix in `compatibility.py` already declares:

```python
(WEIGHT_QUANT_NVFP4, KV_K4V4): (True, "NVFP4 + FusenKV 4-bit, KV quant independent")
(WEIGHT_QUANT_NVFP4, KV_K8V4): (True, "NVFP4 + FusenKV K8V4, KV quant independent")
```

This is architecturally correct: weight quantization (CUTLASS FP4 GEMM) produces BF16
QKV projections; KV cache quantization operates on those BF16 outputs. The two systems
never interact at the kernel level.

### 4. Head Dimensions Supported

The FusenKV kernel supports head_dim in {64, 128, 256, 512} (see `supports_head_size`).
Gemma4 uses D=256 (sliding) and D=512 (global). Both are covered.

The Triton kernel uses `BLOCK_D = triton.next_power_of_2(D)`, so D=256 gives
BLOCK_D=256 and D=512 gives BLOCK_D=512. Both are power-of-2 and tested in the
sweep (Experiment 2 in RESULTS.md confirms correctness at both dims).

### 5. CUDA Graph Safety (NEEDS WORK)

The current FusenKVMetadataBuilder declares `_cudagraph_support = AttentionCGSupport.NEVER`.
This is a problem: our best config (6,615 tok/s) uses CUDA graphs.

The decode kernel itself IS cuda-graph safe when `persistent_buffers=True`:
- `make_decode_fn(..., cuda_graph_safe=True)` pre-allocates mid_out and output buffers
- No Python-level allocations in the hot path
- Triton kernels are graph-capturable

The store kernel uses a separate CUDA stream (`store_async`), which complicates graph
capture. For CUDA graph compatibility, we need:
- Option A: Run store synchronously on the default stream (simplest, slight perf hit)
- Option B: Pre-create the stream and include it in the graph (complex)

**Recommendation:** Start with Option A (sync store). The store kernel is fast
(~0.5ms overhead for 30 layers at D=256) and the decode kernel speedup from CUDA
graphs far outweighs the store overhead.

### 6. Prefill Path (SDPA Fallback)

FusenKV uses `torch.nn.functional.scaled_dot_product_attention` for prefill. This is
correct and efficient for prefill (which is compute-bound, not memory-bound). The
prefill SDPA handles sliding window implicitly via the attention mask.

However, FusenCache does NOT currently implement sliding window masking in prefill.
For Gemma4's sliding layers with window=1024, prefill tokens beyond the window should
be masked. The current code does `is_causal=True` which only applies a causal mask.

**Fix needed:** Apply sliding window mask during prefill for sliding layers.
For short prefills (<= window size), causal mask is sufficient. For long prefills,
need to combine causal + sliding window mask.

### 7. logits_soft_cap (BLOCKING)

Gemma4 uses `attn_logits_soft_cap` (value: 50.0 from config). The current FusenKVImpl
raises `NotImplementedError("FusenKV does not support logits soft cap yet")`.

**This blocks Gemma4 integration.** Soft capping applies `tanh(score / cap) * cap`
to attention logits before softmax. It must be implemented in the Triton decode kernel.

**Fix:** Add soft cap to the stage1 kernel after the QK^T dot product, before softmax.
This is a 3-line change in the kernel: `score = cap * tanh(score / cap)`.

---

## Implementation Plan

### Phase 1: Fix Blocking Issues (Required)

#### 1a. Add logits_soft_cap to Triton decode kernel

File: `kv_cache_gen/kernel.py`

Add `LOGITS_SOFT_CAP: tl.constexpr` parameter to `_universal_decode_stage1`.
After computing `qk_scale = tl.sum(q * k, axis=-1) * sm_scale`, apply:
```python
if LOGITS_SOFT_CAP > 0:
    qk_scale = LOGITS_SOFT_CAP * tl.math.tanh(qk_scale / LOGITS_SOFT_CAP)
```

Propagate through `make_decode_fn` in `generate.py`.

#### 1b. Remove soft_cap NotImplementedError from FusenKVImpl

File: `fusen_kv/backend.py`

Store `logits_soft_cap` and pass it to the decode kernel instead of raising.

#### 1c. Add sliding window support to prefill fallback

File: `fusen_kv/backend.py`

When `self.sliding_window is not None` and prefill length > window size:
```python
# Create sliding window + causal mask
attn_mask = torch.ones(n_prefill, n_prefill, device=query.device, dtype=torch.bool)
attn_mask = torch.triu(attn_mask, diagonal=1)  # causal
attn_mask |= torch.triu(torch.ones_like(attn_mask), diagonal=self.sliding_window)
```

### Phase 2: Update to v1 API (Required)

#### 2a. Rewrite FusenKVMetadataBuilder

File: `fusen_kv/backend.py`

The builder must:
1. Accept `(kv_cache_spec, layer_names, vllm_config, device)` in `__init__`
2. Set `_cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`
3. Implement `build(common_prefix_metadata, common_attn_metadata)` using
   the v1 `CommonAttentionMetadata` fields (block_table_tensor, seq_lens_tensor,
   slot_mapping, num_prefill_tokens, num_decode_tokens)
4. Implement `build_for_cudagraph_capture` returning metadata with max batch sizes

#### 2b. Make decode kernel CUDA-graph safe

File: `fusen_kv/backend.py`

In `FusenKVImpl.__init__`, pass `cuda_graph_safe=True` to `make_decode_fn()`.
Use synchronous store (no separate CUDA stream) for graph compatibility.

### Phase 3: Plugin Integration (Required)

#### 3a. Update backend selection patch for v1

File: `fusen_kv/plugin.py`

The current patch targets `CudaPlatform.get_attn_backend_cls` which is correct for v1.
But the method signature uses `AttentionSelectorConfig` and validation happens via
`validate_configuration()`. We need to ensure the CUSTOM backend passes validation
for FusenKV dtypes.

#### 3b. NVFP4 weight_quant detection

File: `fusen_kv/backend.py`

The `warn_if_untested` call in `FusenKVImpl.__init__` needs the `weight_quant` kwarg.
vLLM does not currently pass weight_quant to AttentionImplBase. Two options:
- Option A: Detect from vllm_config (preferred)
- Option B: Accept as extra_impl_args (requires vLLM model code change)

**Recommendation:** Detect weight quant from the global vllm_config:
```python
from vllm.config import get_current_vllm_config
quant_config = get_current_vllm_config().model_config.quantization
weight_quant = quant_config if quant_config else "none"
```

### Phase 4: Testing

#### 4a. Standalone kernel test

Test FusenKV decode + store with Gemma4 dimensions:
- Sliding: D=256, Hq=16, Hk=8, seq=1024
- Global: D=512, Hq=16, Hk=2, seq=8192
- Verify correctness (cosine sim > 0.99)

#### 4b. vLLM integration test

1. Build FusenKV as pip-installable package with entry_points
2. Install into vllm-gemma4 container
3. Start vLLM with `--kv-cache-dtype k4v4b64`
4. Run serving benchmark: throughput, quality, max concurrency

#### 4c. CUDA graph test

1. Start vLLM with CUDA graphs + FusenKV
2. Verify no tensor allocation in decode path
3. Benchmark vs non-graph baseline

---

## Expected Results

### KV Memory Savings

| Layer Type | BF16 (current) | k4v4b64 | Savings |
|-----------|----------------|---------|---------|
| Sliding (D=256, Hk=8, 1024 tok) | 2 MB/seq | 0.53 MB/seq | 3.8x |
| Global (D=512, Hk=2, 8192 tok) | 16 MB/seq | 4.2 MB/seq | 3.8x |
| **Total per seq (4096 ctx)** | ~12 MB | ~3.2 MB | 3.8x |

### Capacity Impact

- Current: 43K tokens, 15x concurrency at 4096 ctx
- With k4v4b64: ~82K tokens, ~28x concurrency at 4096 ctx (estimate)

### Throughput Projection

From RESULTS.md Experiment 3 (decode simulation):
- B=1: ~44 tok/s (MoE-bound, no change)
- B=64: ~2,000 tok/s (similar to BF16, less memory pressure)
- B=240: ~4,200 tok/s (FusenCache fits, BF16 would OOM)

The real gain is capacity: more concurrent sequences = higher aggregate throughput.
At 2x concurrency, even with identical per-request latency, aggregate tok/s doubles.

### Quality

From sweep results: k4v4b64 gives 0.991 cosine similarity vs BF16 KV.
This is imperceptible in text generation quality.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| logits_soft_cap not in kernel | HIGH (blocks all Gemma4) | Implement in kernel.py first |
| CUDA graph incompatibility | HIGH (4x slowdown without) | Use sync store, persistent buffers |
| v1 API mismatch in builder | HIGH (won't load) | Full rewrite of builder |
| Sliding window prefill mask | MEDIUM (wrong for long prefills) | Add mask; short prefills OK |
| Mixed head_dim layers | LOW (tested in sweep) | Both D=256 and D=512 work |
| NVFP4 interaction | LOW (confirmed orthogonal) | Weight quant is pre-KV-store |

---

## File Changes Summary

| File | Change | Priority |
|------|--------|----------|
| `kv_cache_gen/kernel.py` | Add LOGITS_SOFT_CAP constexpr | P0 |
| `kv_cache_gen/generate.py` | Pass logits_soft_cap to kernel | P0 |
| `fusen_kv/backend.py` | Rewrite builder + impl for v1 API, soft cap, sliding window | P0 |
| `fusen_kv/plugin.py` | Update for v1 registry + validation | P1 |
| `fusen_kv/compatibility.py` | Add k4v2 NVFP4 entries (nice to have) | P2 |
| `test_fusencache_nvfp4.py` | Integration test script | P0 |
