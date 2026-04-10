# PR: Fix FullAttentionSpec.max_memory_usage_bytes() to respect sliding_window

## Summary

When `disable_hybrid_kv_cache_manager=True`, `unify_hybrid_kv_cache_specs()` converts `SlidingWindowSpec` layers to `FullAttentionSpec` while preserving the `sliding_window` field. However, `FullAttentionSpec.max_memory_usage_bytes()` ignores the `sliding_window` field entirely, causing the memory check to allocate `max_model_len` tokens for every layer — even those that only need `sliding_window` tokens.

This causes hybrid models like Gemma 4 (50 sliding window + 10 full attention layers) to vastly overestimate KV cache memory requirements, preventing long context deployment on single GPUs.

## Root Cause

`vllm/v1/kv_cache_interface.py` line 113:

```python
def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    # ... no sliding_window check ...
    return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

The `sliding_window` field exists on `FullAttentionSpec` (line 103) and is correctly populated by `unify_hybrid_kv_cache_specs()` (line 1199), but `max_memory_usage_bytes()` never reads it.

## Fix

```python
def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    # Respect sliding window: layers converted from SlidingWindowSpec
    # only need sliding_window tokens, not the full context length.
    if self.sliding_window is not None:
        max_model_len = min(self.sliding_window, max_model_len)
    ...
```

## Impact

Tested on Gemma 4 31B AWQ (RTX 5090, 32GB):

| Config | Before Fix | After Fix |
|--------|-----------|-----------|
| Max context (disable_hybrid) | 64K | **93K** (+45%) |
| Memory estimate (128K) | 120 GiB | 10.78 GiB |

On Gemma 4 26B-A4B MoE, this enables 128K context on a single RTX 5090.

## Files Changed

- `vllm/v1/kv_cache_interface.py`: 2 lines added to `FullAttentionSpec.max_memory_usage_bytes()`

## Test

```python
from vllm import LLM
llm = LLM(
    model="google/gemma-4-31B-it-AWQ-4bit",
    max_model_len=95360,  # was limited to 65536 before fix
    disable_hybrid_kv_cache_manager=True,
    gpu_memory_utilization=0.92,
    trust_remote_code=True,
    enforce_eager=True,
)
# Previously failed with: "17.04 GiB KV cache needed, 8.06 GiB available"
```
