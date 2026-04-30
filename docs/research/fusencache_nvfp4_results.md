# FusenCache + NVFP4 Integration Results

**Date:** 2026-04-10
**Target:** Gemma4 26B NVFP4 on RTX 5090, FusenCache k4v4b64 KV compression
**Status:** PARTIALLY WORKING -- server loads, 4x KV capacity achieved, decode quality broken

---

## Summary

FusenCache was successfully integrated into vLLM 0.19.1rc1 as a custom attention backend plugin serving Gemma4 26B with NVFP4 weight quantization. The server starts, allocates 4x more KV cache capacity (175K vs 43K tokens), and accepts requests. However, text generation produces incorrect/degraded output due to issues in the attention decode path.

## Results

### KV Cache Capacity (SUCCESS)

| Metric | Baseline (BF16 KV) | FusenCache (k4v4b64) | Change |
|--------|-------------------|---------------------|--------|
| KV cache tokens | 43,712 | 175,072 | **4.0x** |
| Max concurrency (4096 ctx) | 15.0x | 60.07x | **4.0x** |
| Available KV memory | 10.01 GiB | 10.02 GiB | Same |
| Model memory | 17.24 GiB | 17.24 GiB | Same |

### Text Generation Quality (BROKEN)

- Short prompts (1-3 tokens): Generates tokens but output is wrong (repeats/empty)
- Medium prompts (5+ tokens): Produces NaN logits (logits_soft_cap not applied in prefill SDPA)
- The 4-bit quantized decode attention is not producing correct attention outputs

### Server Startup (SUCCESS)

- Server starts in ~10 seconds with `--enforce-eager`
- CUDA graphs: Auto-degraded from FULL to FULL_DECODE_ONLY (expected for custom backend)
- No crashes during startup or inference (except NaN from quality issues)

## Integration Issues Fixed

### 1. Argparse Validation (Fixed)
vLLM's `EngineArgs` uses a pydantic dataclass with frozen Literal choices. Custom dtype strings like "k4v4b64" are rejected at CLI parse time.
**Fix:** Launcher script (`launch_vllm.py`) swaps dtype to "auto" before argparse, then restores it after.

### 2. Pydantic CacheConfig Validation (Fixed)
`CacheConfig.__init__` uses pydantic v2 compiled validators that reject custom dtypes.
**Fix:** Monkey-patch `CacheConfig.__init__` to temporarily use "auto" during pydantic validation, then `object.__setattr__` the real dtype back.

### 3. Backend Name Lookup (Fixed)
vLLM does `AttentionBackendEnum[backend.get_name()]` to map backend to enum member. Our backend returned "FUSEN_KV" but the enum only has "CUSTOM".
**Fix:** `FusenKVBackend.get_name()` returns "CUSTOM" to match the registered enum slot.

### 4. Page Size Calculation (Fixed)
vLLM's `real_page_size_bytes` assumes standard K+V layout with `dtype_size * 2 * head_size`. For 4-bit K+V, the real page size is 4x smaller.
**Fix:** Monkey-patched `real_page_size_bytes` property on `AttentionSpec` and `FullAttentionSpec` to use `slot_bytes` formula when dtype is uint8 (FusenCache marker).

### 5. v1 API Signatures (Fixed)
- `build()` now takes `(common_prefix_len, common_attn_metadata, fast_build=False)`
- `build_for_cudagraph_capture()` takes `(common_attn_metadata)`
- `update_block_table()` implemented for multi-group KV cache
- `CommonAttentionMetadata` fields: `seq_lens` (not `seq_lens_tensor`), `query_start_loc`, `num_reqs`, `max_query_len`

### 6. STR_DTYPE_TO_TORCH_DTYPE Mapping (Fixed)
Added all FusenCache dtype strings (k4v4b64, k8v4b32, etc.) mapped to `torch.uint8`.

### 7. 3D Output Tensor (Fixed)
vLLM v1 passes output as `[num_tokens, num_heads, head_size]` (3D), not `[num_tokens, num_heads * head_size]` (2D).

## Remaining Issues (Blocking)

### A. Decode Quality (CRITICAL)
The 4-bit quantized KV decode kernel produces incorrect attention outputs. This could be due to:
1. **Mismatch between vLLM's slot_mapping and FusenCache's expected layout** -- The slot_mapping indexes into pages differently than FusenCache expects
2. **Scales tensor not properly shared across layers** -- Each layer needs its own scales tensor allocated and passed correctly
3. **Block size mismatch** -- vLLM unified block size to 32 for global layers (D=512) but FusenCache's store/decode kernels may assume block_size=16

### B. Prefill Soft Cap (HIGH)
Gemma4's `logits_soft_cap=50.0` is not applied in the SDPA prefill path. This causes NaN attention scores for longer sequences. Fix requires implementing manual attention with soft_cap in the prefill path.

### C. CUDA Graphs (MEDIUM)
Currently running with `--enforce-eager`. CUDA graph capture needs testing once decode quality is fixed.

## Files Modified

| File | Changes |
|------|---------|
| `fusen_kv/plugin.py` | Added 6 monkey-patches for vLLM integration (dtype, validation, backend selection, page sizing) |
| `fusen_kv/backend.py` | Fixed `get_name()` to return "CUSTOM", updated v1 API signatures, 3D output handling, mixed prefill/decode logic |
| `fusen_kv/launch_vllm.py` | New file: launcher that registers plugin before vLLM argparse |

## Architecture Notes

The integration required extensive monkey-patching because vLLM v1's plugin system assumes:
1. Plugins are pip-installed with entry_points (not just mounted)
2. Custom KV dtypes must be declared in pydantic Literals at class definition time
3. Page size follows standard K+V formula (no sub-byte support)
4. Backend names must be valid enum members

A proper upstream integration would require:
- Adding FusenCache dtypes to vLLM's CacheDType Literal
- Adding a `page_size_override` field to AttentionSpec for custom backends
- Supporting custom backend discovery via PYTHONPATH (not just entry_points)
