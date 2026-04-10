# vLLM Bug Report: MTP num_speculative_tokens > 1 crashes with CUDA illegal memory access at batch >= 4

## Summary

MTP (Multi-Token Prediction) speculative decoding with `num_speculative_tokens > 1` causes `cudaErrorIllegalAddress` when batch size >= 4 on hybrid Mamba/attention models (Qwen3.5) running on SM120 (RTX 5090 Blackwell). The crash occurs in the target model's forward pass on subsequent engine steps after `EagleProposer.propose()` runs the MTP drafting loop. `num_speculative_tokens=1` works correctly at all batch sizes.

## Environment

- **vLLM**: 0.17.1 (also reproduces on 0.17.0)
- **GPU**: NVIDIA RTX 5090 (SM120, Blackwell GB202)
- **Model**: `Kbenkhaled/Qwen3.5-9B-NVFP4` (hybrid attention + Mamba/delta_rule)
- **PyTorch**: 2.10
- **FlashInfer**: latest bundled with vLLM 0.17.1
- **CUDA**: 12.8

## Minimal Reproduction

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

llm = LLM(
    model="Kbenkhaled/Qwen3.5-9B-NVFP4",
    gpu_memory_utilization=0.90,
    max_num_seqs=128,
    max_model_len=4096,
    enable_chunked_prefill=False,
    speculative_config={
        "num_speculative_tokens": 3,  # 1 works fine, 2+ crashes
        "method": "qwen3_5_mtp",
    },
)
sp = SamplingParams(max_tokens=100, temperature=0.0)

# Warmup
llm.generate(["Hello"], sp)
llm.generate(["Hello"], sp)

# batch=2 works
llm.generate(["Hello"] * 2, sp)  # OK

# batch=4 crashes
llm.generate(["Hello"] * 4, sp)  # CUDA illegal memory access
```

## Crash Behavior Matrix

| Config | batch=1 | batch=2 | batch=4 | batch=8 | batch=32 |
|--------|---------|---------|---------|---------|----------|
| `num_speculative_tokens=1` | OK | OK | OK | OK | OK |
| `num_speculative_tokens=2` | OK | OK | OK | CRASH | - |
| `num_speculative_tokens=3` | OK | OK | CRASH | - | - |
| `num_speculative_tokens=3` + `enforce_eager` | OK | OK | OK* | CRASH | - |
| `num_speculative_tokens=3` + `CUDA_LAUNCH_BLOCKING=1` | OK | OK | OK | CRASH | - |

\* Non-deterministic — sometimes passes, sometimes crashes at batch=4.

## Detailed Investigation

### What the error looks like

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The crash is reported at `gpu_model_runner.py:251` in `async_copy_ready_event.synchronize()`, but this is asynchronous error reporting. The actual OOB write occurs during a GPU kernel launched earlier.

### Isolation tests performed

We systematically isolated the crash by monkey-patching `EagleProposer.propose()`:

| Test | What was changed | Result | Conclusion |
|------|-----------------|--------|------------|
| Skip `propose()` entirely (return dummy tokens) | No MTP head forward, no metadata changes | batch=32 OK | Bug is inside `propose()` |
| Set `num_speculative_tokens=1` before calling `propose()` | Early return at line 473, no drafting loop | batch=32 OK | Bug requires MTP >1 code path |
| Run first pass only + manually replicate metadata setup | First MTP forward runs, no loop | batch=32 OK | Bug is NOT in first pass forward |
| Skip drafting loop (set iterations=0) but keep setup code | Lines 515-524 run (metadata mutation) | batch=4 CRASH | Bug triggered by setup code OR loop |
| Deep-copy `common_attn_metadata` before `propose()` | Prevent metadata corruption of main model | batch=4 CRASH | NOT a metadata corruption bug |
| Replace MTP head with no-op (return zeros) | No real model computation, but all metadata/context ops run | batch=4 OK, batch=8 CRASH | Partially helps — model forward contributes but isn't sole cause |
| Disable TRT-LLM decode attention | Route spec decode through FlashInfer native | batch=4 CRASH | Not specific to TRT-LLM kernel |
| Disable `spec-as-decode` (force `reorder_batch_threshold=1`) | Route multi-token verification through prefill path | batch=4 CRASH | Not a decode-vs-prefill routing issue |

### Key finding: the crash correlates with KV cache state corruption

The difference between "works" and "crashes" is whether `propose()` executes the code path that runs the MTP head forward WITH `set_forward_context()`. The `set_forward_context()` establishes attention metadata and slot mappings that the attention layer's KV cache update operation (`do_kv_cache_update` / `reshape_and_cache_flash`) uses to write KV entries.

**Hypothesis**: The MTP head's KV cache writes corrupt shared KV cache state that the target model reads on the next engine step. This is specific to:
1. Hybrid models (Mamba + attention) where the block_size is unusually large (544 tokens) due to Mamba page size alignment
2. `num_speculative_tokens > 1` which changes `uniform_decode_query_len` from 2 to 4, affecting CUDA graph capture sizes and attention metadata structure

### Block size computation

When `num_speculative_tokens > 1`, the Mamba convolution state size increases (extra tokens in conv kernel), which increases `mamba_page_size`, which forces a larger attention `block_size` to ensure page alignment:

```
num_speculative_tokens=0: block_size=528
num_speculative_tokens=1: block_size=528
num_speculative_tokens=2: block_size=544
num_speculative_tokens=3: block_size=544
```

However, forcing `block_size=1024` (power of 2) still crashes, so the block size itself is not the direct cause.

### Attention backends involved

The Qwen3.5-9B model uses:
- **GDNAttentionMetadataBuilder** (kv_cache groups 0-2): for delta_rule/linear attention layers (Mamba-like)
- **FlashAttentionMetadataBuilder** (kv_cache group 3): for standard self-attention layers

Neither uses FlashInfer's TRT-LLM decode path on this model. The `reorder_batch_threshold=4` in GDN allows spec decode batches (4 tokens/request) to be treated as "decode".

### What `num_speculative_tokens=1` does differently

With MTP 1:
- `uniform_decode_query_len = 2`
- `propose()` takes early return at line 473 (before any metadata mutation or drafting loop)
- CUDA graph capture sizes: `[2, 4, 8, ...]`
- MTP head runs once per step (prefill pass only)

With MTP 3:
- `uniform_decode_query_len = 4`
- `propose()` runs the full drafting loop (2 iterations)
- CUDA graph capture sizes: `[4, 8, 16, ...]`
- MTP head runs 3 times per step (1 prefill + 2 decode iterations)
- The drafting loop modifies `common_attn_metadata.seq_lens` in-place (`+= 1` per iteration)
- The drafting loop rebuilds attention metadata via `build_for_drafting()` each iteration

### Additional bug: in-place mutation of shared metadata

`eagle.py` line 567: `common_attn_metadata.seq_lens += 1` modifies the main model's persistent `seq_lens` GPU buffer in-place. The comment at line 563-564 acknowledges this is unsafe for async scheduling but doesn't fix it. This is a secondary bug (data race) separate from the primary OOB crash.

Similarly, lines 519-524 modify `common_attn_metadata` fields (`num_actual_tokens`, `max_query_len`, `query_start_loc`) in-place on the shared object. These should use a copy.

### FlashInfer native decode bug (separate)

In `flashinfer.py` `build()` line 1127:
```python
indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1]
```

`paged_kv_indptr` has `num_reqs + 1` valid entries (indexed by requests). But `num_input_tokens = num_decode_tokens = q_len_per_req * num_reqs`. When `q_len_per_req > 1` (spec decode verification), `num_input_tokens > num_reqs` and this reads past the valid portion of the buffer. This doesn't affect Qwen3.5 (which uses FlashAttention, not FlashInfer), but would crash other models using FlashInfer with MTP >1.

## Root Cause (Confirmed)

The crash is in the **ShortConv and GatedDeltaNet layers** (Mamba/delta_rule hybrid attention) when processing multi-token speculative decode verification batches. Two confirmed bugs:

### Bug 1: `short_conv.py` passes wrong-shape `conv_state_indices` to `causal_conv1d_update`

**File:** `vllm/model_executor/layers/mamba/short_conv.py`, line 186-195

When `num_speculative_tokens > 1`, `state_indices_tensor_d` has shape `[num_requests, 1+num_spec]` (e.g., `[4, 4]` for MTP 3 with batch=4). The code passes this directly to `causal_conv1d_update`, which flattens it to `[16]` and uses each entry as a conv state cache line index. But the 16 entries contain speculative block IDs (columns 1-3) that are NOT the base conv state for each request. Tokens get mapped to wrong conv states, and some block IDs may point to unallocated cache lines → OOB write.

**Fix:** For regular (non-spec) decode, use only column 0: `state_indices_tensor_d[:, 0]`. For spec verification decode, pass `query_start_loc_d` and `num_accepted_tokens` (like `mamba_mixer2.py` does).

```python
# In short_conv.py, replace the decode section:
if has_decode:
    Bx_d = (B_d * x_d).contiguous()
    query_start_loc_d = attn_metadata.query_start_loc_d
    num_accepted_tokens = attn_metadata.num_accepted_tokens
    if query_start_loc_d is not None:
        # Spec decode verification
        Bx = causal_conv1d_update(
            Bx_d, conv_state, conv_weights, self.conv.bias,
            activation=None,
            conv_state_indices=state_indices_tensor_d,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=query_start_loc_d,
            max_query_len=state_indices_tensor_d.size(-1),
        )
    else:
        # Regular single-token decode
        Bx = causal_conv1d_update(
            Bx_d, conv_state, conv_weights, self.conv.bias,
            activation=None,
            conv_state_indices=state_indices_tensor_d[:, 0],
        )
```

### Bug 2: FULL CUDA graphs bake stale Mamba state indices

The `CUDAGraphMode.FULL_AND_PIECEWISE` mode captures the entire model forward (including `short_conv` and `linear_attention` splitting ops) as a single CUDA graph during warmup. The captured graph contains fixed `state_indices_tensor_d` values from the warmup run. During replay, these stale indices are used instead of the real per-step indices, causing OOB writes.

**Fix:** For hybrid Mamba models with MTP > 1, force `enforce_eager=True` (disabling both torch.compile and CUDA graphs). This must be set BEFORE engine initialization via the LLM constructor, not via config patching.

### Bug 3 (secondary): `qwen3_next.py` non-spec decode path slices OOB

**File:** `vllm/model_executor/models/qwen3_next.py`, line 721-722

```python
conv_state_indices=non_spec_state_indices_tensor[:attn_metadata.num_actual_tokens]
```

`non_spec_state_indices_tensor` has shape `[num_reqs]` but `num_actual_tokens` can exceed `num_reqs` when multi-token queries are classified as non-spec decode.

## Suggested fixes

### 1. Short-term: in-place metadata mutation (eagle.py)

Clone `common_attn_metadata` before the drafting loop to prevent corrupting the main model's buffers:

```python
# Before line 519 in eagle.py propose():
from copy import copy
common_attn_metadata = copy(common_attn_metadata)
common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
common_attn_metadata._seq_lens_cpu = None
common_attn_metadata._num_computed_tokens_cpu = None
```

### 2. Short-term: FlashInfer paged_kv_indptr (flashinfer.py)

Use `num_decodes` instead of `num_input_tokens` for paged KV metadata:

```python
# Line 1127 in flashinfer.py:
num_kv_reqs = num_decodes
fast_plan_decode(
    decode_wrapper,
    indptr_cpu=self.paged_kv_indptr.cpu[: num_kv_reqs + 1],
    ...
    last_page_len_cpu=self.paged_kv_last_page_len.cpu[:num_kv_reqs],
    ...
)
```

### 3. Long-term: investigate KV cache corruption

The primary crash requires deeper investigation into how the MTP head's KV cache writes interact with the hybrid Mamba/attention model's block table management on SM120. The crash persists even with metadata isolation, suggesting the corruption is in the KV cache tensor itself (written by `reshape_and_cache_flash` during the MTP forward) or in the compiled CUDA graph state.

## Workaround

Use `num_speculative_tokens=1` for batch serving:

```python
llm = LLM(
    model="Kbenkhaled/Qwen3.5-9B-NVFP4",
    speculative_config={
        "num_speculative_tokens": 1,  # max that works with batching
        "method": "qwen3_5_mtp",
    },
)
```

Performance with MTP 1 on RTX 5090:
- Decode: 157 tok/s (+30% over no-MTP baseline of 121 tok/s)
- Batch 32: 2,703 tok/s (stable)
- No crashes at any batch size

MTP 3 remains viable for single-request latency optimization only (179 tok/s, +48%).

## Related

- vLLM warning at `speculative.py:487`: "Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate" — acknowledges this is an edge case
- The bug does NOT reproduce with pure-attention models (only hybrid Mamba/attention)
- The bug does NOT reproduce with `num_speculative_tokens=1`
- `CUDA_LAUNCH_BLOCKING=1` shifts the crash boundary but does not eliminate it
