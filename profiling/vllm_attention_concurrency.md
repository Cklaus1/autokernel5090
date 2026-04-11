# vLLM Attention Backend Concurrency Analysis

## Investigation: Why FlashInfer crashes at C=16+ with FusenCache + CUDA Graphs

### Environment
- RTX 5090 (SM120, CC 12.0)
- vLLM v1 with FlashInfer attention backend
- Gemma4 27B: 25 sliding-window layers (window=1024) + 5 full-attention layers
- FusenKV registered as CUSTOM backend (handles KV store + decode attention)
- FlashInfer handles the 25 sliding-window layers

### Key Discovery: SM120 Does NOT Use TRTLLM Attention

`supports_trtllm_attention()` returns False for SM120 (only validated for SM100/CC 10.0).
This means ALL FlashInfer attention on RTX 5090 uses the **native FlashInfer path**:
- `BatchDecodeWithPagedKVCacheWrapper` for decode
- `BatchPrefillWithPagedKVCacheWrapper` for prefill
- CUDA graphs only for **pure decode** batches

### Architecture: How vLLM Manages Heterogeneous Attention

**KV Cache Groups:**
Gemma4's 30 layers split into separate KV cache groups because
`FullAttentionSpec` and `SlidingWindowSpec` are different types and cannot merge.

**Attention Groups within init_attn_backend (attn_utils.py:43-89):**
Each KV cache group gets sub-grouped by `(backend_class_name, kv_cache_spec)`.
Result: at least 2 attention groups per KV cache group (FusenKV and FlashInfer),
plus potentially separate groups for sliding vs full attention within FlashInfer.

**CRITICAL: Shared Workspace Buffer (attn_utils.py:83-88):**
```python
if attn_backend_workspace is None:
    if hasattr(builder, "_get_workspace_buffer"):
        attn_backend_workspace = builder._get_workspace_buffer()
else:
    if hasattr(builder, "set_workspace_buffer"):
        builder.set_workspace_buffer(attn_backend_workspace)
```
The FIRST FlashInfer builder allocates a 394 MB workspace buffer.
ALL subsequent FlashInfer builders across ALL KV cache groups get the SAME buffer.

### The Pure Decode vs Mixed Batch Divergence

**At C=1-8 (Pure Decode):**
```python
pure_decode = num_prefills == 0  # True
use_cudagraph = self.enable_cuda_graph and pure_decode and ...  # True
```
- All requests are in decode mode (query_len=1)
- CUDA graphs capture and replay the decode kernel
- FlashInfer uses `_decode_wrappers_cudagraph[batch_size]` (per-batch-size cached)
- `fast_plan_decode()` updates indptr/indices/last_page_len via fast H2D copy
- The captured graph replays with updated metadata buffers

**At C=16+ (Mixed Prefill+Decode):**
```python
pure_decode = num_prefills == 0  # False (some requests still prefilling)
use_cudagraph = ...  # False
```
- Some requests are prefilling while others decode
- CUDA graphs are DISABLED for the decode path
- Both `_prefill_wrapper` and `_decode_wrapper` are used
- The non-CUDA-graph `_decode_wrapper` is a DIFFERENT wrapper than the CG ones
- ALL wrappers share the SAME 394 MB workspace buffer

### Root Cause Hypotheses (Ranked by Likelihood)

#### Hypothesis 1: Mixed Batch Creates Out-of-Bounds Block Table Access
**Likelihood: HIGH**

When FusenKV handles mixed prefill+decode (backend.py:812-901):
1. It iterates requests via `qsl[i]` to find prefill vs decode
2. For decode requests, it gathers block tables: `dec_block_table = attn_metadata.block_table[dec_idx]`
3. The `block_table` in `FusenKVMetadata` comes from `CommonAttentionMetadata.block_table_tensor`

The block_table_tensor is from KV cache group 0 (lines 2191-2192 of gpu_model_runner.py).
For group 1+, it's updated via `cm.block_table_tensor = _get_block_table(kv_cache_gid)`.

**But FusenKV's `update_block_table()` creates a NEW metadata with the updated block_table.**
So each group gets the correct block table. This hypothesis is likely NOT the cause.

#### Hypothesis 2: Workspace Buffer Corruption Between Groups
**Likelihood: MEDIUM**

All FlashInfer builders share the same workspace buffer. The workspace is used as
scratch space during both `plan()` and `run()`. In a mixed batch:

1. FlashInfer Group A (sliding) builder.build() calls `prefill_wrapper.plan()` + `decode_wrapper.plan()`
2. FlashInfer Group B (full) builder.build() calls `prefill_wrapper.plan()` + `decode_wrapper.plan()`
3. Layer 0 (sliding) forward: `prefill_wrapper.run()` -- workspace contains Group B's plan data

If FlashInfer uses the workspace buffer to persist plan data between plan() and run(),
this would cause corruption. However, FlashInfer wrappers typically store plan data
internally, using workspace only as scratch during run(). Sequential layer execution
should prevent concurrent workspace access.

**Risk increases at C=16+ because mixed batches cause both prefill AND decode wrappers
to be active, doubling workspace usage within a single builder.**

#### Hypothesis 3: Non-Blocking H2D Copy Race Condition
**Likelihood: HIGH**

The `_compute_flashinfer_kv_metadata` function (flashinfer.py:787-842) uses:
```python
paged_kv_indptr.copy_(self.paged_kv_indptr_cpu_buffer[...], non_blocking=True)
self.paged_kv_last_page_len.gpu[...].copy_(..., non_blocking=True)
```

These are non-blocking copies from pinned CPU memory. At C=1-8, the CUDA graph
path uses `fast_plan_decode()` which does its own H2D copies. At C=16+, the
eager path uses these non-blocking copies, and if the scheduler calls
`build()` again (for the next step) before the previous copy completes,
the CPU source buffer gets overwritten.

vLLM explicitly acknowledges this risk (flashinfer.py:649-651):
```python
# Since we do not have explicit synchronization in ModelRunnerV2, we do not pin
# reused CPU buffers to avoid a race condition between step N async copies to
# GPU and step N+1 buffer updates.
```
For V1 model runner, `pin_memory=True` by default. The race condition comment
applies to V2 but the same buffers are used in V1.

#### Hypothesis 4: FusenKV Shared Buffer Size Mismatch at Low Batch Counts
**Likelihood: MEDIUM-HIGH**

FusenKV's `_shared_mid_out` is allocated at `_max_B` size (from max_cudagraph_capture_size
or max_num_seqs). The `_num_kv_splits` is computed from `_optimal_splits(max_seq, 1)` -- the
maximum possible splits (for B=1).

In mixed prefill+decode (backend.py:855-901), decode requests are gathered:
```python
n_dec = len(decode_indices)
dec_queries = query[dec_token_starts].contiguous()
attn_out = _decode(dec_queries, kv_cache, ...)
```

If `n_dec` exceeds `_max_B`, the shared output buffer would overflow. At C=16 with
many concurrent requests, `n_dec` could exceed the capture size.

More subtly: `_optimal_splits(max_seq, n_dec)` may return MORE splits than the
pre-allocated buffer dimension when `n_dec` is small (e.g., only 2 decode requests
in a mostly-prefill batch). But the code caps this:
```python
num_kv_splits = min(self._cpp_num_kv_splits, _optimal_splits(self._max_seq, B))
```
This should prevent overflow... unless `B` is passed differently.

#### Hypothesis 5: FlashInfer Sliding Window + Full Attention Global Parameters Conflict
**Likelihood: LOW**

Each FlashInfer metadata builder computes `global_hyperparameters` from its group's layers.
The sliding group gets `window_left=1023`, the full group gets `window_left=-1`.
These are separate builders with separate wrappers, so `plan()` uses the correct
window_left per group. The assert `has_same_window_lefts` only checks within a group.
This should work correctly.

### Recommended Next Steps

1. **Add CUDA synchronization barrier between metadata build and forward pass**
   to test Hypothesis 3:
   ```python
   # In gpu_model_runner.py, after build_attn_metadata
   torch.cuda.synchronize()
   ```

2. **Log batch composition at C=16**: Add logging to show num_decodes vs
   num_prefills when crashes occur. Test if crash always happens on the first
   mixed batch.

3. **Test with `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` doubled** to test
   Hypothesis 2 (workspace overflow in mixed batches).

4. **Test with `--enable-chunked-prefill=false`** to prevent mixed batches
   entirely. If crashes stop, it confirms the mixed batch is the trigger.

5. **Verify FusenKV shared buffer bounds**: Add assertion that `n_dec <= _max_B`
   in the mixed path (backend.py:857).

### Files Examined

| File | Location | Key Finding |
|------|----------|-------------|
| flashinfer.py | /build/vllm/vllm/v1/attention/backends/flashinfer.py (1789 lines) | Shared workspace buffer, pure_decode gate for CUDA graphs |
| flash_attn.py | /build/vllm/vllm/v1/attention/backends/flash_attn.py (1207 lines) | Not used on SM120 for Gemma4 |
| attn_utils.py | /build/vllm/vllm/v1/worker/gpu/attn_utils.py | Workspace buffer sharing across ALL groups |
| workspace.py | /build/vllm/vllm/v1/worker/workspace.py | Per-ubatch workspace (not FlashInfer workspace) |
| gpu_model_runner.py | /build/vllm/vllm/v1/worker/gpu_model_runner.py | Block table per group, metadata build flow |
| gemma4.py | /build/vllm/vllm/model_executor/models/gemma4.py | Sliding vs full attention layer types |
| utils.py | /build/vllm/vllm/v1/attention/backends/utils.py | split_decodes_and_prefills, global_hyperparameters |
| flashinfer_utils.py | /build/vllm/vllm/utils/flashinfer.py | TRTLLM not supported on SM120 |
| backend.py | /root/projects/autokernel/fusen_kv/backend.py (903 lines) | FusenKV shared decode buffers, mixed path |
| plugin.py | /root/projects/autokernel/fusen_kv/plugin.py (294 lines) | Plugin registration, dtype patches |
