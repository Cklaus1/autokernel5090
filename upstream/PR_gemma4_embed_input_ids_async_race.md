# Upstream PR: Fix CUDA illegal-address in Gemma4MultiModal.embed_input_ids under concurrent batched inference

## PR Title

fix(models/gemma4_mm): break `is_multimodal` buffer aliasing and skip PLE mask when no multimodal inputs are present

## PR Body

### Summary

- `Gemma4ForConditionalGeneration.embed_input_ids()` crashes with `CUDA error:
  an illegal memory access was encountered` once the batch contains more than
  a few concurrent text-only requests.
- Root cause is the `is_multimodal` argument: it is a **slice view** into
  `gpu_model_runner.is_mm_embed_buffers[i].gpu`, a buffer that is
  double-buffered on the CPU side only and re-used every other iteration on
  the GPU side. With async scheduling and `cudagraph_mode=FULL`, iteration
  N+2's `copy_to_gpu(..., non_blocking=True)` can overwrite the same GPU
  bytes while iteration N's in-flight CUDA-graph replay is still reading
  them from `torch.where(is_multimodal, zeros_like(input_ids), input_ids)`.
- Fix: (1) clone `is_multimodal` into a fresh tensor so downstream kernels
  no longer alias the shared scheduler buffer, and (2) take the PLE-mask
  path only when real multimodal embeddings are actually present - for
  text-only batches we now match Gemma3n's behavior and feed `input_ids`
  straight into `get_per_layer_inputs`. The masking kernel (and therefore
  the race-prone buffer read) is skipped entirely for text-only traffic.

### Symptom

```
  File "/build/vllm/vllm/v1/worker/gpu_model_runner.py", line 3984, in execute_model
      ) = self._preprocess(...)
  File "/build/vllm/vllm/v1/worker/gpu_model_runner.py", line 3226, in _preprocess
      inputs_embeds_scheduled = self.model.embed_input_ids(...)
  File "/build/vllm/vllm/model_executor/models/gemma4_mm.py", line 1257, in embed_input_ids
      is_multimodal = is_multimodal.to(input_ids.device)
  torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

The `.to()` line is where the fault surfaces - it is the next CUDA API call
after the racy `torch.where` kernel in the previous iteration.

### Trigger

- Gemma4 (any size) served through the v1 engine.
- `--async-scheduling` **on** (the default with `VLLM_USE_V1=1`).
- `cudagraph_mode=FULL`.
- Prefix caching + chunked prefill **on**.
- Concurrency ~>=4, with a mixed prefill+decode batch (scheduler example:
  one cached req at 144 computed tokens + four new reqs at 32 scheduled
  prompt tokens each, `total_num_scheduled_tokens=65`).
- Reproduces fastest with a slow attention backend (quantized KV cache,
  custom backends) because the wider forward window gives iteration N+2's
  prep more time to stomp on iteration N's inputs. Flash-Attention usually
  hides the race.
- Text-only requests trigger it most reliably: when `_gather_mm_embeddings`
  has no `mm_features` to process, `is_mm_embed` is an all-False view but
  the PLE code path still reads it.

### Root cause

`vllm/v1/worker/gpu_model_runner.py` creates a double-buffer pair for the
multimodal mask (intended to hide the prep-vs-forward race):

```python
# gpu_model_runner.py  (around line 721)
self.is_mm_embed_buffers = [
    self._make_buffer(self.max_num_tokens, dtype=torch.bool),
    self._make_buffer(self.max_num_tokens, dtype=torch.bool),
]
self.is_mm_embed_idx = 0
```

In `_gather_mm_embeddings` (~line 2910) the CPU side is swapped between
writes:

```python
self.is_mm_embed_idx = 1 - self.is_mm_embed_idx
is_mm_embed_buf = self.is_mm_embed_buffers[self.is_mm_embed_idx]
...
is_mm_embed = is_mm_embed_buf.copy_to_gpu(total_num_scheduled_tokens)
return mm_embeds, is_mm_embed
```

`CpuGpuBuffer.copy_to_gpu` is a **non-blocking** H2D and returns
`self.gpu[:n]` - a *view* into the buffer's GPU storage. The swap hides the
CPU race, but every other iteration writes into the same two GPU buffers.
Under the conditions above:

1. Iter N prep writes `buf[0].cpu`, launches `buf[0].copy_to_gpu(65)`.
2. Iter N forward (CUDA graph replay) reads `is_mm_embed_buf[0].gpu[:65]`
   inside `Gemma4MultiModal.embed_input_ids` via
   `torch.where(is_multimodal, torch.zeros_like(input_ids), input_ids)`.
3. With `--async-scheduling`, iter N+1 CPU prep starts immediately and uses
   `buf[1]`.
4. Iter N+2 prep swings back to `buf[0]`. Its non-blocking H2D writes
   `buf[0].gpu[:new_n]` while iter N's `torch.where` kernel (launched into
   the CUDA graph) is still reading the same storage. Depending on
   `new_n >= 65`, this also corrupts the bytes iter N's kernel is mid-read.
5. Illegal memory access. The error materialises at the next CUDA call,
   which is the `.to(input_ids.device)` *on the following iteration*,
   matching the traceback.

Two aggravating factors in Gemma4 specifically:

- `embed_input_ids` enters the PLE (per-layer-embedding) path whenever
  `self.per_layer_embeddings is not None`, even for text-only requests.
- It unconditionally runs the `torch.where(is_multimodal, ...)` mask even
  when `multimodal_embeddings` is an empty list. Gemma3n's equivalent
  method does **not** mask - it feeds `input_ids` straight into
  `get_per_layer_input_embeddings(input_ids)`. The Gemma4 port introduced
  the extra masking and with it the racy buffer read.

### Related prior art

This is the same failure *class* as the async-scheduling race fixed in
Discovery #54 and the companion PR *"fix(v1): synchronize model forward
completion before next step's input prep"* (`patches/vllm_async_scheduling_fix.py`),
but a **different code path**. Discovery #54 fences `input_ids`/`block_table`
between steps in `gpu_model_runner.synchronize_input_prep()`. It does **not**
cover the `is_mm_embed_buffers` buffers, and it does not cover reads that
happen **inside** `_preprocess` (which already runs under the prep-event
context manager). So even with Discovery #54 applied, Gemma4 still crashes
on this path.

### The fix (minimal diff)

`vllm/model_executor/models/gemma4_mm.py`, inside `embed_input_ids`:

```diff
-        if self.per_layer_embeddings is not None:
-            # Mask multimodal tokens (image/audio) to 0 for PLE
-            # computation (using token_type_ids == 0 as text_mask).
-            # Replicate this: map image token positions to token 0.
-            if is_multimodal is not None:
-                is_multimodal = is_multimodal.to(input_ids.device)
-                ple_input_ids = torch.where(
-                    is_multimodal, torch.zeros_like(input_ids), input_ids
-                )
-            else:
-                ple_input_ids = input_ids
+        if self.per_layer_embeddings is not None:
+            # Mask multimodal tokens (image/audio) to 0 for PLE
+            # computation (using token_type_ids == 0 as text_mask).
+            # Replicate this: map image token positions to token 0.
+            #
+            # Only run the mask kernel when real multimodal embeddings are
+            # present. For text-only batches, `is_multimodal` is a slice
+            # view into the gpu_model_runner's shared `is_mm_embed_buffers`
+            # double-buffer, and reading it concurrently with the next
+            # iteration's non-blocking H2D copy produces a CUDA illegal
+            # memory access under async scheduling + CUDA-graph FULL.
+            #
+            # When the mask is actually needed, clone `is_multimodal` onto
+            # our own storage so downstream kernels do not alias the
+            # scheduler buffer.
+            has_mm_embeds = (
+                multimodal_embeddings is not None
+                and (
+                    isinstance(multimodal_embeddings, torch.Tensor)
+                    or len(multimodal_embeddings) > 0
+                )
+            )
+            if is_multimodal is not None and has_mm_embeds:
+                is_multimodal = is_multimodal.to(
+                    input_ids.device, non_blocking=False
+                ).clone()
+                ple_input_ids = torch.where(
+                    is_multimodal, torch.zeros_like(input_ids), input_ids
+                )
+            else:
+                ple_input_ids = input_ids
```

Three surgical changes:

1. Gate the masking on `has_mm_embeds` - an empty-list / `None`
   `multimodal_embeddings` is now treated the same way as Gemma3n treats
   the text-only case (no mask, feed `input_ids` directly).
2. Force the H2D (or same-device noop) to `non_blocking=False` so the
   copy is ordered wrt the caller's stream.
3. `.clone()` the resulting tensor so the PLE and `torch.where` kernels
   operate on a fresh allocation owned by this call frame, not the
   scheduler's shared double-buffer.

Overhead: zero for text-only (the kernel is skipped). For real multimodal
batches, the added clone is a single H2D of a `bool[n]` tensor - at
`max_num_batched_tokens=8192` that is 8 KiB, sub-microsecond on Blackwell.

### Verification

1. Reproduce without the patch:
   - `vllm-fusencache:latest` (vLLM `0.1.dev100+gc0c98b8b9.d20260417`)
   - Gemma4-26B NVFP4, FusenCache k4v4b64, `cudagraph_mode=FULL`,
     `--async-scheduling`, prefix caching on, chunked prefill on.
   - 5 concurrent text-only requests at 32-token prompts.
   - Expected (current): illegal memory access inside `embed_input_ids`
     within seconds.

2. Apply `gemma4_embed_input_ids_fix.patch` to
   `vllm/model_executor/models/gemma4_mm.py`, restart the container.

3. Repeat the load test. Expected (fixed): steady throughput with no
   crash over >=10k steps. Verify logits are bit-identical to a C=1 run.

4. Re-run the vLLM test suite:
   - `pytest tests/models/multimodal/test_gemma4.py`
   - `pytest tests/v1/worker/test_gpu_model_runner.py`

### Backport note

`vllm-built:latest` (0.19.1rc1) does **not** contain
`gemma4_mm.py` - Gemma4 support was added on main after 0.19.1, and it
shipped together with this buggy PLE-mask code path. 0.18 / 0.17 only
have `gemma3n_mm.py`, whose `embed_input_ids` never calls
`torch.where(is_multimodal, ...)`. Therefore **no backport is required**;
the fix only needs to land on main (and any release branch that has already
picked up Gemma4).

### Files changed

```
vllm/model_executor/models/gemma4_mm.py
```

Single function touched: `Gemma4ForConditionalGeneration.embed_input_ids`.
