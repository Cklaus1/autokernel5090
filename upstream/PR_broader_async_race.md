# fix(v1/gemma4): two more async-scheduling races — Gemma4Router.root_size and AsyncGPUModelRunnerOutput.get_output

## Summary

- `Gemma4Router.forward()` crashes with `cudaErrorIllegalAddress` at C≥64
  in `cudagraph_mode=FULL` (FULL_DECODE_ONLY) because `root_size.to(x.dtype)`
  emits a device-to-device kernel from inside a CUDA-graph replay, where the
  buffer origin races with the next iteration's async input prep.
- `AsyncGPUModelRunnerOutput.get_output()` crashes in piecewise mode because
  `self.sampled_token_ids_cpu.shape[-1]` is read on line 261 **before**
  `async_copy_ready_event.synchronize()` on line 262. In piecewise graphs the
  copy-stream event may not yet be submitted to the CUDA runtime when `.shape`
  is dereferenced, producing a use-before-ready access on the CPU buffer.

Both sites belong to the same failure *class* identified in Discovery #54 and
fixed in the companion PR for `synchronize_input_prep()` / `gemma4_mm.py`.

## Symptom

**Site 1 — FULL_DECODE_ONLY (C≥64)**
```
File ".../vllm/model_executor/models/gemma4.py", line 199, in forward
    x = x * self.root_size.to(x.dtype)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**Site 2 — piecewise CUDA graphs**
```
File ".../vllm/v1/worker/gpu_model_runner.py", line 262, in get_output
    self.async_copy_ready_event.synchronize()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```
(The error materialises at `synchronize()` but is the deferred result of the
CPU buffer read on the preceding line.)

## Root cause

### Site 1 — `Gemma4Router.forward`

`root_size` is a `register_buffer(..., persistent=False)` holding a scalar
`hidden_size**-0.5` (dtype `float32`, device determined at `__init__` time).
At inference time `x` is `bfloat16`. The expression

```python
x = x * self.root_size.to(x.dtype)   # gemma4.py:199
```

emits a `torch.Tensor.to(dtype)` op that is *not* captured into the CUDA
graph because the buffer lives on a different dtype lane than `x`. Under
async scheduling, iteration N+2's input-prep thread can race with iteration
N's still-in-flight CUDA graph replay that reads the original `root_size`
storage through the `.to()` cast kernel.

Fix: pre-cast `root_size` to the model dtype in `__init__` — or store it as a
plain Python `float` and multiply with a scalar — so `forward()` issues no
device-side type-cast kernel at all.

### Site 2 — `AsyncGPUModelRunnerOutput.get_output`

```python
# gpu_model_runner.py:261-262 (AsyncGPUModelRunnerOutput.get_output)
max_gen_len = self.sampled_token_ids_cpu.shape[-1]   # line 261 -- BEFORE sync
self.async_copy_ready_event.synchronize()             # line 262 -- sync here
```

`sampled_token_ids_cpu` is assigned via `non_blocking=True` inside the
`async_output_copy_stream` context in `__init__`. With piecewise CUDA graphs,
the copy-stream work (and the subsequent `async_copy_ready_event.record()`)
may not have been submitted to the CUDA runtime yet when `get_output()` is
called by the scheduler thread. Reading `.shape` on a tensor whose backing
non-blocking copy is still pending is safe only if the allocation is already
live — but in piecewise mode the allocation itself can be inside the graph,
making it a dangling reference. Swapping the two lines ensures the event
(and therefore the allocation) is committed before any attribute is accessed.

## The fix (minimal diff)

### `vllm/model_executor/models/gemma4.py`

```diff
@@ Gemma4Router.__init__
+        # Pre-cast to avoid a runtime .to(dtype) inside forward(), which
+        # can race against async scheduling under CUDA-graph FULL mode.
+        # The cast is deferred to load_weights() time via a post_init hook;
+        # here we store as float32 and convert lazily in forward() only when
+        # the buffer already matches dtype (no-op path), but see forward().
```

Simplest safe fix — change `forward()` to multiply by a Python scalar:

```diff
--- a/vllm/model_executor/models/gemma4.py
+++ b/vllm/model_executor/models/gemma4.py
@@ -178,7 +178,9 @@ class Gemma4Router(nn.Module):
-        self.register_buffer(
-            "root_size",
-            torch.tensor(self.hidden_size**-0.5),
-            persistent=False,
-        )
+        # Store as a plain Python float so forward() can multiply with a
+        # scalar literal — no device-side cast kernel, no async-race risk.
+        self._root_size_scalar: float = self.hidden_size**-0.5

@@ -196,7 +198,7 @@ class Gemma4Router(nn.Module):
     def forward(self, x: torch.Tensor) -> torch.Tensor:
         """Returns raw router logits [T, E]."""
         x = self.norm(x)
-        x = x * self.root_size.to(x.dtype)
+        x = x * self._root_size_scalar
         x = x * self.scale.to(x.dtype)
         router_logits, _ = self.proj(x)
         return router_logits
```

**Note:** `self.scale.to(x.dtype)` on the same line is a `nn.Parameter`, which
lives on the model device and is always in the same dtype as `x` after
`.to(device)` at model load; at runtime this `.to()` is a no-op identity and
does not emit a kernel. It is left as-is. If future refactors change the
parameter dtype, the same scalar-float treatment should be applied.

### `vllm/v1/worker/gpu_model_runner.py`

```diff
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -258,8 +258,10 @@ class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
     def get_output(self) -> ModelRunnerOutput:
         """Copy the device tensors to the host and return a ModelRunnerOutput.

         This function blocks until the copy is finished.
         """
-        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
         self.async_copy_ready_event.synchronize()
+        # Read .shape only after the async copy event has been committed;
+        # in piecewise graph mode the CPU buffer may not be valid until then.
+        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
```

## Diff size

| Site | Lines removed | Lines added | Net |
|------|--------------|-------------|-----|
| `gemma4.py` (Site 1) | 5 | 4 | −1 |
| `gpu_model_runner.py` (Site 2) | 2 | 3 | +1 |
| **Total** | **7** | **7** | **0** |

**7 lines changed across 2 files** — well within the single-PR threshold.

## Test plan

- [ ] `pytest tests/models/multimodal/test_gemma4.py` — router forward correctness
- [ ] `pytest tests/v1/worker/test_gpu_model_runner.py` — async output copy
- [ ] Serve Gemma4 26B with `cudagraph_mode=FULL`, `--async-scheduling`,
      C=64, C=128 text-only decode; verify no `cudaErrorIllegalAddress` over
      10k+ steps
- [ ] Serve Gemma4 26B with piecewise CUDA graphs, C=64; verify steady
      throughput, no crash in `get_output()`
- [ ] Benchmark: compare tok/s at B=64 before/after — expect < 0.5% regression
      (scalar multiply is faster than `.to(dtype)` round-trip)

## Related

- **Discovery #54 / PR `fix(v1): synchronize model forward completion before
  next step's input prep`** — the original `synchronize_input_prep()` fence
  in `gpu_model_runner.py` that introduced the event-based guard. The current
  two sites slip through because (a) the `root_size` race happens *inside*
  the captured graph, not across the prep boundary, and (b) `get_output()` is
  called from outside the prep context manager.
- **`fix(models/gemma4_mm): avoid aliasing scheduler's is_mm_embed
  double-buffer in embed_input_ids`** — the §4c₁ companion fix for the
  `is_mm_embed_buffers` GPU-storage alias in `Gemma4ForConditionalGeneration`.

## Files changed

```
vllm/model_executor/models/gemma4.py        (+4 / -5)
vllm/v1/worker/gpu_model_runner.py          (+3 / -2)
```

Two functions touched:
1. `Gemma4Router.__init__` + `Gemma4Router.forward`
2. `AsyncGPUModelRunnerOutput.get_output`
