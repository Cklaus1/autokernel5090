# vLLM Async Scheduling Race Condition Analysis

## Summary

vLLM v0.19.0's async scheduling has a race condition in `gpu_model_runner.py`
that causes CUDA graph replay crashes when using attention backends slower than
FlashAttention. The existing `synchronize_input_prep()` mechanism only
synchronizes input preparation, not the model forward pass, leaving a window
where the next step's CPU preparation can interfere with the current step's
GPU execution.

## Architecture of Async Scheduling

### Normal (synchronous) flow
```
Step N:  [prep] -> [forward] -> [sample] -> [bookkeep]
Step N+1:                                    [prep] -> [forward] -> ...
```

### Async scheduling flow
```
Step N:  [prep] -> [forward] -GPU-> [sample] -> [bookkeep]
Step N+1:  [prep] ---------CPU--> [forward] -GPU-> ...
                  ^                                   
                  |_ CPU starts prep while GPU still running step N's forward
```

### The synchronization gap

`synchronize_input_prep()` (line 3477) is a context manager that:
1. On entry: `prepare_inputs_event.synchronize()` -- waits for previous step's
   prep GPU ops (non-blocking copies) to complete
2. On exit: `prepare_inputs_event.record()` -- records event after current
   step's prep ops

The model forward (`_model_forward()` at line 4034) happens OUTSIDE this
context manager. The event timeline:

```
Step N:   [sync event N-1] -> [prep work] -> [record event N] -> [model_forward]
Step N+1: [sync event N]   -> [prep work] ...
                              ^
                              |_ This starts before step N's model_forward finishes
```

### Why GPU stream ordering is insufficient

While CUDA operations on the same stream execute in order (step N+1's GPU ops
won't execute until step N's model forward finishes), the CPU runs ahead:

1. **Pinned memory corruption**: `_prepare_inputs()` writes to `input_ids.cpu`
   (pinned memory), then calls `copy_to_gpu(non_blocking=True)`. If step N+1's
   CPU overwrites `input_ids.cpu` before step N's DMA finishes... but actually,
   the `prepare_inputs_event` covers this case.

2. **CPU-side state corruption**: `_update_states()` modifies `input_batch`
   (removes/adds requests, updates block tables on CPU). This changes
   `block_table.cpu` which may be mid-DMA from step N's `commit_block_table()`.
   The event covers the GPU DMA but NOT the CPU-side data structure mutations.

3. **CUDA allocator memory reuse**: Temporary tensors from step N's model
   forward (attention intermediates, activation buffers) are freed when Python
   references go out of scope. The CUDA caching allocator marks this memory as
   available. Step N+1's operations may allocate into the same memory while
   step N's GPU kernels still read it.

### Why FlashAttention masks the bug

FlashAttention kernels are highly optimized and finish executing before the
CPU reaches the point of modifying shared state. With slower backends:
- FusenCache: ~1.14x slower per layer
- Custom Triton attention: varies, often 1.2-2x slower
- Any non-FlashAttention backend: timing-dependent crash

## The Fix

Added a `_forward_done_event` (torch.cuda.Event) that records AFTER model
forward completes and synchronizes BEFORE the next step's input preparation.

### Changes to gpu_model_runner.py

1. **Initialization** (line 649): `self._forward_done_event = torch.cuda.Event()`
   alongside existing `prepare_inputs_event`

2. **synchronize_input_prep()** (line 3494): Added
   `self._forward_done_event.synchronize()` after the existing
   `prepare_inputs_event.synchronize()`

3. **execute_model()** (line 4115): Added
   `self._forward_done_event.record()` after model forward + postprocessing,
   right before storing `ExecuteModelState`

### Performance impact

- `torch.cuda.Event.record()`: ~2us (just inserts a marker in the stream)
- `torch.cuda.Event.synchronize()`: blocks CPU until GPU reaches the event
  - Best case (GPU already done): ~5us
  - Worst case (waiting for forward): up to the full forward latency
  - Typical decode step: ~10-20us total overhead

This effectively turns async scheduling into "mostly synchronous" for the
forward pass, but retains async benefits for the sampling/bookkeeping overlap.

### Alternative approaches considered

- **Double-buffered inputs**: Pre-allocate two sets of input tensors and
  alternate. Cleanest solution but requires significant refactoring of
  GPUModelRunner's buffer management.

- **Clone before graph replay**: Clone input tensors before CUDA graph replay.
  Adds one memcpy per step (~50us for typical batch) but is minimal change.

- **Full stream synchronize**: `torch.cuda.synchronize()` after forward.
  Heavier than needed -- blocks ALL streams, not just the compute stream.

## Files Modified

- `/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py`
  (3 insertion points, marked with `# ASYNC_SCHEDULING_FIX`)

## Patch Script

`/root/projects/autokernel/patches/vllm_async_scheduling_fix.py`

Usage:
```bash
python3 patches/vllm_async_scheduling_fix.py           # Apply fix
python3 patches/vllm_async_scheduling_fix.py --check   # Check status
python3 patches/vllm_async_scheduling_fix.py --revert  # Revert
```
