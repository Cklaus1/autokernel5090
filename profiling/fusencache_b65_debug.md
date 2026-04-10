# FusenCache B=65 CUDA Assertion Debug Analysis

## Symptom

When FusenCache k4v4b64 is used with inductor + CUDA graphs, the vLLM server
crashes with a CUDA kernel assertion when 65 concurrent requests are active.
Single requests and small batches work. B=65 specifically triggers the crash.

```
CUDA kernel errors might be asynchronously reported at some other API call
Compile with TORCH_USE_CUDA_DSA to enable device-side assertions
```

## Root Cause

**Dynamic tensor slicing in `forward()` breaks CUDA graph replay.**

The decode-only path in `FusenKVImpl.forward()` was doing:

```python
B = attn_metadata.num_reqs     # Python int: 65 (actual)
q = query[:B]                   # Creates tensor [65, Hq, D]
block_table = attn_metadata.block_table[:B]
seq_lens = attn_metadata.seq_lens[:B]
```

Under CUDA graphs, vLLM captures the forward at padded batch sizes (e.g., 72
for actual B=65, since capture sizes are [1, 2, 4, 8, 16, 24, ..., 64, 72, ...]).
During capture, `query[:72]` produces a `[72, Hq, D]` tensor and the Triton
kernel is launched with that shape. During replay with actual B=65:

1. The Python code does NOT re-execute (graph replays CUDA ops only)
2. The kernel runs with the captured [72, Hq, D] tensor shape
3. BUT the persistent buffer was allocated for B=72 at capture time
4. When inductor compiles the graph, the `[:B]` slice is a shape guard --
   if B changes between capture and replay, inductor may re-trace or
   the graph replay may use stale pointers

This mismatch causes undefined behavior in the CUDA runtime, manifesting as
an asynchronous assertion failure.

## Fix (in backend.py)

Eliminated dynamic `[:B]` slicing in the decode-only path. Instead, pass the
full padded tensors to the decode kernel:

```python
# Before (broken):
q = query[:B]
attn_out = self.decode_fn(q, ..., block_table[:B], seq_lens[:B], ...)

# After (fixed):
padded_B = query.shape[0]  # Static shape from captured graph
q = query                   # Full tensor, no slicing
attn_out = self.decode_fn(q, ..., block_table, seq_lens, ...)
```

This is safe because:
- `seq_lens[B:]` = 0 (vLLM zeroes padded entries in `input_batch.py:104`)
- The decode kernel skips entries with seq_len=0 (`split_start >= split_end`)
- The stage2 kernel outputs zeros for seq_len=0 entries (`safe_sum` guard)
- `slot_mapping[B:]` = -1 (PAD_SLOT_ID), and the store kernel checks `slot < 0`

The same fix was applied to the store path for decode-only batches.

## What Was NOT the Problem

The standalone Triton kernel was tested extensively and found to be correct:

1. **Grid calculation**: `cdiv(Hq, VALID_BLOCK_H)` correctly covers all heads
   with proper masking. No OOB writes into `mid_out`.
2. **Split-KV boundaries**: All seq_len values (1 through 4096) work correctly
   at B=65 with NUM_KV_SPLITS=32.
3. **Persistent buffers**: Buffer keyed by `(B, Hq, D, device, dtype)` works
   correctly for all batch sizes, including reuse across different B values.
4. **Block table bounds**: Sequential block allocation stays within bounds.
5. **CUDA graph capture/replay**: Standalone capture at B=72 with replay using
   seq_lens=0 for padded entries works correctly.

## Debug Mode

Added `FUSEN_DEBUG=1` environment variable that enables runtime bounds checking:
- Validates block_table entries < num_blocks
- Validates slot indices < scales tensor capacity
- Zero overhead when disabled (checked at import time)

## Files Changed

- `/root/projects/autokernel/fusen_kv/backend.py` -- Fixed dynamic slicing,
  added FUSEN_DEBUG bounds checking
- `/root/projects/autokernel/test_fusencache_b65_crash.py` -- Standalone
  crash reproduction test (8 test categories, 29 test cases)
