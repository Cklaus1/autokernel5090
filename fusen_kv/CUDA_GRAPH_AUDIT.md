# FusenCache CUDA Graph Compatibility Audit

Audit date: 2026-04-09
Files audited: backend.py (933 lines), generate.py (443 lines), kernel.py (662 lines)

## Previously Fixed Bugs (1-5)

1. Dynamic `[:B]` slicing breaks CUDA graph replay
2. Shared buffer underallocation -- 32x OOB write
3. `output[:padded_B] = attn_out` shape mismatch
4. `output[dec_token_starts] = attn_out` shape mismatch
5. Async CUDA memory recycling race (sync added)

## New Issues Found

### ISSUE #6: `_store_triton` slices `key[:N]` and `value[:N]` -- redundant but creates new view objects

**File:** generate.py:363-364
**What:** `k = key[:N]` where `N = slot_mapping.shape[0]`. During decode-only CUDA graph
path, `key` already has shape `[padded_B, ...]` and `N == padded_B`, so `[:N]` is
semantically a no-op but still creates a new Python view object.
**Why it's OK (not a bug):** PyTorch `[:N]` when `N == tensor.shape[0]` returns a view
sharing the same storage at the same address. The CUDA graph records the underlying
data pointer, not the Python object. Since the view shares storage, the kernel sees
the same GPU address.
**Severity:** SAFE (no fix needed, but could be cleaned up for clarity)

### ISSUE #7: `_store_triton` conditional `if k.ndim == 2: k = k.reshape(N, Hk, D)` -- Python branch varies

**File:** generate.py:365-367
**What:** Python conditional on tensor dimensionality. If key is 3D during capture but
2D during replay (or vice versa), the reshape call would/wouldn't execute.
**Why it's safe:** In the CUDA graph decode-only path, vLLM always passes key as 3D
`[padded_B, num_kv_heads, head_size]`. The reshape is never taken. And even if it were,
`.reshape()` on a contiguous tensor is a view (same address).
**Severity:** SAFE (latent risk only if vLLM changes key format between steps)

### ISSUE #8: `_store_triton` calls `.contiguous()` which may allocate

**File:** generate.py:370-371
**What:** `k = k.contiguous()` and `v = v.contiguous()`. If key/value are NOT contiguous
during CUDA graph capture, this allocates a new tensor. During replay, the captured
tensor address is used. If key/value ARE contiguous (typical), `.contiguous()` is a
no-op returning the same tensor.
**Why it's a latent risk:** If the input key/value contiguity differs between capture
and replay, the CUDA graph would use a stale address. In practice, vLLM always provides
contiguous key/value tensors in decode-only mode.
**Fix recommendation:** Remove `.contiguous()` in the CUDA-graph-safe path, or assert
contiguity instead of silently copying.
**Severity:** LATENT RISK (no crash observed, but fragile)

### ISSUE #9: `_store_triton` has `if not hasattr(layer, '_fc_scales')` -- conditional allocation

**File:** generate.py:378-380
**What:** Conditionally allocates `layer._fc_scales = torch.zeros(...)` on first call.
**Why it's safe:** `backend.py:734` calls `_ensure_scales(layer, kv_cache, query.device)`
before the store, which sets `layer._fc_scales`. So this branch is never taken during
normal forward passes. The `hasattr` check is dead code in practice.
**Fix recommendation:** Remove the redundant allocation in `_store_triton` since the
backend guarantees `_fc_scales` exists before store is called.
**Severity:** SAFE (dead code, but confusing)

### ISSUE #10: `_ensure_scales` allocates on first call -- must happen BEFORE CUDA graph capture

**File:** backend.py:457-475
**What:** `_ensure_scales` allocates `layer._fusen_scales` on the first call. If this
first call happens during CUDA graph capture, `torch.zeros(...)` inside capture would
record a kernel that initializes memory. During replay, the same initialization kernel
runs, zeroing out scale data that may have been populated by prior store operations.
**Why it's safe in practice:** vLLM runs warmup/dummy forward passes before CUDA graph
capture. These warmup passes call `_ensure_scales`, so the allocation happens before
capture. The `_scales_initialized` flag prevents re-allocation.
**But:** If `_scales_initialized` is per-`FusenKVImpl` instance but the same layer is
used across instances, there's no issue because `_ensure_scales` checks `hasattr(layer, '_fusen_scales')`.
**Severity:** SAFE (assumes vLLM warmup runs before capture, which it does)

### ISSUE #11: `query.contiguous()` in decode path may allocate during CUDA graph capture

**File:** backend.py:783
**What:** `q = q.contiguous()` -- if query is not contiguous, allocates a new tensor.
**Analysis:** Same as Issue #8. In decode-only mode, vLLM provides contiguous query.
`.contiguous()` on an already-contiguous tensor returns `self` (no allocation).
**Severity:** LATENT RISK (safe in practice)

### ISSUE #12: Adaptive `NUM_KV_SPLITS` varies per capture size -- different Triton kernels compiled

**File:** generate.py:175-178
**What:** `NUM_KV_SPLITS = _optimal_splits(_seq_hint, B)` where B varies per CUDA graph
capture size. Different B values produce different NUM_KV_SPLITS (a `tl.constexpr`),
causing Triton to compile different kernel binaries.
**Why it's correct:** Each CUDA graph capture records the specific kernel binary for its
B/NUM_KV_SPLITS combination. During replay, the correct binary is used. Triton's
constexpr specialization handles this transparently. The grid `(B, num_head_groups, NUM_KV_SPLITS)` is also baked into each graph.
**Potential issue:** The shared `mid_out` buffer has `_num_kv_splits` splits (the
maximum, computed at B=1). When `NUM_KV_SPLITS < _num_kv_splits`, only a subset of
the buffer's split dimension is used. The strides are correct because they come from
the full buffer shape.
**Severity:** SAFE (correct by design)

### ISSUE #13: `output[:padded_B] = attn_out[:padded_B].to(output.dtype)` -- `.to()` may allocate

**File:** backend.py:821
**What:** `.to(output.dtype)` converts dtype. If `attn_out.dtype != output.dtype`, this
creates a temporary tensor. During CUDA graph capture, this allocation is recorded.
During replay, the same temporary address is reused (CUDA graph replays allocations at
the same addresses from a private memory pool).
**Why it's safe:** CUDA graphs with `torch.cuda.CUDAGraph` manage internal allocations.
Temporary tensors created during capture are replayed at the same addresses. The `.to()`
conversion is deterministic.
**But:** The `attn_out` is the shared buffer (`_shared_output`), which was allocated with
`_out_dtype` matching `vllm_config.model_config.dtype`. The `output` buffer is provided
by vLLM with the model's dtype. If they match, `.to()` is a no-op. If they don't match,
the conversion is captured and replayed correctly.
**Severity:** SAFE

### ISSUE #14: `_cpp_decode` calls `_optimal_splits()` at Python level -- baked into graph

**File:** backend.py:497-501
**What:** `from kv_cache_gen.generate import _optimal_splits; num_kv_splits = min(..., _optimal_splits(self._max_seq, B))`.
This Python computation happens during capture and produces a Python int passed to
`torch.ops.fusencache.decode_attention`. During replay, the same int value is used
(baked into the captured op).
**Why it's safe:** B = padded_B is constant per graph capture. The computation is
deterministic and produces the same result every time for the same B. The Python code
doesn't run during replay.
**Severity:** SAFE

### ISSUE #15: `_cpp_decode` buffer selection has `if B <= ... else ...` branch

**File:** backend.py:507-518
**What:** Selects shared vs temporary buffers based on B. During CUDA graph capture,
B <= max size (by definition), so shared buffers are always used. The else branch
(temporary allocation) is only hit in eager mode.
**Severity:** SAFE

### ISSUE #16: The per-layer synchronize is skipped during CUDA graph replay

**File:** backend.py:930-931
**What:** `if not torch.cuda.is_current_stream_capturing(): torch.cuda.current_stream().synchronize()`.
During CUDA graph capture, this is skipped (correct -- sync would break capture).
During CUDA graph replay, the Python code doesn't execute at all. The sync was added
to fix the async race in eager mode.
**Analysis:** Within a CUDA graph, all operations are serialized on the captured stream.
The shared `_shared_output` buffer is written by decode, then copied to `output`, then
the next layer's forward begins -- all in order within the graph. No race possible.
Between graph replays on the same stream, operations are also serialized.
**The concern:** If vLLM uses different CUDA streams for different steps (pipelining),
two graph replays could overlap, both writing to `_shared_output`. But vLLM's CUDA graph
infrastructure replays on a single stream, so this doesn't happen.
**Severity:** SAFE for single-stream replay. WOULD CRASH with multi-stream replay
(not used by vLLM currently).

### ISSUE #17: The `_pytorch_decode` reference path has `.item()` calls

**File:** backend.py:566, 574
**What:** `seq_lens[b].item()` and `block_table[b, blk_idx].item()` in the PyTorch
reference decode implementation.
**Why it's safe:** `_pytorch_decode` is never called in the hot path. It's only used
for debugging. The main forward() always uses `_cpp_decode` or `self.decode_fn`.
**Severity:** SAFE (debug-only code)

### ISSUE #18: The `_prefill_with_sliding_window` has many dynamic allocations

**File:** backend.py:629-703
**What:** Creates causal masks, calls `torch.arange`, `torch.matmul`, etc.
**Why it's safe:** Prefill is NEVER executed during CUDA graph capture/replay
(`is_decode_only` must be False for prefill, and CUDA graphs only capture decode-only).
**Severity:** SAFE

### ISSUE #19: The mixed prefill+decode path has `.item()` calls and dynamic allocations

**File:** backend.py:841-914
**What:** `qsl[i].item()`, `torch.tensor(decode_indices, ...)`, fancy indexing, etc.
**Why it's safe:** Mixed path runs only in eager mode (CUDA graphs are decode-only).
**Severity:** SAFE

### ISSUE #20: `store_async` creates a new CUDA stream on first call

**File:** generate.py:424-425
**What:** `_store_stream = torch.cuda.Stream()` -- but `store_async` is never called
in the backend. The backend calls `self.store_fn(...)` which calls `store()` (sync),
not `store_async()`.
**Severity:** SAFE (unused code path)

## Complete List of Tensors That Must Be Pre-Allocated for CUDA Graphs

| Tensor | Allocated in | Shape | Purpose |
|--------|-------------|-------|---------|
| `self._shared_mid_out` | backend.py:383-386 | `[_max_B, Hq, _num_kv_splits, D+1]` f32 | Stage1 decode intermediate |
| `self._shared_output` | backend.py:387-390 | `[_max_B, Hq, D]` model dtype | Stage2 decode output |
| `layer._fusen_scales` | backend.py:469-472 | `[max_slots, Hk, num_sb, 2]` f16 | KV quantization scales |
| `query` (input) | vLLM pre-allocates | `[padded_B, Hq, D]` | Query tensor |
| `key` (input) | vLLM pre-allocates | `[padded_B, Hk, D]` | Key tensor |
| `value` (input) | vLLM pre-allocates | `[padded_B, Hk, D]` | Value tensor |
| `output` (input) | vLLM pre-allocates | `[padded_B, Hq, D]` | Output buffer |
| `kv_cache` | vLLM pre-allocates | `[num_blocks, block_size, Hk, slot_bytes]` u8 | Paged KV cache |
| `block_table` | vLLM pre-allocates | `[padded_B, max_blocks]` | Page table |
| `seq_lens` | vLLM pre-allocates | `[padded_B]` | Sequence lengths |
| `slot_mapping` | vLLM pre-allocates | `[padded_B]` | Slot mapping for store |

## Complete List of Code Paths During Decode Forward

1. `forward()` entry (line 705)
2. `_ensure_scales()` -- returns immediately after first call (line 734)
3. Store path: `self.store_fn(key, value, kv_cache, slot_mapping, layer, num_kv_heads)` (line 754)
   - `_store_triton()`: `N = slot_mapping.shape[0]`, `key[:N]`, `k.contiguous()`,
     grid=(N,Hk), launch `_universal_store_kernel`
4. Decode path: `padded_B = query.shape[0]`, `q = query`, `q.contiguous()` (lines 779-783)
5. Decode kernel call: either `_cpp_decode` or `self.decode_fn` (line 814)
   - `self.decode_fn` (Triton): selects shared buffers, computes grid, launches
     `_universal_decode_stage1` then `_universal_decode_stage2`, returns shared output
   - `_cpp_decode` (C++): selects shared buffers, calls `torch.ops.fusencache.decode_attention`
6. Output copy: `output[:padded_B] = attn_out[:padded_B].to(output.dtype)` (line 821)
7. Sync guard: `torch.cuda.current_stream().synchronize()` -- skipped during capture/replay (line 930-931)
8. Return output (line 933)

## Recommendation: Rewrite vs Patch

**Recommendation: NO REWRITE NEEDED.**

The code is fundamentally sound for CUDA graphs. The 5 previously-fixed bugs were the
critical ones. The remaining findings (Issues #6-#20) are all either:
- **SAFE**: Correctly handled by design
- **LATENT RISK**: Would only break if vLLM changes its contract (e.g., passes
  non-contiguous tensors, changes key dimensionality between steps)

The CUDA graph path (decode-only, `is_decode_only=True`) has a clean architecture:
1. All buffers are pre-allocated (shared mid_out and output)
2. All kernel launches use fixed grid sizes (determined by padded_B)
3. All constexpr parameters are deterministic per padded_B
4. No `.item()` calls in the hot path
5. No dynamic allocations in the hot path
6. The sync guard correctly skips during capture

**If C=16+ still crashes, the bug is likely NOT in these 3 files.** Possible locations:
- vLLM's CUDA graph capture/replay infrastructure
- The C++ decode kernel (`fusencache_decode.so`) -- its internal implementation
- Memory pressure causing OOM (not an OOB/logic bug)
- The interaction between FusenKV's shared buffers and vLLM's memory allocator
  (the sync at line 931 is a band-aid; it doesn't fire during CUDA graph replay)

**Minor cleanup recommendations:**
1. Remove the redundant `hasattr(layer, '_fc_scales')` check in `_store_triton`
   (generate.py:378) since `_ensure_scales` guarantees it exists
2. Add an assertion in the decode path: `assert q.is_contiguous()` instead of
   calling `.contiguous()` (backend.py:783)
3. Remove `.contiguous()` calls in `_store_triton` (generate.py:370-371) for the
   CUDA-graph-safe path, replacing with assertions
