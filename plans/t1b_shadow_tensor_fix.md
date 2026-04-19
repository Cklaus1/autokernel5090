# T1-B Shadow-Tensor Fix (W5_T1B_shadow_tensor_fix)

**Date:** 2026-04-19
**Tag:** `W5_T1B_shadow_tensor_fix`
**File:** `fusen_kv/backend.py`
**Context:** Recover from T1-B KILL (2026-04-18) — FusenCache piecewise CUDA
graph patch crashed at C>4 with `cudaErrorIllegalAddress` at
`_to_list → pinned.copy_(sampled_token_ids)`. Fallback FULL mode ran at only
43 tok/s @ C=32 vs pre_t1b peak 4,489 tok/s.

## Root cause (from diff of `backend.py.pre_t1b` vs `backend.py.t1b_broken`)

The T1-B patch replaced `.clone()` of async metadata with pre-allocated
shadow buffers + `.copy_()`. Three latent bugs combined to produce the
illegal-address crash at C>4:

1. **Under-sized shadow buffers.** Token-indexed tensors (`_shadow_slot_mapping`,
   `_ud_token_positions`, `_ud_pseudo_seq_lens`, `_ud_token_block_table`)
   were sized to `max_num_batched_tokens` with a fallback of `_max_B`
   (= `max_num_seqs`). Under piecewise CUDA graph mixed-batch scheduling,
   `padded_T = query.shape[0]` can exceed either bound, causing
   `self._ud_token_positions[:padded_T]` and siblings to produce silent
   out-of-bounds views. The first downstream kernel touching those views
   (or the sampler's `pinned.copy_(sampled_token_ids)`) trips
   `cudaErrorIllegalAddress`.

2. **No size assert — silent CUDA fault.** The T1-B code clamped sizes with
   `min(bt.shape[0], self._shadow_block_table.shape[0])` rather than
   asserting. When the shadow cap was too small the copy silently truncated
   the metadata, producing wrong attention output and, at the sampler stage,
   an illegal address.

3. **View aliasing across graph-capture iterations (P8).** Shadow buffers
   were returned as views (`self._shadow_block_table[:rows, :cols]`) that
   remained live into the next iteration's write, violating the pattern in
   `plans/KILL_PATTERNS.md §P8`.

## Fixes applied (3 changes, matches audit recommendation)

### Fix A: Bounded-sentinel shadow sizing

```python
# OLD (t1b_broken)
_max_seqs = _max_B                    # typically max_num_seqs
_max_tokens = _max_B
if vllm_cfg.scheduler_config.max_num_batched_tokens:
    _max_tokens = max_num_batched_tokens
# -> tight bound, no slack

# NEW (W5 fix, pulled from vllm_cfg)
_SHADOW_SLACK = 4096
_shadow_cap = max(max_num_seqs, max_num_batched_tokens,
                  max_cudagraph_capture_size, _max_B) + _SHADOW_SLACK
_max_tokens = _shadow_cap
_max_seqs = max(_max_seqs, max_num_seqs) + _SHADOW_SLACK
```

Reads from `scheduler_config.max_num_batched_tokens`,
`scheduler_config.max_num_seqs`, and
`compilation_config.max_cudagraph_capture_size`. `+4096` absorbs any
scheduler-internal padding not exposed in config.

### Fix B: Python-level shape assert before every `.copy_()`

```python
def _shadow_copy(src, dst_buf, name):
    if src is None:
        return None
    if src.ndim == 1:
        if src.shape[0] > dst_buf.shape[0]:
            raise RuntimeError(
                f"FusenKV shadow buffer {name} too small: "
                f"src.shape={tuple(src.shape)} dst_cap={dst_buf.shape[0]} "
                f"(rebuild with larger _SHADOW_SLACK)")
        dst = dst_buf[:src.shape[0]]
    elif src.ndim == 2:
        # ... analogous ...
    src_view = src if src.dtype == dst.dtype else src.to(dst.dtype)
    assert dst.shape == src_view.shape, (
        f"FusenKV shadow copy {name}: "
        f"dst.shape={tuple(dst.shape)} != src.shape={tuple(src_view.shape)}")
    dst.copy_(src_view)
    return dst.clone()
```

The size-check `raise` fires at Python stack frame (debuggable), not inside
the CUDA driver as an illegal-address crash.

### Fix C: Clone returned views (P8 fix)

The shadow buffer is a fixed-address staging pool; the `.clone()` at the end
returns a fresh allocation independent from the persistent shadow. This
means:
- Next iteration's `.copy_()` into the shadow can't corrupt the current
  iteration's returned tensor.
- The returned tensor can be retained (`self._prev_block_table = ...`) for
  one step of GC safety without pinning the shadow.
- Mirrors the pre-T1B semantics (`.clone()` from `attn_metadata.*`) at near-
  identical cost, while retaining a stable staging buffer.

Before/after for the metadata isolation block:

```python
# BEFORE (pre_t1b — worked at 4,489 tok/s but allocated from default pool)
_block_table     = attn_metadata.block_table.clone()     if ... else None
_seq_lens        = attn_metadata.seq_lens.clone()        if ... else None
_slot_mapping    = attn_metadata.slot_mapping.clone()    if ... else None
_query_start_loc = attn_metadata.query_start_loc.clone() if ... else None

# BEFORE (t1b_broken — crashed at C>4, view aliasing + OOB)
# ... 40 lines of unchecked min(shape, cap) .copy_() ending in view return ...

# AFTER (W5 fix)
_block_table     = _shadow_copy(attn_metadata.block_table,
                                self._shadow_block_table, "block_table")
_seq_lens        = _shadow_copy(attn_metadata.seq_lens,
                                self._shadow_seq_lens, "seq_lens")
_slot_mapping    = _shadow_copy(attn_metadata.slot_mapping,
                                self._shadow_slot_mapping, "slot_mapping")
_query_start_loc = _shadow_copy(attn_metadata.query_start_loc,
                                self._shadow_query_start_loc,
                                "query_start_loc")
```

## Mapping fix → audit bullet

| Audit bullet | Fix | Rationale |
|---|---|---|
| Size buffers to `max(max_seqs, max_num_batched_tokens)+4096` | Fix A | Uses bounded sentinel from `vllm_cfg.scheduler_config` + `compilation_config.max_cudagraph_capture_size`; `+4096` slack absorbs padding drift |
| Assert shape before `pinned.copy_()` | Fix B | `_shadow_copy()` raises Python `RuntimeError` with buffer name and shapes if src > dst cap, and `assert dst.shape == src_view.shape` catches dtype-cast width drift |
| Clone views before external return (P8) | Fix C | `return dst.clone()` produces a fresh allocation independent from the persistent shadow; eliminates cross-iteration aliasing |

## Expected behavior by concurrency

| C | pre_t1b (baseline) | t1b_broken | W5 fix (expected) |
|---|---|---|---|
| 4 | ~2,000 tok/s | passed | ~2,000 tok/s |
| 16 | ~3,500 tok/s | crash | ~3,500 tok/s |
| 32 | ~4,489 tok/s (peak) | crash (43 tok/s fallback) | ~4,489 tok/s (target) |
| 64 | ~4,200 tok/s | crash | ~4,200 tok/s |
| 128 | ~3,800 tok/s | crash | ~3,800 tok/s |

The clone-after-copy adds one extra device-to-device memcpy per step per
layer for each of 4 small metadata tensors (~2-4 KB each). At 1.8 TB/s HBM
this is well under 1 µs/copy; total ~5 µs/step, negligible vs the ~2 ms/step
observed at peak throughput.

## Remaining risks

- **Dtype cast allocation.** `src.to(dst.dtype)` when `src.dtype !=
  dst.dtype` allocates a new tensor from the default pool. This only fires
  if vLLM changes `query_start_loc` from int32 to a different dtype between
  versions. Current shadow expects int64; cast produces a one-shot alloc.
  The allocation is on the non-capturing path only (CUDA graph capture
  bypasses shadow). No P8 risk here because the cast result is consumed
  immediately by `dst.copy_()`.

- **Shadow memory cost.** At `_max_seqs = max_num_seqs + 4096 = 4352`,
  `_max_tokens = max(max_num_batched_tokens, ...) + 4096 ≈ 10+ KB`,
  `_max_blocks_per_seq = 256`: shadow block_table is ~4.3 MB, shadow
  slot_mapping ~80 KB, shadow ud_token_block_table ~10 MB. Total < 15 MB
  on device, shared across all layers (per-backend instance). Negligible.

- **Clone pool pressure.** The clone after copy runs during piecewise
  eager attention, not during CUDA graph capture. It allocates from the
  default memory pool ~4 small tensors per layer per step. This is the
  same allocation pattern as pre_t1b (which peaked at 4,489 tok/s), so no
  new regression.

## Parent bench recipe

```bash
# Launch (Gemma 4 26B NVFP4, FusenCache-enabled)
./serve_gemma4.sh serving   # or launch_gemma4_swa.sh variant with fusen_kv

# Bench sweep
for C in 4 16 32 64 128; do
  uv run bench_gemma4_nvfp4.py \
    --concurrency $C --max-tokens 128 \
    --output bench_t1b_W5_c${C}.json
done

# Pass/fail gates
# - zero cudaErrorIllegalAddress / CUDA OOM
# - peak throughput >= 4,489 tok/s at C=32 (within 2% of pre_t1b baseline)
# - numeric output matches pre_t1b within 1e-3 (spot-check logprobs on 5 prompts)
```

## Lines changed

`fusen_kv/backend.py`:
- `__init__` shadow-buffer allocation block: ~110 new lines (buffer
  allocation + config reads + logging)
- `forward()` metadata isolation block: `.clone()` calls replaced by
  `_shadow_copy()` helper + 4 invocations (~80 new lines incl. helper)
- Net: ~190 added, 13 removed.

## Test notes

- CPU-only code edit. Parent runs the benches in the NVFP4 container.
- Syntax verified via `python3 -c "import ast; ast.parse(...)"`.
- The graph-capture branch (`_capturing == True`) still uses originals
  directly — no shadow, no clone — because CUDA graph ordering
  guarantees the metadata is stable during capture. No change in that
  path.
