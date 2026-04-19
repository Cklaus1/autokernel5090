# silent_none_fixes_20260419.md

**Tag:** `W7_silent_none_5_fixes`
**Status:** All 5 fixes applied.
**Parent investigation doc:** `plans/silent_none_sweep_20260419.md` (not modified).
**Pattern reference:** `plans/KILL_PATTERNS.md §P1`

---

## Fix 1 — `patches/launch_fused_vllm.py`

**Pattern:** `_FUSED_OP = getattr(..., None)` with no warning when None.

**Before (line ~86 of the FUSED_CODE string injected into gemma4.py):**
```python
_FUSED_OP = getattr(_torch_fused.ops._C, "rms_norm_dynamic_fp4_quant", None)
if _FUSED_OP is not None:
```

**After:**
```python
_FUSED_OP = getattr(_torch_fused.ops._C, "rms_norm_dynamic_fp4_quant", None)
# [FIXED-SILENT-NONE] Fix 1 — Ref: plans/silent_none_sweep_20260419.md §C3, plans/KILL_PATTERNS.md §P1
if _FUSED_OP is None:
    import warnings as _warn_mod
    _warn_mod.warn(
        "[FUSED-LAUNCH] rms_norm_dynamic_fp4_quant not in subprocess — "
        "unfused baseline active (FUSED_KERNEL_SO=<path>)",
        RuntimeWarning,
        stacklevel=1,
    )
if _FUSED_OP is not None:
```

**Module-init banner added:** `logger.info("[FIXED-SILENT-NONE] launch_fused_vllm: ...")` at module top.

**Expected WARNING:** `RuntimeWarning: [FUSED-LAUNCH] rms_norm_dynamic_fp4_quant not in subprocess — unfused baseline active`

---

## Fix 2 — `patches/swa_gemma4_plugin.py`

**Pattern:** `logger.debug(...)` on metadata stash failure; silent fallthrough when `_swa_block_table` / `_swa_seq_lens` is None.

### 2a — metadata stash (line ~100)

**Before:**
```python
except Exception as e:
    logger.debug("[SWA] could not stash block_table/seq_lens: %s", e)
```

**After:**
```python
except Exception as e:
    # [FIXED-SILENT-NONE] Fix 2 — ...
    logger.warning(
        "[SWA] metadata stash FAILED on %s (%s) — SWA kernel will NOT fire; "
        "fallback is silent at forward time (class=%s)", ...)
```

### 2b — forward-time None check (line ~201-205)

**Before:**
```python
if block_table is None or seq_lens is None:
    return _orig_forward(...)
```

**After:**
```python
if block_table is None or seq_lens is None:
    # [FIXED-SILENT-NONE] Fix 2 — ...
    logger.warning(
        "[SWA] _swa_block_table=None/_swa_seq_lens=None on %s — "
        "SWA Triton kernel skipped; falling back to FlashInfer (plugin installed but ineffective)", ...)
    return _orig_forward(...)
```

**Module-init banner added:** `logger.info("[FIXED-SILENT-NONE] swa_gemma4_plugin: ...")` at module top.

**Expected WARNINGs:**
- `[SWA] metadata stash FAILED on FlashInferMetadata (...) — SWA kernel will NOT fire`
- `[SWA] _swa_block_table=None _swa_seq_lens=None on FlashInferMetadata — SWA Triton kernel skipped`

---

## Fix 3 — `patches/fused_norm_fp4_integration.py`

**Pattern:** `else: self._fused_attn_fn = None` / `else: self._fused_mlp_fn = None` with no log.

### 3a — attn path (post-qkv_proj, line ~168)

**Before:**
```python
else:
    self._fused_attn_fn = None
```

**After:**
```python
else:
    # [FIXED-SILENT-NONE] Fix 3 — Ref: plans/silent_none_sweep_20260419.md §C2, plans/KILL_PATTERNS.md §P1
    logger.warning(
        "[FUSED-NORM-GEMMA4] fallthrough on %s — expected .backend attr "
        "(layer=%s); attn fusion DISABLED for this layer", ...)
    self._fused_attn_fn = None
```

### 3b — MLP path (post-gate_up_proj, line ~186)

**Before:**
```python
else:
    self._fused_mlp_fn = None
```

**After:**
```python
else:
    # [FIXED-SILENT-NONE] Fix 3 — ...
    logger.warning(
        "[FUSED-NORM-GEMMA4] fallthrough on %s — expected .backend attr "
        "(layer=%s); MLP fusion DISABLED for this layer", ...)
    self._fused_mlp_fn = None
```

**Module-init banner added:** `logger.info("[FIXED-SILENT-NONE] fused_norm_fp4_integration: ...")` at module top.

**Expected WARNING (per layer, first forward only):**
`[FUSED-NORM-GEMMA4] fallthrough on NoneType — expected .backend attr (layer=0); attn fusion DISABLED`

---

## Fix 4 — `fusen_kv/backend.py`

**Pattern:** `except AttributeError: pass` around `torch.ops.fusencache.store_kv` check.

**Before (line ~49-50):**
```python
except AttributeError:
    pass
```

**After:**
```python
except AttributeError:
    _HAS_CPP_STORE = False
    logging.getLogger(__name__).warning(
        "[FUSEN-KV] C++ store_kv op not registered in %s — "
        "Triton fallback active (reverts the CUDA-graph-safe promise)",
        _FUSENCACHE_CPP_SO,
    )
```

**Module-init banner added:** `logging.getLogger(__name__).info("[FIXED-SILENT-NONE] fusen_kv/backend: ...")` at module top.

**Expected WARNING:** `[FUSEN-KV] C++ store_kv op not registered in /tmp/build_fusencache/fusencache_decode.so — Triton fallback active`

---

## Fix 5 — `patches/prefix_aware_scheduler.py`

**Pattern:** `getattr(request, "block_hashes", None)` returning None silently degrades sort to FIFO.

**Before (line ~49-56):**
```python
bh = getattr(request, "block_hashes", None)
first_hash = 0
if bh:
    ...
```

**After:**
```python
global _bh_none_warned
bh = getattr(request, "block_hashes", None)
# [FIXED-SILENT-NONE] Fix 5 — Ref: plans/silent_none_sweep_20260419.md §C5, plans/KILL_PATTERNS.md §P1
if not _bh_none_warned and bh is None:
    logger.warning(
        "[PREFIX-SCHED] request has no block_hashes — sort degenerates to FIFO "
        "(plugin active but no reordering; prefix caching may be off or "
        "Request.block_hashes renamed upstream)"
    )
    _bh_none_warned = True
first_hash = 0
if bh:
    ...
```

Module-level `_bh_none_warned = False` flag added above `_prefix_key()`.

**Module-init banner added:** `logger.info("[FIXED-SILENT-NONE] prefix_aware_scheduler: ...")` at module top.

**Expected WARNING (once per process):** `[PREFIX-SCHED] request has no block_hashes — sort degenerates to FIFO`

---

## Startup grep verification

```bash
grep -E "\[FIXED-SILENT-NONE\]|\[FUSED-LAUNCH\]|\[SWA\].*FAILED|\[SWA\].*block_table=None|\[FUSED-NORM-GEMMA4\]|\[FUSEN-KV\].*store_kv|\[PREFIX-SCHED\].*block_hashes" /path/to/serve.log
```

One-liner for container startup logs:
```bash
docker logs <container> 2>&1 | grep -E "\[FIXED-SILENT-NONE\]"
```
Expected: 5 `[FIXED-SILENT-NONE]` INFO lines at startup (one per patched module).

---

## Behavioral impact

All 5 fixes are **logging-only**. Fallthrough behavior is 100% preserved. No tensor ops, no control-flow changes, no new imports at hot-path. The only observable change is log output.
