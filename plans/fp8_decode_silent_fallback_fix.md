# FP8 Decode Silent-Fallback Fix

**Tag:** `W5_A1_fp8_decode_silent_fallback`
**Date:** 2026-04-18
**File changed:** `patches/fp8_decode_monkey_patch.py`
**Related:** `plans/tier2_3_audit_20260419.md §5`, `plans/KILL_PATTERNS.md §P1`

---

## What was silenced

### Original fallback at lines 319-335 (pre-fix)

```python
elif isinstance(attn_metadata.decode, FIDecode):
    # FIDecode doesn't carry block_tables/seq_lens directly.
    # Fall back to original FlashInfer decode for this case.
    logger.debug("FIDecode path — falling back to FlashInfer native decode")
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    stride_order = FlashInferBackend.get_kv_cache_stride_order()
    kv_cache_permute = kv_cache.permute(*stride_order)
    decode_wrapper = attn_metadata.decode.wrapper
    decode_wrapper.run(
        decode_query,
        kv_cache_permute,
        k_scale=layer._k_scale_float,
        v_scale=layer._v_scale_float,
        out=output[:num_decode_tokens],
    )
    return output_padded
```

**Problem:** The log line used `logger.debug(...)` — invisible at default INFO level. If vLLM
selected `FIDecode` metadata for the decode wrapper (the common path on SM120 because
`is_device_capability(100)` returns False, which suppresses TRTLLM path selection), the
custom Triton split-K kernel **never executed** and all decode attention silently fell back
to FlashInfer native. The "+71% throughput" T2-I claim would be entirely the FlashInfer
baseline number, not the custom kernel.

A second silent fallback existed at lines 338-342 for any unrecognised decode metadata
type, with no log at all.

---

## What is now visible

### 1. Module-level counter and atexit banner

```python
_fallback_fire_count = 0  # module-level, incremented on every fallback dispatch

def _log_fallback_summary():
    if _fallback_fire_count > 0:
        logger.warning("[FP8-decode] FALLBACK SUMMARY: ... %d time(s) ...", _fallback_fire_count)
    else:
        logger.info("[FP8-decode] FALLBACK SUMMARY: fallback_fire_count=0 — all requests used custom kernel")

atexit.register(_log_fallback_summary)
```

### 2. One-time init banner at patch-apply

Promoted to `logger.warning(...)` at `apply_patch()` time, before the first request is
ever served. Contains explicit "Fallback path X active for metadata type Y" phrasing so
it appears in server startup logs even on an idle server.

### 3. Per-fallback WARNING with count and type name

Both the FIDecode branch and the unknown-type branch now emit:

```
[FP8-decode] FALLBACK #N: decode metadata is FIDecode — routing to FlashInfer native ...
```

The `#N` counter lets you see from logs how many of the 200 test requests actually hit
the fallback versus the fast path.

### 4. Assert-on-fallback env var

`AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1` causes the fallback branches to raise
`RuntimeError` instead of silently routing. Use this to fail-fast during testing to
confirm the kernel is actually running.

### 5. Behavior unchanged in non-asserting mode

All existing launchers (`MODE=patched`) continue to start and serve. Fallbacks still
happen — they are now just loud about it.

---

## Why FIDecode is the common path on SM120

`vllm.v1.attention.backends.flashinfer` selects `TRTLLMDecode` only when
`is_device_capability(100)` returns True (i.e., SM100 / Hopper). SM120 (RTX PRO 6000,
5090) returns False, so vLLM defaults to `FIDecode`. This means on this machine,
**every decode request takes the FIDecode branch**, and the custom kernel was never
exercised in any T2-I bench unless the bench was run with an explicit `TRTLLMDecode`
override.

---

## Root cause class

Pattern P1 — Silent-None dispatch (`plans/KILL_PATTERNS.md §P1`). Same class as the
Qwen3 fused-norm `.kernel` vs `.backend` bug that cost +19% throughput for weeks.

---

## Diagnostic bench plan

**Goal:** determine whether T2-I's banked "+71%" claim is real or was measured on the
FlashInfer fallback path.

### Steps

1. Launch T2-I config:
   ```bash
   # Identify the launcher
   ls patches/ launch_*.sh | grep -i fp8
   # Launch it (MODE=patched)
   ./launch_<fp8_decode>.sh
   ```

2. Send 200 requests via the standard bench script.

3. Grep logs for fallback evidence:
   ```bash
   docker logs CONTAINER 2>&1 | grep -E '\[FP8-decode\] (FALLBACK|INIT|SUMMARY)'
   ```

4. Interpret results:

   | Observation | Meaning | Action |
   |---|---|---|
   | `FALLBACK SUMMARY: fallback_fire_count=0` | Custom kernel fired on all 200 requests. Banked +71% is legitimate. | ENDORSE T2-I |
   | `FALLBACK #N` lines appear (N > 0) | N requests routed to FlashInfer. If N ≈ 200, patch was inert. | Investigate: was the bench that produced +71% run with TRTLLMDecode forced? |
   | `FALLBACK #N` with N ≈ 200 | Patch was completely inert on this hardware. Banked gain is FlashInfer baseline. | Re-bench with `TRTLLM_FORCE=1` or fix FIDecode extraction to get real custom-kernel numbers |

5. If fallback is firing, confirm with assert mode:
   ```bash
   AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1 ./launch_<fp8_decode>.sh
   # Server should hard-crash on first decode request if FIDecode path is active
   ```

---

## Next step if fallback is confirmed

If `fallback_fire_count ≈ 200`, the fix is one of:
- **Option A:** Force TRTLLM decode wrapper selection (env var or vLLM config) so
  `attn_metadata.decode` is always `TRTLLMDecode`. Then our kernel is on the hot path.
- **Option B:** Extract `block_tables` and `seq_lens` from the FIDecode wrapper's
  internal plan (requires reading FlashInfer internals — fragile but doable).
- **Option C:** Replicate the block-table lookup from `attn_metadata` common fields
  instead of metadata.decode-specific fields.

These are follow-on tasks (§5 RECOVER part 2). This fix delivers observability first.
