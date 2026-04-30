# fused_norm_gemma4_audit_20260419.md

**Tag:** `W7_fused_norm_gemma4_symmetric_audit`
**Date:** 2026-04-19
**Auditor:** agent (symmetric audit vs. Qwen3 v1 bug class)
**Verdict:** **CORRECT DISPATCH — but W7 fix (visibility only) already applied; P2 count assertion MISSING**

---

## Section 1: Verdict

**CORRECT-DISPATCH / CORRECT-FIRING (dispatch is clean) — NOT equivalent to Qwen3 v1.**

`fused_norm_fp4_integration.py` uses `.backend` (not `.kernel`) at lines 91, 163, and 187.
The W7 fix (`silent_none_fixes_20260419.md §Fix 3`) was already applied to the file: the `else:` branches at lines 172–179 (attn) and 197–204 (MLP) now emit `logger.warning(...)` per-layer on fallthrough instead of the silent `self._fused_attn_fn = None`.

**The sweep flag (C2) was correct in identifying the topology smell but the fix (WARNING visibility) was already committed.** The dispatch itself was never broken the way Qwen3 v1 was.

**Remaining gap:** No P2 count assertion. The module never logs how many of the 48 layers actually built a non-None `_fused_attn_fn` / `_fused_mlp_fn`. A partial-layer failure (e.g. 1 MoE layer with different `quant_method`) would be invisible at launch time even with W7 visibility fix applied.

**Env gate:** `fused_norm_fp4_integration.py` has NO env gate. It is activated by `launch_fused_vllm.py` which source-patches `gemma4.py` directly (not via this module's `apply_fused_norm_fp4_patch()`). The `FUSED_KERNEL_SO` env var is set by `serve_gemma4_fused.sh:122` — launcher correctly sets it. No dormant gate.

---

## Section 2: Line-by-line comparison table

| Attribute | Qwen3 v1 (`wire_fused_norm_fp4_qwen3.py`) | Qwen3 v2 (`wire_fused_norm_fp4_qwen3_v2.py`) | Gemma4 (`fused_norm_fp4_integration.py`) |
|---|---|---|---|
| **Dispatch attr** | `quant_method.kernel` (line 101) — **WRONG** | `quant_method.backend` (line 122) — correct | `qkv_qm.backend` (line 91, 163, 187) — correct |
| **None check** | `if kernel is None: return None` — silent | `if backend is None: logger.debug(...); return None` | `if not hasattr(qkv_qm, 'backend'): logger.warning(...)` — W7 fix applied |
| **Marlin guard** | `kernel_name not in (...)` — wrong check class | `backend == NvFp4LinearBackend.MARLIN` | `qkv_qm.backend != NvFp4LinearBackend.MARLIN` |
| **Backend dispatch** | if/elif on `kernel_name` string — never reached | if/elif on `backend.value` / enum | if/elif on `backend.value` / enum |
| **Fallthrough log level** | None (silent) | `logger.debug(...)` | `logger.warning(...)` (W7 applied) |
| **P2 count assertion** | Absent | Absent | **Absent** |
| **Env gate** | `AUTOKERNEL_FUSED_NORM_FP4_QWEN3=0` disables | Same | **None** — always on if `apply_fused_norm_fp4_patch()` called |
| **Activation path** | `register()` vLLM plugin entry point | Same | `launch_fused_vllm.py` source-patches `gemma4.py` directly |
| **Actual broken result** | 0/48 fusions — all silently unfused | 48/48 fusions active | Expected 96/96 (48 attn + 48 MLP) — unverified at runtime |

**Key finding:** Gemma4 was the REFERENCE for the Qwen3 v2 fix (per `fused_norm_qwen3_fix_diff.md:23-25` and `wire_fused_norm_fp4_qwen3_v2.py:19-21`). The sweep flag C2 reported the pre-W7 state; W7 already closed it.

---

## Section 3: No broken dispatch — fix N/A

The `.kernel` → `.backend` fix does **not apply** to `fused_norm_fp4_integration.py`. Dispatch was always on `.backend`.

**However:** the following 5-line P2 addition should be applied to `apply_fused_norm_fp4_patch()` to close the count-assertion gap. This is NOT a semantic fix (dispatch is correct) — it converts a silent partial-failure into a loud assertion:

```python
# Add after Gemma4DecoderLayer.forward = _patched_forward (line 283):
def _count_fused_layers(model):
    attn_ok = sum(1 for l in model.layers if getattr(l, '_fused_attn_fn', None) is not None)
    mlp_ok  = sum(1 for l in model.layers if getattr(l, '_fused_mlp_fn',  None) is not None)
    logger.warning("[FUSED-NORM-GEMMA4] P2 count after first forward: "
                   "attn=%d/48 mlp=%d/48 (expect 48 each)", attn_ok, mlp_ok)
```

This requires hooking into first-forward; alternatively add a post-load call site in `launch_fused_vllm.py` after `apply_fused_norm_fp4_patch()` is invoked. **Parent to review before applying — not a trivial single-line fix (requires model-object access).**

---

## Section 4: Parent verify recipe

**Gemma4 fused-norm is dispatch-correct. To confirm it is ALSO firing (P2):**

1. Launch `serve_gemma4_fused.sh` (sets `FUSED_KERNEL_SO`, mounts `/patches`, calls `launch_fused_vllm.py`)
2. After `/health` is ready, run: `docker logs vllm-gemma4 2>&1 | grep -E 'FUSED-NORM-GEMMA4|fused callable built|fused attn|fused mlp'`
3. Expected: **96 lines** of `[FUSED-NORM-GEMMA4] layer X: fused callable built OK` or similar (one per layer per path) — OR the W7 WARNING lines if `.backend` is absent on any layer.
4. If 0 lines of either kind appear: the `_patched_forward` is installed but the lazy-init branch was never reached (first forward not yet run or forward path not hit). Force a single generate call then recheck.
5. **Throughput baseline:** Gemma4 T2-N C=32 → 1,798 tok/s (baseline). Post-verify expected: 1,852–1,924 tok/s (+3–7%) IF fused-norm was previously dormant. If baseline already reflects fused-norm active, no uplift expected.
6. Cross-check: `docker logs vllm-gemma4 2>&1 | grep -c 'fused callable built'` should return **96** (48 attn + 48 MLP paths, each built once on first forward).

**Note on C2 impact re-assessment:** Since Gemma4's dispatch was always `.backend`-based (correct), the +3–7% estimate in the sweep was for the case where the else-branch was silently tripping. With W7 applied, if the WARNING lines do NOT appear in production logs, the fusions ARE firing → baseline already includes the fusion gain → no additional uplift from further fixing. The sweep's P(phantom)=0.20 estimate now looks low; the real scenario is CORRECT-AND-FIRING with no recovery left.

---

*No patch applied — dispatch is correct, W7 visibility fix already committed. P2 count assertion deferred to parent review (requires model-object access in apply call site).*
