# silent_none_sweep_20260419.md

**Tag:** `W6_silent_none_sweep`
**Scope:** exhaustive grep + manual inspection of `patches/*.py`, `fusen_kv/*.py`,
`fusen_solver/*.py`, `autokernel_v2/plugins/*.py`, `kernels/*.py`, root-level `*.py`.
Excluded `_archive/` per instructions.
**Patterns scanned:**
1. `getattr(obj, "attr", None)` silent defaults in plugin/dispatch code
2. `if not hasattr(obj, "attr"):` silent fall-throughs
3. `try: ... except AttributeError: pass` bare silent swallows
4. `os.environ.get("VAR", "0") != "0"` env gates (cross-checked against every `launch_*.sh`)
5. DEBUG-level fallback logs at dispatch sites (KILL_PATTERNS §P1 says WARNING+)
6. Plugin registration vs. actual hot-path installation

**Totals:**
- `getattr(*, *, None)`: **48 instances** across 22 files
- `if not hasattr(*):`: **17 instances**
- `except AttributeError: pass`: **1 instance**
- `os.environ.get(...) == "1"` gates with `"0"` default and no launcher sets the var: **4 instances**
- **Classification: SAFE 38, SUSPECT 12, CRITICAL 5** (confirmed 3 prior bugs already patched).

---

## CRITICAL findings (top 5, ranked by expected throughput impact)

### C1. `swa_gemma4_plugin.py:199-205` — `_swa_block_table`/`_swa_seq_lens` silent disable

**Location:** `patches/swa_gemma4_plugin.py:199-205`, with the paired metadata stash at `swa_gemma4_plugin.py:93-100`.

**Why suspect (P1):**
The metadata attach at line 93-100 catches `Exception` and only `logger.debug(...)` on failure:
```python
try:
    md._swa_block_table = common_attn_metadata.block_table_tensor
    md._swa_seq_lens = common_attn_metadata.seq_lens
except Exception as e:
    logger.debug("[SWA] could not stash block_table/seq_lens: %s", e)   # DEBUG!
```
Then forward() at 199-205 does `getattr(..., "_swa_block_table", None)` and falls through to `_orig_forward` if None, with NO log at all. If vLLM renames `block_table_tensor` or the common metadata field changes, the banner still prints "[SWA] Installed" but the Triton SWA kernel fires zero times. Same topology as the Qwen3 `.kernel` vs `.backend` bug.

**Fix (2-3 lines):**
```python
# line ~99-100:
except Exception as e:
    logger.warning(
        "[SWA] metadata stash FAILED on %s (%s) — SWA kernel will NOT fire; "
        "fall-back is silent at forward time", type(common_attn_metadata).__name__, e)
# line ~201-205: add WARN on first-N Nones, not silent fallthrough
```

**Impact if firing phantom:** the plugin is gated by `--kv-cache-dtype fp8` + SWA window. SWA+FP8 Triton path banked 1.19-1.62× decode throughput per KILL_PATTERNS §5 known-good idioms. If the metadata stash silently fails on an upstream vLLM rename, we're at **0× of that bank** (pure FlashInfer dense path). **Est. recovery: +19-62% decode on Gemma4 FP8 KV.**

---

### C2. `fused_norm_fp4_integration.py:157-168, 174-186` — Gemma4 silent `_fused_*_fn = None`

**Location:** `patches/fused_norm_fp4_integration.py:155-189`.

**Why suspect (P1):**
```python
qkv_qm = getattr(self.self_attn.qkv_proj, 'quant_method', None)
if (qkv_qm is not None and hasattr(qkv_qm, 'backend')):
    ...
else:
    self._fused_attn_fn = None   # silent — no log, no banner change
```
If `quant_method` is None OR lacks `.backend` (a future vLLM attr rename, or a non-NVFP4 model config), fusion is disabled for that layer with **zero visibility** — identical topology to the Qwen3 v1 bug (which cost +19% for weeks). The audit at `plans/kill_audit_20260419.md §3` CONFIRM_KILL'd this plugin on the premise the attribute check is correct; but the silent `else: None` branch is the exact smell pattern. If 1 of 48 layers misses, you get a partial regression invisible to launch-time banners.

**Fix (3 lines):**
```python
if qkv_qm is None or not hasattr(qkv_qm, 'backend'):
    logger.warning("[fused_norm] layer %s: qkv_proj has no .backend (type=%s) "
                   "— fusion DISABLED for this layer",
                   getattr(self, 'layer_idx', '?'),
                   type(qkv_qm).__name__)
    self._fused_attn_fn = None
```
Also log a count-assertion after first forward (per §P2).

**Impact:** Gemma4 fused-norm projects +7.7% e2e per CLAUDE.md; if any layer silently drops back, that partial miss is a significant throughput leak. **Est. recovery if partially dormant: +3-7% e2e on Gemma4 NVFP4.**

---

### C3. `launch_fused_vllm.py:86-87` — `_FUSED_OP` silent-None when .so missing

**Location:** `patches/launch_fused_vllm.py:86-87` (source-patched into gemma4.py).

**Why critical (P1):**
```python
_FUSED_OP = getattr(_torch_fused.ops._C, "rms_norm_dynamic_fp4_quant", None)
if _FUSED_OP is not None:   # silently skips register_fake & fast path
    ...
```
Then the patched forward at line 155 does `if _FUSED_OP is not None and hasattr(...)` and falls through to the ORIGINAL Gemma4 forward (unfused) with no log. If `/tmp/fused_kernel/fused_rms_norm_fp4.so` isn't present in the EngineCore subprocess (separate process from the launcher — common cause: the `torch.ops.load_library` in the parent doesn't carry), the patch applies to source code but **every layer runs unfused**. The launcher prints "[FUSED_NORM_FP4 PATCH] applied" but 0 fusions fire. Same cost class as the Qwen3 bug.

**Fix (3-5 lines):** At apply time, assert the op exists in the *subprocess* and raise with a clear message:
```python
if _FUSED_OP is None:
    import warnings; warnings.warn(
        f"[FUSED] _FUSED_OP is None in EngineCore (FUSED_KERNEL_SO={_FUSED_SO} "
        f"exists={_os_fused.path.exists(_FUSED_SO)}); fusion DISABLED, running "
        f"unfused baseline. Banner 'patch applied' is misleading.",
        stacklevel=2)
```
And log per-layer-first-fire count to verify 48 actual fusions.

**Impact:** Whole-model fusion is either on (+8-15% e2e per wire_fused_norm_fp4.py docstring) or off. Known classical silent-None. **Est. recovery: +8-15% e2e.**

---

### C4. `fusen_kv/backend.py:44-50` — bare `except AttributeError: pass` on `fusencache.store_kv`

**Location:** `fusen_kv/backend.py:44-50`:
```python
try:
    _ = torch.ops.fusencache.store_kv
    _HAS_CPP_STORE = True
    logger.info("FusenCache C++ store kernel loaded (CUDA graph safe)")
except AttributeError:
    pass   # silent — no log, _HAS_CPP_STORE stays False
```
If the C++ .so registered `decode_attention` but not `store_kv` (e.g. partial rebuild), decode works but the CUDA-graph-safe store path silently falls back to Triton — which has the JIT/concurrency bugs the C++ kernel was built to avoid (per CLAUDE.md "bypasses Triton JIT + FlashInfer concurrency bugs"). No `[WARN]` in logs.

**Fix (3 lines):**
```python
except AttributeError:
    logger.warning("FusenCache C++ store_kv NOT registered in %s — "
                   "falling back to Triton store (JIT + concurrency risk)",
                   _FUSENCACHE_CPP_SO)
```

**Impact:** directly tied to T1-B piecewise-graph throughput recovery (kill_audit #2 projects +20× at C=128). If store_kv is silently Triton-routed, the "CUDA graph safe" promise fails. **Est. recovery: gate on the banked T1-B +20× not being a phantom.**

---

### C5. `prefix_aware_scheduler.py:49` — `block_hashes` getattr-None degrades to FIFO

**Location:** `patches/prefix_aware_scheduler.py:49-56`:
```python
bh = getattr(request, "block_hashes", None)
first_hash = 0
if bh:
    h = bh[0]
    first_hash = hash(h)
return (first_hash, request.arrival_time)
```
If `Request.block_hashes` is renamed upstream or not populated (e.g. prefix caching off), `bh` is None for ALL requests → every request bucketed at `first_hash=0` → sort becomes identity (arrival-time only) → the plugin silently degenerates to a no-op FIFO. Banner "[prefix_aware_sched] installed" still prints. Classic §P1.

**Fix (2 lines):**
```python
bh = getattr(request, "block_hashes", None)
if bh is None:
    logger.warning("[prefix_aware_sched] request %s has no .block_hashes "
                   "(prefix caching off?) — reorder degenerates to FIFO",
                   getattr(request, 'request_id', '?'))
```
(Rate-limit to 1 WARNING per process to avoid spam.)

**Impact:** prefix-aware reorder is the mechanism behind the multi-system-prompt serving throughput win (cf. `plans/prefix_aware_sched_*.md`). Silent degeneracy means the launcher proclaims the fix is active while KV cache churn continues. **Est. recovery: 5-15% serving throughput on multi-prompt workloads.**

---

## SUSPECT findings (secondary — worth audit, lower impact)

| File:line | Pattern | Attribute / env | Log level | Est. impact |
|---|---|---|---|---|
| `patches/apply_moe_fix.py:87` | env guard never set | `VLLM_NVFP4_MOE_LOOP` | — | **(confirmed dormant)** |
| `patches/dynamic_chunked_prefill.py:97-98,181` | bare `except Exception: pass`; `AUTOKERNEL_DYNAMIC_CHUNK_AUTO` never set | — | none | dormant auto-path; register() plugin path OK |
| `patches/fused_shuffle_quant_wrapper.py:292` | `AUTOKERNEL_FUSED_RESHAPE_SCALES` never set by launchers | — | — | old code path dormant |
| `fusencache/config.py:54-57` | `FUSEN_SELECTIVE` never set | — | — | selective caching dormant |
| `patches/wire_fused_norm_fp4_qwen3_v2.py:114-125` | DEBUG-level on None-return of `quant_method`/`backend` | `.quant_method`, `.backend` | DEBUG | silent per-layer fallthrough; same topology as C2 |
| `patches/swa_sparse_plugin.py:52` | `AUTOKERNEL_SWA_GUARD_RATIO` default 1.25 — OK but undocumented | — | — | calibration drift risk |
| `patches/fusencache/v1/spec_decode/eagle.py:1448-1462` | chain of None-returning `getattr` then `hasattr(self.model, 'get_top_tokens')` | — | none | eagle draft silent disable |
| `fusen_kv/plugin.py:130,268` | `kv_cache_dtype` None → backend selection silently skips FusenKV | `kv_cache_dtype` | info | infrastructure; config drift silent |
| `fusen_kv/backend.py:499,484` | `num_gpu_blocks`/`max_cudagraph_capture_size` getattr-None with silent defaults | — | none | shadow metadata sized at safe default 256; benign |
| `patches/fusencache/model_executor/layers/attention/attention.py:229` | `kv_cache_scheme` None | — | none | upstream copy; benign |
| `patches/vllm_async_scheduling_fix.py:274` | `use_async_scheduling` missing → event never created, silent no-op | — | none | race fix dormant in non-async mode (expected) but NO banner distinguishes |
| `kernels/csrc/test_*.py` usages | test-only | — | — | benign |

---

## Env-var gates never set by any launcher (verified `launch_*.sh` sweep)

| Env var | Source file | Default | Set-by launcher? |
|---|---|---|---|
| `VLLM_NVFP4_MOE_LOOP` | `patches/apply_moe_fix.py:87` | `"0"` | **NO** (known bug #3) |
| `AUTOKERNEL_DYNAMIC_CHUNK_AUTO` | `patches/dynamic_chunked_prefill.py:181` | `"0"` | **NO** (register() entry still works) |
| `AUTOKERNEL_FUSED_RESHAPE_SCALES` | `patches/fused_shuffle_quant_wrapper.py:292` | `"0"` | **NO** (dormant old code branch) |
| `FUSEN_SELECTIVE` | `fusencache/config.py:54` | `"0"` | **NO** (selective caching dormant) |
| `AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK` | `patches/fp8_decode_monkey_patch.py:42` | `"0"` | only set in one verify launcher; appropriate (dev-only) |
| `AUTOKERNEL_SWA_SPARSE_FP8` | `patches/swa_gemma4_plugin.py:124` | `"1"` | set when intended |
| `AUTOKERNEL_SWA_GUARD_RATIO` | `patches/swa_sparse_plugin.py:52` | `"1.25"` | **NO** (works via default; drift risk) |

Of these, only **`VLLM_NVFP4_MOE_LOOP`** is a confirmed-critical dormancy (MoE loop workaround for the SM120 grouped GEMM bug). The other three dormant env gates are either intentionally off or have parallel `register()` entry-point paths.

---

## Ranked impact table (CRITICAL)

| Rank | ID | File:line | Bug kind | Est. throughput impact if phantom | P(phantom) | Fix LOC |
|---:|---|---|---|---:|---:|---:|
| 1 | C3 | `patches/launch_fused_vllm.py:86` | Missing .so → silent unfused fallback | **+8-15% Gemma4** | 0.30 | 3-5 |
| 2 | C1 | `patches/swa_gemma4_plugin.py:93,199` | Metadata stash DEBUG-log → silent SWA miss | **+19-62% decode SWA+FP8** | 0.15 | 3 |
| 3 | C2 | `patches/fused_norm_fp4_integration.py:167,186` | Per-layer silent `None` branch | **+3-7% Gemma4** | 0.20 | 3 |
| 4 | C4 | `fusen_kv/backend.py:44-50` | Bare `except AttributeError: pass` | **+0-20× T1-B @ C=128** (bank-dependent) | 0.25 | 3 |
| 5 | C5 | `patches/prefix_aware_scheduler.py:49` | `block_hashes` None → FIFO no-op | **+5-15% multi-prompt serving** | 0.20 | 2 |

---

## Systemic recommendation

**Add a lint rule to `tests/` or pre-commit:**

```python
# tests/test_no_silent_none_dispatch.py
# Grep-based guard (CPU-only) over patches/, fusen_kv/, autokernel_v2/plugins/:
# 1. Any `getattr(x, "<name>", None)` on a plugin/dispatch hot-path file must be
#    followed within 10 lines by either (a) a WARN log in the None branch, OR
#    (b) a comment '# SAFE: fallthrough expected for <class>'.
# 2. Any `except AttributeError: pass` in plugin/dispatch code fails the test.
# 3. Any env var `os.environ.get(..., "0") == "1"` guard must be matched by at
#    least one `export <NAME>=` in launch_*.sh; otherwise fails with
#    "dormant env gate — set by no launcher".
```

This catches the 5 CRITICAL patterns above plus any future recurrence. Runs in <1s. The Qwen3 `.kernel` vs `.backend` bug, the FP8 FIDecode fallback, and the `VLLM_NVFP4_MOE_LOOP` dormant gate would ALL have been caught by this one test. Estimate 15 LoC for the lint plus 20 LoC for the launcher-env cross-check.

**Secondary recommendation:** every plugin `register()` entry point should, at first forward, `assert sum(1 for layer in model.layers if layer._fused_fn is not None) >= expected_count` and **log that count at WARNING level** (per KILL_PATTERNS §P2). This converts silent partial fusion misses (like C2 per-layer) into a loud failure.

---

*Sweep methodology: ripgrep + manual inspection, ~90 minutes. False-positive allowance: yes — 12 SUSPECT entries retained in table even where impact is likely benign. Err on inclusion per instructions.*
