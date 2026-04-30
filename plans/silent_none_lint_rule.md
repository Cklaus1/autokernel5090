# silent_none_lint_rule.md

**Tag:** `W8_silent_none_lint_test`
**Parent sweep:** `plans/silent_none_sweep_20260419.md`
**Parent fixes:** `plans/silent_none_fixes_20260419.md`
**Test file:** `tests/test_no_silent_none_dispatch.py`

---

## Purpose

Prevent a recurrence of the Qwen3 fused-norm v1 silent-None bug (`.kernel` vs
`.backend` attribute name, caused +19% throughput to be lost for weeks) and the
4 other CRITICAL patterns identified in the W6 sweep. The test is CPU-only,
runs in <1 s, and requires no GPU or vLLM install.

---

## Rules

### Rule 1 — unguarded `getattr(x, "attr", None)`

**Pattern:** `getattr(x, "attr", None)` in plugin/dispatch code with no
warning on the None branch.

**Requirement:** within 10 lines of the `getattr(` call there must be either:
- A WARNING-level log call (`logger.warning(`, `logging.getLogger().warning(`,
  `warnings.warn(`, `RuntimeWarning`), OR
- A `# SAFE: <reason>` comment explaining why None is expected/safe.

**Why 10 lines:** the typical fix pattern is:
```python
x = getattr(obj, "attr", None)
if x is None:
    logger.warning("[PLUGIN] attr missing on %s", type(obj).__name__)
    return None
```
The warning appears within 3 lines. 10 lines provides comfortable buffer for
multi-line conditions while staying tight enough to catch lapses.

**Implementation:** regex scan of source text per file; no AST needed.

---

### Rule 2 — `except AttributeError: pass`

**Pattern:** `except AttributeError:` block whose body is only `pass`.

**Requirement:** fails immediately. The bare `pass` silently swallows a
missing-attribute error. The correct fix is to log at WARNING level (see
`KILL_PATTERNS §P1`).

**Implementation:** AST walk; `ExceptHandler` node with `type.id ==
"AttributeError"` and `len(body) == 1 and isinstance(body[0], ast.Pass)`.

**Already fixed:** `fusen_kv/backend.py` was the only CRITICAL instance
(W7 Fix 4); its body is now `_HAS_CPP_STORE = False; logger.warning(...)`.

---

### Rule 3 — env-gate not set by any launcher

**Pattern:** `os.environ.get("VAR", "0") == "1"` feature gate in scoped
files where no `launch_*.sh` contains `export VAR=`.

**Requirement:** every such gate must be activated by at least one launcher,
or it must appear in `KNOWN_DORMANT_ENV_GATES` with a documented reason.

**Implementation:** regex scan of all scoped .py files for the gate pattern;
glob `launch_*.sh` for `export VAR=`; set-difference.

---

## Scan scope

| Directory | Note |
|---|---|
| `patches/*.py` (recursive) | plugin/dispatch code; excludes `patches/fusencache/` (upstream vLLM copy) and `patches/flashinfer-upstream-proposal.py` (doc file) |
| `fusen_kv/*.py` (recursive) | excludes `fusen_kv/eval_perplexity.py` (eval-only model introspection) |
| `autokernel_v2/plugins/*.py` | plugin registration code |
| `kernels/*.py` | wrapper .py only; excludes `kernels/test_*.py` |

---

## Known exceptions (SAFE_GETATTR_LOCATIONS)

These specific `getattr(x, "attr", None)` calls are excluded from Rule 1
because they are genuinely safe. Each has an inline comment in the test.

| File:line | Attribute | Reason |
|---|---|---|
| `patches/swa_gemma4_plugin.py:287` | `layer_name` / `prefix` | Cosmetic label for logging only; no dispatch decision |
| `patches/vllm_async_scheduling_fix.py:281,291` | `_forward_done_event` | Optional async machinery; has immediate `is not None` guard on both getattrs |
| `patches/wire_fused_norm_fp4.py:172,173` | `cached_fp4`, `cached_scales` | Thread-local cache; None → fresh allocation below |
| `patches/t1b_piecewise_cudagraph_fix.py:136` | `num_gpu_blocks` | Shadow metadata default 256; benign per sweep §SUSPECT |
| `fusen_kv/backend.py:508` | `num_gpu_blocks` | Same as above |
| `fusen_kv/plugin.py:130,268` | `kv_cache_dtype`, `cache_dtype` | Infrastructure config drift; logged at INFO level; no throughput impact |

---

## Known dormant env gates (KNOWN_DORMANT_ENV_GATES)

These `== "1"` env gates are intentionally not set by any launcher. They
appear in the exclusion list with documented rationale.

| Env var | Source | Reason dormant |
|---|---|---|
| `AUTOKERNEL_FUSED_RESHAPE_SCALES` | `fused_shuffle_quant_wrapper.py` | Old code branch; per sweep §SUSPECT |
| `FUSEN_SELECTIVE` | (not found in scan scope) | Selective caching dormant; register() path works |
| `FUSEN_DEBUG` | `fusen_kv/backend.py` | Debug-only; dev tool |
| `FUSEN_SYNC` | `fusen_kv/backend.py` | Sync-for-crash-pinpointing; dev tool |
| `AUTOKERNEL_DYNAMIC_CHUNK_AUTO` | `patches/dynamic_chunked_prefill.py` | Auto-path dormant; register() entry-point works; per sweep §env-var table |
| `AUTOKERNEL_T2N_SILU_EPILOGUE` | `patches/fused_shuffle_quant_wrapper.py` | T2-N v3 silu epilogue; off by default until bench banks the win |

**Not in exclusion list (intentional violations):**
- `VLLM_NVFP4_MOE_LOOP` — confirmed-critical dormant gate; must be set by
  the MoE launcher when the SM120 grouped GEMM workaround is needed. This
  gate should fail Rule 3 to keep the issue visible.

---

## Expected test results on current codebase

### Should PASS (already-fixed W7 files)

| File | Reason passes |
|---|---|
| `patches/swa_gemma4_plugin.py` | All getattr-None have `logger.warning(...)` within 10 lines (Fix 2) |
| `patches/fused_norm_fp4_integration.py` | `else: logger.warning(...)` added before `self._fused_*_fn = None` (Fix 3) |
| `patches/launch_fused_vllm.py` | `_warn_mod.warn(RuntimeWarning, ...)` added after `_FUSED_OP is None` (Fix 1) |
| `fusen_kv/backend.py` | `except AttributeError:` body now has `logger.warning(...)` not `pass` (Fix 4) |
| `patches/prefix_aware_scheduler.py` | `logger.warning(...)` added in `if bh is None:` branch (Fix 5) |

### Should FAIL (known remaining issues)

| File | Rule | Reason fails |
|---|---|---|
| `patches/wire_fused_norm_fp4_qwen3.py` | Rule 1 | v1: `getattr(quant_method, "kernel", None)` → `return None` silently, no WARNING |
| `patches/wire_fused_norm_fp4_qwen3_v2.py` | Rule 1 | `getattr(..., "backend", None)` → `logger.debug(...)` (not WARNING) per sweep §SUSPECT |
| Any file with `VLLM_NVFP4_MOE_LOOP == "1"` gate | Rule 3 | Confirmed-critical dormant gate; no launcher sets it |

---

## Integration recommendation

**Recommended:** add as a CI step running on every PR that touches
`patches/`, `fusen_kv/`, `autokernel_v2/`, or `kernels/`.

```yaml
# .github/workflows/lint.yml (example)
- name: Silent-None lint
  run: python3 -m pytest tests/test_no_silent_none_dispatch.py -v
```

**Optional:** pre-commit hook for local dev:
```bash
# .pre-commit-config.yaml
- repo: local
  hooks:
  - id: silent-none-lint
    name: Silent-None dispatch lint
    entry: python3 -m pytest tests/test_no_silent_none_dispatch.py -v
    language: system
    pass_filenames: false
    types: [python]
    files: ^(patches|fusen_kv|autokernel_v2/plugins|kernels)/
```

The test has no external dependencies beyond the Python standard library and
`pytest`. It runs in <1 s (regex scan only, no GPU needed).

---

## Relationship to KILL_PATTERNS §P1

The lint rule is the automated enforcement of the "Detection rule" in
`plans/KILL_PATTERNS.md §P1`:

> every `getattr(obj, "attr", default)` in plugin/dispatch code MUST either:
> - Assert the attribute exists
> - Log the resolved class name AND resolution result on first call
> - Document the fallback semantics in a comment with the expected class

Rule 1 enforces the "log" and "document fallback" branches.
Rule 2 enforces the prohibition on `except AttributeError: pass`.
Rule 3 enforces env-gate liveness.

---

*Created: 2026-04-18. Tag: W8_silent_none_lint_test.*
