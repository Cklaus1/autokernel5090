# Dead-Code Audit — patches/ candidates
**Date:** 2026-04-19  
**Tag:** W5_23_dead_code_audit  
**Auditor:** Claude agent (Sonnet 4.6)  
**Scope:** 5 candidate patch files identified in task #23

---

## Audit Method

For each file:
1. `grep -rn` across all `.py` / `.sh` / `.md` (excluding `__pycache__`, `.pyc`, `.venv`).
2. Checked every launcher (`launch_*.sh`, `serve_*.sh`) and `docker/apply_patches.sh`.
3. Checked `fusen_kv/`, `fusen_solver/`, `patches/launch_*.py`.
4. Categorised: **LIVE** / **ORPHANED** / **REFERENCED-IN-PLANS-ONLY**.

---

## Results

| File | Classification | Evidence |
|------|---------------|----------|
| `patches/fusencache_concurrency_fix.py` | **LIVE** | `fusen_kv/backend.py:930,989,1075` contain `torch.cuda.synchronize()` matching the patch; patch was applied as a one-time in-place source edit. Confirmed in `plans/tier2_3_audit_20260419.md:320`. |
| `patches/apply_gemma4_patch.py` | **REFERENCED-IN-PLANS-ONLY** | Mentioned in `patches/wire_fused_norm_fp4.py` usage comments and `plans/tier2_3_audit_20260419.md:321`. No `.sh` launcher auto-runs it; must be invoked manually inside Docker. It is a one-shot in-place editor, not a runtime import. Pending work: wire into `serve_gemma4_fused.sh` Step 2 or the Docker entrypoint. |
| `patches/apply_moe_fix.py` | **REFERENCED-IN-PLANS-ONLY** | Referenced only in `GEMMA4_NVFP4_STATUS.md:93,156` as a workflow step. No launcher auto-runs it. Requires `VLLM_NVFP4_MOE_LOOP=1` to activate the per-expert path once applied. Pending work: add to Docker entrypoint or Gemma4 launcher. |
| `patches/early_abort_wrapper.py` | **ORPHANED** | Zero imports outside `patches/` itself. Referenced only in `plans/early_kv_termination.md` and `plans/rtx_pro6000_experiments.md` as design reference. Not wired into `fusen_solver/`, `fusen_kv/`, or any HTTP serving path. `plans/tier2_3_audit_20260419.md:398` explicitly calls it out as pending wire-or-delete. |
| `patches/bench_throughput.py` | **ORPHANED** | Zero imports, zero launcher references, not mentioned in any plan. Simple ad-hoc HTTP benchmarking script; superseded by `bench_gemma4_nvfp4.py`, `bench_concurrency_sweep.py`, and `bench_event_fence.py`. |

**Summary: LIVE=1, ORPHANED=2, REFERENCED-IN-PLANS-ONLY=2**

---

## Actions Taken

### ORPHANED files — top-of-file comment added (conservative; not deleted)

- `patches/early_abort_wrapper.py` — line 1: `# ORPHANED — not imported by any launcher or runtime path as of 2026-04-19`
- `patches/bench_throughput.py` — line 2: `# ORPHANED — standalone script never imported or called from any launcher as of 2026-04-19; superseded by bench_gemma4_nvfp4.py and bench_concurrency_sweep.py`

### REFERENCED-IN-PLANS-ONLY files — header comment added

- `patches/apply_moe_fix.py` — line 1: `# REFERENCED-IN-PLANS-ONLY — manually run inside Docker; no launcher auto-invokes this`
- `patches/apply_gemma4_patch.py` — line 2: `# REFERENCED-IN-PLANS-ONLY — run manually inside Docker container; no .sh launcher auto-invokes this`

No files were moved to `_archive/` or deleted.

---

## Banked-But-Silently-Not-Firing Findings

### 1. `apply_moe_fix.py` — VLLM_NVFP4_MOE_LOOP=1 guard is silently inert unless env is set AND script was manually applied

The patch adds a dispatcher inside `run_cutlass_moe_fp4` that checks `os.environ.get("VLLM_NVFP4_MOE_LOOP", "0") == "1"`. The check is valid, but the patch must first have been manually run inside the Docker container before the env var matters. No launcher sets `VLLM_NVFP4_MOE_LOOP=1` and no launcher calls `apply_moe_fix.py`. Both conditions must hold simultaneously; currently neither does. **Same failure class as the fused-norm `.kernel` getattr bug: code present, code correct, but path never reached.**

### 2. `apply_gemma4_patch.py` — fused_add_rms_norm import injected into Gemma4 source but `.kernel` reference risk

`apply_gemma4_patch.py` injects `from vllm.model_executor.layers.layernorm import fused_add_rms_norm as _fused_add_rms_norm` and replaces the norm+add+norm sequence. If the vLLM build inside the container does not expose `fused_add_rms_norm` (symbol available only since vLLM ≥0.6.x), the patched model will crash with `ImportError` at load time. No version guard exists. Low risk on the current pinned image, but a silent breakage vector if the base image is ever updated.

### 3. `early_abort_wrapper.py` — designed for `fusen_solver/` but fusen_solver uses direct vLLM AsyncLLM

The docstring says `Usage (fusen_solver / OpenAI-compatible proxy)`, but `fusen_solver/` has no import of this module. Under high concurrency (`C=256+`) described in `plans/rtx_pro6000_experiments.md`, the wrapper would yield a measurable KV-cache benefit; without it, those blocks stay live until natural EOS. This is a **delivery gap**, not a bug — but the intended optimization has been dormant since the file was written.

---

## Recommended Next Steps

1. **Wire `early_abort_wrapper.py`** into `fusen_solver/` generation loop (~20 LoC hookup as described in `plans/tier2_3_audit_20260419.md:398`) or explicitly delete with `git rm`.
2. **Add `apply_gemma4_patch.py` and `apply_moe_fix.py`** to a Docker entrypoint or `serve_gemma4_fused.sh` Step 2, with a version guard on the import symbol.
3. **Rule for all `patches/*.py`**: add a comment block at top stating which launcher(s) invoke it and assert their presence on container startup (see `plans/tier2_3_audit_20260419.md:412`).
