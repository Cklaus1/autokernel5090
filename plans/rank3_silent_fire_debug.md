# Rank 3 `[T2N-UNSHUF-WSUM]` Silent-Fire Debug

Tag: `W8_rank3_silent_fire_debug`
Date: 2026-04-18
Status: fix applied, awaiting parent relaunch verification

## TL;DR

The W7 Rank 3 fused unshuffle+weighted-sum kernel never actually fired under
the Qwen3 launcher. The `[T2N-UNSHUF-WSUM]` banner was missing because the
container's `PYTHONPATH` did not include the repo root `/autokernel`. The
import `from kernels.triton.fused_unshuffle_weightedsum import ...` therefore
raised `ModuleNotFoundError`, and the wrapper silently fell through to the
two-pass `ops.shuffle_rows + sum(dim=1)` path. The observed +7.2% throughput
at C=768 came from reverting `AUTOKERNEL_FUSED_PERSISTENT_BUF=1`, not from
Rank 3.

This is a textbook §P1 silent-None dispatch: the import fails, the `except
Exception` branch logs a WARNING under logger name `fused_shuffle_quant_wrapper`
(which is outside the vLLM `configure_vllm_logger` allowlist in some
deployments), and the caller receives `None` with no user-visible signal.

## Root cause (file:line)

`launch_qwen3_fused_norm_fp4.sh:95` sets:

```bash
export PYTHONPATH=/autokernel/patches:${PYTHONPATH:-}
```

The `patches/` directory contains only monkey-patch shims, not the Triton
kernel package. The Rank 3 kernel lives at
`/autokernel/kernels/triton/fused_unshuffle_weightedsum.py` and is imported
as `kernels.triton.fused_unshuffle_weightedsum` from
`patches/fused_shuffle_quant_wrapper.py:347`. Without `/autokernel` on
`sys.path`, Python cannot resolve the top-level `kernels` package.

Evidence chain (all inside the container):

1. Env var `AUTOKERNEL_FUSED_UNSHUFFLE_WEIGHTEDSUM=1` is set (verified via
   `docker exec ... env`).
2. Wrapper reads the env var at module import
   (`fused_shuffle_quant_wrapper.py:109`) and sets
   `_UNSHUF_WSUM_ENABLED = True`.
3. Wrapper is imported via `fused_shuffle_quant_plugin.register()`.
4. `patch_cutlass_moe_fp4()` installs `_patched_run_cutlass_moe_fp4`.
5. First MoE decode → `fused_unshuffle_weightedsum_moe(...)` is called.
6. `_try_load_unshuffle_weightedsum()` runs
   `from kernels.triton.fused_unshuffle_weightedsum import ...`.
7. Import raises `ModuleNotFoundError: No module named 'kernels'`.
8. `except Exception` logs a WARNING under logger name
   `fused_shuffle_quant_wrapper` (`__name__` of the top-level module as
   loaded via `fused_shuffle_quant_plugin` → `import
   fused_shuffle_quant_wrapper`).
9. vLLM's logger config filters this logger name; the warning never reaches
   stdout.
10. `_unshuf_wsum_fn = None`; `fused_unshuffle_weightedsum_moe` returns
    False; two-pass `ops.shuffle_rows(c3, c_map)` + `(c3.view(...) *
    w).sum(dim=1)` runs silently — no banner, no error.

## Fix

### Primary — launcher PYTHONPATH (`launch_qwen3_fused_norm_fp4.sh:102`)

```diff
-export PYTHONPATH=/autokernel/patches:${PYTHONPATH:-}
+export PYTHONPATH=/autokernel:/autokernel/patches:${PYTHONPATH:-}
```

### Secondary — wrapper belt-and-suspenders sys.path (`patches/fused_shuffle_quant_wrapper.py`, inside `_try_load_unshuffle_weightedsum`)

Added:

```python
import sys
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
```

So the import resolves even if a future launcher forgets `/autokernel` on
PYTHONPATH.

### Tertiary — dual emission (logger + stdout) for all three banners

Load-success, import-failure, and first-call-active banners now
`print(..., flush=True)` in addition to `logger.info/warning`. This
guarantees visibility in `docker logs` regardless of vLLM's logger
filtering.

### Quaternary — eager probe at plugin register time (`patches/fused_shuffle_quant_plugin.py`)

`register()` now calls `wrap.is_unshuffle_weightedsum_available()`
immediately after importing the wrapper, so the
`[T2N-UNSHUF-WSUM] Triton kernel loaded` (or `import failed`) banner fires
at startup instead of first MoE forward. This matches §P1 "eager gate,
visible banner" discipline.

Also prints an explicit probe line:
`[T2N-UNSHUF-WSUM] eager-probe at plugin register: available=<bool>`.

## Expected behaviour post-fix

On relaunch, `docker logs vllm-fused-norm-fp4-qwen3-patched` should show,
before the first request is served:

```
[T2-N] Patched run_cutlass_moe_fp4 via plugin: fused shuffle+quant active
[T2N-UNSHUF-WSUM] Triton kernel loaded (kernels.triton.fused_unshuffle_weightedsum)
[T2N-UNSHUF-WSUM] eager-probe at plugin register: available=True
```

On first MoE forward pass (once traffic hits the server):

```
[T2N-UNSHUF-WSUM] active n_tokens=<M> topk=8 k=2048 (4.24x microbench speedup at Qwen3 shape)
```

`M` will be the decode batch size (e.g. 512 at C=768). `topk=8, k=2048` are
fixed for Qwen3-30B-A3B.

## Parent verification recipe

1. Relaunch: `./launch_qwen3_fused_norm_fp4.sh`.
2. `docker logs vllm-fused-norm-fp4-qwen3-patched 2>&1 | grep -E '\[T2N-UNSHUF-WSUM\]'`
   must show:
   - `Triton kernel loaded` (at startup, from eager probe).
   - `eager-probe at plugin register: available=True`.
   - `active n_tokens=... topk=... k=...` (on first forward pass — send one
     request to trigger).
3. Bench with `bench_qwen3_nvfp4.py`, C=768. Expected +10-15% over
   current 24,923 tok/s banked baseline (the pure-Rank-3 delta before
   `PERSISTENT_BUF` reversion confounds). Microbench projects
   `~1.8 ms per decode step` savings across 48 MoE layers, which at
   Qwen3's step time should be a ~+5-10% e2e throughput gain on top of
   whatever `PERSISTENT_BUF=0` already recovered.
4. If `Triton kernel loaded` appears but `active` never does: the outer
   `_patched_run_cutlass_moe_fp4` is bypassed. Likely cause:
   `VLLM_USE_FLASHINFER_MOE_FP4=1` (launcher sets =0, verify). Deeper
   debug needed — check `docker exec ... env | grep FLASHINFER`.
5. If `import failed` appears: escalated diagnostic — inspect the
   `class=...` and `sys.path[0:3]=...` fields in the WARNING for the real
   cause (likely a Triton/torch version mismatch inside the container).

## Remaining risks

1. **Triton kernel runtime failure.** The import may succeed but the first
   call may raise (e.g. unsupported dtype, grid-size constraint, SM120
   specialization). The `except Exception` at
   `fused_shuffle_quant_wrapper.py:~440` catches and logs
   `[T2N-UNSHUF-WSUM] fused call failed ...` then returns False to trigger
   the two-pass fallback. Banner-visible, benign.
2. **`c3.is_contiguous()` gate.** The fused path requires contiguous c3
   (check at line 802). `_resize_cache(workspace13, (m*topk, k))` should
   always return a contiguous view, but if the workspace has been reshaped
   upstream it could be non-contiguous — fused path silently skipped, no
   banner. Mitigation: consider adding a WARNING on the skip path. Deferred
   pending parent verification.
3. **Logger-filtering may still suppress the WARNING** if a future deploy
   uses a stricter logger config. The stdout `print(..., flush=True)` for
   all three banners is the durable belt-and-suspenders mitigation.
4. **PERSISTENT_BUF interaction.** Launcher currently sets
   `AUTOKERNEL_FUSED_PERSISTENT_BUF=0`. Rank 3 is independent of this flag
   (operates on c3 post-GEMM2, not on scale buffers). No interaction
   expected.

## Files changed

- `launch_qwen3_fused_norm_fp4.sh` — PYTHONPATH fix.
- `patches/fused_shuffle_quant_wrapper.py` — sys.path insert + stdout
  emission for all three banners.
- `patches/fused_shuffle_quant_plugin.py` — eager probe at register().

## KILL_PATTERNS §P1 compliance

All three failure modes of the Rank 3 load path now emit a visible,
class-named log line:

| Condition | Banner | Emission |
|---|---|---|
| Import success | `[T2N-UNSHUF-WSUM] Triton kernel loaded ...` | logger.info + stdout |
| Import failure | `[T2N-UNSHUF-WSUM] import failed class=<cls> msg=<msg> sys.path[0:3]=...` | logger.warning + stdout |
| First-call active | `[T2N-UNSHUF-WSUM] active n_tokens=.. topk=.. k=..` | logger.info + stdout |
| Runtime call failure | `[T2N-UNSHUF-WSUM] fused call failed class=<cls> msg=<msg>` | logger.warning |
| Disabled via env | (no emission — intentional; user opted out) | — |

No silent `None`-returning path remains.
