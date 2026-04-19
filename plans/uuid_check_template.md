# GPU-ISOLATION-CHECK — Canonical Snippet and Rationale

**Tag:** W5_T44_uuid_check_line

## Problem

WSL2 `--gpus 'device=N'` does not guarantee hard GPU isolation. A container
launched with `--gpus 'device=1'` can still see (and report) processes running
on the other GPU if `/dev/nvidia*` device files are world-readable. This caused
a false T2-N regression alarm: a peer container's PID appeared on both GPU UUIDs,
making it look like the patched container had leaked to GPU 0.

Best-effort detection: at container startup, assert the container sees exactly
**one** GPU and record its UUID. If `n_visible != 1`, the leaked state is visible
in logs before any model load begins.

## Canonical Check Snippet

Insert this **inside the INNER_SCRIPT heredoc**, after `set -e` / `unset NAME`
lines, before any `exec python3` call:

```bash
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}', flush=True)
" 2>&1 || true
```

### Notes on quoting inside single-quoted heredocs

When the INNER_SCRIPT is assigned inside single quotes (`INNER_SCRIPT='...'`),
f-string braces and single quotes inside the python3 -c body require the
`'"'"'` escape sequence for single quotes:

```bash
INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
exec python3 "$@"
'
```

When the script is passed as an inline `-c "..."` double-quoted string, use
escaped double quotes for the inner python3 -c:

```bash
-c "
set -e
unset NAME
python3 -c \"
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}', flush=True)
\" 2>&1 || true
..."
```

## Post-launch check for scripts without INNER_SCRIPT

For launchers that call `docker run ... python3` directly (no inner bash -c
heredoc), add after the `docker run` block:

```bash
sleep 10 && docker logs "${CONTAINER_NAME}" 2>&1 | grep "GPU-ISOLATION-CHECK" || true
```

## How to interpret the log line

```
[GPU-ISOLATION-CHECK] visible=1 uuid=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

- `visible=1` — correct: container sees exactly one GPU.
- `visible=2` — WSL2 leak active; both GPUs are visible. Stop and relaunch with
  explicit `-e NVIDIA_VISIBLE_DEVICES=N -e CUDA_VISIBLE_DEVICES=0`.
- Line absent — torch not importable at startup (pre-install container); not
  actionable but not a regression signal either.

## Scripts patched (W5_T44)

| Script | Pattern | Inner scripts patched |
|--------|---------|----------------------|
| `launch_gemma4_swa.sh` | INNER_SCRIPT var | 2 (patched + baseline) |
| `launch_gemma4_swa_fp8.sh` | INNER_SCRIPT var | 2 (patched + baseline) |
| `launch_gemma4_t2n.sh` | INNER_SCRIPT var | 2 (patched + baseline) |
| `launch_gemma4_lmcache_hierarchy.sh` | INNER_SCRIPT var | 3 (baseline + t1_only + patched) |
| `launch_lmcache_smoke_sm120.sh` | inline `-c "..."` | 1 (single block, 2 exec paths) |
| `launch_prefix_aware_sched.sh` | INNER_SCRIPT var | 2 (patched + baseline) |
| `launch_qwen3_fused_norm_fp4.sh` | INNER_SCRIPT var | 2 (patched + baseline) |
| `launch_qwen3_fused_t2n.sh` | INNER_SCRIPT var | 2 (patched + baseline) |
| `launch_qwen3_ngram_spec.sh` | INNER_SCRIPT var | 1 (shared, mode-agnostic) |
| `serve_gemma4.sh` | direct docker run | post-launch `sleep 10 && docker logs` |
| `serve_dual_model.sh` | inline `-c '...'` | 2 (gemma4 + qwen3; upgraded from old form) |
| `serve_disaggregated.sh` | inline `-c '...'` | 2 (prefill + decode; upgraded from old form) |

`_archive/` contents were NOT touched.
