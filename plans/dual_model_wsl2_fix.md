# WSL2 GPU Isolation Fix — dual-model & disaggregated servers

**Tag:** W5_25_wsl2_isolation_fix  
**Date:** 2026-04-18  
**Root cause:** KILL_PATTERNS.md §P4 — on WSL2, `--gpus 'device=N'` alone does NOT isolate GPU visibility. The vLLM process appears resident on ALL GPUs per `nvidia-smi --query-compute-apps`, causing cross-GPU contention (proven: T2-N false regression 9,942 → 19,558 tok/s post-cleanup).

---

## Fix applied to both scripts

For every `docker run` that uses `--gpus 'device=N'`, added the two companion env vars:

```
-e NVIDIA_VISIBLE_DEVICES=N   # host-side: nvidia-container-runtime restricts to device N
-e CUDA_VISIBLE_DEVICES=0     # container-internal: CUDA re-indexes the one visible device to 0
```

The combination means: the container's CUDA runtime sees exactly one device (cuda:0), which maps to host GPU N. Cross-GPU bleed is impossible.

---

## serve_dual_model.sh — changed lines

| Container | Old | Added |
|---|---|---|
| `vllm-gemma4` (GPU 0) | `--gpus '"device=0"'` only | `+ -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0` |
| `vllm-qwen3` (GPU 1) | `--gpus '"device=1"'` only | `+ -e NVIDIA_VISIBLE_DEVICES=1 -e CUDA_VISIBLE_DEVICES=0` |

Both containers switch from `--entrypoint python3` to `--entrypoint bash -c '...'` to allow the isolation diagnostic to run first.

**No overlap:** vllm-gemma4 sees only host GPU 0 (uuid=GPU-xxx); vllm-qwen3 sees only host GPU 1 (uuid=GPU-yyy). The GPU-ISOLATION-CHECK lines in `docker logs` will confirm distinct uuids.

---

## serve_disaggregated.sh — changed lines

| Container | Old | Added |
|---|---|---|
| `vllm-disagg-prefill` (GPU 0) | `--gpus '"device=0"'` only | `+ -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0` |
| `vllm-disagg-decode` (GPU 1) | `--gpus '"device=1"'` only | `+ -e NVIDIA_VISIBLE_DEVICES=1 -e CUDA_VISIBLE_DEVICES=0` |

Same entrypoint wrapper change for the diagnostic.

---

## GPU-ISOLATION-CHECK diagnostic

Each container's bash wrapper runs this before `exec python3`:

```bash
python3 -c "import torch; uuid = torch.cuda.get_device_properties(0).uuid; print(f'[GPU-ISOLATION-CHECK] instance=NAME gpu_uuid={uuid}')" || true
```

Verify after launch:
```bash
docker logs vllm-gemma4 | grep GPU-ISOLATION-CHECK
docker logs vllm-qwen3  | grep GPU-ISOLATION-CHECK
# The two uuids MUST differ. If identical, WSL2 is still leaking — tear down + rerun.
```

---

## unset NAME — not applicable

Neither script uses a `$NAME` variable. Container names are hardcoded literals (`vllm-gemma4`, `vllm-qwen3`, etc.), so P3 does not apply here.
