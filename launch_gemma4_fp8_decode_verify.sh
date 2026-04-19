#!/bin/bash
# W6_fp8_decode_verify_prep — Verify whether the banked T2-I +71% claim is real or phantom.
#
# The W5_A1 finding: on SM120, is_device_capability(100) returns False → vLLM selects
# FIDecode metadata → fp8_decode_monkey_patch.py falls back to FlashInfer native every
# request → custom Triton split-K kernel never fires → banked +71% is FlashInfer baseline.
#
# Usage:
#   MODE=patched  ./launch_gemma4_fp8_decode_verify.sh   # patched: monkey-patch active
#   MODE=baseline ./launch_gemma4_fp8_decode_verify.sh   # baseline: no patch, raw FlashInfer
#
# To assert no fallbacks are tolerated (raises RuntimeError instead of silent fallback):
#   AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1 MODE=patched ./launch_gemma4_fp8_decode_verify.sh
#
# Image: vllm-fusencache:latest (has SWA+FP8 plugins; matches T2-N launcher).
#
# KILL_PATTERNS compliance:
#   §P3 — unset NAME at top (prevents parent shell NAME= overriding container name)
#   §P4 — triple GPU isolation: --gpus device=N + NVIDIA_VISIBLE_DEVICES + CUDA_VISIBLE_DEVICES=0
#          UUID-check printed at container start (confirm single-GPU isolation)
#   §P1/P2 — patch emits WARNING-level init banner + per-fallback WARNING + atexit SUMMARY

set -e
unset NAME  # §P3: prevent parent-shell NAME= from poisoning container name

# ── Config ────────────────────────────────────────────────────────────────────
MODE="${MODE:-patched}"
PORT="${PORT:-8009}"
GPU_DEVICE="${GPU_DEVICE:-0}"
NAME="vllm-fp8-decode-verify-${MODE}"
IMAGE="${IMAGE:-vllm-fusencache:latest}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# Kill any lingering container with the same name before starting.
docker rm -f "${NAME}" 2>/dev/null || true

# ── Common docker args ─────────────────────────────────────────────────────────
COMMON_DOCKER_ARGS=(
  --name "${NAME}"
  --gpus "device=${GPU_DEVICE}"                # §P4: host-side GPU isolation
  --shm-size=8g --ipc=host
  -e CUDA_VISIBLE_DEVICES=0                    # §P4: container sees exactly one GPU at index 0
  -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICE}"    # §P4: belt-and-suspenders isolation
  -v /root/models:/models:ro
  -v /home/cklaus/projects/autokernel:/autokernel
  -p "${PORT}:${PORT}"
  --entrypoint bash
)

# ── vLLM server args ──────────────────────────────────────────────────────────
VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt
  --quantization modelopt
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --kv-cache-dtype fp8
  --trust-remote-code
  --port "${PORT}"
  --served-model-name gemma-4-26B-A4B-it-NVFP4
  --gpu-memory-utilization 0.88
  -cc.mode none
  -cc.cudagraph_mode full
)

# ── Launch: PATCHED mode ───────────────────────────────────────────────────────
if [[ "${MODE}" == "patched" ]]; then
  echo "[FP8-decode verify] starting PATCHED server"
  echo "  patch: patches/fp8_decode_monkey_patch.py"
  echo "  kv-cache-dtype: fp8"
  echo "  port: ${PORT}  gpu: ${GPU_DEVICE}  image: ${IMAGE}"
  echo "  AUTOKERNEL_FP8_DECODE=1 (default on)"
  echo "  AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=${AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK:-0}"

  # Import strategy: PYTHONSTARTUP pointing at a tiny loader script written into
  # the container at startup. This fires before vllm is imported, so apply_patch()
  # runs first and intercepts FlashInferImpl.forward at module-import time.
  #
  # Alternative considered: .pth in site-packages (also works). PYTHONSTARTUP
  # is simpler — no dist-info scaffolding needed, and it's explicit in logs.
  #
  # Note on apply_patch() timing: fp8_decode_monkey_patch.py calls apply_patch()
  # at module import (line 472: `if _ENABLED: apply_patch()`), so merely importing
  # the module is sufficient. PYTHONSTARTUP ensures this happens before any vllm
  # import, so FlashInferImpl.forward is patched for the entire process lifetime.

  INNER_SCRIPT='
set -e
unset NAME

# §P4 UUID check — confirm single GPU visible; same UUID in both containers = isolation failure
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
assert n == 1, f'"'"'ISOLATION FAIL: expected 1 GPU, got {n}'"'"'
" 2>&1 || true

# Write the PYTHONSTARTUP loader that imports fp8_decode_monkey_patch before vllm.
# apply_patch() auto-fires on import (module-level: `if _ENABLED: apply_patch()`).
cat > /tmp/autokernel_fp8_decode_startup.py <<'"'"'STARTUP_EOF'"'"'
import sys
import os
# Ensure patches/ directory is on sys.path so the import resolves
_patches_dir = "/autokernel/patches"
if _patches_dir not in sys.path:
    sys.path.insert(0, _patches_dir)
# Importing the module triggers apply_patch() at module level (line 472).
# This must happen before vllm imports FlashInferImpl so the monkey-patch
# is in place when the backend registers itself.
try:
    import fp8_decode_monkey_patch  # noqa: F401 — side-effects are the point
except Exception as _e:
    import warnings
    warnings.warn(f"[FP8-decode-startup] Failed to import fp8_decode_monkey_patch: {_e}")
STARTUP_EOF

export PYTHONSTARTUP=/tmp/autokernel_fp8_decode_startup.py
export PYTHONPATH=/autokernel/patches:/autokernel/kernels/triton:${PYTHONPATH:-}
export AUTOKERNEL_FP8_DECODE=1
export AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=${AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK:-0}

# Disable other autokernel plugins so their banners don'"'"'t confuse log grepping
export AUTOKERNEL_SWA_SPARSE=0
export AUTOKERNEL_FUSED_SHUFFLE_QUANT=0
unset VLLM_PLUGINS

exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"

# ── Launch: BASELINE mode ─────────────────────────────────────────────────────
else
  echo "[FP8-decode verify] starting BASELINE server (no monkey-patch)"
  echo "  kv-cache-dtype: fp8  port: ${PORT}  gpu: ${GPU_DEVICE}  image: ${IMAGE}"

  INNER_SCRIPT='
set -e
unset NAME

# §P4 UUID check
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
assert n == 1, f'"'"'ISOLATION FAIL: expected 1 GPU, got {n}'"'"'
" 2>&1 || true

# Explicitly disable the patch env gate so even if fp8_decode_monkey_patch
# somehow gets imported (e.g., from a .pth leftover), apply_patch() is a no-op.
export AUTOKERNEL_FP8_DECODE=0
export AUTOKERNEL_SWA_SPARSE=0
export AUTOKERNEL_FUSED_SHUFFLE_QUANT=0
unset PYTHONSTARTUP
unset VLLM_PLUGINS

exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
fi

echo ""
echo "[FP8-decode verify] container '${NAME}' starting on port ${PORT}."
echo ""
echo "  Tail logs:      docker logs -f ${NAME}"
echo "  Watch init:     docker logs ${NAME} 2>&1 | grep '\\[FP8-decode\\]'"
echo "  Health check:   curl -s http://localhost:${PORT}/health"
echo ""
echo "  After bench, trigger atexit banner:"
echo "    docker kill -s SIGTERM ${NAME}"
echo "  Then check:     docker logs ${NAME} 2>&1 | grep 'FALLBACK SUMMARY'"
echo ""
echo "  Interpretation:"
echo "    FALLBACK SUMMARY: fallback_fire_count=0  →  kernel fired; +71% may be real."
echo "    FALLBACK SUMMARY: N time(s) [N≈requests]  →  FIDecode path; +71% is FlashInfer baseline."
echo "    0 < N < requests  →  partial; investigate which metadata types triggered fallback."
