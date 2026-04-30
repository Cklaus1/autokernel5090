#!/bin/bash
# launch_qwen3_speculative.sh
# Launches Qwen3-30B-A3B-NVFP4 (target) with Qwen3-1.7B (draft) for
# same-family speculative decoding.  A/B toggle via MODE=baseline|patched.
#
# Tag: W7_qwen3_samefamily_spec_prep
# Ref: kill_audit §16 — projected +98% aggregate (P=0.80)
#
# Pre-requisites:
#   1. AUTOKERNEL_QWEN3_SPEC=1 must be set in the calling shell (env gate).
#   2. /root/models/Qwen3-1.7B must exist (run fetch_qwen3_draft_model.sh first).
#   3. /root/models/Qwen3-30B-A3B-NVFP4 must exist.
#   4. Docker image vllm-fusencache:latest must be available.
#   5. GPU_DEVICE (default 1) must be free of other workloads.
#
# Usage:
#   MODE=baseline AUTOKERNEL_QWEN3_SPEC=1 ./launch_qwen3_speculative.sh   # target only
#   MODE=patched  AUTOKERNEL_QWEN3_SPEC=1 ./launch_qwen3_speculative.sh   # target + draft
#   MODE=patched  AUTOKERNEL_QWEN3_SPEC=1 PORT=8009 ./launch_qwen3_speculative.sh

set -euo pipefail

# ── Env gate ──────────────────────────────────────────────────────────────────
if [[ "${AUTOKERNEL_QWEN3_SPEC:-0}" != "1" ]]; then
  echo "[error] Set AUTOKERNEL_QWEN3_SPEC=1 to confirm intent before launching."
  echo "        This gate prevents accidental spec-decode launches that consume 2× VRAM."
  exit 1
fi

# ── KILL_PATTERNS hygiene: unset NAME before any assignment (P3 pattern) ─────
unset NAME

# ── Config ────────────────────────────────────────────────────────────────────
MODE="${MODE:-patched}"                        # baseline | patched
PORT="${PORT:-8009}"                           # distinct from ngram (8007) + fused (8005)
GPU_DEVICE="${GPU_DEVICE:-1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-4}"        # §5a: 4 optimal for MoE at low concurrency

TARGET_MODEL="/models/Qwen3-30B-A3B-NVFP4"
DRAFT_MODEL="/models/Qwen3-1.7B"

# ── WSL2 triple-isolation (KILL_PATTERNS §WSL2 GPU-leak) ─────────────────────
# GPU_DEVICE=N leaks to sibling GPUs under WSL2. We triple-isolate:
#   1. Docker --gpus device=N  (container sees only GPU N at driver level)
#   2. -e CUDA_VISIBLE_DEVICES=0  (runtime sees index 0 inside the container)
#   3. -e NVIDIA_VISIBLE_DEVICES=N  (container toolkit applies at CDI layer)
# See: plans/wsl2_gpu_isolation_investigation.md

# ── Derived name (assigned AFTER unset NAME above) ───────────────────────────
NAME="vllm-qwen3-samefamily-spec-${MODE}"

# ── Pre-flight: model presence ───────────────────────────────────────────────
echo "[preflight] Checking model paths ..."
if [[ ! -d "/root/models/Qwen3-30B-A3B-NVFP4" ]]; then
  echo "[error] Target model not found: /root/models/Qwen3-30B-A3B-NVFP4"
  exit 1
fi

if [[ "${MODE}" == "patched" && ! -d "/root/models/Qwen3-1.7B" ]]; then
  echo "[error] Draft model not found: /root/models/Qwen3-1.7B"
  echo "        Run ./fetch_qwen3_draft_model.sh first."
  exit 1
fi

# ── Remove stale container ────────────────────────────────────────────────────
docker rm -f "${NAME}" 2>/dev/null || true

# ── Docker args ───────────────────────────────────────────────────────────────
COMMON_DOCKER_ARGS=(
  --name "${NAME}"
  --gpus "device=${GPU_DEVICE}"
  --shm-size=8g --ipc=host
  -e CUDA_VISIBLE_DEVICES=0
  -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICE}"
  -v /root/models:/models:ro
  -v /home/cklaus/projects/autokernel:/autokernel
  -p "${PORT}:${PORT}"
  --entrypoint bash
)

# ── vLLM args (target model) ─────────────────────────────────────────────────
VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${TARGET_MODEL}"
  --quantization modelopt
  --max-model-len 4096
  --max-num-seqs 256
  --trust-remote-code
  --port "${PORT}"
  --served-model-name Qwen3-30B-A3B-NVFP4
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.88
)
# Note: gpu-memory-utilization lowered to 0.88 (vs 0.90 in ngram launcher) to
# leave headroom for the draft model's KV cache.  Qwen3-1.7B draft KV overhead
# is small (~180 MB at typical concurrency) but must not OOM the target.

# ── Spec decode args (patched only) ──────────────────────────────────────────
if [[ "${MODE}" == "patched" ]]; then
  SPEC_CONFIG="{\"method\":\"draft_model\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"
  VLLM_ARGS+=(
    --speculative-model "${DRAFT_MODEL}"
    --speculative-config "${SPEC_CONFIG}"
  )
  echo "[Qwen3 samefamily spec] PATCHED — target=${TARGET_MODEL##*/} draft=Qwen3-1.7B"
  echo "                         num_speculative_tokens=${NUM_SPEC_TOKENS}  port=${PORT}  GPU=${GPU_DEVICE}"
else
  echo "[Qwen3 samefamily spec] BASELINE (no spec) — target=${TARGET_MODEL##*/}"
  echo "                         port=${PORT}  GPU=${GPU_DEVICE}"
fi

# ── UUID isolation check (inline inner script) ───────────────────────────────
# KILL_PATTERNS hygiene: verify only 1 GPU visible inside container,
# and log its UUID so parent can cross-ref with nvidia-smi.
INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
if torch.cuda.device_count() != 1:
    raise RuntimeError(f'"'"'[GPU-ISOLATION-FAIL] expected 1 GPU, got {torch.cuda.device_count()} — WSL2 leak active'"'"')
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1
exec python3 "$@"
'

# ── Launch ────────────────────────────────────────────────────────────────────
docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
  -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"

echo ""
echo "[launch] Container '${NAME}' starting on port ${PORT}."
echo "[launch] Monitor with: docker logs -f ${NAME}"
echo "[launch] Health check:  curl -s http://localhost:${PORT}/health"
echo "[launch] Bench target:  python3 bench_t2h_qwen3_sweep.py --port ${PORT} --mode ${MODE}"
echo ""
echo "[bench reminder] For A/B comparison:"
echo "  MODE=baseline AUTOKERNEL_QWEN3_SPEC=1 PORT=8009 ./launch_qwen3_speculative.sh"
echo "  MODE=patched  AUTOKERNEL_QWEN3_SPEC=1 PORT=8010 ./launch_qwen3_speculative.sh"
echo "  (run one at a time on same GPU to avoid VRAM contention)"
