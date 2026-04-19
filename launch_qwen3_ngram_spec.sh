#!/bin/bash
# Qwen3-30B-A3B NVFP4 with n-gram GPU speculative decoding (LADE-style).
# §5a PROCEED: zero-draft-model prompt lookup. Projected 1.2-1.5x single-user.
set -e
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8007}"
NAME="${NAME:-vllm-qwen3-ngram-${MODE}}"
GPU_DEVICE="${GPU_DEVICE:-1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"

docker rm -f "${NAME}" 2>/dev/null || true

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

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model /models/Qwen3-30B-A3B-NVFP4
  --quantization modelopt
  --max-model-len 4096
  --max-num-seqs 256
  --trust-remote-code
  --port "${PORT}"
  --served-model-name Qwen3-30B-A3B-NVFP4
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.90
)

if [[ "${MODE}" == "patched" ]]; then
  SPEC_CONFIG='{"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}'
  VLLM_ARGS+=(--speculative-config "${SPEC_CONFIG}")
  echo "[Qwen3 ngram launch] PATCHED (ngram spec, 3 tokens) on GPU ${GPU_DEVICE}"
else
  echo "[Qwen3 ngram launch] BASELINE (no spec) on GPU ${GPU_DEVICE}"
fi

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

docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
  -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"

echo "    container ${NAME} starting on port ${PORT}"
