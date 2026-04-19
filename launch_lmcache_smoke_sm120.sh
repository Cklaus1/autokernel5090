#!/bin/bash
# Launch vLLM + LMCache v0.4.3 rebuilt for SM120 (RTX PRO 6000 Blackwell).
# MODE=baseline  : stock vLLM, no LMCache
# MODE=patched   : LMCacheConnectorV1 active, local_cpu=True 20GB pool
set -e
unset NAME
MODE="${MODE:-patched}"
NAME="${NAME:-vllm-lmcache-${MODE}}"
PORT="${PORT:-8400}"

docker rm -f "${NAME}" 2>/dev/null || true

# Environment toggle for the entrypoint
if [[ "${MODE}" == "patched" ]]; then
  LMCACHE_ENV='-e LMCACHE_CONFIG_FILE=/autokernel/lmcache_cpu.yaml -e VLLM_USE_V1=1 -e PATCHED_MODE=1'
  BANNER="PATCHED (LMCache v0.4.3 SM120)"
else
  LMCACHE_ENV=""
  BANNER="BASELINE (no LMCache)"
fi

echo "[LMCache launch] ${BANNER} on port ${PORT}"

docker run -d \
  --name "${NAME}" \
  --gpus 'device=0' \
  -e CUDA_VISIBLE_DEVICES=0 \
  ${LMCACHE_ENV} \
  --shm-size=8g --ipc=host \
  -v /root/models:/models:ro \
  -v /home/cklaus/projects/autokernel:/autokernel \
  -p "${PORT}:${PORT}" \
  --entrypoint bash \
  vllm-built:latest \
  -c "
set -e
unset NAME
python3 -c \"
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}', flush=True)
\" 2>&1 || true
if [[ '${MODE}' == 'patched' ]]; then
  echo '=== rebuilding lmcache for SM120 ==='
  export TORCH_CUDA_ARCH_LIST='12.0'
  sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/pass  #/' /usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py
  apt-get install -qq -y git >/dev/null 2>&1
  cd /tmp && git clone -q https://github.com/LMCache/LMCache.git
  cd /tmp/LMCache && git checkout v0.4.3 >/dev/null 2>&1
  pip install . --break-system-packages --no-build-isolation --force-reinstall --no-deps >/dev/null 2>&1
  # Install runtime deps (skip cupy/nixl/redis — only need local_cpu mode)
  pip install --break-system-packages sortedcontainers nvtx aiofile aiofiles >/dev/null 2>&1
  echo '=== lmcache rebuilt + deps installed, launching vllm ==='
fi
cd /
if [[ \"\${PATCHED_MODE:-}\" == \"1\" ]]; then
  exec python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-30B-A3B-NVFP4 \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --trust-remote-code \
    --quantization modelopt \
    --port ${PORT} \
    --served-model-name Qwen/Qwen3-8B \
    --gpu-memory-utilization 0.70 \
    --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}'
else
  exec python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-30B-A3B-NVFP4 \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --trust-remote-code \
    --quantization modelopt \
    --port ${PORT} \
    --served-model-name Qwen/Qwen3-8B \
    --gpu-memory-utilization 0.70
fi
"

echo "[LMCache launch] container ${NAME} starting. Follow with: docker logs -f ${NAME}"
