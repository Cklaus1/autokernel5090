#!/bin/bash
# Launch Gemma4 26B NVFP4 with the Triton SWA sparse decode plugin + FP8 KV
# on GPU 0 (Gate #4 bench harness).
#
# This mirrors launch_gemma4_swa.sh but:
#   - Pins to GPU 0 (GPU 1 is busy with other agents).
#   - Adds --kv-cache-dtype fp8 so the SWA kernel runs its FP8 dequant path.
#
# Usage:
#   MODE=patched  ./launch_gemma4_swa_fp8.sh      # SWA plugin + FP8 KV
#   MODE=baseline ./launch_gemma4_swa_fp8.sh      # vanilla FlashInfer + FP8 KV
#
set -e
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8005}"
NAME="${NAME:-vllm-swa-fp8-${MODE}}"
IMAGE="${IMAGE:-vllm-fusencache-gemma4fix:latest}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

# Kill any lingering container with the same name.
docker rm -f "${NAME}" 2>/dev/null || true

COMMON_DOCKER_ARGS=(
  --name "${NAME}"
  --gpus 'device=0'
  --shm-size=8g --ipc=host
  -e CUDA_VISIBLE_DEVICES=0
  -v /root/models:/models:ro
  -v /home/cklaus/projects/autokernel:/autokernel
  -p "${PORT}:${PORT}"
  --entrypoint bash
)

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt
  --quantization modelopt
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --trust-remote-code
  --port "${PORT}"
  --served-model-name gemma-4-26B-A4B-it-NVFP4
  --gpu-memory-utilization 0.88
  -cc.mode none
  -cc.cudagraph_mode full
)

if [[ "${MODE}" == "patched" ]]; then
  echo "[SWA-FP8 launch] starting PATCHED server (AUTOKERNEL_SWA_SPARSE=1,"
  echo "                 AUTOKERNEL_SWA_SPARSE_FP8=1, kv_cache_dtype=${KV_CACHE_DTYPE}) on port ${PORT}"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
mkdir -p /usr/local/lib/python3.12/dist-packages/swa_gemma4_plugin-0.1.dist-info
cat > /usr/local/lib/python3.12/dist-packages/swa_gemma4_plugin-0.1.dist-info/METADATA <<META
Metadata-Version: 2.1
Name: swa-gemma4-plugin
Version: 0.1
META
cat > /usr/local/lib/python3.12/dist-packages/swa_gemma4_plugin-0.1.dist-info/entry_points.txt <<EP
[vllm.general_plugins]
swa_gemma4 = swa_gemma4_plugin:register
EP
cat > /usr/local/lib/python3.12/dist-packages/swa_gemma4_plugin-0.1.dist-info/RECORD <<REC
swa_gemma4_plugin-0.1.dist-info/METADATA,,
swa_gemma4_plugin-0.1.dist-info/entry_points.txt,,
swa_gemma4_plugin-0.1.dist-info/RECORD,,
REC

export PYTHONPATH=/autokernel/patches:/autokernel/kernels/triton:${PYTHONPATH:-}
export AUTOKERNEL_SWA_SPARSE=1
export AUTOKERNEL_SWA_SPARSE_FP8=1
export VLLM_PLUGINS=swa_gemma4

exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
else
  echo "[SWA-FP8 launch] starting BASELINE server (AUTOKERNEL_SWA_SPARSE=0, kv_cache_dtype=${KV_CACHE_DTYPE}) on port ${PORT}"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
unset VLLM_PLUGINS
export AUTOKERNEL_SWA_SPARSE=0
exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
fi

echo "[SWA-FP8 launch] container ${NAME} starting. Tail with:"
echo "    docker logs -f ${NAME}"
