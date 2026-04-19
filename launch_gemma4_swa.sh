#!/bin/bash
# Launch Gemma4 26B NVFP4 with the Triton SWA sparse decode plugin on GPU 1.
#
# Installs a dist-info stub inside the running container so that vLLM's
# load_general_plugins() picks up `swa_gemma4_plugin:register` and monkey-
# patches FlashInferImpl.forward to route sliding-layer decode to the Triton
# sparse kernel at /autokernel/kernels/triton/swa_decode.py.
#
# Usage:
#   MODE=patched  ./launch_gemma4_swa.sh       # SWA plugin active
#   MODE=baseline ./launch_gemma4_swa.sh       # vanilla FlashInfer
#
set -e
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8005}"
NAME="${NAME:-vllm-swa-${MODE}}"
IMAGE="${IMAGE:-vllm-fusencache-gemma4fix:latest}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# Kill any lingering container with the same name.
docker rm -f "${NAME}" 2>/dev/null || true

COMMON_DOCKER_ARGS=(
  --name "${NAME}"
  --gpus 'device=1'
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
  --trust-remote-code
  --port "${PORT}"
  --served-model-name gemma-4-26B-A4B-it-NVFP4
  --gpu-memory-utilization 0.88
  -cc.mode none
  -cc.cudagraph_mode full
)

if [[ "${MODE}" == "patched" ]]; then
  echo "[SWA launch] starting PATCHED server (AUTOKERNEL_SWA_SPARSE=1) on port ${PORT}"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
# Install the plugin dist-info so load_general_plugins() picks us up.
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

# Put our plugin + wrapper + triton kernel dir on PYTHONPATH.
export PYTHONPATH=/autokernel/patches:/autokernel/kernels/triton:${PYTHONPATH:-}
export AUTOKERNEL_SWA_SPARSE=1
export VLLM_PLUGINS=swa_gemma4

exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
else
  echo "[SWA launch] starting BASELINE server (AUTOKERNEL_SWA_SPARSE=0) on port ${PORT}"
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

echo "[SWA launch] container ${NAME} starting. Tail with:"
echo "    docker logs -f ${NAME}"
