#!/bin/bash
# Launch Gemma4 26B NVFP4 with T2-N fused shuffle+quant plugin.
# Cross-apply of Qwen3 T2-N (banked 19,558 tok/s) to Gemma4 (baseline 6,615 tok/s).
# Projected: +34% → ~8,860 tok/s.
set -e
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8006}"
NAME="${NAME:-vllm-gemma4-t2n-${MODE}}"
GPU_DEVICE="${GPU_DEVICE:-0}"

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
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt
  --quantization modelopt
  --max-model-len 4096
  --max-num-seqs 512
  --trust-remote-code
  --port "${PORT}"
  --served-model-name gemma-4-26B-A4B-it-NVFP4
  --gpu-memory-utilization 0.88
  -cc.mode none
  -cc.cudagraph_mode full
)

if [[ "${MODE}" == "patched" ]]; then
  echo "[Gemma4 T2-N launch] starting patched server with fused shuffle+quant plugin on GPU ${GPU_DEVICE}"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
# Install the T2-N plugin dist-info
mkdir -p /usr/local/lib/python3.12/dist-packages/fused_shuffle_quant_plugin-0.1.dist-info
cat > /usr/local/lib/python3.12/dist-packages/fused_shuffle_quant_plugin-0.1.dist-info/METADATA <<META
Metadata-Version: 2.1
Name: fused-shuffle-quant-plugin
Version: 0.1
META
cat > /usr/local/lib/python3.12/dist-packages/fused_shuffle_quant_plugin-0.1.dist-info/entry_points.txt <<EP
[vllm.general_plugins]
fused_shuffle_quant = fused_shuffle_quant_plugin:register
EP
cat > /usr/local/lib/python3.12/dist-packages/fused_shuffle_quant_plugin-0.1.dist-info/RECORD <<REC
fused_shuffle_quant_plugin-0.1.dist-info/METADATA,,
fused_shuffle_quant_plugin-0.1.dist-info/entry_points.txt,,
fused_shuffle_quant_plugin-0.1.dist-info/RECORD,,
REC

export PYTHONPATH=/autokernel/patches:${PYTHONPATH:-}
export AUTOKERNEL_FUSED_SHUFFLE_QUANT=1
export AUTOKERNEL_FUSED_SHUFFLE_QUANT_SO=/autokernel/workspace/fused_shuffle_quant_sm120a.so
export VLLM_PLUGINS=fused_shuffle_quant
export VLLM_USE_FLASHINFER_MOE_FP4=0

exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
else
  echo "[Gemma4 T2-N launch] starting BASELINE (no plugin) on GPU ${GPU_DEVICE}"
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
export AUTOKERNEL_FUSED_SHUFFLE_QUANT=0
exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
fi

echo "[Gemma4 T2-N launch] container ${NAME} starting on port ${PORT}."
echo "    Tail: docker logs -f ${NAME}"
