#!/bin/bash
# Launch Qwen3-30B-A3B NVFP4 with T2-N fused shuffle+quant plugin on GPU 1.
#
# Installs a dist-info stub inside the running container that declares the
# patches/fused_shuffle_quant_plugin.py module as a vllm.general_plugins
# entry point. vLLM's load_general_plugins() then calls register() before
# the model is loaded, which monkey-patches run_cutlass_moe_fp4.
#
# Baseline (no patch) and patched variant share this script; pass
# MODE=baseline or MODE=patched.
set -e
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8003}"
NAME="${NAME:-vllm-t2n-${MODE}}"

# Clean any previous container with the same name
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
  --model /models/Qwen3-30B-A3B-NVFP4
  --quantization modelopt
  --max-model-len 4096
  --max-num-seqs 512
  --trust-remote-code
  --port "${PORT}"
  --served-model-name Qwen3-30B-A3B-NVFP4
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.92
  -cc.mode none
  -cc.cudagraph_mode full
)

if [[ "${MODE}" == "patched" ]]; then
  echo "[T2-N launch] starting patched server with fused shuffle+quant plugin"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
# Install the plugin dist-info so load_general_plugins() picks us up
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

# Put our plugin + wrapper module on PYTHONPATH
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
  echo "[T2-N launch] starting baseline server (no patch)"
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

echo "[T2-N launch] container ${NAME} starting on port ${PORT}. Tail with:"
echo "    docker logs -f ${NAME}"
