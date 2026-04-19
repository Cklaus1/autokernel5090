#!/bin/bash
# Launch vLLM Qwen3-30B-A3B-NVFP4 on GPU 1 for the prefix-aware scheduler
# benchmark.  MODE=baseline (stock FIFO) or MODE=patched (plugin active).
#
#   MODE=baseline PORT=8010 ./launch_prefix_aware_sched.sh
#   MODE=patched  PORT=8011 ./launch_prefix_aware_sched.sh
set -e
unset NAME

MODE="${MODE:-baseline}"
PORT="${PORT:-8010}"
NAME="${NAME:-vllm-prefsched-${MODE}}"
GPU="${GPU:-1}"

docker rm -f "${NAME}" 2>/dev/null || true

COMMON_DOCKER_ARGS=(
  --name "${NAME}"
  --gpus "device=${GPU}"
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
  --max-num-seqs 256
  --trust-remote-code
  --port "${PORT}"
  --served-model-name Qwen3-30B-A3B-NVFP4
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.88
  -cc.mode none
  -cc.cudagraph_mode full
  --enable-prefix-caching
)

if [[ "${MODE}" == "patched" ]]; then
  echo "[prefsched] starting PATCHED server with prefix_aware_sched plugin"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
DIST=/usr/local/lib/python3.12/dist-packages/prefix_aware_sched-0.1.dist-info
mkdir -p "$DIST"
cat > "$DIST/METADATA" <<META
Metadata-Version: 2.1
Name: prefix-aware-sched
Version: 0.1
META
cat > "$DIST/entry_points.txt" <<EP
[vllm.general_plugins]
prefix_aware_sched = prefix_aware_scheduler:register
EP
cat > "$DIST/RECORD" <<REC
prefix_aware_sched-0.1.dist-info/METADATA,,
prefix_aware_sched-0.1.dist-info/entry_points.txt,,
prefix_aware_sched-0.1.dist-info/RECORD,,
REC

export PYTHONPATH=/autokernel/patches:${PYTHONPATH:-}
export VLLM_PLUGINS=prefix_aware_sched

exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
else
  echo "[prefsched] starting BASELINE server (FIFO scheduler)"
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
exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
fi

echo "[prefsched] container ${NAME} on port ${PORT}. Logs: docker logs -f ${NAME}"
