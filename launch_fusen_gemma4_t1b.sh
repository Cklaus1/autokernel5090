#!/bin/bash
# Single-GPU FusenCache Gemma4 launcher for T1-B shadow-tensor fix bench.
# Mounts the local /home/cklaus path (not /root/projects) so W5_T1B fixes are picked up.
set -e
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8000}"
NAME="${NAME:-vllm-fusen-t1b-${MODE}}"
GPU_DEVICE="${GPU_DEVICE:-0}"
FUSEN_HOST="/home/cklaus/projects/autokernel/fusen_kv"

docker rm -f "${NAME}" 2>/dev/null || true

# If MODE=baseline, restore pre_t1b backend before launch
if [[ "${MODE}" == "baseline" ]]; then
  cp "${FUSEN_HOST}/backend.py" "${FUSEN_HOST}/backend.py.w5_t1b_snapshot"
  cp "${FUSEN_HOST}/backend.py.pre_t1b" "${FUSEN_HOST}/backend.py"
  BANNER="BASELINE (pre_t1b backend restored)"
  trap "cp ${FUSEN_HOST}/backend.py.w5_t1b_snapshot ${FUSEN_HOST}/backend.py; echo restored w5_t1b backend" EXIT
else
  BANNER="PATCHED (W5_T1B shadow-tensor fix)"
fi

echo "[Fusen T1-B launch] ${BANNER} on GPU ${GPU_DEVICE}"

docker run -d \
  --name "${NAME}" \
  --gpus "device=${GPU_DEVICE}" \
  --shm-size=8g --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICE}" \
  -v /root/models:/models:ro \
  -v "${FUSEN_HOST}":/fusen/fusen_kv:ro \
  -v /home/cklaus/projects/autokernel/kv_cache_gen:/fusen/kv_cache_gen:ro \
  -p "${PORT}:${PORT}" \
  --entrypoint bash \
  vllm-built:latest \
  -c "
set -e
unset NAME
python3 -c '
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f\"[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}\", flush=True)
' 2>&1 || true
export PYTHONPATH=/fusen:\${PYTHONPATH:-}
exec python3 /fusen/fusen_kv/launch_vllm.py \
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
  --quantization modelopt \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --trust-remote-code \
  --port ${PORT} \
  --served-model-name gemma-4-26B-A4B-it-NVFP4 \
  --kv-cache-dtype k4v4b64 \
  --gpu-memory-utilization 0.85 \
  -cc.mode none \
  -cc.cudagraph_mode full --cudagraph-capture-sizes '[1,2,4,8,16,32]'
"

echo "    container ${NAME} starting on port ${PORT}"
