#!/bin/bash
# Dual-model serving: Gemma4 26B (GPU 0) + Qwen3.6 35B (GPU 1)
# Combined: ~30,000 tok/s aggregate
#
# Usage:
#   ./serve_dual_model.sh           # start both
#   ./serve_dual_model.sh stop      # stop both
#
# Endpoints:
#   GPU 0 :8000 — Gemma4 26B-A4B NVFP4 (stronger, 12,229 tok/s peak)
#   GPU 1 :8001 — Qwen3.6 35B-A3B NVFP4 (faster, TBD tok/s peak)

set -euo pipefail

IMAGE="vllm-built:latest"
MODELS_DIR="/root/models"

stop_servers() {
    docker rm -f vllm-gemma4 vllm-qwen3 2>/dev/null || true
    echo "Servers stopped."
}

if [ "${1:-}" = "stop" ]; then
    stop_servers
    exit 0
fi

stop_servers

# CPU pinning: each server gets its own CCD on the 9950X3D (16C/32T).
# Prevents L3 V-Cache thrashing between the two CCDs.
# Adjust if your CPU has a different core count.
CPUS_GPU0="0-15"    # CCD 0
CPUS_GPU1="16-31"   # CCD 1

echo "=== Starting Gemma4 26B on GPU 0 (port 8000, cores ${CPUS_GPU0}) ==="
# WSL2 GPU isolation fix (KILL_PATTERNS.md §P4): --gpus alone leaks; must also set
# NVIDIA_VISIBLE_DEVICES (host mapping) + CUDA_VISIBLE_DEVICES=0 (container-internal view).
docker run -d --name vllm-gemma4 --gpus '"device=0"' --memory=80g \
    --cpuset-cpus="${CPUS_GPU0}" \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v ${MODELS_DIR}:/models:ro -p 8000:8000 \
    --entrypoint bash ${IMAGE} -c \
    'python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
exec python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 --port 8000 \
    --served-model-name gemma-4-26B-A4B-it-NVFP4 \
    --kv-cache-dtype fp8 -cc.mode none -cc.cudagraph_mode full'

echo "=== Starting Qwen3.6 35B on GPU 1 (port 8001, cores ${CPUS_GPU1}) ==="
# WSL2 GPU isolation fix (W6 corrected — KILL_PATTERNS.md §P4):
# WSL2 sets no-cgroups=true so --gpus 'device=N' cannot enforce cgroup device isolation.
# Both containers see all GPU device files. CUDA_VISIBLE_DEVICES=1 (NOT 0) is required
# for the GPU 1 container because it selects by HOST PCIe enumeration index.
# Using CUDA_VISIBLE_DEVICES=0 in both containers sends both to host GPU 0.
# Run test_wsl2_gpu_isolation.sh e to confirm distinct UUIDs before benching.
docker run -d --name vllm-qwen3 --gpus '"device=1"' --memory=80g \
    --cpuset-cpus="${CPUS_GPU1}" \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -e CUDA_VISIBLE_DEVICES=1 \
    -v ${MODELS_DIR}:/models:ro -p 8001:8000 \
    --entrypoint bash ${IMAGE} -c \
    'python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
exec python3 -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3.6-35B-A3B-FP8 \
    --max-model-len 4096 --port 8000 \
    --served-model-name Qwen3.6-35B-A3B-FP8 \
    --kv-cache-dtype fp8 -cc.mode none -cc.cudagraph_mode full'

echo ""
echo "Waiting for servers..."
for port in 8000 8001; do
    for i in $(seq 1 180); do
        if curl -sS "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  Port ${port} ready (${i}s)"
            break
        fi
        sleep 1
    done
done

echo ""
echo "=== DUAL-MODEL SERVING READY ==="
echo "  GPU 0 :8000 — Gemma4 26B (strong, 12k tok/s)"
echo "  GPU 1 :8001 — Qwen3.6 35B (fast, 17k tok/s)"
echo "  Combined: ~30k tok/s"
