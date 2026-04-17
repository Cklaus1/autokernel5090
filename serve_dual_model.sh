#!/bin/bash
# Dual-model serving: Gemma4 26B (GPU 0) + Qwen3 30B (GPU 1)
# Combined: ~30,000 tok/s aggregate
#
# Usage:
#   ./serve_dual_model.sh           # start both
#   ./serve_dual_model.sh stop      # stop both
#
# Endpoints:
#   GPU 0 :8000 — Gemma4 26B-A4B NVFP4 (stronger, 12,229 tok/s peak)
#   GPU 1 :8001 — Qwen3 30B-A3B NVFP4 (faster, 17,426 tok/s peak)

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
docker run -d --name vllm-gemma4 --gpus '"device=0"' --memory=80g \
    --cpuset-cpus="${CPUS_GPU0}" \
    -v ${MODELS_DIR}:/models:ro -p 8000:8000 \
    --entrypoint python3 ${IMAGE} \
    -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 --port 8000 \
    --served-model-name gemma-4-26B-A4B-it-NVFP4 \
    --kv-cache-dtype fp8 -cc.mode none -cc.cudagraph_mode full

echo "=== Starting Qwen3 30B on GPU 1 (port 8001, cores ${CPUS_GPU1}) ==="
docker run -d --name vllm-qwen3 --gpus '"device=1"' --memory=80g \
    --cpuset-cpus="${CPUS_GPU1}" \
    -v ${MODELS_DIR}:/models:ro -p 8001:8000 \
    --entrypoint python3 ${IMAGE} \
    -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3-30B-A3B-NVFP4 \
    --quantization modelopt --max-model-len 4096 --port 8000 \
    --served-model-name Qwen3-30B-A3B-NVFP4 \
    --kv-cache-dtype fp8 -cc.mode none -cc.cudagraph_mode full

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
echo "  GPU 1 :8001 — Qwen3 30B (fast, 17k tok/s)"
echo "  Combined: ~30k tok/s"
