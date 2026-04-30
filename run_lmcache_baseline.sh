#!/bin/bash
# Launch baseline vLLM (no LMCache) for smoke bench W1_4e_lmcache_smoke
set -e

docker run -d --name vllm-lmcache-smoke \
  --gpus '"device=0"' \
  --ipc=host \
  --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e CUDA_VISIBLE_DEVICES=0 \
  --entrypoint /bin/bash \
  vllm-fusencache:latest -c '
    python3 -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-8B \
      --port 8400 \
      --host 0.0.0.0 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 4096 \
      --enable-prefix-caching \
      --dtype bfloat16 \
      2>&1
  '

echo "Container vllm-lmcache-smoke started"
echo "Waiting for /health on port 8400..."

for i in $(seq 1 120); do
  if curl -sf http://localhost:8400/health > /dev/null 2>&1; then
    echo "Server ready after ${i}x5s = $((i*5))s"
    break
  fi
  sleep 5
  echo "  ${i}/120 waiting..."
done
