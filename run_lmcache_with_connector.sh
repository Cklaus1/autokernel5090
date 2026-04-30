#!/bin/bash
# Launch vLLM with LMCacheConnectorV1 for smoke bench W1_4e_lmcache_smoke
set -e

docker run -d --name vllm-lmcache-smoke \
  --gpus '"device=0"' \
  --ipc=host \
  --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /home/cklaus/projects/autokernel/lmcache_cpu.yaml:/tmp/lmcache_cpu.yaml:ro \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e LMCACHE_CONFIG_FILE=/tmp/lmcache_cpu.yaml \
  -e LMCACHE_CHUNK_SIZE=256 \
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=20 \
  --entrypoint /bin/bash \
  vllm-fusencache:latest -c '
    pip install --break-system-packages --quiet lmcache==0.4.3 &&
    python3 -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-8B \
      --port 8400 \
      --host 0.0.0.0 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 4096 \
      --enable-prefix-caching \
      --dtype bfloat16 \
      --kv-transfer-config '"'"'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'"'"' \
      2>&1
  '

echo "Container vllm-lmcache-smoke (with LMCache) started"
echo "Waiting for /health on port 8400..."

for i in $(seq 1 120); do
  if curl -sf http://localhost:8400/health > /dev/null 2>&1; then
    echo "Server ready after ${i}x5s = $((i*5))s"
    break
  fi
  sleep 5
  echo "  ${i}/120 waiting..."
done
