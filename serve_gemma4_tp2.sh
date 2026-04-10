#!/usr/bin/env bash
# serve_gemma4_tp2.sh -- Launch Gemma4 26B NVFP4 with TP=2 on RTX PRO 6000 workstation
#
# Hardware: 2x RTX PRO 6000 (96GB each, 192GB total, Blackwell SM120)
# Model:    Gemma4 26B-A4B-it NVFP4 (modelopt format, BF16 attention)
#
# Usage:
#   ./serve_gemma4_tp2.sh                       # TP=2 serving, 32K ctx, port 8000
#   ./serve_gemma4_tp2.sh 32768                 # explicit max_model_len
#   ./serve_gemma4_tp2.sh 65536 8001            # 64K context, custom port
#   ./serve_gemma4_tp2.sh 32768 8000 fusen      # enable FusenCache (--kv-cache-dtype fusen)
#
# Projection: 10,000-12,000 tok/s peak (1.6-1.8x single-GPU 6,615)
# KV capacity: ~480K tokens at BF16 / ~960K with FusenCache FP8+int4
#
# Requires:
#   - Docker image 'vllm-built' with TP-aware vLLM build
#   - Both GPUs visible to Docker (--gpus all)
#   - Model at /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt
#   - NCCL + NVLink or PCIe for inter-GPU communication

set -euo pipefail

MAX_LEN="${1:-32768}"
PORT="${2:-8000}"
KV_MODE="${3:-bf16}"      # bf16 | fp8 | fusen

CONTAINER_NAME="vllm-gemma4-tp2"
IMAGE="vllm-built"
MODEL_HOST="/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_CONTAINER="/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_NAME="gemma-4-26B-A4B-it-NVFP4"
LOG_FILE="/tmp/vllm-gemma4-tp2.log"

# ============================================================
# KV cache configuration
# ============================================================
case "$KV_MODE" in
  bf16|default)
    KV_ARGS=()
    echo "=== KV cache: BF16 (default) ==="
    echo "    ~480K token capacity (192GB * 0.92 - 17GB weights / token_overhead)"
    ;;
  fp8)
    KV_ARGS=(--kv-cache-dtype fp8)
    echo "=== KV cache: FP8 (2x capacity, same throughput on Blackwell) ==="
    echo "    ~960K token capacity"
    ;;
  fusen)
    KV_ARGS=(--kv-cache-dtype fusen)
    echo "=== KV cache: FusenCache (FP8+int4, 2.67x compression) ==="
    echo "    ~1.28M token capacity — effectively 768GB of KV"
    echo "    WARNING: FusenCache must be installed in the Docker image"
    ;;
  *)
    echo "ERROR: Unknown KV mode '$KV_MODE'. Use: bf16 | fp8 | fusen"
    exit 1
    ;;
esac

echo ""
echo "=== TP=2 Serving Mode ==="
echo "    No inductor + full CUDA graphs (best batch throughput)"
echo "    Tensor parallel: 2 GPUs (192GB total VRAM)"
echo "    Context: ${MAX_LEN} tokens"
echo "    Port: ${PORT}"
echo "    Log: ${LOG_FILE}"
echo "    Projected peak: 10,000-12,000 tok/s at C=512+"
echo ""

# Remove old container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# ============================================================
# Launch vLLM with TP=2
#
# Key flags:
#   --tensor-parallel-size 2   -- split model across 2 GPUs via NCCL
#   --max-model-len 32768      -- 32K context (192GB makes this trivial)
#   --gpu-memory-utilization 0.92 -- leave 8% headroom for CUDA graphs + NCCL buffers
#   -cc.mode none              -- disable torch.compile/inductor (2x throughput at batch)
#   -cc.cudagraph_mode full    -- capture full decode graph (eliminates Python overhead)
#   --max-num-seqs 1024        -- allow up to 1024 concurrent sequences (was 256 on TP=1)
#   --disable-log-requests     -- reduce log spam at high concurrency
# ============================================================
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=16g \
    -v "$(dirname "$MODEL_HOST"):/models:ro" \
    -p "${PORT}:8000" \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    -e NCCL_DEBUG=WARN \
    "$IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_CONTAINER" \
        --quantization modelopt \
        --tensor-parallel-size 2 \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization 0.92 \
        --max-num-seqs 1024 \
        --port 8000 \
        --served-model-name "$MODEL_NAME" \
        --disable-log-requests \
        -cc.mode none \
        -cc.cudagraph_mode full \
        "${KV_ARGS[@]}" \
    > /dev/null 2>&1

echo "Container started: $CONTAINER_NAME"
echo "Waiting for server (TP=2 startup ~120-180s for CUDA graph capture)..."

# ============================================================
# Health check with extended timeout for TP=2 graph capture
# ============================================================
TIMEOUT=300
for i in $(seq 1 $TIMEOUT); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "=== Server ready in ${i}s ==="
        echo "    API:    http://localhost:${PORT}/v1"
        echo "    Model:  ${MODEL_NAME}"
        echo "    TP:     2 (${MAX_LEN} token context)"
        echo "    KV:     ${KV_MODE}"
        echo ""

        # Parse startup stats from logs
        docker logs "$CONTAINER_NAME" > "$LOG_FILE" 2>&1
        KV_TOKENS=$(grep "GPU KV cache size" "$LOG_FILE" 2>/dev/null | grep -oP '\d+(?= tokens)' || echo "?")
        CONCURRENCY=$(grep "Maximum concurrency" "$LOG_FILE" 2>/dev/null | grep -oP '[\d.]+(?=x)' || echo "?")
        echo "KV cache: ${KV_TOKENS} tokens, ~${CONCURRENCY}x concurrency at ${MAX_LEN} ctx"
        echo ""
        echo "Quick test:"
        echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
        echo ""
        echo "Throughput benchmark:"
        echo "  python bench_tp2.py --port ${PORT} --max-tokens 200"
        echo ""
        echo "Logs: docker logs -f $CONTAINER_NAME"
        exit 0
    fi
    printf "\r    Waiting... %3ds / %ds" "$i" "$TIMEOUT"
    sleep 1
done

echo ""
echo "ERROR: Server failed to start within ${TIMEOUT}s"
echo "Check logs: docker logs $CONTAINER_NAME"
docker logs "$CONTAINER_NAME" 2>&1 | tail -30
exit 1
