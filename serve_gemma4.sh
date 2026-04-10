#!/bin/bash
# Adaptive serving for Gemma4 26B NVFP4 on vLLM in Docker
#
# Usage:
#   ./serve_gemma4.sh                    # serving mode (default, 6,615 tok/s peak)
#   ./serve_gemma4.sh interactive        # single-user mode (127 tok/s, lower latency)
#   ./serve_gemma4.sh serving 8192       # serving mode with 8K context
#   ./serve_gemma4.sh serving 4096 8001  # custom port
#
# Requires: docker, nvidia-container-toolkit, vllm-built image, model at /root/models/

set -euo pipefail

MODE="${1:-serving}"
MAX_LEN="${2:-4096}"
PORT="${3:-8000}"

# Configuration
CONTAINER_NAME="vllm-gemma4"
IMAGE="vllm-built"
MODEL_HOST="/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_CONTAINER="/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_NAME="gemma-4-26B-A4B-it-NVFP4"
MEMORY_LIMIT="36g"
LOG_FILE="/tmp/vllm-gemma4.log"

# Remove old container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Common args
COMMON_ARGS=(
    --model "$MODEL_CONTAINER"
    --quantization modelopt
    --max-model-len "$MAX_LEN"
    --port 8000
    --served-model-name "$MODEL_NAME"
)

# Mode-specific args
case "$MODE" in
  interactive|single|low-latency)
    echo "=== Interactive mode ==="
    echo "    torch.compile + piecewise CUDA graphs"
    echo "    Best for: single user, low latency (~127 tok/s)"
    echo "    Startup: ~90s (CUDA graph + torch.compile)"
    MODE_ARGS=()
    ;;
  serving|batch|throughput)
    echo "=== Serving mode ==="
    echo "    No inductor + full CUDA graphs"
    echo "    Best for: many concurrent users (~6,615 tok/s peak)"
    echo "    Startup: ~80s (CUDA graph capture)"
    MODE_ARGS=(
        -cc.mode none
        -cc.cudagraph_mode full
    )
    ;;
  *)
    echo "Usage: $0 [interactive|serving] [max_model_len] [port]"
    echo ""
    echo "Modes:"
    echo "  interactive  torch.compile, best single-request latency (127 tok/s)"
    echo "  serving      no inductor, best batch throughput (6,615 tok/s)"
    echo ""
    echo "Examples:"
    echo "  $0                        # serving mode, 4096 ctx, port 8000"
    echo "  $0 interactive            # interactive mode"
    echo "  $0 serving 8192 8001      # serving, 8K context, port 8001"
    exit 1
    ;;
esac

echo "    Model: $MODEL_NAME"
echo "    Context: $MAX_LEN tokens"
echo "    Port: $PORT"
echo "    Log: $LOG_FILE"
echo ""

# Launch container
docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --memory="$MEMORY_LIMIT" \
    -v "$(dirname "$MODEL_HOST"):/models:ro" \
    -p "${PORT}:8000" \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    "$IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
        "${COMMON_ARGS[@]}" \
        "${MODE_ARGS[@]}" \
    > /dev/null 2>&1

echo "Container started. Waiting for server..."

# Health check with timeout
TIMEOUT=180
for i in $(seq 1 $TIMEOUT); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "=== Server ready in ${i}s ==="
        echo "    API: http://localhost:${PORT}/v1"
        echo "    Model: $MODEL_NAME"
        echo "    Health: http://localhost:${PORT}/health"
        echo ""
        echo "Test:"
        echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
        echo ""
        echo "Logs: docker logs -f $CONTAINER_NAME"

        # Save startup log
        docker logs "$CONTAINER_NAME" > "$LOG_FILE" 2>&1
        KV_TOKENS=$(grep "GPU KV cache size" "$LOG_FILE" 2>/dev/null | grep -oP '\d+(?= tokens)' || echo "?")
        CONCURRENCY=$(grep "Maximum concurrency" "$LOG_FILE" 2>/dev/null | grep -oP '[\d.]+(?=x)' || echo "?")
        echo "KV cache: ${KV_TOKENS} tokens, ~${CONCURRENCY}x concurrency at ${MAX_LEN} ctx"
        exit 0
    fi
    printf "\r    Waiting... %3ds / %ds" "$i" "$TIMEOUT"
    sleep 1
done

echo ""
echo "ERROR: Server failed to start within ${TIMEOUT}s"
echo "Check logs: docker logs $CONTAINER_NAME"
docker logs "$CONTAINER_NAME" 2>&1 | tail -20
exit 1
