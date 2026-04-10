#!/usr/bin/env bash
# serve_gemma4_dp2.sh -- Launch Gemma4 26B NVFP4 with DP=2 on RTX PRO 6000 workstation
#
# Hardware: 2x RTX PRO 6000 Max-Q (96GB each, PCIe — NOT NVLink)
# Model:    Gemma4 26B-A4B-it NVFP4 (modelopt format, BF16 attention)
#
# WHY DP=2 INSTEAD OF TP=2:
#   PCIe AllReduce adds 50-100 µs per step × 60 layers = 3-6 ms per decode token.
#   At C=1 that is a 30-60% latency penalty. DP=2 has ZERO inter-GPU communication.
#   Two independent servers each handle half the requests → same total throughput,
#   lower latency, no NCCL complexity.
#
# Usage:
#   ./serve_gemma4_dp2.sh                          # BF16 KV, 32K ctx, ports 8000/8001
#   ./serve_gemma4_dp2.sh 32768                    # explicit max_model_len
#   ./serve_gemma4_dp2.sh 65536                    # 64K context (96GB allows this)
#   ./serve_gemma4_dp2.sh 32768 bf16 fusen         # GPU 0 normal, GPU 1 with FusenCache
#   ./serve_gemma4_dp2.sh 32768 fusen fusen        # both GPUs with FusenCache
#   ./serve_gemma4_dp2.sh stop                     # stop both servers
#   ./serve_gemma4_dp2.sh restart                  # restart both servers
#   ./serve_gemma4_dp2.sh status                   # health-check both servers
#
# Projection: ~12,000 tok/s aggregate (2x 6,615 tok/s, zero overhead)
# KV capacity per GPU (BF16):   ~300K tokens  (96GB single-GPU vs 192GB TP=2)
# KV capacity per GPU (FusenCache): ~500K tokens per GPU, ~1M total
#
# Requires:
#   - Docker image 'vllm-built' with vLLM (single-GPU, no TP needed)
#   - Both GPUs visible to Docker via --gpus '"device=N"'
#   - Model at /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt
#   - FusenCache: source at /root/projects/autokernel/fusen_kv (for fusen mode)

set -euo pipefail

MAX_LEN="${1:-32768}"
KV_MODE_GPU0="${2:-bf16}"    # bf16 | fp8 | fusen
KV_MODE_GPU1="${3:-${KV_MODE_GPU0}}"

IMAGE="vllm-built"
MODEL_HOST="/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_DIR_HOST="$(dirname "$MODEL_HOST")"
MODEL_NAME="gemma4-nvfp4"
FUSEN_HOST="/root/projects/autokernel/fusen_kv"

PORT_GPU0=8000
PORT_GPU1=8001
CONTAINER_GPU0="vllm-gpu0"
CONTAINER_GPU1="vllm-gpu1"
LOG_GPU0="/tmp/vllm-gpu0.log"
LOG_GPU1="/tmp/vllm-gpu1.log"

# ============================================================
# Sub-commands
# ============================================================
if [[ "${1:-}" == "stop" ]]; then
    echo "Stopping both vLLM servers..."
    docker rm -f "$CONTAINER_GPU0" 2>/dev/null && echo "  Stopped $CONTAINER_GPU0" || echo "  $CONTAINER_GPU0 not running"
    docker rm -f "$CONTAINER_GPU1" 2>/dev/null && echo "  Stopped $CONTAINER_GPU1" || echo "  $CONTAINER_GPU1 not running"
    exit 0
fi

if [[ "${1:-}" == "restart" ]]; then
    echo "Restarting both vLLM servers..."
    docker rm -f "$CONTAINER_GPU0" "$CONTAINER_GPU1" 2>/dev/null || true
    exec "$0" "$MAX_LEN" "$KV_MODE_GPU0" "$KV_MODE_GPU1"
fi

if [[ "${1:-}" == "status" ]]; then
    echo "=== DP=2 Server Status ==="
    for port in $PORT_GPU0 $PORT_GPU1; do
        label="GPU$((port - PORT_GPU0))"
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            models=$(curl -s "http://localhost:${port}/v1/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(','.join(m['id'] for m in d.get('data',[])))" 2>/dev/null || echo "?")
            echo "  $label (port $port): HEALTHY — models: $models"
        else
            echo "  $label (port $port): NOT READY"
        fi
    done
    exit 0
fi

# ============================================================
# KV cache argument builder
# ============================================================
kv_args_for_mode() {
    local mode="$1"
    case "$mode" in
        bf16|default)
            echo ""
            ;;
        fp8)
            echo "--kv-cache-dtype fp8"
            ;;
        fusen)
            echo "--kv-cache-dtype k4v4b64"
            ;;
        *)
            echo "ERROR: Unknown KV mode '$mode'. Use: bf16 | fp8 | fusen" >&2
            exit 1
            ;;
    esac
}

kv_desc() {
    local mode="$1"
    case "$mode" in
        bf16|default) echo "BF16 (~300K tokens)"  ;;
        fp8)          echo "FP8 (~600K tokens)"   ;;
        fusen)        echo "FusenCache k4v4b64 (~500K tokens)" ;;
    esac
}

KV_ARGS_GPU0="$(kv_args_for_mode "$KV_MODE_GPU0")"
KV_ARGS_GPU1="$(kv_args_for_mode "$KV_MODE_GPU1")"

# ============================================================
# Status banner
# ============================================================
echo ""
echo "=== DP=2 Serving Mode ==="
echo "    Strategy: Two independent servers (zero inter-GPU communication)"
echo "    GPU 0: port ${PORT_GPU0} | KV: $(kv_desc "$KV_MODE_GPU0")"
echo "    GPU 1: port ${PORT_GPU1} | KV: $(kv_desc "$KV_MODE_GPU1")"
echo "    Context: ${MAX_LEN} tokens per GPU"
echo "    Projected peak: ~12,000 tok/s aggregate (2 × 6,000)"
echo ""

# Remove old containers
docker rm -f "$CONTAINER_GPU0" "$CONTAINER_GPU1" 2>/dev/null || true

# ============================================================
# Common vLLM server flags
# ============================================================
VLLM_COMMON_ARGS=(
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt
    --quantization modelopt
    --max-model-len "$MAX_LEN"
    --gpu-memory-utilization 0.92
    --max-num-seqs 512
    --served-model-name "$MODEL_NAME"
    --disable-log-requests
    --port 8000
    -cc.mode none
    -cc.cudagraph_mode full
)

# ============================================================
# GPU 0 — port 8000
# ============================================================
echo "Launching GPU 0 (${CONTAINER_GPU0}, port ${PORT_GPU0})..."

if [[ "$KV_MODE_GPU0" == "fusen" ]]; then
    # FusenCache: mount fusen source, run via launch_vllm.py
    docker run -d \
        --name "$CONTAINER_GPU0" \
        --gpus '"device=0"' \
        --memory=90g \
        -v "${MODEL_DIR_HOST}:/models:ro" \
        -v "${FUSEN_HOST}:/fusen/fusen_kv:ro" \
        -p "${PORT_GPU0}:8000" \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e FUSEN_PATH=/fusen \
        "$IMAGE" \
        python3 /fusen/fusen_kv/launch_vllm.py \
            "${VLLM_COMMON_ARGS[@]}" \
            $KV_ARGS_GPU0 \
        > /dev/null 2>&1
else
    docker run -d \
        --name "$CONTAINER_GPU0" \
        --gpus '"device=0"' \
        --memory=90g \
        -v "${MODEL_DIR_HOST}:/models:ro" \
        -p "${PORT_GPU0}:8000" \
        -e CUDA_VISIBLE_DEVICES=0 \
        "$IMAGE" \
        python3 -m vllm.entrypoints.openai.api_server \
            "${VLLM_COMMON_ARGS[@]}" \
            $KV_ARGS_GPU0 \
        > /dev/null 2>&1
fi

echo "  Container started: $CONTAINER_GPU0"

# ============================================================
# GPU 1 — port 8001
# ============================================================
echo "Launching GPU 1 (${CONTAINER_GPU1}, port ${PORT_GPU1})..."

if [[ "$KV_MODE_GPU1" == "fusen" ]]; then
    docker run -d \
        --name "$CONTAINER_GPU1" \
        --gpus '"device=1"' \
        --memory=90g \
        -v "${MODEL_DIR_HOST}:/models:ro" \
        -v "${FUSEN_HOST}:/fusen/fusen_kv:ro" \
        -p "${PORT_GPU1}:8000" \
        -e CUDA_VISIBLE_DEVICES=1 \
        -e FUSEN_PATH=/fusen \
        "$IMAGE" \
        python3 /fusen/fusen_kv/launch_vllm.py \
            "${VLLM_COMMON_ARGS[@]}" \
            $KV_ARGS_GPU1 \
        > /dev/null 2>&1
else
    docker run -d \
        --name "$CONTAINER_GPU1" \
        --gpus '"device=1"' \
        --memory=90g \
        -v "${MODEL_DIR_HOST}:/models:ro" \
        -p "${PORT_GPU1}:8000" \
        -e CUDA_VISIBLE_DEVICES=1 \
        "$IMAGE" \
        python3 -m vllm.entrypoints.openai.api_server \
            "${VLLM_COMMON_ARGS[@]}" \
            $KV_ARGS_GPU1 \
        > /dev/null 2>&1
fi

echo "  Container started: $CONTAINER_GPU1"

# ============================================================
# Health check — wait for both GPUs concurrently
# ============================================================
TIMEOUT=240

echo ""
echo "Waiting for both servers (single-GPU startup ~60-120s)..."

wait_for_server() {
    local port="$1"
    local label="$2"
    local log="$3"
    local container="$4"
    for i in $(seq 1 $TIMEOUT); do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo "  ${label} ready in ${i}s"
            docker logs "$container" > "$log" 2>&1
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: ${label} failed to start within ${TIMEOUT}s"
    docker logs "$container" 2>&1 | tail -20
    return 1
}

# Run both health-checks in parallel (background subshells)
wait_for_server "$PORT_GPU0" "GPU 0 (port ${PORT_GPU0})" "$LOG_GPU0" "$CONTAINER_GPU0" &
PID0=$!
wait_for_server "$PORT_GPU1" "GPU 1 (port ${PORT_GPU1})" "$LOG_GPU1" "$CONTAINER_GPU1" &
PID1=$!

wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

if [[ $STATUS0 -ne 0 || $STATUS1 -ne 0 ]]; then
    echo ""
    echo "ERROR: One or both servers failed to start."
    echo "  GPU 0 logs: docker logs $CONTAINER_GPU0"
    echo "  GPU 1 logs: docker logs $CONTAINER_GPU1"
    exit 1
fi

# ============================================================
# Parse KV token counts from startup logs
# ============================================================
parse_kv_tokens() {
    local log="$1"
    grep "GPU KV cache size" "$log" 2>/dev/null | grep -oP '\d+(?= tokens)' | head -1 || echo "?"
}

KV0=$(parse_kv_tokens "$LOG_GPU0")
KV1=$(parse_kv_tokens "$LOG_GPU1")
KV_TOTAL=$(( ${KV0//[^0-9]/} + ${KV1//[^0-9]/} )) 2>/dev/null || KV_TOTAL="?"

echo ""
echo "========================================================"
echo "  DP=2 — Both servers ready"
echo "========================================================"
echo ""
echo "  GPU 0: http://localhost:${PORT_GPU0}/v1"
echo "    KV:  $(kv_desc "$KV_MODE_GPU0") — ${KV0} tokens allocated"
echo ""
echo "  GPU 1: http://localhost:${PORT_GPU1}/v1"
echo "    KV:  $(kv_desc "$KV_MODE_GPU1") — ${KV1} tokens allocated"
echo ""
echo "  Aggregate KV capacity: ${KV_TOTAL} tokens"
echo "  Context: ${MAX_LEN} tokens per request"
echo "  Model: ${MODEL_NAME}"
echo ""
echo "  Logs:"
echo "    docker logs -f $CONTAINER_GPU0"
echo "    docker logs -f $CONTAINER_GPU1"
echo ""
echo "  Quick test (GPU 0):"
echo "    curl http://localhost:${PORT_GPU0}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
echo ""
echo "  DP=2 benchmark:"
echo "    python bench_dp2.py --quick"
echo "    python bench_dp2.py --full"
echo ""
echo "  Stop both servers:"
echo "    ./serve_gemma4_dp2.sh stop"
echo ""
echo "  fusen_solver routing (8 agents, 4 per GPU):"
echo "    python -m fusen_solver --config fusen_solver_dp2_config.yaml"
echo ""
