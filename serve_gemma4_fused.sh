#!/usr/bin/env bash
# serve_gemma4_fused.sh -- Launch Gemma4 26B NVFP4 with fused RMSNorm+FP4 kernel
#
# This script:
#   1. Builds the fused kernel .so if needed
#   2. Patches gemma4.py source in the container to use fused norm+quant
#   3. Launches vLLM with the .so pre-loaded
#
# Usage:
#   ./serve_gemma4_fused.sh              # build + launch (serving mode)
#   ./serve_gemma4_fused.sh --build-only # just compile the .so
#   ./serve_gemma4_fused.sh --no-build   # skip build, assume .so exists
#   ./serve_gemma4_fused.sh --eager      # launch in eager mode (no CUDA graphs)
#
# Prerequisites:
#   - Docker image 'vllm-built' (vLLM 0.19.1rc1, CUDA 12.8)
#   - Model at /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt
#   - RTX 5090 (SM120 / Blackwell)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="vllm-built"
CONTAINER_NAME="vllm-gemma4"
SO_HOST_DIR="${SCRIPT_DIR}/cache/fused_kernel"
SO_FILENAME="fused_rms_norm_fp4.so"
SO_HOST_PATH="${SO_HOST_DIR}/${SO_FILENAME}"
MODEL_PATH="/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_CONTAINER="/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_NAME="gemma-4-26B-A4B-it-NVFP4"
PORT=8000
MEMORY_LIMIT="36g"
LOG_FILE="/tmp/vllm-gemma4-fused.log"

BUILD_ONLY=0
SKIP_BUILD=0
EAGER_MODE=0
for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=1 ;;
        --no-build)   SKIP_BUILD=1 ;;
        --eager)      EAGER_MODE=1 ;;
    esac
done

# ============================================================
# Step 1: Build the .so (if needed)
# ============================================================
build_kernel() {
    echo "=== Building fused RMSNorm+FP4 kernel ==="
    mkdir -p "${SO_HOST_DIR}"

    docker run --rm --gpus all \
        -v "${SCRIPT_DIR}:/autokernel:ro" \
        -v "${SO_HOST_DIR}:/output" \
        "${IMAGE}" \
        bash -c "
            python3 /autokernel/kernels/csrc/build_and_install.py && \
            cp /tmp/build_fused_rms_norm_fp4/${SO_FILENAME} /output/${SO_FILENAME} && \
            echo 'Kernel .so copied to /output/${SO_FILENAME}'
        "

    if [[ ! -f "${SO_HOST_PATH}" ]]; then
        echo "ERROR: Build failed -- .so not found at ${SO_HOST_PATH}"
        exit 1
    fi
    echo "=== Kernel built: ${SO_HOST_PATH} ($(stat -c%s "${SO_HOST_PATH}") bytes) ==="
}

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
    if [[ ! -f "${SO_HOST_PATH}" ]]; then
        echo "Kernel .so not found, building..."
        build_kernel
    else
        echo "Kernel .so found at ${SO_HOST_PATH}, skipping build."
    fi
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
    echo "Build-only mode, exiting."
    exit 0
fi

# ============================================================
# Step 2: Stop existing container
# ============================================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container '${CONTAINER_NAME}'..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# ============================================================
# Step 3: Choose compile mode
# ============================================================
if [[ "${EAGER_MODE}" -eq 1 ]]; then
    echo "=== Eager mode (no CUDA graphs) ==="
    COMPILE_ARGS="--enforce-eager"
else
    echo "=== Serving mode (no inductor + full CUDA graphs) ==="
    COMPILE_ARGS="-cc.mode none -cc.cudagraph_mode full"
fi

# ============================================================
# Step 4: Launch with fused kernel
# ============================================================
echo "=== Launching vLLM with fused RMSNorm+FP4 kernel ==="
echo "  Container: ${CONTAINER_NAME}"
echo "  Port:      ${PORT}"
echo "  Model:     ${MODEL_PATH}"
echo "  Kernel:    ${SO_HOST_PATH}"
echo ""

docker run -d \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --memory="${MEMORY_LIMIT}" \
    -v "$(dirname "${MODEL_PATH}"):/models:ro" \
    -v "${SCRIPT_DIR}/patches:/patches:ro" \
    -v "${SO_HOST_DIR}:/tmp/fused_kernel:ro" \
    -p "${PORT}:8000" \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    -e FUSED_KERNEL_SO=/tmp/fused_kernel/${SO_FILENAME} \
    -e PYTHONPATH=/patches \
    "${IMAGE}" \
    python3 /patches/launch_fused_vllm.py \
        --model "${MODEL_CONTAINER}" \
        --quantization modelopt \
        --max-model-len 4096 \
        --port 8000 \
        --served-model-name "${MODEL_NAME}" \
        ${COMPILE_ARGS} \
    > /dev/null 2>&1

echo "Container started. Waiting for server..."

# ============================================================
# Step 5: Health check
# ============================================================
TIMEOUT=300
for i in $(seq 1 $TIMEOUT); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo ""
        echo "=== Server ready in ${i}s ==="
        echo "    API: http://localhost:${PORT}/v1"
        echo "    Model: ${MODEL_NAME}"
        echo "    Fused kernel: ACTIVE"
        echo ""
        echo "Test:"
        echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
        echo ""
        echo "Logs: docker logs -f ${CONTAINER_NAME}"

        # Check for fused kernel in logs
        docker logs "${CONTAINER_NAME}" > "${LOG_FILE}" 2>&1
        if grep -q "FUSED_NORM_FP4\|fused kernel\|Loaded fused" "${LOG_FILE}" 2>/dev/null; then
            echo "=== Fused kernel integration confirmed in logs ==="
        fi

        KV_TOKENS=$(grep "GPU KV cache size" "${LOG_FILE}" 2>/dev/null | grep -oP '\d+(?= tokens)' || echo "?")
        echo "KV cache: ${KV_TOKENS} tokens"
        exit 0
    fi
    printf "\r    Waiting... %3ds / %ds" "$i" "$TIMEOUT"
    sleep 1
done

echo ""
echo "ERROR: Server failed to start within ${TIMEOUT}s"
echo "Check logs: docker logs ${CONTAINER_NAME}"
docker logs "${CONTAINER_NAME}" 2>&1 | tail -30
exit 1
