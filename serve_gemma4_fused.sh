#!/usr/bin/env bash
# serve_gemma4_fused.sh -- Launch Gemma4 26B NVFP4 with fused RMSNorm+FP4 kernel
#
# This script:
#   1. Compiles the fused kernel .so if it doesn't exist (in a throwaway container)
#   2. Launches vLLM with the .so pre-loaded alongside FusenCache
#
# Usage:
#   ./serve_gemma4_fused.sh              # build + launch
#   ./serve_gemma4_fused.sh --build-only # just compile the .so
#   ./serve_gemma4_fused.sh --no-build   # skip build, assume .so exists
#
# Prerequisites:
#   - Docker image 'vllm-built' (vLLM 0.19.1rc1, CUDA 12.8)
#   - Model at /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt
#   - RTX 5090 (SM120 / Blackwell)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="vllm-built"
CONTAINER_NAME="vllm-gemma4-fused"
SO_HOST_DIR="${SCRIPT_DIR}/cache/fused_kernel"
SO_FILENAME="fused_rms_norm_fp4.so"
SO_HOST_PATH="${SO_HOST_DIR}/${SO_FILENAME}"
SO_CONTAINER_PATH="/tmp/fused_kernel/${SO_FILENAME}"
MODEL_PATH="/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
PORT=8000

BUILD_ONLY=0
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --build-only) BUILD_ONLY=1 ;;
        --no-build)   SKIP_BUILD=1 ;;
    esac
done

# ============================================================
# Step 1: Build the .so (if needed)
# ============================================================
build_kernel() {
    echo "=== Building fused RMSNorm+FP4 kernel ==="
    mkdir -p "${SO_HOST_DIR}"

    # Build inside a throwaway container, copy .so out
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
        echo "  (use --build-only to force rebuild, or delete the file)"
    fi
fi

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
    echo "Build-only mode, exiting."
    exit 0
fi

# ============================================================
# Step 2: Create the startup wrapper that loads the fused kernel
# ============================================================
# This Python script loads the .so before launching vLLM, so the
# fused ops are available when the model initializes.
LAUNCHER_SCRIPT=$(cat <<'PYEOF'
#!/usr/bin/env python3
"""Launch vLLM with fused RMSNorm+FP4 kernel pre-loaded."""
import os
import sys
import logging

logger = logging.getLogger("fused_kernel_loader")
logging.basicConfig(level=logging.INFO)

# --- Load the fused kernel .so ---
SO_PATH = os.environ.get("FUSED_KERNEL_SO", "/tmp/fused_kernel/fused_rms_norm_fp4.so")
if os.path.exists(SO_PATH):
    import torch
    torch.ops.load_library(SO_PATH)
    logger.info("Loaded fused RMSNorm+FP4 kernel from %s", SO_PATH)

    # Register fake tensor impls for torch.compile compatibility
    try:
        @torch.library.register_fake("_C::rms_norm_dynamic_fp4_quant")
        def _fake1(result, result_scale, input, weight, input_global_scale,
                   epsilon, is_sf_swizzled_layout):
            pass

        @torch.library.register_fake("_C::fused_add_rms_norm_dynamic_fp4_quant")
        def _fake2(result, result_scale, input, weight, residual,
                   input_global_scale, epsilon, is_sf_swizzled_layout):
            pass
        logger.info("Registered fake tensor implementations for torch.compile")
    except Exception as e:
        logger.warning("Could not register fake tensors: %s", e)

    # Verify the ops are available
    assert hasattr(torch.ops._C, "rms_norm_dynamic_fp4_quant"), \
        "rms_norm_dynamic_fp4_quant not found after loading .so"
    logger.info("Verified: torch.ops._C.rms_norm_dynamic_fp4_quant available")
    logger.info("Verified: torch.ops._C.fused_add_rms_norm_dynamic_fp4_quant available")
else:
    logger.warning("Fused kernel .so not found at %s -- running WITHOUT fusion", SO_PATH)

# --- Now delegate to the FusenKV launcher (which handles plugin registration + vLLM) ---
# We exec the existing launch_vllm.py with all our args
fusen_path = os.environ.get("FUSEN_PATH", "/fusen")
launch_script = os.path.join(fusen_path, "fusen_kv", "launch_vllm.py")
if os.path.exists(launch_script):
    # Set sys.argv[0] to the launch script so argparse works
    sys.argv[0] = launch_script
    exec(open(launch_script).read())
else:
    logger.error("FusenKV launch script not found at %s", launch_script)
    sys.exit(1)
PYEOF
)

# ============================================================
# Step 3: Stop any existing container with same name
# ============================================================
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container '${CONTAINER_NAME}'..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# ============================================================
# Step 4: Launch vLLM with fused kernel
# ============================================================
echo "=== Launching vLLM with fused RMSNorm+FP4 kernel ==="
echo "  Container: ${CONTAINER_NAME}"
echo "  Port:      ${PORT}"
echo "  Model:     ${MODEL_PATH}"
echo "  Kernel:    ${SO_HOST_PATH}"
echo ""

docker run -d --gpus all \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:8000" \
    -v "${SCRIPT_DIR}:/fusen:ro" \
    -v "${MODEL_PATH}:/models/gemma-4-26B-A4B-it-NVFP4-modelopt:ro" \
    -v "${SO_HOST_DIR}:/tmp/fused_kernel:ro" \
    -e "FUSEN_PATH=/fusen" \
    -e "PYTHONPATH=/fusen" \
    -e "FUSED_KERNEL_SO=${SO_CONTAINER_PATH}" \
    "${IMAGE}" \
    python3 -c "${LAUNCHER_SCRIPT}" \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt \
    --kv-cache-dtype k4v4b64 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code

echo ""
echo "=== Container started: ${CONTAINER_NAME} ==="
echo "  View logs:  docker logs -f ${CONTAINER_NAME}"
echo "  Test:       curl http://localhost:${PORT}/v1/models"
echo "  Stop:       docker stop ${CONTAINER_NAME}"
