#!/bin/bash
# Build vLLM from source with community Gemma4/NVFP4 patches
# Uses Docker with memory limit to prevent OOM killing WSL
set -e

CONTAINER_NAME="vllm-build"
IMAGE="nvidia/cuda:12.8.0-devel-ubuntu24.04"
MEMORY_LIMIT="36g"

echo "=== Starting vLLM build container (memory limit: ${MEMORY_LIMIT}) ==="

# Remove old container if exists
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Start container with GPU + memory limit
docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --memory=${MEMORY_LIMIT} \
    --memory-swap=${MEMORY_LIMIT} \
    -v /root/projects/autokernel/patches:/patches:ro \
    -v /root/projects/autokernel/vllm_patches:/vllm_patches:ro \
    ${IMAGE} \
    sleep infinity

echo "=== Container started, installing build deps ==="

# Install system deps
docker exec ${CONTAINER_NAME} bash -c '
    apt-get update -qq && apt-get install -y -qq \
        python3 python3-pip python3-venv python3-dev \
        git wget curl gcc g++ 2>&1 | tail -5
'

echo "=== Installing PyTorch 2.10 + CUDA 12.8 ==="
docker exec ${CONTAINER_NAME} bash -c '
    pip3 install --break-system-packages \
        torch==2.10.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -5
'

echo "=== Cloning vLLM (full clone for PR cherry-picks) ==="
docker exec ${CONTAINER_NAME} bash -c '
    cd /build 2>/dev/null || mkdir -p /build && cd /build
    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    echo "vLLM cloned at commit: $(git rev-parse --short HEAD)"
    echo "Branch: $(git branch --show-current)"
'

echo "=== Fetching and merging community PRs ==="
docker exec ${CONTAINER_NAME} bash -c '
    cd /build/vllm

    # PR #38891 - Per-layer attention backend for Gemma4
    echo "--- PR #38891: per-layer attention backend ---"
    git fetch origin pull/38891/head:pr-38891
    git merge --no-edit pr-38891 && echo "PR #38891 merged OK" || echo "PR #38891 CONFLICT"

    # PR #39084 - Fix NVFP4 expert scale suffix mapping
    echo "--- PR #39084: NVFP4 scale suffix fix ---"
    git fetch origin pull/39084/head:pr-39084
    git merge --no-edit pr-39084 && echo "PR #39084 merged OK" || echo "PR #39084 CONFLICT"

    # PR #39406 - Robust quantized MoE expert weight loading
    echo "--- PR #39406: robust MoE weight loading ---"
    git fetch origin pull/39406/head:pr-39406
    git merge --no-edit pr-39406 && echo "PR #39406 merged OK" || echo "PR #39406 CONFLICT"

    echo ""
    echo "=== Final state ==="
    git log --oneline -10
'

echo "=== Building vLLM from source (MAX_JOBS=4) ==="
echo "This will take 30-60 minutes..."
docker exec -e MAX_JOBS=4 -e CUDA_PARALLEL_JOBS=4 -e TORCH_CUDA_ARCH_LIST="12.0" \
    ${CONTAINER_NAME} bash -c '
    cd /build/vllm
    pip3 install --break-system-packages -e ".[all]" 2>&1 | tee /build/build.log
    echo ""
    echo "=== Build complete ==="
    python3 -c "import vllm; print(\"vLLM version:\", vllm.__version__)"
'

echo ""
echo "=== Done! ==="
echo "Container: ${CONTAINER_NAME}"
echo "To use: docker exec -it --gpus all ${CONTAINER_NAME} bash"
echo "To test: docker exec ${CONTAINER_NAME} python3 -c 'import vllm; print(vllm.__version__)'"
