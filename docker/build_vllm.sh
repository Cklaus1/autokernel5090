#!/bin/bash
# Build vLLM from source with CUDA 13.2 + SM120 fixes
# Uses Docker with memory limit to prevent OOM killing WSL
set -e

CONTAINER_NAME="vllm-build"
IMAGE="nvidia/cuda:13.2.0-devel-ubuntu24.04"
MEMORY_LIMIT="36g"

echo "=== vLLM Build: CUDA 13.2 + Latest Main ==="
echo "=== Starting build container (memory limit: ${MEMORY_LIMIT}) ==="

# Remove old container if exists
docker rm -f ${CONTAINER_NAME} 2>/dev/null || true

# Start container with GPU + memory limit
docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    --memory=${MEMORY_LIMIT} \
    --memory-swap=${MEMORY_LIMIT} \
    -v /root/projects/autokernel/patches:/patches:ro \
    -v /root/projects/autokernel:/workspace:ro \
    ${IMAGE} \
    sleep infinity

echo "=== Container started, installing build deps ==="

# Install system deps
docker exec ${CONTAINER_NAME} bash -c '
    apt-get update -qq && apt-get install -y -qq \
        python3 python3-pip python3-venv python3-dev \
        git wget curl gcc g++ ninja-build 2>&1 | tail -5
'

echo "=== Installing PyTorch nightly + CUDA 13.2 ==="
docker exec ${CONTAINER_NAME} bash -c '
    pip3 install --break-system-packages \
        --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu132 2>&1 | tail -5
    python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.version.cuda}\")"
'

echo "=== Cloning vLLM (latest main) ==="
docker exec ${CONTAINER_NAME} bash -c '
    cd /build 2>/dev/null || mkdir -p /build && cd /build
    git clone --depth=100 https://github.com/vllm-project/vllm.git
    cd vllm
    echo "vLLM cloned at commit: $(git rev-parse --short HEAD)"
    echo "Branch: $(git branch --show-current)"
    echo ""
    echo "=== Recent SM120/Blackwell fixes included ==="
    git log --oneline --all --grep="SM120\|sm120\|Blackwell\|blackwell\|NVFP4\|nvfp4" | head -20
'

echo "=== Building vLLM from source (MAX_JOBS=4) ==="
echo "This will take 30-60 minutes..."
docker exec -e MAX_JOBS=4 -e CUDA_PARALLEL_JOBS=4 -e TORCH_CUDA_ARCH_LIST="12.0" \
    ${CONTAINER_NAME} bash -c '
    cd /build/vllm
    pip3 install --break-system-packages -e ".[all]" 2>&1 | tee /build/build.log | tail -20
    echo ""
    echo "=== Build complete ==="
    python3 -c "import vllm; print(\"vLLM version:\", vllm.__version__)"
    python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.version.cuda}\")"
'

echo ""
echo "=== Committing container as image ==="
docker commit ${CONTAINER_NAME} vllm-cu132:latest
echo "Image saved as vllm-cu132:latest"

echo ""
echo "=== Done! ==="
echo "Container: ${CONTAINER_NAME}"
echo "Image: vllm-cu132:latest"
echo "To use: docker run --gpus all -v /root/models:/models -v /root/.cache/huggingface:/root/.cache/huggingface -v /root/projects/autokernel:/workspace vllm-cu132:latest bash"
