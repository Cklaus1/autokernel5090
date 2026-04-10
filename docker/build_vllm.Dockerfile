FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=4
ENV CUDA_PARALLEL_JOBS=4
ENV TORCH_CUDA_ARCH_LIST="12.0"

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    git wget curl gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (matching host: 2.10 + cu128)
RUN pip3 install --break-system-packages \
    torch==2.10.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Clone vLLM latest
WORKDIR /build
RUN git clone --depth=1 https://github.com/vllm-project/vllm.git

# Cherry-pick the community PRs
WORKDIR /build/vllm
# We'll fetch and cherry-pick PRs at build time via a script
COPY apply_patches.sh /build/apply_patches.sh
RUN chmod +x /build/apply_patches.sh

# Build vLLM from source
# MAX_JOBS=4 prevents the OOM that killed WSL last time
RUN pip3 install --break-system-packages -r requirements/build.txt 2>/dev/null || true
RUN MAX_JOBS=4 pip3 install --break-system-packages -e . 2>&1 | tee /build/build.log

CMD ["python3", "-c", "import vllm; print('vLLM', vllm.__version__, 'OK')"]
