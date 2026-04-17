#!/bin/bash
# SGLang Experiment Environment Setup
# Source this before running experiments

# CUDA 12.8+ for SM120 (RTX 5090 Blackwell) JIT compilation
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Nvidia Python package libraries (cuDNN, cuBLAS, etc.)
NVIDIA_LIBS=$(python -c "import nvidia, os; base=os.path.dirname(nvidia.__file__); libs=[os.path.join(base,d,'lib') for d in os.listdir(base) if os.path.isdir(os.path.join(base,d,'lib'))]; print(':'.join(libs))")
export LD_LIBRARY_PATH=$NVIDIA_LIBS:$LD_LIBRARY_PATH

# SGLang env vars
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# Python
export SGLANG_PYTHON=/root/sglang_env/bin/python

echo "CUDA: $(nvcc --version | grep release)"
echo "SGLang Python: $SGLANG_PYTHON"
