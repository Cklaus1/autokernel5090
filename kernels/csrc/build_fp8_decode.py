#!/usr/bin/env python3
"""
Build the FP8 Paged Decode Attention CUDA kernel and register as torch op.

Usage:
    python3 kernels/csrc/build_fp8_decode.py

Produces: /tmp/build_fp8_decode/fp8_decode.so
Registers: torch.ops.fp8_decode.paged_attention
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "fp8_paged_decode_attention.cu")
BUILD_DIR = "/tmp/build_fp8_decode"
SO_PATH = os.path.join(BUILD_DIR, "fp8_decode.so")

# Prefer /usr/local/cuda/bin/nvcc over /usr/bin/nvcc
NVCC = "/usr/local/cuda/bin/nvcc"
if not os.path.exists(NVCC):
    NVCC = "nvcc"

# Use gcc-12 if available (gcc-13 unsupported by nvcc 12.8)
GCC = "/usr/bin/g++-12" if os.path.exists("/usr/bin/g++-12") else "c++"


def get_nvcc_version():
    out = subprocess.check_output([NVCC, "--version"], text=True)
    m = re.search(r'release (\d+)\.(\d+)', out)
    return (int(m.group(1)), int(m.group(2))) if m else (12, 8)


def build_kernel():
    """Compile kernel using raw nvcc + g++."""
    import torch

    os.makedirs(BUILD_DIR, exist_ok=True)

    nvcc_ver = get_nvcc_version()
    print(f"[BUILD] nvcc: {NVCC} (version {nvcc_ver[0]}.{nvcc_ver[1]})")
    print(f"[BUILD] host compiler: {GCC}")

    # Get GPU arch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}{cap[1]}"
    else:
        arch = "120"

    arch_flag = f"-gencode=arch=compute_{arch},code=sm_{arch}"
    print(f"[BUILD] GPU arch: {arch}, flag: {arch_flag}")

    # Include paths
    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    cuda_include = "/usr/local/cuda/include"
    python_include = subprocess.check_output(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True
    ).strip()

    includes = [
        "-isystem", torch_include,
        "-isystem", torch_api_include,
        "-isystem", cuda_include,
        "-isystem", python_include,
    ]

    # Write torch bindings wrapper
    wrapper_path = os.path.join(BUILD_DIR, "torch_bindings.cpp")
    with open(wrapper_path, 'w') as f:
        f.write(r'''
#include <torch/all.h>

void fp8_paged_decode_attention(
    torch::Tensor& output,
    torch::Tensor const& query,
    torch::Tensor const& kv_cache_k,
    torch::Tensor const& kv_cache_v,
    torch::Tensor const& k_scale,
    torch::Tensor const& v_scale,
    torch::Tensor const& block_table,
    torch::Tensor const& seq_lens,
    torch::Tensor& mid_out,
    double sm_scale,
    double logits_soft_cap,
    int64_t num_kv_splits,
    int64_t head_dim,
    int64_t num_kv_heads,
    int64_t kv_group_size,
    int64_t page_size,
    int64_t per_head_scale);

TORCH_LIBRARY(fp8_decode, ops) {
    ops.def(
        "paged_attention(Tensor! output, Tensor query, Tensor kv_cache_k, "
        "Tensor kv_cache_v, Tensor k_scale, Tensor v_scale, "
        "Tensor block_table, Tensor seq_lens, Tensor! mid_out, "
        "float sm_scale, float logits_soft_cap, "
        "int num_kv_splits, int head_dim, int num_kv_heads, "
        "int kv_group_size, int page_size, int per_head_scale) -> ()");
    ops.impl("paged_attention", torch::kCUDA,
             &fp8_paged_decode_attention);
}
''')

    # Step 1: Compile CUDA kernel
    nvcc_cmd = [
        NVCC,
        "-ccbin", GCC,
        *includes,
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--compiler-options", "-fPIC",
        arch_flag,
        "-O3", "--use_fast_math", "-std=c++20",
        "-c", KERNEL_SRC,
        "-o", os.path.join(BUILD_DIR, "kernel.o"),
    ]
    print("[BUILD] Compiling CUDA kernel...")
    print(" ".join(nvcc_cmd))
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDOUT:\n{r.stdout}")
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] CUDA kernel compiled OK")

    # Step 2: Compile C++ wrapper
    cxx_cmd = [
        GCC,
        *includes,
        "-fPIC", "-std=c++20", "-O3",
        "-c", wrapper_path,
        "-o", os.path.join(BUILD_DIR, "wrapper.o"),
    ]
    print("[BUILD] Compiling C++ wrapper...")
    r = subprocess.run(cxx_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] C++ wrapper compiled OK")

    # Step 3: Link
    torch_lib = os.path.join(torch_dir, "lib")
    link_cmd = [
        GCC, "-shared",
        os.path.join(BUILD_DIR, "kernel.o"),
        os.path.join(BUILD_DIR, "wrapper.o"),
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        "-L/usr/local/cuda/lib64", "-lcudart",
        "-o", SO_PATH,
    ]
    print("[BUILD] Linking...")
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print(f"[BUILD] Linked: {SO_PATH}")

    # Step 4: Load
    torch.ops.load_library(SO_PATH)
    print("[BUILD] Loaded into torch.ops.fp8_decode")
    return SO_PATH


def load_library():
    """Load pre-built library if available."""
    import torch
    if os.path.exists(SO_PATH):
        torch.ops.load_library(SO_PATH)
        return True
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("FP8 Paged Decode Attention CUDA Kernel Builder")
    print("=" * 60)

    so_path = build_kernel()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nShared library: {so_path}")
    print("\nAvailable op:")
    print("  torch.ops.fp8_decode.paged_attention")
