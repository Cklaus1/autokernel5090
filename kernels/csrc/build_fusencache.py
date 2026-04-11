#!/usr/bin/env python3
"""
Build the FusenCache CUDA kernels and register as torch ops.

Usage:
    python3 kernels/csrc/build_fusencache.py

Produces: /tmp/build_fusencache/fusencache_decode.so
Registers:
    torch.ops.fusencache.decode_attention
    torch.ops.fusencache.store_kv
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_DECODE_SRC = os.path.join(SCRIPT_DIR, "fusencache_decode_attention.cu")
KERNEL_STORE_SRC = os.path.join(SCRIPT_DIR, "fusencache_store_kv.cu")
BUILD_DIR = "/tmp/build_fusencache"
SO_PATH = os.path.join(BUILD_DIR, "fusencache_decode.so")

# Prefer /usr/local/cuda/bin/nvcc over /usr/bin/nvcc (Ubuntu package).
# The /usr/local/cuda version has cicc in its nvvm/bin/ dir.
NVCC = "/usr/local/cuda/bin/nvcc"
if not os.path.exists(NVCC):
    NVCC = "nvcc"

# Use gcc-12 if available (gcc-13 is unsupported by nvcc 12.8)
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

    # SM120 (Blackwell) needs compute_120 code
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

void fusencache_decode_attention(
    torch::Tensor& output,
    torch::Tensor const& query,
    torch::Tensor const& kv_cache,
    torch::Tensor const& scales,
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
    int64_t k_bits,
    int64_t v_bits,
    int64_t scale_block_k,
    int64_t scale_block_v,
    double k_offset,
    double v_offset);

void fusencache_store_kv(
    torch::Tensor const& key,
    torch::Tensor const& value,
    torch::Tensor& kv_cache,
    torch::Tensor& scales,
    torch::Tensor const& slot_mapping,
    int64_t head_dim,
    int64_t page_size,
    int64_t k_bits,
    int64_t v_bits,
    int64_t k_scale_block,
    int64_t v_scale_block,
    double k_offset,
    double v_offset);

TORCH_LIBRARY(fusencache, ops) {
    ops.def(
        "decode_attention(Tensor! output, Tensor query, Tensor kv_cache, "
        "Tensor scales, Tensor block_table, Tensor seq_lens, "
        "Tensor! mid_out, float sm_scale, float logits_soft_cap, "
        "int num_kv_splits, int head_dim, int num_kv_heads, "
        "int kv_group_size, int page_size, "
        "int k_bits, int v_bits, int scale_block_k, int scale_block_v, "
        "float k_offset, float v_offset) -> ()");
    ops.impl("decode_attention", torch::kCUDA,
             &fusencache_decode_attention);

    ops.def(
        "store_kv(Tensor key, Tensor value, Tensor! kv_cache, "
        "Tensor! scales, Tensor slot_mapping, "
        "int head_dim, int page_size, "
        "int k_bits, int v_bits, int k_scale_block, int v_scale_block, "
        "float k_offset, float v_offset) -> ()");
    ops.impl("store_kv", torch::kCUDA,
             &fusencache_store_kv);
}
''')

    # Common nvcc flags
    nvcc_common = [
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
        "-O3", "--use_fast_math", "-std=c++17",
    ]

    # Step 1a: Compile decode attention CUDA kernel
    nvcc_cmd = nvcc_common + [
        "-c", KERNEL_DECODE_SRC,
        "-o", os.path.join(BUILD_DIR, "kernel_decode.o"),
    ]
    print("[BUILD] Compiling decode attention CUDA kernel...")
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDOUT:\n{r.stdout}")
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] Decode attention kernel compiled OK")

    # Step 1b: Compile store KV CUDA kernel
    nvcc_cmd_store = nvcc_common + [
        "-c", KERNEL_STORE_SRC,
        "-o", os.path.join(BUILD_DIR, "kernel_store.o"),
    ]
    print("[BUILD] Compiling store KV CUDA kernel...")
    r = subprocess.run(nvcc_cmd_store, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDOUT:\n{r.stdout}")
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] Store KV kernel compiled OK")

    # Step 2: Compile C++ wrapper
    cxx_cmd = [
        GCC,
        *includes,
        "-fPIC", "-std=c++17", "-O3",
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
        os.path.join(BUILD_DIR, "kernel_decode.o"),
        os.path.join(BUILD_DIR, "kernel_store.o"),
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
    print("[BUILD] Loaded into torch.ops.fusencache")
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
    print("FusenCache CUDA Kernel Builder (decode + store)")
    print("=" * 60)

    so_path = build_kernel()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nShared library: {so_path}")
    print("\nAvailable ops:")
    print("  torch.ops.fusencache.decode_attention")
    print("  torch.ops.fusencache.store_kv")
