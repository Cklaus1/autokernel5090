#!/usr/bin/env python3
"""
Build the persistent MoE dispatch kernel.

Usage:
    python3 kernels/csrc/build_persistent_moe.py

Produces: /tmp/build_persistent_moe/persistent_moe.so
Loads into: torch.ops.persistent_moe.dispatch
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "persistent_moe_dispatch.cu")
BUILD_DIR = "/tmp/build_persistent_moe"
SO_PATH = os.path.join(BUILD_DIR, "persistent_moe.so")

# Prefer /usr/local/cuda/bin/nvcc over /usr/bin/nvcc
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
    """Compile persistent MoE dispatch kernel."""
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

    # SM120 code generation
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

    # Write torch bindings wrapper (separate C++ file avoids nvcc issues)
    wrapper_path = os.path.join(BUILD_DIR, "torch_bindings.cpp")
    with open(wrapper_path, 'w') as f:
        f.write(r'''
#include <torch/all.h>

namespace persistent_moe {
void persistent_moe_dispatch(
    torch::Tensor& hidden,
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weights,
    torch::Tensor& a1_gscale,
    torch::Tensor& a2_gscale,
    torch::Tensor& gemm1_output,
    torch::Tensor& gemm2_output,
    torch::Tensor& output,
    torch::Tensor& sorted_hidden,
    torch::Tensor& sorted_fp4,
    torch::Tensor& sorted_sf,
    torch::Tensor& act_fp4,
    torch::Tensor& act_sf,
    torch::Tensor& expert_counts,
    torch::Tensor& expert_offsets,
    torch::Tensor& a_map,
    int64_t M, int64_t N, int64_t K, int64_t E, int64_t top_k,
    int64_t phase_mask);
}

TORCH_LIBRARY(persistent_moe, ops) {
    ops.def(
        "dispatch("
        "Tensor! hidden, Tensor! topk_ids, Tensor! topk_weights, "
        "Tensor! a1_gscale, Tensor! a2_gscale, "
        "Tensor! gemm1_output, Tensor! gemm2_output, "
        "Tensor! output, "
        "Tensor! sorted_hidden, Tensor! sorted_fp4, Tensor! sorted_sf, "
        "Tensor! act_fp4, Tensor! act_sf, "
        "Tensor! expert_counts, Tensor! expert_offsets, Tensor! a_map, "
        "int M, int N, int K, int E, int top_k, int phase_mask) -> ()");
    ops.impl("dispatch", torch::kCUDA,
             &persistent_moe::persistent_moe_dispatch);
}
''')

    # Step 1: Compile CUDA kernel with -rdc true for cooperative_groups
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
        "-rdc", "true",
        arch_flag,
        "-O3", "--use_fast_math", "-std=c++17",
        "-c", KERNEL_SRC,
        "-o", os.path.join(BUILD_DIR, "kernel.o"),
    ]
    print("[BUILD] Compiling CUDA kernel (with -rdc true for cooperative_groups)...")
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDOUT:\n{r.stdout}")
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] CUDA kernel compiled OK")

    # Step 2: Device link (required for -rdc true / cooperative_groups)
    dlink_cmd = [
        NVCC,
        "-ccbin", GCC,
        "--compiler-options", "-fPIC",
        arch_flag,
        "-dlink",
        os.path.join(BUILD_DIR, "kernel.o"),
        "-lcudadevrt",
        "-o", os.path.join(BUILD_DIR, "kernel_dlink.o"),
    ]
    print("[BUILD] Device linking (for cooperative_groups)...")
    r = subprocess.run(dlink_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDOUT:\n{r.stdout}")
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] Device link OK")

    # Step 3: Compile C++ wrapper (no -rdc needed)
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

    # Step 4: Link into shared library
    torch_lib = os.path.join(torch_dir, "lib")
    link_cmd = [
        GCC, "-shared",
        os.path.join(BUILD_DIR, "kernel.o"),
        os.path.join(BUILD_DIR, "kernel_dlink.o"),
        os.path.join(BUILD_DIR, "wrapper.o"),
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        "-L/usr/local/cuda/lib64", "-lcudart", "-lcudadevrt",
        "-o", SO_PATH,
    ]
    print("[BUILD] Linking shared library...")
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDOUT:\n{r.stdout}")
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print(f"[BUILD] Linked: {SO_PATH}")

    # Step 5: Load
    torch.ops.load_library(SO_PATH)
    print("[BUILD] Loaded into torch.ops.persistent_moe")
    return SO_PATH


def check_cooperative_support():
    """Check if device supports cooperative launch."""
    import torch
    device = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device)
    props = torch.cuda.get_device_properties(device)

    print(f"\n[INFO] Device: {name}")
    print(f"[INFO] Compute capability: {cap[0]}.{cap[1]}")
    print(f"[INFO] SM count: {props.multi_processor_count}")
    print(f"[INFO] Cooperative launch requires SM >= 6.0: {'YES' if cap[0] >= 6 else 'NO'}")

    return cap[0] >= 6


def load_library():
    """Load pre-built library if available."""
    import torch
    if os.path.exists(SO_PATH):
        torch.ops.load_library(SO_PATH)
        return True
    return False


def main():
    print("=" * 60)
    print("Persistent MoE Dispatch Kernel Builder")
    print("=" * 60)

    if not check_cooperative_support():
        print("ERROR: Device does not support cooperative launch")
        sys.exit(1)

    so_path = build_kernel()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nShared library: {so_path}")
    print("\nAvailable ops:")
    print("  torch.ops.persistent_moe.dispatch")


if __name__ == "__main__":
    main()
