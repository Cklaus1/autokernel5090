#!/usr/bin/env python3
"""
Build the mega-graph Gemma4 2-layer prototype kernel.

Produces: /tmp/build_mega_graph_gemma4/libmega_graph_gemma4.so

Usage:
    python3 kernels/csrc/build_mega_graph_gemma4.py
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "mega_graph_gemma4.cu")
BUILD_DIR = "/tmp/build_mega_graph_gemma4"
SO_PATH = os.path.join(BUILD_DIR, "libmega_graph_gemma4.so")

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
if not os.path.exists(NVCC):
    NVCC = "/usr/local/cuda/bin/nvcc"
if not os.path.exists(NVCC):
    NVCC = "nvcc"

GCC = "/usr/bin/g++-12" if os.path.exists("/usr/bin/g++-12") else "c++"


def build():
    os.makedirs(BUILD_DIR, exist_ok=True)
    arch_flag = "-gencode=arch=compute_120a,code=sm_120a"
    print(f"[BUILD] nvcc:         {NVCC}")
    print(f"[BUILD] host compiler: {GCC}")
    print(f"[BUILD] arch flag:    {arch_flag}")

    # Step 1: compile .cu -> .o with -rdc true (for cooperative_groups)
    nvcc_cmd = [
        NVCC, "-ccbin", GCC,
        "--expt-relaxed-constexpr",
        "--compiler-options", "-fPIC",
        "-rdc", "true",
        arch_flag,
        "-O3", "--use_fast_math", "-std=c++17",
        "-c", KERNEL_SRC,
        "-o", os.path.join(BUILD_DIR, "mega_graph_gemma4.o"),
    ]
    print("[BUILD] Compiling CUDA kernel ...")
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n" + r.stdout)
        print("STDERR:\n" + r.stderr)
        sys.exit(1)
    print("[BUILD] OK")

    # Step 2: device-link
    dlink_cmd = [
        NVCC, "-ccbin", GCC,
        "--compiler-options", "-fPIC",
        arch_flag,
        "-dlink",
        os.path.join(BUILD_DIR, "mega_graph_gemma4.o"),
        "-lcudadevrt",
        "-o", os.path.join(BUILD_DIR, "mega_graph_gemma4_dlink.o"),
    ]
    print("[BUILD] Device-linking ...")
    r = subprocess.run(dlink_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n" + r.stdout)
        print("STDERR:\n" + r.stderr)
        sys.exit(1)
    print("[BUILD] OK")

    # Step 3: link .so
    cuda_lib = "/usr/local/cuda-12.8/lib64"
    if not os.path.exists(cuda_lib):
        cuda_lib = "/usr/local/cuda/lib64"
    link_cmd = [
        GCC, "-shared",
        os.path.join(BUILD_DIR, "mega_graph_gemma4.o"),
        os.path.join(BUILD_DIR, "mega_graph_gemma4_dlink.o"),
        f"-L{cuda_lib}",
        "-lcudart", "-lcudadevrt",
        "-o", SO_PATH,
    ]
    print("[BUILD] Linking shared library ...")
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n" + r.stdout)
        print("STDERR:\n" + r.stderr)
        sys.exit(1)
    print(f"[BUILD] DONE -> {SO_PATH}")


if __name__ == "__main__":
    build()
