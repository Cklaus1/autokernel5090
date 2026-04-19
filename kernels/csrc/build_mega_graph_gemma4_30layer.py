#!/usr/bin/env python3
"""
Build the mega-graph Gemma4 30-layer prototype kernel.

Produces: /tmp/build_mega_graph_gemma4_30/libmega_graph_gemma4_30.so
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "mega_graph_gemma4_30layer.cu")
BUILD_DIR = "/tmp/build_mega_graph_gemma4_30"
SO_PATH = os.path.join(BUILD_DIR, "libmega_graph_gemma4_30.so")

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

    nvcc_cmd = [
        NVCC, "-ccbin", GCC,
        "--expt-relaxed-constexpr",
        "--compiler-options", "-fPIC",
        "-rdc", "true",
        arch_flag,
        "-O3", "-std=c++17",  # NB: removed --use_fast_math to match
                              # torch.sigmoid/rsqrt precision across 30 layers.
        "-c", KERNEL_SRC,
        "-o", os.path.join(BUILD_DIR, "mega_graph_gemma4_30.o"),
    ]
    print("[BUILD] Compiling CUDA kernel ...")
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n" + r.stdout)
        print("STDERR:\n" + r.stderr)
        sys.exit(1)
    print("[BUILD] OK")

    dlink_cmd = [
        NVCC, "-ccbin", GCC,
        "--compiler-options", "-fPIC",
        arch_flag,
        "-dlink",
        os.path.join(BUILD_DIR, "mega_graph_gemma4_30.o"),
        "-lcudadevrt",
        "-o", os.path.join(BUILD_DIR, "mega_graph_gemma4_30_dlink.o"),
    ]
    print("[BUILD] Device-linking ...")
    r = subprocess.run(dlink_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n" + r.stdout)
        print("STDERR:\n" + r.stderr)
        sys.exit(1)
    print("[BUILD] OK")

    cuda_lib = "/usr/local/cuda-12.8/lib64"
    if not os.path.exists(cuda_lib):
        cuda_lib = "/usr/local/cuda/lib64"
    link_cmd = [
        GCC, "-shared",
        os.path.join(BUILD_DIR, "mega_graph_gemma4_30.o"),
        os.path.join(BUILD_DIR, "mega_graph_gemma4_30_dlink.o"),
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
