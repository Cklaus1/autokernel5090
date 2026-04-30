#!/usr/bin/env python3
"""Build v7.1 Phase-0 K-gang FP4 real-MoE microbench."""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "mega_graph_v7_1_phase0_microbench.cu")
BUILD_DIR = "/tmp/build_mega_graph_v7_1_phase0"
SO_PATH = os.path.join(BUILD_DIR, "libmega_graph_v7_1_phase0.so")

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

    obj = os.path.join(BUILD_DIR, "mega_graph_v7_1_phase0.o")
    dlink = os.path.join(BUILD_DIR, "mega_graph_v7_1_phase0_dlink.o")

    nvcc_cmd = [
        NVCC, "-ccbin", GCC,
        "--expt-relaxed-constexpr",
        "--compiler-options", "-fPIC",
        "-rdc", "true",
        arch_flag,
        "-O3", "-std=c++17",
        "-Xptxas", "-v",
        "-c", KERNEL_SRC,
        "-o", obj,
    ]
    print("[BUILD] Compiling CUDA kernel ...")
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.stderr:
        print("STDERR:\n" + r.stderr)
    if r.returncode != 0:
        print("STDOUT:\n" + r.stdout)
        sys.exit(1)
    print("[BUILD] compile OK")

    dlink_cmd = [
        NVCC, "-ccbin", GCC,
        "--compiler-options", "-fPIC",
        arch_flag,
        "-dlink",
        obj,
        "-lcudadevrt",
        "-o", dlink,
    ]
    print("[BUILD] Device-linking ...")
    r = subprocess.run(dlink_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n" + r.stdout)
        print("STDERR:\n" + r.stderr)
        sys.exit(1)
    print("[BUILD] dlink OK")

    cuda_lib = "/usr/local/cuda-12.8/lib64"
    if not os.path.exists(cuda_lib):
        cuda_lib = "/usr/local/cuda/lib64"
    link_cmd = [
        GCC, "-shared",
        obj, dlink,
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
