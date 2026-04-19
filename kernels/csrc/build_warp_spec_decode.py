#!/usr/bin/env python3
"""
Build the warp-specialized decode attention CUDA kernel (SM120a).
Produces /tmp/build_warp_spec_decode/warp_spec_decode.so and registers
torch.ops.warp_spec_decode.decode.
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "warp_spec_decode_attention.cu")
BUILD_DIR = "/tmp/build_warp_spec_decode"
SO_PATH = os.path.join(BUILD_DIR, "warp_spec_decode.so")

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
if not os.path.exists(NVCC):
    NVCC = "/usr/local/cuda/bin/nvcc"
if not os.path.exists(NVCC):
    NVCC = "nvcc"

GCC = "/usr/bin/g++-12" if os.path.exists("/usr/bin/g++-12") else "c++"


def build_kernel():
    # Ensure we resolve venv torch first if present (avoids ABI mismatch)
    venv_site = "/home/cklaus/projects/autokernel/.venv/lib/python3.12/site-packages"
    if os.path.isdir(venv_site) and venv_site not in sys.path:
        sys.path.insert(0, venv_site)
    import torch
    os.makedirs(BUILD_DIR, exist_ok=True)

    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    cuda_include = "/usr/local/cuda/include"
    if not os.path.isdir(cuda_include):
        cuda_include = "/usr/local/cuda-12.8/include"

    python_include = subprocess.check_output(
        [sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True,
    ).strip()

    includes = [
        "-isystem", torch_include,
        "-isystem", torch_api_include,
        "-isystem", cuda_include,
        "-isystem", python_include,
    ]

    # Compile CUDA
    kernel_o = os.path.join(BUILD_DIR, "kernel.o")
    nvcc_cmd = [
        NVCC, "-ccbin", GCC,
        *includes,
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--compiler-options", "-fPIC",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-O3", "-std=c++20",
        "-c", KERNEL_SRC,
        "-o", kernel_o,
    ]
    print("[BUILD] compile:", " ".join(nvcc_cmd))
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        sys.exit(1)

    # Link
    torch_lib = os.path.join(torch_dir, "lib")
    cuda_lib = "/usr/local/cuda/lib64"
    if not os.path.isdir(cuda_lib):
        cuda_lib = "/usr/local/cuda-12.8/lib64"
    link_cmd = [
        GCC, "-shared", kernel_o,
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        f"-L{cuda_lib}", "-lcudart",
        "-o", SO_PATH,
    ]
    print("[BUILD] link:", " ".join(link_cmd))
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:", r.stderr)
        sys.exit(1)

    torch.ops.load_library(SO_PATH)
    print(f"[BUILD] OK -> {SO_PATH}")
    return SO_PATH


def load_library():
    import torch
    if os.path.exists(SO_PATH):
        torch.ops.load_library(SO_PATH)
        return True
    return False


if __name__ == "__main__":
    build_kernel()
    print("torch.ops.warp_spec_decode.decode registered")
