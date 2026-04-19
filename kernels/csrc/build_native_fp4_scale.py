#!/usr/bin/env python3
"""Build the native_fp4_scale_quant kernel and register it as a torch op."""

import os
import subprocess
import sys
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "native_fp4_scale_kernel.cu")
BUILD_DIR = "/tmp/build_native_fp4_scale"
SO_PATH   = os.path.join(BUILD_DIR, "native_fp4_scale.so")


def build():
    import torch
    os.makedirs(BUILD_DIR, exist_ok=True)

    cap = torch.cuda.get_device_capability()
    arch = f"{cap[0]}{cap[1]}"
    # SM120a needed for e2m1/e4m3 packed PTX
    arch_flag = f"-gencode=arch=compute_{arch}a,code=sm_{arch}a"

    torch_dir = os.path.dirname(torch.__file__)
    torch_inc = os.path.join(torch_dir, "include")
    torch_api_inc = os.path.join(torch_inc, "torch", "csrc", "api", "include")
    cuda_inc = "/usr/local/cuda/include"
    py_inc = subprocess.check_output(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True).strip()

    includes = ["-isystem", torch_inc, "-isystem", torch_api_inc,
                "-isystem", cuda_inc, "-isystem", py_inc]

    wrapper = os.path.join(BUILD_DIR, "wrapper.cpp")
    with open(wrapper, "w") as f:
        f.write(r'''
#include <torch/all.h>

void native_fp4_scale_quant(
    torch::Tensor& out_fp4,
    torch::Tensor& out_sf,
    torch::Tensor const& input,
    torch::Tensor const& input_global_scale);

TORCH_LIBRARY_FRAGMENT(_Cnfs, ops) {
    ops.def(
        "native_fp4_scale_quant(Tensor! out_fp4, Tensor! out_sf, "
        "Tensor input, Tensor input_global_scale) -> ()");
    ops.impl("native_fp4_scale_quant", torch::kCUDA, &native_fp4_scale_quant);
}
''')

    # Compile kernel
    r = subprocess.run([
        "nvcc", *includes,
        "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr", "--compiler-options", "-fPIC",
        arch_flag, "-O3", "--use_fast_math", "-std=c++17",
        "-c", KERNEL_SRC,
        "-o", os.path.join(BUILD_DIR, "kernel.o"),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("NVCC STDERR:\n" + r.stderr); sys.exit(1)
    print("[BUILD] kernel.o OK")

    r = subprocess.run([
        "c++", *includes, "-fPIC", "-std=c++17", "-O3",
        "-c", wrapper, "-o", os.path.join(BUILD_DIR, "wrapper.o"),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("CXX STDERR:\n" + r.stderr); sys.exit(1)
    print("[BUILD] wrapper.o OK")

    torch_lib = os.path.join(torch_dir, "lib")
    r = subprocess.run([
        "c++", "-shared",
        os.path.join(BUILD_DIR, "kernel.o"),
        os.path.join(BUILD_DIR, "wrapper.o"),
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        "-L/usr/local/cuda/lib64", "-lcudart",
        "-o", SO_PATH,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print("LINK STDERR:\n" + r.stderr); sys.exit(1)
    print(f"[BUILD] linked {SO_PATH}")

    torch.ops.load_library(SO_PATH)
    print("[BUILD] loaded into torch.ops._Cnfs")
    return SO_PATH


if __name__ == "__main__":
    build()
