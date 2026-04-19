#!/usr/bin/env python3
"""
Build fused_silu_fp4_epilogue.cu into workspace/fused_silu_fp4_sm120a.so.

Mirrors build_fused_shuffle_quant.py (importlib-loadable pybind11 .so).
Targets SM120a (CUTLASS NVFP4 Blackwell consumer path).

Usage:
    python3 kernels/csrc/build_fused_silu_fp4.py
"""
import os
import re
import subprocess
import sys
import sysconfig

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Auto-inject the project venv site-packages when invoked under system python,
# so we link against the same torch the serving wrapper uses.
_venv_sp = os.path.join(REPO_ROOT, ".venv", "lib", "python3.12", "site-packages")
if os.path.isdir(_venv_sp) and _venv_sp not in sys.path:
    sys.path.insert(0, _venv_sp)
KERNEL_SRC = os.path.join(SCRIPT_DIR, "fused_silu_fp4_epilogue.cu")
BUILD_DIR = "/tmp/build_fused_silu_fp4"
SO_NAME = "fused_silu_fp4_sm120a.so"
SO_OUT = os.path.join(REPO_ROOT, "workspace", SO_NAME)
SO_BUILD = os.path.join(BUILD_DIR, SO_NAME)

NVCC = "/usr/local/cuda/bin/nvcc"
if not os.path.exists(NVCC):
    NVCC = "nvcc"

GCC = "/usr/bin/g++-12" if os.path.exists("/usr/bin/g++-12") else "c++"


def get_nvcc_version():
    try:
        out = subprocess.check_output([NVCC, "--version"], text=True)
        m = re.search(r"release (\d+)\.(\d+)", out)
        return (int(m.group(1)), int(m.group(2))) if m else (12, 8)
    except Exception:
        return (12, 8)


def build():
    import torch

    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SO_OUT), exist_ok=True)

    nvcc_ver = get_nvcc_version()
    print(f"[BUILD] nvcc: {NVCC} version {nvcc_ver[0]}.{nvcc_ver[1]}")
    print(f"[BUILD] host compiler: {GCC}")

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}{cap[1]}"
    else:
        arch = "120"
    if int(arch) >= 120:
        arch_flag = f"-gencode=arch=compute_{arch}a,code=sm_{arch}a"
    else:
        arch_flag = f"-gencode=arch=compute_{arch},code=sm_{arch}"
    print(f"[BUILD] arch_flag: {arch_flag}")

    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    cuda_include = "/usr/local/cuda/include"
    python_include = sysconfig.get_path("include")

    pybind11_include = os.path.join(torch_include)

    includes = [
        "-isystem", torch_include,
        "-isystem", torch_api_include,
        "-isystem", cuda_include,
        "-isystem", python_include,
        "-isystem", pybind11_include,
    ]

    nvcc_cmd = [
        NVCC,
        "-ccbin", GCC,
        *includes,
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-DTORCH_EXTENSION_NAME=fused_silu_fp4_epilogue",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"),
        "--expt-relaxed-constexpr",
        "--compiler-options", "-fPIC",
        arch_flag,
        "-O3", "--use_fast_math", "-std=c++17",
        "-c", KERNEL_SRC,
        "-o", os.path.join(BUILD_DIR, "kernel.o"),
    ]
    print("[BUILD] Compiling fused_silu_fp4_epilogue.cu ...")
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        sys.exit(1)
    print("[BUILD] Compile OK")

    torch_lib = os.path.join(torch_dir, "lib")
    link_cmd = [
        GCC, "-shared",
        os.path.join(BUILD_DIR, "kernel.o"),
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        "-ltorch_python",
        "-L/usr/local/cuda/lib64", "-lcudart",
        "-o", SO_BUILD,
    ]
    print("[BUILD] Linking ...")
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:", r.stdout)
        print("STDERR:", r.stderr)
        sys.exit(1)
    print(f"[BUILD] Linked: {SO_BUILD}")

    import shutil
    shutil.copy2(SO_BUILD, SO_OUT)
    print(f"[BUILD] Installed: {SO_OUT}")

    return SO_OUT


if __name__ == "__main__":
    build()
