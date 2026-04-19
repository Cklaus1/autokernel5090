#!/usr/bin/env python3
"""Build the cp.async prototype FusenCache decode kernel as a separate shared lib.

Usage:
    python3 kernels/csrc/build_fusencache_cpasync.py

Produces: /tmp/build_fusencache_cpasync/fusencache_cpasync.so
Registers torch op: torch.ops.fusencache_cpasync.decode_attention
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "fusencache_decode_cpasync.cu")
BUILD_DIR = "/tmp/build_fusencache_cpasync"
SO_PATH = os.path.join(BUILD_DIR, "fusencache_cpasync.so")

NVCC = "/usr/local/cuda/bin/nvcc"
if not os.path.exists(NVCC):
    for v in ("12.8", "12.9", "13.0"):
        cand = f"/usr/local/cuda-{v}/bin/nvcc"
        if os.path.exists(cand):
            NVCC = cand
            break
    else:
        NVCC = "nvcc"

GCC = "/usr/bin/g++-12" if os.path.exists("/usr/bin/g++-12") else "c++"


def get_nvcc_version():
    out = subprocess.check_output([NVCC, "--version"], text=True)
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else (12, 8)


def build():
    import torch

    os.makedirs(BUILD_DIR, exist_ok=True)

    v = get_nvcc_version()
    print(f"[BUILD] nvcc: {NVCC} ({v[0]}.{v[1]})")
    print(f"[BUILD] host cc: {GCC}")

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}{cap[1]}"
    else:
        arch = "120"

    # RTX PRO 6000 is sm_120a (arch suffix 'a' enables TMA / new cp.async.bulk intrinsics).
    # For 16-B cp.async used here, plain sm_120 is sufficient; pick '120a' to match
    # the rest of this repo's kernels (sm120a Blackwell consumer).
    arch_flag = f"-gencode=arch=compute_{arch}a,code=sm_{arch}a"
    print(f"[BUILD] arch flag: {arch_flag}")

    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    cuda_include = os.path.join(os.path.dirname(os.path.dirname(NVCC)), "include")
    python_include = subprocess.check_output(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True,
    ).strip()

    includes = [
        "-isystem", torch_include,
        "-isystem", torch_api_include,
        "-isystem", cuda_include,
        "-isystem", python_include,
    ]

    wrapper_path = os.path.join(BUILD_DIR, "torch_bindings.cpp")
    with open(wrapper_path, "w") as f:
        f.write(r'''
#include <torch/all.h>

void fusencache_decode_cpasync(
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

TORCH_LIBRARY(fusencache_cpasync, ops) {
    ops.def(
        "decode_attention(Tensor! output, Tensor query, Tensor kv_cache, "
        "Tensor scales, Tensor block_table, Tensor seq_lens, "
        "Tensor! mid_out, float sm_scale, float logits_soft_cap, "
        "int num_kv_splits, int head_dim, int num_kv_heads, "
        "int kv_group_size, int page_size, "
        "int k_bits, int v_bits, int scale_block_k, int scale_block_v, "
        "float k_offset, float v_offset) -> ()");
    ops.impl("decode_attention", torch::kCUDA, &fusencache_decode_cpasync);
}
''')

    nvcc_common = [
        NVCC, "-ccbin", GCC,
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

    obj_kernel = os.path.join(BUILD_DIR, "kernel.o")
    obj_wrap   = os.path.join(BUILD_DIR, "wrap.o")

    print("[BUILD] nvcc compile ...")
    r = subprocess.run(nvcc_common + ["-c", KERNEL_SRC, "-o", obj_kernel],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("STDOUT:\n", r.stdout)
        print("STDERR:\n", r.stderr)
        sys.exit(1)
    print("[BUILD] nvcc OK")

    print("[BUILD] cxx wrapper ...")
    r = subprocess.run(
        [GCC, *includes, "-fPIC", "-std=c++17", "-O3",
         "-c", wrapper_path, "-o", obj_wrap],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:\n", r.stderr)
        sys.exit(1)

    torch_lib = os.path.join(torch_dir, "lib")
    cuda_lib  = os.path.join(os.path.dirname(os.path.dirname(NVCC)), "lib64")
    link_cmd = [
        GCC, "-shared", obj_kernel, obj_wrap,
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        f"-L{cuda_lib}", "-lcudart",
        "-o", SO_PATH,
    ]
    print("[BUILD] link ...")
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("STDERR:\n", r.stderr)
        sys.exit(1)

    torch.ops.load_library(SO_PATH)
    print(f"[BUILD] loaded: {SO_PATH}")
    return SO_PATH


def load_library():
    import torch
    if os.path.exists(SO_PATH):
        torch.ops.load_library(SO_PATH)
        return True
    return False


if __name__ == "__main__":
    print("=" * 60)
    print("FusenCache cp.async prototype builder")
    print("=" * 60)
    so_path = build()
    print("\nTorch op: torch.ops.fusencache_cpasync.decode_attention")
    print("Shared lib:", so_path)
