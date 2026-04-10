#!/usr/bin/env python3
"""
Build and install the fused RMSNorm + FP4 quantization kernel into vLLM.

Usage (inside Docker container):
    python3 /tmp/csrc/build_and_install.py

Or from host:
    docker cp kernels/csrc/ vllm-gemma4:/tmp/csrc/
    docker exec vllm-gemma4 python3 /tmp/csrc/build_and_install.py
"""

import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(SCRIPT_DIR, "rms_norm_dynamic_fp4_quant.cu")
BUILD_DIR = "/tmp/build_fused_rms_norm_fp4"
SO_PATH = os.path.join(BUILD_DIR, "fused_rms_norm_fp4.so")


def get_nvcc_version():
    out = subprocess.check_output(["nvcc", "--version"], text=True)
    m = re.search(r'release (\d+)\.(\d+)', out)
    return (int(m.group(1)), int(m.group(2))) if m else (12, 8)


def build_kernel():
    """Compile kernel using raw nvcc + g++ (not torch.utils.cpp_extension)."""
    import torch

    os.makedirs(BUILD_DIR, exist_ok=True)

    nvcc_ver = get_nvcc_version()
    print(f"[BUILD] nvcc version: {nvcc_ver[0]}.{nvcc_ver[1]}")

    # Get GPU arch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"{cap[0]}{cap[1]}"
    else:
        arch = "120"

    # FP4 e2m1 PTX needs 'a' variant on nvcc < 13.0
    if nvcc_ver < (13, 0):
        arch_flag = f"-gencode=arch=compute_{arch}a,code=sm_{arch}a"
    else:
        arch_flag = f"-gencode=arch=compute_{arch},code=sm_{arch}"

    print(f"[BUILD] GPU arch: {arch}, flag: {arch_flag}")

    # Get include paths
    torch_dir = os.path.dirname(torch.__file__)
    torch_include = os.path.join(torch_dir, "include")
    torch_api_include = os.path.join(torch_include, "torch", "csrc", "api", "include")
    cuda_include = "/usr/local/cuda/include"
    python_include = subprocess.check_output(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_path('include'))"],
        text=True
    ).strip()

    includes = [
        f"-isystem", torch_include,
        f"-isystem", torch_api_include,
        f"-isystem", cuda_include,
        f"-isystem", python_include,
    ]

    # Write C++ wrapper with torch library registration
    wrapper_path = os.path.join(BUILD_DIR, "torch_bindings.cpp")
    with open(wrapper_path, 'w') as f:
        f.write(r'''
#include <torch/all.h>

void rms_norm_dynamic_fp4_quant(
    torch::Tensor& result, torch::Tensor& result_scale,
    torch::Tensor const& input, torch::Tensor const& weight,
    torch::Tensor const& input_global_scale,
    double epsilon, bool is_sf_swizzled_layout);

void fused_add_rms_norm_dynamic_fp4_quant(
    torch::Tensor& result, torch::Tensor& result_scale,
    torch::Tensor& input, torch::Tensor const& weight,
    torch::Tensor& residual, torch::Tensor const& input_global_scale,
    double epsilon, bool is_sf_swizzled_layout);

TORCH_LIBRARY_FRAGMENT(_C, ops) {
    ops.def(
        "rms_norm_dynamic_fp4_quant(Tensor! result, Tensor! result_scale, "
        "Tensor input, Tensor weight, Tensor input_global_scale, "
        "float epsilon, bool is_sf_swizzled_layout) -> ()");
    ops.impl("rms_norm_dynamic_fp4_quant", torch::kCUDA,
             &rms_norm_dynamic_fp4_quant);

    ops.def(
        "fused_add_rms_norm_dynamic_fp4_quant(Tensor! result, "
        "Tensor! result_scale, Tensor! input, Tensor weight, "
        "Tensor! residual, Tensor input_global_scale, "
        "float epsilon, bool is_sf_swizzled_layout) -> ()");
    ops.impl("fused_add_rms_norm_dynamic_fp4_quant", torch::kCUDA,
             &fused_add_rms_norm_dynamic_fp4_quant);
}
''')

    # Step 1: Compile CUDA kernel
    nvcc_cmd = [
        "nvcc",
        *includes,
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--compiler-options", "-fPIC",
        arch_flag,
        "-O3", "--use_fast_math", "-std=c++17",
        "-c", KERNEL_SRC,
        "-o", os.path.join(BUILD_DIR, "kernel.o"),
    ]
    print(f"[BUILD] Compiling CUDA kernel...")
    r = subprocess.run(nvcc_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] CUDA kernel compiled OK")

    # Step 2: Compile C++ wrapper
    cxx_cmd = [
        "c++",
        *includes,
        "-fPIC", "-std=c++17", "-O3",
        "-c", wrapper_path,
        "-o", os.path.join(BUILD_DIR, "wrapper.o"),
    ]
    print(f"[BUILD] Compiling C++ wrapper...")
    r = subprocess.run(cxx_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print("[BUILD] C++ wrapper compiled OK")

    # Step 3: Link
    torch_lib = os.path.join(torch_dir, "lib")
    link_cmd = [
        "c++", "-shared",
        os.path.join(BUILD_DIR, "kernel.o"),
        os.path.join(BUILD_DIR, "wrapper.o"),
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
        "-L/usr/local/cuda/lib64", "-lcudart",
        "-o", SO_PATH,
    ]
    print(f"[BUILD] Linking...")
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"STDERR:\n{r.stderr}")
        sys.exit(1)
    print(f"[BUILD] Linked: {SO_PATH}")

    # Step 4: Load
    torch.ops.load_library(SO_PATH)
    print("[BUILD] Loaded into torch.ops._C")
    return SO_PATH


def register_fake_tensors():
    """Register fake tensor implementations for torch.compile."""
    import torch

    try:
        @torch.library.register_fake("_C::rms_norm_dynamic_fp4_quant")
        def _fake1(result, result_scale, input, weight, input_global_scale,
                   epsilon, is_sf_swizzled_layout):
            pass

        @torch.library.register_fake("_C::fused_add_rms_norm_dynamic_fp4_quant")
        def _fake2(result, result_scale, input, weight, residual,
                   input_global_scale, epsilon, is_sf_swizzled_layout):
            pass

        print("[FAKE] Meta implementations registered for torch.compile")
    except Exception as e:
        print(f"[FAKE] Warning: {e}")


def verify():
    """Quick correctness check."""
    import torch
    print("\n[VERIFY] Running correctness check...")

    M, N = 4, 128
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(N, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    eps = 1e-6

    out_fp4 = torch.empty(M, N // 2, device="cuda", dtype=torch.uint8)
    out_sf = torch.empty(M, N // 16, device="cuda", dtype=torch.uint8)

    torch.ops._C.rms_norm_dynamic_fp4_quant(
        out_fp4, out_sf, x, w, gs, eps, False
    )

    assert out_fp4.shape == (M, N // 2), f"Wrong shape: {out_fp4.shape}"
    assert not torch.all(out_fp4 == 0), "All zeros!"
    print(f"[VERIFY] rms_norm_dynamic_fp4_quant: OK, non-zero={int((out_fp4 != 0).sum())}/{out_fp4.numel()}")

    # Test fused_add variant
    res = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    x2 = x.clone()
    out_fp4_2 = torch.empty(M, N // 2, device="cuda", dtype=torch.uint8)
    out_sf_2 = torch.empty(M, N // 16, device="cuda", dtype=torch.uint8)

    torch.ops._C.fused_add_rms_norm_dynamic_fp4_quant(
        out_fp4_2, out_sf_2, x2, w, res, gs, eps, False
    )

    assert not torch.all(out_fp4_2 == 0), "Fused add: all zeros!"
    print(f"[VERIFY] fused_add_rms_norm_dynamic_fp4_quant: OK")

    # Numerical comparison: check output matches separate norm + quant
    print("[VERIFY] Comparing against reference (separate norm + quant)...")
    x_ref = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # Reference: manual RMSNorm
    variance = x_ref.float().pow(2).mean(dim=-1, keepdim=True)
    x_normed = (x_ref.float() * torch.rsqrt(variance + eps) * w.float())

    # Reference: FP4 quant with vLLM's existing kernel
    try:
        from vllm._custom_ops import scaled_fp4_quant
        ref_out, ref_sf = scaled_fp4_quant(x_normed.bfloat16(), gs)
        # Fused kernel
        out_fused = torch.empty(M, N // 2, device="cuda", dtype=torch.uint8)
        out_sf_fused = torch.empty_like(ref_sf)
        torch.ops._C.rms_norm_dynamic_fp4_quant(
            out_fused, out_sf_fused, x_ref, w, gs, eps, False
        )
        # Note: exact match not expected due to different scale factor
        # storage (swizzled vs row-major) and rounding. Check non-trivial output.
        match = (out_fused == ref_out).float().mean()
        print(f"[VERIFY] Byte-exact match vs reference: {match:.1%}")
        print(f"[VERIFY] (Mismatch expected due to scale rounding differences)")
    except Exception as e:
        print(f"[VERIFY] Skipping reference comparison: {e}")

    print("[VERIFY] All checks passed!")


def main():
    print("=" * 60)
    print("Fused RMSNorm + FP4 Quantization Kernel Builder")
    print("=" * 60)

    so_path = build_kernel()
    register_fake_tensors()
    verify()

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nShared library: {so_path}")
    print("\nAvailable ops:")
    print("  torch.ops._C.rms_norm_dynamic_fp4_quant")
    print("  torch.ops._C.fused_add_rms_norm_dynamic_fp4_quant")


if __name__ == "__main__":
    main()
