#!/usr/bin/env python3
"""Patch enable_sm90 to include __CUDA_ARCH__ == 1200, then re-compile and capture real errors."""
import os, subprocess, sys, shutil

UTILS = "/tmp/flash-attention/hopper/utils.h"
BACKUP = UTILS + ".bak"

# 1) backup
if not os.path.exists(BACKUP):
    shutil.copy(UTILS, BACKUP)
with open(UTILS) as f:
    src = f.read()
orig = src
# 2) patch
patched = src.replace(
    "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)\n        Kernel::operator()(std::forward<Args>(args)...);",
    "#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ == 900) || (__CUDA_ARCH__ == 1200))\n        Kernel::operator()(std::forward<Args>(args)...);",
)
assert patched != orig, "patch did not apply"
with open(UTILS, "w") as f:
    f.write(patched)
print("Patched enable_sm90 to accept sm_120 as well.")

# 3) now compile
import torch
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
FA_ROOT = "/tmp/flash-attention"
SRC = f"{FA_ROOT}/hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu"
CUTLASS_INC = f"{FA_ROOT}/csrc/cutlass/include"
HOPPER_INC = f"{FA_ROOT}/hopper"
TORCH_INC = os.path.join(os.path.dirname(torch.__file__), "include")
TORCH_INC_TOP = os.path.join(TORCH_INC, "torch", "csrc", "api", "include")
PY_INC = subprocess.check_output([sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"], text=True).strip()

ARCH = "120a"
cmd = [
    NVCC, "-c", SRC,
    f"-I{HOPPER_INC}", f"-I{CUTLASS_INC}", f"-I{TORCH_INC}", f"-I{TORCH_INC_TOP}", f"-I{PY_INC}",
    "-O3", "-std=c++17", "--use_fast_math",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED", "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG",
    "-DFLASHATTENTION_DISABLE_BACKWARD", "-DFLASHATTENTION_DISABLE_FP8",
    "-DFLASHATTENTION_DISABLE_HDIM64", "-DFLASHATTENTION_DISABLE_HDIM96",
    "-DFLASHATTENTION_DISABLE_HDIM192", "-DFLASHATTENTION_DISABLE_HDIM256",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF64", "-DFLASHATTENTION_DISABLE_HDIMDIFF192",
    "-DFLASHATTENTION_DISABLE_SM80", "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DFLASHATTENTION_DISABLE_PACKGQA", "-DFLASHATTENTION_DISABLE_PAGEDKV",
    "-DFLASHATTENTION_DISABLE_APPENDKV", "-DFLASHATTENTION_DISABLE_SPLIT",
    "-DFLASHATTENTION_DISABLE_LOCAL", "-DFLASHATTENTION_DISABLE_VARLEN",
    "-DFLASHATTENTION_DISABLE_CLUSTER", "-DFLASHATTENTION_DISABLE_FP16",
    "-gencode", f"arch=compute_{ARCH},code=sm_{ARCH}",
    "--threads", "1",
    "-o", f"/tmp/fa3_probe_{ARCH}_patched.o",
    "--ftemplate-backtrace-limit=1",
    "--expt-relaxed-constexpr",
]
print("CMD:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
print("RC:", r.returncode)

# Filter stderr to errors only
lines = (r.stderr or "").splitlines()
errs = [l for l in lines if 'error' in l.lower() or 'INVALID_CONTROL_PATH' in l or 'CUTE_STATIC_ASSERT' in l or 'CUTLASS_STATIC_ASSERT' in l]
print(f"=== ERROR-ish lines ({len(errs)}) ===")
for l in errs[:80]:
    print(l)
print("=== TAIL STDERR ===")
print("\n".join(lines[-120:]))
