#!/usr/bin/env python3
"""Probe: compile ONE FA3 sm90 source file with sm_120a target, capture errors."""
import os, subprocess, sys, torch

NVCC = "/usr/local/cuda-12.8/bin/nvcc"
FA_ROOT = "/tmp/flash-attention"
SRC = f"{FA_ROOT}/hopper/instantiations/flash_fwd_hdim128_bf16_sm90.cu"
CUTLASS_INC = f"{FA_ROOT}/csrc/cutlass/include"
HOPPER_INC = f"{FA_ROOT}/hopper"
TORCH_INC = os.path.join(os.path.dirname(torch.__file__), "include")
TORCH_INC_TOP = os.path.join(TORCH_INC, "torch", "csrc", "api", "include")
PY_INC = subprocess.check_output([sys.executable, "-c", "import sysconfig; print(sysconfig.get_path('include'))"], text=True).strip()

# Target arch
ARCH = sys.argv[1] if len(sys.argv) > 1 else "120a"

cmd = [
    NVCC, "-c",
    SRC,
    f"-I{HOPPER_INC}",
    f"-I{CUTLASS_INC}",
    f"-I{TORCH_INC}",
    f"-I{TORCH_INC_TOP}",
    f"-I{PY_INC}",
    "-O3", "-std=c++17",
    "--use_fast_math",
    "--resource-usage",
    "-lineinfo",
    "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
    "-DCUTLASS_ENABLE_GDC_FOR_SM90",
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
    "-DNDEBUG",
    "-DFLASHATTENTION_DISABLE_BACKWARD",
    "-DFLASHATTENTION_DISABLE_FP8",
    "-DFLASHATTENTION_DISABLE_HDIM64",
    "-DFLASHATTENTION_DISABLE_HDIM96",
    "-DFLASHATTENTION_DISABLE_HDIM192",
    "-DFLASHATTENTION_DISABLE_HDIM256",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF64",
    "-DFLASHATTENTION_DISABLE_HDIMDIFF192",
    "-DFLASHATTENTION_DISABLE_SM80",
    "-DFLASHATTENTION_DISABLE_SOFTCAP",
    "-DFLASHATTENTION_DISABLE_PACKGQA",
    "-DFLASHATTENTION_DISABLE_PAGEDKV",
    "-DFLASHATTENTION_DISABLE_APPENDKV",
    "-DFLASHATTENTION_DISABLE_SPLIT",
    "-DFLASHATTENTION_DISABLE_LOCAL",
    "-DFLASHATTENTION_DISABLE_VARLEN",
    "-DFLASHATTENTION_DISABLE_CLUSTER",
    "-DFLASHATTENTION_DISABLE_FP16",
    "-gencode", f"arch=compute_{ARCH},code=sm_{ARCH}",
    "--threads", "1",
    "-o", f"/tmp/fa3_probe_{ARCH}.o",
    "--ftemplate-backtrace-limit=0",
]
print("CMD:", " ".join(cmd), flush=True)
r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
print("RC:", r.returncode)
print("--- STDOUT ---")
print(r.stdout[-20000:])
print("--- STDERR ---")
print(r.stderr[-80000:])
