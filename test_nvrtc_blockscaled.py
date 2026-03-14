"""Test SM120 block-scaled MMA via NVRTC compilation.
First step: compile a simple FP16 matmul via NVRTC to verify the pipeline works."""

import torch
import ctypes
import tempfile
import os
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

# Load NVRTC 12.8 from venv
NVRTC_LIB = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
NVRTC_BUILTINS = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.8'
PTXAS_BIN = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin/ptxas'

nvrtc = ctypes.cdll.LoadLibrary(NVRTC_LIB)

# NVRTC error checking
def check_nvrtc(result, func_name=""):
    if result != 0:
        log_size = ctypes.c_size_t()
        # Try to get error string
        raise RuntimeError(f"NVRTC {func_name} failed with code {result}")

# Simple FP16 matmul kernel using WMMA (to verify NVRTC pipeline)
CUDA_SOURCE = r"""
#include <cuda_fp16.h>

// Simple naive FP16 matmul for pipeline testing
extern "C" __global__ void simple_matmul(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}
"""

# Compile with NVRTC
print("=== Compiling with NVRTC 12.8 ===")

# Create program
prog = ctypes.c_void_p()
source = CUDA_SOURCE.encode('utf-8')
result = nvrtc.nvrtcCreateProgram(
    ctypes.byref(prog),
    source,
    b"matmul.cu",
    0,  # num headers
    None,  # headers
    None,  # include names
)
print(f"nvrtcCreateProgram: {result}")

# Compile for SM120
options = [b"--gpu-architecture=sm_120", b"-default-device"]
options_arr = (ctypes.c_char_p * len(options))(*options)
result = nvrtc.nvrtcCompileProgram(prog, len(options), options_arr)
print(f"nvrtcCompileProgram: {result}")

if result != 0:
    # Get compilation log
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    print(f"Compilation log:\n{log.value.decode()}")
else:
    # Get PTX
    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    nvrtc.nvrtcGetPTX(prog, ptx)
    ptx_str = ptx.value.decode()
    print(f"PTX generated: {len(ptx_str)} bytes")

    # Show first few lines
    lines = ptx_str.split('\n')
    print(f"PTX target: {[l for l in lines[:10] if 'target' in l.lower()]}")

    # Load module via CUDA driver API
    cuda = ctypes.cdll.LoadLibrary('libcuda.so')

    module = ctypes.c_void_p()
    result = cuda.cuModuleLoadData(ctypes.byref(module), ptx.value)
    print(f"cuModuleLoadData: {result}")

    if result == 0:
        # Get function
        func = ctypes.c_void_p()
        result = cuda.cuModuleGetFunction(ctypes.byref(func), module, b"simple_matmul")
        print(f"cuModuleGetFunction: {result}")

        if result == 0:
            print("SUCCESS: NVRTC 12.8 -> SM120 PTX -> CUDA module loaded!")

            # Now let's try a more interesting kernel: FP16 tiled matmul using mma.sync
            print("\n=== Now testing mma.sync.aligned FP16 matmul ===")
    else:
        print(f"Failed to load module (error {result})")

nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

# Now try the real thing: block-scaled MMA
print("\n=== Testing block-scaled MMA PTX instruction ===")

BLOCK_SCALED_SOURCE = r"""
// Test if mma.sync.aligned.block_scale PTX instruction assembles on SM120
extern "C" __global__ void test_block_scale_avail()
{
    // Just check if the PTX assembler accepts the instruction
    // This is a no-op kernel that uses inline PTX
    unsigned int tid = threadIdx.x;
    if (tid >= 1024) {  // Never actually executes
        // The mma.sync.aligned.block_scale instruction:
        // mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.ue8m0
        // {d0, d1, d2, d3}, {a0, a1, a2, a3}, {b0, b1}, {c0, c1, c2, c3}, {sfa}, {sfb}
        unsigned int a0=0, a1=0, a2=0, a3=0;
        unsigned int b0=0, b1=0;
        float c0=0, c1=0, c2=0, c3=0;
        float d0, d1, d2, d3;
        unsigned int sfa=0, sfb=0;

        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.ue8m0 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13}, "
            "{%14}, "
            "{%15};"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3),
              "r"(sfa), "r"(sfb)
        );
    }
}
"""

prog2 = ctypes.c_void_p()
source2 = BLOCK_SCALED_SOURCE.encode('utf-8')
result = nvrtc.nvrtcCreateProgram(
    ctypes.byref(prog2),
    source2,
    b"block_scaled.cu",
    0, None, None,
)

options2 = [b"--gpu-architecture=sm_120", b"-default-device"]
options_arr2 = (ctypes.c_char_p * len(options2))(*options2)
result = nvrtc.nvrtcCompileProgram(prog2, len(options2), options_arr2)
print(f"Block-scaled compile: {result}")

if result != 0:
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog2, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog2, log)
    print(f"Compilation log:\n{log.value.decode()}")
else:
    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(prog2, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    nvrtc.nvrtcGetPTX(prog2, ptx)
    ptx_str = ptx.value.decode()

    # Check for block_scale instruction in PTX
    block_scale_lines = [l.strip() for l in ptx_str.split('\n') if 'block_scale' in l or 'mxf4' in l]
    if block_scale_lines:
        print(f"Block-scaled MMA instructions found in PTX ({len(block_scale_lines)}):")
        for l in block_scale_lines[:5]:
            print(f"  {l}")
    else:
        print("No block_scale instructions found in PTX")

    # Try loading
    cuda = ctypes.cdll.LoadLibrary('libcuda.so')
    module2 = ctypes.c_void_p()
    result = cuda.cuModuleLoadData(ctypes.byref(module2), ptx.value)
    if result == 0:
        print("SUCCESS: Block-scaled MMA kernel loaded on SM120!")
    else:
        print(f"Module load failed: {result}")

nvrtc.nvrtcDestroyProgram(ctypes.byref(prog2))
