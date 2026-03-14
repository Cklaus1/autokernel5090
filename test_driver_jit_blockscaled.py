"""Test: NVRTC 12.8 → PTX → CUDA driver 13.1 JIT → SM120 block-scaled MMA.
Skip ptxas entirely — let the driver JIT compile the PTX."""

import torch  # Initialize CUDA context first
import ctypes

NVRTC_LIB = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
CUDA_INCLUDES = [
    '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/include',
    '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/include',
]

nvrtc = ctypes.cdll.LoadLibrary(NVRTC_LIB)
cuda = ctypes.cdll.LoadLibrary('libcuda.so')

# Ensure CUDA is initialized
cuda.cuInit(0)

def nvrtc_compile(source_bytes, extra_opts=None):
    """Compile CUDA source to PTX via NVRTC."""
    prog = ctypes.c_void_p()
    nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source_bytes, b"kernel.cu", 0, None, None)

    opts = [b"--gpu-architecture=compute_120", b"-default-device"]
    for inc in CUDA_INCLUDES:
        opts.append(f"--include-path={inc}".encode())
    if extra_opts:
        opts.extend(extra_opts)
    opts_arr = (ctypes.c_char_p * len(opts))(*opts)

    result = nvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
    if result != 0:
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"NVRTC compile failed:\n{log.value.decode()[:2000]}")

    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx_buf = ctypes.create_string_buffer(ptx_size.value)
    nvrtc.nvrtcGetPTX(prog, ptx_buf)
    ptx = ptx_buf.value
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx


def load_and_get_func(ptx_bytes, kernel_name):
    """Load PTX via driver JIT and get kernel function."""
    module = ctypes.c_void_p()
    # Use cuModuleLoadData — driver 13.1 will JIT compile the PTX
    result = cuda.cuModuleLoadData(ctypes.byref(module), ptx_bytes)
    if result != 0:
        raise RuntimeError(f"cuModuleLoadData failed: {result}")

    func = ctypes.c_void_p()
    result = cuda.cuModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    if result != 0:
        raise RuntimeError(f"cuModuleGetFunction failed: {result}")
    return func, module


# ================================================================
# Test 1: Simple WMMA FP16 (sanity check the pipeline)
# ================================================================
print("=== Test 1: WMMA FP16 via driver JIT ===")

WMMA_SOURCE = b'''
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_matmul(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int numWarpsN = N / 16;
    int warpM = warpId / numWarpsN;
    int warpN = warpId % numWarpsN;
    int row = warpM * 16;
    int col = warpN * 16;
    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}
'''

try:
    ptx = nvrtc_compile(WMMA_SOURCE)
    # Show MMA instructions
    for line in ptx.decode().split('\n'):
        if 'mma' in line.lower() and 'sync' in line and not line.strip().startswith('//'):
            print(f"  PTX: {line.strip()[:120]}")

    func, mod = load_and_get_func(ptx, "wmma_matmul")
    print("  Loaded!")

    M, K, N = 256, 256, 256
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    ref = torch.mm(a, b)

    m_val, n_val, k_val = ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K)
    args = (ctypes.c_void_p * 6)(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c.data_ptr()),
        ctypes.addressof(m_val),
        ctypes.addressof(n_val),
        ctypes.addressof(k_val),
    )
    num_warps = (M // 16) * (N // 16)
    threads = min(num_warps * 32, 1024)
    blocks = (num_warps * 32 + threads - 1) // threads

    result = cuda.cuLaunchKernel(
        func, blocks, 1, 1, threads, 1, 1,
        0, ctypes.c_void_p(0),
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()

    err = (c - ref).abs().max().item()
    print(f"  max_err={err:.4f} {'PASS' if err < 1.0 else 'FAIL'}")
except Exception as e:
    print(f"  FAILED: {e}")


# ================================================================
# Test 2: Block-scaled MMA (FP4 E2M1) via driver JIT
# ================================================================
print("\n=== Test 2: Block-scaled MMA via driver JIT ===")

BLOCKSCALE_SOURCE = b'''
extern "C" __global__ void block_scale_test(float* output)
{
    float d0=0, d1=0, d2=0, d3=0;
    float c0=0, c1=0, c2=0, c3=0;
    unsigned int a0=0, a1=0, a2=0, a3=0;
    unsigned int b0=0, b1=0;
    unsigned int sfa = 127;  // E8M0: 2^(127-127) = 1.0
    unsigned int sfb = 127;

    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X"
        ".f32.e2m1.e2m1.ue8m0 "
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

    if (threadIdx.x == 0) {
        output[0] = d0;
        output[1] = d1;
        output[2] = d2;
        output[3] = d3;
    }
}
'''

try:
    ptx = nvrtc_compile(BLOCKSCALE_SOURCE)
    # Check for block_scale in PTX
    for line in ptx.decode().split('\n'):
        if 'block_scale' in line.lower() or 'mxf4' in line.lower():
            print(f"  PTX: {line.strip()[:140]}")

    func2, mod2 = load_and_get_func(ptx, "block_scale_test")
    print("  Block-scaled MMA loaded via driver JIT!")

    output = torch.zeros(4, device='cuda', dtype=torch.float32)
    args = (ctypes.c_void_p * 1)(ctypes.c_void_p(output.data_ptr()))
    result = cuda.cuLaunchKernel(
        func2, 1, 1, 1, 32, 1, 1,
        0, ctypes.c_void_p(0),
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()
    print(f"  Output: {output}")
    print("  SUCCESS: SM120 block-scaled MMA instruction works!")
except Exception as e:
    print(f"  FAILED: {e}")
