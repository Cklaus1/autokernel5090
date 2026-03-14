"""Test: NVRTC 12.8 → PTX → CUDA driver 13.1 JIT → SM120 block-scaled MMA.
Fix: proper CUDA context management."""

import torch  # Initialize CUDA context
_ = torch.zeros(1, device='cuda')  # Force CUDA init

import ctypes

NVRTC_LIB = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
CUDA_INCLUDES = [
    '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/include',
    '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/include',
]

nvrtc = ctypes.cdll.LoadLibrary(NVRTC_LIB)
cuda = ctypes.cdll.LoadLibrary('libcuda.so')

# Check CUDA driver context
ctx = ctypes.c_void_p()
r = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
print(f"cuCtxGetCurrent: result={r}, ctx={ctx.value}")

if ctx.value is None or ctx.value == 0:
    # No context — create one
    device = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(device), 0)
    r = cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
    print(f"cuCtxCreate: result={r}, ctx={ctx.value}")

# Check driver version
driver_ver = ctypes.c_int()
cuda.cuDriverGetVersion(ctypes.byref(driver_ver))
print(f"CUDA Driver version: {driver_ver.value}")


def nvrtc_compile(source_bytes, extra_opts=None):
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


def load_ptx(ptx_bytes, kernel_name):
    module = ctypes.c_void_p()
    # Try cuModuleLoadDataEx for better error info
    jit_log = ctypes.create_string_buffer(4096)
    jit_err = ctypes.create_string_buffer(4096)

    CU_JIT_INFO_LOG_BUFFER = 0
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 1
    CU_JIT_ERROR_LOG_BUFFER = 2
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 3

    opt_keys = (ctypes.c_uint * 4)(
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    )
    opt_vals = (ctypes.c_void_p * 4)(
        ctypes.cast(jit_log, ctypes.c_void_p),
        ctypes.c_void_p(4096),
        ctypes.cast(jit_err, ctypes.c_void_p),
        ctypes.c_void_p(4096),
    )

    result = cuda.cuModuleLoadDataEx(
        ctypes.byref(module), ptx_bytes,
        4, opt_keys, opt_vals)

    if result != 0:
        err_msg = jit_err.value.decode('utf-8', errors='replace')
        info_msg = jit_log.value.decode('utf-8', errors='replace')
        raise RuntimeError(
            f"cuModuleLoadDataEx failed: {result}\n"
            f"  Error log: {err_msg[:500]}\n"
            f"  Info log: {info_msg[:500]}")

    func = ctypes.c_void_p()
    result = cuda.cuModuleGetFunction(ctypes.byref(func), module, kernel_name.encode())
    if result != 0:
        raise RuntimeError(f"cuModuleGetFunction failed: {result}")
    return func, module


# ================================================================
# Test 1: Simple kernel (absolute minimum)
# ================================================================
print("\n=== Test 1: Simplest possible kernel ===")

SIMPLE_SOURCE = b'''
extern "C" __global__ void simple_add(float* out, float a, float b) {
    if (threadIdx.x == 0) out[0] = a + b;
}
'''

try:
    ptx = nvrtc_compile(SIMPLE_SOURCE)
    print(f"  PTX size: {len(ptx)} bytes")
    # Show target line
    for line in ptx.decode().split('\n'):
        if '.target' in line or '.address_size' in line:
            print(f"  {line.strip()}")

    func, mod = load_ptx(ptx, "simple_add")
    print("  Loaded!")

    out = torch.zeros(1, device='cuda', dtype=torch.float32)
    a_val = ctypes.c_float(3.0)
    b_val = ctypes.c_float(4.0)
    args = (ctypes.c_void_p * 3)(
        ctypes.c_void_p(out.data_ptr()),
        ctypes.addressof(a_val),
        ctypes.addressof(b_val),
    )
    r = cuda.cuLaunchKernel(
        func, 1, 1, 1, 32, 1, 1, 0, ctypes.c_void_p(0),
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()
    print(f"  Result: {out[0].item()} (expected 7.0) {'PASS' if abs(out[0].item() - 7.0) < 0.01 else 'FAIL'}")
except Exception as e:
    print(f"  FAILED: {e}")


# ================================================================
# Test 2: Block-scaled MMA
# ================================================================
print("\n=== Test 2: Block-scaled MMA ===")

BLOCKSCALE_SOURCE = b'''
extern "C" __global__ void block_scale_test(float* output)
{
    float d0=0, d1=0, d2=0, d3=0;
    float c0=0, c1=0, c2=0, c3=0;
    unsigned int a0=0, a1=0, a2=0, a3=0;
    unsigned int b0=0, b1=0;
    unsigned int sfa = 127;
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
    for line in ptx.decode().split('\n'):
        if 'mma' in line and 'block_scale' in line:
            print(f"  PTX: {line.strip()[:140]}")

    func2, mod2 = load_ptx(ptx, "block_scale_test")
    print("  Block-scaled MMA loaded!")

    output = torch.zeros(4, device='cuda', dtype=torch.float32)
    args = (ctypes.c_void_p * 1)(ctypes.c_void_p(output.data_ptr()))
    r = cuda.cuLaunchKernel(
        func2, 1, 1, 1, 32, 1, 1, 0, ctypes.c_void_p(0),
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()
    print(f"  Output: {output}")
    print("  SUCCESS!")
except Exception as e:
    print(f"  FAILED: {e}")
