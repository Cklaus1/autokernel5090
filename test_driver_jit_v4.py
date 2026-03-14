"""Test: Use WSL's libcuda.so (CUDA 13.1) for driver JIT."""

import ctypes, os

# Load WSL libcuda FIRST, before torch
cuda = ctypes.CDLL('/usr/lib/wsl/lib/libcuda.so.1', mode=ctypes.RTLD_GLOBAL)

ver = ctypes.c_int()
cuda.cuDriverGetVersion(ctypes.byref(ver))
print(f"CUDA Driver version: {ver.value}")

r = cuda.cuInit(0)
print(f"cuInit: {r}")

# Get device
device = ctypes.c_int()
r = cuda.cuDeviceGet(ctypes.byref(device), 0)
print(f"cuDeviceGet: {r}, device={device.value}")

# Get device name
name = ctypes.create_string_buffer(256)
cuda.cuDeviceGetName(name, 256, device)
print(f"Device: {name.value.decode()}")

# Create context
ctx = ctypes.c_void_p()
r = cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device)
print(f"cuCtxCreate: {r}")

if r != 0:
    print("Failed to create context, exiting")
    exit(1)

# Test: load simple PTX
simple_ptx = b""".version 8.8
.target sm_120
.address_size 64
.visible .entry simple_test(.param .u64 out_param) {
    .reg .u64 %rd<2>;
    .reg .f32 %f<2>;
    ld.param.u64 %rd0, [out_param];
    mov.f32 %f0, 0f40E00000;
    st.global.f32 [%rd0], %f0;
    ret;
}
"""

module = ctypes.c_void_p()
r = cuda.cuModuleLoadData(ctypes.byref(module), simple_ptx)
print(f"cuModuleLoadData (sm_120 PTX): {r}")

if r == 0:
    # Allocate device memory
    d_out = ctypes.c_void_p()
    cuda.cuMemAlloc_v2(ctypes.byref(d_out), 4)

    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), module, b"simple_test")

    args = (ctypes.c_void_p * 1)(d_out)
    r = cuda.cuLaunchKernel(
        func, 1, 1, 1, 1, 1, 1, 0, ctypes.c_void_p(0),
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()

    result = ctypes.c_float()
    cuda.cuMemcpyDtoH_v2(ctypes.byref(result), d_out, 4)
    print(f"Simple kernel result: {result.value} (expected 7.0)")
    cuda.cuMemFree_v2(d_out)

    # Now try block-scaled MMA PTX
    print("\n=== Block-scaled MMA ===")
    NVRTC_LIB = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
    nvrtc = ctypes.cdll.LoadLibrary(NVRTC_LIB)

    BLOCKSCALE_SOURCE = b'''
extern "C" __global__ void bs_test(float* output)
{
    float d0=0, d1=0, d2=0, d3=0;
    float c0=0, c1=0, c2=0, c3=0;
    unsigned int a0=0, a1=0, a2=0, a3=0;
    unsigned int b0=0, b1=0;
    unsigned int sfa = 127, sfb = 127;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X"
        ".f32.e2m1.e2m1.ue8m0 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13}, "
        "{%14}, {%15};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3),
          "r"(sfa), "r"(sfb)
    );
    if (threadIdx.x == 0) {
        output[0] = d0; output[1] = d1; output[2] = d2; output[3] = d3;
    }
}
'''
    prog = ctypes.c_void_p()
    nvrtc.nvrtcCreateProgram(ctypes.byref(prog), BLOCKSCALE_SOURCE, b"k.cu", 0, None, None)
    opts = [b"--gpu-architecture=compute_120", b"-default-device"]
    opts_arr = (ctypes.c_char_p * len(opts))(*opts)
    r2 = nvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
    if r2 != 0:
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        log = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, log)
        print(f"NVRTC failed: {log.value.decode()[:500]}")
    else:
        ptx_size = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        nvrtc.nvrtcGetPTX(prog, ptx_buf)
        ptx = ptx_buf.value

        # Show the block-scaled instruction
        for line in ptx.decode().split('\n'):
            if 'mma' in line and 'block_scale' in line:
                print(f"  PTX: {line.strip()[:140]}")

        module2 = ctypes.c_void_p()
        r3 = cuda.cuModuleLoadData(ctypes.byref(module2), ptx)
        print(f"  cuModuleLoadData: {r3}")
        if r3 == 0:
            func2 = ctypes.c_void_p()
            cuda.cuModuleGetFunction(ctypes.byref(func2), module2, b"bs_test")
            d_out2 = ctypes.c_void_p()
            cuda.cuMemAlloc_v2(ctypes.byref(d_out2), 16)
            args2 = (ctypes.c_void_p * 1)(d_out2)
            cuda.cuLaunchKernel(
                func2, 1, 1, 1, 32, 1, 1, 0, ctypes.c_void_p(0),
                ctypes.cast(args2, ctypes.POINTER(ctypes.c_void_p)),
                ctypes.c_void_p(0))
            cuda.cuCtxSynchronize()
            result_buf = (ctypes.c_float * 4)()
            cuda.cuMemcpyDtoH_v2(result_buf, d_out2, 16)
            print(f"  Output: [{result_buf[0]}, {result_buf[1]}, {result_buf[2]}, {result_buf[3]}]")
            print("  *** SM120 BLOCK-SCALED MMA WORKS! ***")
            cuda.cuMemFree_v2(d_out2)
        elif r3 == 218:
            print("  CUDA_ERROR_INVALID_PTX")
        elif r3 == 222:
            print("  CUDA_ERROR_UNSUPPORTED_PTX_VERSION")
        else:
            print(f"  Error code: {r3}")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
else:
    print(f"Failed to load simple PTX. Error codes: 218=INVALID_PTX, 222=UNSUPPORTED_PTX_VERSION")
