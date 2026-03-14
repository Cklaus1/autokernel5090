"""Use the already-loaded libcuda.so via RTLD_NOLOAD after torch init."""

import torch
_ = torch.zeros(1, device='cuda')  # Force CUDA init with real driver

import ctypes

# Get the already-loaded libcuda.so (torch loaded the real WSL driver)
# The actual driver is at /usr/lib/wsl/drivers/nvhdci.inf_amd64.../libcuda.so.1.1
RTLD_NOLOAD = 4
cuda = ctypes.CDLL('libcuda.so.1', mode=RTLD_NOLOAD)

ver = ctypes.c_int()
r = cuda.cuDriverGetVersion(ctypes.byref(ver))
print(f"Driver version: {ver.value}, result: {r}")

# Check context (torch should have already set one up)
ctx = ctypes.c_void_p()
r = cuda.cuCtxGetCurrent(ctypes.byref(ctx))
print(f"cuCtxGetCurrent: result={r}, ctx={ctx.value}")

# Try loading sm_120 PTX
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
print(f"cuModuleLoadData: {r}")

if r == 0:
    print("SM120 PTX loading works!")
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), module, b"simple_test")

    out = torch.zeros(1, device='cuda', dtype=torch.float32)
    out_ptr = ctypes.c_void_p(out.data_ptr())
    args = (ctypes.c_void_p * 1)(out_ptr)
    cuda.cuLaunchKernel(
        func, 1, 1, 1, 1, 1, 1, 0, ctypes.c_void_p(0),
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()
    print(f"Result: {out[0].item()} (expected 7.0)")

    # Block-scaled MMA
    print("\n=== Block-scaled MMA ===")
    NVRTC_LIB = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
    nvrtc = ctypes.cdll.LoadLibrary(NVRTC_LIB)

    BS_SRC = b'''
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
    nvrtc.nvrtcCreateProgram(ctypes.byref(prog), BS_SRC, b"k.cu", 0, None, None)
    opts = [b"--gpu-architecture=compute_120", b"-default-device"]
    opts_arr = (ctypes.c_char_p * len(opts))(*opts)
    r2 = nvrtc.nvrtcCompileProgram(prog, len(opts), opts_arr)
    if r2 != 0:
        ls = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(ls))
        lg = ctypes.create_string_buffer(ls.value)
        nvrtc.nvrtcGetProgramLog(prog, lg)
        print(f"NVRTC fail: {lg.value.decode()[:500]}")
    else:
        ps = ctypes.c_size_t()
        nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ps))
        pb = ctypes.create_string_buffer(ps.value)
        nvrtc.nvrtcGetPTX(prog, pb)
        ptx = pb.value
        for line in ptx.decode().split('\n'):
            if 'block_scale' in line:
                print(f"  PTX: {line.strip()[:140]}")
        m2 = ctypes.c_void_p()
        r3 = cuda.cuModuleLoadData(ctypes.byref(m2), ptx)
        print(f"  Load: {r3}")
        if r3 == 0:
            f2 = ctypes.c_void_p()
            cuda.cuModuleGetFunction(ctypes.byref(f2), m2, b"bs_test")
            out2 = torch.zeros(4, device='cuda', dtype=torch.float32)
            a2 = (ctypes.c_void_p * 1)(ctypes.c_void_p(out2.data_ptr()))
            cuda.cuLaunchKernel(f2, 1, 1, 1, 32, 1, 1, 0, ctypes.c_void_p(0),
                ctypes.cast(a2, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_void_p(0))
            cuda.cuCtxSynchronize()
            print(f"  Output: {out2}")
            print("  *** BLOCK-SCALED MMA WORKS! ***")
        else:
            print(f"  Error {r3}: ", end="")
            if r3 == 218: print("INVALID_PTX")
            elif r3 == 222: print("UNSUPPORTED_PTX_VERSION")
            else: print("unknown")
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
else:
    print(f"PTX load failed: {r}")
    if r == 218: print("INVALID_PTX")
    elif r == 222: print("UNSUPPORTED_PTX_VERSION")
    elif r == 3: print("NOT_INITIALIZED - driver API context issue")
