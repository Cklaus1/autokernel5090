"""Minimal test: load SM120 PTX via CUDA 13.1 driver JIT."""

import torch
_ = torch.zeros(1, device='cuda')

import ctypes

RTLD_NOLOAD = 4
cuda = ctypes.CDLL('libcuda.so.1', mode=RTLD_NOLOAD)

ver = ctypes.c_int()
cuda.cuDriverGetVersion(ctypes.byref(ver))
print(f"Driver: {ver.value}")

ctx = ctypes.c_void_p()
cuda.cuCtxGetCurrent(ctypes.byref(ctx))
print(f"Context: {ctx.value}")

# Test 1: Simple SM120 PTX
print("\n=== Simple SM120 PTX ===")
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
print(f"Load: {r}")

if r == 0:
    func = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func), module, b"simple_test")
    out = torch.zeros(1, device='cuda', dtype=torch.float32)
    out_ptr = ctypes.c_void_p(out.data_ptr())
    args = (ctypes.c_void_p * 1)(out_ptr)
    cuda.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, ctypes.c_void_p(0),
        ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()
    print(f"Result: {out[0].item()} (want 7.0)")

# Test 2: Block-scaled MMA raw PTX (hand-written, no NVRTC)
print("\n=== Block-scaled MMA PTX ===")
bs_ptx = b""".version 8.8
.target sm_120a
.address_size 64
.visible .entry bs_test(.param .u64 out_param) {
    .reg .u64 %rd<2>;
    .reg .f32 %f<10>;
    .reg .b32 %r<16>;

    // Zero accumulators
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;

    // Zero A operands (4 x u32 = 64 FP4 values)
    mov.b32 %r1, 0;
    mov.b32 %r2, 0;
    mov.b32 %r3, 0;
    mov.b32 %r4, 0;

    // Zero B operands (2 x u32 = 32 FP4 values)
    mov.b32 %r5, 0;
    mov.b32 %r6, 0;

    // Scale factors (E8M0: 127 = 2^0 = 1.0)
    mov.b32 %r7, 127;
    mov.b32 %r8, 127;

    // Block-scaled MMA: 16x8x64, FP4 E2M1 with E8M0 scales
    mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.ue8m0
        {%f5, %f6, %f7, %f8},
        {%r1, %r2, %r3, %r4},
        {%r5, %r6},
        {%f1, %f2, %f3, %f4},
        {%r7},
        {%r8};

    // Store results
    ld.param.u64 %rd1, [out_param];
    st.global.f32 [%rd1], %f5;
    st.global.f32 [%rd1+4], %f6;
    st.global.f32 [%rd1+8], %f7;
    st.global.f32 [%rd1+12], %f8;
    ret;
}
"""
module2 = ctypes.c_void_p()
r = cuda.cuModuleLoadData(ctypes.byref(module2), bs_ptx)
print(f"Load: {r}")
if r == 0:
    func2 = ctypes.c_void_p()
    cuda.cuModuleGetFunction(ctypes.byref(func2), module2, b"bs_test")
    out2 = torch.zeros(4, device='cuda', dtype=torch.float32)
    args2 = (ctypes.c_void_p * 1)(ctypes.c_void_p(out2.data_ptr()))
    cuda.cuLaunchKernel(func2, 1, 1, 1, 32, 1, 1, 0, ctypes.c_void_p(0),
        ctypes.cast(args2, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_void_p(0))
    cuda.cuCtxSynchronize()
    print(f"Output: {out2}")
    print("*** SM120 BLOCK-SCALED MMA WORKS! ***")
elif r == 218:
    print("CUDA_ERROR_INVALID_PTX")
elif r == 222:
    print("CUDA_ERROR_UNSUPPORTED_PTX_VERSION")
elif r == 209:
    print("CUDA_ERROR_NO_BINARY_FOR_GPU")
else:
    print(f"Error: {r}")
