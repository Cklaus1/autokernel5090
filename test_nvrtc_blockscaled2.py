"""Test SM120 block-scaled MMA via NVRTC 12.8 + CUDA driver API."""

import torch  # Initialize CUDA context
import ctypes

NVRTC_LIB = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12'
CUDA_INCLUDE = '/root/projects/autokernel/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/include'

nvrtc = ctypes.cdll.LoadLibrary(NVRTC_LIB)
cuda = ctypes.cdll.LoadLibrary('libcuda.so')

# Block-scaled MMA kernel that actually executes the instruction
BLOCK_SCALED_SOURCE = r"""
// Test block-scaled MMA on SM120
// Instruction: mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.ue8m0
extern "C" __global__ void test_block_scale(float* output)
{
    unsigned int a0=0, a1=0, a2=0, a3=0;
    unsigned int b0=0, b1=0;
    float c0=0, c1=0, c2=0, c3=0;
    float d0, d1, d2, d3;
    unsigned int sfa=127;  // E8M0 scale = 2^(127-127) = 1.0
    unsigned int sfb=127;

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

    if (threadIdx.x == 0) {
        output[0] = d0;
        output[1] = d1;
        output[2] = d2;
        output[3] = d3;
    }
}
"""

print("=== Compiling block-scaled MMA kernel with NVRTC 12.8 ===")

prog = ctypes.c_void_p()
result = nvrtc.nvrtcCreateProgram(
    ctypes.byref(prog),
    BLOCK_SCALED_SOURCE.encode('utf-8'),
    b"block_scaled.cu",
    0, None, None,
)

options = [
    b"--gpu-architecture=sm_120",
    b"-default-device",
    f"--include-path={CUDA_INCLUDE}".encode('utf-8'),
]
options_arr = (ctypes.c_char_p * len(options))(*options)
result = nvrtc.nvrtcCompileProgram(prog, len(options), options_arr)

if result != 0:
    log_size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
    log = ctypes.create_string_buffer(log_size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    print(f"Compile FAILED ({result}):\n{log.value.decode()}")
else:
    print("Compile SUCCESS!")

    # Get PTX
    ptx_size = ctypes.c_size_t()
    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size))
    ptx = ctypes.create_string_buffer(ptx_size.value)
    nvrtc.nvrtcGetPTX(prog, ptx)
    ptx_str = ptx.value.decode()

    # Check for block_scale instruction
    for line in ptx_str.split('\n'):
        if 'mma' in line.lower() and 'block_scale' in line.lower():
            print(f"  PTX MMA: {line.strip()}")

    # Load as CUDA module (torch already initialized CUDA)
    module = ctypes.c_void_p()
    result = cuda.cuModuleLoadData(ctypes.byref(module), ptx.value)
    print(f"cuModuleLoadData: {result}")

    if result == 0:
        func = ctypes.c_void_p()
        result = cuda.cuModuleGetFunction(ctypes.byref(func), module, b"test_block_scale")
        print(f"cuModuleGetFunction: {result}")

        if result == 0:
            # Allocate output
            output = torch.zeros(4, device='cuda', dtype=torch.float32)
            output_ptr = ctypes.c_void_p(output.data_ptr())

            # Launch: 1 block, 32 threads (1 warp)
            args = (ctypes.c_void_p * 1)(output_ptr)
            args_ptr = ctypes.cast(args, ctypes.POINTER(ctypes.c_void_p))

            result = cuda.cuLaunchKernel(
                func,
                1, 1, 1,   # grid
                32, 1, 1,  # block (1 warp)
                0,         # shared mem
                ctypes.c_void_p(0),  # stream (default)
                args_ptr,
                ctypes.c_void_p(0),  # extra
            )
            print(f"cuLaunchKernel: {result}")

            cuda.cuCtxSynchronize()
            print(f"Output: {output}")
            print("\nSUCCESS: SM120 block-scaled MMA instruction works!")
    else:
        print(f"Module load FAILED (error {result})")
        # Error 222 = CUDA_ERROR_UNSUPPORTED_PTX_VERSION
        # Error 218 = CUDA_ERROR_INVALID_PTX
        if result == 222:
            print("  CUDA_ERROR_UNSUPPORTED_PTX_VERSION - driver too old for this PTX")
        elif result == 218:
            print("  CUDA_ERROR_INVALID_PTX")

nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
