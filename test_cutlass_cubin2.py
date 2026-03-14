"""Extract cubin from CUTLASS DSL - try F16 MMA first, then MXF4."""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass.cute as cute
from cutlass import Float16, Float32, Float4E2M1FN, Float8E8M0FNU
from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op, MmaMXF4Op

# Test 1: F16 MMA - this should definitely work
@cute.kernel
def f16_kernel():
    mma_op = MmaF16BF16Op(
        ab_dtype=Float16,
        acc_dtype=Float16,
        shape_mnk=(16, 8, 16),
    )
    mma_atom = cute.make_mma_atom(mma_op)
    tiled_mma = cute.make_tiled_mma(mma_atom)

    # Dispatch rule (V) x (V) => (V): sizes must be equal
    tA = cute.make_rmem_tensor(cute.make_layout((8,)), Float16)
    tB = cute.make_rmem_tensor(cute.make_layout((8,)), Float16)
    tC = cute.make_rmem_tensor(cute.make_layout((4,)), Float16)
    tD = cute.make_rmem_tensor(cute.make_layout((4,)), Float16)

    cute.gemm(tiled_mma, tD, tA, tB, tC)
    return

@cute.jit
def host_f16():
    f16_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1])


# Test 2: Empty MXF4 kernel (no gemm call, just atom creation)
@cute.kernel
def mxf4_empty():
    mma_op = MmaMXF4Op(
        ab_dtype=Float4E2M1FN,
        acc_dtype=Float32,
        sf_type=Float8E8M0FNU,
    )
    mma_atom = cute.make_mma_atom(mma_op)
    tiled_mma = cute.make_tiled_mma(mma_atom)
    return

@cute.jit
def host_mxf4_empty():
    mxf4_empty().launch(grid=[1, 1, 1], block=[32, 1, 1])


import tempfile, glob

# Test F16
print("=== F16 MMA Cubin ===")
with tempfile.TemporaryDirectory() as tmp:
    try:
        compiled = cute.compile(host_f16, options=f"--dump-dir={tmp} --keep-ptx --keep-cubin")
        print("Compiled!")
        for f in sorted(glob.glob(f"{tmp}/**/*", recursive=True)):
            if os.path.isfile(f):
                sz = os.path.getsize(f)
                ext = os.path.splitext(f)[1]
                print(f"  {os.path.basename(f)} ({sz} bytes)")
                if ext == '.ptx':
                    with open(f) as fp:
                        for line in fp:
                            if 'mma' in line.lower() and 'sync' in line:
                                print(f"    {line.strip()[:140]}")
                            if '.target' in line:
                                print(f"    {line.strip()}")
                elif ext == '.cubin':
                    with open(f, 'rb') as fp:
                        cubin = fp.read()
                    print(f"    CUBIN extracted! {len(cubin)} bytes")
                    # Try loading via driver
                    import torch
                    _ = torch.zeros(1, device='cuda')
                    import ctypes
                    cuda_drv = ctypes.CDLL('libcuda.so.1', mode=4)
                    module = ctypes.c_void_p()
                    r = cuda_drv.cuModuleLoadData(ctypes.byref(module), cubin)
                    print(f"    cuModuleLoadData: {r}")
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")

# Test MXF4 empty
print("\n=== MXF4 Empty Kernel Cubin ===")
with tempfile.TemporaryDirectory() as tmp:
    try:
        compiled = cute.compile(host_mxf4_empty, options=f"--dump-dir={tmp} --keep-ptx --keep-cubin")
        print("Compiled!")
        for f in sorted(glob.glob(f"{tmp}/**/*", recursive=True)):
            if os.path.isfile(f):
                sz = os.path.getsize(f)
                ext = os.path.splitext(f)[1]
                print(f"  {os.path.basename(f)} ({sz} bytes)")
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")
