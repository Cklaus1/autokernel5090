"""Test CUTLASS DSL SM120 block-scaled MMA GEMM - inspect generated PTX."""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import Float4E2M1FN, Float8E8M0FNU, Float32
from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op, MmaF16BF16Op
from cutlass import Float16

# First: a simple F16 MMA to verify the pipeline works end-to-end
@cute.kernel
def f16_mma_kernel():
    mma_op = MmaF16BF16Op(
        ab_dtype=Float16,
        acc_dtype=Float16,
        shape_mnk=(16, 8, 16),
    )
    mma_atom = cute.make_mma_atom(mma_op)
    tiled_mma = cute.make_tiled_mma(mma_atom)

    tA = cute.make_rmem_tensor(cute.make_layout((8,)), Float16)
    tB = cute.make_rmem_tensor(cute.make_layout((4,)), Float16)
    tC = cute.make_rmem_tensor(cute.make_layout((4,)), Float16)
    tD = cute.make_rmem_tensor(cute.make_layout((4,)), Float16)

    cute.gemm(tiled_mma, tD, tA, tB, tC)
    return

@cute.jit
def host_fn_f16():
    f16_mma_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1])


# Second: SM120 block-scaled MXF4 MMA
@cute.kernel
def mxf4_mma_kernel():
    mma_op = MmaMXF4Op(
        ab_dtype=Float4E2M1FN,
        acc_dtype=Float32,
        sf_type=Float8E8M0FNU,
    )
    mma_atom = cute.make_mma_atom(mma_op)
    tiled_mma = cute.make_tiled_mma(mma_atom)

    tA = cute.make_rmem_tensor(cute.make_layout((32,)), Float4E2M1FN)
    tB = cute.make_rmem_tensor(cute.make_layout((16,)), Float4E2M1FN)
    tC = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
    tD = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)

    cute.gemm(tiled_mma, tD, tA, tB, tC)
    return

@cute.jit
def host_fn_mxf4():
    mxf4_mma_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1])


import tempfile, glob

# Test F16 first
print("=== Test 1: F16 MMA PTX ===")
try:
    with tempfile.TemporaryDirectory() as tmp:
        compiled = cute.compile(host_fn_f16, options=f"--dump-dir={tmp} --keep-ptx")
        for f in glob.glob(f"{tmp}/*.ptx"):
            with open(f) as fp:
                ptx = fp.read()
            for line in ptx.split('\n'):
                if 'mma' in line.lower() and not line.strip().startswith('//'):
                    print(f"  {line.strip()}")
        print("F16 MMA: OK")
except Exception as e:
    print(f"F16 MMA FAILED: {type(e).__name__}: {e}")

# Test MXF4
print("\n=== Test 2: MXF4 block-scaled MMA PTX ===")
try:
    with tempfile.TemporaryDirectory() as tmp:
        compiled = cute.compile(host_fn_mxf4, options=f"--dump-dir={tmp} --keep-ptx")
        for f in glob.glob(f"{tmp}/*.ptx"):
            with open(f) as fp:
                ptx = fp.read()
            for line in ptx.split('\n'):
                if 'mma' in line.lower() and not line.strip().startswith('//'):
                    print(f"  {line.strip()}")
            for line in ptx.split('\n'):
                if '.target' in line:
                    print(f"  Target: {line.strip()}")
        print("MXF4 MMA: OK")
except Exception as e:
    print(f"MXF4 MMA FAILED: {type(e).__name__}: {e}")
