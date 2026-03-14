"""Smoke test: CUTLASS DSL on SM120."""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import Float4E2M1FN, Float8E8M0FNU, Float8E4M3FN, Float32, Float16

print(f"CUTLASS DSL CUDA version: {cutlass.CUDA_VERSION}")
print(f"CUTLASS DSL version: {cutlass.__version__}")

# Test 1: Simple kernel compilation
print("\n=== Test 1: Empty kernel compilation ===")

@cute.kernel
def empty_kernel():
    return

@cute.jit
def host_fn():
    empty_kernel().launch(
        grid=[1, 1, 1],
        block=[1, 1, 1],
    )

try:
    compiled = cute.compile(host_fn)
    print(f"Compilation succeeded! Type: {type(compiled)}")
except Exception as e:
    print(f"Compilation failed: {type(e).__name__}: {e}")

# Test 2: MMA atom instantiation inside a kernel
print("\n=== Test 2: MMA atom in kernel context ===")

from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op

@cute.kernel
def mma_test_kernel():
    mma_op = MmaMXF4Op(
        ab_dtype=Float4E2M1FN,
        acc_dtype=Float32,
        sf_type=Float8E8M0FNU,
    )
    mma_atom = cute.make_mma_atom(mma_op)
    return

@cute.jit
def host_fn2():
    mma_test_kernel().launch(
        grid=[1, 1, 1],
        block=[32, 1, 1],
    )

try:
    compiled2 = cute.compile(host_fn2)
    print(f"MMA kernel compilation succeeded!")
except Exception as e:
    print(f"MMA kernel compilation failed: {type(e).__name__}: {e}")

# Test 3: MmaF16BF16Op (legacy warp-level, should work)
print("\n=== Test 3: F16 MMA atom ===")

from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op

@cute.kernel
def f16_mma_kernel():
    mma_op = MmaF16BF16Op(
        ab_dtype=Float16,
        acc_dtype=Float16,
        shape_mnk=(16, 8, 16),
    )
    mma_atom = cute.make_mma_atom(mma_op)
    return

@cute.jit
def host_fn3():
    f16_mma_kernel().launch(
        grid=[1, 1, 1],
        block=[32, 1, 1],
    )

try:
    compiled3 = cute.compile(host_fn3)
    print(f"F16 MMA kernel compilation succeeded!")
except Exception as e:
    print(f"F16 MMA kernel compilation failed: {type(e).__name__}: {e}")
