"""CUTLASS DSL: block-scaled MMA with proper tensor partitioning."""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass.cute as cute
from cutlass import Float16, Float32, Float4E2M1FN, Float8E8M0FNU
from cutlass.cute.nvgpu.warp.mma import MmaF16BF16Op, MmaMXF4Op
from cutlass.cutlass_dsl import Int32

# Test 1: F16 MMA with partition approach
@cute.kernel
def f16_partition_kernel():
    mma_op = MmaF16BF16Op(
        ab_dtype=Float16,
        acc_dtype=Float32,
        shape_mnk=(16, 8, 16),
    )
    mma_atom = cute.make_mma_atom(mma_op)
    tiled_mma = cute.make_tiled_mma(mma_atom)

    # Create full tile tensors in register memory
    # MMA tile: M=16, N=8, K=16
    sA = cute.make_rmem_tensor(cute.make_layout((16, 16)), Float16)  # M x K
    sB = cute.make_rmem_tensor(cute.make_layout((8, 16)), Float16)   # N x K
    sC = cute.make_rmem_tensor(cute.make_layout((16, 8)), Float32)   # M x N
    sD = cute.make_rmem_tensor(cute.make_layout((16, 8)), Float32)   # M x N

    # Get thread slice and partition
    tidx = cute.arch.thread_idx()
    thr_mma = tiled_mma.get_slice(tidx)

    tA = thr_mma.partition_A(sA)
    tB = thr_mma.partition_B(sB)
    tC = thr_mma.partition_C(sC)
    tD = thr_mma.partition_C(sD)

    cute.gemm(tiled_mma, tD, tA, tB, tC)
    return

@cute.jit
def host_f16():
    f16_partition_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1])


# Test 2: MXF4 block-scaled MMA with partition approach
@cute.kernel
def mxf4_partition_kernel():
    mma_op = MmaMXF4Op(
        ab_dtype=Float4E2M1FN,
        acc_dtype=Float32,
        sf_type=Float8E8M0FNU,
    )
    mma_atom = cute.make_mma_atom(mma_op)
    tiled_mma = cute.make_tiled_mma(mma_atom)

    # MXF4 MMA tile: M=16, N=8, K=64
    sA = cute.make_rmem_tensor(cute.make_layout((16, 64)), Float4E2M1FN)
    sB = cute.make_rmem_tensor(cute.make_layout((8, 64)), Float4E2M1FN)
    sC = cute.make_rmem_tensor(cute.make_layout((16, 8)), Float32)
    sD = cute.make_rmem_tensor(cute.make_layout((16, 8)), Float32)

    tidx = cute.arch.thread_idx()

    tA = thr_mma.partition_A(sA)
    tB = thr_mma.partition_B(sB)
    tC = thr_mma.partition_C(sC)
    tD = thr_mma.partition_C(sD)

    cute.gemm(tiled_mma, tD, tA, tB, tC)
    return

@cute.jit
def host_mxf4():
    mxf4_partition_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1])


import tempfile, glob

for name, fn in [("F16", host_f16), ("MXF4", host_mxf4)]:
    print(f"\n=== {name} MMA ===")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            compiled = cute.compile(fn, options=f"--dump-dir={tmp} --keep-ptx --keep-cubin")
            print("Compiled!")
            for f in sorted(glob.glob(f"{tmp}/**/*", recursive=True)):
                if os.path.isfile(f):
                    sz = os.path.getsize(f)
                    print(f"  {os.path.basename(f)} ({sz} bytes)")
                    if f.endswith('.ptx'):
                        with open(f) as fp:
                            for line in fp:
                                if 'mma' in line.lower() and ('sync' in line or 'block_scale' in line):
                                    if not line.strip().startswith('//'):
                                        print(f"    {line.strip()[:140]}")
                    elif f.endswith('.cubin'):
                        with open(f, 'rb') as fp:
                            cubin = fp.read()
                        # Save cubin
                        out_path = f"/tmp/{name.lower()}_mma.cubin"
                        with open(out_path, 'wb') as fp:
                            fp.write(cubin)
                        print(f"    Saved to {out_path}")
        except Exception as e:
            print(f"Failed: {type(e).__name__}: {str(e)[:500]}")
