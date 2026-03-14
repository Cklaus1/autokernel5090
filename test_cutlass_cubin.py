"""Extract cubin from CUTLASS DSL compilation of block-scaled MMA kernel."""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass.cute as cute
from cutlass import Float4E2M1FN, Float8E8M0FNU, Float32
from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op

@cute.kernel
def mxf4_kernel():
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
def host_fn():
    mxf4_kernel().launch(grid=[1, 1, 1], block=[32, 1, 1])


import tempfile, glob, sys

with tempfile.TemporaryDirectory() as tmp:
    try:
        compiled = cute.compile(host_fn, options=f"--dump-dir={tmp} --keep-ptx --keep-cubin")
        print(f"Compiled! Type: {type(compiled)}")

        # Check attributes
        for attr in ['__ptx__', '__cubin__', '__mlir__', 'artifacts', 'ir_module', 'engine']:
            if hasattr(compiled, attr):
                val = getattr(compiled, attr)
                if val is not None:
                    if isinstance(val, (bytes, str)):
                        print(f"  {attr}: {len(val)} bytes")
                    else:
                        print(f"  {attr}: {type(val)}")
                else:
                    print(f"  {attr}: None")

        # Check tmp directory
        for f in sorted(glob.glob(f"{tmp}/**/*", recursive=True)):
            if os.path.isfile(f):
                sz = os.path.getsize(f)
                print(f"  File: {os.path.basename(f)} ({sz} bytes)")
                if f.endswith('.ptx'):
                    with open(f) as fp:
                        content = fp.read()
                    for line in content.split('\n'):
                        if 'mma' in line.lower() and not line.strip().startswith('//'):
                            if 'sync' in line or 'block_scale' in line:
                                print(f"    MMA: {line.strip()[:140]}")
                        if '.target' in line:
                            print(f"    {line.strip()}")
                elif f.endswith('.cubin'):
                    # Save cubin for later use
                    with open(f, 'rb') as fp:
                        cubin = fp.read()
                    with open('/tmp/mxf4_kernel.cubin', 'wb') as fp:
                        fp.write(cubin)
                    print(f"    Saved cubin ({len(cubin)} bytes)")

    except Exception as e:
        print(f"Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
