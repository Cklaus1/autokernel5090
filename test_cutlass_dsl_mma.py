"""Use CUTLASS DSL to compile SM120 block-scaled MMA and inspect PTX."""

import os
os.environ["CUTE_DSL_ARCH"] = "sm_120a"

import cutlass
import cutlass.cute as cute
from cutlass import Float4E2M1FN, Float8E8M0FNU, Float32
from cutlass.cute.nvgpu.warp.mma import MmaMXF4Op

@cute.kernel
def mma_kernel():
    # Create SM120 block-scaled MMA atom
    mma_op = MmaMXF4Op(
        ab_dtype=Float4E2M1FN,
        acc_dtype=Float32,
        sf_type=Float8E8M0FNU,
    )
    mma_atom = cute.make_mma_atom(mma_op)
    tiled_mma = cute.make_tiled_mma(mma_atom)
    return

@cute.jit
def host_fn():
    mma_kernel().launch(
        grid=[1, 1, 1],
        block=[32, 1, 1],
    )

import tempfile
try:
    with tempfile.TemporaryDirectory() as tmp:
        compiled = cute.compile(host_fn, options=f"--dump-dir={tmp} --keep-ptx --keep-cubin")
        print(f"Compilation succeeded!")

        # Check for dumped files
        import glob
        for pat in ['**/*.ptx', '**/*.ll', '**/*.mlir', '**/*']:
            files = glob.glob(f"{tmp}/{pat}", recursive=True)
            if files:
                for f in files:
                    if os.path.isfile(f):
                        print(f"  {f}")

        ptx_files = glob.glob(f"{tmp}/**/*.ptx", recursive=True)
        for f in ptx_files:
            with open(f) as fp:
                content = fp.read()
            for line in content.split('\n'):
                if 'mma' in line.lower():
                    print(f"  MMA: {line.strip()}")

        # Try getting artifacts
        if hasattr(compiled, 'artifacts'):
            arts = compiled.artifacts
            for attr in dir(arts):
                if not attr.startswith('_'):
                    val = getattr(arts, attr)
                    if isinstance(val, (bytes, str)):
                        print(f"  Artifact {attr}: {len(val)} bytes")

except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
