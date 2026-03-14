"""Test CUTLASS Python API FP16 GEMM on SM120."""

import torch
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)

try:
    import cutlass

    # Try basic GEMM
    plan = cutlass.op.Gemm(
        element=torch.float16,
        layout_a=cutlass.LayoutType.RowMajor,
        layout_b=cutlass.LayoutType.RowMajor,
        layout_c=cutlass.LayoutType.RowMajor,
    )

    C = torch.zeros(M, N, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(3):
        plan.run(A, B, C, C)
    torch.cuda.synchronize()

    t = do_bench(lambda: plan.run(A, B, C, C), warmup=25, rep=100)
    tflops = flops / (t * 1e-3) / 1e12
    print(f"CUTLASS GEMM: {tflops:.1f} TFLOPS")

    # Check accuracy
    ref = torch.mm(A, B)
    max_err = (C - ref).abs().max().item()
    print(f"Max error vs torch.mm: {max_err:.6f}")

except ImportError as e:
    print(f"CUTLASS not available: {e}")
except Exception as e:
    print(f"CUTLASS error: {type(e).__name__}: {e}")

# Also try: cutlass with different accumulator
try:
    plan2 = cutlass.op.Gemm(
        element_a=torch.float16,
        element_b=torch.float16,
        element_c=torch.float16,
        element_d=torch.float16,
        element_accumulator=torch.float16,  # FP16 accumulator
        layout_a=cutlass.LayoutType.RowMajor,
        layout_b=cutlass.LayoutType.RowMajor,
        layout_c=cutlass.LayoutType.RowMajor,
    )
    C2 = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    for _ in range(3):
        plan2.run(A, B, C2, C2)
    torch.cuda.synchronize()
    t = do_bench(lambda: plan2.run(A, B, C2, C2), warmup=25, rep=100)
    tflops = flops / (t * 1e-3) / 1e12
    max_err = (C2 - ref).abs().max().item()
    print(f"CUTLASS FP16 accum: {tflops:.1f} TFLOPS, max_err={max_err:.6f}")
except Exception as e:
    print(f"CUTLASS FP16 accum: {type(e).__name__}: {e}")
