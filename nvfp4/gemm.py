"""NVFP4 GEMM via PyTorch's cuBLASLt block-scaled path.

Achieves ~1271 TFLOPS on RTX 5090 (SM120) for large matrices.
Requires M >= 128 for correct scale padding. For M < 128, fall back to FP16 cuBLAS.

API: torch._scaled_mm with float4_e2m1fn_x2 data + float8_e4m3fn block scales.
"""

import torch
from .quantize import quantize_to_nvfp4


def nvfp4_gemm(A_fp16: torch.Tensor, B_fp16: torch.Tensor):
    """NVFP4 GEMM: C = A @ B.T with on-the-fly quantization to FP4.

    Args:
        A_fp16: [M, K] float16 activations (M must be >= 128)
        B_fp16: [N, K] float16 weights

    Returns:
        C: [M, N] float16 output
    """
    A_fp4, scale_a = quantize_to_nvfp4(A_fp16, block_size=16)
    B_fp4, scale_b = quantize_to_nvfp4(B_fp16, block_size=16)
    B_col = B_fp4.t()
    return torch._scaled_mm(
        A_fp4, B_col, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16
    )


def nvfp4_gemm_prequantized(
    A_fp4: torch.Tensor,
    scale_a: torch.Tensor,
    B_fp4_col: torch.Tensor,
    scale_b: torch.Tensor,
):
    """NVFP4 GEMM with pre-quantized inputs (weight-stationary inference).

    Args:
        A_fp4: [M, K//2] float4_e2m1fn_x2 (row-major)
        scale_a: flat float8_e4m3fn scale tensor for A
        B_fp4_col: [K//2, N] float4_e2m1fn_x2 (col-major, i.e. B_fp4.t())
        scale_b: flat float8_e4m3fn scale tensor for B

    Returns:
        C: [M, N] float16 output
    """
    return torch._scaled_mm(
        A_fp4, B_fp4_col, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16
    )
