"""NVFP4 (e2m1) quantization and GEMM for SM120 (Blackwell).

Provides:
  - quantize_to_nvfp4: FP16 → packed FP4 + block scales
  - dequantize_nvfp4: packed FP4 + scales → FP32 (for verification)
  - nvfp4_gemm: quantize-on-the-fly GEMM at ~1271 TFLOPS
  - NVFP4Linear: drop-in nn.Linear replacement storing FP4 weights
  - convert_model: convert all Linear layers in a model to NVFP4
"""

from .quantize import quantize_to_nvfp4, dequantize_nvfp4
from .gemm import nvfp4_gemm, nvfp4_gemm_prequantized
from .linear import NVFP4Linear, convert_model

__all__ = [
    "quantize_to_nvfp4",
    "dequantize_nvfp4",
    "nvfp4_gemm",
    "nvfp4_gemm_prequantized",
    "NVFP4Linear",
    "convert_model",
]
