"""NVFP4 (e2m1) quantization and dequantization.

NVFP4 representable values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
Block size: 16 elements per e4m3fn scale (hardware requirement).
Scale padding: cuBLASLt requires M padded to 128 and num_blocks padded to 4.

Requirements:
    - PyTorch >= 2.10 with float4_e2m1fn_x2 dtype
    - CUDA GPU (quantization runs on GPU)
"""

import math
import os
import torch
from torch.utils.cpp_extension import load as _load_ext


def quantize_to_nvfp4(x_fp16: torch.Tensor, block_size: int = 16):
    """Quantize FP16 tensor to NVFP4 with per-block e4m3fn scales.

    Args:
        x_fp16: [M, K] float16 tensor on CUDA
        block_size: elements per scale block (16 for HW path)

    Returns:
        x_fp4: [M, K//2] float4_e2m1fn_x2 (2 FP4 values per element)
        scales: flat float8_e4m3fn scale tensor, padded per cuBLAS layout
    """
    M, K = x_fp16.shape
    assert K % block_size == 0, f"K={K} must be divisible by block_size={block_size}"

    x = x_fp16.float()
    num_blocks = K // block_size
    x_blocks = x.reshape(M, num_blocks, block_size)

    # Per-block scale = max_abs / 6.0 (FP4 max magnitude)
    block_max = x_blocks.abs().amax(dim=-1)
    scales_fp32 = (block_max / 6.0).clamp(min=1e-12)

    # Scale values into FP4 range and quantize via nearest-value lookup
    x_scaled = x_blocks / scales_fp32.unsqueeze(-1)

    nvfp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device
    )
    x_flat = x_scaled.reshape(-1)
    x_sign = x_flat.sign()
    x_abs = x_flat.abs()

    diffs = (x_abs.unsqueeze(-1) - nvfp4_values.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)

    # E2M1 encoding: magnitude codes 0-7, sign bit is MSB (bit 3)
    e2m1_encoding = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7], device=x.device, dtype=torch.uint8
    )
    fp4_codes = e2m1_encoding[indices]
    fp4_codes = fp4_codes | ((x_sign < 0).to(torch.uint8) << 3)

    # Pack two FP4 values per byte (low nibble = first element)
    fp4_codes = fp4_codes.reshape(-1, 2)
    packed = (fp4_codes[:, 1] << 4) | fp4_codes[:, 0]
    packed = packed.reshape(M, K // 2)

    # Pad scales to cuBLAS block-scaled layout:
    #   padded_M = ceil(M / 128) * 128
    #   padded_num_blocks = ceil(num_blocks / 4) * 4
    padded_num_blocks = math.ceil(num_blocks / 4) * 4
    padded_M = math.ceil(M / 128) * 128

    scales_padded = torch.zeros(
        padded_M, padded_num_blocks, device=x.device, dtype=torch.float32
    )
    scales_padded[:M, :num_blocks] = scales_fp32
    scales_flat = scales_padded.reshape(-1).to(torch.float8_e4m3fn)

    x_fp4 = packed.contiguous().view(torch.float4_e2m1fn_x2)
    return x_fp4, scales_flat


# ---------------------------------------------------------------------------
# Fast CUDA quantization backend (v3 > v2 > v1 > Python fallback)
# ---------------------------------------------------------------------------
_nvfp4_dir = os.path.dirname(os.path.abspath(__file__))
_CUDA_SRC_V3 = os.path.join(_nvfp4_dir, "quantize_cuda_v3.cu")
_CUDA_SRC_V2 = os.path.join(_nvfp4_dir, "quantize_cuda_v2.cu")
_CUDA_SRC_V1 = os.path.join(_nvfp4_dir, "quantize_cuda.cu")

_cuda_mod = None
_cuda_version = 0  # 0 = not loaded, 1 = v1, 2 = v2/v3 (same API)
_scale_buf = {}


def _load_cuda_quant():
    """JIT-compile the fastest available CUDA quantization kernel."""
    global _cuda_mod, _cuda_version
    if _cuda_mod is not None:
        return _cuda_mod

    _cflags = ['-O3', '--use_fast_math', '-arch=sm_80']

    # Try v3 (vectorized half2 + manual FP8, fastest)
    if os.path.exists(_CUDA_SRC_V3):
        try:
            _cuda_mod = _load_ext('nvfp4_quant_v3', sources=[_CUDA_SRC_V3],
                                  extra_cuda_cflags=_cflags, verbose=False)
            _cuda_version = 2  # same API as v2
            return _cuda_mod
        except Exception:
            pass

    # Try v2 (fused FP8 scales via __nv_fp8_e4m3)
    if os.path.exists(_CUDA_SRC_V2):
        try:
            _cuda_mod = _load_ext('nvfp4_quant_v2', sources=[_CUDA_SRC_V2],
                                  extra_cuda_cflags=_cflags, verbose=False)
            _cuda_version = 2
            return _cuda_mod
        except Exception:
            pass

    # Try v1 (FP16 scales, needs Python-side padding)
    if os.path.exists(_CUDA_SRC_V1):
        try:
            _cuda_mod = _load_ext('nvfp4_quant', sources=[_CUDA_SRC_V1],
                                  extra_cuda_cflags=_cflags, verbose=False)
            _cuda_version = 1
            return _cuda_mod
        except Exception:
            pass

    return None


def quantize_to_nvfp4_fast(x_fp16: torch.Tensor, block_size: int = 16):
    """Quantize FP16 tensor to NVFP4 using the fastest available backend.

    Tries CUDA kernels (v3 → v2 → v1), falls back to Python searchsorted.
    CUDA v3 is ~15x faster than Python (24µs vs 358µs for 2048x5120).

    Args:
        x_fp16: [M, K] float16 tensor on CUDA
        block_size: elements per scale block (16 for HW path)

    Returns:
        x_fp4: [M, K//2] float4_e2m1fn_x2
        scales: flat float8_e4m3fn scale tensor, padded per cuBLAS layout
    """
    M, K = x_fp16.shape
    num_blocks = K // block_size
    padded_num_blocks = math.ceil(num_blocks / 4) * 4
    padded_M = math.ceil(M / 128) * 128

    cuda_mod = _load_cuda_quant()
    if cuda_mod is not None:
        if _cuda_version == 2:
            # v2/v3: outputs pre-padded FP8 scales directly
            packed, scales_uint8 = cuda_mod.quantize_nvfp4(
                x_fp16, padded_M, padded_num_blocks)
            x_fp4 = packed.view(torch.float4_e2m1fn_x2)
            scales_flat = scales_uint8.view(torch.float8_e4m3fn)
            return x_fp4, scales_flat
        else:
            # v1: FP16 scales, needs Python-side padding
            packed, scales_fp16 = cuda_mod.quantize_nvfp4(x_fp16)
            buf_key = (padded_M, padded_num_blocks, x_fp16.device)
            if buf_key not in _scale_buf:
                _scale_buf[buf_key] = torch.zeros(
                    padded_M, padded_num_blocks,
                    device=x_fp16.device, dtype=torch.float32)
            scales_padded = _scale_buf[buf_key]
            scales_padded.zero_()
            scales_padded[:M, :num_blocks] = scales_fp16.float()
            scales_flat = scales_padded.reshape(-1).to(torch.float8_e4m3fn)
            x_fp4 = packed.view(torch.float4_e2m1fn_x2)
            return x_fp4, scales_flat

    # Fallback: Python searchsorted
    return quantize_to_nvfp4(x_fp16, block_size)


def dequantize_nvfp4(
    x_fp4_packed: torch.Tensor,
    scales: torch.Tensor,
    M: int,
    K: int,
    block_size: int = 16,
):
    """Dequantize NVFP4 tensor back to FP32.

    Args:
        x_fp4_packed: [M, K//2] float4_e2m1fn_x2 tensor
        scales: flat float8_e4m3fn scale tensor
        M, K: original matrix dimensions
        block_size: elements per scale block

    Returns:
        x_fp32: [M, K] float32 tensor (approximate reconstruction)
    """
    packed = x_fp4_packed.view(torch.uint8).reshape(M, K // 2)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    fp4_codes = torch.stack([lo, hi], dim=-1).reshape(M, K)

    sign_bit = (fp4_codes >> 3) & 1
    magnitude = fp4_codes & 0x07
    nvfp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x_fp4_packed.device
    )
    values = nvfp4_values[magnitude.long()] * (1 - 2 * sign_bit.float())

    num_blocks = K // block_size
    padded_num_blocks = math.ceil(num_blocks / 4) * 4
    padded_M = math.ceil(M / 128) * 128
    scales_fp32 = scales.reshape(padded_M, padded_num_blocks).float()[:M, :num_blocks]

    values = values.reshape(M, num_blocks, block_size) * scales_fp32.unsqueeze(-1)
    return values.reshape(M, K)
