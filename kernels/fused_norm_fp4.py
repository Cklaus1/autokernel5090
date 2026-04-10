"""
Fused RMSNorm + FP4-E2M1 Quantization Triton Kernel

Performs in ONE kernel launch:
1. RMSNorm: x_norm = x * rsqrt(mean(x^2) + eps) * weight
2. Per-block scaling: compute max absolute value per block of 16 elements
3. FP4-E2M1 quantization: map normalized values to nearest FP4 value
4. Byte packing: pack two FP4 values per uint8 byte (low nibble first)

Input:  x [B, H] bf16/fp16, weight [H] bf16/fp16, epsilon float, global_scale [1] float32
Output: x_fp4 [B, H/2] uint8 (packed), block_scale [B, H/block_size] fp8_e4m3fn
"""

KERNEL_TYPE = "fused_norm_fp4"

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# FP4-E2M1 values: 0->0.0, 1->0.5, 2->1.0, 3->1.5, 4->2.0, 5->3.0, 6->4.0, 7->6.0
# ---------------------------------------------------------------------------
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_MAX = 6.0


def _generate_autotune_configs():
    """Generate autotune configurations for the fused kernel."""
    configs = []
    # BLOCK_H must be power-of-2 (Triton tl.arange requirement)
    # and divisible by QUANT_BLOCK_SIZE (16)
    for block_h in [256, 512, 1024, 2048, 4096]:
        for num_warps in [4, 8, 16]:
            for num_stages in [1, 2, 3]:
                configs.append(
                    triton.Config(
                        {"BLOCK_H": block_h},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.jit
def _quantize_to_fp4_code(val_scaled):
    """Map |value / scale| to FP4 magnitude code 0-7 using boundary thresholds."""
    code = (val_scaled * 0).to(tl.int32)  # zeros with matching shape
    code = tl.where(val_scaled > 0.25, 1, code)
    code = tl.where(val_scaled > 0.75, 2, code)
    code = tl.where(val_scaled > 1.25, 3, code)
    code = tl.where(val_scaled > 1.75, 4, code)
    code = tl.where(val_scaled > 2.5, 5, code)
    code = tl.where(val_scaled > 3.5, 6, code)
    code = tl.where(val_scaled > 5.0, 7, code)
    return code


@triton.autotune(
    configs=_generate_autotune_configs(),
    key=["H"],
)
@triton.jit
def fused_rms_norm_fp4_quant_kernel(
    # Pointers
    X_ptr,
    W_ptr,
    OUT_fp4_ptr,
    OUT_scale_ptr,
    GS_ptr,
    # Dimensions
    B,
    H: tl.constexpr,
    stride_x_b,
    stride_out_b,
    stride_scale_b,
    # Parameters
    eps,
    # Constexpr tuning
    BLOCK_H: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HAVE_WEIGHT: tl.constexpr,
    VARIANCE_SIZE: tl.constexpr,
):
    """Fused RMSNorm + FP4-E2M1 quantization.

    One program per row. Two passes over the row:
      Pass 1: sum of squares for RMS
      Pass 2: normalize, per-block scale, quantize, pack
    """
    row = tl.program_id(0)
    global_scale = tl.load(GS_ptr).to(tl.float32)

    VAR_DIM: tl.constexpr = H if VARIANCE_SIZE == 0 else VARIANCE_SIZE
    NUM_ITERS_VAR: tl.constexpr = (VAR_DIM + BLOCK_H - 1) // BLOCK_H
    NUM_ITERS: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    QBLOCKS_PER_ITER: tl.constexpr = BLOCK_H // QUANT_BLOCK_SIZE
    HALF_BLOCK: tl.constexpr = BLOCK_H // 2
    HALF_QBS: tl.constexpr = QUANT_BLOCK_SIZE // 2

    # ---- Pass 1: Compute sum-of-squares ----
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for _i in range(NUM_ITERS_VAR):
        offs = _i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < VAR_DIM
        x = tl.load(X_ptr + row * stride_x_b + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    rrms = 1.0 / tl.sqrt(sum_sq / VAR_DIM + eps)

    # ---- Pass 2: Normalize + quantize + pack ----
    for _i in range(NUM_ITERS):
        base = _i * BLOCK_H

        # Load even elements (positions 0, 2, 4, ...) and odd elements (1, 3, 5, ...)
        # This naturally gives us the pairs for byte packing.
        even_offs = base + tl.arange(0, HALF_BLOCK) * 2       # [HALF_BLOCK]
        odd_offs = even_offs + 1                                # [HALF_BLOCK]
        even_mask = even_offs < H
        odd_mask = odd_offs < H

        # Load even/odd elements
        x_even = tl.load(X_ptr + row * stride_x_b + even_offs, mask=even_mask, other=0.0).to(tl.float32)
        x_odd = tl.load(X_ptr + row * stride_x_b + odd_offs, mask=odd_mask, other=0.0).to(tl.float32)

        # Normalize
        xn_even = x_even * rrms
        xn_odd = x_odd * rrms

        # Apply weight
        if HAVE_WEIGHT:
            w_even = tl.load(W_ptr + even_offs, mask=even_mask, other=1.0).to(tl.float32)
            w_odd = tl.load(W_ptr + odd_offs, mask=odd_mask, other=1.0).to(tl.float32)
            xn_even = xn_even * w_even
            xn_odd = xn_odd * w_odd

        # ---- Per-block scaling ----
        # We need max(abs(block)) for each QUANT_BLOCK_SIZE block.
        # Each block has QUANT_BLOCK_SIZE elements = HALF_QBS even + HALF_QBS odd.
        # Reshape even/odd to [QBLOCKS_PER_ITER, HALF_QBS]
        abs_even_2d = tl.abs(tl.reshape(xn_even, [QBLOCKS_PER_ITER, HALF_QBS]))
        abs_odd_2d = tl.abs(tl.reshape(xn_odd, [QBLOCKS_PER_ITER, HALF_QBS]))

        # Max within each block (combine even and odd)
        max_even = tl.max(abs_even_2d, axis=1)  # [QBLOCKS_PER_ITER]
        max_odd = tl.max(abs_odd_2d, axis=1)    # [QBLOCKS_PER_ITER]
        block_max = tl.maximum(max_even, max_odd)

        block_scale = block_max / (6.0 * global_scale)
        block_scale = tl.minimum(block_scale, 448.0)  # fp8_e4m3fn max

        # Store scales as fp8_e4m3fn
        s_offs = _i * QBLOCKS_PER_ITER + tl.arange(0, QBLOCKS_PER_ITER)
        s_mask = s_offs < (H // QUANT_BLOCK_SIZE)
        tl.store(OUT_scale_ptr + row * stride_scale_b + s_offs, block_scale, mask=s_mask)

        # Read back fp8-quantized scale for exact match with C++ kernel behavior
        block_scale_fp8 = tl.load(
            OUT_scale_ptr + row * stride_scale_b + s_offs, mask=s_mask, other=0.0
        ).to(tl.float32)

        # ---- Quantize using fp8-rounded scales ----
        bs_2d = tl.reshape(block_scale_fp8, [QBLOCKS_PER_ITER, 1])
        bs_2d = tl.broadcast_to(bs_2d, [QBLOCKS_PER_ITER, HALF_QBS])
        denom = tl.reshape(bs_2d, [HALF_BLOCK]) * global_scale
        denom = tl.where(denom > 0.0, denom, 1.0)

        # Scaled absolute values
        vs_even = tl.abs(xn_even) / denom
        vs_odd = tl.abs(xn_odd) / denom

        # Map to FP4 magnitude codes
        code_even = _quantize_to_fp4_code(vs_even)
        code_odd = _quantize_to_fp4_code(vs_odd)

        # Apply sign bits
        sign_even = tl.where(xn_even < 0.0, 8, 0).to(tl.int32)
        sign_odd = tl.where(xn_odd < 0.0, 8, 0).to(tl.int32)
        fp4_even = code_even | sign_even  # low nibble
        fp4_odd = code_odd | sign_odd     # high nibble

        # Pack: byte = lo | (hi << 4)
        packed = (fp4_even & 0xF) | ((fp4_odd & 0xF) << 4)

        # Store packed bytes
        byte_offs = _i * HALF_BLOCK + tl.arange(0, HALF_BLOCK)
        byte_mask = byte_offs < (H // 2)
        tl.store(
            OUT_fp4_ptr + row * stride_out_b + byte_offs,
            packed.to(tl.uint8),
            mask=byte_mask,
        )


@triton.jit
def fused_add_rms_norm_fp4_quant_kernel(
    X_ptr,
    R_ptr,
    W_ptr,
    OUT_fp4_ptr,
    OUT_scale_ptr,
    GS_ptr,
    B,
    H: tl.constexpr,
    stride_x_b,
    stride_r_b,
    stride_out_b,
    stride_scale_b,
    eps,
    BLOCK_H: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HAVE_WEIGHT: tl.constexpr,
    VARIANCE_SIZE: tl.constexpr,
):
    """Fused residual-add + RMSNorm + FP4 quantization.

    hidden = x + residual (stored back to X_ptr), then RMSNorm(hidden) -> FP4.
    """
    row = tl.program_id(0)
    global_scale = tl.load(GS_ptr).to(tl.float32)

    VAR_DIM: tl.constexpr = H if VARIANCE_SIZE == 0 else VARIANCE_SIZE
    NUM_ITERS_VAR: tl.constexpr = (VAR_DIM + BLOCK_H - 1) // BLOCK_H
    NUM_ITERS: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    QBLOCKS_PER_ITER: tl.constexpr = BLOCK_H // QUANT_BLOCK_SIZE
    HALF_BLOCK: tl.constexpr = BLOCK_H // 2
    HALF_QBS: tl.constexpr = QUANT_BLOCK_SIZE // 2

    # ---- Pass 1: Add residual, store back, compute sum-of-squares ----
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for _i in range(NUM_ITERS_VAR):
        offs = _i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < VAR_DIM
        x_raw = tl.load(X_ptr + row * stride_x_b + offs, mask=mask, other=0.0)
        r_raw = tl.load(R_ptr + row * stride_r_b + offs, mask=mask, other=0.0)
        hidden_raw = x_raw + r_raw  # bf16/fp16 addition
        # Store back in original dtype for the residual stream
        tl.store(X_ptr + row * stride_x_b + offs, hidden_raw, mask=mask)
        # Upcast to float32 for variance accumulation
        hidden = hidden_raw.to(tl.float32)
        sum_sq += tl.sum(hidden * hidden, axis=0)

    rrms = 1.0 / tl.sqrt(sum_sq / VAR_DIM + eps)

    # ---- Pass 2: Normalize + quantize + pack (same as non-fused variant) ----
    for _i in range(NUM_ITERS):
        base = _i * BLOCK_H
        even_offs = base + tl.arange(0, HALF_BLOCK) * 2
        odd_offs = even_offs + 1
        even_mask = even_offs < H
        odd_mask = odd_offs < H

        # Load updated hidden values
        x_even = tl.load(X_ptr + row * stride_x_b + even_offs, mask=even_mask, other=0.0).to(tl.float32)
        x_odd = tl.load(X_ptr + row * stride_x_b + odd_offs, mask=odd_mask, other=0.0).to(tl.float32)

        xn_even = x_even * rrms
        xn_odd = x_odd * rrms

        if HAVE_WEIGHT:
            w_even = tl.load(W_ptr + even_offs, mask=even_mask, other=1.0).to(tl.float32)
            w_odd = tl.load(W_ptr + odd_offs, mask=odd_mask, other=1.0).to(tl.float32)
            xn_even = xn_even * w_even
            xn_odd = xn_odd * w_odd

        abs_even_2d = tl.abs(tl.reshape(xn_even, [QBLOCKS_PER_ITER, HALF_QBS]))
        abs_odd_2d = tl.abs(tl.reshape(xn_odd, [QBLOCKS_PER_ITER, HALF_QBS]))
        max_even = tl.max(abs_even_2d, axis=1)
        max_odd = tl.max(abs_odd_2d, axis=1)
        block_max = tl.maximum(max_even, max_odd)

        block_scale = block_max / (6.0 * global_scale)
        block_scale = tl.minimum(block_scale, 448.0)

        s_offs = _i * QBLOCKS_PER_ITER + tl.arange(0, QBLOCKS_PER_ITER)
        s_mask = s_offs < (H // QUANT_BLOCK_SIZE)
        tl.store(OUT_scale_ptr + row * stride_scale_b + s_offs, block_scale, mask=s_mask)

        # Read back fp8-quantized scale
        block_scale_fp8 = tl.load(
            OUT_scale_ptr + row * stride_scale_b + s_offs, mask=s_mask, other=0.0
        ).to(tl.float32)

        bs_2d = tl.reshape(block_scale_fp8, [QBLOCKS_PER_ITER, 1])
        bs_2d = tl.broadcast_to(bs_2d, [QBLOCKS_PER_ITER, HALF_QBS])
        denom = tl.reshape(bs_2d, [HALF_BLOCK]) * global_scale
        denom = tl.where(denom > 0.0, denom, 1.0)

        vs_even = tl.abs(xn_even) / denom
        vs_odd = tl.abs(xn_odd) / denom
        code_even = _quantize_to_fp4_code(vs_even)
        code_odd = _quantize_to_fp4_code(vs_odd)
        sign_even = tl.where(xn_even < 0.0, 8, 0).to(tl.int32)
        sign_odd = tl.where(xn_odd < 0.0, 8, 0).to(tl.int32)
        fp4_even = code_even | sign_even
        fp4_odd = code_odd | sign_odd

        packed = (fp4_even & 0xF) | ((fp4_odd & 0xF) << 4)

        byte_offs = _i * HALF_BLOCK + tl.arange(0, HALF_BLOCK)
        byte_mask = byte_offs < (H // 2)
        tl.store(
            OUT_fp4_ptr + row * stride_out_b + byte_offs,
            packed.to(tl.uint8),
            mask=byte_mask,
        )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_rms_norm_fp4_quant(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    global_scale: torch.Tensor,
    epsilon: float = 1e-6,
    residual: torch.Tensor | None = None,
    quant_block_size: int = 16,
    variance_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + FP4-E2M1 quantization.

    Args:
        x: Input tensor [B, H] in bf16/fp16
        weight: RMSNorm weight [H] or None
        global_scale: Global FP4 scale [1] float32
        epsilon: RMSNorm epsilon
        residual: Optional residual tensor [B, H] to add before norm
        quant_block_size: FP4 quantization block size (default 16)
        variance_size: Subset of dims for variance (0 = use full H)

    Returns:
        (fp4_packed [B, H/2] uint8, block_scales [B, H/quant_block_size] fp8_e4m3fn)
        If residual is provided, x is modified in-place (x += residual).
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    B, H = x.shape
    assert H % quant_block_size == 0, f"H={H} not divisible by quant_block_size={quant_block_size}"
    assert H % 2 == 0, f"H={H} must be even for FP4 packing"

    fp4_out = torch.empty((B, H // 2), device=x.device, dtype=torch.uint8)
    num_scales = H // quant_block_size
    scale_out = torch.empty((B, num_scales), device=x.device, dtype=torch.float8_e4m3fn)

    have_weight = weight is not None
    if not have_weight:
        weight = torch.empty(0, device=x.device, dtype=x.dtype)

    grid = (B,)

    if residual is not None:
        assert residual.shape == x.shape
        # Pick BLOCK_H: largest power-of-2 <= H, capped at 4096
        block_h = min(4096, 1 << (H - 1).bit_length())
        if block_h > H:
            block_h = block_h // 2
        block_h = max(block_h, 256)
        # Ensure BLOCK_H >= QUANT_BLOCK_SIZE and is a multiple of it
        assert block_h % quant_block_size == 0
        fused_add_rms_norm_fp4_quant_kernel[grid](
            x, residual, weight,
            fp4_out, scale_out, global_scale,
            B, H,
            x.stride(0), residual.stride(0),
            fp4_out.stride(0), scale_out.stride(0),
            epsilon,
            BLOCK_H=block_h,
            QUANT_BLOCK_SIZE=quant_block_size,
            HAVE_WEIGHT=have_weight,
            VARIANCE_SIZE=variance_size,
            num_warps=8,
            num_stages=1,
        )
    else:
        fused_rms_norm_fp4_quant_kernel[grid](
            x, weight,
            fp4_out, scale_out, global_scale,
            B, H,
            x.stride(0),
            fp4_out.stride(0), scale_out.stride(0),
            epsilon,
            QUANT_BLOCK_SIZE=quant_block_size,
            HAVE_WEIGHT=have_weight,
            VARIANCE_SIZE=variance_size,
        )

    return fp4_out, scale_out


# ---------------------------------------------------------------------------
# Swizzle utility: convert non-swizzled scales to CUTLASS 128x4 swizzled layout
# ---------------------------------------------------------------------------

def swizzle_scales_cuda(scales: torch.Tensor) -> torch.Tensor:
    """
    GPU-accelerated scale swizzle from [B, num_scales] fp8_e4m3fn
    to CUTLASS 128x4 swizzled layout.

    Row mapping: swizzled_row = (row % 32) * 4 + (row // 32)
    """
    B, num_scales = scales.shape
    rounded_B = ((B + 127) // 128) * 128
    rounded_scales = ((num_scales + 3) // 4) * 4

    out = torch.zeros((rounded_B, rounded_scales), device=scales.device, dtype=torch.uint8)
    scale_bytes = scales.view(torch.uint8)

    rows = torch.arange(B, device=scales.device)
    swizzled_rows = (rows % 32) * 4 + (rows // 32)

    padded = torch.nn.functional.pad(scale_bytes, (0, rounded_scales - num_scales), value=0)
    out.index_copy_(0, swizzled_rows, padded)

    return out.view(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# torch.library registration for vLLM fusion pass compatibility
# ---------------------------------------------------------------------------

_lib = torch.library.Library("autokernel", "DEF")
_lib.define(
    "fused_rms_norm_fp4_quant(Tensor x, Tensor? weight, Tensor global_scale, "
    "float epsilon, Tensor? residual, int quant_block_size, int variance_size) "
    "-> (Tensor, Tensor)"
)


@torch.library.impl(_lib, "fused_rms_norm_fp4_quant", "CUDA")
def _fused_rms_norm_fp4_quant_impl(x, weight, global_scale, epsilon, residual, quant_block_size, variance_size):
    return fused_rms_norm_fp4_quant(x, weight, global_scale, epsilon, residual, quant_block_size, variance_size)


@torch.library.impl(_lib, "fused_rms_norm_fp4_quant", "Meta")
def _fused_rms_norm_fp4_quant_fake(x, weight, global_scale, epsilon, residual, quant_block_size, variance_size):
    """Fake tensor implementation for torch.compile."""
    B, H = x.shape
    fp4_out = torch.empty((B, H // 2), device=x.device, dtype=torch.uint8)
    scale_out = torch.empty(
        (B, H // quant_block_size), device=x.device, dtype=torch.float8_e4m3fn
    )
    return fp4_out, scale_out
