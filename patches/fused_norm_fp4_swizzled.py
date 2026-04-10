#!/usr/bin/env python3
"""
Fused RMSNorm + FP4-E2M1 Quantization with CUTLASS 128x4 swizzled scale output.

Produces output compatible with vLLM's cutlass_scaled_fp4_mm.
"""
import torch
import triton
import triton.language as tl


FP4_MAX = 6.0


@triton.jit
def _quantize_to_fp4_code(val_scaled):
    """Map |value / scale| to FP4 magnitude code 0-7."""
    code = (val_scaled * 0).to(tl.int32)
    code = tl.where(val_scaled > 0.25, 1, code)
    code = tl.where(val_scaled > 0.75, 2, code)
    code = tl.where(val_scaled > 1.25, 3, code)
    code = tl.where(val_scaled > 1.75, 4, code)
    code = tl.where(val_scaled > 2.5, 5, code)
    code = tl.where(val_scaled > 3.5, 6, code)
    code = tl.where(val_scaled > 5.0, 7, code)
    return code


def _generate_autotune_configs():
    configs = []
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


@triton.autotune(
    configs=_generate_autotune_configs(),
    key=["H"],
)
@triton.jit
def fused_rms_norm_fp4_quant_swizzled_kernel(
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
    # Swizzle params
    numKTiles: tl.constexpr,
    # Parameters
    eps,
    # Constexpr
    BLOCK_H: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HAVE_WEIGHT: tl.constexpr,
    VARIANCE_SIZE: tl.constexpr,
):
    """Fused RMSNorm + FP4 quant with CUTLASS 128x4 swizzled scale output."""
    row = tl.program_id(0)
    global_scale = tl.load(GS_ptr).to(tl.float32)

    VAR_DIM: tl.constexpr = H if VARIANCE_SIZE == 0 else VARIANCE_SIZE
    NUM_ITERS_VAR: tl.constexpr = (VAR_DIM + BLOCK_H - 1) // BLOCK_H
    NUM_ITERS: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    QBLOCKS_PER_ITER: tl.constexpr = BLOCK_H // QUANT_BLOCK_SIZE
    HALF_BLOCK: tl.constexpr = BLOCK_H // 2
    HALF_QBS: tl.constexpr = QUANT_BLOCK_SIZE // 2

    # Precompute swizzle constants for this row
    mTileIdx = row // 128
    outerMIdx = row % 32
    innerMIdx = (row // 32) % 4

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

        even_offs = base + tl.arange(0, HALF_BLOCK) * 2
        odd_offs = even_offs + 1
        even_mask = even_offs < H
        odd_mask = odd_offs < H

        x_even = tl.load(X_ptr + row * stride_x_b + even_offs, mask=even_mask, other=0.0).to(tl.float32)
        x_odd = tl.load(X_ptr + row * stride_x_b + odd_offs, mask=odd_mask, other=0.0).to(tl.float32)

        xn_even = x_even * rrms
        xn_odd = x_odd * rrms

        if HAVE_WEIGHT:
            w_even = tl.load(W_ptr + even_offs, mask=even_mask, other=1.0).to(tl.float32)
            w_odd = tl.load(W_ptr + odd_offs, mask=odd_mask, other=1.0).to(tl.float32)
            xn_even = xn_even * w_even
            xn_odd = xn_odd * w_odd

        # Per-block scaling
        abs_even_2d = tl.abs(tl.reshape(xn_even, [QBLOCKS_PER_ITER, HALF_QBS]))
        abs_odd_2d = tl.abs(tl.reshape(xn_odd, [QBLOCKS_PER_ITER, HALF_QBS]))
        max_even = tl.max(abs_even_2d, axis=1)
        max_odd = tl.max(abs_odd_2d, axis=1)
        block_max = tl.maximum(max_even, max_odd)

        block_scale = block_max / (6.0 * global_scale)
        block_scale = tl.minimum(block_scale, 448.0)

        # Compute swizzled offsets for scale storage
        kIdx_base = _i * QBLOCKS_PER_ITER + tl.arange(0, QBLOCKS_PER_ITER)
        kTileIdx = kIdx_base // 4
        innerKIdx = kIdx_base % 4
        SFOffset = (mTileIdx * numKTiles + kTileIdx) * 512 + outerMIdx * 16 + innerMIdx * 4 + innerKIdx
        s_mask = kIdx_base < (H // QUANT_BLOCK_SIZE)

        # Store scales in swizzled layout
        tl.store(OUT_scale_ptr + SFOffset, block_scale, mask=s_mask)

        # Read back fp8-quantized scale from swizzled location
        block_scale_fp8 = tl.load(OUT_scale_ptr + SFOffset, mask=s_mask, other=0.0).to(tl.float32)

        # Quantize using fp8-rounded scales
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


@triton.jit
def fused_add_rms_norm_fp4_quant_swizzled_kernel(
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
    numKTiles: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HAVE_WEIGHT: tl.constexpr,
    VARIANCE_SIZE: tl.constexpr,
):
    """Fused residual-add + RMSNorm + FP4 quant with swizzled scales."""
    row = tl.program_id(0)
    global_scale = tl.load(GS_ptr).to(tl.float32)

    VAR_DIM: tl.constexpr = H if VARIANCE_SIZE == 0 else VARIANCE_SIZE
    NUM_ITERS_VAR: tl.constexpr = (VAR_DIM + BLOCK_H - 1) // BLOCK_H
    NUM_ITERS: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    QBLOCKS_PER_ITER: tl.constexpr = BLOCK_H // QUANT_BLOCK_SIZE
    HALF_BLOCK: tl.constexpr = BLOCK_H // 2
    HALF_QBS: tl.constexpr = QUANT_BLOCK_SIZE // 2

    mTileIdx = row // 128
    outerMIdx = row % 32
    innerMIdx = (row // 32) % 4

    # ---- Pass 1: Add residual, store back, compute sum-of-squares ----
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for _i in range(NUM_ITERS_VAR):
        offs = _i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < VAR_DIM
        x_raw = tl.load(X_ptr + row * stride_x_b + offs, mask=mask, other=0.0)
        r_raw = tl.load(R_ptr + row * stride_r_b + offs, mask=mask, other=0.0)
        hidden_raw = x_raw + r_raw
        tl.store(X_ptr + row * stride_x_b + offs, hidden_raw, mask=mask)
        hidden = hidden_raw.to(tl.float32)
        sum_sq += tl.sum(hidden * hidden, axis=0)

    rrms = 1.0 / tl.sqrt(sum_sq / VAR_DIM + eps)

    # ---- Pass 2: Normalize + quantize + pack with swizzled scales ----
    for _i in range(NUM_ITERS):
        base = _i * BLOCK_H
        even_offs = base + tl.arange(0, HALF_BLOCK) * 2
        odd_offs = even_offs + 1
        even_mask = even_offs < H
        odd_mask = odd_offs < H

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

        kIdx_base = _i * QBLOCKS_PER_ITER + tl.arange(0, QBLOCKS_PER_ITER)
        kTileIdx = kIdx_base // 4
        innerKIdx = kIdx_base % 4
        SFOffset = (mTileIdx * numKTiles + kTileIdx) * 512 + outerMIdx * 16 + innerMIdx * 4 + innerKIdx
        s_mask = kIdx_base < (H // QUANT_BLOCK_SIZE)

        tl.store(OUT_scale_ptr + SFOffset, block_scale, mask=s_mask)
        block_scale_fp8 = tl.load(OUT_scale_ptr + SFOffset, mask=s_mask, other=0.0).to(tl.float32)

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
# Python wrappers
# ---------------------------------------------------------------------------

def fused_rms_norm_fp4_quant_swizzled(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    global_scale: torch.Tensor,
    epsilon: float = 1e-6,
    residual: torch.Tensor | None = None,
    quant_block_size: int = 16,
    variance_size: int = 0,
    hidden_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + FP4 quant with CUTLASS 128x4 swizzled scales.

    Args:
        x: [B, H] bf16/fp16
        weight: [H] or None
        global_scale: [1] float32 -- the ORIGINAL global scale (NOT inverse)
                      Our formula: block_scale = amax / (6 * gs)
                      To match vLLM: pass input_global_scale (not _inv)
        epsilon: RMSNorm epsilon
        residual: optional [B, H] for fused add
        quant_block_size: FP4 block size (16)
        variance_size: subset of dims for variance (0 = full H)
        hidden_dim: original hidden dimension for swizzle (defaults to H)
    """
    assert x.ndim == 2
    B, H = x.shape
    assert H % quant_block_size == 0
    assert H % 2 == 0

    if hidden_dim == 0:
        hidden_dim = H

    fp4_out = torch.empty((B, H // 2), device=x.device, dtype=torch.uint8)
    num_scales = H // quant_block_size
    numKTiles = (hidden_dim + 63) // 64
    rounded_B = ((B + 127) // 128) * 128
    rounded_scales = ((num_scales + 3) // 4) * 4

    # Allocate swizzled scale output
    scale_out = torch.zeros(
        rounded_B * rounded_scales, device=x.device, dtype=torch.float8_e4m3fn
    )

    have_weight = weight is not None
    if not have_weight:
        weight = torch.empty(0, device=x.device, dtype=x.dtype)

    grid = (B,)

    if residual is not None:
        assert residual.shape == x.shape
        block_h = min(4096, 1 << (H - 1).bit_length())
        if block_h > H:
            block_h = block_h // 2
        block_h = max(block_h, 256)
        assert block_h % quant_block_size == 0
        fused_add_rms_norm_fp4_quant_swizzled_kernel[grid](
            x, residual, weight,
            fp4_out, scale_out, global_scale,
            B, H,
            x.stride(0), residual.stride(0),
            fp4_out.stride(0),
            numKTiles,
            epsilon,
            BLOCK_H=block_h,
            QUANT_BLOCK_SIZE=quant_block_size,
            HAVE_WEIGHT=have_weight,
            VARIANCE_SIZE=variance_size,
            num_warps=8,
            num_stages=1,
        )
    else:
        fused_rms_norm_fp4_quant_swizzled_kernel[grid](
            x, weight,
            fp4_out, scale_out, global_scale,
            B, H,
            x.stride(0),
            fp4_out.stride(0),
            numKTiles,
            epsilon,
            QUANT_BLOCK_SIZE=quant_block_size,
            HAVE_WEIGHT=have_weight,
            VARIANCE_SIZE=variance_size,
        )

    scale_out = scale_out.reshape(rounded_B, rounded_scales)
    return fp4_out, scale_out


if __name__ == "__main__":
    import time
    from vllm._custom_ops import scaled_fp4_quant

    n = 2816
    gs = torch.tensor([1.0], device='cuda', dtype=torch.float32)
    eps = 1e-6

    print("=== Correctness Tests ===")
    for m_t in [1, 4, 32, 128, 256]:
        x = torch.randn(m_t, n, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(n, device='cuda', dtype=torch.bfloat16).abs() + 0.5

        # Reference: manual norm + vLLM quant
        x_f32 = x.float()
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = (x_f32 * torch.rsqrt(var + eps)).to(torch.bfloat16) * weight
        fp4_ref, scale_ref = scaled_fp4_quant(normed, gs, is_sf_swizzled_layout=True)

        # Our fused kernel
        fp4_ours, scale_ours = fused_rms_norm_fp4_quant_swizzled(
            x.clone(), weight, gs, eps, hidden_dim=n
        )

        fp4_diff = (fp4_ref != fp4_ours).sum().item()
        scale_diff = (scale_ref.view(torch.uint8) != scale_ours.view(torch.uint8)).sum().item()
        fp4_pct = 100 * fp4_diff / fp4_ref.numel()
        scale_pct = 100 * scale_diff / scale_ref.view(torch.uint8).numel()
        print(f"  m={m_t:4d}: FP4 diff {fp4_pct:.1f}%, scale diff {scale_pct:.1f}%")

    print("\n=== Fused-Add Correctness ===")
    for m_t in [1, 32, 128]:
        x = torch.randn(m_t, n, device='cuda', dtype=torch.bfloat16)
        res = torch.randn(m_t, n, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(n, device='cuda', dtype=torch.bfloat16).abs() + 0.5

        # Reference
        combined = x + res
        x_f32 = combined.float()
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = (x_f32 * torch.rsqrt(var + eps)).to(torch.bfloat16) * weight
        fp4_ref, scale_ref = scaled_fp4_quant(normed, gs, is_sf_swizzled_layout=True)

        # Our kernel
        x_copy = x.clone()
        fp4_ours, scale_ours = fused_rms_norm_fp4_quant_swizzled(
            x_copy, weight, gs, eps, residual=res, hidden_dim=n
        )

        fp4_diff = (fp4_ref != fp4_ours).sum().item()
        fp4_pct = 100 * fp4_diff / fp4_ref.numel()
        # Verify x was updated in-place to x + res
        x_updated_match = torch.allclose(x_copy, x + res, atol=1e-3)
        print(f"  m={m_t:4d}: FP4 diff {fp4_pct:.1f}%, residual updated: {x_updated_match}")

    print("\n=== Benchmarks ===")
    for m_b in [1, 32, 128, 256]:
        x = torch.randn(m_b, n, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn(n, device='cuda', dtype=torch.bfloat16).abs() + 0.5

        # Warmup fused
        for _ in range(10):
            fused_rms_norm_fp4_quant_swizzled(x.clone(), weight, gs, eps, hidden_dim=n)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(1000):
            fused_rms_norm_fp4_quant_swizzled(x.clone(), weight, gs, eps, hidden_dim=n)
        torch.cuda.synchronize()
        fused_us = (time.perf_counter() - t0) / 1000 * 1e6

        # Warmup separate
        x_f32 = x.float()
        var = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = (x_f32 * torch.rsqrt(var + eps)).to(torch.bfloat16) * weight
        for _ in range(10):
            scaled_fp4_quant(normed, gs, is_sf_swizzled_layout=True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(1000):
            # Simulate separate norm + quant
            x_f32 = x.float()
            var = x_f32.pow(2).mean(dim=-1, keepdim=True)
            normed = (x_f32 * torch.rsqrt(var + eps)).to(torch.bfloat16) * weight
            scaled_fp4_quant(normed, gs, is_sf_swizzled_layout=True)
        torch.cuda.synchronize()
        separate_us = (time.perf_counter() - t0) / 1000 * 1e6

        print(f"  m={m_b:4d}: fused={fused_us:.1f}us, separate={separate_us:.1f}us, speedup={separate_us/fused_us:.2f}x")
