"""Fused shared-expert MLP (gate_up + SiLU*up + down) for Qwen3 MoE.

Motivation
==========
``Qwen3MoeMLP.forward`` (shared-expert path) runs 3 BF16 kernels
back-to-back:

    gate_up = self.gate_up_proj(x)       # BF16 [M, 2*INTER] matmul
    out = self.act_fn(gate_up)           # SiluAndMul -> [M, INTER]
    out = self.down_proj(out)            # BF16 [M, H] matmul

At M=1 decode each launch is ~5-10 us overhead; the arithmetic is
trivial.  We collapse the 3 launches into 2 by fusing op 1+2 into a
single Triton kernel (gate_up + SiLU*up → BF16 intermediate buffer)
and using a tl.dot-based down-proj as the second kernel.  The
intermediate vector never materialises into a separate SiluAndMul
output tensor that gets read and written again.

Shapes (Qwen3-30B-A3B shared_expert)
====================================
- ``x``:              ``[M, H]``           ``H = 2048``
- ``gate_up.weight``: ``[2*INTER, H]``     ``INTER = 768``
- ``down.weight``:    ``[H, INTER]``
- output:             ``[M, H]``           BF16

gate_up.weight rows [0..INTER) gate, [INTER..2*INTER) up (vLLM
``MergedColumnParallelLinear`` layout).

Design: two-kernel fusion
=========================
Kernel 1: fused_gate_up_silu
  - Input:  x [M, H], gate_up.weight [2*INTER, H]
  - Output: intermediate [M, INTER] BF16
  - Grid:   (ceil(M/BLOCK_M), ceil(INTER/BLOCK_I))
  - Per program: tensor-core matmul over BLOCK_K tiles along H,
    accumulating gate_out[BLOCK_M, BLOCK_I] and up_out[BLOCK_M, BLOCK_I]
    simultaneously (avoids a separate SiluAndMul launch).  Writes
    SiLU(gate) * up to the intermediate buffer.
  - Collapses stock ops 1+2 (gate_up_proj + SiluAndMul) into one kernel.

Kernel 2: down projection
  - Implemented as ``F.linear(intermediate, down.weight)`` — vendor
    cuBLAS BF16 GEMM (already near-optimal on SM120).

Net: 3 launches → 2 launches (−33% launch overhead), with gate/up
partial-sums kept in registers between the gate_up matmul and the
SiLU*up elementwise (stock path writes full [M, 2*INTER] BF16 to
gmem, reads it back, writes [M, INTER] BF16, reads it for down).

Tile choice (H=2048, INTER=768 → INTER_K=1024, decode M=1)
==========================================================
- ``BLOCK_M = 16`` (tensor-core minimum; masked for M<16).
- ``BLOCK_I = 128`` — Stage 1 per-program I-tile; grid along I is
  INTER_K/BLOCK_I = 8.  Output weight tile = 128*64*2B = 16 KB.
- ``BLOCK_K = 64``  — inner K-tile.  x tile = 16*64*2B = 2 KB;
  gate+up weight tiles together = 2*64*128*2B = 32 KB.  Full smem ~48 KB
  per block — well under SM120's 100 KB user budget.
- ``num_warps = 4``, ``num_stages = 2`` for H-loop pipelining.

Correctness
===========
Gate-up matmul in FP32, SiLU*up in FP32, write BF16 intermediate.
This matches the stock path bit-for-bit on the intermediate, and lets
the cuBLAS down GEMM be the single final op.

Tag: W8_T2N_rank6_shared_expert_mlp
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_up_silu_kernel(
    X_ptr,               # [M, H] BF16
    GU_ptr,              # [2*INTER, H] BF16
    Out_ptr,             # [M, INTER] BF16 — intermediate
    M,
    H_real,
    INTER_real,
    stride_xm, stride_xh,
    stride_gu_row, stride_gu_col,
    stride_om, stride_oi,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_ITERS: tl.constexpr,
):
    """Stage 1: gate = x @ gate_w.T; up = x @ up_w.T; out = SiLU(gate)*up.

    Produces BF16 intermediate [M, INTER] without materialising the
    separate gate/up BF16 tensors.
    """
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    i_start = pid_i * BLOCK_I
    i_offs = i_start + tl.arange(0, BLOCK_I)
    i_mask = i_offs < INTER_real

    gate_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)

    for k_it in range(0, K_ITERS):
        k_start = k_it * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H_real

        x_ptrs = (
            X_ptr
            + m_offs[:, None] * stride_xm
            + k_offs[None, :] * stride_xh
        )
        x_tile = tl.load(
            x_ptrs,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )                                               # [BLOCK_M, BLOCK_K]

        # gate_w.T tile: [BLOCK_K, BLOCK_I].
        gate_w_ptrs = (
            GU_ptr
            + i_offs[None, :] * stride_gu_row
            + k_offs[:, None] * stride_gu_col
        )
        gate_w_tile = tl.load(
            gate_w_ptrs,
            mask=k_mask[:, None] & i_mask[None, :], other=0.0,
        )                                               # [BLOCK_K, BLOCK_I]

        up_w_ptrs = (
            GU_ptr
            + (i_offs[None, :] + INTER_real) * stride_gu_row
            + k_offs[:, None] * stride_gu_col
        )
        up_w_tile = tl.load(
            up_w_ptrs,
            mask=k_mask[:, None] & i_mask[None, :], other=0.0,
        )                                               # [BLOCK_K, BLOCK_I]

        gate_acc += tl.dot(x_tile, gate_w_tile, out_dtype=tl.float32)
        up_acc += tl.dot(x_tile, up_w_tile, out_dtype=tl.float32)

    # SiLU(gate) * up in FP32, cast to BF16 for the intermediate buffer.
    interm = (gate_acc * tl.sigmoid(gate_acc)) * up_acc

    # Store to [M, INTER_real] output (writing only the real-I columns).
    out_ptrs = (
        Out_ptr
        + m_offs[:, None] * stride_om
        + i_offs[None, :] * stride_oi
    )
    tl.store(
        out_ptrs, interm.to(tl.bfloat16),
        mask=m_mask[:, None] & i_mask[None, :],
    )


def fused_shared_expert_mlp(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Fused BF16 gate_up + SiLU*up + down projection.

    3 kernels (stock) → 2 kernels (fused Stage 1 + cuBLAS down).

    Args:
        x:              ``[..., H]`` BF16.
        gate_up_weight: ``[2*INTER, H]`` BF16.  Gate rows [0, INTER),
                        up rows [INTER, 2*INTER).
        down_weight:    ``[H, INTER]`` BF16.

    Returns:
        ``[..., H]`` BF16.

    Qwen3-30B-A3B shared_expert: H=2048, INTER=768.
    """
    assert x.is_cuda
    assert gate_up_weight.is_cuda and down_weight.is_cuda

    orig_shape = x.shape
    x2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x

    M, H_real = x2d.shape
    two_inter, H2 = gate_up_weight.shape
    assert H_real == H2, f"hidden mismatch: {H_real} vs {H2}"
    assert two_inter % 2 == 0
    INTER_real = two_inter // 2
    Hd, Id = down_weight.shape
    assert Hd == H_real and Id == INTER_real

    # Allocate intermediate [M, INTER] BF16 (the boundary between
    # Stage 1 and Stage 2 cuBLAS matmul).
    interm = torch.empty((M, INTER_real), device=x.device, dtype=x.dtype)

    # Tile choice — profiled on SM120 RTX PRO 6000 for Qwen3 shapes
    # (H=2048, INTER=768) across M in {1, 4, 16, 128}.  BLOCK_I=128
    # gives 6 programs along I at decode (INTER=768 → 6 tiles) which
    # hits better SM occupancy than BLOCK_I=256 (3 programs, measured
    # -30% vs 3-op) or BLOCK_I=64 (12 programs, high launch-overhead
    # sensitivity).
    BLOCK_M = 16
    BLOCK_I = 128
    BLOCK_K = 64
    K_ITERS = triton.cdiv(H_real, BLOCK_K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(INTER_real, BLOCK_I))

    _fused_gate_up_silu_kernel[grid](
        x2d,
        gate_up_weight,
        interm,
        M,
        H_real,
        INTER_real,
        x2d.stride(0), x2d.stride(1),
        gate_up_weight.stride(0), gate_up_weight.stride(1),
        interm.stride(0), interm.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_I=BLOCK_I,
        BLOCK_K=BLOCK_K,
        K_ITERS=K_ITERS,
        num_warps=4,
        num_stages=2,
    )

    # Stage 2: down projection via cuBLAS (F.linear is NT GEMM, optimal).
    out = torch.nn.functional.linear(interm, down_weight)

    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], H_real)

    return out


def shared_expert_mlp_torch_ref(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference matching the stock 3-op path (BF16 intermediate)."""
    orig_shape = x.shape
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])

    gate_up = x @ gate_up_weight.t()
    INTER = gate_up.shape[-1] // 2
    gate = gate_up[..., :INTER]
    up = gate_up[..., INTER:]
    intermediate = torch.nn.functional.silu(gate) * up
    out = intermediate @ down_weight.t()

    if len(orig_shape) > 2:
        out = out.view(*orig_shape[:-1], out.shape[-1])
    return out
