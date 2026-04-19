#!/usr/bin/env python3
"""
T2-N scale swizzle converter.

The fused_shuffle_quant CUDA kernel writes per-sorted-row FP8 scale bytes in a
ROW-MAJOR layout of shape [padded_M, padded_nb]. cutlass_fp4_moe_mm requires
scales in CUTLASS 128x4 swizzled layout with per-expert base offset. Forward map:

    row_in_buf = blockscale_offsets[e] + (dst_row - expert_offsets[e])
    mTileIdx  = row_in_buf >> 7
    outerMIdx = row_in_buf & 31
    innerMIdx = (row_in_buf >> 5) & 3
    kTileIdx  = kIdx >> 2
    innerKIdx = kIdx & 3
    byte_off  = ((mTileIdx * numKTiles + kTileIdx) << 9)
                | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx

numKTiles = (K + 63) // 64.

This module provides a Triton kernel that does both the per-expert remap and
CUTLASS swizzle in one pass, using a small per-row lookup table `row_in_buf[dst_row]`
built on GPU from (expert_offsets, blockscale_offsets) to avoid the 128-way expert
scan inside the swizzle kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _build_row_lookup_kernel(
    ROW_IN_BUF_ptr,           # int32* [M_sorted]
    EXPERT_OFFSETS_ptr,       # int32* [E+1]
    BLOCKSCALE_OFFSETS_ptr,   # int32* [E+1]
    NUM_EXPERTS: tl.constexpr,
    M_SORTED,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    mask = rows < M_SORTED

    # Find expert via scan — but done ONCE, not per block_idx.
    expert_start = tl.zeros([BLOCK_ROWS], dtype=tl.int32)
    bs_off = tl.zeros([BLOCK_ROWS], dtype=tl.int32)
    for e in tl.static_range(NUM_EXPERTS):
        cur = tl.load(EXPERT_OFFSETS_ptr + e)
        nxt = tl.load(EXPERT_OFFSETS_ptr + e + 1)
        hit = (rows >= cur) & (rows < nxt)
        expert_start = tl.where(hit, cur, expert_start)
        bs_off = tl.where(hit, tl.load(BLOCKSCALE_OFFSETS_ptr + e), bs_off)

    row_in_buf = bs_off + (rows - expert_start)
    tl.store(ROW_IN_BUF_ptr + rows, row_in_buf, mask=mask)


@triton.jit
def _swizzle_kernel_lookup(
    SRC_ptr,                 # uint8* src[padded_M, padded_nb]
    DST_ptr,                 # uint8* dst (CUTLASS swizzled)
    ROW_IN_BUF_ptr,          # int32* [M_sorted] — precomputed per-row swizzle row index
    N_BLOCKS: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    PADDED_NB: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_SORTED,
):
    """One program per dst_row; vectorized across N_BLOCKS."""
    dst_row = tl.program_id(0)
    if dst_row >= M_SORTED:
        return

    row_in_buf = tl.load(ROW_IN_BUF_ptr + dst_row)

    mTileIdx = row_in_buf // 128
    outerMIdx = row_in_buf % 32
    innerMIdx = (row_in_buf // 32) % 4
    base_row_off = (mTileIdx * NUM_K_TILES) * 512 + outerMIdx * 16 + innerMIdx * 4

    kIdx = tl.arange(0, BLOCK_K)
    mask = kIdx < N_BLOCKS

    src_base = dst_row * PADDED_NB
    sf_bytes = tl.load(SRC_ptr + src_base + kIdx, mask=mask, other=0)

    kTileIdx = kIdx // 4
    innerKIdx = kIdx % 4
    dst_off = base_row_off + kTileIdx * 512 + innerKIdx
    tl.store(DST_ptr + dst_off, sf_bytes, mask=mask)


def rowmajor_kernel_scales_to_cutlass_swizzled(
    fused_scales_flat: torch.Tensor,
    padded_M: int,
    padded_nb: int,
    M_sorted: int,
    K: int,
    expert_offsets: torch.Tensor,
    blockscale_offsets: torch.Tensor,
    max_tpe_times_topk: int,
    out_buf: torch.Tensor | None = None,
    row_in_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused remap + CUTLASS swizzle.

    Returns out: [max_tpe_times_topk, padded_nb // 4] int32.
    """
    assert fused_scales_flat.dtype == torch.uint8
    n_blocks = K // 16
    num_k_tiles = (K + 63) // 64
    num_experts = expert_offsets.numel() - 1
    device = fused_scales_flat.device

    if out_buf is None:
        out_buf = torch.empty(
            max_tpe_times_topk * padded_nb, dtype=torch.uint8, device=device
        )

    # Build per-row lookup (small, fast).
    if row_in_buf is None:
        row_in_buf = torch.empty(M_sorted, dtype=torch.int32, device=device)
    BLOCK_ROWS = 128
    _build_row_lookup_kernel[((M_sorted + BLOCK_ROWS - 1) // BLOCK_ROWS,)](
        row_in_buf,
        expert_offsets.to(torch.int32),
        blockscale_offsets.to(torch.int32),
        NUM_EXPERTS=num_experts,
        M_SORTED=M_sorted,
        BLOCK_ROWS=BLOCK_ROWS,
    )

    BLOCK_K = 1
    while BLOCK_K < n_blocks:
        BLOCK_K *= 2

    _swizzle_kernel_lookup[(M_sorted,)](
        fused_scales_flat.contiguous(),
        out_buf,
        row_in_buf,
        N_BLOCKS=n_blocks,
        NUM_K_TILES=num_k_tiles,
        PADDED_NB=padded_nb,
        BLOCK_K=BLOCK_K,
        M_SORTED=M_sorted,
    )

    return out_buf.view(torch.int32).reshape(max_tpe_times_topk, padded_nb // 4)
