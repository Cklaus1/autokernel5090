"""Fused unshuffle + topk-weighted-sum kernel for Qwen3-30B-A3B MoE post-GEMM2.

Target op (T2-N ceiling analysis Rank 3, lines 466-478 of the original flow):

    c3_unshuffled = shuffle_rows(c3, c_map)            # scatter: sorted -> token order
    output = (
        c3_unshuffled.view(m, num_topk, k)
        * topk_weights.view(m, num_topk, 1).to(out_dtype)
    ).sum(dim=1)

Semantics of shuffle_rows(src, idx): out[i, :] = src[idx[i], :].
So for row `i = m*topk + t`, c3_unshuffled[i] = c3[c_map[m*topk + t]].
Each destination token m receives exactly `topk` contributions — no atomics
needed; a single program per (m, k_block) does the parallel reduction.

Two-pass memory cost (baseline):
  pass 1 (shuffle_rows):  M*topk*K BF16 read + M*topk*K BF16 write
  pass 2 (weighted sum):  M*topk*K BF16 read + M*K BF16 write
  total:                  3*M*topk*K + M*K BF16 traffic (~12 MB at Qwen3 M=512)

Fused:
  M*topk*K BF16 read (c3)  +  M*topk weight loads  +  M*K BF16 write
  total:                  M*topk*K + M*K BF16 traffic (~4.5 MB at Qwen3 M=512)

Roughly 2.5-3x less HBM traffic at the Qwen3 shape.

Tag: W7_T2N_rank3_unshuffle_weightedsum
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_unshuffle_weightedsum_kernel(
    c3_ptr,                 # [M_sorted, K]  BF16/FP16 (sorted expert order)
    c_map_ptr,              # [M*topk]       int32 (sorted-index per (m, t))
    weights_ptr,            # [M, topk]      same dtype as output
    out_ptr,                # [M, K]         BF16/FP16
    stride_c3_m,
    stride_cmap,
    stride_w_m,
    stride_out_m,
    M,
    K,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid:  (num_tokens_M,  K / BLOCK_K)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # Load the topk sorted-row indices for this destination token.
    # c_map layout: c_map[m*topk + t] = sorted row index.
    cmap_base = pid_m * TOPK
    t_idx = tl.arange(0, TOPK)
    src_rows = tl.load(c_map_ptr + cmap_base + t_idx).to(tl.int64)

    # Load topk weights for this destination token.
    # weights layout: [M, topk], stride_w_m rows.
    w = tl.load(weights_ptr + pid_m * stride_w_m + t_idx).to(tl.float32)

    # Accumulate in FP32. TOPK is small (8 for Qwen3) so we unroll.
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    # Rely on Triton loop-unroll since TOPK is constexpr.
    for t in tl.static_range(0, TOPK):
        row = tl.load(c_map_ptr + cmap_base + t).to(tl.int64)
        ptr = c3_ptr + row * stride_c3_m + offs_k
        v = tl.load(ptr, mask=mask_k, other=0.0).to(tl.float32)
        wt = tl.load(weights_ptr + pid_m * stride_w_m + t).to(tl.float32)
        acc += v * wt

    # Write back.
    out_addrs = out_ptr + pid_m * stride_out_m + offs_k
    tl.store(out_addrs, acc.to(out_ptr.dtype.element_ty), mask=mask_k)


def fused_unshuffle_weightedsum(
    c3: torch.Tensor,               # [M_sorted, K]  BF16/FP16 in sorted order
    c_map: torch.Tensor,            # [M*topk]       int32
    topk_weights: torch.Tensor,     # [M, topk]      cast to out_dtype internally
    M: int,
    topk: int,
    out_dtype: torch.dtype | None = None,
    output: torch.Tensor | None = None,
    BLOCK_K: int = 256,
) -> torch.Tensor:
    """Fused equivalent of:
        c3u = shuffle_rows(c3, c_map)            # [M*topk, K]
        output = (c3u.view(M, topk, K)
                  * topk_weights.view(M, topk, 1).to(out_dtype)).sum(dim=1)

    Writes into `output` if provided; else allocates a new [M, K] tensor.
    """
    assert c3.dim() == 2
    assert c_map.dim() == 1
    assert c_map.numel() == M * topk, (
        f"c_map numel {c_map.numel()} != M*topk {M * topk}"
    )
    assert topk_weights.dim() == 2 and topk_weights.shape == (M, topk), (
        f"topk_weights shape {tuple(topk_weights.shape)} != ({M}, {topk})"
    )
    assert c3.is_cuda and c_map.is_cuda and topk_weights.is_cuda
    K = c3.shape[1]

    if out_dtype is None:
        out_dtype = c3.dtype
    if output is None:
        output = torch.empty((M, K), dtype=out_dtype, device=c3.device)
    else:
        assert output.shape == (M, K) and output.dtype == out_dtype

    # Weights get cast to out_dtype in the reference path; we do FP32 accumulate
    # so we just need them contiguous. Keep dtype as-is (typically BF16 or FP32).
    # The kernel casts to float32 internally.
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    if not c3.is_contiguous():
        c3 = c3.contiguous()
    if not c_map.is_contiguous():
        c_map = c_map.contiguous()

    # Pad TOPK up to a power-of-two if needed (Qwen3 topk=8 already is).
    # For non-power-of-two topks we do not support here; assert.
    assert (topk & (topk - 1)) == 0, (
        f"topk must be power of two (got {topk})"
    )
    # Pick BLOCK_K: for K=2048, BLOCK_K=256 gives 8 k-blocks per token,
    # M*8 = 4096 programs at Qwen3 M=512 -> saturates SMs on SM120.
    if K % BLOCK_K != 0:
        # Fall back to a divisor of K.
        for bk in (256, 128, 64, 32):
            if K % bk == 0:
                BLOCK_K = bk
                break

    grid = (M, triton.cdiv(K, BLOCK_K))

    _fused_unshuffle_weightedsum_kernel[grid](
        c3,
        c_map,
        topk_weights,
        output,
        c3.stride(0),
        c_map.stride(0),
        topk_weights.stride(0),
        output.stride(0),
        M,
        K,
        TOPK=topk,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return output


# ---------------------------------------------------------------------------
# Reference (for correctness tests).
# ---------------------------------------------------------------------------


def ref_unshuffle_weightedsum(
    c3: torch.Tensor,
    c_map: torch.Tensor,
    topk_weights: torch.Tensor,
    M: int,
    topk: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Reference using PyTorch ops matching the exact two-pass path."""
    if out_dtype is None:
        out_dtype = c3.dtype
    K = c3.shape[1]
    # shuffle_rows(c3, c_map) = c3[c_map] (gather along dim 0).
    unshuffled = c3[c_map.long()]  # [M*topk, K]
    out = (
        unshuffled.view(M, topk, K)
        * topk_weights.view(M, topk, 1).to(out_dtype)
    ).sum(dim=1)
    return out.to(out_dtype)
