# SPDX-License-Identifier: Apache-2.0
"""FusenCache v4 Triton decode: FP8 K + FP8 V, head-batched, graph-safe.

Simpler than v1 — no int4 unpack, no V scales. Both K and V are FP8.
Combined cache layout: [k_fp8 (D bytes) | v_fp8 (D bytes)] per slot.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.logger import init_logger

logger = init_logger(__name__)


@triton.jit
def _fc_v4_decode_stage1(
    Q_ptr,
    KV_cache_ptr,         # uint8 combined [num_blocks, bs, Hk, 2*D]
    Block_table_ptr,
    Seq_lens_ptr,
    Mid_out_ptr,          # float32 [B, Hq, splits, D+1]
    # Strides
    stride_qb, stride_qh,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b,
    stride_mid_b, stride_mid_h, stride_mid_s,
    # Constants
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Head-batched FP8+FP8 decode with tl.dot."""
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head_id // tl.cdiv(KV_GROUP_SIZE, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if KV_GROUP_SIZE > BLOCK_H else KV_GROUP_SIZE
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < Q_HEAD_NUM)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM

    seq_len = tl.load(Seq_lens_ptr + cur_batch)
    kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = kv_len_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

    if split_start >= split_end:
        return

    # Load Q: [BLOCK_H, BLOCK_D]
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q_ptr + offs_q,
                mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    # Accumulators
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    kv_range = tl.arange(0, BLOCK_KV)

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        # Page table lookup
        block_nums = tl.load(
            Block_table_ptr + cur_batch * stride_bt_b + kv_offs // PAGE_SIZE,
            mask=kv_mask, other=0)
        page_off = kv_offs % PAGE_SIZE
        slot_bases = (block_nums * stride_cache_block
                      + page_off * stride_cache_pos
                      + cur_kv_head * stride_cache_head)

        # === K: [BLOCK_D, BLOCK_KV] (transposed for tl.dot) ===
        k_addrs = slot_bases[None, :] + offs_d[:, None]
        k_raw = tl.load(KV_cache_ptr + k_addrs,
                         mask=(mask_d[:, None]) & (kv_mask[None, :]), other=0)
        k = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

        # QK^T: [BLOCK_H, BLOCK_D] @ [BLOCK_D, BLOCK_KV] → [BLOCK_H, BLOCK_KV]
        qk = tl.dot(q.to(tl.float32), k) * sm_scale
        qk = tl.where(mask_h[:, None] & (kv_mask[None, :]), qk, float("-inf"))

        # === V: [BLOCK_KV, BLOCK_D] — FP8, offset by HEAD_DIM ===
        v_addrs = slot_bases[:, None] + HEAD_DIM + offs_d[None, :]
        v_raw = tl.load(KV_cache_ptr + v_addrs,
                         mask=(kv_mask[:, None]) & (mask_d[None, :]), other=0)
        v = v_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

        # Online softmax
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])

        # P×V: [BLOCK_H, BLOCK_KV] @ [BLOCK_KV, BLOCK_D] → [BLOCK_H, BLOCK_D]
        acc = acc * re_scale[:, None] + tl.dot(p.to(tl.float32), v)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Store results
    safe_l = tl.where(e_sum > 0.0, e_sum, 1.0)
    acc = acc / safe_l[:, None]
    lse = e_max + tl.log(safe_l)

    head_indices = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = (head_indices < Q_HEAD_NUM) & mask_h

    mid_bases = (cur_batch * stride_mid_b
                 + head_indices * stride_mid_h
                 + split_kv_id * stride_mid_s)

    # Store output: [BLOCK_H, BLOCK_D]
    out_addrs = mid_bases[:, None] + offs_d[None, :]
    tl.store(Mid_out_ptr + out_addrs, acc,
             mask=head_mask[:, None] & mask_d[None, :])

    # Store LSE at position HEAD_DIM
    lse_addrs = mid_bases + HEAD_DIM
    tl.store(Mid_out_ptr + lse_addrs, lse, mask=head_mask)


@triton.jit
def _fc_v4_decode_stage2(
    Mid_out_ptr, Out_ptr, Seq_lens_ptr,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr, NUM_Q_HEADS: tl.constexpr,
    stride_mid_b, stride_mid_h, stride_mid_s,
    stride_out_b, stride_out_h,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    seq_len = tl.load(Seq_lens_ptr + bid)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    mid_base = bid * stride_mid_b + hid * stride_mid_h

    for split_id in range(NUM_KV_SPLITS):
        split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = split_len * split_id
        split_end = tl.minimum(split_start + split_len, seq_len)
        if split_end > split_start:
            off = mid_base + split_id * stride_mid_s
            tv = tl.load(Mid_out_ptr + off + d_offs, mask=d_mask, other=0.0)
            tlogic = tl.load(Mid_out_ptr + off + HEAD_DIM)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc = acc * old_scale + tl.exp(tlogic - n_e_max) * tv
            e_sum = e_sum * old_scale + tl.exp(tlogic - n_e_max)
            e_max = n_e_max

    tl.store(Out_ptr + bid * stride_out_b + hid * stride_out_h + d_offs,
             acc / e_sum, mask=d_mask)


def fusencache_v4_decode(
    query: torch.Tensor,       # [B, Hq, D]
    kv_cache: torch.Tensor,    # [num_blocks, bs, Hk, 2*D] uint8
    block_table: torch.Tensor, # [B, max_blocks]
    seq_lens: torch.Tensor,    # [B]
    scale: float,
    num_kv_heads: int,
) -> torch.Tensor:
    """FusenCache v4 Triton decode — FP8+FP8, head-batched, graph-safe."""
    B, Hq, D = query.shape
    Hk = num_kv_heads
    kv_group_size = Hq // Hk
    block_size = kv_cache.shape[1]

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_KV = 32
    BLOCK_H = min(16, kv_group_size)
    NUM_KV_SPLITS = 64

    # Use pre-allocated buffer if available (avoids allocation during graph capture)
    mid_key = (B, Hq, NUM_KV_SPLITS, D + 1)
    if not hasattr(fusencache_v4_decode, '_mid_buf') or \
       fusencache_v4_decode._mid_buf.shape != mid_key or \
       fusencache_v4_decode._mid_buf.device != query.device:
        fusencache_v4_decode._mid_buf = torch.empty(
            *mid_key, dtype=torch.float32, device=query.device)
    mid_out = fusencache_v4_decode._mid_buf

    num_head_groups = triton.cdiv(Hq, BLOCK_H)
    grid1 = (B, num_head_groups, NUM_KV_SPLITS)

    _fc_v4_decode_stage1[grid1](
        query, kv_cache, block_table, seq_lens, mid_out,
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
        scale,
        HEAD_DIM=D, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
        BLOCK_H=BLOCK_H, NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size, Q_HEAD_NUM=Hq,
        PAGE_SIZE=block_size, NUM_KV_HEADS=Hk,
        num_warps=4, num_stages=2,
    )

    out_key = (B, Hq, D)
    if not hasattr(fusencache_v4_decode, '_out_buf') or \
       fusencache_v4_decode._out_buf.shape != out_key or \
       fusencache_v4_decode._out_buf.device != query.device or \
       fusencache_v4_decode._out_buf.dtype != query.dtype:
        fusencache_v4_decode._out_buf = torch.empty(
            *out_key, dtype=query.dtype, device=query.device)
    output = fusencache_v4_decode._out_buf
    grid2 = (B, Hq)
    _fc_v4_decode_stage2[grid2](
        mid_out, output, seq_lens,
        HEAD_DIM=D, BLOCK_D=BLOCK_D,
        NUM_KV_SPLITS=NUM_KV_SPLITS, NUM_Q_HEADS=Hq,
        stride_mid_b=mid_out.stride(0), stride_mid_h=mid_out.stride(1),
        stride_mid_s=mid_out.stride(2),
        stride_out_b=output.stride(0), stride_out_h=output.stride(1),
        num_warps=2,
    )
    return output
