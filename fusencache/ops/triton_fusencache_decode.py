# SPDX-License-Identifier: Apache-2.0
"""FusenCache Triton decode kernels.

v1 dense: FP8 K + int4 V with GQA head batching and tl.dot.
v3.1 selective: position-list based sparse attention.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.logger import init_logger

logger = init_logger(__name__)


# ====================================================================
# v1 Dense decode — GQA head-batched with tl.dot
# ====================================================================

@triton.jit
def _fc_grouped_decode_stage1(
    Q_ptr,
    KV_cache_ptr,         # uint8 combined [num_blocks, block_size, Hk, slot_size]
    V_scales_ptr,         # float16 [max_slots, Hk]
    Block_table_ptr,      # [batch, max_blocks]
    Seq_lens_ptr,         # [batch]
    Mid_out_ptr,          # float32 [batch, Hq, splits, D+1]
    # Strides
    stride_qb, stride_qh,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b,
    stride_mid_b, stride_mid_h, stride_mid_s,
    # Constants
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,        # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,       # KV tokens per tile
    BLOCK_H: tl.constexpr,        # query heads per program
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,   # Hq // Hk
    Q_HEAD_NUM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    V_PACKED_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Head-batched decode: processes BLOCK_H query heads per program."""
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # Map head group to KV head
    cur_kv_head = cur_head_id // tl.cdiv(KV_GROUP_SIZE, BLOCK_H)

    # Which query heads this program handles
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if KV_GROUP_SIZE > BLOCK_H else KV_GROUP_SIZE
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < Q_HEAD_NUM)

    # Dimension offsets
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    offs_v = tl.arange(0, BLOCK_D // 2)
    mask_v = offs_v < V_PACKED_SIZE

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

    # Online softmax accumulators: [BLOCK_H] for max/sum, [BLOCK_H, D/2] for V
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_even = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)

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

        # === Load K: [BLOCK_D, BLOCK_KV] (transposed for tl.dot) ===
        k_addrs = slot_bases[None, :] + offs_d[:, None]
        k_raw = tl.load(
            KV_cache_ptr + k_addrs,
            mask=(mask_d[:, None]) & (kv_mask[None, :]),
            other=0)
        # FP8 bitcast and convert
        k = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

        # QK^T: [BLOCK_H, BLOCK_D] @ [BLOCK_D, BLOCK_KV] → [BLOCK_H, BLOCK_KV]
        qk = tl.dot(q.to(tl.float32), k)
        qk *= sm_scale
        qk = tl.where(
            mask_h[:, None] & (kv_mask[None, :]), qk, float("-inf"))

        # === Load V: int4 packed, unpack to [BLOCK_KV, D/2] ===
        v_addrs = slot_bases[:, None] + HEAD_DIM + offs_v[None, :]
        v_packed = tl.load(
            KV_cache_ptr + v_addrs,
            mask=(kv_mask[:, None]) & (mask_v[None, :]),
            other=0).to(tl.int32)

        v_lo = v_packed & 0xF
        v_hi = (v_packed >> 4) & 0xF
        v_lo = tl.where(v_lo > 7, v_lo - 16, v_lo)
        v_hi = tl.where(v_hi > 7, v_hi - 16, v_hi)

        # V scales
        flat_slots = block_nums * PAGE_SIZE + page_off
        v_scales = tl.load(
            V_scales_ptr + flat_slots * NUM_KV_HEADS + cur_kv_head,
            mask=kv_mask, other=0).to(tl.float32)

        v_lo_f = v_lo.to(tl.float32) * v_scales[:, None]  # [BLOCK_KV, D/2]
        v_hi_f = v_hi.to(tl.float32) * v_scales[:, None]  # [BLOCK_KV, D/2]

        # === Online softmax ===
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)  # [BLOCK_H]
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])  # [BLOCK_H, BLOCK_KV]

        # P×V: [BLOCK_H, BLOCK_KV] @ [BLOCK_KV, D/2] → [BLOCK_H, D/2]
        acc_even = acc_even * re_scale[:, None] + tl.dot(p.to(tl.float32), v_lo_f)
        acc_odd = acc_odd * re_scale[:, None] + tl.dot(p.to(tl.float32), v_hi_f)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Normalize and store interleaved
    safe_l = tl.where(e_sum > 0.0, e_sum, 1.0)
    acc_even = acc_even / safe_l[:, None]
    acc_odd = acc_odd / safe_l[:, None]

    # Store per-head results: [BLOCK_H, D/2] → interleaved [BLOCK_H, D+1]
    half_d = tl.arange(0, BLOCK_D // 2)
    half_mask = half_d < V_PACKED_SIZE
    lse = e_max + tl.log(safe_l)  # [BLOCK_H]

    # 2D store: all heads at once
    head_indices = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = (head_indices < Q_HEAD_NUM) & mask_h

    mid_bases = (cur_batch * stride_mid_b
                 + head_indices * stride_mid_h
                 + split_kv_id * stride_mid_s)  # [BLOCK_H]

    # Store even dims: mid[h, 2*i] = acc_even[h, i]
    even_addrs = mid_bases[:, None] + half_d[None, :] * 2  # [BLOCK_H, D/2]
    tl.store(Mid_out_ptr + even_addrs, acc_even,
             mask=head_mask[:, None] & half_mask[None, :])

    # Store odd dims: mid[h, 2*i+1] = acc_odd[h, i]
    odd_addrs = mid_bases[:, None] + half_d[None, :] * 2 + 1
    tl.store(Mid_out_ptr + odd_addrs, acc_odd,
             mask=head_mask[:, None] & half_mask[None, :])

    # Store LSE at position HEAD_DIM
    lse_addrs = mid_bases + HEAD_DIM
    tl.store(Mid_out_ptr + lse_addrs, lse, mask=head_mask)


@triton.jit
def _fc_decode_stage2(
    Mid_out_ptr,
    Out_ptr,
    Seq_lens_ptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
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

    out_off = bid * stride_out_b + hid * stride_out_h
    tl.store(Out_ptr + out_off + d_offs, acc / e_sum, mask=d_mask)


def fusencache_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    v_scales: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
) -> torch.Tensor:
    """FusenCache v1 Triton decode with GQA head batching + tl.dot."""
    B, Hq, D = query.shape
    Hk = num_kv_heads
    kv_group_size = Hq // Hk
    block_size = kv_cache.shape[1]
    max_slots = kv_cache.shape[0] * block_size
    v_packed_size = D // 2

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_KV = 32
    BLOCK_H = min(16, kv_group_size)
    NUM_KV_SPLITS = 64

    # Mid output: [B, Hq, splits, D+1]
    mid_out = torch.empty(B, Hq, NUM_KV_SPLITS, D + 1,
                          dtype=torch.float32, device=query.device)

    # Grid: fewer programs thanks to BLOCK_H head batching
    num_head_groups = triton.cdiv(Hq, BLOCK_H)
    grid1 = (B, num_head_groups, NUM_KV_SPLITS)

    _fc_grouped_decode_stage1[grid1](
        query, kv_cache, v_scales, block_table, seq_lens, mid_out,
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
        scale,
        HEAD_DIM=D, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
        BLOCK_H=BLOCK_H, NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size, Q_HEAD_NUM=Hq,
        PAGE_SIZE=block_size, V_PACKED_SIZE=v_packed_size,
        NUM_KV_HEADS=Hk,
        num_warps=4, num_stages=2,
    )

    # Stage 2: reduce across splits
    output = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)
    grid2 = (B, Hq)
    _fc_decode_stage2[grid2](
        mid_out, output, seq_lens,
        HEAD_DIM=D, BLOCK_D=BLOCK_D,
        NUM_KV_SPLITS=NUM_KV_SPLITS, NUM_Q_HEADS=Hq,
        stride_mid_b=mid_out.stride(0), stride_mid_h=mid_out.stride(1),
        stride_mid_s=mid_out.stride(2),
        stride_out_b=output.stride(0), stride_out_h=output.stride(1),
        num_warps=2,
    )

    return output


# ====================================================================
# v3.1: Selective decode — position-list based
# ====================================================================

@triton.jit
def _fc_selective_decode(
    Q_ptr, KV_cache_ptr, V_scales_ptr,
    Positions_ptr, Num_positions_ptr, Block_table_ptr, Out_ptr,
    stride_qb, stride_qh,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b, stride_pos_b,
    sm_scale,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr, KV_GROUP_SIZE: tl.constexpr,
    PAGE_SIZE: tl.constexpr, V_PACKED_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Single-pass selective decode over position list."""
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kv_head = hid // KV_GROUP_SIZE
    num_pos = tl.load(Num_positions_ptr + bid)
    if num_pos <= 0:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    q = tl.load(Q_ptr + bid * stride_qb + hid * stride_qh + d_offs,
                mask=d_mask, other=0.0).to(tl.float32)

    v_offs = tl.arange(0, BLOCK_D // 2)
    v_mask = v_offs < V_PACKED_SIZE
    m_prev = -float("inf")
    l_prev = 0.0
    acc_even = tl.zeros([BLOCK_D // 2], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_D // 2], dtype=tl.float32)
    kv_range = tl.arange(0, BLOCK_KV)
    pos_base = bid * stride_pos_b

    for start_n in range(0, num_pos, BLOCK_KV):
        tile_offs = start_n + kv_range
        tile_mask = tile_offs < num_pos
        positions = tl.load(Positions_ptr + pos_base + tile_offs,
                            mask=tile_mask, other=0)
        page_idx = positions // PAGE_SIZE
        page_off = positions % PAGE_SIZE
        block_nums = tl.load(
            Block_table_ptr + bid * stride_bt_b + page_idx,
            mask=tile_mask, other=0)
        slot_bases = (block_nums * stride_cache_block
                      + page_off * stride_cache_pos
                      + kv_head * stride_cache_head)

        k_addrs = slot_bases[:, None] + d_offs[None, :]
        k_raw = tl.load(KV_cache_ptr + k_addrs,
                         mask=tile_mask[:, None] & d_mask[None, :], other=0)
        k = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
        qk = tl.where(tile_mask, qk, -float("inf"))

        v_addrs = slot_bases[:, None] + HEAD_DIM + v_offs[None, :]
        v_packed = tl.load(KV_cache_ptr + v_addrs,
                            mask=tile_mask[:, None] & v_mask[None, :], other=0).to(tl.int32)
        v_lo = v_packed & 0xF
        v_hi = (v_packed >> 4) & 0xF
        v_lo = tl.where(v_lo > 7, v_lo - 16, v_lo)
        v_hi = tl.where(v_hi > 7, v_hi - 16, v_hi)
        flat_slots = block_nums * PAGE_SIZE + page_off
        v_scales = tl.load(V_scales_ptr + flat_slots * NUM_KV_HEADS + kv_head,
                            mask=tile_mask, other=0).to(tl.float32)
        v_lo_f = v_lo.to(tl.float32) * v_scales[:, None]
        v_hi_f = v_hi.to(tl.float32) * v_scales[:, None]

        m_new = tl.maximum(tl.max(qk, 0), m_prev)
        rescale = tl.exp(m_prev - m_new)
        p = tl.exp(qk - m_new)
        acc_even = acc_even * rescale + tl.sum(p[:, None] * v_lo_f, axis=0)
        acc_odd = acc_odd * rescale + tl.sum(p[:, None] * v_hi_f, axis=0)
        l_prev = l_prev * rescale + tl.sum(p, 0)
        m_prev = m_new

    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out_base = bid * stride_qb + hid * stride_qh
    half_d = tl.arange(0, BLOCK_D // 2)
    half_mask = half_d < V_PACKED_SIZE
    tl.store(Out_ptr + out_base + half_d * 2, acc_even / safe_l, mask=half_mask)
    tl.store(Out_ptr + out_base + half_d * 2 + 1, acc_odd / safe_l, mask=half_mask)


def fusencache_selective_decode(
    query, kv_cache, v_scales, positions, num_positions,
    block_table, scale, num_kv_heads,
):
    B, Hq, D = query.shape
    Hk = num_kv_heads
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    output = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)
    grid = (B, Hq)
    _fc_selective_decode[grid](
        query, kv_cache, v_scales, positions, num_positions,
        block_table, output,
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), positions.stride(0),
        scale, HEAD_DIM=D, BLOCK_D=BLOCK_D, BLOCK_KV=32,
        KV_GROUP_SIZE=Hq // Hk, PAGE_SIZE=block_size,
        V_PACKED_SIZE=D // 2, NUM_KV_HEADS=Hk,
        num_warps=4, num_stages=2,
    )
    return output
