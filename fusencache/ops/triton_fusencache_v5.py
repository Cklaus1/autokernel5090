# SPDX-License-Identifier: Apache-2.0
"""FusenCache v5 Triton kernel: K=int8 V=int4 per-block-16 scales.

Cache layout: [k_int8 (D bytes) | v_int4 (D/2 bytes)] per slot.
K scales + V scales in separate tensor: (max_slots, Hk, D/16, 2) float16.

K dequant: k_int8 * k_scale_per_block
V dequant: (v_nibble - 7.5) * v_scale_per_block
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.logger import init_logger

logger = init_logger(__name__)

BLOCK_SCALE = 16  # elements per scale block


@triton.jit
def _fc_v5_decode_stage1(
    Q_ptr,
    KV_cache_ptr,       # uint8 [num_blocks, bs, Hk, slot_size]
    Scales_ptr,         # float16 [max_slots, Hk, D/16, 2]
    Block_table_ptr,
    Seq_lens_ptr,
    Mid_out_ptr,        # float32 [B, Hq, splits, D+1]
    # Strides
    stride_qb, stride_qh,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b,
    stride_mid_b, stride_mid_h, stride_mid_s,
    stride_sc_slot, stride_sc_head, stride_sc_block, stride_sc_kv,
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
    V_PACKED_SIZE: tl.constexpr,  # D/2
    NUM_SCALE_BLOCKS: tl.constexpr,  # D/16
):
    """K=int8 V=int4 with per-block-16 scales, head-batched."""
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
    kv_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = kv_len * split_kv_id
    split_end = tl.minimum(split_start + kv_len, seq_len)

    if split_start >= split_end:
        return

    # Load Q: [BLOCK_H, BLOCK_D]
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q_ptr + offs_q, mask=mask_h[:, None] & mask_d[None, :], other=0.0)

    # V half-dim offsets
    v_offs = tl.arange(0, BLOCK_D // 2)
    v_mask = v_offs < V_PACKED_SIZE

    # Accumulators
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_even = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)
    acc_odd = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)

    kv_range = tl.arange(0, BLOCK_KV)

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        # Page table
        block_nums = tl.load(
            Block_table_ptr + cur_batch * stride_bt_b + kv_offs // PAGE_SIZE,
            mask=kv_mask, other=0)
        page_off = kv_offs % PAGE_SIZE
        slot_bases = (block_nums * stride_cache_block
                      + page_off * stride_cache_pos
                      + cur_kv_head * stride_cache_head)

        # === K: int8 → exact per-block-16 dequant → tl.dot ===
        k_addrs = slot_bases[None, :] + offs_d[:, None]
        k_raw = tl.load(KV_cache_ptr + k_addrs,
                         mask=mask_d[:, None] & kv_mask[None, :], other=0)
        k_i8 = k_raw.to(tl.int8).to(tl.float32)  # [BLOCK_D, BLOCK_KV]

        # Load per-block-16 K scales and expand to per-element
        flat_slots = block_nums * PAGE_SIZE + page_off
        sc_base = flat_slots * stride_sc_slot + cur_kv_head * stride_sc_head

        # Build per-dim scale: each group of 16 dims shares one scale
        # offs_d // 16 gives the scale block index for each dim
        sc_block_idx = offs_d // 16  # [BLOCK_D]
        k_sc_addrs = sc_base[None, :] + sc_block_idx[:, None] * stride_sc_block
        k_sc = tl.load(Scales_ptr + k_sc_addrs,
                        mask=mask_d[:, None] & kv_mask[None, :],
                        other=1.0).to(tl.float32)  # [BLOCK_D, BLOCK_KV]

        # Dequant K: int8 * per-element scale
        k_scaled = k_i8 * k_sc  # [BLOCK_D, BLOCK_KV]

        # QK^T
        qk = tl.dot(q.to(tl.float32), k_scaled) * sm_scale
        qk = tl.where(mask_h[:, None] & kv_mask[None, :], qk, float("-inf"))

        # === V: int4 packed ===
        v_addrs = slot_bases[:, None] + HEAD_DIM + v_offs[None, :]
        v_packed = tl.load(KV_cache_ptr + v_addrs,
                            mask=kv_mask[:, None] & v_mask[None, :], other=0).to(tl.int32)

        v_lo = (v_packed & 0xF).to(tl.float32) - 7.5
        v_hi = ((v_packed >> 4) & 0xF).to(tl.float32) - 7.5

        # V scales: per-block-16, expand to per-element (half-dim)
        # v_lo covers even dims (0,2,4..), v_hi covers odd dims (1,3,5..)
        # Both dim i and i+1 share the same scale block (i//16)
        v_sc_block_idx = v_offs // 8  # each v_offs maps to 2 dims, /16 per block = /8
        v_sc_addrs = sc_base[:, None] + v_sc_block_idx[None, :] * stride_sc_block + stride_sc_kv
        v_sc = tl.load(Scales_ptr + v_sc_addrs,
                        mask=kv_mask[:, None] & v_mask[None, :],
                        other=1.0).to(tl.float32)  # [BLOCK_KV, D/2]

        v_lo = v_lo * v_sc
        v_hi = v_hi * v_sc

        # Online softmax
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])

        acc_even = acc_even * re_scale[:, None] + tl.dot(p.to(tl.float32), v_lo)
        acc_odd = acc_odd * re_scale[:, None] + tl.dot(p.to(tl.float32), v_hi)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Store
    safe_l = tl.where(e_sum > 0.0, e_sum, 1.0)
    acc_even = acc_even / safe_l[:, None]
    acc_odd = acc_odd / safe_l[:, None]
    lse = e_max + tl.log(safe_l)

    head_indices = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = (head_indices < Q_HEAD_NUM) & mask_h
    mid_bases = (cur_batch * stride_mid_b + head_indices * stride_mid_h
                 + split_kv_id * stride_mid_s)

    half_d = tl.arange(0, BLOCK_D // 2)
    half_mask = half_d < V_PACKED_SIZE
    tl.store(Mid_out_ptr + mid_bases[:, None] + half_d[None, :] * 2,
             acc_even, mask=head_mask[:, None] & half_mask[None, :])
    tl.store(Mid_out_ptr + mid_bases[:, None] + half_d[None, :] * 2 + 1,
             acc_odd, mask=head_mask[:, None] & half_mask[None, :])
    tl.store(Mid_out_ptr + mid_bases + HEAD_DIM, lse, mask=head_mask)


@triton.jit
def _fc_v5_decode_stage2(
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

    for s in range(NUM_KV_SPLITS):
        sl = tl.cdiv(seq_len, NUM_KV_SPLITS)
        if tl.minimum(sl * s + sl, seq_len) > sl * s:
            off = mid_base + s * stride_mid_s
            tv = tl.load(Mid_out_ptr + off + d_offs, mask=d_mask, other=0.0)
            tlogic = tl.load(Mid_out_ptr + off + HEAD_DIM)
            n = tl.maximum(tlogic, e_max)
            r = tl.exp(e_max - n)
            acc = acc * r + tl.exp(tlogic - n) * tv
            e_sum = e_sum * r + tl.exp(tlogic - n)
            e_max = n

    tl.store(Out_ptr + bid * stride_out_b + hid * stride_out_h + d_offs,
             acc / e_sum, mask=d_mask)


def fusencache_v5_decode(query, kv_cache, scales, block_table, seq_lens,
                          scale, num_kv_heads):
    """K8V4B16 Triton decode."""
    B, Hq, D = query.shape
    Hk = num_kv_heads
    kv_group_size = Hq // Hk
    block_size = kv_cache.shape[1]

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_KV = 16   # reduced from 32 to fit shared memory
    BLOCK_H = min(8, kv_group_size)  # reduced from 16
    NUM_KV_SPLITS = 64

    mid_out = torch.empty(B, Hq, NUM_KV_SPLITS, D + 1,
                          dtype=torch.float32, device=query.device)

    num_head_groups = triton.cdiv(Hq, BLOCK_H)
    grid1 = (B, num_head_groups, NUM_KV_SPLITS)

    _fc_v5_decode_stage1[grid1](
        query, kv_cache, scales, block_table, seq_lens, mid_out,
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
        scales.stride(0), scales.stride(1), scales.stride(2), scales.stride(3),
        scale,
        HEAD_DIM=D, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
        BLOCK_H=BLOCK_H, NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size, Q_HEAD_NUM=Hq,
        PAGE_SIZE=block_size, NUM_KV_HEADS=Hk,
        V_PACKED_SIZE=D // 2,
        NUM_SCALE_BLOCKS=D // BLOCK_SCALE,
        num_warps=4, num_stages=1,  # reduced to fit shared memory
    )

    output = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)
    grid2 = (B, Hq)
    _fc_v5_decode_stage2[grid2](
        mid_out, output, seq_lens,
        HEAD_DIM=D, BLOCK_D=BLOCK_D,
        NUM_KV_SPLITS=NUM_KV_SPLITS, NUM_Q_HEADS=Hq,
        stride_mid_b=mid_out.stride(0), stride_mid_h=mid_out.stride(1),
        stride_mid_s=mid_out.stride(2),
        stride_out_b=output.stride(0), stride_out_h=output.stride(1),
        num_warps=2,
    )
    return output
