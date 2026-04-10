"""Universal KV cache decode kernel — one kernel for all quantization specs.

Uses tl.constexpr dispatch so Triton eliminates dead branches at compile time.
The compiled PTX for K_BITS=4 is identical to a hand-written int4 kernel.
"""

import torch
import triton
import triton.language as tl


# ===== Stage 1: Split-KV decode with data-driven dequant =====

@triton.jit
def _universal_decode_stage1(
    Q_ptr,
    KV_cache_ptr,
    Scales_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Mid_out_ptr,
    # Strides
    stride_qb, stride_qh,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b,
    stride_mid_b, stride_mid_h, stride_mid_s,
    stride_sc_slot, stride_sc_head, stride_sc_block, stride_sc_kv,
    # Scalar
    sm_scale,
    # Layout constexprs
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    # ---- Data-driven constexprs ----
    K_BITS: tl.constexpr,
    V_BITS: tl.constexpr,
    K_OFFSET: tl.constexpr,       # 0.0 for int8, 7.5 for int4, 1.5 for int2
    V_OFFSET: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,  # 0 = no scales
    SCALE_BLOCK_V: tl.constexpr,
    SCALE_BLOCK_STORE: tl.constexpr,  # storage granularity = min(K, V)
    K_REGION_BYTES: tl.constexpr, # bytes of K data per head in slot
    V_REGION_START: tl.constexpr, # byte offset where V starts in slot
    LOGITS_SOFT_CAP: tl.constexpr = 0.0,  # 0 = disabled, >0 = tanh soft cap
):
    # ---- Fixed: grid mapping ----
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head_id // tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if KV_GROUP_SIZE > BLOCK_H else KV_GROUP_SIZE
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < Q_HEAD_NUM)

    seq_len = tl.load(Seq_lens_ptr + cur_batch)
    kv_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = kv_len * split_kv_id
    split_end = tl.minimum(split_start + kv_len, seq_len)

    if split_start >= split_end:
        return

    # ---- Data-driven: load Q based on K packing ----
    HALF_D: tl.constexpr = BLOCK_D // 2
    half_offs = tl.arange(0, HALF_D)

    if K_BITS >= 8:
        # Full-dim Q load
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < HEAD_DIM
        q_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_d[None, :]
        q = tl.load(Q_ptr + q_addrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    else:
        # Split Q into even/odd for packed K dot product
        half_mask = half_offs < (HEAD_DIM // 2)
        even_idx = half_offs * 2
        odd_idx = half_offs * 2 + 1
        q_even_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + even_idx[None, :]
        q_odd_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + odd_idx[None, :]
        q_even = tl.load(Q_ptr + q_even_addrs, mask=mask_h[:, None] & half_mask[None, :], other=0.0).to(tl.float32)
        q_odd = tl.load(Q_ptr + q_odd_addrs, mask=mask_h[:, None] & half_mask[None, :], other=0.0).to(tl.float32)

    # ---- Fixed: accumulators ----
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)

    if V_BITS >= 8:
        offs_d = tl.arange(0, BLOCK_D)
        acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    elif V_BITS == 4:
        acc_even = tl.zeros([BLOCK_H, HALF_D], dtype=tl.float32)
        acc_odd = tl.zeros([BLOCK_H, HALF_D], dtype=tl.float32)
    elif V_BITS == 2:
        QUARTER_D: tl.constexpr = BLOCK_D // 4
        acc_v0 = tl.zeros([BLOCK_H, QUARTER_D], dtype=tl.float32)
        acc_v1 = tl.zeros([BLOCK_H, QUARTER_D], dtype=tl.float32)
        acc_v2 = tl.zeros([BLOCK_H, QUARTER_D], dtype=tl.float32)
        acc_v3 = tl.zeros([BLOCK_H, QUARTER_D], dtype=tl.float32)

    kv_range = tl.arange(0, BLOCK_KV)

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        # ---- Fixed: page table lookup ----
        block_nums = tl.load(
            Block_table_ptr + cur_batch * stride_bt_b + kv_offs // PAGE_SIZE,
            mask=kv_mask, other=0)
        page_off = kv_offs % PAGE_SIZE
        # Cast to int64 to prevent overflow at large seq_len * batch
        block_nums_i64 = block_nums.to(tl.int64)
        slot_bases = (block_nums_i64 * stride_cache_block
                      + page_off * stride_cache_pos
                      + cur_kv_head * stride_cache_head)

        # Scale base addresses (int64 to avoid overflow)
        flat_slots = block_nums_i64 * PAGE_SIZE + page_off
        sc_base = flat_slots * stride_sc_slot + cur_kv_head * stride_sc_head

        # ============ K: load + dequant + QK^T ============

        if K_BITS == 8:
            # uint8: 1 byte per element, load as unsigned then dequant
            k_offs = tl.arange(0, BLOCK_D)
            k_mask = k_offs < HEAD_DIM
            k_addrs = slot_bases[None, :] + k_offs[:, None]
            k_raw = tl.load(KV_cache_ptr + k_addrs,
                            mask=k_mask[:, None] & kv_mask[None, :], other=0)
            k_vals = k_raw.to(tl.float32)  # uint8 codes 0-255

            # Dequant: (code - offset) * scale
            if SCALE_BLOCK_K > 0:
                sc_idx = k_offs // SCALE_BLOCK_STORE
                k_sc_addrs = sc_base[None, :] + sc_idx[:, None] * stride_sc_block
                k_sc = tl.load(Scales_ptr + k_sc_addrs,
                               mask=k_mask[:, None] & kv_mask[None, :],
                               other=1.0).to(tl.float32)
                k_vals = (k_vals - K_OFFSET) * k_sc

            # QK^T: direct dot
            qk = tl.dot(q, k_vals) * sm_scale

        elif K_BITS == 4:
            # int4: 2 values per byte, nibble unpack
            half_mask_k = half_offs < (HEAD_DIM // 2)
            k_addrs = slot_bases[:, None] + half_offs[None, :]
            k_packed = tl.load(KV_cache_ptr + k_addrs,
                               mask=kv_mask[:, None] & half_mask_k[None, :], other=0).to(tl.int32)
            k_lo = (k_packed & 0xF).to(tl.float32) - K_OFFSET
            k_hi = ((k_packed >> 4) & 0xF).to(tl.float32) - K_OFFSET

            if SCALE_BLOCK_K > 0:
                sc_idx = half_offs // (SCALE_BLOCK_STORE // 2)
                k_sc_addrs = sc_base[:, None] + sc_idx[None, :] * stride_sc_block
                k_sc = tl.load(Scales_ptr + k_sc_addrs,
                               mask=kv_mask[:, None] & half_mask_k[None, :],
                               other=1.0).to(tl.float32)
                k_lo = k_lo * k_sc
                k_hi = k_hi * k_sc

            # QK^T: split even/odd dot
            qk = (tl.dot(q_even, tl.trans(k_lo)) + tl.dot(q_odd, tl.trans(k_hi))) * sm_scale

        elif K_BITS == 2:
            # int2: 4 values per byte
            quarter = HEAD_DIM // 4
            q_offs = tl.arange(0, BLOCK_D // 4)
            q_mask = q_offs < quarter
            k_addrs = slot_bases[:, None] + q_offs[None, :]
            k_packed = tl.load(KV_cache_ptr + k_addrs,
                               mask=kv_mask[:, None] & q_mask[None, :], other=0).to(tl.int32)
            k_0 = (k_packed & 0x3).to(tl.float32) - K_OFFSET
            k_1 = ((k_packed >> 2) & 0x3).to(tl.float32) - K_OFFSET
            k_2 = ((k_packed >> 4) & 0x3).to(tl.float32) - K_OFFSET
            k_3 = ((k_packed >> 6) & 0x3).to(tl.float32) - K_OFFSET

            if SCALE_BLOCK_K > 0:
                sc_idx = q_offs // (SCALE_BLOCK_STORE // 4)
                k_sc_addrs = sc_base[:, None] + sc_idx[None, :] * stride_sc_block
                k_sc = tl.load(Scales_ptr + k_sc_addrs,
                               mask=kv_mask[:, None] & q_mask[None, :],
                               other=1.0).to(tl.float32)
                k_0 = k_0 * k_sc
                k_1 = k_1 * k_sc
                k_2 = k_2 * k_sc
                k_3 = k_3 * k_sc

            # Load Q in 4 quarter-dim slices
            q0_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + (q_offs * 4)[None, :]
            q1_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + (q_offs * 4 + 1)[None, :]
            q2_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + (q_offs * 4 + 2)[None, :]
            q3_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + (q_offs * 4 + 3)[None, :]
            q_0 = tl.load(Q_ptr + q0_addrs, mask=mask_h[:, None] & q_mask[None, :], other=0.0).to(tl.float32)
            q_1 = tl.load(Q_ptr + q1_addrs, mask=mask_h[:, None] & q_mask[None, :], other=0.0).to(tl.float32)
            q_2 = tl.load(Q_ptr + q2_addrs, mask=mask_h[:, None] & q_mask[None, :], other=0.0).to(tl.float32)
            q_3 = tl.load(Q_ptr + q3_addrs, mask=mask_h[:, None] & q_mask[None, :], other=0.0).to(tl.float32)

            qk = (tl.dot(q_0, tl.trans(k_0)) + tl.dot(q_1, tl.trans(k_1))
                  + tl.dot(q_2, tl.trans(k_2)) + tl.dot(q_3, tl.trans(k_3))) * sm_scale

        # ---- Logits soft cap: tanh(qk / cap) * cap ----
        # tanh(x) = 1 - 2/(exp(2x) + 1) since tl.math has no tanh
        if LOGITS_SOFT_CAP > 0:
            _x = qk / LOGITS_SOFT_CAP
            _e2x = tl.math.exp(_x * 2.0)
            _tanh = 1.0 - 2.0 / (_e2x + 1.0)
            qk = LOGITS_SOFT_CAP * _tanh

        qk = tl.where(mask_h[:, None] & kv_mask[None, :], qk, float("-inf"))

        # ============ V: load + dequant ============

        if V_BITS == 8:
            v_d_offs = tl.arange(0, BLOCK_D)
            v_d_mask = v_d_offs < HEAD_DIM
            v_addrs = slot_bases[:, None] + V_REGION_START + v_d_offs[None, :]
            v_raw = tl.load(KV_cache_ptr + v_addrs,
                            mask=kv_mask[:, None] & v_d_mask[None, :], other=0)
            v_vals = v_raw.to(tl.float32)  # uint8 codes 0-255

            if SCALE_BLOCK_V > 0:
                v_sc_idx = v_d_offs // SCALE_BLOCK_STORE
                v_sc_addrs = sc_base[:, None] + v_sc_idx[None, :] * stride_sc_block + stride_sc_kv
                v_sc = tl.load(Scales_ptr + v_sc_addrs,
                               mask=kv_mask[:, None] & v_d_mask[None, :],
                               other=1.0).to(tl.float32)
                v_vals = (v_vals - V_OFFSET) * v_sc

        elif V_BITS == 4:
            v_half_mask = half_offs < (HEAD_DIM // 2)
            v_addrs = slot_bases[:, None] + V_REGION_START + half_offs[None, :]
            v_packed = tl.load(KV_cache_ptr + v_addrs,
                               mask=kv_mask[:, None] & v_half_mask[None, :], other=0).to(tl.int32)
            v_lo = (v_packed & 0xF).to(tl.float32) - V_OFFSET
            v_hi = ((v_packed >> 4) & 0xF).to(tl.float32) - V_OFFSET

            if SCALE_BLOCK_V > 0:
                v_sc_idx = half_offs // (SCALE_BLOCK_STORE // 2)
                v_sc_addrs = sc_base[:, None] + v_sc_idx[None, :] * stride_sc_block + stride_sc_kv
                v_sc = tl.load(Scales_ptr + v_sc_addrs,
                               mask=kv_mask[:, None] & v_half_mask[None, :],
                               other=1.0).to(tl.float32)
                v_lo = v_lo * v_sc
                v_hi = v_hi * v_sc

        elif V_BITS == 2:
            v_quarter = HEAD_DIM // 4
            v_q_offs = tl.arange(0, BLOCK_D // 4)
            v_q_mask = v_q_offs < v_quarter
            v_addrs = slot_bases[:, None] + V_REGION_START + v_q_offs[None, :]
            v_packed = tl.load(KV_cache_ptr + v_addrs,
                               mask=kv_mask[:, None] & v_q_mask[None, :], other=0).to(tl.int32)
            v_0 = (v_packed & 0x3).to(tl.float32) - V_OFFSET
            v_1 = ((v_packed >> 2) & 0x3).to(tl.float32) - V_OFFSET
            v_2 = ((v_packed >> 4) & 0x3).to(tl.float32) - V_OFFSET
            v_3 = ((v_packed >> 6) & 0x3).to(tl.float32) - V_OFFSET

            if SCALE_BLOCK_V > 0:
                v_sc_idx = v_q_offs // (SCALE_BLOCK_STORE // 4)
                v_sc_addrs = sc_base[:, None] + v_sc_idx[None, :] * stride_sc_block + stride_sc_kv
                v_sc = tl.load(Scales_ptr + v_sc_addrs,
                               mask=kv_mask[:, None] & v_q_mask[None, :],
                               other=1.0).to(tl.float32)
                v_0 = v_0 * v_sc
                v_1 = v_1 * v_sc
                v_2 = v_2 * v_sc
                v_3 = v_3 * v_sc

        # ============ Fixed: online softmax + V accumulate ============

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])

        if V_BITS == 8:
            acc = acc * re_scale[:, None] + tl.dot(p.to(tl.float32), v_vals)
        elif V_BITS == 4:
            acc_even = acc_even * re_scale[:, None] + tl.dot(p.to(tl.float32), v_lo)
            acc_odd = acc_odd * re_scale[:, None] + tl.dot(p.to(tl.float32), v_hi)
        elif V_BITS == 2:
            # V2: 4 quarter-dim tensors, each [BLOCK_KV, D/4]
            # v_0 has dims 4i, v_1 has 4i+1, v_2 has 4i+2, v_3 has 4i+3
            # Accumulate separately, interleave when storing
            pf = p.to(tl.float32)
            acc_v0 = acc_v0 * re_scale[:, None] + tl.dot(pf, v_0)
            acc_v1 = acc_v1 * re_scale[:, None] + tl.dot(pf, v_1)
            acc_v2 = acc_v2 * re_scale[:, None] + tl.dot(pf, v_2)
            acc_v3 = acc_v3 * re_scale[:, None] + tl.dot(pf, v_3)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # ============ Fixed: normalize and store ============

    safe_l = tl.where(e_sum > 0.0, e_sum, 1.0)
    lse = e_max + tl.log(safe_l)

    head_indices = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = (head_indices < Q_HEAD_NUM) & mask_h
    mid_bases = (cur_batch * stride_mid_b + head_indices * stride_mid_h
                 + split_kv_id * stride_mid_s)

    if V_BITS >= 8:
        acc = acc / safe_l[:, None]
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < HEAD_DIM
        tl.store(Mid_out_ptr + mid_bases[:, None] + d_offs[None, :],
                 acc, mask=head_mask[:, None] & d_mask[None, :])
    elif V_BITS == 4:
        acc_even = acc_even / safe_l[:, None]
        acc_odd = acc_odd / safe_l[:, None]
        half_d = tl.arange(0, HALF_D)
        half_d_mask = half_d < (HEAD_DIM // 2)
        tl.store(Mid_out_ptr + mid_bases[:, None] + half_d[None, :] * 2,
                 acc_even, mask=head_mask[:, None] & half_d_mask[None, :])
        tl.store(Mid_out_ptr + mid_bases[:, None] + half_d[None, :] * 2 + 1,
                 acc_odd, mask=head_mask[:, None] & half_d_mask[None, :])
    elif V_BITS == 2:
        # Normalize each quarter-dim accumulator
        acc_v0 = acc_v0 / safe_l[:, None]
        acc_v1 = acc_v1 / safe_l[:, None]
        acc_v2 = acc_v2 / safe_l[:, None]
        acc_v3 = acc_v3 / safe_l[:, None]
        # Interleave: output dim 4i from acc_v0[i], 4i+1 from acc_v1[i], etc.
        q_d = tl.arange(0, BLOCK_D // 4)
        q_d_mask = q_d < (HEAD_DIM // 4)
        tl.store(Mid_out_ptr + mid_bases[:, None] + q_d[None, :] * 4,
                 acc_v0, mask=head_mask[:, None] & q_d_mask[None, :])
        tl.store(Mid_out_ptr + mid_bases[:, None] + q_d[None, :] * 4 + 1,
                 acc_v1, mask=head_mask[:, None] & q_d_mask[None, :])
        tl.store(Mid_out_ptr + mid_bases[:, None] + q_d[None, :] * 4 + 2,
                 acc_v2, mask=head_mask[:, None] & q_d_mask[None, :])
        tl.store(Mid_out_ptr + mid_bases[:, None] + q_d[None, :] * 4 + 3,
                 acc_v3, mask=head_mask[:, None] & q_d_mask[None, :])

    tl.store(Mid_out_ptr + mid_bases + HEAD_DIM, lse, mask=head_mask)


# ===== Stage 2: reduce across splits (identical for all specs) =====

@triton.jit
def _universal_decode_stage2(
    Mid_out_ptr, Out_ptr, Seq_lens_ptr,
    HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
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
    # Cast to int64 to prevent overflow for large batch * head * split strides
    mid_base = bid.to(tl.int64) * stride_mid_b + hid.to(tl.int64) * stride_mid_h

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

    # Guard against seq_len=0 (padding entries in CUDA graph batches):
    # when no splits contributed, e_sum is 0 and acc/e_sum would be NaN.
    # Output zero instead, which is safe for padded entries.
    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    tl.store(Out_ptr + bid * stride_out_b + hid * stride_out_h + d_offs,
             acc / safe_sum, mask=d_mask)


# ===== Store kernel: quantize + pack + scatter into paged KV cache =====
#
# Strategy: For each (token, head), load the full HEAD_DIM FP16 values,
# quantize per scale-block, then pack into bytes.
#
# For sub-byte packing (4-bit, 2-bit), we can't gather from register vectors
# in Triton, so we load the source data in packed layout (pairs/quads)
# and quantize each element with its block's scale. We compute scales first
# in a pass over the full vector, store them, then re-load source in packed
# order and quantize using stored scales.

@triton.jit
def _store_compute_scales(
    src_ptr, scales_ptr,
    src_base, sc_base,
    stride_sc_block, stride_sc_kv,
    kv_index,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OFFSET: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    SCALE_BLOCK_STORE: tl.constexpr,
):
    """Compute per-block scales from FP source and store them."""
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    vals = tl.load(src_ptr + src_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    NUM_BLOCKS: tl.constexpr = BLOCK_D // SCALE_BLOCK
    block_id = d_offs // SCALE_BLOCK

    for sb in range(NUM_BLOCKS):
        if sb < HEAD_DIM // SCALE_BLOCK:
            bmask = (block_id == sb) & d_mask
            absmax = tl.max(tl.where(bmask, tl.abs(vals), 0.0))
            scale = absmax / OFFSET
            # Store scale (with repeat for mismatched block sizes)
            repeat = SCALE_BLOCK // SCALE_BLOCK_STORE
            for r in range(0, repeat if repeat > 0 else 1):
                sc_idx = sb * repeat + r
                tl.store(scales_ptr + sc_base + sc_idx * stride_sc_block + kv_index * stride_sc_kv,
                         scale.to(tl.float16))


@triton.jit
def _store_quantize_pack_8bit(
    src_ptr, cache_ptr, scales_ptr,
    src_base, cache_base, sc_base, region_start,
    stride_sc_block, stride_sc_kv,
    kv_index,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OFFSET: tl.constexpr,
    LEVELS_MINUS_1: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    SCALE_BLOCK_STORE: tl.constexpr,
):
    """8-bit: load full vector, quantize, store directly."""
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    vals = tl.load(src_ptr + src_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Read scales and quantize
    sc_idx = (d_offs // SCALE_BLOCK) * (SCALE_BLOCK // SCALE_BLOCK_STORE)
    sc = tl.load(scales_ptr + sc_base + sc_idx * stride_sc_block + kv_index * stride_sc_kv,
                 mask=d_mask, other=1.0).to(tl.float32)
    safe_sc = tl.where(sc > 1e-8, sc, 1e-8)

    max_code = LEVELS_MINUS_1 + 0.0
    codes = tl.extra.cuda.libdevice.rint(
        tl.minimum(tl.maximum(vals / safe_sc + OFFSET, 0.0), max_code)
    ).to(tl.uint8)

    tl.store(cache_ptr + cache_base + region_start + d_offs, codes, mask=d_mask)


@triton.jit
def _store_quantize_pack_4bit(
    src_ptr, cache_ptr, scales_ptr,
    src_base, cache_base, sc_base, region_start,
    stride_sc_block, stride_sc_kv,
    kv_index,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OFFSET: tl.constexpr,
    LEVELS_MINUS_1: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    SCALE_BLOCK_STORE: tl.constexpr,
):
    """4-bit: load even/odd pairs, quantize, pack into bytes."""
    HALF_D: tl.constexpr = BLOCK_D // 2
    half_offs = tl.arange(0, HALF_D)
    half_mask = half_offs < (HEAD_DIM // 2)
    even_idx = half_offs * 2
    odd_idx = half_offs * 2 + 1

    val_even = tl.load(src_ptr + src_base + even_idx, mask=half_mask, other=0.0).to(tl.float32)
    val_odd = tl.load(src_ptr + src_base + odd_idx, mask=half_mask, other=0.0).to(tl.float32)

    # Scale lookup: even and odd elements in same pair are always in same scale block
    # (scale_block >= 16, so adjacent elements share a scale)
    sc_idx = (even_idx // SCALE_BLOCK) * (SCALE_BLOCK // SCALE_BLOCK_STORE)
    sc = tl.load(scales_ptr + sc_base + sc_idx * stride_sc_block + kv_index * stride_sc_kv,
                 mask=half_mask, other=1.0).to(tl.float32)
    safe_sc = tl.where(sc > 1e-8, sc, 1e-8)

    max_code = LEVELS_MINUS_1 + 0.0
    code_even = tl.extra.cuda.libdevice.rint(
        tl.minimum(tl.maximum(val_even / safe_sc + OFFSET, 0.0), max_code)
    ).to(tl.int32)
    code_odd = tl.extra.cuda.libdevice.rint(
        tl.minimum(tl.maximum(val_odd / safe_sc + OFFSET, 0.0), max_code)
    ).to(tl.int32)

    packed = (code_even | (code_odd << 4)).to(tl.uint8)
    tl.store(cache_ptr + cache_base + region_start + half_offs, packed, mask=half_mask)


@triton.jit
def _store_quantize_pack_2bit(
    src_ptr, cache_ptr, scales_ptr,
    src_base, cache_base, sc_base, region_start,
    stride_sc_block, stride_sc_kv,
    kv_index,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OFFSET: tl.constexpr,
    LEVELS_MINUS_1: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    SCALE_BLOCK_STORE: tl.constexpr,
):
    """2-bit: load quads, quantize, pack 4 values per byte."""
    QUARTER_D: tl.constexpr = BLOCK_D // 4
    q_offs = tl.arange(0, QUARTER_D)
    q_mask = q_offs < (HEAD_DIM // 4)
    idx0 = q_offs * 4
    idx1 = q_offs * 4 + 1
    idx2 = q_offs * 4 + 2
    idx3 = q_offs * 4 + 3

    v0 = tl.load(src_ptr + src_base + idx0, mask=q_mask, other=0.0).to(tl.float32)
    v1 = tl.load(src_ptr + src_base + idx1, mask=q_mask, other=0.0).to(tl.float32)
    v2 = tl.load(src_ptr + src_base + idx2, mask=q_mask, other=0.0).to(tl.float32)
    v3 = tl.load(src_ptr + src_base + idx3, mask=q_mask, other=0.0).to(tl.float32)

    sc_idx = (idx0 // SCALE_BLOCK) * (SCALE_BLOCK // SCALE_BLOCK_STORE)
    sc = tl.load(scales_ptr + sc_base + sc_idx * stride_sc_block + kv_index * stride_sc_kv,
                 mask=q_mask, other=1.0).to(tl.float32)
    safe_sc = tl.where(sc > 1e-8, sc, 1e-8)

    max_code = LEVELS_MINUS_1 + 0.0
    c0 = tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(v0 / safe_sc + OFFSET, 0.0), max_code)).to(tl.int32)
    c1 = tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(v1 / safe_sc + OFFSET, 0.0), max_code)).to(tl.int32)
    c2 = tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(v2 / safe_sc + OFFSET, 0.0), max_code)).to(tl.int32)
    c3 = tl.extra.cuda.libdevice.rint(tl.minimum(tl.maximum(v3 / safe_sc + OFFSET, 0.0), max_code)).to(tl.int32)

    packed = (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).to(tl.uint8)
    tl.store(cache_ptr + cache_base + region_start + q_offs, packed, mask=q_mask)


@triton.jit
def _universal_store_kernel(
    Key_ptr, Value_ptr,
    KV_cache_ptr, Scales_ptr, Slot_mapping_ptr,
    stride_kv_n, stride_kv_h,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_sc_slot, stride_sc_head, stride_sc_block, stride_sc_kv,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    K_BITS: tl.constexpr, V_BITS: tl.constexpr,
    K_OFFSET: tl.constexpr, V_OFFSET: tl.constexpr,
    K_LEVELS_MINUS_1: tl.constexpr, V_LEVELS_MINUS_1: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr, SCALE_BLOCK_V: tl.constexpr,
    SCALE_BLOCK_STORE: tl.constexpr,
    V_REGION_START: tl.constexpr,
):
    """Store one token, one KV head: quantize K and V, pack, scatter to cache.

    Two-pass approach:
    1. Compute and store per-block scales (full HEAD_DIM pass)
    2. Re-load source in packed layout, quantize using stored scales, pack, store

    This avoids the Triton limitation of not being able to gather from register vectors.
    """
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    slot = tl.load(Slot_mapping_ptr + token_id)
    if slot < 0:
        return

    blk_idx = slot // PAGE_SIZE
    blk_off = slot % PAGE_SIZE

    cache_base = (blk_idx.to(tl.int64) * stride_cache_block
                  + blk_off.to(tl.int64) * stride_cache_pos
                  + head_id.to(tl.int64) * stride_cache_head)
    sc_base = slot.to(tl.int64) * stride_sc_slot + head_id.to(tl.int64) * stride_sc_head

    k_src_base = token_id.to(tl.int64) * stride_kv_n + head_id.to(tl.int64) * stride_kv_h
    v_src_base = token_id.to(tl.int64) * stride_kv_n + head_id.to(tl.int64) * stride_kv_h

    # ---- Pass 1: compute and store scales ----
    _store_compute_scales(
        Key_ptr, Scales_ptr, k_src_base, sc_base,
        stride_sc_block, stride_sc_kv, 0,
        HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=K_OFFSET,
        SCALE_BLOCK=SCALE_BLOCK_K, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
    )
    _store_compute_scales(
        Value_ptr, Scales_ptr, v_src_base, sc_base,
        stride_sc_block, stride_sc_kv, 1,
        HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=V_OFFSET,
        SCALE_BLOCK=SCALE_BLOCK_V, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
    )

    # ---- Pass 2: quantize + pack + store K ----
    if K_BITS == 8:
        _store_quantize_pack_8bit(
            Key_ptr, KV_cache_ptr, Scales_ptr,
            k_src_base, cache_base, sc_base, 0,
            stride_sc_block, stride_sc_kv, 0,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=K_OFFSET,
            LEVELS_MINUS_1=K_LEVELS_MINUS_1,
            SCALE_BLOCK=SCALE_BLOCK_K, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
        )
    elif K_BITS == 4:
        _store_quantize_pack_4bit(
            Key_ptr, KV_cache_ptr, Scales_ptr,
            k_src_base, cache_base, sc_base, 0,
            stride_sc_block, stride_sc_kv, 0,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=K_OFFSET,
            LEVELS_MINUS_1=K_LEVELS_MINUS_1,
            SCALE_BLOCK=SCALE_BLOCK_K, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
        )
    elif K_BITS == 2:
        _store_quantize_pack_2bit(
            Key_ptr, KV_cache_ptr, Scales_ptr,
            k_src_base, cache_base, sc_base, 0,
            stride_sc_block, stride_sc_kv, 0,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=K_OFFSET,
            LEVELS_MINUS_1=K_LEVELS_MINUS_1,
            SCALE_BLOCK=SCALE_BLOCK_K, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
        )

    # ---- Pass 2: quantize + pack + store V ----
    if V_BITS == 8:
        _store_quantize_pack_8bit(
            Value_ptr, KV_cache_ptr, Scales_ptr,
            v_src_base, cache_base, sc_base, V_REGION_START,
            stride_sc_block, stride_sc_kv, 1,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=V_OFFSET,
            LEVELS_MINUS_1=V_LEVELS_MINUS_1,
            SCALE_BLOCK=SCALE_BLOCK_V, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
        )
    elif V_BITS == 4:
        _store_quantize_pack_4bit(
            Value_ptr, KV_cache_ptr, Scales_ptr,
            v_src_base, cache_base, sc_base, V_REGION_START,
            stride_sc_block, stride_sc_kv, 1,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=V_OFFSET,
            LEVELS_MINUS_1=V_LEVELS_MINUS_1,
            SCALE_BLOCK=SCALE_BLOCK_V, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
        )
    elif V_BITS == 2:
        _store_quantize_pack_2bit(
            Value_ptr, KV_cache_ptr, Scales_ptr,
            v_src_base, cache_base, sc_base, V_REGION_START,
            stride_sc_block, stride_sc_kv, 1,
            HEAD_DIM=HEAD_DIM, BLOCK_D=BLOCK_D, OFFSET=V_OFFSET,
            LEVELS_MINUS_1=V_LEVELS_MINUS_1,
            SCALE_BLOCK=SCALE_BLOCK_V, SCALE_BLOCK_STORE=SCALE_BLOCK_STORE,
        )
