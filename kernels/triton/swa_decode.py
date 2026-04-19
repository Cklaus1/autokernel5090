"""Sparse sliding-window-attention decode kernel for Gemma4 26B (BF16 KV).

Target: Gemma4 sliding layers (25 of 30 layers). Decode step (M=1 per batch).
At seq_len > window, we only attend to the last `window` tokens. This kernel
skips pages that fall entirely outside the window -- programs don't launch
iterations for them, saving both HBM traffic and FLOPs.

Structure is a direct port of `kernels/fp8_decode_attention.py`:
  - Stage 1: split-KV paged decode with online softmax, one page per iter.
  - Stage 2: merge splits via LSE.

Differences vs fp8_decode_attention.py:
  - BF16 KV (FP8 folded in later; Discovery #37).
  - Page loop starts at `start_page = max(0, seq_len - window) // page_size`.
  - Leading page applies a left mask `abs_pos >= seq_len - window`.
  - Split-K partitions the *window* pages, not full seq_len.

Call `swa_decode_attention(q, k_cache, v_cache, block_table, seq_lens,
window, sm_scale=None, logits_soft_cap=0.0)`.
"""

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _swa_decode_stage1(
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    K_scale_ptr,
    V_scale_ptr,
    Mid_out_ptr,
    stride_qb, stride_qh,
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    stride_bt_b,
    stride_mid_b, stride_mid_h, stride_mid_s,
    sm_scale,
    WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    LOGITS_SOFT_CAP: tl.constexpr = 0.0,
    PER_HEAD_SCALE: tl.constexpr = 0,
    FP8_KV: tl.constexpr = 0,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head_id // tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if KV_GROUP_SIZE > BLOCK_H else KV_GROUP_SIZE
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < NUM_Q_HEADS)

    seq_len = tl.load(Seq_lens_ptr + cur_batch)

    # Sliding-window range: attend only to [window_start, seq_len).
    # window_start is clamped at 0 (no effect when seq_len <= window).
    window_start = tl.maximum(seq_len - WINDOW, 0)
    start_page = window_start // PAGE_SIZE
    end_page = tl.cdiv(seq_len, PAGE_SIZE)
    window_pages = end_page - start_page  # >= 1 when seq_len > 0

    if window_pages <= 0:
        return

    # Partition the *window* page range across splits.
    pages_per_split = tl.cdiv(window_pages, NUM_KV_SPLITS)
    local_start = pages_per_split * split_kv_id
    local_end = tl.minimum(local_start + pages_per_split, window_pages)

    if local_start >= local_end:
        return

    page_start = start_page + local_start
    page_end = start_page + local_end

    # Load Q.
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    q_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q_ptr + q_addrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Load FP8 scales (only meaningful when FP8_KV=1; compiler DCEs for BF16 path).
    if FP8_KV:
        if PER_HEAD_SCALE:
            k_scale = tl.load(K_scale_ptr + cur_kv_head).to(tl.float32)
            v_scale = tl.load(V_scale_ptr + cur_kv_head).to(tl.float32)
        else:
            k_scale = tl.load(K_scale_ptr).to(tl.float32)
            v_scale = tl.load(V_scale_ptr).to(tl.float32)
        # Fold k_scale into sm_scale -- saves one multiply per K element.
        qk_scale = sm_scale * k_scale
    else:
        qk_scale = sm_scale
        v_scale = 1.0  # unused in BF16 path

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    pos_range = tl.arange(0, PAGE_SIZE)

    for page_idx in range(page_start, page_end):
        phys_block = tl.load(Block_table_ptr + cur_batch * stride_bt_b + page_idx)
        phys_block_i64 = phys_block.to(tl.int64)

        abs_pos = page_idx * PAGE_SIZE + pos_range
        # Valid tokens = in [window_start, seq_len).
        pos_mask = (abs_pos < seq_len) & (abs_pos >= window_start)

        k_page_base = (phys_block_i64 * stride_kb
                       + cur_kv_head.to(tl.int64) * stride_kh)
        k_addrs = k_page_base + pos_range[:, None].to(tl.int64) * stride_kp + offs_d[None, :]
        k_addrs = tl.multiple_of(k_addrs, 16)  # W5_B2: hint 16-byte alignment for vectorised BF16/FP8 loads
        k_load_mask = pos_mask[:, None] & mask_d[None, :]
        if FP8_KV:
            # other=0.0 is not safely representable for FP8 e4m3 -- use explicit mask-then-zero.
            k_fp8 = tl.load(K_cache_ptr + k_addrs, mask=k_load_mask)
            k_vals = k_fp8.to(tl.float32)
            k_vals = tl.where(k_load_mask, k_vals, 0.0)
        else:
            k_vals = tl.load(K_cache_ptr + k_addrs, mask=k_load_mask, other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(k_vals)) * qk_scale

        if LOGITS_SOFT_CAP > 0:
            _x = qk / LOGITS_SOFT_CAP
            _e2x = tl.math.exp(_x * 2.0)
            _tanh = 1.0 - 2.0 / (_e2x + 1.0)
            qk = LOGITS_SOFT_CAP * _tanh

        qk = tl.where(mask_h[:, None] & pos_mask[None, :], qk, float("-inf"))

        v_page_base = (phys_block_i64 * stride_vb
                       + cur_kv_head.to(tl.int64) * stride_vh)
        v_addrs = v_page_base + pos_range[:, None].to(tl.int64) * stride_vp + offs_d[None, :]
        v_addrs = tl.multiple_of(v_addrs, 16)  # W5_B2: hint 16-byte alignment for vectorised BF16/FP8 loads
        v_load_mask = pos_mask[:, None] & mask_d[None, :]
        if FP8_KV:
            v_fp8 = tl.load(V_cache_ptr + v_addrs, mask=v_load_mask)
            v_vals = v_fp8.to(tl.float32) * v_scale
            v_vals = tl.where(v_load_mask, v_vals, 0.0)
        else:
            v_vals = tl.load(V_cache_ptr + v_addrs, mask=v_load_mask, other=0.0).to(tl.float32)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc = acc * re_scale[:, None] + tl.dot(p.to(tl.float32), v_vals)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    safe_l = tl.where(e_sum > 0.0, e_sum, 1.0)
    lse = e_max + tl.log(safe_l)
    acc = acc / safe_l[:, None]

    head_indices = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = (head_indices < NUM_Q_HEADS) & mask_h
    mid_bases = (cur_batch * stride_mid_b + head_indices * stride_mid_h
                 + split_kv_id * stride_mid_s)

    tl.store(Mid_out_ptr + mid_bases[:, None] + offs_d[None, :],
             acc, mask=head_mask[:, None] & mask_d[None, :])
    tl.store(Mid_out_ptr + mid_bases + HEAD_DIM, lse, mask=head_mask)


@triton.jit
def _swa_decode_stage2(
    Mid_out_ptr, Out_ptr, Seq_lens_ptr,
    WINDOW: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    stride_mid_b, stride_mid_h, stride_mid_s,
    stride_out_b, stride_out_h,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    seq_len = tl.load(Seq_lens_ptr + bid)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    window_start = tl.maximum(seq_len - WINDOW, 0)
    start_page = window_start // PAGE_SIZE
    end_page = tl.cdiv(seq_len, PAGE_SIZE)
    window_pages = end_page - start_page

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    mid_base = bid.to(tl.int64) * stride_mid_b + hid.to(tl.int64) * stride_mid_h

    for s in range(NUM_KV_SPLITS):
        pages_per_split = tl.cdiv(window_pages, NUM_KV_SPLITS)
        local_start = pages_per_split * s
        local_end = tl.minimum(local_start + pages_per_split, window_pages)
        if local_start < local_end:
            off = mid_base + s * stride_mid_s
            tv = tl.load(Mid_out_ptr + off + d_offs, mask=d_mask, other=0.0)
            tlse = tl.load(Mid_out_ptr + off + HEAD_DIM)
            n = tl.maximum(tlse, e_max)
            r = tl.exp(e_max - n)
            acc = acc * r + tl.exp(tlse - n) * tv
            e_sum = e_sum * r + tl.exp(tlse - n)
            e_max = n

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    tl.store(Out_ptr + bid * stride_out_b + hid * stride_out_h + d_offs,
             acc / safe_sum, mask=d_mask)


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def swa_decode_attention(
    q: torch.Tensor,              # [batch, num_q_heads, head_dim] BF16
    k_cache: torch.Tensor,        # [num_blocks, page_size, num_kv_heads, head_dim] BF16 or FP8
    v_cache: torch.Tensor,        # [num_blocks, page_size, num_kv_heads, head_dim] BF16 or FP8
    block_table: torch.Tensor,    # [batch, max_num_blocks] int32
    seq_lens: torch.Tensor,       # [batch] int32
    window: int,                  # sliding window size (in tokens)
    sm_scale: float = None,
    logits_soft_cap: float = 0.0,
    num_kv_splits: int = 0,       # 0 => auto
    k_scale: torch.Tensor = None,  # FP32 scalar tensor or [num_kv_heads]; required for FP8 KV
    v_scale: torch.Tensor = None,  # FP32 scalar tensor or [num_kv_heads]; required for FP8 KV
) -> torch.Tensor:
    """Sparse sliding-window decode attention.

    Supports BF16/FP16 KV (q and KV match) and FP8 KV (Q stays BF16/FP16,
    KV in float8_e4m3fn or float8_e5m2 with per-tensor scales).
    """
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    kv_group_size = num_q_heads // num_kv_heads

    assert k_cache.dtype in (torch.bfloat16, torch.float16) + _FP8_DTYPES
    assert v_cache.dtype == k_cache.dtype
    fp8_kv = k_cache.dtype in _FP8_DTYPES
    if fp8_kv:
        assert q.dtype in (torch.bfloat16, torch.float16), \
            f"Q must be BF16/FP16 for FP8 KV path, got {q.dtype}"
        assert k_scale is not None and v_scale is not None, \
            "FP8 KV requires k_scale and v_scale"
    else:
        assert q.dtype == k_cache.dtype

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Use `window` as the upper bound for the number of pages any single
    # sequence touches (the kernel internally clamps to actual seq_len, so
    # launching extra splits when seq_len < window is correct but wasteful
    # — avoided in the non-capture path via the max()).
    # Avoid host sync during CUDA graph capture: .item() is unsupported there.
    if torch.cuda.is_current_stream_capturing():
        eff_len = window  # upper bound
    else:
        max_seq = int(seq_lens.max().item())
        eff_len = min(max_seq, window)
    # Window can span at most ceil(window / page_size) + 1 pages (due to misalignment).
    window_pages = (eff_len + page_size - 1) // page_size + 1

    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_H = min(kv_group_size, 4)
    BLOCK_H = max(triton.next_power_of_2(BLOCK_H), 16)  # Triton dot needs M>=16
    num_head_blocks = triton.cdiv(num_q_heads, min(kv_group_size, 4))
    # Actually BLOCK_H as used in addressing should match. Pad to 16 only if tl.dot needs.
    # The existing fp8 kernel uses unpadded BLOCK_H=4 via tl.dot; but tl.dot requires M>=16
    # in some triton versions. Keep 4 for GQA=4; tl.dot supports M=4 on recent triton.
    BLOCK_H = min(kv_group_size, 4)
    BLOCK_H = triton.next_power_of_2(BLOCK_H)
    num_head_blocks = triton.cdiv(num_q_heads, BLOCK_H)

    if num_kv_splits == 0:
        base_blocks = batch * num_head_blocks
        target_blocks = 170 * 4
        if base_blocks >= target_blocks:
            num_kv_splits = 1
        else:
            num_kv_splits = max(1, min(window_pages, target_blocks // max(base_blocks, 1)))
            pages_per_split = window_pages // max(num_kv_splits, 1)
            while pages_per_split < 2 and num_kv_splits > 1:
                num_kv_splits = num_kv_splits // 2
                pages_per_split = window_pages // max(num_kv_splits, 1)

    mid_out = torch.empty(
        (batch, num_q_heads, num_kv_splits, head_dim + 1),
        dtype=torch.float32, device=q.device)
    output = torch.empty_like(q)

    # Scale tensors (dummy for BF16 path so the kernel launch signature is uniform).
    if fp8_kv:
        k_scale_t = k_scale.to(device=q.device, dtype=torch.float32)
        v_scale_t = v_scale.to(device=q.device, dtype=torch.float32)
        per_head_scale = 1 if k_scale_t.numel() > 1 else 0
    else:
        # Dummy 1-element tensor; kernel won't read it when FP8_KV=0 (DCE'd).
        k_scale_t = torch.ones(1, dtype=torch.float32, device=q.device)
        v_scale_t = k_scale_t
        per_head_scale = 0

    grid_stage1 = (batch, num_head_blocks, num_kv_splits)
    _swa_decode_stage1[grid_stage1](
        q,
        k_cache, v_cache,
        block_table, seq_lens,
        k_scale_t, v_scale_t,
        mid_out,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
        sm_scale,
        WINDOW=window,
        HEAD_DIM=head_dim,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=page_size,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        KV_GROUP_SIZE=kv_group_size,
        NUM_Q_HEADS=num_q_heads,
        LOGITS_SOFT_CAP=logits_soft_cap,
        PER_HEAD_SCALE=per_head_scale,
        FP8_KV=1 if fp8_kv else 0,
        num_stages=1,
        num_warps=8,  # W5_B2: 4→8 warps; wider issue for BW-bound HBM loads (30%→40-50% BW target)
    )

    grid_stage2 = (batch, num_q_heads)
    _swa_decode_stage2[grid_stage2](
        mid_out, output, seq_lens,
        WINDOW=window,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        BLOCK_D=BLOCK_D,
        NUM_KV_SPLITS=num_kv_splits,
        stride_mid_b=mid_out.stride(0),
        stride_mid_h=mid_out.stride(1),
        stride_mid_s=mid_out.stride(2),
        stride_out_b=output.stride(0),
        stride_out_h=output.stride(1),
    )

    return output


def bf16_swa_decode_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    window: int,
    sm_scale: float = None,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    """PyTorch reference: gather KV, slice window, full softmax attention in FP32."""
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    kv_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    output = torch.zeros_like(q, dtype=torch.float32)

    for b in range(batch):
        sl = int(seq_lens[b].item())
        if sl == 0:
            continue

        n_pages = (sl + page_size - 1) // page_size
        pages = block_table[b, :n_pages]

        k_gathered, v_gathered = [], []
        for i, page_id in enumerate(pages):
            start = i * page_size
            end = min(start + page_size, sl)
            count = end - start
            k_gathered.append(k_cache[page_id, :count])
            v_gathered.append(v_cache[page_id, :count])
        k_all = torch.cat(k_gathered, dim=0).float()  # [sl, kv, d]
        v_all = torch.cat(v_gathered, dim=0).float()

        # Slice window.
        ws = max(0, sl - window)
        k_all = k_all[ws:sl]
        v_all = v_all[ws:sl]

        q_b = q[b].float()
        for h in range(num_q_heads):
            kv_h = h // kv_group_size
            scores = (q_b[h] @ k_all[:, kv_h, :].T) * sm_scale
            if logits_soft_cap > 0:
                scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)
            weights = torch.softmax(scores, dim=-1)
            output[b, h] = weights @ v_all[:, kv_h, :]

    return output.to(q.dtype)
