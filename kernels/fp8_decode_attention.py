"""FP8-native paged decode attention kernel for SM120 (RTX 5090).

Reads K/V directly from paged KV cache as float8_e4m3fn, dequantizes to FP32
in registers, computes attention with online softmax + split-K reduction.

Key advantage over FA2: half the memory traffic (1 byte vs 2 bytes per element).
FA2 on SM120 does NOT support FP8 KV, so this is the only path to FP8 decode.

Architecture:
  - Page-aligned inner loop: each iteration processes one full page (contiguous)
  - No per-element page table lookup: one lookup per page
  - Split-K across pages for parallelism

Supports:
  - Paged KV cache (block_table indirection)
  - GQA (arbitrary group_size)
  - Split-K for long sequences
  - Logits soft cap (tanh, Gemma4 requires cap=50.0)
  - head_dim=256 (sliding) and head_dim=512 (global)
  - Per-tensor or per-head FP8 scales
"""

import math
import torch
import triton
import triton.language as tl


# =====================================================================
# Stage 1: Split-KV decode — page-aligned FP8 loads
# =====================================================================

@triton.jit
def _fp8_decode_stage1(
    Q_ptr,           # [batch, num_q_heads, head_dim] BF16
    K_cache_ptr,     # [num_blocks, block_size, num_kv_heads, head_dim] FP8
    V_cache_ptr,     # [num_blocks, block_size, num_kv_heads, head_dim] FP8
    Block_table_ptr, # [batch, max_num_blocks] int32
    Seq_lens_ptr,    # [batch] int32
    K_scale_ptr,     # scalar or [num_kv_heads] float32
    V_scale_ptr,     # scalar or [num_kv_heads] float32
    Mid_out_ptr,     # [batch, num_q_heads, num_splits, head_dim+1] FP32
    # Q strides (in elements)
    stride_qb, stride_qh,
    # K/V cache strides (in elements — 1 element = 1 byte for FP8)
    stride_kb, stride_kp, stride_kh,
    stride_vb, stride_vp, stride_vh,
    # Block table strides
    stride_bt_b,
    # Mid output strides
    stride_mid_b, stride_mid_h, stride_mid_s,
    # Scalars
    sm_scale,
    # Constexprs
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,       # next_power_of_2(HEAD_DIM)
    PAGE_SIZE: tl.constexpr,     # KV page size (16)
    BLOCK_H: tl.constexpr,       # Q heads per block (for GQA grouping)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # num_q_heads // num_kv_heads
    NUM_Q_HEADS: tl.constexpr,
    LOGITS_SOFT_CAP: tl.constexpr = 0.0,
    PER_HEAD_SCALE: tl.constexpr = 0,  # 0=per-tensor, 1=per-head
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    # Map block to KV head and Q head range
    cur_kv_head = cur_head_id // tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
    VALID_BLOCK_H: tl.constexpr = BLOCK_H if KV_GROUP_SIZE > BLOCK_H else KV_GROUP_SIZE
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < NUM_Q_HEADS)

    # Sequence length -> number of pages
    seq_len = tl.load(Seq_lens_ptr + cur_batch)
    num_pages = tl.cdiv(seq_len, PAGE_SIZE)

    # Split range in pages
    pages_per_split = tl.cdiv(num_pages, NUM_KV_SPLITS)
    page_start = pages_per_split * split_kv_id
    page_end = tl.minimum(page_start + pages_per_split, num_pages)

    if page_start >= page_end:
        return

    # Load Q vector(s) into registers [BLOCK_H, BLOCK_D] as FP32
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    q_addrs = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q_ptr + q_addrs, mask=mask_h[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Load FP8 scales
    if PER_HEAD_SCALE:
        k_scale = tl.load(K_scale_ptr + cur_kv_head).to(tl.float32)
        v_scale = tl.load(V_scale_ptr + cur_kv_head).to(tl.float32)
    else:
        k_scale = tl.load(K_scale_ptr).to(tl.float32)
        v_scale = tl.load(V_scale_ptr).to(tl.float32)

    # Fold k_scale into sm_scale to save a multiply per element
    qk_scale = sm_scale * k_scale

    # Accumulators
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    pos_range = tl.arange(0, PAGE_SIZE)  # [0, 1, ..., PAGE_SIZE-1]

    # Page-aligned iteration: one page table lookup per page
    for page_idx in range(page_start, page_end):
        # Single page table lookup for this page
        phys_block = tl.load(Block_table_ptr + cur_batch * stride_bt_b + page_idx)
        phys_block_i64 = phys_block.to(tl.int64)

        # Absolute positions in this page
        abs_pos = page_idx * PAGE_SIZE + pos_range
        pos_mask = abs_pos < seq_len

        # K base address for this physical block + KV head (contiguous read!)
        # Layout: [num_blocks, block_size, num_kv_heads, head_dim]
        k_page_base = (phys_block_i64 * stride_kb
                       + cur_kv_head.to(tl.int64) * stride_kh)

        # Load K: we want [BLOCK_D, PAGE_SIZE] for QK^T = [H, D] @ [D, S]
        # K[pos, dim] -> load as [pos, dim] then transpose
        k_addrs = k_page_base + pos_range[:, None].to(tl.int64) * stride_kp + offs_d[None, :]
        k_load_mask = pos_mask[:, None] & mask_d[None, :]
        k_fp8 = tl.load(K_cache_ptr + k_addrs, mask=k_load_mask)
        k_vals = k_fp8.to(tl.float32)
        k_vals = tl.where(k_load_mask, k_vals, 0.0)

        # QK^T: [BLOCK_H, BLOCK_D] @ [BLOCK_D, PAGE_SIZE]
        qk = tl.dot(q, tl.trans(k_vals)) * qk_scale

        # Logits soft cap: tanh(qk / cap) * cap
        if LOGITS_SOFT_CAP > 0:
            _x = qk / LOGITS_SOFT_CAP
            _e2x = tl.math.exp(_x * 2.0)
            _tanh = 1.0 - 2.0 / (_e2x + 1.0)
            qk = LOGITS_SOFT_CAP * _tanh

        qk = tl.where(mask_h[:, None] & pos_mask[None, :], qk, float("-inf"))

        # Load V: [PAGE_SIZE, BLOCK_D]
        v_page_base = (phys_block_i64 * stride_vb
                       + cur_kv_head.to(tl.int64) * stride_vh)
        v_addrs = v_page_base + pos_range[:, None].to(tl.int64) * stride_vp + offs_d[None, :]
        v_load_mask = pos_mask[:, None] & mask_d[None, :]
        v_fp8 = tl.load(V_cache_ptr + v_addrs, mask=v_load_mask)
        v_vals = v_fp8.to(tl.float32) * v_scale
        v_vals = tl.where(v_load_mask, v_vals, 0.0)

        # Online softmax + accumulate
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])

        acc = acc * re_scale[:, None] + tl.dot(p.to(tl.float32), v_vals)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    # Normalize and store partial results
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


# =====================================================================
# Stage 2: Reduce across splits
# =====================================================================

@triton.jit
def _fp8_decode_stage2(
    Mid_out_ptr, Out_ptr, Seq_lens_ptr,
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

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    mid_base = bid.to(tl.int64) * stride_mid_b + hid.to(tl.int64) * stride_mid_h

    for s in range(NUM_KV_SPLITS):
        num_pages = tl.cdiv(seq_len, 16)  # PAGE_SIZE assumed 16
        pages_per_split = tl.cdiv(num_pages, NUM_KV_SPLITS)
        page_start = pages_per_split * s
        page_end = tl.minimum(page_start + pages_per_split, num_pages)
        if page_start < page_end:
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


# =====================================================================
# Python wrapper
# =====================================================================

def fp8_decode_attention(
    q: torch.Tensor,              # [batch, num_q_heads, head_dim] BF16
    k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim] FP8
    v_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim] FP8
    block_table: torch.Tensor,    # [batch, max_num_blocks] int32
    seq_lens: torch.Tensor,       # [batch] int32
    k_scale: torch.Tensor,        # scalar or [num_kv_heads] float32
    v_scale: torch.Tensor,        # scalar or [num_kv_heads] float32
    sm_scale: float = None,
    logits_soft_cap: float = 0.0,
    num_kv_splits: int = 0,       # 0 = auto
) -> torch.Tensor:
    """FP8-native paged decode attention.

    Args:
        q: Query tensor [batch, num_q_heads, head_dim] in BF16.
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim] in FP8 E4M3.
        v_cache: Value cache, same layout as k_cache, FP8 E4M3.
        block_table: Page table [batch, max_num_blocks] mapping logical blocks to physical.
        seq_lens: Sequence lengths [batch].
        k_scale: FP8 dequant scale for keys (scalar or per-head).
        v_scale: FP8 dequant scale for values (scalar or per-head).
        sm_scale: Softmax scale (default: 1/sqrt(head_dim)).
        logits_soft_cap: Tanh soft cap for logits (0=disabled, 50.0 for Gemma4).
        num_kv_splits: Number of split-K partitions (0=auto based on SMs).

    Returns:
        Output tensor [batch, num_q_heads, head_dim] in BF16.
    """
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    kv_group_size = num_q_heads // num_kv_heads

    assert k_cache.dtype == torch.float8_e4m3fn, f"K cache must be FP8 E4M3, got {k_cache.dtype}"
    assert v_cache.dtype == torch.float8_e4m3fn, f"V cache must be FP8 E4M3, got {v_cache.dtype}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Number of pages
    max_seq = seq_lens.max().item()
    num_pages = (max_seq + block_size - 1) // block_size

    # Auto-select split count for good SM utilization
    if num_kv_splits == 0:
        # GQA blocking
        BLOCK_H = min(kv_group_size, 4)
        BLOCK_H = triton.next_power_of_2(BLOCK_H)
        num_head_blocks = triton.cdiv(num_q_heads, BLOCK_H)
        base_blocks = batch * num_head_blocks

        # Target enough blocks to fill 170 SMs with ~4 blocks each
        target_blocks = 170 * 4
        if base_blocks >= target_blocks:
            num_kv_splits = 1
        else:
            num_kv_splits = max(1, min(num_pages, target_blocks // max(base_blocks, 1)))
            # Ensure each split has at least 2 pages of work
            pages_per_split = num_pages // num_kv_splits
            while pages_per_split < 2 and num_kv_splits > 1:
                num_kv_splits = num_kv_splits // 2
                pages_per_split = num_pages // num_kv_splits

    # Round BLOCK_D to next power of 2
    BLOCK_D = triton.next_power_of_2(head_dim)

    # GQA blocking: process multiple Q heads per block
    BLOCK_H = min(kv_group_size, 4)  # Up to 4 Q heads per block
    BLOCK_H = triton.next_power_of_2(BLOCK_H)

    num_head_blocks = triton.cdiv(num_q_heads, BLOCK_H)

    # Per-head scale detection
    per_head_scale = 1 if k_scale.numel() > 1 else 0

    # Allocate mid output: [batch, num_q_heads, num_kv_splits, head_dim+1]
    mid_out = torch.empty(
        (batch, num_q_heads, num_kv_splits, head_dim + 1),
        dtype=torch.float32, device=q.device)

    # Output
    output = torch.empty_like(q)

    # Grid: (batch, head_blocks, splits)
    grid_stage1 = (batch, num_head_blocks, num_kv_splits)

    _fp8_decode_stage1[grid_stage1](
        q,
        k_cache, v_cache,
        block_table, seq_lens,
        k_scale, v_scale,
        mid_out,
        # Q strides
        q.stride(0), q.stride(1),
        # K cache strides
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        # V cache strides
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        # Block table strides
        block_table.stride(0),
        # Mid strides
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
        # Scalars
        sm_scale,
        # Constexprs
        HEAD_DIM=head_dim,
        BLOCK_D=BLOCK_D,
        PAGE_SIZE=block_size,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=num_kv_splits,
        KV_GROUP_SIZE=kv_group_size,
        NUM_Q_HEADS=num_q_heads,
        LOGITS_SOFT_CAP=logits_soft_cap,
        PER_HEAD_SCALE=per_head_scale,
        num_stages=1,
        num_warps=4,
    )

    # Stage 2: reduce splits
    grid_stage2 = (batch, num_q_heads)
    _fp8_decode_stage2[grid_stage2](
        mid_out, output, seq_lens,
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


# =====================================================================
# BF16 reference (same kernel structure, for correctness comparison)
# =====================================================================

def bf16_decode_attention_ref(
    q: torch.Tensor,              # [batch, num_q_heads, head_dim] BF16
    k_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim] BF16
    v_cache: torch.Tensor,        # [num_blocks, block_size, num_kv_heads, head_dim] BF16
    block_table: torch.Tensor,    # [batch, max_num_blocks] int32
    seq_lens: torch.Tensor,       # [batch] int32
    sm_scale: float = None,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    """PyTorch reference: unpaged gather + attention in FP32."""
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]
    kv_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    output = torch.zeros_like(q, dtype=torch.float32)

    for b in range(batch):
        sl = seq_lens[b].item()
        if sl == 0:
            continue

        # Gather K/V from paged cache
        num_blocks_needed = (sl + block_size - 1) // block_size
        pages = block_table[b, :num_blocks_needed]

        # Collect all positions
        k_gathered = []  # [sl, num_kv_heads, head_dim]
        v_gathered = []
        for i, page_id in enumerate(pages):
            start = i * block_size
            end = min(start + block_size, sl)
            count = end - start
            k_gathered.append(k_cache[page_id, :count])
            v_gathered.append(v_cache[page_id, :count])

        k_all = torch.cat(k_gathered, dim=0).float()  # [sl, num_kv_heads, head_dim]
        v_all = torch.cat(v_gathered, dim=0).float()  # [sl, num_kv_heads, head_dim]

        q_b = q[b].float()  # [num_q_heads, head_dim]

        for h in range(num_q_heads):
            kv_h = h // kv_group_size
            q_vec = q_b[h]  # [head_dim]
            k_h = k_all[:, kv_h, :]  # [sl, head_dim]
            v_h = v_all[:, kv_h, :]  # [sl, head_dim]

            scores = (q_vec @ k_h.T) * sm_scale  # [sl]

            if logits_soft_cap > 0:
                scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)

            weights = torch.softmax(scores, dim=-1)  # [sl]
            output[b, h] = weights @ v_h  # [head_dim]

    return output.to(q.dtype)


# =====================================================================
# Convenience: create test data
# =====================================================================

def create_test_data(
    batch: int = 32,
    num_q_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 256,
    seq_len: int = 2048,
    block_size: int = 16,
    device: str = "cuda",
):
    """Create test data for FP8 decode attention."""
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch * num_blocks_per_seq + 64  # extra slack

    # Create BF16 KV cache first (ground truth), then quantize to FP8
    k_cache_bf16 = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device=device) * 0.1
    v_cache_bf16 = torch.randn(
        total_blocks, block_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device=device) * 0.1

    # Quantize to FP8 with per-tensor scale
    k_amax = k_cache_bf16.float().abs().max().item()
    v_amax = v_cache_bf16.float().abs().max().item()
    # FP8 E4M3 max value is 448.0
    fp8_max = 448.0
    k_scale = k_amax / fp8_max if k_amax > 0 else 1.0
    v_scale = v_amax / fp8_max if v_amax > 0 else 1.0

    k_cache_fp8 = (k_cache_bf16.float() / k_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    v_cache_fp8 = (v_cache_bf16.float() / v_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    # Block table: simple sequential assignment
    block_table = torch.zeros(batch, num_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(batch):
        for i in range(num_blocks_per_seq):
            block_table[b, i] = b * num_blocks_per_seq + i

    # Sequence lengths (all same for benchmarking)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Query
    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1

    # Scales as tensors
    k_scale_t = torch.tensor([k_scale], dtype=torch.float32, device=device)
    v_scale_t = torch.tensor([v_scale], dtype=torch.float32, device=device)

    return {
        "q": q,
        "k_cache_fp8": k_cache_fp8,
        "v_cache_fp8": v_cache_fp8,
        "k_cache_bf16": k_cache_bf16,
        "v_cache_bf16": v_cache_bf16,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "k_scale": k_scale_t,
        "v_scale": v_scale_t,
        "k_scale_val": k_scale,
        "v_scale_val": v_scale,
    }
