"""Fused per-head RMSNorm(q) + RMSNorm(k) + RoPE kernel for Qwen3 MoE.

Motivation
==========
In Qwen3MoeAttention.forward, after the qkv_proj split, three separate
kernels run back-to-back:

    q_by_head = self.q_norm(q.view(..., head_dim))   # RMSNorm launch
    k_by_head = self.k_norm(k.view(..., head_dim))   # RMSNorm launch
    q, k = self.rotary_emb(positions, q, k)          # RoPE launch

At M=1 decode with head_dim=128, per-launch overhead (~5-10 us each on
SM120 inside CUDA graph capture, ~15-25 us eager) dominates the actual
compute (BW for 32*128 + 4*128 BF16 = ~9 KB per token). Fusing into a
single kernel lets us read q/k once, normalize, rotate, and write back.

W8 RANK 4 EXTENSION (KV-cache-update fusion)
============================================
On FlashInfer decode, after RoPE vLLM launches ANOTHER separate kernel:

    torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)

which scatter-writes the rotated K and raw V into the paged KV cache
at slots indicated by ``slot_mapping[token_idx]``. At M=1 that kernel
is pure launch overhead (~3-5 us × 48 layers = ~144-240 us/step).

This kernel's K programs now ALSO, when a ``slot_mapping`` and
``key_cache`` pointer are supplied, scatter-write the rotated K
into the paged cache in the same pass (no extra launch, no extra K
roundtrip — K lives in registers from RoPE output). A separate
V-scatter kernel handles value (which never passes through RoPE).

KV cache layout reference (FlashInfer, NHD logical):
    kv_cache.shape == (num_blocks, 2, block_size, num_kv_heads, head_dim)
    key_cache   = kv_cache[:, 0]   # 4-D view
    value_cache = kv_cache[:, 1]
Physical memory may be HND-permuted; strides on ``key_cache`` and
``value_cache`` carry that info correctly. For each token t:
    slot = slot_mapping[t]
    block_idx    = slot // block_size
    block_offset = slot %  block_size
    key_cache[block_idx, block_offset, head, :] = rotated_K[t, head, :]

IMPORTANT: ``slot_mapping`` may contain sentinel ``-1`` entries for
padding tokens (CUDA-graph capture pads the batch dim). Programs MUST
early-exit when slot < 0 — writing to ``key_cache[-1, ...]`` corrupts
the last page silently (worst bug class).

Block layout
============
Grid: (M, num_q_heads + num_kv_heads).  One program per (token, head).

Each program:
  1. Loads two half-vectors of a head (``HALF_DIM`` = ``head_dim/2``).
  2. Loads the matching RMSNorm weight (halves too).
  3. Computes RMSNorm in FP32 over the concatenation (sum-of-squares is
     additive over the two halves).
  4. Loads cos/sin slice for this token's position.
  5. Applies NEOX-style RoPE:
         out_first  = x_first  * cos - x_second * sin
         out_second = x_second * cos + x_first  * sin
  6. Writes the two halves back.

head_dim MUST equal rotary_dim for this kernel (no partial rotary / no
passthrough tail).  Qwen3-30B satisfies this: head_dim=rotary_dim=128.

Why one block per head (not per token)
======================================
At M=1 decode:
  - 32 Q heads + 4 K heads = 36 programs
  - Each program touches ~128 BF16 elements + 128 weight + 128 cos/sin
    = ~2 KB — fits in L1.  Kernel is pure launch-overhead bound.
  - Wider blocks (one program per token, loop over heads) would reduce
    launch count but serialize heads; at M=1 that's the opposite of what
    we want.

At prefill (M>=1024) the grid is M*36.  Per-program work unchanged.

Call:
    fused_qknorm_rope(
        q,                    # [M, num_q_heads, head_dim] BF16 or [M, Q*HD]
        k,                    # [M, num_kv_heads, head_dim] BF16 or [M, K*HD]
        positions,            # [M] int
        q_norm_weight,        # [head_dim]
        k_norm_weight,        # [head_dim]
        cos_sin_cache,        # [max_pos, head_dim] — cos concat sin along -1
        eps,                  # float
        is_neox=True,
    )

Returns (q, k) — the same tensors, modified in place (or a fresh tensor
if the input was not contiguous — see wrapper).

Tag: W7_T2N_rank2_qknorm_rope
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qknorm_rope_kernel(
    QK_ptr,                  # base pointer to q or k (BF16)
    Positions_ptr,           # [M] int32/int64
    Norm_w_ptr,              # [HEAD_DIM] RMSNorm weight
    Cos_sin_ptr,             # [max_pos, HEAD_DIM] — first HALF cos, second HALF sin
    stride_m,                # stride along token dim (in elements)
    stride_h,                # stride along head dim (in elements)
    stride_cs_m,             # stride along position dim of cos_sin_cache (in elements)
    eps,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    """Fused RMSNorm + NEOX-style RoPE for one (token, head) block."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    half_offs = tl.arange(0, HALF_DIM)

    base = QK_ptr + pid_m * stride_m + pid_h * stride_h
    ptr_first = base + half_offs
    ptr_second = base + HALF_DIM + half_offs

    # Load both halves as FP32.
    x_first = tl.load(ptr_first).to(tl.float32)
    x_second = tl.load(ptr_second).to(tl.float32)

    # ---- RMSNorm in FP32 (one reduction across full head) ----
    ss = tl.sum(x_first * x_first) + tl.sum(x_second * x_second)
    inv_rms = 1.0 / tl.sqrt(ss / HEAD_DIM + eps)

    w_first = tl.load(Norm_w_ptr + half_offs).to(tl.float32)
    w_second = tl.load(Norm_w_ptr + HALF_DIM + half_offs).to(tl.float32)

    x_first = x_first * inv_rms * w_first
    x_second = x_second * inv_rms * w_second

    # Match the two-step BF16 rounding that vLLM's stock path does:
    #   RMSNorm writes BF16, RoPE reads BF16.
    # Round through BF16 here so correctness tests align bit-for-bit with the
    # reference.
    x_first = x_first.to(tl.bfloat16).to(tl.float32)
    x_second = x_second.to(tl.bfloat16).to(tl.float32)

    # ---- RoPE (neox) ----
    pos = tl.load(Positions_ptr + pid_m).to(tl.int64)
    cs_base = Cos_sin_ptr + pos * stride_cs_m
    cos = tl.load(cs_base + half_offs).to(tl.float32)
    sin = tl.load(cs_base + HALF_DIM + half_offs).to(tl.float32)

    out_first = x_first * cos - x_second * sin
    out_second = x_second * cos + x_first * sin

    # Store back; cast to the input dtype inferred at the pointer type.
    tl.store(ptr_first, out_first)
    tl.store(ptr_second, out_second)


@triton.jit
def _fused_qknorm_rope_qk_kernel(
    Q_ptr, K_ptr,
    Positions_ptr,
    Qw_ptr, Kw_ptr,
    Cos_sin_ptr,
    stride_qm, stride_qh,
    stride_km, stride_kh,
    stride_cs_m,
    # --- Rank 4 additions: paged KV-cache scatter for K. -----
    KC_ptr,                  # key_cache base ptr (same dtype as K); 0 if disabled
    stride_kc_block,         # element stride of block axis in key_cache
    stride_kc_off,           # element stride of block_size axis
    stride_kc_h,             # element stride of num_kv_heads axis
    SlotMap_ptr,             # int slot_mapping[M]; 0 if disabled
    # --- end Rank 4 additions. --------------------------------
    eps,
    BLOCK_SIZE_KV: tl.constexpr,  # page block size
    WRITE_KV_CACHE: tl.constexpr, # bool — emit the scatter
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    """One-kernel fused RMSNorm+RoPE for BOTH q and k.

    Grid: (M, NUM_Q_HEADS + NUM_KV_HEADS).
    Programs with pid_h < NUM_Q_HEADS process Q; the rest process K
    (with pid_h - NUM_Q_HEADS being the kv-head index).

    Collapsing q and k into a single launch eliminates one of our two
    kernel launches — the per-launch overhead (5-10 us each in cuda
    graph capture, ~15-25 us eager) was the dominant cost vs vLLM's
    stock 3-op CUDA path at M=1.

    Rank 4: when ``WRITE_KV_CACHE`` is True, K programs additionally
    scatter-write the rotated K into ``key_cache[block, offset, head, :]``
    at ``slot_mapping[pid_m]``. Early-exit guard on sentinel ``-1`` slot.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    half_offs = tl.arange(0, HALF_DIM)

    is_q = pid_h < NUM_Q_HEADS
    # Dispatch base pointer and stride + norm weight based on q vs k.
    if is_q:
        base = Q_ptr + pid_m * stride_qm + pid_h * stride_qh
        w_ptr = Qw_ptr
    else:
        kv_h = pid_h - NUM_Q_HEADS
        base = K_ptr + pid_m * stride_km + kv_h * stride_kh
        w_ptr = Kw_ptr

    ptr_first = base + half_offs
    ptr_second = base + HALF_DIM + half_offs

    x_first = tl.load(ptr_first).to(tl.float32)
    x_second = tl.load(ptr_second).to(tl.float32)

    # RMSNorm (FP32)
    ss = tl.sum(x_first * x_first) + tl.sum(x_second * x_second)
    inv_rms = 1.0 / tl.sqrt(ss / HEAD_DIM + eps)
    w_first = tl.load(w_ptr + half_offs).to(tl.float32)
    w_second = tl.load(w_ptr + HALF_DIM + half_offs).to(tl.float32)
    x_first = x_first * inv_rms * w_first
    x_second = x_second * inv_rms * w_second

    # BF16 round-trip to match reference semantics.
    x_first = x_first.to(tl.bfloat16).to(tl.float32)
    x_second = x_second.to(tl.bfloat16).to(tl.float32)

    # RoPE
    pos = tl.load(Positions_ptr + pid_m).to(tl.int64)
    cs_base = Cos_sin_ptr + pos * stride_cs_m
    cos = tl.load(cs_base + half_offs).to(tl.float32)
    sin = tl.load(cs_base + HALF_DIM + half_offs).to(tl.float32)

    out_first = x_first * cos - x_second * sin
    out_second = x_second * cos + x_first * sin

    tl.store(ptr_first, out_first)
    tl.store(ptr_second, out_second)

    # ---- Rank 4: scatter rotated K into paged KV cache. ----
    # Only K programs write to the cache. V gets its own kernel (below).
    if WRITE_KV_CACHE:
        if not is_q:
            slot = tl.load(SlotMap_ptr + pid_m).to(tl.int64)
            # Sentinel guard: slot == -1 means "padded token, do not scatter."
            # Writing to key_cache[-1, ...] would corrupt the last page.
            if slot >= 0:
                block_idx = slot // BLOCK_SIZE_KV
                block_off = slot %  BLOCK_SIZE_KV
                kv_h2 = pid_h - NUM_Q_HEADS
                kc_base = (
                    KC_ptr
                    + block_idx * stride_kc_block
                    + block_off * stride_kc_off
                    + kv_h2    * stride_kc_h
                )
                # key_cache last-dim stride == 1 (head_dim contiguous); we
                # checked in the wrapper.
                tl.store(kc_base + half_offs,            out_first)
                tl.store(kc_base + HALF_DIM + half_offs, out_second)


@triton.jit
def _v_scatter_kernel(
    V_ptr,
    VC_ptr,                  # value_cache base ptr
    SlotMap_ptr,             # int slot_mapping[M]
    stride_vm, stride_vh,
    stride_vc_block,
    stride_vc_off,
    stride_vc_h,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Scatter V tokens into the paged value cache.

    Grid: (M, NUM_KV_HEADS). One program per (token, kv-head). V is not
    passed through RoPE, so this is a pure copy with a gather-store
    pattern. Fused with the qk kernel's launch under the same custom op
    call site: two launches (qk + v) replace three (2 norms + rope +
    unified_kv_cache_update).

    NOTE: V could in principle be co-launched with the qk kernel using a
    larger grid dim (pid_h >= NUM_Q_HEADS + NUM_KV_HEADS reserved for
    V), but keeping it separate (a) keeps the qk register pressure
    low, (b) lets V scatter run in parallel on SM fabric with qk's tail
    — the two kernels are both tiny at M=1. At M>=1024 this separation
    is also better for occupancy.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    slot = tl.load(SlotMap_ptr + pid_m).to(tl.int64)
    # Sentinel guard — see _fused_qknorm_rope_qk_kernel.
    if slot < 0:
        return

    offs = tl.arange(0, HEAD_DIM)
    v = tl.load(V_ptr + pid_m * stride_vm + pid_h * stride_vh + offs)

    block_idx = slot // BLOCK_SIZE_KV
    block_off = slot %  BLOCK_SIZE_KV
    vc_base = (
        VC_ptr
        + block_idx * stride_vc_block
        + block_off * stride_vc_off
        + pid_h    * stride_vc_h
    )
    tl.store(vc_base + offs, v)


def fused_qknorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    is_neox: bool = True,
    # --- Rank 4 additions ---
    v: torch.Tensor | None = None,
    key_cache: torch.Tensor | None = None,
    value_cache: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm(q) + RMSNorm(k) + NEOX RoPE.

    Shape convention:
      q: [M, num_q_heads,  head_dim]   BF16, contiguous on last 2 dims
         (or flattened to [M, num_q_heads*head_dim] — we reshape)
      k: [M, num_kv_heads, head_dim]   BF16
      positions: [M] int32/int64
      q_norm_weight, k_norm_weight: [head_dim]
      cos_sin_cache: [max_pos, head_dim] — cos and sin concatenated.

    Modifies q and k in place. Returns (q, k).

    Rank 4 extension: if ALL of ``v``, ``key_cache``, ``value_cache``,
    ``slot_mapping`` are passed, additionally scatter-write rotated K
    (produced here) and the raw V into the paged KV cache at the slots
    indicated by ``slot_mapping[token_idx]`` — replacing the separate
    ``torch.ops.vllm.unified_kv_cache_update`` launch.
    """
    assert is_neox, "This kernel implements NEOX-style rotary only."

    # Normalise shapes to [M, H, D].
    head_dim = q_norm_weight.shape[0]
    assert cos_sin_cache.shape[-1] == head_dim, (
        f"cos_sin_cache last dim ({cos_sin_cache.shape[-1]}) must equal head_dim "
        f"({head_dim})"
    )
    assert k_norm_weight.shape[0] == head_dim

    if q.dim() == 2:
        # q may be a strided slice of a larger qkv tensor (from .split); use
        # .reshape if .view fails to avoid a silent copy-on-non-contiguous.
        try:
            q3 = q.view(q.shape[0], -1, head_dim)
        except RuntimeError:
            q3 = q.reshape(q.shape[0], -1, head_dim)
    else:
        q3 = q
    if k.dim() == 2:
        try:
            k3 = k.view(k.shape[0], -1, head_dim)
        except RuntimeError:
            k3 = k.reshape(k.shape[0], -1, head_dim)
    else:
        k3 = k

    # q3/k3 may be non-contiguous (strided slices of qkv); the kernel uses
    # explicit strides so this is fine.  We only require that the last dim
    # (head_dim) has stride 1, i.e., per-head elements are contiguous.
    assert q3.stride(-1) == 1, f"q last-dim stride must be 1, got {q3.stride()}"
    assert k3.stride(-1) == 1, f"k last-dim stride must be 1, got {k3.stride()}"

    M = q3.shape[0]
    num_q_heads = q3.shape[1]
    num_kv_heads = k3.shape[1]
    assert q3.shape[2] == head_dim and k3.shape[2] == head_dim

    HALF_DIM = head_dim // 2
    assert HALF_DIM * 2 == head_dim, "head_dim must be even for RoPE."

    # --- Rank 4: validate KV-cache scatter params. ---
    write_kv_cache = (
        key_cache is not None
        and value_cache is not None
        and slot_mapping is not None
        and v is not None
    )
    if write_kv_cache:
        # Layout sanity checks. key_cache / value_cache from vLLM are
        # 4-D views: (num_blocks, block_size, num_kv_heads, head_dim)
        # in NHD logical layout. Physical memory may be HND-permuted,
        # so we rely on strides (not shape ordering) for indexing.
        assert key_cache.dim() == 4, (
            f"key_cache must be 4-D (num_blocks, block_size, num_kv_heads, "
            f"head_dim); got shape {tuple(key_cache.shape)}"
        )
        assert value_cache.dim() == 4
        assert key_cache.shape == value_cache.shape, (
            "key/value cache shape mismatch"
        )
        assert key_cache.shape[-1] == head_dim, (
            f"key_cache last dim {key_cache.shape[-1]} != head_dim {head_dim}"
        )
        assert key_cache.shape[-2] == num_kv_heads, (
            f"key_cache num_kv_heads axis {key_cache.shape[-2]} != "
            f"{num_kv_heads}"
        )
        # Last dim must be contiguous (stride 1) in both the K/V tensors
        # and the cache — we scatter head_dim contiguous elements.
        assert key_cache.stride(-1) == 1
        assert value_cache.stride(-1) == 1
        # slot_mapping is a 1-D int index. vLLM supplies int32 OR int64
        # depending on backend; the kernel casts to int64. It should be
        # at least M long (may be longer because CUDA-graph padding).
        assert slot_mapping.dim() == 1
        assert slot_mapping.shape[0] >= M, (
            f"slot_mapping has {slot_mapping.shape[0]} entries but "
            f"M={M} tokens"
        )
        # V tensor: [M, num_kv_heads, head_dim] with last-dim stride 1.
        if v.dim() == 2:
            try:
                v3 = v.view(v.shape[0], -1, head_dim)
            except RuntimeError:
                v3 = v.reshape(v.shape[0], -1, head_dim)
        else:
            v3 = v
        assert v3.shape[0] == M and v3.shape[1] == num_kv_heads
        assert v3.shape[2] == head_dim
        assert v3.stride(-1) == 1
        block_size_kv = key_cache.shape[1]
        kc_block, kc_off, kc_h = (
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
        )
        vc_block, vc_off, vc_h = (
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
        )
    else:
        # Dummy values so the Triton compiler sees valid scalars in the
        # constexpr path. WRITE_KV_CACHE=False means the kernel skips
        # the scatter entirely — these are never dereferenced.
        block_size_kv = 1
        kc_block = kc_off = kc_h = 0
        # Pass q itself as a placeholder pointer (never loaded/stored to).
        key_cache = q
        slot_mapping = positions  # same dtype, same device, same shape

    # Single-kernel launch (q and k programs share a grid).
    grid = (M, num_q_heads + num_kv_heads)
    _fused_qknorm_rope_qk_kernel[grid](
        q3, k3,
        positions,
        q_norm_weight, k_norm_weight,
        cos_sin_cache,
        q3.stride(0), q3.stride(1),
        k3.stride(0), k3.stride(1),
        cos_sin_cache.stride(0),
        key_cache,
        kc_block, kc_off, kc_h,
        slot_mapping,
        eps=eps,
        BLOCK_SIZE_KV=block_size_kv,
        WRITE_KV_CACHE=write_kv_cache,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        HALF_DIM=HALF_DIM,
        num_warps=2,
        num_stages=1,
    )

    # --- Rank 4: separate V scatter kernel. V never gets RoPE/RMSNorm. ---
    if write_kv_cache:
        v_grid = (M, num_kv_heads)
        _v_scatter_kernel[v_grid](
            v3, value_cache, slot_mapping,
            v3.stride(0), v3.stride(1),
            vc_block, vc_off, vc_h,
            BLOCK_SIZE_KV=block_size_kv,
            HEAD_DIM=head_dim,
            num_warps=2,
            num_stages=1,
        )

    return q, k


def qknorm_rope_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference: per-head RMSNorm then NEOX-style RoPE."""
    head_dim = q_norm_weight.shape[0]
    orig_q_shape = q.shape
    orig_k_shape = k.shape
    if q.dim() == 2:
        q = q.view(q.shape[0], -1, head_dim)
    if k.dim() == 2:
        k = k.view(k.shape[0], -1, head_dim)

    # RMSNorm per head.
    def _rms(x, w):
        x_fp32 = x.to(torch.float32)
        ms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return (x_fp32 * torch.rsqrt(ms + eps) * w.to(torch.float32)).to(x.dtype)

    q = _rms(q, q_norm_weight)
    k = _rms(k, k_norm_weight)

    # RoPE (neox): split last dim into two halves.
    cos_sin = cos_sin_cache.index_select(0, positions.long())  # [M, head_dim]
    cos, sin = cos_sin[..., : head_dim // 2], cos_sin[..., head_dim // 2 :]
    cos = cos.unsqueeze(1).to(q.dtype)  # [M, 1, HALF]
    sin = sin.unsqueeze(1).to(q.dtype)

    def _rope(x):
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    q = _rope(q)
    k = _rope(k)
    return q.view(orig_q_shape), k.view(orig_k_shape)


def kv_cache_scatter_torch_ref(
    k_rotated: torch.Tensor,            # [M, num_kv_heads, head_dim] — already rotated
    v: torch.Tensor,                    # [M, num_kv_heads, head_dim]
    key_cache: torch.Tensor,            # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,          # same shape as key_cache
    slot_mapping: torch.Tensor,         # [M]
) -> None:
    """Reference scatter-write mirroring
    ``torch.ops._C_cache_ops.reshape_and_cache_flash`` for BF16 KV.

    Only tokens with ``slot_mapping[t] >= 0`` are written (matches vLLM's
    sentinel-skip semantics).
    """
    M = slot_mapping.shape[0]
    block_size = key_cache.shape[1]
    slots = slot_mapping[:M].long()
    valid = slots >= 0
    valid_slots = slots[valid]
    block_idx = valid_slots // block_size
    block_off = valid_slots %  block_size
    key_cache[block_idx, block_off] = k_rotated[:M][valid]
    value_cache[block_idx, block_off] = v[:M][valid]
