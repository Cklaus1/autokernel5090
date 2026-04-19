"""Triton masked attention for I-DLM v2 extend window.

Simple scaled dot-product attention with a per-request bool mask.
Designed for small sequence lengths (N <= 256) where the full attention
row fits in SRAM. Not intended to compete with FlashAttention -- just
needs to be correct and fast enough for the 7-token extend window.

Shapes:
    q, k, v : [N, H, D]   (ragged batch, all requests concatenated)
    mask    : [N, N]        bool, True = attend
    output  : [N, H, D]
    lse     : [N, H]        float32, log-sum-exp per (query, head)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def masked_attention_kernel(
    Q, K, V, Out, Mask, LSE,
    stride_qn, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kd,
    stride_vn, stride_vh, stride_vd,
    stride_on, stride_oh, stride_od,
    stride_mn, stride_mk,
    stride_lse_n, stride_lse_h,
    N, sm_scale,
    BLOCK_D: tl.constexpr,
    MAX_N: tl.constexpr,
):
    # One program per (query_position, head)
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Pointers to this query row: q[pid_n, pid_h, :]
    q_ptrs = Q + pid_n * stride_qn + pid_h * stride_qh + tl.arange(0, BLOCK_D) * stride_qd
    q = tl.load(q_ptrs, mask=tl.arange(0, BLOCK_D) < BLOCK_D)  # [D]

    # Load mask row: mask[pid_n, :]
    kv_idx = tl.arange(0, MAX_N)
    mask_ptrs = Mask + pid_n * stride_mn + kv_idx * stride_mk
    m = tl.load(mask_ptrs, mask=kv_idx < N, other=False)  # [MAX_N] bool

    # Compute QK^T for all K positions, apply mask
    # k[j, pid_h, :] dot q  for each j
    k_base = K + pid_h * stride_kh
    v_base = V + pid_h * stride_vh

    # Accumulate softmax numerator and denominator
    m_i = float("-inf")  # running max
    l_i = 0.0            # running sum of exp
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # weighted V accumulator

    for j in range(0, N):
        # Check mask
        valid = tl.load(Mask + pid_n * stride_mn + j * stride_mk)
        if valid:
            # Load k[j]
            k_ptrs = k_base + j * stride_kn + tl.arange(0, BLOCK_D) * stride_kd
            k_j = tl.load(k_ptrs)
            # dot product
            s = tl.sum(q * k_j, axis=0) * sm_scale

            # Online softmax update
            m_new = tl.maximum(m_i, s)
            alpha = tl.math.exp2((m_i - m_new) * 1.44269504)  # log2(e)
            p = tl.math.exp2((s - m_new) * 1.44269504)
            l_i = l_i * alpha + p
            acc = acc * alpha

            # Accumulate v[j]
            v_ptrs = v_base + j * stride_vn + tl.arange(0, BLOCK_D) * stride_vd
            v_j = tl.load(v_ptrs)
            acc += p * v_j

            m_i = m_new

    # Normalize
    acc = acc / l_i

    # Store output
    o_ptrs = Out + pid_n * stride_on + pid_h * stride_oh + tl.arange(0, BLOCK_D) * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty))

    # Store log-sum-exp: lse = m_i + log(l_i)
    # Uses natural log via tl.log; m_i was tracked in natural-log space via exp2
    # but scaled with log2(e), so convert back: m_i_nat = m_i (scores are raw),
    # and l_i accumulates exp2(...*log2(e)) = exp(...), so tl.log(l_i) is correct.
    lse_val = m_i + tl.log(l_i)
    lse_ptr = LSE + pid_n * stride_lse_n + pid_h * stride_lse_h
    tl.store(lse_ptr, lse_val.to(tl.float32))


def masked_attention(q, k, v, mask, sm_scale=None, return_lse=True):
    """Attention with custom bool mask.

    Args:
        q: [N, H, D] query
        k: [N, H, D] key
        v: [N, H, D] value
        mask: [N, N] bool (True = attend)
        sm_scale: softmax scale (default 1/sqrt(D))
        return_lse: if True (default), return (out, lse); if False, return out only
    Returns:
        out: [N, H, D]
        lse: [N, H] float32, log-sum-exp values (only when return_lse=True)
             Compatible with FlashInfer's merge_state(o1, s1, o2, s2).
    """
    N, H, D = q.shape
    assert k.shape == v.shape == q.shape
    assert mask.shape == (N, N) and mask.dtype == torch.bool

    if sm_scale is None:
        sm_scale = 1.0 / (D ** 0.5)

    out = torch.empty_like(q)
    lse = torch.empty((N, H), dtype=torch.float32, device=q.device)

    # Round D up to power of 2 for Triton
    BLOCK_D = triton.next_power_of_2(D)
    # Round N up to power of 2 for mask load
    MAX_N = triton.next_power_of_2(N)

    grid = (N, H)
    masked_attention_kernel[grid](
        q, k, v, out, mask, lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        mask.stride(0), mask.stride(1),
        lse.stride(0), lse.stride(1),
        N, sm_scale,
        BLOCK_D=BLOCK_D,
        MAX_N=MAX_N,
    )

    if return_lse:
        return out, lse
    return out
