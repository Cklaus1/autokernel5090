"""
AutoKernel -- The file the agent modifies.

Current kernel: Flash Attention (block-wise online softmax)
Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass

Optimized flash attention with:
- Autotuned BLOCK_M/BLOCK_N
- FP16 QK dot products for 2x tensor core throughput
- Causal early termination
"""

KERNEL_TYPE = "flash_attention"

import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_stages=2, num_warps=4),
    ],
    key=['M_size', 'N_size', 'D'],
)
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M_size, N_size,
    D: tl.constexpr,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash attention with online softmax and FP16 dot products."""
    pid_z = tl.program_id(2)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(0)

    qkv_offset = pid_z * stride_qz + pid_h * stride_qh
    k_offset = pid_z * stride_kz + pid_h * stride_kh
    v_offset = pid_z * stride_vz + pid_h * stride_vh
    o_offset = pid_z * stride_oz + pid_h * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Load Q block [BLOCK_M, D]
    q_ptrs = Q_ptr + qkv_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < M_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize online softmax state
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Causal: only iterate up to the diagonal
    if IS_CAUSAL:
        kv_end = tl.minimum(N_size, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = N_size

    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block [BLOCK_N, D]
        k_ptrs = K_ptr + k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < N_size
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # QK^T with FP16 dot for 2x throughput
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float16).to(tl.float32)
        qk *= sm_scale

        # Causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # OOB mask
        kv_mask = offs_n[None, :] < N_size
        qk = tl.where(kv_mask, qk, float("-inf"))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V and accumulate
        v_ptrs = V_ptr + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n[:, None] < N_size
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v).to(tl.float32)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store
    o_ptrs = O_ptr + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < M_size
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)


def kernel_fn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
) -> torch.Tensor:
    """Flash attention entry point."""
    Z, H, M_size, D = Q.shape
    _, _, N_size, _ = K.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)

    assert D in (16, 32, 64, 128, 256), f"Head dim {D} not supported"

    grid = lambda META: (triton.cdiv(M_size, META['BLOCK_M']), H, Z)

    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, M_size, N_size,
        D=D,
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
    )

    return O
