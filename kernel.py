"""
AutoKernel -- The file the agent modifies.

Current kernel: W4A16 Quantized Matrix Multiplication
Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass
"""

KERNEL_TYPE = "quantized_matmul_w4a16"

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['K', 'N'],
)
@triton.jit
def dequant_kernel(
    QW_ptr, S_ptr, Z_ptr, W_ptr,
    K, N,
    stride_qwk, stride_qwn,
    stride_skg, stride_sn,
    stride_zkg, stride_zn,
    stride_wk, stride_wn,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
):
    """Dequantize INT4 packed weights to FP16."""
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_mask = offs_k < K
    n_mask = offs_n < N

    g = offs_k // QUANT_GROUP_SIZE

    s_ptrs = S_ptr + g[:, None] * stride_skg + offs_n[None, :] * stride_sn
    z_ptrs = Z_ptr + g[:, None] * stride_zkg + offs_n[None, :] * stride_zn
    scales = tl.load(s_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=1.0)
    zeros = tl.load(z_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

    packed_k_idx = offs_k // 8
    bit_shift = ((offs_k & 7) * 4).to(tl.int32)

    qw_ptrs = QW_ptr + packed_k_idx[:, None] * stride_qwk + offs_n[None, :] * stride_qwn
    qw_packed = tl.load(qw_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
    int4_vals = (qw_packed >> bit_shift[:, None]) & 0xF

    w_dequant = (int4_vals.to(scales.dtype) - zeros) * scales

    w_ptrs = W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    tl.store(w_ptrs, w_dequant, mask=k_mask[:, None] & n_mask[None, :])


def kernel_fn(
    activation: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Entry point called by bench.py."""
    assert activation.is_cuda
    M, K = activation.shape
    N = packed_weights.shape[1]

    W = torch.empty((K, N), device=activation.device, dtype=activation.dtype)

    def dequant_grid(META):
        return (triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(N, META['BLOCK_SIZE_N']))

    dequant_kernel[dequant_grid](
        packed_weights, scales, zeros, W,
        K, N,
        packed_weights.stride(0), packed_weights.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        W.stride(0), W.stride(1),
        QUANT_GROUP_SIZE=group_size,
    )

    return torch.mm(activation, W)
