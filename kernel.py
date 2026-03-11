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
        # M=256 variants — best for large prefill
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        # M=128 variants
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        # M=64 variants
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=1),
        # Small M for decode
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def quantized_matmul_w4a16_kernel(
    A_ptr, QW_ptr, S_ptr, Z_ptr, C_ptr,
    M, N, K,
    group_size,
    stride_am, stride_ak,
    stride_qwk, stride_qwn,
    stride_skg, stride_sn,
    stride_zkg, stride_zn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """W4A16 quantized matmul with L2 swizzle and autotune."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # L2 swizzle
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        # Load activation tile
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Unpack int4 weights
        packed_k_idx = offs_k // 8
        bit_shift = ((offs_k & 7) * 4).to(tl.int32)

        qw_ptrs = QW_ptr + packed_k_idx[:, None] * stride_qwk + offs_n[None, :] * stride_qwn
        qw_mask = (packed_k_idx[:, None] < (K // 8)) & (offs_n[None, :] < N)
        qw_packed = tl.load(qw_ptrs, mask=qw_mask, other=0)
        int4_vals = (qw_packed >> bit_shift[:, None]) & 0xF

        # Per-group dequantize
        group_idx = offs_k // group_size
        s_ptrs = S_ptr + group_idx[:, None] * stride_skg + offs_n[None, :] * stride_sn
        z_ptrs = Z_ptr + group_idx[:, None] * stride_zkg + offs_n[None, :] * stride_zn
        s_mask = (group_idx[:, None] < (K // group_size)) & (offs_n[None, :] < N)
        scales = tl.load(s_ptrs, mask=s_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=s_mask, other=0.0)

        w_dequant = (int4_vals.to(a.dtype) - zeros) * scales
        acc += tl.dot(a, w_dequant)

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


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

    C = torch.empty((M, N), device=activation.device, dtype=activation.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    quantized_matmul_w4a16_kernel[grid](
        activation, packed_weights, scales, zeros, C,
        M, N, K, group_size,
        activation.stride(0), activation.stride(1),
        packed_weights.stride(0), packed_weights.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
