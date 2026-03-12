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
        # M=256
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        # M=128
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=1),
        # M=64
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=1),
        # Decode
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=1),
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
    """Persistent W4A16 matmul with flat K loop."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n

    for tile_id in range(pid, num_tiles, tl.num_programs(0)):
        # L2 swizzle
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        m_mask = offs_m < M

        num_k_steps = K // BLOCK_SIZE_K
        for k_step in range(0, num_k_steps):
            k_start = k_step * BLOCK_SIZE_K
            offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

            # Determine group index for this K block
            g = k_start // group_size

            # Load scales/zeros for this group — [BLOCK_SIZE_N]
            s_ptrs = S_ptr + g * stride_skg + offs_n * stride_sn
            z_ptrs = Z_ptr + g * stride_zkg + offs_n * stride_zn
            scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
            zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)

            # Load activation tile
            a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)

            # Unpack int4 from int32
            packed_k_idx = offs_k // 8
            bit_shift = ((offs_k & 7) * 4).to(tl.int32)

            qw_ptrs = QW_ptr + packed_k_idx[:, None] * stride_qwk + offs_n[None, :] * stride_qwn
            qw_packed = tl.load(qw_ptrs, mask=n_mask[None, :], other=0)
            int4_vals = (qw_packed >> bit_shift[:, None]) & 0xF

            # Dequantize with hoisted scales/zeros
            w_dequant = (int4_vals.to(a.dtype) - zeros[None, :]) * scales[None, :]
            acc = tl.dot(a, w_dequant, acc)

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

    def grid(META):
        num_m = triton.cdiv(M, META['BLOCK_SIZE_M'])
        num_n = triton.cdiv(N, META['BLOCK_SIZE_N'])
        total = num_m * num_n
        num_programs = min(total, 680)
        return (num_programs,)

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
