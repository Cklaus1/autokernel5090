"""
AutoKernel -- The file the agent modifies.

Current kernel: W4A16 Quantized Matrix Multiplication
Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass

W4A16 scheme:
  - Weights are packed as INT32 (8 x 4-bit values per int32)
  - Per-group scales (FP16) and zero-points (FP16), group_size=128
  - output = activation @ dequantize(packed_weights, scales, zeros)

The agent can change anything in this file:
  - Block sizes, warps, stages
  - Dequantization strategy (per-tile vs per-block)
  - Memory access patterns, shared memory staging
  - Any Triton feature or trick

The agent CANNOT change bench.py, reference.py, or the evaluation.
"""

KERNEL_TYPE = "quantized_matmul_w4a16"

import torch
import triton
import triton.language as tl


@triton.jit
def quantized_matmul_w4a16_kernel(
    # Activation: [M, K], float16
    A_ptr,
    # Packed weights: [K // 8, N], int32
    QW_ptr,
    # Scales: [K // group_size, N], float16
    S_ptr,
    # Zeros: [K // group_size, N], float16
    Z_ptr,
    # Output: [M, N], float16
    C_ptr,
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
):
    """W4A16 quantized matmul: dequantize weights on-the-fly and multiply."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K in steps of BLOCK_SIZE_K (must be multiple of 8)
    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        # Load activation tile: A[offs_m, offs_k]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Dequantize weight tile: unpack int4 from int32
        packed_k_idx = offs_k // 8
        bit_shift = ((offs_k % 8) * 4).to(tl.int32)

        # Load packed weights: QW[packed_k_idx, offs_n]
        qw_ptrs = QW_ptr + packed_k_idx[:, None] * stride_qwk + offs_n[None, :] * stride_qwn
        qw_mask = (packed_k_idx[:, None] < (K // 8)) & (offs_n[None, :] < N)
        qw_packed = tl.load(qw_ptrs, mask=qw_mask, other=0)

        # Extract 4-bit values
        int4_vals = (qw_packed >> bit_shift[:, None]) & 0xF

        # Load per-group scales and zeros
        group_idx = offs_k // group_size
        s_ptrs = S_ptr + group_idx[:, None] * stride_skg + offs_n[None, :] * stride_sn
        z_ptrs = Z_ptr + group_idx[:, None] * stride_zkg + offs_n[None, :] * stride_zn
        s_mask = (group_idx[:, None] < (K // group_size)) & (offs_n[None, :] < N)
        scales = tl.load(s_ptrs, mask=s_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=s_mask, other=0.0)

        # Dequantize: w = (int4_val - zero) * scale
        w_dequant = (int4_vals.to(a.dtype) - zeros) * scales

        # Accumulate
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
    """Entry point called by bench.py. Must match reference.quantized_matmul_w4a16_ref signature."""
    assert activation.is_cuda
    M, K = activation.shape
    N = packed_weights.shape[1]

    C = torch.empty((M, N), device=activation.device, dtype=activation.dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32  # must be multiple of 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    quantized_matmul_w4a16_kernel[grid](
        activation, packed_weights, scales, zeros, C,
        M, N, K, group_size,
        activation.stride(0), activation.stride(1),
        packed_weights.stride(0), packed_weights.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return C
