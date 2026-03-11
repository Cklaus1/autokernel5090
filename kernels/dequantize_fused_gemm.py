"""
AutoKernel -- Fused Dequantize + SwiGLU MLP kernel.

Current kernel: Fused Dequant + Gate + Up projection + SiLU + elementwise multiply + Down projection
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Fuses the following operations with on-the-fly W4A16 dequantization:
  gate = x @ dequant(packed_w_gate)
  up   = x @ dequant(packed_w_up)
  hidden = silu(gate) * up
  out  = hidden @ dequant(packed_w_down)

W4A16 scheme:
  - Weights are packed as INT32 (8 x 4-bit values per int32)
  - Per-group scales (FP16) and zero-points (FP16), group_size=128

The agent can change anything in this file.
The agent CANNOT change bench.py, reference.py, or the evaluation.
"""

KERNEL_TYPE = "dequantize_fused_gemm"

import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_matmul_kernel(
    X_ptr,
    QW_ptr,
    S_ptr,
    Z_ptr,
    C_ptr,
    M, N, K,
    group_size,
    stride_xm, stride_xk,
    stride_qwk, stride_qwn,
    stride_skg, stride_sn,
    stride_zkg, stride_zn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Quantized matmul building block: X @ dequant(QW) with per-group scales/zeros."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        # Load activation tile
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Unpack int4 from int32
        packed_k_idx = offs_k // 8
        bit_shift = ((offs_k % 8) * 4).to(tl.int32)

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

        w_dequant = (int4_vals.to(x.dtype) - zeros) * scales

        acc += tl.dot(x, w_dequant)

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def fused_dequant_gate_up_kernel(
    X_ptr,
    QW_gate_ptr, S_gate_ptr, Z_gate_ptr,
    QW_up_ptr, S_up_ptr, Z_up_ptr,
    Out_ptr,
    M, N, K,
    group_size,
    stride_xm, stride_xk,
    stride_qg_k, stride_qg_n,
    stride_sg_kg, stride_sg_n,
    stride_zg_kg, stride_zg_n,
    stride_qu_k, stride_qu_n,
    stride_su_kg, stride_su_n,
    stride_zu_kg, stride_zu_n,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused: SiLU(x @ dequant(w_gate)) * (x @ dequant(w_up))."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)

        # Load activation
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        packed_k_idx = offs_k // 8
        bit_shift = ((offs_k % 8) * 4).to(tl.int32)
        group_idx = offs_k // group_size

        # --- Dequantize gate weights ---
        qg_ptrs = QW_gate_ptr + packed_k_idx[:, None] * stride_qg_k + offs_n[None, :] * stride_qg_n
        qg_mask = (packed_k_idx[:, None] < (K // 8)) & (offs_n[None, :] < N)
        qg = tl.load(qg_ptrs, mask=qg_mask, other=0)
        int4_gate = (qg >> bit_shift[:, None]) & 0xF

        sg_ptrs = S_gate_ptr + group_idx[:, None] * stride_sg_kg + offs_n[None, :] * stride_sg_n
        zg_ptrs = Z_gate_ptr + group_idx[:, None] * stride_zg_kg + offs_n[None, :] * stride_zg_n
        sg_mask = (group_idx[:, None] < (K // group_size)) & (offs_n[None, :] < N)
        s_gate = tl.load(sg_ptrs, mask=sg_mask, other=1.0)
        z_gate = tl.load(zg_ptrs, mask=sg_mask, other=0.0)

        w_gate = (int4_gate.to(tl.float16) - z_gate) * s_gate

        # --- Dequantize up weights ---
        qu_ptrs = QW_up_ptr + packed_k_idx[:, None] * stride_qu_k + offs_n[None, :] * stride_qu_n
        qu_mask = (packed_k_idx[:, None] < (K // 8)) & (offs_n[None, :] < N)
        qu = tl.load(qu_ptrs, mask=qu_mask, other=0)
        int4_up = (qu >> bit_shift[:, None]) & 0xF

        su_ptrs = S_up_ptr + group_idx[:, None] * stride_su_kg + offs_n[None, :] * stride_su_n
        zu_ptrs = Z_up_ptr + group_idx[:, None] * stride_zu_kg + offs_n[None, :] * stride_zu_n
        su_mask = (group_idx[:, None] < (K // group_size)) & (offs_n[None, :] < N)
        s_up = tl.load(su_ptrs, mask=su_mask, other=1.0)
        z_up = tl.load(zu_ptrs, mask=su_mask, other=0.0)

        w_up = (int4_up.to(tl.float16) - z_up) * s_up

        acc_gate += tl.dot(x, w_gate)
        acc_up += tl.dot(x, w_up)

    # SiLU(gate) * up
    gate_activated = acc_gate * tl.sigmoid(acc_gate)
    result = gate_activated * acc_up

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result.to(Out_ptr.dtype.element_ty), mask=out_mask)


def _run_dequant_matmul(x, packed_w, scales, zeros, N_out, group_size):
    """Helper: run a single quantized matmul via the building-block kernel."""
    M, K = x.shape
    C = torch.empty((M, N_out), device=x.device, dtype=x.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_out, BLOCK_N))

    _dequant_matmul_kernel[grid](
        x, packed_w, scales, zeros, C,
        M, N_out, K, group_size,
        x.stride(0), x.stride(1),
        packed_w.stride(0), packed_w.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )
    return C


def kernel_fn(
    x: torch.Tensor,
    packed_w_gate: torch.Tensor,
    packed_w_up: torch.Tensor,
    packed_w_down: torch.Tensor,
    scales_gate: torch.Tensor,
    zeros_gate: torch.Tensor,
    scales_up: torch.Tensor,
    zeros_up: torch.Tensor,
    scales_down: torch.Tensor,
    zeros_down: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Entry point called by bench.py. Fused dequant + SwiGLU MLP.

    SwiGLU:
      gate = x @ dequant(packed_w_gate)   -- [M, K] @ [K, N] -> [M, N]
      up   = x @ dequant(packed_w_up)     -- [M, K] @ [K, N] -> [M, N]
      hidden = silu(gate) * up            -- [M, N]
      out  = hidden @ dequant(packed_w_down) -- [M, N] @ [N, K] -> [M, K]
    """
    assert x.is_cuda
    M, K = x.shape
    N = packed_w_gate.shape[1]  # intermediate_size

    # Fused gate + up projection
    hidden = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_dequant_gate_up_kernel[grid](
        x,
        packed_w_gate, scales_gate, zeros_gate,
        packed_w_up, scales_up, zeros_up,
        hidden,
        M, N, K, group_size,
        x.stride(0), x.stride(1),
        packed_w_gate.stride(0), packed_w_gate.stride(1),
        scales_gate.stride(0), scales_gate.stride(1),
        zeros_gate.stride(0), zeros_gate.stride(1),
        packed_w_up.stride(0), packed_w_up.stride(1),
        scales_up.stride(0), scales_up.stride(1),
        zeros_up.stride(0), zeros_up.stride(1),
        hidden.stride(0), hidden.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
    )

    # Down projection: hidden [M, N] @ dequant(packed_w_down) [N, K] -> [M, K]
    N_down = packed_w_down.shape[1]  # output dim (= K, the hidden_size)
    out = _run_dequant_matmul(hidden, packed_w_down, scales_down, zeros_down, N_down, group_size)

    return out
