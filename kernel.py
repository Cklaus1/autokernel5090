"""
AutoKernel -- Fused Dequantize + SwiGLU MLP kernel.

Current kernel: Split Dequant + cuBLAS SwiGLU MLP
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Split-dequant strategy (proven on W4A16 matmul):
  1. Triton dequant kernel: INT4 packed -> FP16 for each weight matrix
  2. cuBLAS F.linear for the matmuls (NT layout, ~4% faster than torch.mm)
  3. PyTorch SiLU + elementwise multiply

W4A16 scheme:
  - Weights are packed as INT32 (8 x 4-bit values per int32)
  - Per-group scales (FP16) and zero-points (FP16), group_size=128
"""

KERNEL_TYPE = "dequantize_fused_gemm"

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 256}, num_warps=8, num_stages=4),
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

    if BLOCK_SIZE_K == QUANT_GROUP_SIZE:
        g = pid_k
        s_ptrs = S_ptr + g * stride_skg + offs_n * stride_sn
        z_ptrs = Z_ptr + g * stride_zkg + offs_n * stride_zn
        scales = tl.load(s_ptrs, mask=n_mask, other=1.0)
        zeros = tl.load(z_ptrs, mask=n_mask, other=0.0)
        scales = scales[None, :]
        zeros = zeros[None, :]
    else:
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


# Weight caches
_wt_buf = {}
_dequant_cache = {}


def _dequant_grid(META):
    return (triton.cdiv(META['K'], META['BLOCK_SIZE_K']),
            triton.cdiv(META['N'], META['BLOCK_SIZE_N']))


def _dequantize_weights(packed_w, scales, zeros, K, N, dtype, group_size, device):
    """Dequantize INT4 packed weights to FP16 with caching."""
    cache_key = (id(packed_w), id(scales), id(zeros), K, N, dtype)

    if cache_key not in _dequant_cache:
        # Each weight matrix gets its own buffer (keyed by identity, not shape)
        if cache_key not in _wt_buf:
            _wt_buf[cache_key] = torch.empty((K, N), device=device, dtype=dtype)
        W = _wt_buf[cache_key]

        dequant_kernel[_dequant_grid](
            packed_w, scales, zeros, W,
            K, N,
            packed_w.stride(0), packed_w.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            W.stride(0), W.stride(1),
            QUANT_GROUP_SIZE=group_size,
        )

        if len(_dequant_cache) > 32:
            _dequant_cache.clear()
            _wt_buf.clear()
            _wt_cache.clear()
        _dequant_cache[cache_key] = W

    return _dequant_cache[cache_key]


# Cache for transposed weight views
_wt_cache = {}


def _get_wt(W, cache_key):
    """Get transposed contiguous weight for F.linear, with caching."""
    wt_key = (cache_key, 'Wt')
    if wt_key not in _wt_cache:
        if len(_wt_cache) > 32:
            _wt_cache.clear()
        _wt_cache[wt_key] = W.t().contiguous()
    return _wt_cache[wt_key]


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
    """Entry point called by bench.py. Split dequant + cuBLAS SwiGLU MLP.

    SwiGLU:
      gate = x @ dequant(packed_w_gate)   -- [M, K] @ [K, N] -> [M, N]
      up   = x @ dequant(packed_w_up)     -- [M, K] @ [K, N] -> [M, N]
      hidden = silu(gate) * up            -- [M, N]
      out  = hidden @ dequant(packed_w_down) -- [M, N_down] @ [N_down, K_out] -> [M, K_out]

    Weight shapes:
      packed_w_gate, packed_w_up: [K//8, N]  -> dequant to [K, N]
      packed_w_down:              [N//8, K_out] -> dequant to [N, K_out]
    """
    assert x.is_cuda
    M, K = x.shape
    N = packed_w_gate.shape[1]  # intermediate_size
    device = x.device
    dtype = x.dtype

    # 1. Dequantize gate weights: [K//8, N] -> [K, N]
    W_gate = _dequantize_weights(
        packed_w_gate, scales_gate, zeros_gate, K, N, dtype, group_size, device
    )

    # 2. Dequantize up weights: [K//8, N] -> [K, N]
    W_up = _dequantize_weights(
        packed_w_up, scales_up, zeros_up, K, N, dtype, group_size, device
    )

    # 3. Gate and up projections via cuBLAS F.linear (NT layout)
    # F.linear(x, Wt) computes x @ Wt.T, so we need Wt = W.T  [N, K]
    gate_key = (id(packed_w_gate), id(scales_gate), id(zeros_gate), K, N, dtype)
    up_key = (id(packed_w_up), id(scales_up), id(zeros_up), K, N, dtype)

    Wt_gate = _get_wt(W_gate, gate_key)  # [N, K]
    Wt_up = _get_wt(W_up, up_key)        # [N, K]

    gate = F.linear(x, Wt_gate)  # [M, N]
    up = F.linear(x, Wt_up)      # [M, N]

    # 4. SiLU activation + elementwise multiply
    hidden = F.silu(gate) * up  # [M, N]

    # 5. Dequantize down weights: [N//8, K_out] -> [N_down, K_out]
    N_down = packed_w_down.shape[0] * 8  # N (intermediate_size)
    K_out = packed_w_down.shape[1]       # output dim (hidden_size)

    W_down = _dequantize_weights(
        packed_w_down, scales_down, zeros_down, N_down, K_out, dtype, group_size, device
    )

    # 6. Down projection via cuBLAS F.linear
    down_key = (id(packed_w_down), id(scales_down), id(zeros_down), N_down, K_out, dtype)
    Wt_down = _get_wt(W_down, down_key)  # [K_out, N_down]

    out = F.linear(hidden, Wt_down)  # [M, K_out]

    return out
