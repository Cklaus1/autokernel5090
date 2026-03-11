"""
Reference implementations -- PyTorch-only ground truth for correctness verification.
DO NOT MODIFY. These are the oracles that the benchmark harness checks against.
"""

import torch
import torch.nn.functional as F

# Matrix Multiplication
def matmul_ref(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiplication. A @ B."""
    return torch.matmul(A, B)

# Softmax
def softmax_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard softmax along dim."""
    return F.softmax(x, dim=dim)

# Layer Normalization
def layernorm_ref(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Layer normalization over last dimension."""
    normalized_shape = x.shape[-1:]
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

# RMS Normalization
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms) * weight

# Flash Attention
def flash_attention_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True, sm_scale: float = None) -> torch.Tensor:
    """Standard scaled dot-product attention."""
    if sm_scale is None:
        sm_scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if causal:
        seq_len_q, seq_len_k = Q.shape[-2], K.shape[-2]
        mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)

# Fused MLP (SwiGLU-style)
def fused_mlp_ref(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, activation: str = "silu") -> torch.Tensor:
    """SwiGLU-style fused MLP: down(activation(gate(x)) * up(x))."""
    gate = x @ w_gate.T
    up = x @ w_up.T
    if activation == "silu":
        gate = F.silu(gate)
    elif activation == "gelu":
        gate = F.gelu(gate)
    elif activation == "relu2":
        gate = F.relu(gate) ** 2
    return (gate * up) @ w_down.T

# Cross Entropy Loss
def cross_entropy_ref(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross entropy loss."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

# Rotary Position Embedding
def rotary_embedding_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)

# Parallel Reductions
def reduce_sum_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sum reduction."""
    return x.sum(dim=dim)

def reduce_max_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Max reduction."""
    return x.max(dim=dim).values


# ---------------------------------------------------------------------------
# W4A16 Quantization helpers
# ---------------------------------------------------------------------------

def _dequantize_w4a16(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    K: int,
    N: int,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Vectorized dequantization of INT4 weights packed in INT32.

    packed_weights: [K // 8, N] int32 — 8 x 4-bit values per element.
    scales:         [K // group_size, N] float16/bf16
    zeros:          [K // group_size, N] float16/bf16
    Returns:        [K, N] in *dtype*.
    """
    shifts = torch.arange(0, 32, 4, device=packed_weights.device, dtype=torch.int32)  # [8]
    # [K//8, N, 1] >> [1, 1, 8] -> [K//8, N, 8]
    unpacked = (packed_weights.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
    # Reshape to [K, N]: reorder so the 8 sub-indices run along dim-0
    unpacked = unpacked.permute(0, 2, 1).reshape(K, N).to(dtype)
    # Per-group dequantize
    group_indices = torch.arange(K, device=packed_weights.device) // group_size  # [K]
    weight_fp16 = (unpacked - zeros[group_indices]) * scales[group_indices]
    return weight_fp16


# W4A16 Quantized Matrix Multiplication
def quantized_matmul_w4a16_ref(
    activation: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """W4A16 quantized matmul: dequantize INT4 weights then matmul.

    activation:     [M, K] float16/bf16
    packed_weights: [K // 8, N] int32
    scales:         [K // group_size, N] float16/bf16
    zeros:          [K // group_size, N] float16/bf16
    Returns:        [M, N] same dtype as activation.
    """
    M, K = activation.shape
    N = packed_weights.shape[1]
    weight = _dequantize_w4a16(packed_weights, scales, zeros, K, N, group_size, activation.dtype)
    return activation @ weight


# Fused Dequantize + SwiGLU MLP
def dequantize_fused_gemm_ref(
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
    """Fused dequant + SwiGLU MLP reference.

    x:            [M, K] float16/bf16
    packed_w_gate, packed_w_up: [K // 8, N] int32
    packed_w_down:              [N // 8, K] int32
    scales/zeros_gate/up:       [K // group_size, N] float16/bf16
    scales/zeros_down:          [N // group_size, K] float16/bf16
    Returns:      [M, K] same dtype as x.
    """
    gate_out = quantized_matmul_w4a16_ref(x, packed_w_gate, scales_gate, zeros_gate, group_size)
    up_out = quantized_matmul_w4a16_ref(x, packed_w_up, scales_up, zeros_up, group_size)
    gate_out = F.silu(gate_out)
    hidden = gate_out * up_out
    return quantized_matmul_w4a16_ref(hidden, packed_w_down, scales_down, zeros_down, group_size)
