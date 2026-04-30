"""Fused post_attention_layernorm (RMSNorm + residual add) + router-gate
BF16 linear for Qwen3 MoE (Rank 5 of T2-N ceiling analysis).

Motivation
==========
In Qwen3MoeDecoderLayer.forward the following chain runs after attention:

    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    #   = (rms_norm(hidden_states + residual) * norm.weight,  hidden_states + residual)
    router_logits, _ = self.gate(hidden_states)
    #   = F.linear(hidden_states, gate.weight) for gate.weight [E, H]
    shared_out, fused_out = self.experts(...)

The three ops (``rms_norm_add`` + ``F.linear`` + residual-write back) are
three separate kernel launches. At M=1 decode on SM120 each launch is
~3-6 us of pure overhead for sub-microsecond arithmetic work. Fusing
them collapses three launches into one.

What this kernel does
=====================
Grid: (M,). One program per token.
Each program:
  1. Reads ``hidden_states[m, :]`` and ``residual[m, :]`` (BF16 [H]).
  2. Computes ``r = (hidden + residual)`` in FP32 and writes it to
     ``residual_out`` as BF16 (this is the NEW residual for the MoE
     block's downstream add).
  3. Computes ``rms = sqrt(mean(r^2) + eps)``.
  4. Computes ``x_norm = (r / rms).to(bf16) * norm.weight.to(bf16)`` and
     writes it to ``hidden_out`` (needed by the shared-expert and FP4
     quant paths downstream).
  5. For each of the E routing experts, computes
     ``router_logits[m, e] = sum_h(x_norm[h] * gate.weight[e, h])``
     and writes it to ``router_logits_out``.
     The H=2048 feature row is looped over in BLOCK_H-sized chunks;
     the E=128 experts live in a single axis that we tile by BLOCK_E.

Why one block per token (not one per (token, expert))
======================================================
At M=1 decode the grid is (1,). Launch overhead is the bottleneck,
not arithmetic: 2048*128 = 256K BF16 multiplies per token ~= a few us
of compute on SM120. A single persistent program holds the normed
activation in registers and streams the [E=128, H=2048] BF16 weight
through shared memory (~512 KB) once — keeping launch count = 1.

At larger M (e.g., prefill M=1024) the grid scales to (1024,). Each
program still handles one token but now the weights are resident in
L2 so effective bandwidth is fine. The real regime of interest is
M=1..32 (decode + small speculative).

Semantics vs the stock path
===========================
vLLM's RMSNorm.forward_static computes:
    r = (hidden + residual).to(fp32)            # fp32
    residual_out = r.to(bf16)                   # NEW residual
    rms = sqrt(mean(r^2) + eps)                 # fp32
    x = (r / rms).to(bf16) * weight.to(bf16)    # bf16 out (weight mul in bf16)

We replicate this EXACTLY — including the intermediate bf16 round
before the weight multiply — so the residual and hidden_normed are
bit-identical to stock and the downstream router-gate / FP4-quant
paths see the same rounded values.

Call:
    fused_post_attn_norm_router_gate(
        hidden_states,   # [M, H] BF16  (attn output)
        residual,        # [M, H] BF16  (pre-attn residual)
        norm_weight,     # [H]    BF16
        gate_weight,     # [E, H] BF16  (router)
        eps,             # float
        router_logits_out=None,  # optional [M, E] BF16 buffer
        hidden_out=None,         # optional [M, H] BF16 buffer (normed)
        residual_out=None,       # optional [M, H] BF16 buffer (new residual)
    )

Returns (hidden_out, residual_out, router_logits_out).

Tag: W8_T2N_rank5_post_attn_router
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_post_attn_norm_router_gate_kernel(
    H_ptr,                 # [M, H] BF16 hidden_states
    R_ptr,                 # [M, H] BF16 residual
    Nw_ptr,                # [H]    BF16 norm.weight
    Gw_ptr,                # [E, H] BF16 gate.weight
    Hout_ptr,              # [M, H] BF16 hidden_out (normed)
    Rout_ptr,              # [M, H] BF16 residual_out
    Lout_ptr,              # [M, E] BF16 router_logits_out
    stride_hm, stride_hh,
    stride_rm, stride_rh,
    stride_gw_e, stride_gw_h,
    stride_hom, stride_hoh,
    stride_rom, stride_roh,
    stride_lm, stride_le,
    eps,
    H: tl.constexpr,
    E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    """Fused RMSNorm+residual + router gate GEMM for one token.

    Grid: (M,).  ``H`` and ``E`` are full sizes (compile-time), H is
    assumed to be a power of two >= BLOCK_H.  We load the entire row
    ``x[m, :]`` into a single [BLOCK_H] register vector (BLOCK_H == H
    for Qwen3-30B at H=2048 fits in registers with num_warps=8).
    """
    pid_m = tl.program_id(0)

    # Offsets along H for the full row.
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # --- Load hidden + residual in FP32 ---
    h_ptr = H_ptr + pid_m * stride_hm + h_offs * stride_hh
    r_ptr = R_ptr + pid_m * stride_rm + h_offs * stride_rh
    x_h = tl.load(h_ptr, mask=h_mask, other=0.0).to(tl.float32)
    x_r = tl.load(r_ptr, mask=h_mask, other=0.0).to(tl.float32)

    # Residual-add: new_residual = (hidden + residual) in FP32,
    # then stored as BF16 to residual_out.
    new_res_fp32 = x_h + x_r
    new_res_bf16 = new_res_fp32.to(tl.bfloat16)

    # Write new residual.
    rout_ptr = Rout_ptr + pid_m * stride_rom + h_offs * stride_roh
    tl.store(rout_ptr, new_res_bf16, mask=h_mask)

    # --- RMSNorm (FP32 reduction over the full H row) ---
    # variance_epsilon lives in FP32; mean is sum/H to match torch's
    # .pow(2).mean(-1) behaviour exactly.
    sq = new_res_fp32 * new_res_fp32
    ss = tl.sum(sq, axis=0)
    inv_rms = 1.0 / tl.sqrt(ss / H + eps)

    # x = (r / rms).to(bf16) * weight.to(bf16)
    # We reproduce vLLM forward_static: cast to BF16 BEFORE multiplying
    # by weight so the rounding matches bit-for-bit.
    x_normed_fp32 = new_res_fp32 * inv_rms
    x_normed_bf16 = x_normed_fp32.to(tl.bfloat16)
    w_norm = tl.load(Nw_ptr + h_offs, mask=h_mask, other=0.0)
    # w_norm is BF16; (BF16 * BF16) will upcast internally but the final
    # result is cast back to BF16 for the hidden_out write.
    x_out_bf16 = (x_normed_bf16.to(tl.float32) * w_norm.to(tl.float32)).to(
        tl.bfloat16
    )

    # Write normed hidden_states (needed by shared-expert + FP4 path).
    hout_ptr = Hout_ptr + pid_m * stride_hom + h_offs * stride_hoh
    tl.store(hout_ptr, x_out_bf16, mask=h_mask)

    # --- Router gate GEMM: router_logits[e] = sum_h(x_norm[h] * gw[e, h]) ---
    # We keep x_out in FP32 for the dot-product; gw rows are BF16 and
    # upcast on load. Accumulate per-expert in FP32 then cast to BF16.
    x_for_gemm = x_out_bf16.to(tl.float32)

    # Loop over experts in chunks of BLOCK_E. For Qwen3-30B E=128 and
    # BLOCK_E=32 → 4 iterations; each iter loads a [BLOCK_E, H] BF16 tile
    # (~16 KB at BLOCK_E=32,H=2048) and does one reduction.
    for e_start in tl.static_range(0, E, BLOCK_E):
        e_offs = e_start + tl.arange(0, BLOCK_E)
        # Build [BLOCK_E, BLOCK_H] pointer grid.
        gw_ptrs = (
            Gw_ptr
            + e_offs[:, None] * stride_gw_e
            + h_offs[None, :] * stride_gw_h
        )
        gw_mask = (e_offs[:, None] < E) & (h_offs[None, :] < H)
        gw_tile = tl.load(gw_ptrs, mask=gw_mask, other=0.0).to(tl.float32)

        # Row-wise dot: sum over H for each of BLOCK_E experts.
        # Broadcast x along E axis.
        prod = gw_tile * x_for_gemm[None, :]  # [BLOCK_E, BLOCK_H]
        logits_fp32 = tl.sum(prod, axis=1)    # [BLOCK_E]

        logits_bf16 = logits_fp32.to(tl.bfloat16)
        lout_ptrs = Lout_ptr + pid_m * stride_lm + e_offs * stride_le
        tl.store(lout_ptrs, logits_bf16, mask=e_offs < E)


def fused_post_attn_norm_router_gate(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    eps: float,
    router_logits_out: torch.Tensor | None = None,
    hidden_out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
    block_e: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused post-attn RMSNorm+residual + router gate linear.

    Inputs:
      hidden_states: [M, H] BF16 (attention output)
      residual:      [M, H] BF16 (pre-attn residual)
      norm_weight:   [H]    BF16 (post_attention_layernorm.weight)
      gate_weight:   [E, H] BF16 (router gate.weight; standard NT layout)
      eps:           float (post_attention_layernorm.variance_epsilon)

    Returns:
      hidden_out:        [M, H] BF16 — normed hidden states (feeds MoE experts)
      residual_out:      [M, H] BF16 — NEW residual (= hidden + residual)
      router_logits_out: [M, E] BF16 — logits for router

    If the ``*_out`` kwargs are None, buffers are allocated fresh.
    Allocating output buffers is launch-overhead-heavy at M=1 — callers
    should pre-allocate and reuse across forward calls when possible.
    """
    assert hidden_states.shape == residual.shape
    assert hidden_states.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert norm_weight.dtype == torch.bfloat16
    assert gate_weight.dtype == torch.bfloat16

    M, H = hidden_states.shape
    E, Hg = gate_weight.shape
    assert Hg == H, f"gate_weight H mismatch: {Hg} != {H}"
    assert norm_weight.shape == (H,)

    # H must be a power of 2 and fit in one block (we stream it as one
    # register vector).  Qwen3-30B-A3B is H=2048 which is fine.
    # Pick BLOCK_H = next_pow2(H) so small H values (e.g., testing) work.
    BLOCK_H = triton.next_power_of_2(H)

    device = hidden_states.device
    if hidden_out is None:
        hidden_out = torch.empty_like(hidden_states)
    if residual_out is None:
        residual_out = torch.empty_like(residual)
    if router_logits_out is None:
        router_logits_out = torch.empty(
            (M, E), dtype=torch.bfloat16, device=device,
        )

    # BLOCK_E: small enough that [BLOCK_E, BLOCK_H] BF16 tile fits in
    # shared memory. At BLOCK_H=2048, BF16 = 2B → BLOCK_E=32 → 128 KB
    # register tile; with num_warps=8 that's 16 KB per warp — well within
    # register budget on SM120.
    BLOCK_E = block_e
    # E must be divisible by BLOCK_E for the static_range loop to cover
    # it exactly; round up if not.
    if E % BLOCK_E != 0:
        # Fall back to a BLOCK_E that divides E (E=128 -> 32 ok).
        for cand in (32, 16, 8, 4, 2, 1):
            if E % cand == 0:
                BLOCK_E = cand
                break

    grid = (M,)
    _fused_post_attn_norm_router_gate_kernel[grid](
        hidden_states, residual, norm_weight, gate_weight,
        hidden_out, residual_out, router_logits_out,
        hidden_states.stride(0), hidden_states.stride(1),
        residual.stride(0), residual.stride(1),
        gate_weight.stride(0), gate_weight.stride(1),
        hidden_out.stride(0), hidden_out.stride(1),
        residual_out.stride(0), residual_out.stride(1),
        router_logits_out.stride(0), router_logits_out.stride(1),
        eps,
        H=H, E=E,
        BLOCK_H=BLOCK_H, BLOCK_E=BLOCK_E,
        num_warps=8,
        num_stages=2,
    )

    return hidden_out, residual_out, router_logits_out


def post_attn_norm_router_gate_torch_ref(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference matching vLLM's RMSNorm.forward_static + F.linear.

    Replicates (verbatim):
        r = (hidden + residual).to(fp32)
        residual_out = r.to(bf16)
        rms = sqrt(mean(r^2) + eps)
        x = (r / rms).to(bf16) * weight.to(bf16)  # bf16 out
        router_logits = F.linear(x, gate_weight)
    """
    H = norm_weight.shape[0]
    x_fp32 = hidden_states.to(torch.float32) + residual.to(torch.float32)
    residual_out = x_fp32.to(torch.bfloat16)
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(variance + eps)
    x_norm = x_norm.to(torch.bfloat16)
    hidden_out = x_norm * norm_weight  # bf16 * bf16 (broadcast)
    router_logits = torch.nn.functional.linear(hidden_out, gate_weight)
    # F.linear with bf16 inputs keeps bf16 output.
    return hidden_out, residual_out, router_logits
