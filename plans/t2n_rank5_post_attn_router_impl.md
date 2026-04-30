# T2-N Rank 5 — post_attention_layernorm + router-gate BF16 fusion

Tag: `W8_T2N_rank5_post_attn_router`

## Goal

Collapse the 3-kernel sequence that runs after attention in every
Qwen3-30B-A3B MoE decoder layer:

```python
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
router_logits, _ = self.mlp.gate(hidden_states)
# hidden_states then feeds shared_expert + fused MoE experts
```

into a single Triton kernel. At M=1 decode each of the 3 ops is launch-
overhead-bound (~3-6 µs on SM120 in CUDA graph capture); fusing them
removes 2 launches per layer × 48 layers per step.

Projected e2e: **+1.2-1.8 %** on the banked 23,254 tok/s T2-N baseline
(per `plans/t2n_ceiling_analysis_extended.md` §Rank 5, P=0.55).

## Design

### Kernel — `kernels/triton/fused_post_attn_router.py`

Grid: `(M,)` — one program per token.

Per program:
1. Load `hidden_states[m, :]` and `residual[m, :]` as BF16 vectors of length H.
2. `new_res_fp32 = hidden + residual` in FP32; store `new_res_fp32.to(bf16)` to `residual_out[m, :]`.
3. `inv_rms = 1/sqrt(mean(new_res^2) + eps)` — FP32 reduction.
4. `x_norm_bf16 = (new_res_fp32 * inv_rms).to(bf16); x_out = (x_norm_bf16 * norm.weight).to(bf16)`.
   Round order matches vLLM `RMSNorm.forward_static` bit-for-bit so residuals drift zero layer-to-layer.
5. Store `x_out` to `hidden_out[m, :]` (feeds shared_expert + FP4 experts).
6. For `e` in `range(0, E, BLOCK_E)`: load a `[BLOCK_E, H]` BF16 tile of
   `gate.weight`, multiply by `x_out` broadcast along E, reduce along H
   in FP32, cast to BF16, store to `router_logits_out[m, e:e+BLOCK_E]`.

Qwen3-30B constants: H=2048, E=128, BLOCK_H=2048 (next_power_of_2),
BLOCK_E=32, `num_warps=8`, `num_stages=2`.

### Plugin wiring — `patches/wire_fused_norm_fp4_qwen3_v2.py`

- New builder `_build_fused_post_attn_router_fn(layer, max_num_tokens)`
  with §P1 silent-None guards on `mlp.gate`, `mlp.gate.weight`,
  `post_attention_layernorm.weight`; warns + returns None on missing attrs.
- Pre-allocates `_hidden_buf`, `_residual_buf`, `_logits_buf` once per
  decoder layer (static GPU memory reuse → CUDA-graph friendly).
- Extended `_patched_decoder_forward` lazy-builds the callable on first
  forward, then replaces the 3-op tail with one fused call.
- Patched `Qwen3MoeSparseMoeBlock.forward` honours a per-instance
  `_prerouted_logits` attribute; when set (by the fused decoder path)
  it skips `self.gate(hidden_states)` and reuses the precomputed logits.
- §P2 banner: `[fused_post_attn_router] active layers=N/M` prints once
  after all 48 layers have been visited on the first forward.
- Env gate `AUTOKERNEL_FUSED_POST_ATTN_ROUTER=0` falls through to stock
  3-kernel path (BC preserved).

### Correctness semantics

vLLM's `RMSNorm.forward_static` with residual:
```
r = (hidden + residual).to(fp32)
residual_out = r.to(bf16)
rms = sqrt(mean(r^2) + eps)
x = (r / rms).to(bf16) * weight   # bf16 mul
```
then `router_logits = F.linear(x, gate.weight)`.

The Triton kernel replicates the two-step BF16 round exactly (cast
`r * inv_rms` to BF16 before multiplying by `weight`, then BF16 round
the result). The router gate GEMM is FP32-accumulated then BF16-cast
on output — identical to `F.linear` with BF16 inputs and BF16 output
dtype.

## Correctness gate

(Pending standalone run — sandbox does not permit Python execution.
Recipe: `CUDA_VISIBLE_DEVICES=1 python kernels/triton/test_fused_post_attn_router.py`.)

- `cos(fused, ref) >= 0.9999` for hidden_out, residual, router_logits
- `max_abs(fused, ref) <= 5e-3` for hidden_out/router_logits (BF16 regime)
- `max_abs(residual) <= 2e-3` — residual must not drift layer-to-layer
- Reference: `post_attn_norm_router_gate_torch_ref` in the kernel module
  (replicates vLLM's forward_static verbatim).

## Performance gate

- **PASS:** e2e ≥ +1.2% vs banked 23,254 (→ ≥23,533 tok/s)
- **KILL:** e2e < +0.3% (→ <23,324 tok/s)
- Kill-only-if-regressed threshold: < -0.5% with correctness failures

## Parent bench recipe

```bash
# Standalone microbench (GPU 1):
CUDA_VISIBLE_DEVICES=1 python kernels/triton/test_fused_post_attn_router.py

# E2E bench — plugin env gate default ON; to ablate:
./launch_qwen3_fused_norm_fp4.sh              # bank the default
AUTOKERNEL_FUSED_POST_ATTN_ROUTER=0 \
  ./launch_qwen3_fused_norm_fp4.sh            # control

# Sweep:
python bench_t2h_qwen3_sweep.py \
    --tag W8_T2N_rank5_post_attn_router \
    --compare-env AUTOKERNEL_FUSED_POST_ATTN_ROUTER=0,1 \
    --concurrency 1024
```

Look for `[fused_post_attn_router] active layers=48/48` in the server
log to confirm the hot path is covered on every MoE layer (§P2).

## §P11 categorisation

Category 1 (bug-fix/code-level PROCEED): the 3-kernel chain is visible
in `qwen3_moe_copy.py:236` and `_patched_decoder_forward`:475 (pre-W8),
the fusion boundary is clean (single BF16 consumer chain for router
path), and the kernel is a new one written from scratch (not cross-
applied from a different regime). P~0.55 is realistic.

## Risks / failure modes

- **BF16 numerical drift in residual** — would compound across 48 layers.
  Mitigated by bit-identical reference matching and the test's
  `torch.equal(residual_fused, residual_ref)` spot-check. If this
  fails, the fix is to ensure `(hidden.to(fp32) + residual.to(fp32)).to(bf16)`
  round order matches; the kernel already does this.
- **Gate has bias** — not supported; the builder returns None + warns
  so the layer falls through to stock. Qwen3 gates have bias=False so
  this is a future-proofing guard, not an active code path.
- **Large-M prefill** — pre-allocated buffer size is `max_num_tokens`
  (default 512, env `AUTOKERNEL_FUSED_NORM_MAX_TOKENS` overrides).
  Prefill chunks above this allocate fresh; CUDA graphs only capture
  the pre-allocated path.
- **Kernel-launch parity at M=1024** — at larger M the router is a
  serial dependency of the whole MoE; if the fused kernel's per-expert
  loop is slower than cuBLAS's `F.linear` at large M, a KILL threshold
  check on the M=1024 microbench is the guardrail before shipping.

## Files

- `kernels/triton/fused_post_attn_router.py` — Triton kernel + wrapper + torch reference
- `kernels/triton/test_fused_post_attn_router.py` — correctness + microbench harness
- `patches/wire_fused_norm_fp4_qwen3_v2.py` — extended with W8 builder, decoder-forward edits, `Qwen3MoeSparseMoeBlock.forward` monkey-patch, env gate
- `plans/t2n_rank5_post_attn_router_impl.md` — this file
