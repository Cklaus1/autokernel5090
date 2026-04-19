# Fused RMSNorm + FP4 Quant wiring for Qwen3-30B-A3B NVFP4

## Goal

Port the fused RMSNorm + FP4-quant kernel (`kernels/csrc/rms_norm_dynamic_fp4_quant.cu`,
2.95x vs separate ops on SM120a) from the Gemma4 wiring to Qwen3 MoE.

## Qwen3 MoE layer structure (`vllm/model_executor/models/qwen3_moe.py`)

```
Qwen3MoeDecoderLayer.forward(positions, hidden_states, residual)
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)           # (1) norm
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)  # (1') fused_add_norm
    hidden_states = self.self_attn(positions, hidden_states)           # (2) qkv_proj is NVFP4

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual) # (3)
    hidden_states = self.mlp(hidden_states)                            # (4) MoE (experts) or MLP
    return hidden_states, residual
```

Attention:
```
Qwen3MoeAttention.forward(positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)   # <-- NVFP4 (ModelOptNvFp4LinearMethod)
    q, k, v = qkv.split(...)
    q_by_head = self.q_norm(q.view(...))
    k_by_head = self.k_norm(k.view(...))
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)     # NVFP4 (BF16 input)
```

MoE block:
```
Qwen3MoeSparseMoeBlock.forward(hidden_states):
    router_logits, _ = self.gate(hidden_states)          # gate is BF16 (excluded from quant)
    shared_out, fused_out = self.experts(hidden_states, router_logits)  # MoE has internal NVFP4 quant
```

## Fusable norm -> NVFP4 paths per decoder layer

1. **`input_layernorm` -> `self_attn.qkv_proj`**  FUSABLE (clean FP4-only consumer)
   - First layer: plain RMSNorm + scaled_fp4_quant -> `rms_norm_dynamic_fp4_quant`
   - Subsequent layers: `fused_add_rms_norm` + scaled_fp4_quant -> `fused_add_rms_norm_dynamic_fp4_quant`

2. **`post_attention_layernorm` -> `self.mlp`**
   - The normed output feeds BOTH `gate` (BF16 linear, not quantized) AND `experts` (internal FP4 quant).
   - NOT fusable: we need the BF16 normed tensor for the gate, so materialising FP4-only would require
     a second dequant pass. Skip.

3. **`o_proj`, `shared_expert.down_proj`**: input comes from attention output / activation, not from a
   dedicated RMSNorm. Not fusable.

## Shape check (hidden_size = 2048)

Qwen3-30B-A3B: hidden_size=2048, intermediate_size=6144, moe_intermediate_size=768,
num_heads=32, num_kv_heads=4, head_dim=128 -> qkv total output =
32*128 + 4*128 + 4*128 = 4096 + 512 + 512 = 5120.

The fused norm kernel only depends on the input width (hidden_size=2048), which matches Gemma4.
`2048 % 16 == 0` -> scaled_fp4_quant block_size constraint satisfied. No shape adaptation needed.

## Wiring approach

Reuse the Gemma4 strategy from `patches/fused_norm_fp4_integration.py`:

- Monkey-patch `Qwen3MoeDecoderLayer.forward` to bypass the separate
  `input_layernorm(x)` + `qkv_proj` pair with a fused call that directly produces
  (fp4_x, sf_x) and invokes the matmul backend (cutlass/flashinfer/fbgemm).
- Keep `post_attention_layernorm` + mlp unchanged.
- Lazily build the per-layer fused callable on first forward (after weights are loaded and
  `process_weights_after_loading` has set `layer.weight_scale`, `alpha`, `input_global_scale_inv`).

Plugin entry point: `patches/wire_fused_norm_fp4_qwen3.py`, exposed as
`[vllm.general_plugins] fused_norm_fp4_qwen3 = wire_fused_norm_fp4_qwen3:register`.
Both this plugin and the T2-N `fused_shuffle_quant` plugin are activated via
`VLLM_PLUGINS=fused_shuffle_quant,fused_norm_fp4_qwen3`.

Shared library: `workspace/fused_rms_norm_fp4_cu13.so` (rebuilt in the
`vllm-fusencache:latest` CUDA-13 image since the existing `*_pro6000.so` was linked
against CUDA 12 and fails to load under the cu130 container).

## Expected performance

- 48 layers * 1 fusable pair (attn side) = 48 fusions per forward
- Each fusion eliminates 1 kernel launch + 1 BF16 materialisation
- Microbenchmark headroom: 2.95x on the fused step; projected e2e +3-6% on Qwen3-30B-A3B
  (Gemma4 saw +12.9% projected with 2 fusions/layer; Qwen3 only has 1/layer, so ~half)

## Pass/Kill criteria

- Both `[T2-N]` and `[fused_norm_fp4_qwen3]` log lines visible at startup
- >= +2% peak tokens/s vs T2-N-only -> PASS
- Regression -> KILL
- No measurable delta -> PARTIAL
