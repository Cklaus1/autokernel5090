# EAGLE3 Draft Model Compatibility with Gemma4 26B MoE

**Date:** 2026-04-09  
**Conclusion: INCOMPATIBLE — hidden_size mismatch (5376 vs 2816)**

---

## Available EAGLE3 Draft Models

| Model | Downloads | Framework |
|-------|-----------|-----------|
| `RedHatAI/gemma-4-31B-it-speculator.eagle3` | 112 | speculators lib (custom code) |
| `thoughtworks/Gemma-4-31B-Eagle3` | 648 | transformers/sglang |

Both drafts were trained against `google/gemma-4-31B-it` (hidden_size=5376).

---

## Dimension Comparison

| Dimension | 26B MoE target | 31B EAGLE3 draft expects | Compatible? |
|-----------|---------------|--------------------------|-------------|
| **hidden_size** | **2816** | **5376** | **NO — 1.91x mismatch** |
| vocab_size | 262144 | 262144 | YES |
| head_dim | 256 | 256 (RedHatAI) / 128 (thoughtworks) | Partial |
| num_attention_heads | 16 | 32 (RedHatAI) / 42 (thoughtworks) | NO |
| num_key_value_heads | 8 | 16 (RedHatAI) / 14 (thoughtworks) | NO |
| num_hidden_layers | 30 | 1 (draft layer) | n/a (draft is shallow by design) |
| intermediate_size | 2112 (dense) / 704 (MoE per expert) | 21504 (RedHatAI) | NO |

---

## How EAGLE3 Works and Why This Fails

EAGLE3 draft models receive **hidden states from the target model** as input. The draft's transformer layer has `hidden_size == target_hidden_size` by design — it processes the target's internal representations directly. The connection points are:

1. **Feature extraction**: EAGLE3 taps hidden states from specific layers of the target (e.g., layers [2, 29/30, 56/57] for the 31B). These states have dimension 5376 for the 31B.
2. **Draft transformer input**: The single-layer draft Transformer operates in the target's hidden dimension (5376), not a compressed space.
3. **Projection**: Even if a linear projection were inserted (2816→5376), the draft weights (trained on 31B activations) would not align with 26B activations statistically — the draft would produce garbage token proposals.

The `target_hidden_size: 5376` field in the thoughtworks config makes this explicit.

---

## 26B MoE Config (Full Relevant Fields)

```
hidden_size:          2816
num_hidden_layers:    30
vocab_size:           262144
head_dim:             256
num_attention_heads:  16
num_key_value_heads:  8
intermediate_size:    2112   (dense MLP)
moe_intermediate_size: 704   (per-expert)
num_experts:          128
top_k_experts:        8
layer_types:          alternating sliding/full attention (5:1 ratio)
sliding_window:       1024
```

---

## RedHatAI Speculator Config (Key Fields)

```
architecture:         Eagle3DraftModel (speculators custom)
target hidden_size:   5376
draft transformer:    hidden_size=5376, 1 layer, Llama-style
vocab_size:           262144 (full), draft_vocab_size=32000 (small head)
eagle_aux_layers:     [2, 30, 57]  (tapped from 31B target)
verifier:             google/gemma-4-31B-it
```

---

## What Would Be Needed to Train a 26B-Specific Draft

To build a working EAGLE3 draft for the Gemma4 26B MoE:

1. **Train from scratch** — no existing draft weights can be reused. The draft must be trained with the 26B as the verifier, capturing its hidden states (dim=2816) at selected layers.

2. **Architecture choices for a 26B draft:**
   - hidden_size: 2816 (must match 26B)
   - Single Llama-style transformer layer (standard EAGLE3 design)
   - head_dim: 256 (match 26B)
   - intermediate_size: ~4096–8192 (typical 4x hidden_size for draft FFN)
   - aux_hidden_state_layer_ids: need to choose 3 layers from 26B's 30 layers (e.g., [1, 14, 28])
   - vocab_size: 262144

3. **Training data:** Standard speculative decoding training — run the 26B on a large text corpus, record hidden states + logits, train the draft to predict next token from the tapped hidden states.

4. **MoE consideration:** The 26B alternates between sliding-attention (5/6 of layers) and full-attention (1/6) with MoE FFN blocks. The EAGLE3 draft itself is a simple dense Llama layer, so no MoE needed in the draft. But the hidden states it receives will reflect MoE routing patterns from the 26B — this is handled naturally by training.

5. **Estimated training cost:** Small (draft is 1 layer, ~200M params at hidden_size=2816). A few hundred GPU-hours on the target model is the main cost (data generation). The draft fine-tuning itself is cheap.

---

## Alternative Approaches for Speculative Decoding on 26B

Since no compatible EAGLE3 draft exists:

| Option | Feasibility | Notes |
|--------|-------------|-------|
| Train 26B-specific EAGLE3 | Feasible, ~1-2 weeks | Needs data gen pass over 26B |
| Use a smaller Gemma4 variant as draft | Check dimensions | Gemma4 9B has hidden_size=3840 — also mismatches 2816 |
| Medusa / lookahead decoding | No draft needed | Lower acceptance rate but zero training |
| Token recycling (n-gram) | No draft needed | Works for repetitive text, ~1.2x speedup |
| ngram speculation in SGLang | Built-in | Enable with `--speculative-algorithm ngram` |

The ngram/token-recycling path in SGLang is available today with zero setup cost and gives modest gains on repetitive patterns (code, structured output). For larger gains, training a 26B-specific EAGLE3 draft is the right long-term investment.
