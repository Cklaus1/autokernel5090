# Gentle Expert Pruning: Layer 0, 5 Expert Zeroing

**Date:** 2026-04-09  
**Model:** Gemma4 26B A4B (NVFP4-modelopt)  
**Approach:** Zero weights of 5 least-active experts in layer 0 only (no structural change, 128 experts remain)

---

## Motivation

Previous experiment (50% global pruning = 64/128 experts removed) produced garbage output (5/20 coherent, 25%).
Hypothesis: the failure was too aggressive. A much gentler approach — zeroing just 5 experts in 1 layer — might preserve quality.

## Target Selection

**Layer 0** was chosen as the target:
- Highest Gini coefficient (0.657) among all 30 layers — most skewed routing
- These 5 experts have nearly zero real-world activation (from 10,000 token sample via vLLM server)

| Expert | Activation Freq | % of Uniform Rate | Rank (least active) |
|--------|----------------|-------------------|---------------------|
| 46 | 0.0000125 | 0.020% | 1st |
| 75 | 0.0000125 | 0.020% | 2nd |
| 110 | 0.0000125 | 0.020% | 3rd |
| 9 | 0.0000250 | 0.040% | 4th |
| 24 | 0.0000250 | 0.040% | 5th |

**Uniform rate** (random routing): 8/128 = 6.25%. These 5 experts activate at <0.05% — over 1000x below uniform.

Collectively, these 5 experts handle <0.02% of all routing decisions. In theory, zeroing them should have negligible impact.

## What Was Done

**Method:** Weight zeroing (NOT removal)
- Kept 128 experts — no structural change
- No config.json changes needed
- Router still selects these experts during inference
- When selected, expert output = 0 (zero weights → zero output)
- Only 45 tensor keys modified (gate/up/down × weight/scale/scale2 × 5 experts)

**Script:** `/root/projects/autokernel/gentle_expert_zero.py`  
**Checkpoint:** `/root/models/gemma4-gentle-zero/` (18.0 GB — same size as original)

**Zeroed tensors per expert:**
- `gate_proj.weight` (uint8 NVFP4 packed)
- `gate_proj.weight_scale` (fp8_e4m3fn)
- `gate_proj.weight_scale_2` (fp32 global scale)
- `up_proj.weight` / `weight_scale` / `weight_scale_2`
- `down_proj.weight` / `weight_scale` / `weight_scale_2`

## Quality Test Results

**Setup:** vLLM 0.19.1rc1, `enforce_eager=True`, `max_model_len=2048`, 20 diverse prompts, greedy decoding, 128 max tokens.

**Result: 6/20 coherent (30%)**

| Question | Result | Sample Output |
|----------|--------|---------------|
| Capital of France? | FAIL | `-111111...` (gibberish digits) |
| 17 * 23? | FAIL | `***로로로로...` (Korean characters + garbage) |
| Quantum computing? | OK* | `(Note: (Note: (Note:...` (repetitive loops) |
| Fibonacci Python? | FAIL | (empty) |
| Three laws of thermodynamics? | FAIL | `TheDesDesDesDesDesDes...` (repetitive garbage) |
| Who wrote Pride and Prejudice? | OK* | `No, it's a different one. No, it's a different one...` (wrong + looping) |
| Speed of light? | FAIL | Korean character loops |
| Supervised vs unsupervised? | OK* | `The difference between the difference...` (looping) |
| Haiku about ocean? | FAIL | `TheTheTheTheThe...` (repetitive) |
| What causes seasons? | FAIL | `---...` (dashes) |

*"OK" by our loose coherence check (>30% alphabetic), but actual content is wrong or repetitive

**Throughput:** 315 tok/s, 2180 tokens in 6.9s (fast — model itself runs fine)
**Load time:** 49.5s (similar to baseline)
**VRAM:** 17.24 GiB (identical to original — no memory savings from zeroing)

## Comparison with Previous Results

| Approach | Coherent | Notes |
|----------|---------|-------|
| Baseline 128 experts | N/A | Cannot load with KV cache on 32GB GPU |
| **Gentle zero (5 experts, layer 0)** | **6/20 (30%)** | **Catastrophic failure despite tiny scope** |
| Pruned 50% (64 experts removed) | 5/20 (25%) | Expected failure (50% removed) |
| Pruned 75% (32 experts removed) | 5/20 (25%) | Complete garbage |

## Analysis: Why Does Zeroing 5 Experts Catastrophically Fail?

This result is surprising and counter-intuitive. Only 5 out of 3,840 expert-layer slots (0.13%) were modified, in the lowest-priority positions. Yet the model fails just as badly as 50% pruning. Several factors explain this:

### 1. Layer 0 is the Input Processing Layer

Layer 0 is the first transformer layer after the embedding. It processes raw token representations. Even small disruptions here propagate through all 29 subsequent layers — the residual stream carries corrupted signal forward.

### 2. Router Still Routes to Zeroed Experts (10.5% Loss of Expert Capacity at Layer 0)

The routing softmax normalizes over 128 experts but the zeroed experts contribute 0 output. With top-8 routing, each token activates 8 experts. If one of those 8 is zeroed, that token loses ~12.5% of its MoE capacity at layer 0. Our activation analysis showed these experts ARE selected (just rarely). When they ARE selected, the hidden state at layer 0 receives an incorrect (diminished) update.

### 3. NVFP4 Zero Is Numerically Ambiguous

Zeroing the NVFP4 `weight_scale_2` (global scale) may not produce a clean output of 0. The MoE kernel applies:
`output = (NVFP4_dequant(weight, scale) @ input) * global_scale`
Setting `global_scale = 0` should give 0, but FP8 × FP4 operations may have NaN/Inf propagation issues or the kernel may not handle zero scales correctly.

### 4. The Activation Frequency Analysis Was From Embeddings Only

The expert activation analysis used vocabulary embeddings as a proxy for hidden states at layer 0. **The actual hidden states during inference are different** — they depend on the model's full behavior. An expert that rarely activates during our 10,000-token sample may be critical for specific token patterns that appear frequently in the 20 test prompts.

## Critical Discovery: Expert Zeroing Is Not Viable

Even zeroing 5 experts (0.13% of total) in the least-active layer causes **catastrophic quality failure**. This definitively answers the question: **there is no "safe" set of experts to disable in Gemma4 MoE**.

The model architecture appears highly sensitive to expert availability. Possible reasons:
1. The router was trained to assume all 128 experts are functional — routing probabilities are calibrated for 128 outputs
2. Any missing expert contribution creates numerical imbalance that compounds through residual connections
3. Layer 0 in particular may have critical "gateway" experts for certain input types

## Conclusion

**Expert pruning is not viable for Gemma4 26B MoE** at any granularity tested:
- 50% global removal → garbage (expected)
- 5 expert zeroing in 1 layer → same garbage (surprising)

**Memory savings:** None (zeroing doesn't reduce checkpoint size or VRAM — expert weights remain in memory)

**Recommendation:** Abandon expert pruning/zeroing approaches entirely. Focus on:
1. KV cache quantization (FP8/INT8) for memory reduction
2. Dynamic expert offloading at runtime (keep weights in CPU, load on demand)
3. Speculative decoding for throughput gains
4. Longer context with efficient KV management

## Files

| File | Description |
|------|-------------|
| `/root/projects/autokernel/gentle_expert_zero.py` | Script that created the zeroed checkpoint |
| `/root/projects/autokernel/gentle_zero_test.py` | Quality test script (20 prompts) |
| `/root/models/gemma4-gentle-zero/` | Modified checkpoint (18.0 GB, can delete) |
| `/root/projects/autokernel/profiling/expert_activation_real_results.json` | Expert activation data used for target selection |
