# Layer Pruning Analysis: Gemma 4 26B-A4B-it NVFP4

**Date:** 2026-04-09
**GPU:** RTX 5090
**Model:** `/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/`

## Model Architecture

- **30 layers total:** 25 sliding_attention + 5 full_attention
- **Full attention at positions:** 5, 11, 17, 23, 29 (every 6th layer)
- **MoE:** 128 experts, top-8 routing per layer
- **Hidden size:** 2816, **Head dim:** 256 (sliding) / 512 (global)
- **Quantization:** NVFP4 (MLP + experts), attention weights excluded from FP4

## Methodology

Five weight-based importance metrics computed per layer:

1. **Weight Magnitude (L2 norm):** Total energy in layer weights. Lower = less impactful on activations.
2. **Layer Scalar:** Gemma 4 uses per-layer residual scaling. Lower scalar = layer contributes less to residual stream.
3. **Adjacent Layer Similarity:** Cosine similarity between neighboring layers' weight vectors. Higher similarity = one layer is more redundant.
4. **Router Expert Scale CV:** Coefficient of variation of per-expert routing scales. Lower = more uniform/less specialized routing.
5. **Weight Entropy:** Information content of packed weight distributions.

Composite importance score combines these with protection rules for global attention layers and first/last layers.

## Key Findings

### Layer Scalars Reveal Clear Structure

| Layer | Scalar | Interpretation |
|-------|--------|---------------|
| 0 | 0.070 | Very low -- initialization layer, residual barely modified |
| 1 | 0.203 | Low -- early processing |
| 2 | 0.170 | Low -- **most prunable** |
| 29 | 0.195 | Low -- final layer, but needed for output |
| 24 | 0.816 | Highest -- critical computation |
| 20-22 | 0.762 | High -- important middle-late layers |

The layer scalar is the strongest single signal. Layers with scalars below 0.6 contribute proportionally less to the residual stream.

### Weight Magnitude Distribution

Layers with smallest L2 norms (least weight energy):
- **Layer 8:** 2.83B (smallest by far)
- **Layer 6:** 2.99B
- **Layer 4:** 3.50B
- **Layer 7:** 3.75B
- **Layer 20:** 3.83B

Largest L2 norms (most weight energy):
- **Layer 25:** 7.85B
- **Layer 23:** 7.31B (full attention)
- **Layer 15:** 6.99B

### Adjacent Layer Similarity

All adjacent pairs have cosine similarity > 0.997, indicating high redundancy across the board. The most similar pairs (most redundant):
- Layers 12-13: 0.99854
- Layers 19-20: 0.99856

The least similar (most distinct):
- Layers 27-28: 0.99776
- Layers 21-22: 0.99790

### Router Analysis

Expert routing is remarkably uniform across layers -- coefficient of variation ranges only 0.0109 to 0.0127. This means no layer has dramatically different expert utilization patterns, so router structure alone doesn't differentiate importance.

## Importance Ranking (least important first)

| Rank | Layer | Type | Importance | Key Reason |
|------|-------|------|-----------|------------|
| 1 | 2 | sliding | 0.180 | Low scalar (0.17), low L2, high neighbor similarity |
| 2 | 4 | sliding | 0.223 | Low L2 (3rd smallest), mid scalar |
| 3 | 8 | sliding | 0.229 | Smallest L2 norm of all layers |
| 4 | 6 | sliding | 0.237 | 2nd smallest L2 norm |
| 5 | 20 | sliding | 0.248 | Low L2, high neighbor similarity |
| 6 | 7 | sliding | 0.255 | Low L2, low neighbor similarity penalty |
| 7 | 10 | sliding | 0.256 | High neighbor similarity |
| 8 | 3 | sliding | 0.258 | Mid-range across all metrics |
| 9 | 19 | sliding | 0.261 | High neighbor similarity (with 20) |
| 10 | 13 | sliding | 0.278 | Lowest router CV |
| ... | ... | ... | ... | ... |
| 28 | 5 | **full** | 0.537 | Protected: global attention |
| 29 | 17 | **full** | 0.591 | Protected: global attention |
| 30 | 23 | **full** | 0.653 | Most important layer overall |

## Pruning Tiers

### Tier 1: Remove 1 layer (Layer 2)
- **Layers removed:** [2]
- **Remaining:** 29 layers (24 sliding + 5 full)
- **Estimated speedup:** 1.03x (~3%)
- **Estimated decode:** ~71.8 tok/s (from ~69.4)
- **Risk:** Minimal -- layer 2 has lowest importance score and very low residual scalar (0.17)
- **Checkpoint:** `/root/models/gemma-4-26B-pruned-29L-rm2/` (17.5 GB)

### Tier 2: Remove 3 layers (Layers 2, 4, 8)
- **Layers removed:** [2, 4, 8]
- **Remaining:** 27 layers (22 sliding + 5 full)
- **Estimated speedup:** 1.11x (~10%)
- **Estimated decode:** ~77.2 tok/s
- **Risk:** Low -- all early sliding layers with low weight magnitudes
- **Checkpoint:** `/root/models/gemma-4-26B-pruned-27L-rm2_4_8/` (16.5 GB)

### Tier 3: Remove 5 layers (Layers 2, 4, 6, 8, 20)
- **Layers removed:** [2, 4, 6, 8, 20]
- **Remaining:** 25 layers (20 sliding + 5 full)
- **Estimated speedup:** 1.20x (~17%)
- **Estimated decode:** ~83.3 tok/s
- **Risk:** Moderate -- removes 4 early layers plus 1 mid-late layer. All 5 global attention layers preserved.
- **Checkpoint:** `/root/models/gemma-4-26B-pruned-25L-rm2_4_6_8_20/` (15.5 GB)

## Quality Baseline (30L Original)

30 test prompts across 6 categories evaluated on running vLLM server:

| Category | Correct/Reasonable | Notes |
|----------|-------------------|-------|
| Math (5) | 5/5 | All arithmetic correct |
| Code (5) | 5/5 | Correct implementations and explanations |
| Knowledge (5) | 5/5 | All factual answers correct |
| Creative (5) | 5/5 | Appropriate completions |
| Reasoning (5) | 5/5 | Bat-and-ball, sequences, logic all correct |
| Instruction (5) | 4/5 | "Respond in one word" gave "Four" instead of "4" |

Baseline responses saved to `profiling/quality_baseline_30L.json` for comparison after pruning.

## Observations and Recommendations

1. **Early layers (0-8) are most prunable.** They have the lowest weight magnitudes and layer scalars, suggesting the model front-loads compression/tokenization and back-loads reasoning.

2. **Global attention layers are critical.** Positions 5, 11, 17, 23, 29 handle long-range dependencies with 512-dim heads. Never prune these.

3. **Layer 2 is the safest single removal.** It has the lowest composite importance (0.180), the 3rd lowest layer scalar (0.170), and high similarity to both neighbors.

4. **Tier 2 (27L) is the sweet spot.** Removing 3 early sliding layers gives ~10% speedup with minimal quality risk. All removed layers are in the early block (positions 2, 4, 8) where redundancy is highest.

5. **Tier 3 (25L) needs quality validation.** Removing layer 20 alongside the early layers crosses into the mid-network region. Test carefully.

6. **VRAM savings are modest.** The model is already NVFP4 quantized. Removing 5 layers saves ~2 GB. The primary benefit is latency reduction.

## Next Steps

1. Swap vLLM server to 29L checkpoint, run quality baseline, compare
2. If quality holds, try 27L checkpoint
3. Measure actual decode tok/s improvement (estimated vs actual)
4. If Tier 2 passes quality, that becomes the production model

## Files

- `tools/analyze_layers.py` -- Layer importance scoring script
- `tools/prune_layers.py` -- Checkpoint pruning script (--tier 1/2/3)
- `profiling/layer_importance.json` -- Full importance scores and metrics
- `profiling/quality_baseline_30L.json` -- Quality baseline responses
- `/root/models/gemma-4-26B-pruned-29L-rm2/` -- Tier 1 checkpoint
- `/root/models/gemma-4-26B-pruned-27L-rm2_4_8/` -- Tier 2 checkpoint
- `/root/models/gemma-4-26B-pruned-25L-rm2_4_6_8_20/` -- Tier 3 checkpoint
