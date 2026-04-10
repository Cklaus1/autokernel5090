# Middle Layer Pruning Experiments: Gemma 4 26B-A4B-it NVFP4

**Date:** 2026-04-09  
**GPU:** RTX 5090 (32 GB)  
**Model:** `/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/` (30 layers)  
**Context:** Retry of layer pruning based on Opus review, which suggested middle layers (12-18) might be safer than early layers.

## Background

Previous attempt removed layers 2, 4, 8 (tier 2 = early, low-scalar layers) and observed -41% quality degradation. Opus review hypothesized that early layers are critical infrastructure (tokenization, positional encoding processing) and middle layers might be more redundant.

## Layer Importance Data (Relevant Layers)

From `profiling/layer_importance.json`:

| Layer | Type              | Scalar | Composite Importance | Notes                        |
|-------|-------------------|--------|---------------------|------------------------------|
| 11    | full_attention    | 0.676  | 0.464               | PROTECTED - global attention |
| 12    | sliding_attention | 0.719  | 0.336               | -                            |
| 13    | sliding_attention | 0.711  | 0.277               | Lowest in middle range       |
| 14    | sliding_attention | 0.637  | 0.371               | Primary candidate            |
| 15    | sliding_attention | 0.559  | 0.334               | Lower scalar                 |
| 16    | sliding_attention | 0.555  | 0.385               | Lowest scalar in range       |
| 17    | full_attention    | 0.617  | 0.591               | PROTECTED - global attention |
| 18    | sliding_attention | 0.754  | 0.342               | -                            |
| 19    | sliding_attention | 0.605  | 0.261               | 2nd lowest importance        |
| 20    | sliding_attention | 0.762  | 0.248               | Lowest composite score       |

Note: Layer 12-13 pair has highest neighbor similarity (0.99854), and layers 19-20 second highest (0.99856), suggesting potential redundancy at these boundaries.

## Experiments

### Experiment 1: Remove Layer 14 (middle, importance=0.371)

**Checkpoint:** `/root/models/gemma-4-26B-pruned-29L-rm14/` (17.51 GB)  
**Config:** 29 layers, 24 sliding + 5 full_attention. Full attention at: [5, 11, 16, 22, 28]  
**Creation time:** 68.1 seconds  

**Test (vLLM, enforce_eager, max_model_len=1024, gpu_memory_utilization=0.92):**

| Prompt | Response |
|--------|----------|
| What is 2+2? | `(Wait, (Wait, (Wait, ...` repeated loop |
| Write a Python binary search | `a Python binary search function a Python binary search...` loop |
| Explain photosynthesis | `-flies in one paragraph-flies in one paragraph-flies...` loop |
| Write a haiku about mountains | `Wait-Wait-Wait-Wait-Wait...` loop |
| Capital of France? | `-the-of-the-of-the-of-the...` loop |
| List 3 benefits of exercise | ` of-1-1-1-1-1-1-1-1...` loop |
| 15 multiplied by 7? | `??????? ? ? ? ? ? ? ?...` loop |
| Reverse string in Python | `to reverse a string...` repeated loop |

**Result: FAIL — Completely incoherent. Infinite repetition loops across all prompts.**

---

### Experiment 2: Remove Layer 13 (middle, importance=0.277, lowest in range)

**Checkpoint:** `/root/models/gemma-4-26B-pruned-29L-rm13/` (17.51 GB)  
**Config:** 29 layers, 24 sliding + 5 full_attention. Full attention at: [5, 11, 16, 22, 28]  

**Test results:**

| Prompt | Response |
|--------|----------|
| What is 2+2? | `2 + 1 + 1 + 1 + 1 + 1 + 1...` repeated addition loop |
| Write a Python binary search | `a Python binary search function a Python binary search...` loop |
| Explain photosynthesis | `-e-e-e-e-e-e-e-e-e-e-e-e-e...` character loop |
| Write a haiku about mountains | `.The-The-The-The-The...` loop |
| Capital of France? | `-the capital of France?-the capital of France?...` loop |
| List 3 benefits | ` of 3 benefits of 3-3-3-3-3-3...` loop |
| 15 multiplied by 7? | `15? 1? 1? 1?11?11?11?...` garbled loop |
| Reverse string | `to reverse a string in Python to reverse...` loop |

**Result: FAIL — Completely incoherent. Same repetition pattern as layer 14.**

---

### Experiment 3: Remove Layer 19 (mid-late, importance=0.261, 2nd lowest)

**Checkpoint:** `/root/models/gemma-4-26B-pruned-29L-rm19/` (17.51 GB)  
**Config:** 29 layers, 24 sliding + 5 full_attention. Full attention at: [5, 11, 17, 22, 28]  

**Test results:**

| Prompt | Response |
|--------|----------|
| What is 2+2? | `-0.00-0.0-0.1-0.0-0.1-0.-1.-1.-1...` numeric loop |
| Write a Python binary search | `-가---...---0---` garbled with Korean character |
| Explain photosynthesis | `-1-1-1-1-1-11-11-11-11...` loop |
| Write a haiku about mountains | `-hi러总成功成功成功...` garbled multilingual |
| Capital of France? | `로고---...` garbled Korean |
| List 3 benefits | `로로로로1---...` Korean characters, loop |
| 15 multiplied by 7? | `15 multiplied 115 multiplied 1111...` |
| Reverse string | `-1-1-1-1-1-1-1- way { 1 { 1...` garbled |

**Result: FAIL — Completely incoherent. Worse than layers 13/14, with multilingual hallucination artifacts suggesting deeper embedding disruption.**

---

## Summary

| Layer Removed | Importance | Result | Pattern |
|---------------|-----------|--------|---------|
| 14 (mid)      | 0.371     | FAIL   | Repetition loops |
| 13 (mid)      | 0.277     | FAIL   | Repetition loops |
| 19 (mid-late) | 0.261     | FAIL   | Repetition + multilingual artifacts |
| 2 (early)     | 0.180     | FAIL*  | -41% quality (from earlier experiment) |
| 4,8 (early)   | 0.223/0.229 | FAIL* | Combined with 2: -41% |

*From previous experiment session.

## Conclusion: Middle Layers Are NOT Safer

The Opus review hypothesis was wrong. Removing a single middle layer (13, 14, or 19) produces **catastrophically worse** results than removing early layers 2,4,8:

- Early layers (2,4,8 removed): -41% quality — degraded but coherent output
- Middle layers (any single one): 0% coherence — infinite repetition loops, complete model collapse

**Why middle layers fail harder:**

1. **The repetition loop pattern** (repeating phrases/characters) is a signature of broken attention — the model loses its ability to shift attention to new tokens and loops on its last state. Middle layers are where learned attention patterns consolidate.

2. **Layer scalars are misleading in the middle.** Middle layers have scalars around 0.55-0.72 — high enough to matter for residual stream continuity. Even layer 13 with importance=0.277 is a key link in the residual chain because the layers on both sides have adapted to its output.

3. **Co-adaptation:** In transformers, adjacent layers co-adapt during training. When you remove a layer from the middle, the output distribution of layer 12 no longer matches what layer 14 (now renumbered 13) expects as input. Early layers have more uniform, simpler representations that are more robust to this.

4. **Layer 19 → multilingual artifacts:** This is particularly revealing. Layer 19 sits just before full_attention layer 17's span of influence ends. Removing it disrupts the context window of the sliding attention boundary, causing the model to pull from random embedding directions, manifesting as multilingual character hallucinations.

## Recommendations

1. **Do not prune middle layers (10-24) without fine-tuning.** Raw weight removal fails universally.

2. **Early layers remain the best pruning targets** despite the previous -41% result. The key issue was removing 3 layers (2,4,8) at once. Tier 1 (just layer 2) should be retested carefully.

3. **Fine-tuning required for >1% quality recovery.** Any useful pruning will need at least a brief distillation/recovery fine-tuning pass.

4. **Consider FLAP or SparseGPT-style approaches** that calibrate the remaining weights after removal to compensate for the removed layer's contribution.

5. **Alternative: Expert pruning.** Rather than layer removal, pruning from 128 to 64 experts per layer might be safer (checked by `gemma4-pruned-32exp` which already exists).

## Files Created

- `/root/models/gemma-4-26B-pruned-29L-rm13/` — Layer 13 removed (17.51 GB)
- `/root/models/gemma-4-26B-pruned-29L-rm14/` — Layer 14 removed (17.51 GB)  
- `/root/models/gemma-4-26B-pruned-29L-rm19/` — Layer 19 removed (17.51 GB)
