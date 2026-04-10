# 27-Layer Pruned Model Quality Validation

**Date:** 2026-04-10  
**Pruned checkpoint:** `/root/models/gemma-4-26B-pruned-27L-rm2_4_8`  
**Layers removed:** 2, 4, 8 (30 -> 27 layers, -10%)  
**Baseline:** `gemma-4-26B-A4B-it-NVFP4` (30 layers, NVFP4 quantized)  
**Validation tool:** `tools/validate_pruning.py` (113 tests, 6 domains)  
**Raw results:** `profiling/layer_pruning_27L_quality_raw.json`

---

## Step 1: Checkpoint Validation

The pruned checkpoint exists and is structurally valid:

```
/root/models/gemma-4-26B-pruned-27L-rm2_4_8/
  config.json           (5.4 KB) — text_config.num_hidden_layers = 27
  model.safetensors     (15.4 GB, single shard)
  model.safetensors.index.json (3.7 MB)
  tokenizer.json        (30.7 MB)
  hf_quant_config.json  — NVFP4 quantization (group_size=16)
```

Weight map confirms 27 contiguous layer indices (0-26). The renumbering after
removing original layers 2, 4, 8 is correct — layers are sequential with no gaps.

---

## Step 2: vLLM Load Test

**Result: LOADED SUCCESSFULLY**

The model loaded in vLLM (vllm-built image) with these settings:
- `--quantization modelopt`
- `--max-model-len 2048`
- `--enforce-eager`
- `--gpu-memory-utilization 0.90`

Key observations from load:
- Weights loaded in ~8s (single 15.4 GB shard, NVFP4)
- GPU memory used: ~17 GiB (vs ~20 GiB for the 30L model)
- No weight shape errors or config mismatches
- NVFP4 quantization recognized correctly (exclude_modules references layers 0-26, matching pruned count)
- vLLM backend: TRITON_ATTN (SM120 path, RTX 5090)

The quantization config in `config.json` correctly excludes attention layers
0-26 from NVFP4 (attention stays in BF16), confirming the config was properly
updated during pruning.

---

## Step 3: Quality Results

### Overall Score

| Metric | Pruned 27L | Baseline 30L | Delta |
|--------|-----------|--------------|-------|
| Overall | **0.525** | **0.895** | **-0.370 (-41.3%)** |

### Domain Breakdown

| Domain | Pruned | Baseline | Delta | Status |
|--------|--------|----------|-------|--------|
| coding | 0.397 | 0.976 | -0.579 | **REGRESSION** |
| math | 0.159 | 0.545 | -0.386 | **REGRESSION** |
| reasoning | 0.350 | 0.900 | -0.550 | **REGRESSION** |
| knowledge | 0.524 | 0.952 | -0.428 | **REGRESSION** |
| language | 0.853 | 1.000 | -0.147 | **REGRESSION** |
| instruction | 0.867 | 1.000 | -0.133 | **REGRESSION** |

**All 6 domains regressed. 5 of 6 exceed the 10% regression threshold.**

### Sub-domain Breakdown

**Coding (0.397):**
- javascript: 1.000 (unchanged — only surviving sub-domain)
- python: 0.667 (-0.333)
- algorithms: 0.500 (-0.500)
- sql: 0.375 (-0.500)
- bash: 0.167 (-0.833)
- debugging: 0.000 (-1.000) — complete collapse
- rust: 0.250 (-0.750)
- typescript: 0.000 (-1.000) — complete collapse

**Math (0.159):**
- calculus: 0.750 (-0.250)
- algebra: 0.100 (-0.300)
- arithmetic: 0.000 (-0.800) — complete collapse
- probability: 0.000 (-0.500)
- word_problem: 0.000 (was already 0.0)

**Reasoning (0.350):**
- counterfactual: 0.500 (-0.500)
- cognitive: 0.500 (-0.500)
- pattern: 0.333 (unchanged)
- logic: 0.400 (-0.600)
- analogy: 0.333 (-0.667)
- causal: 0.000 (-1.000) — complete collapse

**Knowledge (0.524):**
- chemistry: 0.667 (unchanged)
- technology: 0.667 (-0.333)
- geography: 0.600 (-0.400)
- literature: 0.500 (-0.500)
- history: 0.500 (-0.500)
- biology: 0.000 (-1.000) — complete collapse
- physics: 0.000 (-1.000) — complete collapse

**Language (0.853):** — best retained domain
- creative, formal, simplification, summarization: 1.000 each
- tone: 0.750 (-0.250)
- translation: 0.600 (-0.400)

**Instruction (0.867):**
- constraint, multistep, refusal, structured: 1.000 each
- list: 0.200 (-0.800) — severe degradation

---

## Step 4: Failure Analysis

### Failure Mode 1: Repetition / Degeneration (most common)
Responses loop or degrade into repeated tokens/patterns:
```
"```bash\n```bash\n```bash\n```bash\n```bash\n..."
"******\n****\n**\n**\n**\n..."
"1\n1\n1\n1\n1\n1\n..."
```
This is the dominant failure mode across coding, reasoning, and instruction domains.
It indicates the pruned layers (2, 4, 8) were critical for controlling repetition
and maintaining generation coherence — likely early transformer layers that set up
attention patterns and embedding distributions.

### Failure Mode 2: Hallucination / Wrong Facts (knowledge domain)
- "What is the speed of light?" → "397700" (correct: ~300,000 km/s)
- "Who wrote Pride and Prejudice?" → "Jodi Pillen" (correct: Jane Austen)
- "In what year did WWII end?" → "1" (truncated/wrong)
- "What programming language did Guido van Rossum create?" → "ALGOLON" (correct: Python)
- "What is the powerhouse of the cell?" → infinite tautology loop

### Failure Mode 3: Arithmetic Errors (math domain)
- "What is 144 / 12?" → "11" (correct: 12)
- "What is 2^10?" → "1" (correct: 1024)
- "What is 156 + 287?" → "477" (correct: 443) — note: hallucinated plausible number
- "What is 17 * 23?" → "12" (correct: 391)

The arithmetic failures are near-random, confirming structural damage rather than
slight degradation.

### Failure Mode 4: Instruction Following Breakdown
- "Name 3 programming languages. Only 3." → "1\n" (single character)
- "List exactly 5 planets, numbered 1-5" → "1\n1\n2\n3\n4\n4\n1\n1\n..." (repetition loop)

---

## Conclusion

### The Pruned Model FAILS Quality Validation

The 27-layer model (layers 2, 4, 8 removed) is **not viable for production use**:

1. **Overall quality drops 41.3%** (0.895 -> 0.525) — far exceeding the 10% regression threshold
2. **Dominant failure mode is generation degeneration** (repetition loops, garbled output),
   not just accuracy degradation. This is a structural problem, not a fine-tuning gap.
3. **Layers 2, 4, 8 appear to be critical early-layer infrastructure**, likely involved
   in residual stream normalization, attention sink formation, and early token routing.
   Their removal breaks the generation process for structured/constrained outputs.
4. **Language generation is partially preserved** (0.853) — open-ended creative tasks
   can still complete without tight constraints. This suggests later layers handle
   fluency while early layers handle precision and structure.

### What Would Need to Change

To make layer pruning viable at 10% layer removal:

1. **Different layer selection:** Layers 2, 4, 8 are poor candidates. Analysis of
   layer importance scores (see `profiling/layer_importance.json`) should guide
   selection toward later, more redundant layers with high cosine similarity to
   their neighbors.
2. **Recovery fine-tuning:** Even well-chosen pruning requires 100-500 steps of
   continued pretraining or distillation to recover the generation stability lost
   from structural changes. Without fine-tuning, degeneration is expected.
3. **Gradual pruning:** Remove 1 layer at a time, evaluating quality after each,
   rather than removing 3 simultaneously.
4. **Test with layers from the middle/later range** (e.g., layers 20-28 of 30),
   which tend to be more redundant and less foundational than early layers.

### Memory / Speed Impact of 27L

The 27L model loaded in ~17 GiB GPU memory vs ~20 GiB for 30L — a ~15% VRAM
savings. Generation speed was not benchmarked in this validation run but would be
approximately 10% faster due to reduced layer count (assuming no throughput
bottleneck changes).
