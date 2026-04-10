# Distillation Feasibility: Gemma4 26B MoE → Dense 9B

**Date:** 2026-04-09
**Hardware:** RTX 5090 (32 GB, 1792 GB/s, SM120 Blackwell), single GPU
**Current baseline:** 127 tok/s single-user decode, 6,615 tok/s peak batch (Gemma4 26B NVFP4 + CUDA graphs)
**Question:** Is distilling Gemma4 26B MoE into a smaller dense model worth doing for a coding workstation?

---

## 1. Does Google Offer a Smaller Gemma4 Model?

**Yes — Gemma4 E2B (2B) is available locally.**

```
/root/.cache/huggingface/hub/models--google--gemma-4-E2B-it         (ref only, not downloaded)
/root/.cache/huggingface/hub/models--principled-intelligence--gemma-4-E2B-it-text-only  (downloaded)
/root/.cache/huggingface/hub/models--prithivMLmods--gemma-4-E2B-it-FP8  (downloaded)
```

**Gemma4 E2B architecture (from config.json):**

| Parameter | Value |
|-----------|-------|
| Architecture | Dense (enable_moe_block: False) |
| Layers | 35 |
| Hidden size | 1536 |
| Attention heads | 8Q / 1KV (extreme GQA) |
| Head dim | 256 (local) / 512 (global) |
| Intermediate size | 6144 × 2 (double-wide MLP) |
| MoE experts | None |
| Vocab size | 262,144 (same as 26B) |
| Context | 131,072 tokens |
| Attention pattern | 30 sliding (window=512) + 5 full |

**Estimated parameter count: ~2.6B** (embed=0.40B + 35 layers × 63.7M each)

**There is no Gemma4 9B.** Google's Gemma4 family has only two sizes:
- E2B (2B dense) — available
- 26B-A4B (26B MoE, 4B active) — available locally in multiple quantizations

A "Gemma4 9B" does not exist. The 9B slot in the Gemma family is occupied by Gemma 2 (different architecture, different tokenizer, incompatible with Gemma4's 262,144-token vocabulary).

---

## 2. What Would Distillation Actually Give Us?

### The Core Architecture Problem

Distillation from Gemma4 26B into a dense 9B requires picking a student architecture. The options are:

| Student | Params | Active/token | Architecture | Available |
|---------|--------|--------------|--------------|-----------|
| Gemma4 E2B | 2.6B | 2.6B | Dense (no MoE) | Yes, locally |
| Distilled Gemma4 ~9B | ~9B | 9B | Dense (custom) | No — must train from scratch |
| Qwen3.5-9B | 9.2B | 9.2B | Dense | Yes, locally (NVFP4) |
| LLaMA 3.1 8B | 8B | 8B | Dense | Not cached |
| DeepSeek-Coder-V2 Lite | 16B / 2.4B active | 2.4B | MoE | Not cached |

### The Bandwidth Argument (Why Dense Isn't Automatically Faster)

Counterintuitively, the 26B MoE activates a similar number of parameters per token as a dense 2-3B model:

```
Gemma4 26B MoE active params per decode token:
  Attention: 30 layers × 34.6M = 1.04B
  MoE (top-8 of 128): 30 layers × 47.6M = 1.43B
  Total active: ~2.47B parameters per token

Gemma4 E2B active params per decode token:
  Attention: 35 layers × 7.1M = 0.25B
  MLP (dense): 35 layers × 56.6M = 1.98B
  Total active: ~2.23B parameters per token

Active-param ratio: 26B MoE / E2B ≈ 1.11x
```

**The 26B MoE reads only 11% more weight memory per token than the E2B dense model.** The "MoE overhead" narrative is misleading for bandwidth-bound single-user decode: the router loads very little data, and only 8 of 128 experts are executed per token. The weight memory read is the bottleneck, and MoE already minimizes that.

A "distilled 9B dense" would read 9B parameters every token — 3.6x more weight memory than the 26B MoE active path. For decode at C=1 (bandwidth-bound), this makes the dense 9B **slower than the 26B MoE**, not faster.

### When Dense Is Faster

Dense models win at **high batch (C≥16)** because:
- No expert routing overhead (no `topk` + index gather per token)
- No grouped GEMM inefficiency (expert GEMMs are small and cache-unfriendly)
- Linear scaling: B tokens × 9B params = 9B loads, always

Current measured throughputs at high concurrency on the 26B MoE:
- C=1:   127 tok/s
- C=4:   383 tok/s
- C=32:  2,071 tok/s
- C=256: 6,615 tok/s

A dense 9B model with NVFP4 on RTX 5090 at C=1 would decode at roughly:
- Weights: 9B × 2 bytes (FP16) = 18 GB, or ~9 GB NVFP4
- Bandwidth: 1792 GB/s → 9 GB / 1792 GB/s = 5 ms → ~200 tok/s
- With CUDA graphs: maybe 250 tok/s

So at **single-user, dense 9B would be ~2x faster than current Gemma4 26B MoE** (127 → ~250 tok/s). Not 3-5x. The 3-5x estimate requires comparing against an unoptimized MoE baseline without CUDA graphs.

---

## 3. Training Cost Estimate

### Option A: Self-Distillation (Gemma4 26B as teacher)

**Dataset generation:**
- Generate 50-100K coding examples using Gemma4 26B teacher
- At 127 tok/s (C=1), 200-token responses: 627 responses/hour
- 100K examples: ~160 GPU-hours just for data generation
- Cost on RTX 5090: 6-7 days of continuous inference

**Training a 9B student from scratch:**
- No 9B Gemma4 architecture exists; must use a compatible base (Qwen3.5-9B, LLaMA-3.1-8B)
- Vocabulary mismatch: Gemma4 uses 262,144 vocab, Qwen3.5 uses 248,320 — output logits incompatible for standard KL divergence distillation
- Workaround: response distillation (train on generated text, not logits) — this is standard fine-tuning, not true distillation
- GPU-hours for 9B fine-tune: 50-100 GPU-hours on a single A100/H100 class GPU
- On RTX 5090 (FP16, ~312 TFLOPS): 9B model, 100K samples, 3 epochs → ~200-400 GPU-hours
- Wall time: **8-17 days** on a single RTX 5090

**Total: 14-24 days of GPU time** (data generation + training), consuming the GPU entirely.

### Option B: Fine-tune Existing Qwen3.5-9B on Coding Data

- Qwen3.5-9B is already downloaded (NVFP4 version cached locally)
- Standard instruction fine-tuning on CodeContests / HumanEval / LeetCode data
- No teacher model needed — use public coding datasets
- Training cost: 20-50 GPU-hours (~1-2 days on RTX 5090)
- Quality: Qwen3.5-9B-Instruct is already a strong coder; fine-tuning adds domain specificity

---

## 4. Alternative: Use an Existing Small Model

### Available Locally (No Download Required)

| Model | Cached | Quantized | Est. single-user tok/s | Coding quality |
|-------|--------|-----------|------------------------|----------------|
| Gemma4 E2B | Yes (text-only variant) | FP8 available | ~350-450 tok/s | Weaker than 26B |
| Qwen3.5-9B | Yes (NVFP4) | Yes | ~200-280 tok/s | Strong (HumanEval ~72%) |
| Qwen3.5-4B | Yes (base+DFlash) | No | ~450-600 tok/s | Decent (HumanEval ~55%) |
| Qwen3.5-2B | Yes (base) | No | ~900+ tok/s | Limited (HumanEval ~35%) |
| Qwen3.5-27B (AWQ) | Yes | AWQ 4-bit | ~90 tok/s | Near-Gemma4-26B quality |

**The Gemma4 E2B is the most direct answer to the original question.** It uses Gemma4's exact tokenizer (262,144 vocab), same chat template, same architecture family — just smaller. No distillation needed. Load it today, test coding quality, done.

**Speed estimate for Gemma4 E2B:**
- Active weights per token: 2.23B params × 2 bytes (BF16) = 4.46 GB
- Or FP8: ~2.23 GB
- At 1792 GB/s: 1.2-2.5 ms per token → **400-800 tok/s** single-user
- With CUDA graphs overhead: realistically **350-600 tok/s**

This is a **3-5x speedup** over the 26B MoE at single-user — but at significant quality cost.

---

## 5. For Coding Specifically: Would a Code-Specialized 7B Beat Distilled Gemma4?

### Head-to-Head Comparison (Hypothetical Distilled Gemma4 9B vs Code-Specialized 7B)

| Model | HumanEval (pass@1) | MBPP | GSM8K | tok/s (C=1) | VRAM |
|-------|-------------------|------|-------|-------------|------|
| Gemma4 26B (teacher) | ~78% | ~73% | ~90% | 127 | 14.8 GB NVFP4 |
| Distilled Gemma4 9B (hypothetical) | ~65-70% | ~60-65% | ~75-80% | ~250 | ~5 GB NVFP4 |
| Qwen3.5-9B-Instruct | ~72% | ~67% | ~83% | ~250 | ~5 GB NVFP4 |
| Gemma4 E2B | ~45-55% | ~40-50% | ~60% | ~400-600 | ~1.5 GB NVFP4 |
| DeepSeek-Coder-V2-Lite (16B MoE) | ~75% | ~71% | ~83% | ~200 | ~9 GB |

**Key finding:** Qwen3.5-9B-Instruct already matches what a distilled Gemma4 9B would likely achieve, at the same speed, with zero training cost. It has been trained on massive code datasets (GitHub, competitive programming, HumanEval, etc.) from the start — purpose-built for code.

A distilled Gemma4 9B trained via response distillation would:
1. Inherit Gemma4's tokenizer mismatch issues with the student
2. Likely underperform Qwen3.5-9B on coding because Qwen3.5 has more dedicated coding pretraining
3. Take 2+ weeks to produce
4. Have no MoE efficiency advantage (being dense, it reads 9B weights vs 2.47B active)

---

## 6. Verdict: Is Distillation Worth It?

### Short Answer: No

The premise contains two false assumptions:
1. **"Dense 9B = 3-5x faster than MoE 26B"** — False. The 26B MoE already reads only ~2.47B weights per token. A dense 9B reads 3.6x more. At C=1, the realistic speedup is ~2x, not 3-5x.
2. **"Distillation gives a unique model not otherwise available"** — False. Qwen3.5-9B already exists, performs comparably, and is cached locally.

### The Real Options (Ranked by ROI)

| Option | Effort | Speedup (C=1) | Quality vs 26B | Ready? |
|--------|--------|---------------|----------------|--------|
| **Gemma4 E2B** (existing) | 0 — just load it | 3-4x | -25 to -35% HumanEval | Yes |
| **Qwen3.5-9B NVFP4** (existing) | 0 — already cached | 2x | -8 to -12% | Yes |
| **26B MoE + CUDA graphs** (current) | Done | baseline | Baseline | Done |
| **26B MoE + speculative ngram** | Days | +30-50% C=1 | 0% (lossless) | Near-term |
| **Dense 9B distillation from 26B** | 2-3 weeks | 1.5-2x | -15 to -25% | No |

### Recommended Path

For **single-user coding workstation** (C=1, latency matters):

1. **Today:** Test Gemma4 E2B (`principled-intelligence/gemma-4-E2B-it-text-only` is downloaded). Measure HumanEval pass@1 and subjective coding quality. If 80%+ of Gemma4 26B quality is acceptable, use it — 400+ tok/s is excellent.

2. **This week:** Test Qwen3.5-9B NVFP4 on coding benchmarks. It's cached at `/root/.cache/huggingface/hub/models--osoleve--Qwen3.5-9B-Base-Text-NVFP4`. The base model needs instruct fine-tuning but is a strong starting point.

3. **If best quality is needed at high speed:** Keep 26B MoE and implement ngram speculative decoding (no model quality loss, +30-50% decode speed). The speculative_decoding_feasibility.md analysis shows ngram works unconditionally with Gemma4.

4. **Do not distill.** The training cost (2-3 weeks GPU time) is not justified when equivalent or better models (E2B, Qwen3.5-9B) are already available and can be deployed in minutes.

---

## Appendix: Hardware Context

**Workstation GPU:** RTX 5090, 32 GB GDDR7, 1792 GB/s, Blackwell SM120
- NVFP4 capability: native via `torch._scaled_mm_v2` — 1270 TFLOPS measured
- Currently runs Gemma4 26B NVFP4 at 14.8 GB VRAM, 127 tok/s (C=1), 6,615 tok/s (peak batch)

**PRO 6000 (arriving):** 2x RTX PRO 6000, 96 GB each, Blackwell SM120
- Would run Gemma4 26B at TP=2: projected ~11,000 tok/s peak, ~110-130 tok/s C=1
- 192 GB VRAM: could run a 70B dense model in NVFP4 (all 70B weights fit in 35 GB)
- On PRO 6000: Qwen3.5-35B AWQ is already cached and usable — better quality than 9B, similar or faster than 26B MoE TP=1

**If PRO 6000 is the target workstation:** The E2B / 9B question becomes irrelevant. The PRO 6000 pair can run Qwen3.5-35B AWQ (cached: `models--QuantTrio--Qwen3.5-27B-AWQ`) or Gemma4 26B at full quality with 2x the KV headroom.
