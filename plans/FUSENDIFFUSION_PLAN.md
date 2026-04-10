# FusenDiffusion: Diffusion-Based Text Generation for FusenAI

**Date:** 2026-04-09
**Author:** ASI Planning Module
**Goal:** Increase single-user generation speed from 120 tok/s to 500-1000+ tok/s
**Hardware:** RTX 5090 (32GB, current) → PRO 6000 DP=2 (2x96GB, arriving)
**Target model:** Gemma4 26B MoE NVFP4 (hidden_size=2816, 262K vocab, 30 layers, top-8/128 experts)

---

## Table of Contents

1. [Background and Constraints](#1-background-and-constraints)
2. [Approach A: Draft Model with Diffusion Head](#2-approach-a-draft-model-with-diffusion-head)
3. [Approach B: Full Diffusion Model](#3-approach-b-full-diffusion-model)
4. [Approach C: Hybrid AR + Diffusion](#4-approach-c-hybrid-ar--diffusion)
5. [Approach D: Training Variations](#5-approach-d-training-variations)
6. [Approach E: Architecture Variations](#6-approach-e-architecture-variations)
7. [Approach F: Inference Optimizations](#7-approach-f-inference-optimizations)
8. [Unified Scoring Matrix](#8-unified-scoring-matrix)
9. [Recommended Timeline](#9-recommended-timeline)
10. [Appendix: Key Numbers](#10-appendix-key-numbers)

---

## 1. Background and Constraints

### Current State

| Metric | Value |
|--------|-------|
| Gemma4 26B NVFP4 single-user decode | 120-127 tok/s |
| Gemma4 26B NVFP4 batch peak (C=256) | 6,615 tok/s |
| Active params per token (MoE top-8/128) | ~2.47B |
| Weight memory read per token | ~5 GB (NVFP4) |
| RTX 5090 bandwidth | 1,792 GB/s |
| Decode bottleneck | Memory bandwidth (weight reads) |
| DFlash on Qwen3.5-9B | 94.6 tok/s (proven, patched into vLLM) |
| N-gram spec decode (projected) | 160-320 tok/s (zero training, proven) |
| FusenCache KV compression | 4x (K4V4B64) |

### Why Diffusion?

Standard AR generation is fundamentally sequential: 1 token per forward pass. At 120 tok/s, the GPU spends ~8.3 ms per token, mostly reading 5 GB of weights from VRAM. The compute units are underutilized.

Diffusion generates N tokens in parallel per forward pass. If the forward pass costs 1.5-2x a single AR step (due to longer sequence), but produces 4-8 tokens, the effective throughput is 2-5x higher. This is the core value proposition.

### Hard Constraints

1. **No Gemma4 2B draft exists** — hidden_size=2816 is unique to Gemma4 26B; no smaller model shares it
2. **EAGLE3 for Gemma4 31B is incompatible** — different hidden_size (3072 vs 2816)
3. **262K vocabulary** — any draft/diffusion model must match this exact vocab size or use embedding-space methods
4. **VRAM budget** — 32 GB on 5090 (14.8 GB for model, ~17 GB remaining); 96 GB per GPU on PRO 6000
5. **Training budget** — 3-5 days max GPU time on PRO 6000 DP=2
6. **Quality bar** — must maintain >90% of Gemma4 26B quality on coding tasks

### Why Not Just Use N-gram?

N-gram spec decode (projected 160-320 tok/s) is the zero-effort baseline and should be deployed immediately. But n-gram has structural limits:
- Acceptance rate drops to 20% for novel generation (general chat, creative coding)
- Cannot exceed ~2.5x speedup even for highly repetitive workloads
- Cannot achieve 500+ tok/s target

Diffusion methods can potentially achieve 4-8x effective parallelism, reaching 500-1000 tok/s. The question is whether we can train a good enough diffusion head within the budget.

---

## 2. Approach A: Draft Model with Diffusion Head

### A1. FusenDiffusion-Lite: Minimal DFlash-Style Head

**Concept:** Train a small diffusion head (3-5 transformer layers) that takes hidden states from the middle of Gemma4 26B and proposes N=4-8 tokens in parallel using 2-4 denoising steps. This is the closest analog to DFlash (which works for Qwen3.5) adapted for diffusion.

**Architecture:**
```
Target model forward (30 layers) → extract hidden states at layer 15
  → Project: Linear(2816, 2816)
  → 3x Transformer blocks (2816 dim, 8 heads, 256 head_dim)
  → Noise schedule: absorbing mask diffusion (2-4 steps)
  → Output: N=4-8 token logits over 262K vocab
  → Speculative verify: target model scores all N tokens in one pass
```

**How it works:**
1. Target model runs normally, producing hidden states at layer 15
2. Diffusion head receives these hidden states + N mask tokens
3. Denoising step 1: predict all N tokens from noise (using cross-attention to layer-15 states)
4. Denoising step 2-3: refine predictions (re-run head with partially denoised tokens)
5. Target model verifies: run full 30 layers on all N proposed tokens (single batched forward pass)
6. Accept tokens that match target distribution (standard speculative rejection sampling)

**Parameters to train:** ~180M
- 3 transformer blocks: 3 × (4 × 2816² + 2 × 2816 × 11264) ≈ 160M
- Input projection: 2816² ≈ 8M
- Output LM head: shared with target (frozen) ≈ 0 trainable
- Noise embedding: 4 × 2816 ≈ 11K (negligible)

**Training data:** Extract hidden states from Gemma4 26B on 50K prompts × 200 tokens = 10M tokens. For each token position, record (layer_15_hidden, next_4_tokens). Total dataset: ~10M × (2816 × 2 bytes + 4 × 4 bytes) ≈ 56 GB of hidden states. Can be generated in streaming fashion (no full storage needed).

**Training procedure:**
1. Generate hidden states: 10M tokens at 120 tok/s = ~23 hours on 5090
2. Train diffusion head: 180M params, 10M samples, 3 epochs, batch=64
   - FLOPs: 3 epochs × 10M × 2 × 180M ≈ 1.1 × 10^16
   - RTX 5090 at 312 TFLOPS (FP16): 1.1e16 / 312e12 ≈ 35 seconds of raw compute
   - With memory overhead and data loading: ~2-4 hours realistic
3. Total: ~25-27 hours

**Training feasibility on PRO 6000 DP=2:** Easily feasible. One GPU generates data, the other trains (or both generate, then both train). Total: ~15 hours with DP=2.

**Expected speedup:**
- Denoising steps: 3 (each step ≈ 1 forward through 3-layer head ≈ 0.3 ms)
- Proposal cost: 3 × 0.3 ms = 0.9 ms for N=4 tokens
- Target verification: 1 forward pass with N=4 tokens ≈ 8.5 ms (slightly more than single token due to longer attention)
- Total step: 9.4 ms
- Expected acceptance: alpha ≈ 0.55-0.65 (layer 15 captures ~60-70% of final distribution)
- Effective tokens per step: (1 - 0.6^5) / 0.4 ≈ 2.3 tokens
- Effective tok/s: 2.3 / 9.4 ms ≈ **245 tok/s**
- With N=8 and alpha=0.55: (1 - 0.55^9) / 0.45 ≈ 2.2 / 10.0 ms ≈ **220 tok/s**

**Expected quality:** Lossless — speculative decoding with rejection sampling guarantees identical output distribution to the target model.

**Implementation complexity:** 20-30 ASI-hours
- Adapt DFlash proposer code (already have working `dflash.py`)
- Add diffusion noise schedule to the head
- Training script for hidden state extraction + head training
- Integration with vLLM spec decode scheduler

**Plugin compatibility:** Yes — fits directly into vLLM's spec decode framework as a new proposer type. The fusen_solver routes to the same backend.

**Risk level:** LIKELY. DFlash already works for Qwen3.5 at 94.6 tok/s. The diffusion variant adds denoising steps but proposes more tokens per step. The architecture is proven; the novelty is training a Gemma4-specific head.

---

### A2. FusenDiffusion-Deep: Multi-Layer Extraction

**Concept:** Instead of extracting hidden states from only layer 15, extract from layers 5, 10, 15, 20, 25 and use cross-attention between the diffusion head and all five extraction points. Richer signal should yield higher acceptance rates.

**Architecture:**
```
Target model forward (30 layers):
  → Extract h5 at layer 5   (2816 dim)
  → Extract h10 at layer 10 (2816 dim)
  → Extract h15 at layer 15 (2816 dim)
  → Extract h20 at layer 20 (2816 dim)
  → Extract h25 at layer 25 (2816 dim)
  
Diffusion head:
  → Concatenate: [h5; h10; h15; h20; h25] through gated fusion
  → 5x Transformer blocks with cross-attention to each extraction layer
  → Denoising: 2-3 steps
  → Output: N=8 token proposals
```

**Parameters to train:** ~500M
- 5 gated projection layers: 5 × 2816² ≈ 40M
- Fusion gate: 5 × 2816 ≈ 14K
- 5 transformer blocks with cross-attention: 5 × (4 × 2816² + 5 × 2 × 2816² + 2 × 2816 × 11264) ≈ 450M
- Output head: shared

**Training data:** Same as A1 but store 5 hidden states per position. Storage: 10M × 5 × 2816 × 2 ≈ 280 GB. Must stream — too large to store fully. Alternative: generate on-the-fly during training (teacher forward + student forward in same batch).

**Training time:**
1. On-the-fly training: each batch runs Gemma4 26B forward (8.3 ms/token) + head backward (~1 ms)
2. 10M tokens × 3 epochs / batch_size_64 ≈ 470K steps
3. Each step: ~0.6 seconds (forward teacher + forward/backward head)
4. Total: ~78 hours on single GPU, ~40 hours on DP=2

**Training feasibility on PRO 6000 DP=2:** Feasible within 2-day budget. Each GPU runs an independent replica with different data shards.

**Expected speedup:**
- Higher acceptance rate due to richer signal: alpha ≈ 0.65-0.75
- Larger head = slower proposal: 5 layers × 0.5 ms = 2.5 ms for 3 denoising steps = 7.5 ms
- Verification: 8.5 ms
- Total: 16 ms
- Effective tokens (N=8, alpha=0.70): (1 - 0.7^9) / 0.3 ≈ 3.2
- Effective tok/s: 3.2 / 16 ms ≈ **200 tok/s**

Wait — the larger head and longer denoising eat the acceptance gain. This is actually slower than A1 because proposal cost dominates.

**Revised with 2 denoising steps:**
- Proposal: 2 × 2.5 ms = 5 ms
- Total: 13.5 ms
- Alpha drops to ~0.60 with fewer steps
- Effective tokens (N=8, alpha=0.60): (1 - 0.6^9) / 0.4 ≈ 2.5
- Effective tok/s: 2.5 / 13.5 ms ≈ **185 tok/s**

**Verdict:** A2 is inferior to A1 unless the acceptance rate gain from multi-layer extraction is dramatic (alpha > 0.80). The cross-attention overhead to 5 layers is expensive. Not recommended as a first approach.

**Risk level:** SPECULATIVE. No existing work on multi-layer extraction for diffusion drafting. The acceptance rate improvement is unproven.

---

### A3. FusenDiffusion-MoE: Expert-Aware Draft

**Concept:** The draft model receives not just hidden states but also the router logits (which experts were activated). This gives the draft information about the token's "expert trajectory" — which should improve proposals for MoE-specific models.

**Architecture:**
```
Target model forward:
  → Extract hidden_states at layer 15 (2816 dim)
  → Extract router_logits at layers 0-29 (30 × 128 = 3840 dim)
  → Compress router logits: Linear(3840, 512) → router_embedding

Diffusion head:
  → Input: [hidden_states; router_embedding] → 3328 dim
  → 3x Transformer blocks (3328 dim)
  → Denoising: 2-3 steps
  → Output: N=4-8 token proposals
```

**Parameters to train:** ~250M
- Router compression: 3840 × 512 ≈ 2M
- 3 transformer blocks at 3328 dim: ~210M
- Projection layers: ~38M

**Training data:** Must extract router logits during teacher forward. Gemma4 26B computes `top_k(gating_output, k=8)` at each of 30 layers — the full gating output is 128-dim per layer. Storing all 30 layers' router logits: 30 × 128 × 4 bytes = 15 KB/token. For 10M tokens: 150 GB. Feasible to stream.

**Training time:** Similar to A1 (~25-30 hours) since the head size is similar and the extra router extraction is cheap.

**Expected speedup:** Similar to A1 (245 tok/s) but with potentially better acceptance for MoE models. The router logits tell the head which expert "subspace" the token lives in, potentially improving alpha from 0.60 to 0.65-0.70.

**Expected quality:** Lossless (speculative decode).

**Implementation complexity:** 30-40 ASI-hours. Requires modifying vLLM to expose router logits during decode (currently discarded after top-k selection). Non-trivial vLLM surgery.

**Risk level:** RESEARCH. No existing work on MoE-aware diffusion drafting. The hypothesis that router logits improve draft quality is plausible but unproven. The vLLM modification to expose router logits is the main engineering risk.

---

### A4. FusenDiffusion-Cascade: Multi-Resolution

**Concept:** Two-stage diffusion cascade. Stage 1 generates many coarse proposals fast. Stage 2 refines the best ones. Target verifies the refined candidates.

**Architecture:**
```
Stage 1 (Coarse): MLP-only head, 1 denoising step
  → Input: layer 15 hidden states
  → Output: N=32 token sketches (rough logits)
  → Select top-8 by confidence

Stage 2 (Fine): 3-layer transformer head, 2 denoising steps
  → Input: layer 15 hidden states + stage 1's top-8 tokens
  → Output: 8 refined token proposals

Stage 3: Target verification (standard spec decode)
```

**Parameters to train:** ~200M total (Stage 1: ~20M MLP, Stage 2: ~180M transformer)

**Training data:** Same as A1, but Stage 1 trains on all-tokens prediction and Stage 2 trains on refining Stage 1's outputs. Requires two-phase training.

**Training time:** ~35-40 hours (two models, two phases)

**Expected speedup:**
- Stage 1: 0.1 ms (MLP, trivial)
- Stage 2: 2 × 0.9 ms = 1.8 ms (3-layer head, 2 denoising steps on 8 tokens)
- Verification: ~9 ms (8 tokens)
- Total: 11 ms
- The cascade should yield alpha ≈ 0.60-0.65 (stage 2 refining helps)
- Effective tokens (N=8, alpha=0.62): (1 - 0.62^9) / 0.38 ≈ 2.6
- Effective tok/s: 2.6 / 11 ms ≈ **236 tok/s**

**Verdict:** Marginal improvement over A1 for significant complexity. The cascade buys ~10% more tokens through better candidate selection but adds training and inference complexity.

**Risk level:** SPECULATIVE. Cascaded diffusion works well in image generation but has not been validated for discrete token drafting.

---

### A5. Self-Speculative Diffusion: No Separate Draft Model

**Concept:** Use Gemma4 26B itself but skip layers (run only layers 0, 5, 10, 15, 20, 25 — 6 of 30). Add lightweight diffusion noise to the 6-layer output and denoise with 2-3 passes through the same 6 layers. No additional model to train.

**Architecture:**
```
Draft forward: Run layers {0, 5, 10, 15, 20, 25} → 6-layer forward
  → Add absorbing noise to N=4 positions
  → Denoise: 2 passes through same 6 layers
  → Output: N=4 token proposals

Target forward: Full 30 layers verify
```

**Parameters to train:** ~0 (only a noise schedule to calibrate — a few hundred parameters)

**Training data:** Calibration only — run 1K prompts through full model vs 6-layer model, measure divergence, tune noise schedule to minimize KL divergence. ~1 hour.

**Expected speedup:**
- 6-layer forward cost: c = 6/30 = 0.20 of full model
- But 3 denoising passes: effective c = 3 × 0.20 = 0.60
- With N=4, alpha ≈ 0.45 (6 layers capture far less than 30):
  - Effective tokens: (1 - 0.45^5) / 0.55 ≈ 1.65
  - Step cost: 1 + 4 × 0.60 = 3.4
  - Speedup: 1.65 / 3.4 = **0.49x — SLOWER**

**Even with N=8 and alpha=0.50:**
- Effective tokens: (1 - 0.5^9) / 0.5 ≈ 2.0
- Step cost: 1 + 8 × 0.60 = 5.8
- Speedup: 2.0 / 5.8 = **0.34x — MUCH SLOWER**

The problem: MoE models are bandwidth-bound, and each of the 6 layers still reads the same expert weights. The "skipped" layers only save bandwidth proportional to their share. Three denoising passes triple the cost, destroying any advantage.

**With 1 denoising step (no iterative refinement):**
- c = 0.20
- N=4, alpha ≈ 0.35 (1 step is very noisy)
- Effective tokens: (1 - 0.35^5) / 0.65 ≈ 1.40
- Step cost: 1 + 4 × 0.20 = 1.80
- Speedup: 1.40 / 1.80 = **0.78x — still slower**

**Verdict:** Self-speculative diffusion is NOT viable for MoE models on bandwidth-bound hardware. The layer-skipping savings are too small relative to the cost of multiple denoising passes. This matches the finding in `speculative_decoding_feasibility.md` that pruned-layer drafts (c ≈ 0.83) cause net slowdown.

**Risk level:** PROVEN NEGATIVE. The math shows this cannot work for Gemma4 26B MoE regardless of noise schedule quality.

---

## 3. Approach B: Full Diffusion Model (Standalone Generation)

### B1. Distill Gemma4 into Small MDLM-Style Diffusion Model

**Concept:** Train a standalone 2B-parameter masked diffusion language model (MDLM) that learns to generate text matching Gemma4 26B's output distribution. No speculative decode — this replaces the AR model entirely for speed-critical tasks.

**Architecture:**
```
MDLM (Masked Diffusion Language Model):
  → Embed: 262144 × 1536 (shared with Gemma4 E2B embedding)
  → 24 transformer layers, dim=1536, 12 heads
  → Absorbing-state noise schedule (tokens either correct or [MASK])
  → Denoising steps: 8-16 for high quality, 2-4 for speed
  → Output: 262144-dim logits per position
```

**Parameters to train:** ~2.3B (full model from scratch)

**Training data:** 
- Generate 500K examples from Gemma4 26B teacher (coding focus)
- At 120 tok/s, 500 tokens/example: ~580 GPU-hours for data generation
- Alternative: use public code datasets (The Stack, CodeContests) — faster but no Gemma4 flavor

**Training procedure:**
- Score matching objective: predict clean tokens from noised tokens
- 2.3B params, 250M training tokens, 5 epochs
- FLOPs: 5 × 250M × 2 × 2.3B × 6 ≈ 3.5 × 10^18
- PRO 6000 DP=2 at 624 TFLOPS: 3.5e18 / 624e12 ≈ 5,600 seconds ≈ 1.6 hours raw
- With real overhead (3-5x): **5-8 hours**

**BUT:** This vastly underestimates. 250M tokens is tiny for a 2.3B model. For reasonable quality, need 5-50B training tokens:
- 5B tokens × 5 epochs × 2 × 2.3B × 6 ≈ 3.5 × 10^20
- PRO 6000 DP=2: 3.5e20 / 624e12 ≈ 560,000 seconds ≈ **6.5 days**

**Training feasibility on PRO 6000 DP=2:** Marginal. 5B tokens requires ~6.5 days, which is at the edge of the 3-5 day budget. Quality at 5B tokens for a 2.3B model will be mediocre — typical LLM training uses 20-100x the parameter count in tokens.

**Expected speedup:**
- Diffusion model generates all tokens in parallel
- At 16 denoising steps, each step is one forward pass through 2.3B params
- Forward pass: 2.3B × 2 bytes / 1792 GB/s ≈ 2.6 ms (bandwidth-bound on PRO 6000)
- 16 steps × 2.6 ms = 41.6 ms
- For a 200-token response: 200 tokens / 41.6 ms = **4,808 tok/s**
- For 4 denoising steps: 200 / (4 × 2.6 ms) = **19,231 tok/s**

These numbers look absurdly high because diffusion generates ALL tokens simultaneously. The real question is quality.

**Expected quality:**
- At 16 steps with 5B training tokens: maybe 50-60% of Gemma4 26B quality
- At 4 steps: maybe 30-40% quality
- Code generation: likely incoherent for complex logic, acceptable for boilerplate
- The fundamental issue: discrete diffusion models in 2025-2026 (MDLM, SEDD, Mercury, Plaid) achieve ~70-80% of same-size AR model quality, and we're training a 2B model to match a 26B teacher

**Implementation complexity:** 80-120 ASI-hours
- Build MDLM architecture from scratch (no Gemma4-compatible MDLM exists)
- Training pipeline with score matching loss
- Integration with fusen_solver as a "fast" backend
- Sampling with configurable denoising steps

**Plugin compatibility:** Yes — serves as a separate backend in fusen_solver. The solver routes "fast/simple" queries to diffusion, "complex" to AR.

**Risk level:** SPECULATIVE. Diffusion LLMs are an active research area. No production-quality diffusion LLM exists at 2B scale that matches AR quality. Training on PRO 6000 is at the edge of feasibility.

---

### B2. Fine-Tune Existing Diffusion LLM on Gemma4 Outputs

**Concept:** Download a pretrained diffusion LLM (Mercury Coder, MDLM base, Plaid, SEDD) and fine-tune on Gemma4-generated code.

**Available checkpoints (as of April 2026):**
- **Mercury Coder 7B** — Inception Labs' diffusion coding model, ~3x faster than Llama 3.1 8B
- **MDLM base** — research checkpoint from Sahoo et al., small scale
- **SEDD** — Score Entropy Discrete Diffusion, research scale
- **Dream** — Diffusion Reasoning Model, recent (2025)

**Problem:** None of these use Gemma4's 262K vocabulary. Fine-tuning requires:
1. Replace the embedding + output layers (262K vocab)
2. Randomly initialize new embeddings
3. Fine-tune entire model on Gemma4-tokenized data
4. The random embedding initialization means we're essentially training from scratch for the vocabulary portion

**Parameters to train:** ~7B (full model fine-tune with new embeddings)

**Training data:** 100K-500K code examples tokenized with Gemma4 tokenizer

**Training time:** 
- 7B model, full fine-tune, 500K examples × 500 tokens = 250M tokens
- 250M tokens is grossly insufficient for 7B params — need 10B+ tokens minimum
- At 10B tokens: FLOPs ≈ 10B × 2 × 7B × 6 ≈ 8.4 × 10^20
- PRO 6000 DP=2: 8.4e20 / 624e12 ≈ 1.35M seconds ≈ **15.6 days**

**Training feasibility on PRO 6000 DP=2:** NOT feasible within budget. Even with LoRA (training ~1% of params), the forward pass through 7B params for each training step takes the same time, so LoRA only saves memory, not compute.

**Expected quality:** If we somehow had enough compute — 7B diffusion model fine-tuned on enough data could reach 70-75% of Gemma4 26B quality on coding tasks. But the vocab mismatch and limited compute budget make this unrealistic.

**Risk level:** NOT RECOMMENDED. The vocab mismatch and training budget make this impractical.

---

### B3. Continuous Diffusion in Embedding Space

**Concept:** Instead of discrete token diffusion (absorbing/masking tokens), diffuse in the continuous embedding space. Start from Gaussian noise in R^2816, denoise to embeddings, snap to nearest vocabulary token.

**Architecture:**
```
Noise: z ~ N(0, I) ∈ R^{N × 2816}
Denoiser: 12-layer transformer, dim=2816
  → Input: noisy embeddings at each position
  → Conditioning: prompt embeddings from Gemma4 encoder
  → Output: denoised embeddings
  → Snap: nearest-neighbor in Gemma4 embedding matrix (262K × 2816)
```

**Parameters to train:** ~1.5B (12 layers at 2816 dim)

**Training data:** Gemma4 26B embedding outputs for 50K-500K examples. Extract the final embedding at each token position. Train with flow matching or DDPM objective in embedding space.

**Training time:** ~3-4 days on PRO 6000 DP=2 (similar compute to B1 but slightly smaller model)

**Expected speedup:**
- 8 denoising steps × ~1.7 ms per step = 13.6 ms for 200 tokens
- 200 / 13.6 ms = **14,700 tok/s** (theoretical)
- Plus nearest-neighbor lookup: 200 × 262K × 2816 dot products — this is a single matmul (200, 2816) × (2816, 262K) ≈ 0.3 ms on PRO 6000
- Realistic: **~10,000 tok/s** for 200-token generation with 8 steps

**Expected quality:** POOR. Continuous diffusion in embedding space has a fundamental problem: the embedding space is not smooth. Nearby embeddings can correspond to completely unrelated tokens. The "snap to nearest" step introduces discrete errors that compound. Research results (Li et al., 2022; Dieleman et al., 2022) show continuous diffusion in embedding space consistently underperforms discrete diffusion by 10-30% on text metrics.

**Risk level:** RESEARCH. Interesting direction but quality is likely unacceptable for coding tasks.

---

## 4. Approach C: Hybrid AR + Diffusion

### C1. AR Planning + Diffusion Expansion

**Concept:** Gemma4 26B generates a skeleton (outline/pseudocode/key tokens), then a diffusion model fills in the details. This exploits the observation that in code, ~20-30% of tokens carry the semantic load (function names, control flow) while ~70-80% are syntactic filler (brackets, common patterns, boilerplate).

**Architecture:**
```
Phase 1 (AR, Gemma4 26B): Generate skeleton tokens
  "def sort_list(arr):\n    if len(arr) <= 1: return arr\n    ..."
  → Gemma4 outputs key tokens: [def, sort, list, arr, if, len, arr, <=, 1, return, arr, ...]
  → ~30 tokens at 120 tok/s = 250 ms

Phase 2 (Diffusion, 2B model): Fill in syntax/boilerplate
  → Input: skeleton tokens at fixed positions
  → Diffusion generates the ~70 filler tokens between anchors
  → 4 denoising steps at 2.6 ms each = 10.4 ms for all 70 tokens

Total: 250 ms + 10.4 ms ≈ 260 ms for 100 tokens → ~385 tok/s
```

**Parameters to train:** ~2B (diffusion infill model) + 0 (Gemma4 frozen)

**Training data:** 
- For each code example, identify "skeleton" tokens (function names, keywords, operators) vs "filler" tokens
- Train diffusion model on infilling: given skeleton positions, predict filler tokens
- 50K code examples, ~500 tokens each = 25M tokens
- Skeleton identification: can be rule-based (Python keywords, bracket matching) or learned

**Training time:** ~8-15 hours on PRO 6000 DP=2 (small model, moderate data)

**Expected speedup:**
- Skeleton generation: 30% of tokens at 120 tok/s → 250 ms for 100-token output
- Infill: 70 tokens in 4 denoising steps → ~10 ms
- Overhead (orchestration, concatenation): ~2 ms
- Total: ~262 ms for 100 tokens → **382 tok/s**
- For longer outputs (500 tokens): 150 skeleton + 350 filler
  - Skeleton: 150 / 120 = 1.25 seconds
  - Filler: 4 × 2.6 ms = 10.4 ms
  - Total: 1.26 seconds → **397 tok/s**

**Expected quality:** Depends entirely on skeleton quality (which is Gemma4 quality) and the diffusion model's ability to infill correctly. For code, syntax infilling is highly constrained (matching brackets, correct indentation, common patterns), so quality should be high — maybe 85-95% of full AR.

**Implementation complexity:** 60-80 ASI-hours
- Skeleton extractor (rule-based for Python/JS/Go, or trained classifier)
- Infill diffusion model training
- Orchestration in fusen_solver: route phase 1 to Gemma4, phase 2 to diffusion backend
- Streaming: must buffer until infill is complete (latency spike at boundaries)

**Plugin compatibility:** Yes — fusen_solver already supports multi-backend routing. Add `strategy: skeleton_infill` to config.

**Risk level:** LIKELY for code, SPECULATIVE for natural language. Code has enough structure that skeleton+infill is well-defined. Natural language has no clear "skeleton" — every word can be semantically important.

---

### C2. Interleaved AR + Diffusion (Anchor Points)

**Concept:** Generate every K-th token with AR (anchor points), then fill in the K-1 tokens between anchors with diffusion. AR provides global coherence, diffusion provides local speed.

**Architecture:**
```
For K=4:
  Step 1: AR generates token at position 0 (8.3 ms)
  Step 2: Diffusion generates tokens at positions 1,2,3 (2 denoising steps, ~1.5 ms)
  Step 3: AR generates token at position 4, conditioned on 0-3 (8.3 ms)
  Step 4: Diffusion generates tokens at positions 5,6,7 (1.5 ms)
  ...

Per 4 tokens: 8.3 + 1.5 = 9.8 ms → 408 tok/s
```

**Parameters to train:** ~180M (small diffusion infill head, same as A1)

**Training data:** For each group of K=4 tokens in training data, train diffusion to predict tokens 1,2,3 given token 0 and the preceding context. 10M tokens, ~25 hours.

**Expected speedup:**
- K=4: 4 tokens in 9.8 ms → **408 tok/s**
- K=8: 8 tokens in (8.3 + 2.5) ms → **740 tok/s**
- K=16: 16 tokens in (8.3 + 4.0) ms → **1,300 tok/s**

**BUT:** Quality degrades rapidly with K. The diffusion model must predict K-1 tokens without seeing the AR model's output for those positions. At K=4, the 3 infilled tokens depend only on the anchor and context — they miss the AR model's conditioning signal for their own positions.

**Expected quality:**
- K=4: ~90% of AR quality (3 tokens of local context is usually sufficient)
- K=8: ~75% quality (7 tokens without AR guidance → drift accumulates)
- K=16: ~50% quality (15 tokens blind → significant coherence loss)

**Critical issue:** This is NOT lossless. Unlike speculative decode, there is no verification step. The diffusion tokens are accepted as-is. Quality loss is real and permanent.

**Making it lossless (with verification):**
- After diffusion fills in positions 1-3, run AR verification on all 4 positions
- Verification cost: ~8.5 ms (slightly more than single token)
- Total per 4 tokens: 8.3 + 1.5 + 8.5 = 18.3 ms → **218 tok/s**
- This is just fancy speculative decode with extra steps — worse than A1

**Risk level:** SPECULATIVE without verification (quality loss), LIKELY with verification (but then it's just slower A1).

---

### C3. Diffusion Draft + AR Refinement

**Concept:** Diffusion generates N tokens fast (possibly with errors), then AR does a single editing pass to fix mistakes.

**Architecture:**
```
Step 1: Diffusion generates 8 tokens (3 denoising steps, ~3 ms)
Step 2: AR model sees all 8 tokens + context, outputs corrected logits for all 8 positions
  → This is a single batched forward pass: ~9 ms
Step 3: For each position, if AR logit != diffusion token, replace
```

This is exactly speculative decoding. The "editing pass" is the verification step. This approach reduces to A1.

**Verdict:** Not a distinct approach — collapses to A1/standard speculative decode.

---

### C4. Progressive Refinement (Jacobi-Diffusion Hybrid)

**Concept:** Initialize N token positions with diffusion (1 step = rough draft). Then iteratively refine using the AR model's logits in a Jacobi-style parallel iteration. Each refinement fixes some tokens while keeping others. Converges to AR-quality output in 3-5 iterations instead of N sequential steps.

**Architecture:**
```
Iteration 0: Diffusion head generates initial guess for positions 1..N (1 step, ~1 ms)
Iteration 1: AR model scores all N positions in parallel → update tokens where argmax changed
Iteration 2: AR model re-scores all N positions → fewer tokens change
Iteration 3: AR model re-scores → most tokens fixed, maybe 1-2 still changing
Iteration 4: No tokens change → converged → emit all N tokens
```

**Parameters to train:** ~180M (diffusion head for initial guess) + 0 (AR model frozen)

The key insight: standard Jacobi decoding (without diffusion initialization) starts from random tokens and needs 5-15 iterations to converge. Diffusion initialization gives a much better starting point, reducing iterations to 3-5.

**Training data:** Same as A1 (diffusion head training)

**Expected speedup:**
- Diffusion init: 1 ms
- Each Jacobi iteration: ~9 ms (full AR forward on N tokens)
- 4 iterations × 9 ms + 1 ms = 37 ms for N=8 tokens
- 8 / 37 ms = **216 tok/s**

Compare to pure Jacobi (no diffusion init):
- 8 iterations × 9 ms = 72 ms for N=8 tokens → 111 tok/s (SLOWER than baseline)

Compare to A1 (speculative with 3-step diffusion head):
- Proposal: 0.9 ms, Verify: 8.5 ms = 9.4 ms for ~2.3 effective tokens → 245 tok/s

**Verdict:** A1 is faster. Progressive refinement needs too many AR forward passes. The AR forward pass is the bottleneck (weight reads), and doing 4 of them is 4x the baseline cost for only ~3-4x the tokens.

**Risk level:** LIKELY to work, but PROVEN INFERIOR to standard speculative decode for bandwidth-bound models.

---

## 5. Approach D: Training Variations

These apply to any of the trainable approaches above (primarily A1, A3, B1, C1).

### D1. Noise Schedule Variations

| Schedule | Description | Best For | Training Impact |
|----------|-------------|----------|-----------------|
| **Absorbing (recommended)** | Tokens are either correct or [MASK]. Binary state. | Discrete tokens, spec decode heads | Stable training, clean gradients, proven for text |
| **Linear** | Gaussian noise added linearly in embedding space | Continuous diffusion (B3) | Simple, well-understood |
| **Cosine** | More noise preserved at early steps, gradual degradation | Long sequences where later tokens are harder | 5-10% quality improvement for long outputs |
| **Uniform** | Equal corruption probability at all timesteps | Baseline comparison | No particular advantage |

**Recommendation:** Use absorbing schedule for all discrete approaches (A1-A4, C1-C4). Use cosine for continuous (B3) if pursued.

**Training impact:** Absorbing schedule is the default for MDLM/DFlash-family models. Switching to cosine adds ~2 hours of hyperparameter search.

### D2. Training Objective Variations

| Objective | Description | Pros | Cons |
|-----------|-------------|------|------|
| **Score matching (recommended)** | Predict denoised token at each masked position | Standard, stable, well-studied | Requires many samples for good gradients |
| **ELBO** | Variational lower bound on log-likelihood | Principled, measures true likelihood | Loose bound → suboptimal for generation |
| **Flow matching** | Learn velocity field from noise to data | Fewer denoising steps needed (2-4 vs 8-16) | Harder to train, recent technique |
| **Consistency** | One-step generation (no iterative denoising) | Fastest inference (1 step) | Hardest to train, quality drops |

**Recommendation:** Start with score matching (proven). If 4+ denoising steps are too slow, try flow matching to reduce to 2 steps. Consistency training is a stretch goal.

### D3. Data Variations

| Data Source | Tokens | Quality Signal | Best For |
|-------------|--------|----------------|----------|
| **Gemma4 26B generated code** | 10-50M | Direct teacher signal | Self-distillation heads (A1-A4) |
| **Public code (The Stack v2)** | 100B+ | General code distribution | Full diffusion models (B1-B2) |
| **HumanEval + CodeContests** | 1M | High-quality competitive programming | Fine-tuning / evaluation |
| **Gemma4 26B hidden states** | 10M positions | Internal representations | Draft heads (A1-A4) — required |
| **Multi-language text (C4, RedPajama)** | 100B+ | General text distribution | General-purpose diffusion models |

**Recommendation for A1:** Use Gemma4 26B hidden states (required) + generated code (for the token sequences). 10M tokens is sufficient for a 180M parameter head.

---

## 6. Approach E: Architecture Variations for the Diffusion Head

These apply to approaches A1-A4 where a diffusion head is trained.

### E1. Transformer Diffusion Head (Standard)

**Architecture:** Standard pre-norm transformer blocks with self-attention + cross-attention to teacher hidden states.

| Spec | Value |
|------|-------|
| Layers | 3-5 |
| Dim | 2816 (match teacher) |
| Heads | 8-16 |
| Head dim | 256 |
| Params | 180-300M |
| Latency per denoising step | 0.3-0.5 ms |
| Quality | Best |

**Recommendation:** Default choice. Proven in DFlash/EAGLE family.

### E2. Mamba Diffusion Head (Linear Attention)

**Architecture:** Replace self-attention with Mamba SSM blocks. O(N) instead of O(N^2) in sequence length.

| Spec | Value |
|------|-------|
| Layers | 4-6 |
| Dim | 2816 |
| State dim | 128 |
| Params | 150-250M |
| Latency per denoising step | 0.2-0.3 ms |
| Quality | ~5-10% worse than transformer for short sequences |

For N=4-8 speculative tokens, the sequence is extremely short. O(N^2) attention on 8 tokens is negligible. Mamba's advantage only appears at N > 64.

**Verdict:** Not beneficial for spec decode (N is too small). Could be useful for B1-type full diffusion models generating 200+ tokens.

### E3. MLP-Only Diffusion Head

**Architecture:** Stack of feed-forward layers with residual connections. No attention.

| Spec | Value |
|------|-------|
| Layers | 6-8 |
| Dim | 2816 |
| Params | 100-150M |
| Latency per denoising step | 0.1-0.2 ms |
| Quality | ~20-30% worse than transformer |

**Use case:** Only for the "coarse" stage in A4 (cascade), where speed matters more than quality. Not viable as the primary draft head.

### E4. Mixture-of-Experts Diffusion Head

**Architecture:** MoE within the diffusion head — route different token categories to different expert denoisers.

| Spec | Value |
|------|-------|
| Layers | 3 |
| Experts | 4 (top-2) |
| Dim | 2816 |
| Params | 350-500M total, ~200M active |
| Latency per denoising step | 0.4-0.6 ms |
| Quality | Potentially +5% vs dense (if experts specialize) |

**Risk:** Over-engineering for a 180M head. MoE makes sense at 7B+ scale, not at 180M. The routing overhead exceeds any quality benefit.

**Verdict:** Not recommended.

### E5. Convolutional Diffusion Head

**Architecture:** 1D convolutions over the token sequence, good for local pattern detection.

| Spec | Value |
|------|-------|
| Layers | 8-12 |
| Channels | 2816 |
| Kernel size | 3-7 |
| Params | 100-200M |
| Latency per denoising step | 0.15-0.25 ms |
| Quality | Good for code (local patterns), poor for long-range dependencies |

**Use case:** Hybrid with transformer — conv layers for local pattern, one transformer layer for global attention.

**Verdict:** Niche. Could be interesting as a speed optimization within E1, but not worth pursuing independently.

---

## 7. Approach F: Inference Optimizations for Diffusion

These apply to any diffusion approach and are independent of training.

### F1. Classifier-Free Guidance (CFG)

**What:** Scale the diffusion output toward or away from a conditioning signal (e.g., "generate code" vs "generate prose"). Multiply logits by guidance_scale > 1.0 to steer generation.

**Cost:** 2x forward passes per denoising step (conditional + unconditional).

**Benefit:** Better adherence to coding prompts, potentially +5-10% quality.

**Verdict:** Too expensive for spec decode heads (doubles proposal cost). Useful for B1-type standalone diffusion models where quality is the bottleneck.

### F2. Adaptive Denoising Steps

**What:** Use fewer denoising steps for "easy" token positions (common words, predictable syntax) and more steps for "hard" positions (variable names, logic).

**Implementation:**
- After step 1, compute entropy of predictions at each position
- Low entropy (< 0.5 nats) → accept immediately
- High entropy (> 2.0 nats) → continue denoising
- Medium → 1 more step

**Expected benefit:** Reduce average denoising steps from 3 to ~1.8, saving ~40% of proposal time.

**Cost:** Entropy computation is trivial (softmax + sum). Irregular computation pattern may prevent CUDA graph capture.

**Verdict:** Worth implementing. ~40% proposal cost reduction with minimal quality impact.

### F3. Cached Denoising

**What:** If denoising step N doesn't change most token predictions, skip recomputation for unchanged positions.

**Implementation:**
- After each step, compare predictions to previous step
- Only re-run the diffusion head on positions that changed
- Use a mask to select active positions

**Expected benefit:** Steps 2+ typically change <30% of positions → ~70% compute saving per step.

**Problem:** Masking creates irregular computation that prevents efficient GPU utilization. The overhead of managing the mask may exceed the savings for N=4-8 tokens.

**Verdict:** Not beneficial for small N (spec decode). Could help for B1-type models generating 200+ tokens.

### F4. Parallel Denoising on GPU

**What:** All N token positions denoise simultaneously — this is already inherent in the diffusion architecture. No additional optimization needed.

**Verdict:** Already the default. Diffusion's parallelism is the whole point.

### F5. CUDA Graph Capture for Denoising Loop

**What:** Capture the entire multi-step denoising loop as a single CUDA graph. Eliminates kernel launch overhead between denoising steps.

**Implementation:**
- CUDA graph captures: input → step1 → step2 → step3 → output
- Fixed N (number of tokens) and fixed number of denoising steps
- Replay graph instead of launching individual kernels

**Expected benefit:** Eliminate ~0.1-0.3 ms of launch overhead per denoising step. For 3 steps: save ~0.5 ms.

**Cost:** Must pre-capture graphs for each (N, num_steps) combination. Memory cost: ~50-100 MB per graph.

**Verdict:** Essential optimization. vLLM already uses CUDA graphs for AR decode; extending to denoising is straightforward.

---

## 8. Unified Scoring Matrix

Score formula: `(speedup × quality × probability) / (training_days + ASI_build_days)`

| # | Approach | Speedup (tok/s) | Quality | Probability | Train (days) | Build (days) | Score | Rank |
|---|----------|----------------|---------|-------------|-------------|-------------|-------|------|
| A1 | FusenDiffusion-Lite | 245 | 1.00 | 0.75 | 1.0 | 1.5 | 73.5 | **1** |
| C1 | AR Planning + Diffusion Expansion | 385 | 0.90 | 0.55 | 0.6 | 3.0 | 53.0 | **2** |
| A3 | FusenDiffusion-MoE | 260 | 1.00 | 0.50 | 1.5 | 2.0 | 37.1 | 3 |
| A4 | FusenDiffusion-Cascade | 236 | 1.00 | 0.45 | 1.5 | 2.5 | 26.6 | 4 |
| C2 | Interleaved (K=4, no verify) | 408 | 0.90 | 0.40 | 1.0 | 2.0 | 49.0 | — [a] |
| B1 | MDLM 2B distillation | 4808 | 0.55 | 0.25 | 6.5 | 5.0 | 57.5 | — [b] |
| A2 | FusenDiffusion-Deep | 200 | 1.00 | 0.55 | 1.7 | 2.0 | 29.7 | 5 |
| C4 | Jacobi-Diffusion | 216 | 1.00 | 0.70 | 1.0 | 1.5 | 60.5 | — [c] |
| B3 | Continuous embedding diffusion | 10000 | 0.40 | 0.15 | 3.5 | 4.0 | 80.0 | — [d] |
| A5 | Self-speculative diffusion | <120 | 1.00 | 0.00 | 0 | 0.5 | 0.0 | DEAD |
| B2 | Fine-tune existing diffusion LLM | ~5000 | 0.50 | 0.10 | 15.6 | 4.0 | 12.8 | DEAD |

Notes:
- [a] C2 without verification has unacceptable quality risk for production use
- [b] B1 has astronomically high speedup but quality is too uncertain — score is misleading
- [c] C4 is proven inferior to A1 for bandwidth-bound models — the math shows it clearly
- [d] B3 has theoretical numbers that won't survive contact with reality — embedding space diffusion quality is poor

**Quality = 1.00 means lossless** (speculative decode with rejection sampling). Any score < 1.00 means lossy.

### Adjusted Rankings (Lossless Only)

For a coding workstation where correctness matters:

| Rank | Approach | tok/s | Risk | Time to Deploy |
|------|----------|-------|------|----------------|
| 1 | **N-gram spec decode** (baseline) | 160-320 | None | Today |
| 2 | **A1: FusenDiffusion-Lite** | ~245 | Low | 2.5 days |
| 3 | **A3: FusenDiffusion-MoE** | ~260 | Medium | 3.5 days |
| 4 | **A4: FusenDiffusion-Cascade** | ~236 | Medium | 4 days |
| 5 | **A2: FusenDiffusion-Deep** | ~200 | Medium | 3.7 days |

### Adjusted Rankings (Lossy Allowed for Non-Critical Tasks)

For boilerplate, documentation, test generation where 90% quality is acceptable:

| Rank | Approach | tok/s | Quality | Time to Deploy |
|------|----------|-------|---------|----------------|
| 1 | **C1: AR Planning + Diffusion** | ~385 | 90% | 3.6 days |
| 2 | **C2: Interleaved K=4** | ~408 | 90% | 3 days |
| 3 | **B1: MDLM 2B distillation** | ~4800 | 55% | 11.5 days |

---

## 9. Recommended Timeline

### THIS WEEK (Before PRO 6000 — RTX 5090 Only)

**Day 1-2: Deploy N-gram speculative decode (zero training)**

This is the unambiguous first move. Zero risk, zero training, immediate 1.5-2.5x speedup.

```python
# vLLM launch config
speculative_config = {
    "method": "ngram_gpu",
    "num_speculative_tokens": 4,
    "prompt_lookup_min": 1,
    "prompt_lookup_max": 5,
}
```

Expected result: 160-320 tok/s depending on workload.

**Day 3-4: Begin A1 data generation**

Start extracting hidden states from Gemma4 26B on coding prompts.
- Run Gemma4 on 50K coding prompts (HumanEval, CodeContests, The Stack samples)
- Extract layer-15 hidden states and next-4 tokens at each position
- Store as streaming dataset (HuggingFace datasets format)
- This runs alongside normal workload — just capture hidden states during inference

**Day 5-7: A1 architecture + training scaffold**

Build the FusenDiffusion-Lite training pipeline:
- Define the 3-layer transformer diffusion head (PyTorch)
- Implement absorbing noise schedule
- Implement score matching loss
- Test training on a small dataset (~100K tokens) to verify convergence
- Profile training speed to validate PRO 6000 time estimates

### NEXT WEEK (First Week with PRO 6000)

**Day 1-2: PRO 6000 benchmarking (per pro6000_projections.md)**

Run the DP=2 benchmark suite. Establish baseline numbers. This is critical — all speed projections depend on actual hardware measurements.

**Day 2-3: A1 training on PRO 6000 DP=2**

- GPU 0: Continue hidden state extraction (if not complete) + serve model
- GPU 1: Train FusenDiffusion-Lite head
- ~15-25 hours total

**Day 4-5: A1 integration with vLLM**

- Implement `FusenDiffusionProposer` (extending `DFlashProposer` — we already have the DFlash code)
- Key modifications from DFlash:
  - Add denoising loop (DFlash is single-pass; diffusion needs 2-3 steps)
  - Add noise schedule to input preparation
  - Modify `set_inputs_first_pass` to handle iterative refinement
- Integration point: `v1/spec_decode/fusendiffusion.py`

**Day 6-7: Testing and benchmarking**

- Measure acceptance rate on coding prompts
- Measure end-to-end tok/s
- Compare to n-gram baseline
- If acceptance rate < 0.50: try N=6 instead of N=4 (fewer tokens but higher acceptance)
- If acceptance rate > 0.65: try N=8 (more tokens per step)

### THIS MONTH (Weeks 3-4)

**Week 3: Optimize A1 based on measurements**

- Implement F2 (adaptive denoising steps) — reduce average steps from 3 to ~1.8
- Implement F5 (CUDA graph for denoising loop)
- If A1 acceptance is low, try A3 (MoE-aware head with router logits)
- Sweep noise schedule: absorbing vs cosine, measure impact

**Week 4: Explore C1 (AR Planning + Diffusion) for non-critical tasks**

- Build skeleton extractor for Python code
- Train diffusion infill model (separate from A1)
- Integrate with fusen_solver: route boilerplate/test-gen to C1, complex coding to Gemma4 AR
- This is additive — does not replace A1, provides a second "fast lane"

### RESEARCH (Do Not Build Yet)

**B1: Full MDLM distillation** — Wait for:
- Diffusion LLM quality to improve (Mercury v2, Plaid 2.0)
- Access to more compute (multi-node training)
- Better understanding of A1 acceptance rates (if A1 gets alpha > 0.70, B1 is unnecessary)

**B3: Continuous embedding diffusion** — Wait for:
- Academic results showing embedding-space diffusion matching discrete quality
- Currently a 10-30% quality penalty is unacceptable

**A3: MoE-aware draft** — Only if A1 acceptance rate is disappointing (< 0.50):
- The hypothesis that router logits help drafting is unproven
- vLLM surgery to expose router logits is non-trivial
- Only justified if A1 fails to meet expectations

**E2: Mamba diffusion head** — Only if generating N > 32 tokens per step:
- At N=4-8, transformer attention is negligible
- Mamba only helps for long-sequence diffusion (B1-type models)

---

## 10. Appendix: Key Numbers

### Gemma4 26B MoE Decode Cost Breakdown

```
Per-token decode (C=1, RTX 5090):
  Total:           8.3 ms
  Weight reads:    5.0 ms (60%) — 2.47B active params × 2 bytes / 1792 GB/s ≈ 2.75 ms
                                   + MoE routing/gather overhead ≈ 2.25 ms
  KV read/write:   1.5 ms (18%) — 30 layers × 2 × 8 heads × 256 dim × 2 bytes × seq_len
  Attention:        0.8 ms (10%) — FlashAttention on 30 layers
  Overhead:         1.0 ms (12%) — CUDA graph replay, Python scheduler, etc.
```

### Speculative Decode Math Reference

```
Effective tokens per step = (1 - alpha^(N+1)) / (1 - alpha)
Step cost (with draft) = T_target + N × T_draft
Speedup = effective_tokens × T_baseline / step_cost

Where:
  alpha = acceptance rate (probability draft token matches target)
  N = number of speculative tokens
  T_target = target model forward pass time
  T_draft = draft model forward pass time (per token)
  T_baseline = single-token AR decode time
```

### Diffusion Denoising Step Cost

```
For a K-layer, D-dim transformer head with N tokens:
  Self-attention: N^2 × D × K flops
  Cross-attention: N × C × D × K flops (C = context length)
  FFN: N × D × 4D × 2 × K flops
  
At D=2816, K=3, N=8:
  Self-attention: 64 × 2816 × 3 ≈ 0.5M flops (negligible)
  FFN: 8 × 2816 × 11264 × 2 × 3 ≈ 1.5B flops
  At 312 TFLOPS: 1.5B / 312T ≈ 0.005 ms (compute-bound time)
  
  But bandwidth-bound: 180M params × 2 bytes / 1792 GB/s ≈ 0.2 ms
  Real time with overhead: ~0.3 ms per denoising step
```

### PRO 6000 DP=2 Training Throughput

```
Per GPU (96 GB, SM120 Blackwell):
  FP16 TFLOPS: ~312 (same SM count as 5090)
  Memory: 96 GB → can hold 2.3B student + 14.8B teacher in NVFP4
  Training batch: can fit batch=64 of 512-token sequences

DP=2 training throughput:
  2 × 312 = 624 TFLOPS aggregate
  No communication overhead (independent data shards)
  
180M head training (A1):
  FLOPs for 10M samples × 3 epochs: ~10^16
  At 624 TFLOPS: ~16 seconds raw, ~2-4 hours real
  
2.3B model training (B1):
  FLOPs for 5B tokens: ~3.5 × 10^20
  At 624 TFLOPS: ~560K seconds ≈ 6.5 days
```

### Vocabulary Compatibility Matrix

| Model | Vocab Size | Compatible with Gemma4 262K? |
|-------|-----------|------------------------------|
| Gemma4 26B | 262,144 | Yes (target) |
| Gemma4 E2B | 262,144 | Yes |
| Qwen3.5 family | 248,320 | No |
| LLaMA 3.x | 128,256 | No |
| Mercury Coder | 32,000 | No |
| Custom trained | 262,144 | Yes (if using Gemma4 tokenizer) |

**Any diffusion head for speculative decode must output logits over the full 262,144 vocab.** This is a 262K × 2816 output projection matrix = 738M parameters just for the LM head. In practice, this is shared (frozen) from the target model.

---

## Summary

The path to 500+ tok/s single-user on Gemma4 26B MoE is:

1. **Today:** N-gram spec decode → 160-320 tok/s (free, zero risk)
2. **Week 2:** FusenDiffusion-Lite (A1) → ~245 tok/s (low risk, 2.5 days)
3. **Week 3:** A1 + optimizations (F2, F5) → ~280-320 tok/s (medium effort)
4. **Week 4:** C1 skeleton+infill for non-critical tasks → ~385 tok/s (lossy, code only)

The 500 tok/s target with lossless quality is **not achievable** with current diffusion techniques on Gemma4 26B MoE. The fundamental limit is that the target model verification step costs 8.3 ms per step regardless of how fast the draft is. At 8.3 ms per step, even with perfect acceptance and N=8, the ceiling is 8 / 8.3 ms = 964 tok/s — but this requires alpha=1.0 (impossible). At realistic alpha=0.70, N=8: 3.2 effective tokens / 9 ms ≈ 356 tok/s.

**To break 500 tok/s with lossless quality requires:**
- Faster target model verification (TP=2 on NVLink → ~5 ms/step → 640 tok/s ceiling)
- OR a smaller target model (Gemma4 E2B → 400-600 tok/s baseline, no diffusion needed)
- OR lossy generation (C1 at 385 tok/s, B1 at 4800 tok/s with 55% quality)

The honest answer: **diffusion-based speculative decoding can reach ~300 tok/s lossless on PRO 6000. For 500+ tok/s, you must either accept quality loss or use a smaller model.**
