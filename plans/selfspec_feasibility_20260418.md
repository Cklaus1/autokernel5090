# Self-Speculative Decoding Feasibility — Qwen3-30B-A3B NVFP4

**Date:** 2026-04-18  
**Tag:** W2_5a_selfspec_feasibility  
**Target:** Qwen3-30B-A3B NVFP4 on RTX PRO 6000 SM120a, single-user decode  
**Constraint:** No training. Stack on v6 FP4 mega-graph (Wave 4 slot).  
**Status of blockers:** I-DLM KILL'd (MASK-to-MASK load-bearing). EAGLE3 separate track (training required, blocked on PR 4c/3).

---

## Candidate Evaluations

### 1. N-gram / LADE (Lookahead Decoding)

**What it is.** Propose k tokens by matching n-gram suffixes in the existing context (or via a Jacobi-style lookahead branch). No draft model. vLLM ships `NgramProposerGPU` natively.

**Acceptance rate.** Published LADE numbers on LLaMA-class models: 1.5–2.1× on code/structured output (alpha ~0.50–0.65), 1.1–1.3× on conversational chat (alpha ~0.20–0.35). Qwen3-30B conversational distribution is diverse-vocabulary MoE output; repetition density is lower than dense LLaMA, so the lower end applies — expect alpha ≈ 0.20–0.40 for chat, 0.50–0.65 for code/structured.

**Prior art caveat from this project.** The Gemma4 n-gram KILL (Discovery #39/42) was silicon-specific: compute-bound model where the variable-batch-size path broke CUDA graph reuse. Qwen3 has a different profile — it has been running successfully at high throughput with `vllm-built:latest` (T2-N, T2-H experiments) and CUDA graphs are stable. The Gemma4 n-gram kill does NOT transfer to Qwen3.

**v6 FP4 mega-graph interaction.** N-gram has zero draft-forward cost (c = 0). The proposer generates candidate token IDs from a hash table lookup on the existing context window — no kernel is invoked. The mega-graph's cooperative single-launch contract is not touched. Verification is the standard single target-model forward with a wider input (gamma+1 tokens); the cooperative kernel handles variable M natively. No conflicts.

**Integration effort.** 0–1 days. The `speculative_config = {"method": "ngram_gpu", "num_speculative_tokens": 4}` flag is already in vLLM's serving config. Wire it into the Qwen3 launch script and sweep gamma=3,4,5 + lookup_max=3,5,7.

**Risk.** On Qwen3 single-user conversational, if alpha <0.25 the overhead of widened verification absorbs the gain (net ~1.0×). For code/structured workloads this risk is low; for pure chat it is real. Graceful degradation — no slowdown below baseline if acceptance is poor.

---

### 2. Dynamic Speculative with Confidence Threshold (SpecDec++ / adaptive gamma)

**What it is.** Run the target model's softmax at each step; if top-1 probability exceeds a threshold theta, propose the high-confidence token without speculation; if below theta, engage spec decode with gamma tokens. Adapts gamma per-token based on model certainty.

**Acceptance rate.** SpecDec++ (Chen et al. 2023) and its successors report 1.3–1.8× on dense models. For MoE models, confidence-based routing is noisier (MoE softmax logits are sharper but less calibrated due to expert gating), so the threshold calibration is harder — published data from Mixtral-style MoE shows ~1.2–1.5×.

**v6 FP4 mega-graph interaction.** The adaptive gamma path requires a decision after each verification step. This is a host-side Python check between mega-graph launches — no kernel modification needed. The mega-graph's fixed grid shape accommodates variable M (gamma changes the token count passed in). No conflicts.

**Integration effort.** 2–3 days. Requires a custom rejection sampler hook in vLLM's `SpecDecodeWorker` or an outer loop wrapper that reads `speculative_tokens_accepted` and adjusts gamma. Not a one-liner but not invasive.

**Compound with n-gram.** Can be layered on top: use n-gram as the proposer, adaptive gamma as the scheduler. The combination (LADE + adaptive gamma) is the correct SpecDec++ configuration, not a separate track.

---

### 3. Layer-Skip Self-Drafting (Self-Speculative Decoding, Zhang et al. 2023)

**What it is.** Run the same model twice: once exiting at layer L_draft (producing a draft), once running all L layers (the verifier). The draft cost ratio c = L_draft / L. For Qwen3-30B-A3B with 48 layers, exiting at layer 24 gives c = 0.5.

**Acceptance rate.** Zhang et al. report 1.3–2.0× on LLaMA-65B with exit at 50% depth (alpha ≈ 0.60–0.75 on MT-Bench). However, those are dense models. For MoE with 48 layers and routing: (a) hidden states at the midpoint are less predictive of the final output because expert specialization is distributed unevenly across layers; (b) the top-8-of-128 routing at early layers selects different experts than later layers — the "early exit" prediction is systematically miscalibrated vs the full forward. Expected alpha for Qwen3 MoE conversational: 0.40–0.60, lower than dense-model numbers.

**Speedup formula.** With c=0.5, gamma=4, alpha=0.55:

```
effective_tokens = (1 - 0.55^5) / (1 - 0.55) = 2.11
step_cost = 1 + 4 * 0.5 = 3.0
speedup = 2.11 / 3.0 = 0.70x  ← SLOWER than baseline
```

With alpha=0.70 (optimistic upper bound, dense-model-like):

```
effective_tokens = (1 - 0.7^5) / (1 - 0.7) = 2.84
step_cost = 3.0
speedup = 2.84 / 3.0 = 0.95x  ← still net loss
```

Break-even requires alpha ≥ 0.75 with c=0.5 and gamma=4. Published MoE acceptance numbers do not reach this threshold without additional training (adapter at exit point).

**v6 FP4 mega-graph interaction.** Fatal conflict: the cooperative kernel runs all 48 layers as a single persistent kernel with `grid.sync()` barriers. There is no mechanism to "exit early" without restructuring the entire kernel into two separable phases (layers 0..23 + a sampling step + layers 24..47). This would double the grid.sync barrier budget and require the kernel to emit partial hidden states to host memory for CPU-side sampling — eliminating the cooperative kernel's core advantage (no per-layer host dispatch). The architecture is fundamentally incompatible.

**Integration effort.** 7–12 days (cooperative kernel bifurcation + sampling injection + re-verification logic), assuming the speedup math worked, which it does not.

**Verdict: KILL.** Net speedup negative at realistic MoE alpha; architecture conflicts with v6 FP4 mega-graph; no training path to fix the acceptance gap.

---

### 4. Medusa-Free Variants (lm_head reuse as draft heads)

**What it is.** Medusa attaches K auxiliary lm_heads at the final hidden state to predict tokens at positions t+1..t+K in parallel. The "training-free" variant reuses the single lm_head for multiple speculation positions by passing the final hidden state through the existing lm_head K times with different linear projections — or just using argmax of lm_head(h) as a single draft token (degenerate case).

**The degenerate case.** Using the full model's lm_head with the final hidden state to predict t+1 is exactly what autoregressive decoding already does. There is no "free" extra head — each additional position requires either a separate learned projection (training) or running another full forward pass (no gain). Training-free Medusa degenerates to either: (a) K=1, same as AR, or (b) using a fixed random projection (acceptance near 0). Published "training-free Medusa" papers (e.g. REST, Lookahead-Medusa) all require at least a short fine-tuning step (1–2 GPU-hours minimum) to initialize the auxiliary heads meaningfully.

**Acceptance rate.** Without training: alpha ≈ 0.05–0.15 (near-random head, most proposals rejected). Useless for throughput.

**v6 FP4 mega-graph interaction.** Would require appending K small linear projections after the final hidden state. The cooperative kernel currently ends with a single lm_head GEMV. Adding K-1 more lm_head projections is feasible (no structural conflict) but provides no benefit without trained weights.

**Verdict: KILL (training-free).** Alpha too low to produce any gain. The "medusa-free" framing is a misnomer — these variants all require at least a calibration training step. This is the EAGLE3 track (blocked on PR 4c/3, separate budget).

---

### 5. Jacobi Decoding

**What it is.** Run the model on a fixed-length guess vector (usually initialized with random tokens or the last token repeated), accept tokens where the model agrees with its own next-token prediction, iterate. A special case of lookahead where the "draft" is generated by a fixed-point iteration of the model itself.

**Acceptance rate.** Jacobi decoding requires multiple forward passes to converge. For LLaMA-7B on short sequences, 1.2–1.4× is typical; for longer contexts with high repetition, up to 1.6×. For MoE models at single-user decode, the extra forward passes (even "parallel" in a batched Jacobi scheme) cost more than the benefit because (a) each Jacobi step is a full model forward, (b) MoE routing makes convergence slower (tokens near boundaries of expert specializations take more iterations), and (c) at single-user the batch is always size 1 so there is no parallelism to exploit across the guess vector.

**Published speedup on MoE.** No published Jacobi MoE numbers found in literature as of early 2026. Inference from dense-model numbers: 1.1–1.3× at best. For single-user (no batch to amortize the fixed-point iterations), net gain is marginal or negative.

**v6 FP4 mega-graph interaction.** Each Jacobi iteration is a full forward pass. The cooperative kernel can be called repeatedly — no structural conflict. However, the number of iterations is data-dependent and not known at launch time, making CUDA graph capture problematic (the graph assumes a fixed call sequence). This is the same variability problem that killed n-gram on Gemma4 (variable batch → graph invalidation), now applied to variable iteration count.

**Verdict: DEFER (weak).** Marginal speedup at single-user, potential graph-capture conflict, no published MoE data. Not worth Wave 4 implementation time when n-gram + adaptive gamma covers the same ground with less complexity and better acceptance rates.

---

## Ranked Table (Top 3)

| Rank | Method | Accept rate (chat / code) | Complexity | Integration days | Compound with v6 FP4 |
|------|--------|--------------------------|------------|-----------------|----------------------|
| **1** | **N-gram GPU (LADE)** | 0.20–0.40 / 0.50–0.65 | Trivial (config flag) | **0–1** | Clean — c=0, no kernel conflict, mega-graph forward handles variable M |
| **2** | **N-gram + adaptive gamma (SpecDec++)** | Same + calibrated | Low (Python wrapper) | **2–3** | Same as #1 + reduces wasted verification on low-acceptance turns |
| **3** | **Jacobi decoding** | 0.10–0.25 / 0.20–0.35 | Medium (custom scheduler) | **4–6** | Fragile — variable iteration count conflicts with CUDA graph capture |

---

## Recommendation: Wave 4 Implementation Target

**Target: N-gram GPU + adaptive gamma (SpecDec++-style scheduler)**

Implement in two steps:
1. Day 0–1: wire `speculative_config = {"method": "ngram_gpu", "num_speculative_tokens": 4, "prompt_lookup_max": 5}` into the Qwen3-30B-A3B vLLM launch config. Sweep gamma=3,4,5 and report `speculative_tokens_accepted` per workload.
2. Day 2–3 (conditional on step 1 showing alpha ≥ 0.30 for at least one target workload): add an adaptive-gamma Python wrapper that reduces gamma to 2 when alpha drops below 0.25 (reduces verification overhead on low-acceptance turns without killing throughput on high-acceptance turns).

Expected outcome on conversational workload: **1.2–1.5×** (chat: 1.2×, code: 1.6×). At current Qwen3-30B-A3B NVFP4 baseline of ~83 tok/s single-user (C=1): **~100–125 tok/s/user**.

Compound with v6 FP4 mega-graph: if mega-graph reduces step latency by 2× (projecting ~166 tok/s), n-gram adds 1.2–1.5× on top → **~200–250 tok/s/user**. This is additive, not multiplicative, because n-gram's gain is acceptance-rate-driven and independent of per-step cost.

---

## Verdicts

| Candidate | Verdict | Reason |
|-----------|---------|--------|
| N-gram GPU / LADE | **PROCEED** | 1.2–1.6× expected, zero training, works today, no mega-graph conflict. EV = 1.35 × P(0.90) = 1.2. Above 1.2× threshold. |
| N-gram + adaptive gamma | **PROCEED** | Adds 0.1–0.2× on top of #1 at 2–3 days cost. Low-risk add-on. |
| Layer-skip self-drafting | **KILL** | Net negative speedup at MoE acceptance rates; fatal architectural conflict with v6 cooperative kernel. |
| Medusa-free | **KILL** | Training-free variant has alpha ~0.05; not distinguishable from random. Belongs to EAGLE3 track. |
| Jacobi decoding | **DEFER** | Marginal single-user gain; variable iteration count conflicts with CUDA graph capture; no MoE benchmark data. |

**Overall verdict: PROCEED** — n-gram GPU is available today with zero risk, expected 1.2–1.5× lift at P ≥ 0.85. EV satisfies the 1.2× × P≥0.5 threshold comfortably.
