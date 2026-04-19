# Multi-Tenant Speculative Serving: Shared Draft Model Feasibility

**Tag:** W2_5d_multitenant_spec_feasibility  
**Date:** 2026-04-18  
**Hardware:** 2× RTX PRO 6000 (96 GB each, Blackwell SM120)  
**Banked baseline:** 12,229 tok/s Gemma4 26B (GPU 0) + 17,426 tok/s Qwen3.6-35B (GPU 1)  
**Hypothesis:** one small draft model (Qwen3-1.7B or Qwen3-8B) speculates for both targets simultaneously, amortizing draft forward cost.

---

## 1. Tokenizer Compatibility (Hard Blocker)

| Model | Tokenizer family | Vocab size |
|---|---|---|
| Qwen3-30B / Qwen3.6-35B | Qwen BPE | 151,669 |
| Gemma4-26B | Gemma SentencePiece | 262,144 |
| Qwen3-1.7B (candidate draft) | Qwen BPE | 151,669 |

**Mismatch is fundamental, not incidental.** vLLM's `verify_equal_vocab_size_if_draft_model` check will reject a Qwen3 draft paired against a Gemma4 target at construction time. The check exists because spec decode requires both models to assign probabilities over an identical token ID space: the draft proposes IDs 0–151,668 but Gemma4's softmax covers 0–262,143. There is no legal mapping between the two distributions without retokenization.

**Cross-vocabulary research exists** (OmniDraft, NeurIPS 2025; transplant-vocab; SpecVocab) but none is integrated into vLLM. OmniDraft's n-gram cache approach achieves 1.5–2× speedup with a Llama-68M draft across Vicuna/Qwen/Llama3 targets — but only across models that share approximately the same token space. The Qwen↔Gemma delta (151K vs 262K, entirely different byte-pair rules) is a harder class of mismatch than the padding deltas within the Qwen family that vLLM PR #13849 already handled (≤128 token delta, same underlying BPE).

**Retokenization path:** Draft produces a Qwen token → map to surface string → retokenize as Gemma4 token → verify. This is sound in theory but adds a CPU retokenization round-trip per speculative token, per request, in the hot path. At 4 speculative tokens and even 10 µs per retokenization call, that is 40 µs per step — the Gemma4 26B decode step on PRO 6000 is roughly 8 ms, so retokenization overhead is negligible in absolute terms. However the semantic mapping is lossy: one Qwen token may span multiple Gemma4 tokens (or vice versa), making one-to-one ID substitution incorrect. The draft accepts or rejects must be computed in Gemma4 token space, meaning the draft must propose in Gemma4 token space — which it cannot do natively. Partial-word boundaries propagate into mis-aligned probability distributions, collapsing acceptance rate toward zero.

**Estimated acceptance rate (cross-tokenizer, no fine-tuning):** < 0.25 (literature on cross-family, incompatible-tokenizer pairs; same-family Qwen3-0.6B↔Qwen3-7B achieves ~0.60–0.70 with matched tokenizers).

**Conclusion:** A Qwen draft cannot usefully serve Gemma4 without either (a) a trained cross-vocabulary adapter or (b) a completely separate Gemma4-family draft. Option (b) reverts to independent spec decode with no amortization. Option (a) does not exist in any shipping codebase as of April 2026.

---

## 2. vLLM Architecture Support

vLLM's speculative decode (v0.x and V1) is a **per-engine feature**. The current dual-model setup runs two entirely separate Docker containers (vllm-gemma4, vllm-qwen3) with no shared memory, shared KV cache, or shared draft model. There is no vLLM API or configuration for a single draft engine to service two separate target engines.

The V1 engine architecture (`EngineCore` + isolated `EngineCore` loop) makes cross-engine draft sharing harder, not easier: each EngineCore has its own scheduler, KV cache manager, and model executor. Draft proposals are generated inside the target engine's forward pass scheduler, not as an external service.

**What would be needed for shared-draft:**
1. A draft-model-as-service subprocess (IPC) that both target engines query.
2. Batching of draft requests from two engines with potentially different step cadences and batch sizes — requires a new synchronization protocol.
3. KV cache for the draft model shared or duplicated across two engines — complex memory management.
4. vLLM core changes: the `DraftModelProposer` is tightly coupled to one `ModelRunner` instance.

**Effort estimate:** 10–20 engineering days for a prototype; upstream PR acceptance is unlikely without broader community interest. This is a fork.

**EAGLE3 note:** EAGLE3 (available today for Qwen3-30B via the trained head at `eagle3_qwen3_30b_training.md`) is engine-internal by design — its speculation head is bolted onto the target model's own hidden states. It is not shareable across model families by definition.

---

## 3. GPU Layout and Draft Placement

With both GPUs fully occupied (GPU 0: Gemma4 96 GB, GPU 1: Qwen3.6 96 GB), draft placement options are:

| Option | VRAM impact | Bandwidth | Verdict |
|---|---|---|---|
| Draft on GPU 0 (co-resident with Gemma4) | Qwen3-1.7B FP8 ≈ 1.7 GB; leaves 94.3 GB for Gemma4 + KV | No cross-GPU hop for Gemma4 target | Only works for Qwen→Gemma pairing (blocked by tokenizer) |
| Draft on GPU 1 (co-resident with Qwen3.6) | Same 1.7 GB; leaves 94.3 GB | No hop for Qwen target | Works for same-family Qwen draft, but no sharing to GPU 0 |
| Draft on CPU (system RAM) | 0 VRAM | NVLink/PCIe hop per step ≈ 2–5 ms | Adds ~25–60% latency per step; unacceptable |
| Separate third GPU (hypothetical) | N/A — not available | NVLink hop | N/A |

**Bottom line:** even if tokenizers matched, the two-GPU topology offers no free slot for a truly shared draft. Intra-GPU co-residence does not help the opposite target, and cross-GPU draft traffic is prohibitively expensive on PCIe (no NVLink between consumer/workstation Blackwell SKUs in this build).

---

## 4. Throughput Projection

### Scenario A: Independent spec decode (realistic ceiling)

Each engine runs its own same-family draft:
- GPU 0: Gemma4 26B + Gemma4 E2B draft (262K vocab match). E2B at 400–600 tok/s is fast enough (c ≈ 0.08). With alpha = 0.65, gamma = 4: speedup = 2.4/(1 + 4×0.08) = **1.8×**. Gemma4 throughput: 12,229 × 1.8 ≈ **22,000 tok/s**.
- GPU 1: Qwen3.6-35B + Qwen3-1.7B draft (151K vocab match). Alpha ≈ 0.68, c ≈ 0.05, gamma = 4: speedup ≈ 2.5/(1 + 4×0.05) = **2.1×**. Qwen3.6 throughput: 17,426 × 2.1 ≈ **36,600 tok/s**.
- Combined: **~58,600 tok/s** vs 29,655 tok/s banked baseline → **+98% aggregate**.

### Scenario B: Shared single Qwen draft for both (the hypothesis)

Cross-tokenizer acceptance rate for Gemma4 ≈ 0.15–0.25. At alpha = 0.20, gamma = 4, c = 0.05:
```
effective tokens/step = (1 - 0.2^5) / 0.8 = 1.25
step cost factor      = 1 + 4 × 0.05 = 1.20
speedup               = 1.25 / 1.20 = 1.04×
```
Draft cost overhead essentially cancels the marginal accepted tokens. Net gain over baseline: near zero for Gemma4. Qwen3.6 side retains full ~2.1× (same tokenizer). **Combined: ~37,000 tok/s** — worse than Scenario A and only marginally better than baseline.

### Scenario C: Shared draft with a cross-vocab adapter (research track)

If an OmniDraft-style adapter is trained for Qwen3-1.7B → Gemma4 token mapping, literature suggests alpha could reach 0.45–0.55 for code/structured tasks. At alpha = 0.50:
```
speedup = (1 - 0.5^5) / (0.5 × 1.20) = 1.94 / 0.60 = 1.61×
```
Combined: Gemma4 ~19,700 + Qwen3.6 ~36,600 = **~56,300 tok/s**. Still marginally worse than Scenario A (independent spec decode with proper same-family drafts), with a 20–30 day training investment to build the adapter.

---

## 5. Summary Table

| Dimension | Finding |
|---|---|
| Qwen↔Gemma tokenizer overlap | ~0% functional overlap — Qwen BPE 151K vs Gemma SentencePiece 262K; hard mismatch |
| vLLM shared-draft support | No — each engine owns its own draft; no IPC draft-as-service exists |
| Needs fork? | Yes — 10–20 days minimum; not upstreamable near-term |
| Cross-tokenizer acceptance rate | ~0.15–0.25 (estimated); spec decode provides near-zero gain for Gemma4 target |
| Best-case shared draft speedup (Gemma4) | ~1.04× (negligible) |
| Independent spec decode (realistic) | +98% aggregate (Scenario A); achievable with existing models |
| Integration days (shared draft) | 10–20 days vLLM fork + 20–30 days adapter training = 30–50 days |
| Integration days (independent spec) | 2–4 days (Gemma4 E2B already downloaded; Qwen3-1.7B available) |

---

## 6. Verdict: KILL

**Multi-tenant shared draft is not viable for the Qwen3 ↔ Gemma4 pair.** The tokenizer mismatch is not a configuration issue — it is an architectural incompatibility that collapses acceptance rate to the point where draft overhead cancels all gain. Even with a trained cross-vocabulary adapter (a 30–50 day project), the ceiling is ~56K tok/s, which is worse than the 2–4 day alternative of independent per-engine spec decode (~58K tok/s).

**Recommended path instead:**
1. **Deploy independent spec decode per engine** (Scenario A). Add Gemma4 E2B as draft on GPU 0 (`language_model_only=True`); add Qwen3-1.7B as draft on GPU 1. Expected combined throughput: ~58K tok/s (+98% vs baseline). Effort: 2–4 days.
2. **For Gemma4 specifically:** if E2B acceptance is poor (alpha < 0.50 measured), fall back to ngram_gpu spec decode (zero VRAM, zero tokenizer issue, ~1.5–2.5× on code workloads).
3. **Monitor OmniDraft integration into vLLM** — if cross-vocab spec decode ships as a first-class vLLM feature, the economics change (no fork cost). Re-evaluate in 6 months.

**The alternative framing (large model as draft for smaller)** is unambiguously worse: a 30B model as draft for a 26B target has c ≈ 1.0+, guaranteed net slowdown per the spec decode speedup formula.

---

## References

- [OmniDraft (NeurIPS 2025)](https://arxiv.org/abs/2507.02659) — cross-vocabulary n-gram cache approach, 1.5–2× speedup
- [vLLM Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — current architecture
- [vLLM PR #13849](https://github.com/vllm-project/vllm/pull/13849) — vocab size delta handling (Qwen family only, ≤128 token delta)
- [vLLM Issue #7252](https://github.com/vllm-project/vllm/issues/7252) — feature request for different-vocab draft (open, unimplemented)
- [Disparate Impacts of Speculative Decoding (arXiv 2510.02128)](https://arxiv.org/html/2510.02128v1) — acceptance rate as primary speedup driver
- [Google EAGLE3 for Gemma4 (HuggingFace blog)](https://huggingface.co/blog/lujangusface/tw-eagle3-gemma4) — 1.72× speedup with trained EAGLE3 head
- [Standalone draft support in vLLM v1 (forum)](https://discuss.vllm.ai/t/standalone-draft-model-spec-decode-support-in-v0-x-and-v1/2241) — removed in v0.x >10, not in V1
