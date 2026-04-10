# Speculative Decoding Feasibility: Gemma4 26B MoE on RTX 5090

**Date:** 2026-04-09  
**Model:** gemma-4-26B-A4B-it-NVFP4-modelopt  
**Baseline:** 127 tok/s single-user decode (CUDA graphs enabled)  
**Hardware:** RTX 5090, 32 GB VRAM  

---

## 1. Does Gemma4 Support Native MTP?

**No.** The `config.json` (`text_config` section) contains zero MTP-related keys. There is no `mtp_*`, `num_draft_tokens`, or `predict_*` field. Gemma4 26B has no built-in multi-token prediction heads, unlike DeepSeek-V3, Qwen3.5, or GLM-4 which have dedicated MTP modules registered in vLLM's `MTPModelTypes` list.

vLLM 0.19.0's supported MTP architectures are: DeepSeekMTP, QwenMTP variants, MiMoMTP, GLM4MoeMTP, ErnieMTP, NemotronH, ExaoneMoeMTP, LongCatFlash, Step3p5. **Gemma4 is absent.**

---

## 2. Architecture Summary

| Parameter | Value |
|-----------|-------|
| Layers | 30 |
| Attention types | 25 sliding (window=1024) + 5 full |
| Hidden size | 2816 |
| Attention heads | 16Q / 8KV |
| Head dim | 256 (local) / 512 (global full-attn) |
| MoE experts | 128 total, top-8 per token |
| MoE intermediate | 704 |
| Vocab size | 262,144 |
| KV per token | 240 KB (BF16, 30 layers × 2 × 8 heads × 256 dim) |
| NVFP4 model VRAM | ~14.8 GB (MoE in NVFP4, attn in BF16) |

**Active compute per decode token:** top-8 experts fire, so 8 × 704 × 2816 × 3 = 47.6M MoE params per layer, ×30 = 1.43B active weights. Attention is 30 × ~23M = 690M. Total active per token: ~2.1B weights read from VRAM. This is the bottleneck for any draft model comparison.

---

## 3. vLLM Spec Decode Support for Gemma4

### Multimodal Restriction

Gemma4 is `Gemma4ForConditionalGeneration` — a multimodal architecture. vLLM's `SpecDecodeBaseProposer` (parent of `DraftModelProposer` and `EagleProposer`) raises `NotImplementedError` when `supports_mm_inputs=True`:

```python
def _raise_if_multimodal(self):
    if self.supports_mm_inputs:
        raise NotImplementedError(
            "Speculative Decoding with draft models or parallel drafting "
            "does not support multimodal models yet"
        )
```

**Workaround:** `language_model_only=True` in the LLM constructor sets `MultiModalConfig.language_model_only=True`, which causes `get_limit_per_prompt()` to return 0 for all modalities, which causes `supports_multimodal_inputs()` to return `False`. This bypasses the check. Draft model spec decode with `language_model_only=True` **should work** — but it is untested for Gemma4 specifically.

**Ngram is unrestricted.** `NgramProposer` and `NgramProposerGPU` are standalone classes that do not inherit from `SpecDecodeBaseProposer` and contain no multimodal check. They work with any model including Gemma4, regardless of `language_model_only`.

### Supported Methods in vLLM 0.19.0

| Method | Status for Gemma4 |
|--------|-------------------|
| `ngram` | Works today — no model required, no multimodal restriction |
| `ngram_gpu` | Works today — GPU-accelerated variant |
| `draft_model` | Needs `language_model_only=True` + vocab-matched draft; untested |
| `eagle` / `eagle3` | No Gemma4 Eagle head exists; would need custom training |
| `medusa` | No Gemma4 Medusa head exists |
| `mtp` (native) | Not supported — Gemma4 has no MTP module |

---

## 4. Draft Model Candidates

### Option A: Gemma4 26B Pruned 25L as Draft Model

**Files available:** `/root/models/gemma-4-26B-pruned-25L-rm2_4_6_8_20` (15 GB, NVFP4), `/root/models/gemma-4-26B-pruned-27L-rm2_4_8` (16 GB, NVFP4)

**Vocab match:** 262,144 — passes the `verify_equal_vocab_size_if_draft_model` check.

**VRAM budget:**
- Target (30L): ~14.8 GB
- Draft (25L, NVFP4): ~12.3 GB  
- Combined + 3 GB overhead: ~30.1 GB  
- KV cache headroom: ~1.9 GB (~8K tokens max)

**Compute cost ratio:** The draft has 25 layers vs target's 30 layers (c = 25/30 = 0.83). For memory-bandwidth-bound MoE decode, this is a near-linear reduction. Active MoE compute per token is **identical** (top-8 out of 128 experts, same sizes), so the savings come only from 5 fewer attention+MoE passes.

**Theoretical speedup** (gamma=4 draft tokens, alpha=0.65 acceptance, c=0.83):

```
effective tokens/step = (1 - 0.65^5) / (1 - 0.65) = 2.4
step cost factor      = 1 + 4 × 0.83 = 4.32
speedup               = 2.4 / 4.32 = 0.56x  ← SLOWER
```

**Verdict:** Pruned-layer draft models are not viable for this architecture. The draft cost is too close to the target cost (c close to 1), and each speculative step incurs a net slowdown. The speedup formula requires c << 1 to gain; with c=0.83 and realistic acceptance rates, spec decode makes inference ~40% slower.

### Option B: Gemma4 26B Expert-Pruned 32-Expert as Draft Model

**File:** `/root/models/gemma4-pruned-32exp` (7.9 GB, NVFP4)

**VRAM budget:**
- Target (30L, 128 exp): ~14.8 GB
- Draft (30L, 32 exp, NVFP4): ~6.3 GB
- Combined + 3 GB overhead: ~24.1 GB
- KV cache headroom: ~7.9 GB (~33K tokens)

**Compute cost problem:** Both models select top-8 experts per token. The draft with 32 experts still forwards through exactly 8 expert FFNs per layer — same as the 128-expert target. The only reduction is in the gating computation (32 vs 128 scores) and the weight loading for the router. Expert forward passes are **identical** in cost. Effective c ≈ 0.90.

**MoE routing divergence:** A 32-expert model was trained with different routing distributions. At top-8 out of 32, each expert fires 25% of the time (vs 6.25% for 128-expert target). Proposed tokens will frequently route through different experts than the target model expects, leading to high rejection rates (alpha ≈ 0.40–0.55 expected).

**Theoretical speedup** (gamma=4, alpha=0.50, c=0.90):

```
effective tokens/step = (1 - 0.5^5) / (1 - 0.5) = 1.94
step cost factor      = 1 + 4 × 0.90 = 4.60
speedup               = 1.94 / 4.60 = 0.42x  ← MUCH SLOWER
```

**Verdict:** Expert-pruned draft is worse than layer-pruned and is not viable.

### Option C: True Small Draft (Gemma4 2B or Similar)

No Gemma4 2B model exists (Google has not released one as of April 2026). Any Gemma4 draft model would need to match the 262,144-token vocabulary. Using a non-Gemma4 model (e.g., Gemma 2B at 256K vocab) fails the vocab size check.

**If such a model existed:** A true 2B parameter model with the same hidden size (2816) and matching vocab would have c ≈ 0.10–0.15. At alpha=0.70, gamma=4:
```
effective tokens/step = (1 - 0.7^5) / 0.3 = 2.84
step cost factor      = 1 + 4 × 0.12 = 1.48
speedup               = 2.84 / 1.48 = 1.92x → ~244 tok/s
```
This would be compelling. The blocker is the model does not exist.

### Option D: N-gram Speculative Decoding (Recommended)

N-gram proposes tokens by matching suffix patterns from the existing context — no draft model needed. Works on any model. In vLLM, uses `NgramProposer` (CPU, numba-accelerated) or `NgramProposerGPU` (Triton-compiled).

**Zero VRAM overhead.** All 32 GB is available for target model + KV cache.

**No multimodal restriction.** Confirmed by source inspection.

**Configuration:**
```python
llm = LLM(
    model="/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt",
    language_model_only=True,
    speculative_config={
        "method": "ngram_gpu",   # or "ngram" for CPU version
        "num_speculative_tokens": 4,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 5,
    },
    ...
)
```

**Workload-dependent acceptance rates and speedup (gamma=4):**

| Workload | alpha | Effective tok/step | Speedup | Projected tok/s |
|----------|-------|--------------------|---------|-----------------|
| Structured output (JSON/CSV) | 0.65 | 2.5 | 2.5x | 321 |
| Code generation | 0.55 | 2.1 | 2.1x | 268 |
| Code completion | 0.45 | 1.8 | 1.8x | 227 |
| Summarization | 0.35 | 1.5 | 1.5x | 194 |
| General chat | 0.20 | 1.2 | 1.2x | 159 |

Note: n-gram speedup formula simplifies to `effective_toks / 1.0` because the proposer has zero model cost (c=0). This is unusually favorable vs draft model methods.

---

## 5. MoE-Specific Challenges for Speculative Decoding

### Routing Consistency Problem

For any draft-model-based approach, the core challenge with MoE is that different models (or even the same model with fewer experts) route tokens through different experts. This means draft token hidden states are computed with different expert activations than the target model would use, causing systematic quality degradation. The target model must reject tokens whose logit distributions deviate from what it would produce.

In Gemma4 26B, top-8 of 128 experts are selected per layer per token. A pruned draft with 25 layers already commits to routing decisions the target may override. The mismatch accumulates over the 4 speculative tokens, so acceptance rate degrades with each additional speculative token.

**N-gram avoids this entirely:** proposals are pure token sequence patterns; the full target model evaluates each token independently.

### KV Cache Duplication

For draft-model spec decode, vLLM allocates KV cache entries for both the draft and target models. The 25L pruned draft has 25 × 2 × 8 × 256 = ~100KB per token (vs 240KB for target). Combined: ~340KB per token. At 7.9 GB available (25L draft scenario), maximum context is ~23K tokens — adequate but leaves less headroom than pure ngram.

### CUDA Graphs Compatibility

CUDA graphs are currently working at 127 tok/s single-user. Adding a draft model disables CUDA graphs for the draft forward passes in some vLLM configurations. The ngram proposer has no forward pass, so CUDA graphs for the target model remain fully enabled.

---

## 6. Self-Speculative Decoding (Layer Skipping)

True self-speculative decoding (where the same model skips some layers to produce draft tokens) is not supported by vLLM natively for Gemma4. It would require:

1. A custom vLLM model class that implements layer-skip draft proposals
2. Integration with the spec decode scheduler
3. Re-training or careful calibration of which layers to skip

Our pruned models (`rm2_4_8`, `rm2_4_6_8_20`) were produced by layer removal during fine-tuning, not layer skipping during inference. They function as separate models (separate weight files), not as a self-speculative mechanism.

**The computation cost for self-speculative is identical to using the pruned model as a draft** (c = n_skipped/30), so the speedup analysis in Option A applies. Given c ≈ 0.83 for skipping 5 layers, self-speculative is also not viable.

---

## 7. Recommended Approach

**Primary recommendation: N-gram GPU spec decode**

Rationale:
1. Works today with zero code changes beyond `speculative_config`
2. No VRAM penalty — full KV cache budget preserved
3. No multimodal restriction (NgramProposerGPU does not inherit SpecDecodeBaseProposer)
4. CUDA graphs for target model remain enabled
5. 1.5x–2.5x speedup for structured/code workloads
6. For general chat: still 1.2x at worst, negligible cost if acceptance is low

**Secondary recommendation: Wait for Gemma4-family small model**

If Google releases a Gemma4 2B or 4B model with the same tokenizer (vocab=262,144), the `draft_model` method with `language_model_only=True` becomes viable and would deliver ~1.9x speedup at alpha=0.70. The VRAM math works: 2B BF16 ≈ 4 GB + 14.8 GB target = 18.8 GB, leaving 10+ GB for KV cache.

**Not recommended:**
- Pruned 25L/27L/29L as draft: c too high (0.83+), net slowdown
- Expert-pruned 32-exp as draft: same active compute, low acceptance
- Eagle/Medusa: no trained heads exist for Gemma4
- Native MTP: not in architecture

---

## 8. Implementation Plan for N-gram

**Effort:** Low (2–4 hours including testing)

**Steps:**
1. Add `speculative_config` to existing LLM launch configuration
2. Sweep `num_speculative_tokens` (2, 4, 6) and `prompt_lookup_max` (3, 5, 7)
3. Benchmark on target workloads (code generation is best case)
4. Check that acceptance metrics are reported in vLLM logs (`speculative_tokens_accepted`)

**Risk:** Near zero. If acceptance is poor for a given workload, vLLM falls back to standard decode. No correctness risk (spec decode is mathematically equivalent to standard decode when using exact rejection sampling).

**Expected result:** ~200–270 tok/s for code/structured workloads, ~160 tok/s for chat.

---

## 9. Throughput Summary

| Approach | VRAM overhead | Speedup | Projected tok/s | Feasible today |
|----------|--------------|---------|-----------------|----------------|
| Baseline (no spec decode) | 0 GB | 1.0x | 127 | Yes |
| N-gram (code/structured) | 0 GB | 1.5–2.5x | 190–320 | Yes |
| N-gram (general chat) | 0 GB | 1.2x | 152 | Yes |
| 25L pruned draft | ~12 GB | ~0.56x | ~71 | No — net slowdown |
| 32-expert draft | ~6 GB | ~0.42x | ~53 | No — net slowdown |
| Gemma4 2B draft (hypothetical) | ~4 GB | ~1.9x | ~241 | No — model doesn't exist |

---

## 10. Key Blockers and Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| No native MTP in Gemma4 | High | Use ngram instead |
| All available drafts too expensive (c close to 1) | High | Confirmed by compute analysis |
| Draft model spec decode requires `language_model_only=True` | Medium | Known workaround, untested for Gemma4 |
| N-gram acceptance low for diverse chat workloads | Low | Graceful degradation, no slowdown below baseline |
| CUDA graph compatibility with draft models | Medium | Ngram avoids this entirely |
| Gemma4 2B never released | Medium | Would unlock 1.9x speedup — worth monitoring |
