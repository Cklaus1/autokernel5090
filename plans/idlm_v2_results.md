# I-DLM v2 Results — Phase 1 Kill Criterion Hit

**Date:** 2026-04-17  
**Hardware:** RTX PRO 6000 Blackwell (SM120a)  
**Model:** I-DLM-Qwen3-8B (gen_block_size=4, block_size=7)

## Experiment: v1 (causal) vs v2 (correct per-request mask)

### Setup
- **v1**: Original FlashInfer paged attention with `dllm_force_causal=True`. MASKs attend to each other via standard causal mask.
- **v2**: Triton masked attention for extend window (MASKs only attend to committed tokens) + FlashInfer paged for prefix KV + merge_state. Semantically correct masking.

### Results (C=1, 20 prompts)

| Metric | v1 (causal) | v2 (correct mask) |
|---|---|---|
| Mean tok/s | **138.5** | 77.6 |
| Acceptance rate | **42%** | 8.6% |
| Tokens/forward | **2.20** | 1.41 |
| Total tokens | 9,026 | 9,028 |

### Analysis

The **mask contamination theory is wrong**. MASK-to-MASK attention is not a bug — it's a feature.

When MASK tokens attend to each other via causal masking:
- Each MASK sees the predictions of preceding MASKs in the same block
- This creates an implicit chain: MASK[0] predicts token[0], MASK[1] conditions on MASK[0]'s prediction + committed context
- This is essentially **auto-regressive within the speculative block**, just done in parallel
- The model learned to leverage these peer signals during training (the training used causal masking)

When MASK tokens are isolated (v2 correct masking):
- Each MASK independently predicts from committed context only
- All 3 speculative tokens are conditionally independent
- This is "correct" in a diffusion model sense but wrong for this model's training distribution
- Acceptance rate collapses because the 3 predictions can be mutually inconsistent

### Kill Criterion

> "After Phase 1: if acceptance rate doesn't improve by ≥5%, the mask contamination theory is wrong → stop"

**Acceptance rate decreased by 33 percentage points (42% → 8.6%).** Phase 1 failed. I-DLM v2 with correct masking is abandoned.

### What This Means

1. **The v1 causal approach is correct for this model.** The model was trained with causal attention, so MASKs naturally produce coherent speculative chains under causal masking.
2. **Batch scaling requires a different approach.** Since causal masking works well but is hard to batch across different diffusion phases, the batching problem remains.
3. **Alternative: use CUDA graphs for single-user speedup** (v1 already gets 138.5 tok/s at C=1 without graphs; with graphs it should reach ~200+ tok/s).
4. **Alternative: Medusa/EAGLE3** for batch-friendly speculative decoding on the same model.

### Implications for the v2 Plan

- Phase 2 (batched verify): **cancelled** — the mask fix doesn't help, so no reason to pursue batched masking
- Phase 3 (fused classify): **cancelled**
- Phase 4 (CUDA graphs): **still valuable** but for v1, not v2
- **Recommendation**: Focus on v1 + CUDA graphs, or switch to Medusa/EAGLE3 for batch scaling
