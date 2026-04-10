# Expert Activation Analysis: Gemma4 26B-A4B MoE

**Model:** google/gemma-4-26B-A4B-it (NVFP4 quantized via ModelOpt)
**Architecture:** 30 MoE layers, 128 experts/layer, top-8 routing, hidden_size=2816, moe_intermediate_size=704
**Date:** 2026-04-09

## Methodology

Three complementary analyses were performed:

1. **Simulated routing with random hidden states** (5 seeds x 10,000 tokens) -- baseline uniformity check
2. **Simulated routing with real token embeddings** (10,000 sampled vocabulary embeddings through actual router weights) -- primary analysis
3. **550 diverse prompts through running vLLM server** (30,270 total tokens processed) -- confirms model is serving correctly

The router implements: `logits = (hidden * scale) @ W^T * per_expert_scale`, then selects top-8 experts per token with softmax weighting over selected experts.

**Important caveat:** This analysis uses the embedding layer as a proxy for hidden states at each layer. In practice, hidden states evolve through attention and MLP blocks before reaching each layer's router. The actual activation patterns during inference will differ, particularly in deeper layers. The results should be interpreted as a structural analysis of routing capacity, not a recording of live routing decisions.

## Key Findings

### 1. Activation Distribution is Moderately Skewed (NOT Uniform)

With real token embeddings, the routing shows meaningful skew:

| Metric | Value | Uniform Baseline |
|--------|-------|-----------------|
| Gini coefficient | 0.30 - 0.66 | 0.0 |
| Entropy | 5.85 - 6.80 bits | 7.0 bits |
| Max/uniform ratio | 2.9x - 10.5x | 1.0x |
| Top-32 coverage | 44-74% of activations | 25% |

**Layer 0 is the most skewed** (Gini=0.66, entropy=5.85). Expert 17 in layer 0 captures 8.2% of all activations (10.5x the uniform rate). The later layers (26-29) also show elevated skew (Gini 0.45-0.54).

**Middle layers (8-20) are most balanced** (Gini 0.29-0.34), suggesting the model distributes routing more evenly for intermediate representations.

### 2. No Experts Are Completely Dead

- **Zero-activation experts:** 0 across all 30 layers (with 10,000 token sample)
- **Near-zero (<10% of uniform rate):** 73 out of 3,840 expert-layer slots (1.9%)
  - Concentrated in layer 0 (40 experts) and layer 29 (13 experts)
  - Middle layers have essentially no near-zero experts

### 3. No Experts Are Consistently Hot or Cold Across Layers

- **Consistently cold (bottom 25% in 20+/30 layers):** 0 experts
- **Consistently hot (top 25% in 20+/30 layers):** 0 experts
- **Cross-layer Jaccard similarity (consecutive layers):** 0.14 mean (very low)

Each layer maintains its own independent routing pattern. This means pruning must be done per-layer, not globally -- there is no "universally useless" expert.

### 4. Hot Set Coverage

| Hot Set Size | Avg Coverage | Min | Max |
|-------------|-------------|-----|-----|
| Top 16 (12.5%) | 31.6% | 25.8% | 49.6% |
| Top 32 (25%) | 50.4% | 44.2% | 74.3% |
| Top 48 (37.5%) | 64.5% | 57.6% | 87.9% |
| Top 64 (50%) | 75.8% | 69.6% | 95.5% |
| Top 96 (75%) | 91.7% | 89.0% | 99.5% |

The top-64 experts handle ~76% of routing on average. The bottom 32 experts contribute only ~8.3% of activations.

### 5. Expert Weight Similarity (Merging Candidates)

**Router vector similarity:**
- Mean cosine similarity between expert routing vectors: 0.02 - 0.17 (very low -- experts are well-separated in routing space)
- Max similarity: 0.83 (layer 29, E18-E81)
- Zero pairs with similarity > 0.9 (except 2 pairs in layer 29)
- Only 7 pairs across all layers exceed 0.8

**Weight fingerprint similarity (via NVFP4 scale tensors):**
- Mean similarity: 0.90 - 0.93 (HIGH -- all experts have similar magnitude structure due to shared initialization/training)
- This is a known artifact of MoE training: experts share a common "backbone" and specialize through small perturbations
- The high weight similarity does NOT mean experts are interchangeable -- routing vectors differentiate them

**Top merge candidates (combined router + weight similarity):**

| Layer | Expert A | Expert B | Combined Score | Router Sim | Weight Sim |
|-------|----------|----------|---------------|------------|------------|
| 29 | 18 | 81 | 0.876 | 0.831 | 0.924 |
| 29 | 78 | 111 | 0.874 | 0.817 | 0.934 |
| 26 | 98 | 110 | 0.860 | 0.803 | 0.920 |
| 24 | 1 | 106 | 0.836 | 0.752 | 0.930 |
| 29 | 22 | 49 | 0.831 | 0.741 | 0.932 |
| 27 | 73 | 102 | 0.826 | 0.725 | 0.941 |
| 1 | 12 | 100 | 0.825 | 0.726 | 0.937 |

Layer 29 has the most merge candidates, consistent with it having the most skewed routing.

### 6. Pruning Impact Estimates

| Prune Bottom | Experts Removed/Layer | Memory Saved | Routing Loss |
|-------------|----------------------|--------------|-------------|
| 10% | 12 | 1,021 MB (9%) | 2.1% |
| 20% | 25 | 2,127 MB (20%) | 5.8% |
| 30% | 38 | 3,233 MB (30%) | 10.7% |
| 50% | 64 | 5,445 MB (50%) | 24.2% |

**Per-expert NVFP4 size:** ~2.84 MB (gate_proj + up_proj + down_proj)
**Total expert memory:** ~10,890 MB (10.6 GB) across all 30 layers

### 7. Router Weight Norm vs Activation Frequency

Correlation between router weight L2 norm and activation frequency: **-0.007** (essentially zero). Router norm does not predict which experts get activated -- the selection depends on the interaction between hidden states and routing vectors, not the magnitude of the routing vectors alone.

## Recommendations

### For Pruning

1. **Layer 0 is the best target:** 40 experts have <10% of uniform activation rate. Pruning 12-25 of the coldest experts in layer 0 would save memory with minimal routing loss.

2. **Layers 26-29 are secondary targets:** Higher skew means more prunable experts, but these are deeper layers where errors compound.

3. **Middle layers are poor pruning targets:** Layers 8-20 have near-uniform routing. Pruning here would cause proportionally more disruption.

4. **Conservative recommendation:** Prune bottom 10% per layer (12 experts) for ~1 GB memory savings with only 2.1% routing loss. This is likely recoverable with minimal fine-tuning.

5. **Aggressive recommendation:** Prune bottom 20% (25 experts) for ~2.1 GB savings. The 5.8% routing loss would require fine-tuning to recover quality.

### For Merging

1. **Layer 29 has the strongest merge candidates** (6+ pairs with combined score > 0.80). Merging E18+E81 and E78+E111 would be the lowest-risk first experiments.

2. **Merge strategy:** Average the weights of merged experts, sum their routing logits (or take the max). Fine-tune the router for 100-1000 steps to adapt.

3. **Expected gain from merging top-5 pairs per layer:** ~150 experts removed globally, saving ~426 MB with minimal quality impact (both router and weight similarity are high for these pairs).

### For Serving Optimization (Without Model Changes)

1. **Expert offloading:** The bottom 32 experts per layer (handling only ~8.3% of activations) could be offloaded to CPU/slower memory and fetched on-demand. This would reduce GPU memory by ~2.7 GB with a throughput impact only on the 8.3% of tokens that route to those experts.

2. **Expert caching:** With top-64 handling 76% of activations, a GPU cache of 64 experts per layer would achieve a 76% hit rate. Combined with async prefetch, this could enable serving with only 50% of experts in GPU memory.

3. **Per-layer expert budgets:** Layer 0 could run with just 48 experts (covering 88% of activations), while middle layers need all 128.

## Files

| File | Description |
|------|-------------|
| `expert_activation.py` | Router simulation with random + diverse hidden states |
| `expert_similarity.py` | Pairwise cosine similarity (router vectors + weight fingerprints) |
| `expert_activation_results.json` | Full results from random hidden state simulation |
| `expert_activation_real_results.json` | Full results from real token embedding simulation |
| `expert_similarity_results.json` | Full similarity matrices and merge candidates |
| `inference_token_stats.json` | Token counts from 550 live inference prompts |
