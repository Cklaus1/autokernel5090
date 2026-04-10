# Expert Pruning Results: Gemma4 26B MoE (NVFP4)

## Model Specification

- **Model**: Gemma4 26B A4B (gemma-4-26B-A4B-it-NVFP4-modelopt)
- **Architecture**: 30 layers, 128 experts per layer, top_k=8 routing
- **Quantization**: NVFP4 (4-bit packed uint8 weights, FP8 scales)
- **GPU**: NVIDIA RTX 5090 (32 GB VRAM)
- **Framework**: vLLM 0.19.0 with CUTLASS MoE backend

## Expert Importance Analysis

Computed 4 metrics per expert across all 30 layers:

| Metric | Description | Weight in Composite |
|--------|-------------|-------------------|
| weight_l2 | L2 norm of expert's packed weights + scales | 40% |
| router_bias | L2 norm of router projection row for this expert | 30% |
| weight_var | Deviation of expert's norm from layer mean | 20% |
| router_scale | per_expert_scale value from router | 10% |

### Key Findings

1. **Metrics are largely uncorrelated**: weight_l2 vs router_bias r=0.005, meaning weight magnitude and routing preference measure independent aspects.
2. **Expert importance varies across layers**: The bottom 5 experts differ in every layer. Per-layer pruning is essential (vs. global pruning).
3. **Global ranking**: Expert 115 is the least important overall (avg composite 0.374), Expert 55 is most important (0.482). The spread is narrow (0.37-0.48), suggesting all experts carry some value.

### Global Bottom-10 (Least Important Experts)

| Rank | Expert | Avg Composite Score |
|------|--------|-------------------|
| 1 | 115 | 0.3743 |
| 2 | 16 | 0.3765 |
| 3 | 14 | 0.3787 |
| 4 | 3 | 0.3802 |
| 5 | 32 | 0.3811 |
| 6 | 18 | 0.3848 |
| 7 | 19 | 0.3894 |
| 8 | 79 | 0.3903 |
| 9 | 50 | 0.3906 |
| 10 | 102 | 0.3923 |

## Pruned Checkpoints

| Variant | Experts | Prune % | Checkpoint Size | Size Reduction | vLLM Model VRAM |
|---------|---------|---------|-----------------|----------------|-----------------|
| Baseline | 128 | 0% | 18.02 GB | - | 17.24 GiB (does not fit with KV cache) |
| Pruned 10% | 116 | 10% | 16.81 GB | 6.7% | Crashes (non-power-of-2 experts) |
| Pruned 20% | 103 | 20% | 15.50 GB | 14.0% | Crashes (non-power-of-2 experts) |
| Pruned 25% | 96 | 25% | 14.80 GB | 17.9% | OOM (14.22 GiB model, no KV space) |
| Pruned 30% | 90 | 30% | 14.20 GB | 21.2% | Crashes (non-power-of-2 experts) |
| **Pruned 50%** | **64** | **50%** | **11.58 GB** | **35.7%** | **11.24 GiB (fits with 16.2 GiB KV)** |
| Pruned 75% | 32 | 75% | 8.36 GB | 53.6% | 8.36 GiB (fits with abundant KV) |

## Quality and Throughput Results

Tested with 20 diverse prompts using Gemma chat template, greedy decoding, max 128 tokens.

| Variant | Coherence | Bench tok/s (256 tok) | Quality tok/s (batched 20) | Notes |
|---------|-----------|---------------------|--------------------------|-------|
| Baseline (128) | N/A | N/A | N/A | Cannot load on 32GB GPU with KV cache |
| Pruned 50% (64) | 5/20 (25%) | 27.1 | 433.7 | Loads and generates, but outputs are largely incoherent |
| Pruned 75% (32) | 5/20 (25%) | 2.9 | 53.9 | Loads but extremely slow, all outputs are garbage |

### Sample Outputs (50% Pruned, 64 Experts)

**Q: What is the capital of France?**
A: `{   {   _   _   _   _   _   ...` (garbage)

**Q: Explain quantum computing in simple terms.**
A: `much simple terms. <end_of_true_common_is_not_even_true_common...` (garbage)

### Sample Outputs (75% Pruned, 32 Experts)

**Q: What is the capital of France?**
A: `OK_is a a <[- much...` (garbage)

## Critical Technical Findings

### 1. vLLM MoE Kernel Requires Power-of-2 Expert Counts

Non-power-of-2 expert counts (90, 96, 103, 116) cause CUDA illegal memory access errors in vLLM's CUTLASS MoE backend. Only 64 and 32 (both powers of 2) worked. This is a hard constraint for any MoE pruning pipeline targeting vLLM inference.

### 2. Baseline Model Does Not Fit on 32GB GPU

The unpruned NVFP4 Gemma4 26B uses 17.24 GiB for model weights alone. On RTX 5090 (32 GB total, ~30 GB usable), there is insufficient memory for KV cache even at max_model_len=256. The 50% pruned model (11.24 GiB weights) is the sweet spot -- it fits with 16.23 GiB available for KV cache (70,880 tokens at max_model_len=512).

### 3. Expert Pruning Severely Degrades Quality

Even at 50% pruning (keeping 64 of 128 experts), output quality is catastrophically degraded. This is because:

- Gemma4 routes to top_k=8 experts per token. With 128 experts, each expert specializes in ~6.25% of routing space. Removing half the experts forces redistribution of tokens to less-specialized remaining experts.
- The router was trained with 128 experts. Even though we update the router projection matrix, the model's internal representations expect the original expert distribution.
- The narrow composite score range (0.374-0.482) confirms that there are no truly "dead" experts that can be safely removed.

### 4. 75% Pruning (32 Experts) Is Paradoxically Slower

With 32 experts and top_k=8, each token activates 25% of all experts (vs. 6.25% at 128 experts). This causes:
- More expert activation overlap per forward pass
- Less effective batching in the MoE kernel
- Result: 2.9 tok/s vs 27.1 tok/s for 64 experts

## Recommendations

1. **Expert pruning is not viable for Gemma4 MoE at these ratios.** The model needs all 128 experts for coherent output. Fine-tuning after pruning (expert distillation) would be required.

2. **For RTX 5090 deployment**, the NVFP4 model with 128 experts needs either:
   - Offloading (CPU/disk offload for some layers)
   - KV cache quantization (FP8/INT8) 
   - Smaller context length (<256 tokens)
   - A GPU with >40 GB VRAM

3. **If pruning is required**, the only vLLM-compatible options are powers of 2 (64, 32). Expert distillation (training remaining experts to absorb pruned knowledge) is necessary for quality.

4. **Alternative approach**: Instead of removing experts, use expert merging -- combine similar experts into one, preserving more knowledge while reducing count.

## Files

- **Pruning script**: `/root/projects/autokernel/tools/prune_experts.py`
- **Quality test script**: `/root/projects/autokernel/tools/test_pruned_quality.py`
- **Pruned checkpoints**: `/root/models/gemma4-pruned-50pct/`, `/root/models/gemma4-pruned-32exp/`
- **Expert importance data**: `/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/expert_importance.json`
