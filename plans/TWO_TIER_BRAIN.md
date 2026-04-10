# Two-Tier "Fast Brain / Slow Brain" Inference Architecture

## Overview

Model-level Mixture of Experts: instead of discarding pruned components, keep them as a CPU-resident fallback tier. The fast brain (pruned model on GPU) handles 95%+ of tokens at full speed. When it detects uncertainty, the slow brain (pruned experts/layers on CPU) loads on-demand.

**Key insight**: Pruning gives speed but loses quality on edge cases. This architecture recovers that quality without giving back the speed — because only the rare "hard" tokens pay the slow-path cost.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Token Stream                               │
│                        │                                      │
│                  ┌─────▼──────┐                               │
│                  │  Router /  │                                │
│                  │ Detector   │  ← 3 signals:                 │
│                  └─────┬──────┘    1. Router confidence       │
│                   ┌────┴────┐      2. Residual norm           │
│                   │         │      3. Output entropy           │
│              fast │         │ slow                             │
│              path │         │ path                             │
│           ┌──────▼──┐  ┌───▼────────┐                         │
│           │  Fast   │  │ CPU→GPU    │                         │
│           │  Brain  │  │ Transfer   │                         │
│           │  (GPU)  │  │ (~0.1ms)   │                         │
│           │         │  │    │       │                         │
│           │ Pruned  │  │ ┌──▼─────┐ │                         │
│           │ Model   │  │ │ Slow   │ │                         │
│           │ 90 exp  │  │ │ Expert │ │                         │
│           │ 27 lay  │  │ │ on GPU │ │                         │
│           └────┬────┘  │ └──┬─────┘ │                         │
│                │       │    │       │                         │
│                └───┬───┘    │       │                         │
│                    │  ◄─────┘       │                         │
│                ┌───▼────┐           │                         │
│                │ Merge  │           │                         │
│                │ Output │           │                         │
│                └───┬────┘           │                         │
│                    │                │                         │
│                    ▼                │                         │
│               Next Token            │                         │
└──────────────────────────────────────────────────────────────┘
```

## Three Detection Signals

### 1. Router Confidence (MoE layers)

The original router produces logits for all 128 experts. If top-k selection includes a pruned expert ID with weight above threshold, that token needs the slow brain.

- **When**: Every MoE layer forward pass
- **Cost**: ~0.01ms (softmax + top-k on 128-dim vector)
- **Precision**: High — directly measures whether the token *wants* the removed expert

### 2. Residual Norm (Pruned layers)

At positions where layers were removed, measure the hidden state norm. Outlier norms indicate the skipped layer would have had a large effect.

- **When**: At each pruned layer position
- **Cost**: ~0.005ms (vector norm)
- **Calibration**: Run calibration batch to set per-layer threshold at 95th percentile

### 3. Output Entropy (Final logits)

If the fast brain's output distribution has high entropy (uncertain about next token), the slow brain might resolve the ambiguity.

- **When**: After final LM head, before sampling
- **Cost**: ~0.02ms (softmax + entropy on vocab)
- **Note**: This is the most expensive signal but catches cases where accumulated small errors cause confusion

## Slow Brain Components

### Pruned Experts (CPU → GPU on demand)

| Property | Value |
|----------|-------|
| Storage | CPU RAM (pinned for fast DMA) |
| Per-expert size | ~3 MB (NVFP4 packed) |
| Total (30% prune, 30 layers) | ~3.4 GB CPU RAM |
| Transfer latency | ~0.1ms per expert (PCIe 5.0) |
| GPU LRU cache | 64 experts, 256 MB budget |
| Cache hit = | Zero additional latency |

### Pruned Layers (CPU → GPU on demand)

| Property | Value |
|----------|-------|
| Storage | CPU RAM |
| Per-layer size | ~200 MB (full transformer block) |
| Total (3 layers) | ~600 MB CPU RAM |
| Transfer latency | ~1ms per layer |
| Usage frequency | Very rare (< 1% of tokens) |

## Performance Projections

### Gemma 4 26B on RTX 5090

```
Configuration              Throughput    Quality    VRAM
─────────────────────────  ──────────    ───────    ────
Full model (baseline)      6,685 tok/s   100%      ~14 GB
Pure prune (30% experts)   ~8,700 tok/s   ~97%     ~10 GB
Pure prune (+ 3 layers)    ~9,200 tok/s   ~94%      ~9 GB

Two-tier (30% experts):
  Fast path (95% tokens)   ~8,700 tok/s   100%*    ~10 GB
  Slow path (5% tokens)    ~2,000 tok/s   100%      +256 MB GPU cache
  Effective average         ~8,400 tok/s   ~100%    ~10 GB GPU + 3.4 GB CPU

Two-tier (30% + 3 layers):
  Fast path (96% tokens)   ~9,200 tok/s   100%*    ~9 GB
  Slow path (4% tokens)    ~1,500 tok/s   100%      +256 MB GPU cache
  Effective average         ~8,900 tok/s   ~100%    ~9 GB GPU + 4 GB CPU
```

*Quality on fast path tokens matches full model because those tokens didn't need the pruned components anyway.

### Effective Throughput Formula

```
T_effective = T_fast * (1 - hit_rate) + T_slow * hit_rate

Where:
  T_fast = throughput with pruned model (fast brain only)
  T_slow = throughput when slow brain is invoked
  hit_rate = fraction of tokens needing slow brain
```

### Hit Rate Sensitivity

| Hit Rate | Effective tok/s | % of Pruned Speed |
|----------|----------------|-------------------|
| 1%       | 8,633          | 99.2%             |
| 5%       | 8,365          | 96.1%             |
| 10%      | 8,030          | 92.3%             |
| 20%      | 7,360          | 84.6%             |
| 50%      | 5,350          | 61.5%             |

The architecture is only valuable when hit rate < ~20%.

## Memory Budget

### GPU (Fast Brain)

| Component | Size |
|-----------|------|
| Model weights (pruned) | ~10 GB |
| KV cache (FusenCache) | ~2 GB |
| Slow expert GPU cache | 256 MB |
| Activation workspace | ~500 MB |
| **Total** | **~12.8 GB** |

### CPU (Slow Brain)

| Component | Size |
|-----------|------|
| Pruned expert weights | ~3.4 GB |
| Pruned layer weights | ~600 MB |
| Transfer staging buffers | ~100 MB |
| **Total** | **~4.1 GB** |

### Compared to Full Model

| Setup | GPU | CPU | Quality |
|-------|-----|-----|---------|
| Full model | 14 GB | 0 | 100% |
| Pure pruning | 10 GB | 0 | ~94-97% |
| Two-tier | 10-11 GB | 4 GB | ~100% |

Two-tier trades 4 GB CPU RAM for near-zero quality loss.

## Implementation Files

| File | Purpose |
|------|---------|
| `tools/two_tier_brain.py` | Core classes: TwoTierModel, SlowBrainExperts, SlowBrainLayers, SlowBrainDetector, GPUExpertCache, TwoTierGeneric |
| `tools/prepare_two_tier.py` | Checkpoint splitting: full model → fast brain + slow experts + slow layers |
| `tools/test_two_tier.py` | 10-category test suite (detection, cache, quality, hit rate, etc.) |

## Key Design Decisions

### 1. CPU RAM over SSD

Expert weights are ~3 MB each. CPU RAM random access: ~0.1ms. NVMe SSD: ~1ms. The 10x latency difference matters because slow path tokens are already paying a penalty. CPU RAM keeps the slow path tolerable.

### 2. GPU LRU Cache

Frequently-requested slow experts stay on GPU. With a 64-expert / 256 MB cache, the most common slow experts are always ready. This turns repeated slow-path requests into fast-path latency.

### 3. Batch Splitting

In a batch of 32 tokens, if 2 need the slow brain, we process the other 30 at full fast-brain speed and handle the 2 separately. This prevents slow tokens from blocking fast tokens.

### 4. Async Prefetch

During layer N's MoE computation, we predict which slow experts layer N+1 might need and start the CPU→GPU transfer on a separate CUDA stream. By the time layer N+1 runs, the expert is already on GPU.

### 5. Graceful Degradation

If the CPU→GPU transfer takes too long (configurable, default 2ms), skip the slow brain for that token. Accept minor quality loss rather than blocking the entire batch. This makes latency bounded.

## Comparison to Alternatives

| Approach | Speed | Quality | Complexity | Memory |
|----------|-------|---------|------------|--------|
| Full model | 1x | 100% | Low | High GPU |
| Pure pruning | 1.3x | 94-97% | Low | Low GPU |
| MoE offloading (all) | 0.3x | 100% | Medium | Low GPU, high CPU |
| Speculative decoding | 1.5-2x | 100% | High | 2x GPU |
| **Two-tier brain** | **1.26x** | **~100%** | **Medium** | **Med GPU + Med CPU** |

## Generalizations

The two-tier pattern extends beyond pruning:

### Small Model + Large Model
- Fast brain: 9B model (full speed)
- Slow brain: 26B model (invoked for hard reasoning)
- Detection: output entropy, confidence calibration

### Quantized + Full Precision
- Fast brain: FP4 quantized (fast, slightly lossy)
- Slow brain: BF16 full precision (for precision-sensitive tokens)
- Detection: activation outlier detection, gradient sensitivity

### Local + Cloud
- Fast brain: on-device model
- Slow brain: API call to larger model
- Detection: confidence threshold, domain classifier

## Next Steps

1. **Run prepare_two_tier.py** on Gemma 4 26B checkpoint to create the split
2. **Measure actual hit rates** on diverse evaluation prompts
3. **Benchmark end-to-end** with vLLM integration (custom scheduler hook)
4. **Calibrate thresholds** using validation set to minimize hit rate while preserving quality
5. **Integrate with FusenCache** — KV cache compression + two-tier brain = maximum memory efficiency
