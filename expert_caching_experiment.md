# Expert Weight Reordering L2 Cache Experiment

## Hypothesis

Reordering expert indices in the Gemma4 26B NVFP4 checkpoint so that frequently-routed
experts are stored contiguously (experts 0-31 = most popular) would improve L2 cache
hit rates and throughput, since hot experts would be more likely to stay resident in the
RTX 5090's 96 MB L2 cache between consecutive MoE layers.

## Result: NEGATIVE -- Expert reordering provides zero measurable benefit

The hypothesis is falsified by three independent lines of evidence:

1. **Routing is near-uniform across experts** -- there is no stable "hot set" to exploit
2. **L2 temporal reuse between layers is negligible** -- each layer has different weights
3. **Expert memory layout does not affect access bandwidth** -- gather is compute-bound, not cache-bound

## Hardware Context

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX 5090 |
| L2 cache | 96 MB |
| DRAM bandwidth | 1,792 GB/s |
| Expert count | 128 per MoE layer |
| MoE layers | 30 |
| Top-k routing | 8 |
| Expert size (NVFP4) | 3.19 MB (gate_up + down + scales) |
| L2 expert capacity | ~30 experts (weights only) |
| MoE backend | VLLM_CUTLASS (grouped GEMM) |

## Experiment 1: Routing Frequency Analysis

### Method
Loaded router projection weights from the NVFP4 checkpoint and routed real token
embeddings (315 tokens from 32 diverse prompts) through all 30 MoE layers.

### Findings

**Per-layer routing entropy: 5.77 - 6.55 bits (vs 7.00 uniform)**

The routing IS skewed within each layer -- the top-32 experts per layer handle 55-76%
of routing. However:

| Metric | Value | Implication |
|--------|-------|-------------|
| Global entropy (across all layers) | 6.97 / 7.00 | Near-uniform globally |
| Cross-layer Jaccard similarity of top-32 sets | 0.143 | Each layer has DIFFERENT hot experts |
| Adjacent layer top-32 overlap | 8 / 32 | Only 25% overlap between consecutive layers |
| If global top-32 mapped to experts 0-31 | 22-40% hit per layer | Most layers would still miss |

**Key insight**: While individual layers have skewed routing, each layer's hot set is
different. There is no stable global hot set that would benefit from contiguous placement.
The top-32 globally hottest experts only capture 31.7% of total routing decisions --
barely above the 25% expected from uniform distribution.

### Per-layer routing distribution

```
Layer  Entropy  Top16%  Top32%  Gini    Max%   Active
    0    6.084    45.0    68.9  0.6004   5.44     115
    5    6.286    42.1    64.8  0.5395   3.77     122
   10    6.474    36.0    57.7  0.4680   3.69     126
   15    6.552    33.7    55.0  0.4351   2.98     126
   20    6.266    43.3    63.1  0.5366   5.40     122
   25    6.397    39.4    61.5  0.4978   4.21     126
   29    5.765    55.4    75.6  0.6681   8.10     115
```

Layer 29 (final) shows the most skew, but even there, 115/128 experts are active.

## Experiment 2: L2 Cache Access Pattern Benchmarks

### Method
Allocated tensors matching actual NVFP4 expert dimensions:
- w13_weight: [128, 1408, 1408] uint8 (1.89 MB/expert)
- w2_weight: [128, 2816, 352] uint8 (0.95 MB/expert)

Measured gather bandwidth with different expert index patterns.

### Findings

**Expert access pattern has ZERO effect on bandwidth:**

| Access Pattern | Time (us) | Bandwidth (GB/s) |
|---------------|-----------|-------------------|
| Contiguous [0:8] | 43.0 | 343 |
| Contiguous [60:68] | 42.7 | 346 |
| Random 8 experts | 42.6 | 346 |
| Max stride [0,16,32,...] | 42.6 | 346 |

All patterns achieve identical bandwidth (~345 GB/s). This is because:
1. Each expert is 1.89 MB -- far larger than any cache line (128B)
2. The GPU's memory controller handles large, aligned accesses regardless of stride
3. CUDA's memory subsystem parallelizes across memory channels

### L2 Temporal Reuse (consecutive layers)

| Scenario | Time (us) | Speedup |
|----------|-----------|---------|
| 8 experts, same set (L2 hit) | 69.8 | -- |
| 8 experts, different set (L2 miss) | 70.9 | 1.016x |
| 32 experts, same set (L2 hit) | 306.2 | -- |
| 32 experts, different set (L2 miss) | 308.2 | 1.006x |

**L2 temporal reuse provides only 0.6-1.6% speedup**, even when the SAME experts are
accessed consecutively. This is because:
1. Each MoE layer has its OWN weight tensors -- layer N's w13 is a completely different
   allocation than layer N+1's w13
2. Even if the same expert IDs are routed, the actual memory addresses are different
3. The L2 cache can only help with temporal reuse on the SAME memory addresses

### Realistic B=32 scenario (30-layer pass)

| Order | Time (ms) | Speedup |
|-------|-----------|---------|
| Natural order | 30.45 | -- |
| Reordered (hot experts = low indices) | 30.60 | 0.995x |

**Reordering is slightly SLOWER** (within noise) because the reorder computation
itself adds overhead with no L2 benefit.

## Experiment 3: Architecture Analysis

### Why expert reordering cannot help in this architecture

1. **Each layer has independent weight tensors**: Layer 0's experts are at completely
   different GPU memory addresses than layer 1's experts. L2 cache lines from layer 0
   are evicted before layer 1 runs because layer 1's weights are elsewhere in VRAM.

2. **CUTLASS grouped GEMM uses pointer arithmetic**: The kernel computes
   `base_ptr + expert_id * stride` to find each expert's weights. The expert_id only
   determines an offset into the tensor -- whether expert 0 is "hot" or "cold" makes
   no difference to the memory access pattern of the GEMM kernel.

3. **Expert weights are too large for meaningful L2 benefit**: At 3.19 MB per expert,
   the L2 can hold ~30 experts. But with B=32, top_k=8, approximately 109 out of 128
   experts are active per layer. The working set far exceeds L2 capacity.

4. **Bandwidth is not the bottleneck**: Measured gather bandwidth is ~345 GB/s for
   8 experts, which is only 19% of the RTX 5090's 1,792 GB/s peak DRAM bandwidth.
   The operation is compute-bound (the .sum() reduction), not memory-bound.

5. **The real MoE bottleneck is the grouped GEMM compute**: For NVFP4 matmul,
   the CUTLASS kernel is compute-bound at these small per-expert matrix sizes
   (1408x1408 per expert). Memory layout optimization targets the wrong bottleneck.

## Conclusion

Expert weight reordering provides **zero measurable throughput benefit** for Gemma4 26B
NVFP4 on RTX 5090. The optimization is theoretically unsound for three independent reasons:

1. No stable cross-layer hot expert set exists (Jaccard similarity = 0.14)
2. Each layer's weights are at different memory addresses (no cross-layer L2 reuse possible)
3. Expert gather bandwidth is pattern-independent at these sizes

**Recommendation**: Do NOT pursue expert reordering. Instead, focus optimization effort on:
- Reducing the number of active experts (expert pruning/merging)
- Improving the grouped GEMM kernel efficiency for small per-expert matrices
- KV cache optimization (much larger potential impact)

## Scripts

| Script | Purpose |
|--------|---------|
| `exp_routing_freq.py` | Routing frequency with random hidden states (baseline) |
| `exp_routing_real2.py` | Routing frequency with real BF16 token embeddings |
| `exp_l2_cache_sim.py` | L2 cache simulation with various access patterns |
| `exp_l2_ncu_profile.py` | Detailed L2 profiling with NVFP4 dimensions |

All scripts are in `/root/projects/autokernel/`.
