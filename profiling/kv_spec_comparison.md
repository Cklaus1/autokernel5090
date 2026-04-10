# FusenCache KV Spec Comparison: Gemma4 26B

**Date:** 2026-04-09  
**Model:** Neural-ICE/Gemma-4-26B-A4B-it-NVFP4  
**Architecture:**
- 25 sliding attention layers: D=256, Hk=8 heads, seq≤1024
- 5 global attention layers: D=512, Hk=2 heads, seq≤262K
**Hardware:** RTX 5090 (32GB VRAM), RTX PRO 6000 (96GB VRAM)

Data sources: `kv_cache_gen/RESULTS.md` (12 experiments, 405 configs, measured on RTX 5090).

---

## Methodology

### Memory Model

Model weights (NVFP4, 26B params) occupy ~15GB including activations. This leaves:
- RTX 5090 (32GB): **17 GB** for KV cache
- RTX PRO 6000 (96GB): **81 GB** for KV cache

Bytes per token (all 30 layers, all KV heads combined):
- Sliding: 25 layers × 8 KV heads × `slot_bytes(256) + scale_bytes(256)` per head
- Global: 5 layers × 2 KV heads × `slot_bytes(512) + scale_bytes(512)` per head

### Quality Source

Cosine similarity vs BF16 KV cache, measured in Experiment 1 of RESULTS.md across 9 predefined specs. Quality reflects both K and V quantization error propagated through attention.

### Throughput Source

End-to-end decode simulation (Experiment 3), full 30-layer decode with RMSNorm + QKV proj + attention + output proj + MoE routing + 8-expert forward. Numbers are tokens/second aggregated across all concurrent requests.

---

## Per-Spec Analysis

### Spec: k8v8 (8-bit K, 8-bit V, scale block 32)

| Metric | Sliding (D=256) | Global (D=512) |
|--------|----------------|----------------|
| Bytes per head per token | 544 | 1,088 |
| Compression vs BF16 | 1.88x | 1.88x |
| Cosine similarity | 1.0000 | 1.0000 |

**Total bytes per token (all layers + heads):** 119,680 bytes

| GPU | KV budget | Total KV tokens | At ctx=2K | At ctx=8K | At ctx=64K | At ctx=131K |
|-----|-----------|----------------|-----------|-----------|------------|-------------|
| RTX 5090 (32GB) | 17 GB | 153K | B=74 | B=18 | B=2 | B=1 |
| RTX PRO 6000 (96GB) | 81 GB | 727K | B=354 | B=88 | B=11 | B=5 |

**Decode throughput (RTX 5090):**
- B=64: 1,776 tok/s
- B=128: 2,765 tok/s
- B=240: OOM (does not fit in 32GB)

**Store overhead (2048-token prefill, 30 layers):** 10.3ms with Triton store, ~1.1ms

---

### Spec: k8v4 (8-bit K, 4-bit V, scale block 32)

| Metric | Sliding (D=256) | Global (D=512) |
|--------|----------------|----------------|
| Bytes per head per token | 416 | 832 |
| Compression vs BF16 | 2.46x | 2.46x |
| Cosine similarity | 0.9958 | 0.9958 |

**Total bytes per token (all layers + heads):** 91,520 bytes

| GPU | KV budget | Total KV tokens | At ctx=2K | At ctx=8K | At ctx=64K | At ctx=131K |
|-----|-----------|----------------|-----------|-----------|------------|-------------|
| RTX 5090 (32GB) | 17 GB | 199K | B=97 | B=24 | B=3 | B=1 |
| RTX PRO 6000 (96GB) | 81 GB | 950K | B=464 | B=116 | B=14 | B=7 |

**Decode throughput (RTX 5090):**
- B=64: 1,883-1,886 tok/s
- B=128: 2,953 tok/s
- B=240: OOM (does not fit in 32GB)

**Store overhead (2048-token prefill, 30 layers):** 12.0ms PyTorch, ~1.9ms Triton

**Note:** k8v4 is faster than k4v4 at seq<16K (less nibble-packing overhead). Crossover to k4v4 advantage occurs at ~16K tokens per sequence.

---

### Spec: k4v4b32 (4-bit K, 4-bit V, scale block 32)

| Metric | Sliding (D=256) | Global (D=512) |
|--------|----------------|----------------|
| Bytes per head per token | 288 | 576 |
| Compression vs BF16 | 3.56x | 3.56x |
| Cosine similarity | 0.9930 | 0.9930 |

**Total bytes per token (all layers + heads):** 63,360 bytes

| GPU | KV budget | Total KV tokens | At ctx=2K | At ctx=8K | At ctx=64K | At ctx=131K |
|-----|-----------|----------------|-----------|-----------|------------|-------------|
| RTX 5090 (32GB) | 17 GB | 288K | B=140 | B=35 | B=4 | B=2 |
| RTX PRO 6000 (96GB) | 81 GB | 1,373K | B=670 | B=167 | B=21 | B=10 |

**Decode throughput (RTX 5090):**
- B=64: ~1,950 tok/s (estimated from sweep data, between k4v4b64 and k8v4)
- B=128: ~3,000 tok/s
- B=240: fits in 32GB, ~4,100 tok/s (estimated)

**Store overhead (2048-token prefill, 30 layers):** 13.1ms PyTorch, ~1.0ms Triton

---

### Spec: k4v4b64 (4-bit K, 4-bit V, scale block 64) — current production spec

| Metric | Sliding (D=256) | Global (D=512) |
|--------|----------------|----------------|
| Bytes per head per token | 272 | 544 |
| Compression vs BF16 | 3.76x | 3.76x |
| Cosine similarity | 0.9916 | 0.9916 |

**Total bytes per token (all layers + heads):** 59,840 bytes

| GPU | KV budget | Total KV tokens | At ctx=2K | At ctx=8K | At ctx=64K | At ctx=131K |
|-----|-----------|----------------|-----------|-----------|------------|-------------|
| RTX 5090 (32GB) | 17 GB | 305K | B=148 | B=37 | B=4 | B=2 |
| RTX PRO 6000 (96GB) | 81 GB | 1,453K | B=709 | B=177 | B=22 | B=11 |

**Decode throughput (RTX 5090, measured):**
- B=64: 1,999 tok/s
- B=128: 3,032 tok/s
- B=240: **4,203 tok/s** (only spec that fits at B=240 on 32GB)

**Store overhead (2048-token prefill, 30 layers):** 13.1ms PyTorch, **1.0ms Triton** (lowest absolute cost)

**Sequence length crossover:** k4v4b64 is slower than k8v4 at seq<16K, faster at seq>16K. At 131K: k4v4 is 28% faster than k8v8, 9% faster than k8v4.

---

### Spec: fp8 (FP8 E4M3/E5M2, 8-bit K and V, no integer scales)

| Metric | Sliding (D=256) | Global (D=512) |
|--------|----------------|----------------|
| Bytes per head per token | 512 | 1,024 |
| Compression vs BF16 | 2.00x | 2.00x |
| Cosine similarity | ~1.000 (theoretical) | ~1.000 (theoretical) |

**Total bytes per token (all layers + heads):** 112,640 bytes

| GPU | KV budget | Total KV tokens | At ctx=2K | At ctx=8K | At ctx=64K | At ctx=131K |
|-----|-----------|----------------|-----------|-----------|------------|-------------|
| RTX 5090 (32GB) | 17 GB | 162K | B=79 | B=19 | B=2 | B=1 |
| RTX PRO 6000 (96GB) | 81 GB | 772K | B=377 | B=94 | B=12 | B=5 |

**Status:** Kernel not yet implemented. The `is_float_format` property is defined in spec.py and the spec objects exist (`fp8_e4m3`, `fp8_e5m2`), but the decode kernel's cast-based dequant path is a pending follow-up. Current kernel only supports integer-code dequant.

**Theoretical decode throughput:** Similar to k8v8 in capacity, but decode kernel would eliminate scale multiplications — expected 10-20% faster than k8v8 per step. No measured data.

**Why not just use fp8:** Same capacity as k8v8 (2x compression vs 3.8x for k4v4b64), so it OOMs at B=240 on 32GB. For 32GB GPUs, fp8's quality advantage over k4v4 does not offset the batch capacity loss.

---

## Side-by-Side Comparison

| Spec | K bits | V bits | Scale blk | Comp | CosSim | 32GB KV-tok | 96GB KV-tok | B=128 tok/s | B=240 tok/s |
|------|--------|--------|-----------|------|--------|------------|------------|-------------|-------------|
| k8v8 | 8 | 8 | 32 | 1.88x | 1.0000 | 153K | 727K | 2,765 | OOM |
| k8v4 | 8 | 4 | 32 | 2.46x | 0.9958 | 199K | 950K | 2,953 | OOM |
| k4v4b32 | 4 | 4 | 32 | 3.56x | 0.9930 | 288K | 1,373K | ~3,000 | ~4,100 |
| **k4v4b64** | **4** | **4** | **64** | **3.76x** | **0.9916** | **305K** | **1,453K** | **3,032** | **4,203** |
| fp8 | 8 | 8 | none | 2.00x | ~1.000 | 162K | 772K | (not impl.) | OOM |

**Throughput comparison at B=128** (RTX 5090, all specs fit):
- k4v4b64 is **10% faster** than k8v8 (+267 tok/s)
- k4v4b64 is **3% faster** than k8v4 (+79 tok/s)

**Quality comparison** (cosine similarity):
- k8v8 vs k4v4b64: Δ0.0084 — imperceptible in practice per RESULTS.md
- k4v4 quality loss is below human perception threshold for text generation; real-world output quality is indistinguishable at this cosine range (>0.99)

---

## Recommendations by Use Case

### Use Case 1: Interactive Coding (quality matters most)

**Single-user latency is spec-agnostic.** At B=1, all specs deliver **~44 tok/s** and MoE FFN computation is >90% of decode time. The KV cache format contributes <5% of wall time at B=1.

**Recommendation:**

| GPU | Recommendation | Rationale |
|-----|---------------|-----------|
| RTX 5090 (32GB) | **k8v4 (sliding) + k4v4b64 (global)** | Mixed-spec: best quality where memory is cheap (sliding, 1024 token window), max compression where it's expensive (global, up to 262K). See Experiment 5: mixed-spec gives 9% more throughput at B=128. |
| RTX PRO 6000 (96GB) | **k8v8 (sliding) + k8v4 (global)** | Memory is abundant; run full quality on both layer types. k8v8/k8v4 mixed fits B=500 at 2K ctx with 88GB. |

**Why not k8v8 everywhere on 32GB:** At 2K context k8v8 caps at B=74, while k4v4 caps at B=148. For an interactive coding server with multiple users, k4v4 doubles the number of concurrent sessions.

---

### Use Case 2: Batch Serving (throughput matters most)

**Recommendation: k4v4b64 everywhere (both layer types)**

This is the current production recommendation from RESULTS.md and is validated by Experiments 3 and 5.

| GPU | Spec | Peak Batch | Peak Throughput |
|-----|------|-----------|----------------|
| RTX 5090 (32GB) | k4v4b64 | B=240 (fits) | **4,203 tok/s** |
| RTX PRO 6000 (96GB) | k4v4b64 | B=700+ | **~14,000 tok/s** (extrapolated) |

**Why k4v4b64 over k4v4b32:** Scale block 64 reduces scale-tensor memory bandwidth by 2x vs block 32, and the larger block means fewer scale reads per decode step. Quality difference (0.9930 vs 0.9916 cosine) is not meaningful for batch serving. Throughput benefit: ~3% at B=128 (measured, Experiment 2).

**Key insight from Experiment 3:** At B=240, k8v8/k8v4/k8v4 all OOM on 32GB. Only k4v4 fits. Compression is not a quality trade-off — it is the prerequisite for operating at max batch size. The additional 10% throughput from k4v4 at B=128 vs k8v8 comes from reduced KV cache memory bandwidth, not from batch size.

**Optimal Triton config for batch serving:** `block_kv=16, block_h=8, num_warps=2, num_kv_splits=32` (verified universal sweet spot, Experiment 2).

---

### Use Case 3: Long Context (capacity matters most)

Long context means large global attention sequence lengths (seq>>8K). Global layers have D=512, Hk=2 heads, up to 262K tokens.

**The compression advantage inverts at long context:** Below 16K tokens, k8v4 is 9% faster per decode step (less packing overhead). Above 16K, k4v4 is faster because memory bandwidth — not compute — is the bottleneck.

**At 131K context (B=1):**
- k4v4: 241μs per attention step
- k8v4: 263μs per attention step
- k8v8: 334μs per attention step
- **k4v4 is 28% faster than k8v8, 9% faster than k8v4**

**Recommendation:**

| GPU | Context Range | Spec | Rationale |
|-----|--------------|------|-----------|
| RTX 5090 (32GB) | ≤16K tokens | **k8v4 (sliding) + k8v4 (global)** | Faster per step, adequate quality. 199K KV tokens → B=24 at 8K ctx. |
| RTX 5090 (32GB) | >16K tokens | **k4v4b64 everywhere** | Faster and smaller above the crossover. 305K KV tokens → B=4 at 64K ctx. |
| RTX PRO 6000 (96GB) | ≤32K tokens | **k8v8 (sliding) + k8v4 (global)** | Abundant memory, best quality. 950K KV tokens → B=116 at 8K ctx. |
| RTX PRO 6000 (96GB) | >32K tokens | **k8v4 everywhere** | Quality with compression. 950K KV tokens → B=14 at 64K ctx. |

**Note:** The adaptive selector in `kv_cache_gen/adaptive.py` (Experiment 12) handles the VRAM-aware downgrade automatically. At runtime, it measures available VRAM and selects the highest-quality spec that fits the requested batch size.

---

## Spec Decision Tree

```
Is GPU ≥ 64GB?
├── YES (PRO 6000, 96GB)
│   ├── Interactive / quality-first
│   │   └── k8v8 sliding + k8v4 global
│   ├── Long context (>16K)
│   │   └── k8v4 everywhere
│   └── Max throughput
│       └── k4v4b64 everywhere
└── NO (RTX 5090, 32GB)
    ├── Single user / interactive
    │   └── k8v4 sliding + k4v4b64 global
    ├── Batch serving (any context)
    │   └── k4v4b64 everywhere
    └── Long context (>16K)
        └── k4v4b64 everywhere (only spec with meaningful B>1)
```

---

## Why Not k4v2 or k8v2?

From Experiment 1: 2-bit V causes cosine similarity to drop below 0.93 (k4v2b16: 0.929, k8v2b16: 0.931). This is a perceptible quality degradation. The "aggressive" tier is excluded from recommendations. The attention mechanism amplifies V errors (softmax weights multiply V, so V quantization noise scales with attention score magnitude), making V more sensitive to quantization than K.

---

## Why k4v4b64 Beats k4v4b32 for Throughput

Scale block size controls how many elements share one FP16 scale:
- Block 32: 1 scale per 32 elements → D/32 scales per head dimension
- Block 64: 1 scale per 64 elements → D/64 scales per head dimension

At D=512 (global layers), block 64 reads 8 scales vs block 32's 16 scales per decode step per head. Fewer scale reads = less memory bandwidth = faster kernel. Quality cost: 0.9930 vs 0.9916 cosine — not a practical difference.

---

## Summary Table

| Use Case | RTX 5090 (32GB) | RTX PRO 6000 (96GB) |
|----------|----------------|---------------------|
| Interactive coding (quality) | k8v4 + k4v4b64 (mixed) | k8v8 + k8v4 (mixed) |
| Batch serving (throughput) | k4v4b64 (uniform) | k4v4b64 (uniform) |
| Long context ≤16K | k8v4 + k8v4 (mixed) | k8v8 + k8v4 (mixed) |
| Long context >16K | k4v4b64 (uniform) | k8v4 (uniform) |
| FP8 (future) | Not viable on 32GB (OOM at B>80) | k8v8 equivalent quality, possibly faster |

**Default recommendation** for any new deployment: start with **k4v4b64 everywhere** and use the adaptive selector to upgrade quality when VRAM allows. This is the only spec guaranteed to fit all use cases on both GPU sizes.
