# Adaptive Sparse Attention Feasibility — H2O + SWA Hybrid

**Tag:** W2_5e_adaptive_sparse_feasibility  
**Date:** 2026-04-18  
**Scope:** CPU-only literature + static analysis. No new benchmarks run.

---

## 1. Proposed Scheme

**H2O + SWA hybrid** (most promising path):

- Sliding layers (25/30): keep existing SWA decode kernel — geometric sparsity, window=4096.  
  No adaptive component here; Gemma4 is trained with SWA and any deviation from the
  geometric window risks quality on the layers that consume the bulk of HBM traffic.
- Global layers (5/30): apply H2O-style adaptive eviction, identical in spirit to the
  `semantic_kv_eviction.md` FusenEvict design (EMA decay α=0.95, sink=16, recency=512).
  These are the only layers where KV grows unboundedly and adaptive selection can add
  marginal benefit beyond what SWA already provides.

**Token budget at seq_len=16384:**

| Component | Tokens retained | Fraction |
|-----------|----------------|----------|
| SWA window (sliding layers) | 4096 | 25.0% |
| H2O heavy hitters, global layers (5%) | ~400 | 2.4% additive |
| Recency + sinks (global) | 528 | 3.2% additive |
| **Effective (geometric union)** | **~4500** | **~27%** |

Geometric sparsity via SWA alone is already 4096/16384 = 25% of KV; the H2O component
adds ~400 heavy-hitter tokens on 5 layers only — a second-order effect on overall FLOPs.

---

## 2. Top-k Selection Cost Analysis

**Core problem:** top-k adaptive selection on global layers does NOT reduce Q·K^T cost.
You must compute all attention logits first, then select top-k indices, then accumulate
V·softmax(P_topk). The savings come ONLY from the V·P accumulation (O(k·d) vs O(S·d))
and from the reduced KV eviction keeping cache pressure lower over time.

At decode M=1, the bottleneck is HBM bandwidth, not FLOP count:
- Q·K^T for one global layer at S=16K: 16384 × 256 × 2B (BF16) = 8 MB KV load.
  This is already loaded; you cannot skip it for adaptive selection.
- Savings from not accumulating V for evicted tokens: 400/16384 ≈ 2.4% V-load reduction
  per global layer at 16K, scaling linearly with sparsity ratio.

**Net ops overhead** of the top-k scatter/gather indirection inside the Triton kernel:
estimated +5-10% kernel launch overhead (index lookup, non-contiguous gather) vs.
contiguous page iteration in `swa_decode.py`.

For sliding layers (25/30), there is no Q·K^T to run outside the window — adding
adaptive selection would require computing the full S logits before selecting,
**eliminating the SWA speedup entirely** for those layers. This path is infeasible.

---

## 3. Context Length Breakpoint

Define breakpoint as: seq_len at which H2O global-layer savings exceed the top-k
indirection overhead AND compound with SWA geometric savings.

**Geometric SWA savings** (already realized):
- At S=8192: 2× KV reduction on 25 layers → ~1.9× effective e2e (observed 2.52× includes
  other overhead reduction and batch effects at C=8).
- At S=16384: 4× KV reduction on 25 layers.

**Adaptive H2O savings** (global layers only, 5 layers):
- Each global layer KV budget: 4096 (recency+sinks+heavy-hitters) vs S tokens.
- Fractional layer weight: 5/30 ≈ 16.7% of total attention cost.
- At S=16384: H2O saves (16384-4096)/16384 = 75% of global-layer KV reads.
- Contribution to e2e: 0.167 × 0.75 = 12.5% reduction in attention HBM traffic.
- After overhead tax (~8%): net ~4-5% e2e benefit.

**Breakpoint:** The overhead of top-k gather indirection (≈5-10% kernel latency) eats the
H2O benefit until S is large enough that the 5-layer savings outpace overhead. Crossing
point is roughly:

```
0.167 * (1 - 4096/S) > 0.08  →  S > 4096 / (1 - 0.48) ≈ 7,900 tokens
```

**Practical breakpoint: S ≈ 8K–10K tokens.** Below 8K, overhead dominates.
Above 16K, H2O adds ~4-5% on top of SWA's dominant geometric savings.

The SWA benefit scales strongly (linear with S/window) while H2O benefit is bounded to 5
layers and saturates once those 5 layers are well-sparse. At S=64K, SWA contributes
~16× KV reduction on sliding layers; H2O contributes at most ~15× on global layers (5 of
30), net ~3.5% additive e2e improvement — marginal at that point.

---

## 4. Projected Lift vs Current 2.52×

Baseline: 2.52× e2e at C=8, S=12K, SWA window=4096 (geometric only).

| Scheme | Approx e2e speedup | Delta vs SWA |
|--------|--------------------|-------------|
| Pure SWA geometric (current) | 2.52× | — |
| H2O on global layers, S=12K | ~2.62× | +4% |
| H2O on global layers, S=32K | ~2.85× | +13% |
| H2O on global layers, S=64K | ~2.95× | +17% |

The gains are real but modest. SWA already handles 83% of attention layers geometrically;
H2O only improves the remaining 17%. To get a 3× compound requires S≥32K where H2O
provides ~12-15% over baseline SWA — meaningful for very-long-context workloads only.

**T3-L comparison:** The prior semantic eviction result (2.5× at 90% retention, S=1.5K)
was measured at short context where the window itself is the bottleneck, not the
token-importance distribution. At S=12K the picture changes: SWA already shrinks the
window to 4096 tokens, and H2O operating on those 4096 tokens within global layers would
provide genuine additional pruning. **Longer context does meaningfully change the picture
for global-layer eviction, not for sliding layers.**

---

## 5. Correctness Risk

**Sliding layers (SWA, no change):** Zero additional risk. Current kernel is validated
cos>0.9999 vs FlashInfer.

**Global layers (H2O adaptive):**
- Gemma4 global layers use full attention — no trained sparsity pattern. Arbitrary
  top-k selection introduces an out-of-distribution attention mask.
- H2O paper: top-20% KV budget retains >95% generation quality on summarization/QA.
  At our 5-layer scope with a 25% budget (4096/16384), expected degradation is <2% on
  most benchmarks.
- **RoPE is safe:** absolute-position RoPE on retained tokens is mathematically
  consistent regardless of which tokens are evicted (confirmed in semantic_kv_eviction.md).
- **Attention sinks must be pinned** (first 16 tokens, confirmed by StreamingLLM).
- **RULER/InfiniteBench risk:** tasks requiring retrieval of a specific fact from an
  evicted position will fail. H2O does not guarantee the needle is retained. Projected
  RULER score degradation at 50% KV budget: ~5-8% per literature. At 25% budget: 2-4%.

**Verdict:** Medium correctness risk, concentrated in needle-in-haystack retrieval.
Acceptable for throughput-optimized serving; unacceptable for precision RAG applications
without benchmark-specific tuning of the eviction budget.

---

## 6. Triton Kernel Integration

**`kernels/triton/swa_decode.py` is not the right integration point for H2O.**

The SWA kernel's sparsity is geometric (page-level table truncation at `start_page`).
Extending it to adaptive top-k would require:
1. A pre-pass sorting/ranking step to build a non-contiguous block list.
2. Replacing the contiguous `for page_idx in range(page_start, page_end)` loop with a
   gather over an index list (`selected_pages[i]`).
3. Passing an additional `selected_pages` tensor per batch entry.

**Triton feasibility:** The gather-style loop is expressible in Triton using an index
buffer; `phys_block = tl.load(Selected_pages_ptr + cur_batch*stride_sp + local_idx)`
instead of the current page table lookup. This is ~30-40 LoC change to stage 1.

**But this is only correct for global layers.** SWA kernel targets sliding layers
(`window=4096`, head_dim=128). Global layers use `head_dim=256` and are currently
handled by FlashInfer's full-attention decode path. The adaptive gather kernel should be
a **new kernel** (`h2o_decode.py`) targeting global-layer shape (head_dim=256, GQA
group=4, no geometric window), not a fork of `swa_decode.py`.

**Integration estimate:**
- H2O score accumulation hook in global-layer attention: 1 day
- `h2o_decode.py` gather-style Triton kernel (new, based on `swa_decode.py` structure): 2 days
- vLLM block eviction plumbing (re-uses FusenEvict design): 1 day
- Correctness validation + RULER/InfiniteBench eval: 1 day
- **Total: 5 engineering days**

The SWA kernel itself requires no modification.

---

## 7. Verdict

**DEFER.**

**Rationale:**
1. Breakpoint at S≈8-10K — below that, overhead eats the gain. Current production
   benchmark (S=12K, 2.52×) is just above breakpoint; marginal uplift of ~4% at that
   operating point is within measurement noise.
2. Maximum realistic compound uplift at S≤16K is +4-5% additive over existing 2.52×
   (→ ~2.62×). This does not justify 5 engineering days at the current roadmap cadence.
3. The scheme becomes genuinely worthwhile at S≥32K where it projects +13% lift (→
   ~2.85×), but 32K+ seq_len workloads are not the current benchmark target.
4. Correctness risk on RULER-style benchmarks is non-trivial and would require a
   validation suite not yet in the repo.
5. FusenEvict (semantic_kv_eviction.md) already captures most of the design; if 32K+
   workloads become the target, merge the two designs (H2O scoring + FusenEvict eviction
   engine) rather than building a separate adaptive-sparse kernel.

**Revisit trigger:** If production seq_len target moves to ≥32K or a RULER regression
budget of ≤3% is accepted, upgrade to PROCEED with the H2O-on-global-layers only design.
