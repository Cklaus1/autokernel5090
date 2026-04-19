# Gemma4 T2-N Post-Mortem
**Tag:** W5_CA_gemma4_t2n_postmortem  
**Date:** 2026-04-18  
**Result:** Qwen3-30B-A3B +34% (13,994 → 18,756 tok/s @ C=512); Gemma4-26B-A4B **-38%** (1,798 → 1,113 tok/s @ C=32)

---

## 1. MoE Config Side-by-Side

| Parameter | Qwen3-30B-A3B | Gemma4-26B-A4B | Delta |
|---|---|---|---|
| `hidden_size` (K) | 2,048 | 2,816 | **+38%** |
| `moe_intermediate_size` | 768 | 704 | -8% |
| `num_experts` (E) | 128 | 128 | same |
| `top_k` | 8 | 8 | same |
| `num_hidden_layers` | 48 | 30 | -37.5% |
| MoE dtype (activation) | FP16 (from quant config) | BF16 | different kernel path |
| Bench concurrency (C) | **512** | **32** | **16x difference** |
| M_sorted (C × topk) | **4,096** | **256** | **16x difference** |
| Quant group_size | 16 | 16 | same |

### Derived kernel parameters (`fused_shuffle_quant.cu`)

| Parameter | Qwen3 | Gemma4 | Formula |
|---|---|---|---|
| `n_blocks` (K/16) | 128 | **176** | `K // 16` |
| `numKTiles` (ceil(K/64)) | 32 | **44** | `(K + 63) // 64` |
| `padded_k_int32` | 32 | **44** | `(n_blocks + 3) // 4` |
| kernel `block` dim | 128 threads | **176 threads** | `min(n_blocks, 1024)` |
| grid dim | **4,096 blocks** | 256 blocks | `M_sorted` |
| scale buffer size | 5,120 KB | **7,040 KB** | `E*320 * padded_k * 4` |
| scale buffer row util | 4096/40960 = 10% | 256/40960 = **0.6%** | actual / allocated |

---

## 2. Root-Cause Analysis

### Primary: H4 — Batch-size-dependent overhead (dominant)

**Evidence:** The bench concurrencies differ by 16×: Qwen3 at C=512 (M_sorted=4,096) vs Gemma4 at C=32 (M_sorted=256). The fused kernel has per-call fixed costs that do not scale with M_sorted:

1. **`torch.empty([40960, 44], dtype=torch.kInt32)`** — a 7,040 KB allocation per MoE call, regardless of actual token count. At C=32, only 256 of 40,960 rows are written (0.6% utilization). vLLM's native two-op path (`shuffle_rows` + `scaled_fp4_experts_quant`) allocates only what it needs.

2. **kernel launch fixed overhead** — 256-block grid with 176 threads/block. At C=32, the GPU kernel is dispatched for 45,056 threads of real work but incurs the same launch latency as any other CUDA dispatch (~3–10 µs). The two-op path at C=32 shuffles and quantizes ~1.44 MB BF16 and is complete in 2–4 µs total; our kernel + overhead is estimated at 5–20× that.

3. **Gemma4 has only 30 MoE layers** (vs Qwen3's 48), so each saved µs/call is worth less, and each added µs/call hurts 37.5% more per-unit than on Qwen3.

**Quantitative check:** Gemma4 baseline is 1,798 tok/s at C=32 → ~17.8 ms per forward (32 tokens / 1798 tok/s). If the fused kernel adds ~30 µs net overhead per layer × 30 layers = +900 µs per forward → 17.8 ms → 18.7 ms → ~1,709 tok/s → 0.95× degradation minimum. At C=32 the actual two-op path is very fast (small M), so the full 0.62× result is consistent with 30–60 µs overhead per call amortized poorly at small batch.

### Secondary: H1 — Shape mismatch (amplifies H4)

Gemma4's K=2,816 vs Qwen3's K=2,048 means:
- 38% more threads per block (176 vs 128), worse occupancy on SM120a
- 40% larger scale buffer allocation (7 MB vs 5 MB) per call
- 37.5% fewer layers means overhead penalty is less diluted across forwards

At large C (≥256), H1 alone would degrade the Gemma4 gain estimate from the Qwen3 +34% to perhaps +20–25%. But at C=32, H4 entirely dominates; H1 magnifies it.

### Ruled out:
- **H2 (swizzle offset bug):** Both models use identical E=128, topk=8. The `numKTiles` formula is correct for both; no per-expert offset arithmetic differs. Scale buffer is sized generously (40960 rows).
- **H3 (routing pattern difference):** E=128, topk=8 is identical. The linear expert-lookup scan (`O(E)` per row) is the same.
- **H5 (dtype mismatch):** The `.cu` has dedicated BF16 and FP16 kernels (`fused_shuffle_quant_bf16_kernel` / `fused_shuffle_quant_kernel`). Gemma4 uses BF16 → routes to the correct variant.

---

## 3. Is a Gemma4-Specific Variant Feasible?

**Yes, conditionally — at C≥128 only.**

The kernel itself is architecturally sound for Gemma4 (correct E, topk, swizzle formula, BF16 path). The regression is deployment-context-specific (C=32 chosen as a single-GPU decode serving baseline). Two changes make it viable:

### Fix A: Persistent scale buffer (eliminates H4's largest cost)
Move `torch.empty([40960, padded_k_int32], ...)` out of the hot path into a module-level cached slab (see `t2n_polish_findings.md` §1 "Follow-up"). Allocate once at layer init. The CUDA caching allocator already reuses same-size slabs, but the Python-side `torch.empty` + shape-check overhead is non-trivial at C=32.

This fix is already planned in `fused_shuffle_quant_wrapper.py` `_scale_buf_cache` (present but keyed to `AUTOKERNEL_FUSED_RESHAPE_SCALES=1` path only). Promote it to the default path for non-reshape mode as well.

**Effort:** 1–2 hours. **Expected benefit:** removes ~5–15 µs per call.

### Fix B: Grid-stride dispatch with a C-threshold kill switch
Add a Python-side guard:
```python
if M_sorted < FUSED_MIN_TOKENS:   # e.g. 512
    return ops.shuffle_rows(a, a_map), ops.scaled_fp4_experts_quant(...)
```
This automatically falls back to two-op at small batch (where fused has no advantage) and engages fused only where the amortization works.

**Effort:** 30 minutes. **Expected benefit:** eliminates the regression entirely at C≤64; fused path active C≥128 where it may break even or slightly win.

### Fix C: Gemma4-tuned tile (smaller grid blocks)
The 176-thread block (for K=2,816) wastes ~48 threads on SM120a's 128-thread warp boundary preference. Pad `n_blocks` to 192 (nearest multiple of 32) with zero-ops, or split rows into two passes, to fit the SM120a warp scheduler better.

**Effort:** 4–6 hours. **Projected gain vs unfused:** +15–25% at C≥128 (extrapolating from Qwen3 K=2,048 at its own C).

### Projected recovery at Gemma4 C=256 (production target)

| Scenario | Projected tok/s | vs Gemma4 baseline (6,615) |
|---|---|---|
| Current patched C=32 | 1,113 | **-38% (REGRESSION)** |
| Fix A+B only, C=32 | ~1,750–1,800 | ~0% (neutral) |
| Fix A+B+C, C=256 | ~7,200–7,800 | **+9–18%** |
| Fix A+B+C, C=512 | ~8,000–8,800 | **+21–33%** |

Full +34% replication at Gemma4 is **unlikely** due to shape overhead (K=2,816 vs 2,048 = 38% more work/row with proportionally more shuffle cost). Realistic ceiling at the best batch size is **+20–28%**.

---

## 4. KILL_PATTERNS §P11 — Category Assessment

This is a **textbook Category 3 failure** (Cross-apply PROCEED, P~0.3 realized):

- The kernel logic is correct and architecturally compatible.
- The optimization was projected from a different shape regime (C=512, K=2,048) to a different regime (C=32, K=2,816).
- No code-level bug; the regression is entirely from deployment-context mismatch.

Per §P11: "Cross-apply Category 3 projections are P~0.3 — this was predictable." The `launch_gemma4_t2n.sh` comment itself says "Projected: +34% → ~8,860 tok/s" with no caveat for batch-size differences. The correct pre-dispatch framing should have been:

> "Worth testing at Gemma4's production concurrency (C=128+). Do NOT assume +34% — Qwen3 was benched at C=512; Gemma4 at C=32 is a 16× smaller batch."

**Was this an unforeseen class or a known Category 3?** Known class. §P11 examples already included "T5.6 T2-N on Gemma4 — audit P=0.70, projected +34%. Reality: 0.62× regression. T2-N kernel tile shapes tuned for Qwen3 MoE dispatch don't transfer." The post-mortem is recorded in §P11 correctly. This experiment *confirmed* rather than *discovered* the pattern.

---

## 5. What Classes of Optimizations DO Transfer Across NVFP4 MoE Models

For cross-apply to succeed across Gemma4 and Qwen3 (same E=128, topk=8, NVFP4, BF16):

| Class | Transfer probability | Why |
|---|---|---|
| Bug fixes (wrong dtype, wrong attribute, wrong env var) | P~0.9 | Logic bug is model-independent |
| Scale-format / swizzle correctness fixes | P~0.8 | CUTLASS 128×4 formula is shared |
| Persistent-buffer allocation (amortized overhead) | P~0.7 | Python overhead is model-independent |
| Large-batch kernel optimizations (C≥256) | P~0.5 | Shape-specific but similar regime |
| Small-batch kernel tuning at specific C | P~0.2 | Highly regime-specific |
| Throughput projections from different-C benchmarks | P~0.2 | Regime mismatch is the main failure mode |

---

## 6. Recommended Next Steps

1. **Add `FUSED_MIN_TOKENS` threshold** to `fused_shuffle_quant_wrapper.py` (Fix B, 30 min).
2. **Promote persistent buffer** from opt-in reshape path to default (Fix A, 1–2 hr).
3. **Re-bench Gemma4 at C=256** with Fix A+B before concluding on Fix C viability.
4. **Do not bank Gemma4 T2-N until C=256+ bench confirms positive gain.**

---

*Cite: `plans/KILL_PATTERNS.md §P11` for cross-apply category definitions. All model config values from `/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/config.json` and `/root/models/Qwen3-30B-A3B-NVFP4/config.json`.*
