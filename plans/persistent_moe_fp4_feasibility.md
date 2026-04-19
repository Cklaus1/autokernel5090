# Persistent MoE + FP4 Expert GEMM Co-Design: Feasibility Study

**Tag:** `W2_5f_persistent_moe_feasibility`  
**Date:** 2026-04-18  
**Status:** Analysis only — CPU dry-run, no kernel execution  
**Verdict:** **DEFER**

---

## 1. Hypothesis

Combine `persistent_moe_dispatch.cu` (cooperative routing/shuffle/quant) with v6 FP4 expert GEMMs into one mega-kernel that replaces vLLM's full MoE path for shapes that fit.

---

## 2. Smem Budget

### Existing allocations

| Region | Size | Source |
|---|---|---|
| v5a mega-graph (attn + RMSNorm + MLP routing) | ~29 KB | roadmap_20260418.md §3 / v5a.1 measurement |
| v4b FP4 tile staging (Option A: fp4 + fp8 scale + bf16 dequant) | ~0.66 KB | mega_graph_v4b_fp4_spec.md §6 (128+16+512 B per tile) |
| `persistent_moe_dispatch.cu` shared memory | **0 B** | cudaLaunchCooperativeKernel call, line 579 — `0, // shared mem` |
| Routing tables: `expert_counts[E]` + `expert_offsets[E+1]` = 129 × int32 × 2 | ~1.0 KB | 258 × 4 B |
| `a_map[M*top_k]` at B=128, top_k=8 → 1024 × 4 B | 4 KB | scales with batch |
| FP4 expert weight tile (16 rows × K/2 bytes, K=4096): 16 × 2048 | 32 KB | one tile per active GEMM step |
| BF16 dequant output tile (16 × K × 2 B, K=4096): 16 × 4096 × 2 | 128 KB | full tile in smem for WMMA |

**Key finding:** The FP4 expert weight tile for real Gemma4 shapes (K=4096 hidden dim) inflates the dequant buffer to **128 KB per tile** if we use the same 16-row WMMA tile pattern as v4b. That alone consumes 56% of the 228 KB SM cap, leaving only ~100 KB for all other regions — tight and requiring careful union-overlay analysis. The v4b spec used K=16 WMMA step tiles (128 B + 512 B = 640 B per tile); at K=4096 these are 256 outer tiles deep, each needing the same 640 B staging. Total weight staging across all 256 K-tiles if double-buffered: ~330 KB — exceeds cap.

**Practical smem estimate (per SM, one active expert GEMM tile at a time):**

| Region | Size |
|---|---|
| v5a base (attn + norm + dispatch bookkeeping) | 29 KB |
| Routing tables + a_map | ~5 KB |
| FP4 B-tile staging (16 × 16 × 0.5 B = 128 B, double-buffered) | 0.25 KB |
| FP8 scale staging (double-buffered) | 0.03 KB |
| BF16 dequant output tile (16 × 16 × 2 B = 512 B) | 0.5 KB |
| **Estimated total (tile-at-a-time, K-loop style)** | **~35 KB** |

**Smem verdict: FITS** — 35 KB << 228 KB cap, well under the ~100 KB practical cooperative limit. The key is that per-WMMA-step tiles are 16×16 (K-step = 16), not full K=4096 tiles. Budget passes.

---

## 3. Grid.sync Cost Re-Evaluation

- Isolated microbench (Discovery #35): ~0.82 µs / barrier
- Empirical cost with real compute load (v5a.1 measurement): **~15 µs / barrier**
- Current mega-graph uses ~5 barriers/layer × 30 layers = 150 barriers/step = **~2,250 µs = 25% of step time**

A persistent MoE kernel inside the mega-graph loop would add barriers:
- Phase 1 route+shuffle: 3 grid.syncs (count, prefix-sum, scatter)
- Phase 2 FP4 quant: 1 grid.sync
- [External GEMM1]
- Phase 4 SiLU+requant: 1 grid.sync
- [External GEMM2]
- Phase 6 unshuffle: 1 grid.sync (implied)

Total: **6 additional barriers per MoE layer × 30 layers = 180 new barriers = ~2,700 µs overhead**, which is **larger than the entire current step budget gain from v5b** (estimated -900 µs). This is disqualifying on its own.

**atomicAdd race in phase 6:** Confirmed in `persistent_moe_dispatch.cu` line 376-383 — weighted accumulate has a BF16 read-modify-write race when multiple sorted positions map to the same token. The comment acknowledges this as a V1 correctness hole that V2 must fix with atomic float workspace. This must be resolved before any production use, independent of the FP4 co-design question.

---

## 4. Integration with Mega-Graph

The mega-graph assigns each SM a **contiguous row-stripe** of the hidden dimension. MoE dispatch requires **all SMs to collectively route tokens across all experts** — tokens for expert E need to be gathered from across all SMs' row-stripes. These are incompatible partitioning strategies:

- Mega-graph: SM-local hidden-dim stripe ownership
- Persistent MoE dispatch: token-level scatter/gather that crosses SM boundaries

Integrating MoE as a phase inside the mega-graph cooperative kernel would require a full ownership handoff at each MoE layer (all SMs abandon their stripe, participate in token scatter, then resume stripe ownership). The grid.sync cost of this handoff makes it prohibitive — see §3 above.

---

## 5. Expert-Batch Size and Amdahl Cliff

### Model configurations

| Model | Experts | top_k | Active experts/token | Active SMs at B=1 |
|---|---|---|---|---|
| Gemma4 26B-A4B | 128 | 8 | 8 | 8 of 188 = 4.3% |
| Qwen3-30B-A3B | 128 | 8 | 8 | 8 of 188 = 4.3% |

At batch B=128: 1,024 routed positions → 128 × top_k=8 / 128 experts ≈ 8 tokens/expert. Only 8 experts active per token means **only 8 SMs have useful GEMM work** if we assign one expert per SM. The remaining 180 SMs spin at the grid.sync barrier.

This is the same Amdahl cliff identified in v5a.1's multi-head pack failure, but worse: multi-head had 8 heads active across 188 SMs (4.3% utilization); MoE dispatch is identical. Even with per-expert × per-N-tile work decomposition (splitting each expert GEMM across N tiles distributed to multiple SMs), the routing tables and expert_offsets must still be broadcast globally before GEMM dispatch can start — adding a mandatory non-parallelizable prefix.

**Amdahl estimate:** With 8 active experts and tile-parallelism across P=4 SMs per expert: 32 SMs useful, 156 idle = 17% utilization ceiling. At a decode step time of ~9,887 µs (v3 baseline), MoE GEMMs are roughly 35% of step time (~3,460 µs). 17% SM utilization on a ~15 µs barrier × 6 barriers = 90 µs per layer × 30 = 2,700 µs pure barrier overhead vs. at most 3,460 µs of GEMM compute. Net: negative.

---

## 6. Projected Lift vs. v6 Standalone

v6 (standalone FP4 expert GEMMs via CUTLASS, as specified in the roadmap): projected **1.6–2.2× on expert HBM traffic**, translating to roughly **1.3–1.6× on MoE layer wall time** (assuming compute becomes the bottleneck after BW compression).

Persistent MoE co-design ON TOP of v6:
- The dispatch scaffolding (route+shuffle+quant+unshuffle) currently costs ~8–10 µs per MoE layer at B=128 (measured in `moe_shuffle_fusion_analysis.md`).
- Total non-GEMM dispatch cost: 10 µs × 30 layers = 300 µs of the ~9,887 µs step.
- Even eliminating 100% of dispatch overhead is **3% of step time**.
- The merge adds 6 grid.syncs/layer × 30 layers × 15 µs = 2,700 µs of new overhead.
- **Net delta: -3% gain, +27% regression = approximately 1.24× slowdown over v6 standalone.**

**Relationship is subtractive, not additive or multiplicative.** The persistent MoE wrapper degrades v6.

---

## 7. Integration Effort Estimate

Even setting aside the negative net delta:

- Fix phase 6 atomicAdd race (requires float workspace or token-level partitioning): 1–2 days
- Thread the FP4 weight tile loads and dequant inline into persistent dispatch phases: 3–4 days
- Reconcile row-stripe vs. token-scatter partitioning: 2–3 days (may be architecturally infeasible)
- Retune grid.sync barrier placement to avoid cliff: 2–3 days
- Correctness validation and numerical comparison vs. CUTLASS MoE path: 2 days

**Estimated total: 10–14 days** on top of the 5–7 day v6 base — roughly doubling v6 integration cost for a negative-expected-value outcome.

---

## 8. Verdict: DEFER

**Do not pursue for v6.**

| Question | Answer |
|---|---|
| Smem budget fits? | YES (35 KB << 228 KB cap) |
| Grid.sync overhead acceptable? | NO (6 new barriers/layer = +2,700 µs; dwarfs non-GEMM dispatch savings of 300 µs) |
| Integrates with mega-graph per-SM stripe? | NO (incompatible partitioning; requires expensive ownership handoff) |
| Amdahl cliff manageable? | NO (8 active experts / 188 SMs = 4.3% utilization; tile-sharding buys back only ~17%) |
| Additive lift over v6 standalone? | NO (net slowdown ~1.24× relative to v6 without persistent wrapper) |
| Phase 6 race condition fixed? | NO (flagged in code as V2 work) |

**Conditions for revisiting (DEFER not KILL permanently):**
1. v5b lands, reducing barrier count from 5→3 per layer — freeing bandwidth budget for 3–4 additional MoE-specific syncs.
2. Workload shifts to batch B ≥ 512 where per-expert token counts reach 32+ — Amdahl utilization rises to ~30–40%.
3. The phase 6 atomicAdd race is fixed independently (low-cost hygiene).

At B=512, 4,096 routed positions / 128 experts = 32 tokens/expert, utilization with 4 SMs/expert = 16/188 = 8.5% — still poor but improving. Revisit at B≥1024 (82 tokens/expert) where utilization with N-tile sharding reaches ~30–50%. Tag for v7 or later.

**Standalone deliverable from this analysis:** The `persistent_moe_dispatch.cu` phase 6 atomicAdd race should be patched regardless (it is a correctness bug for V1 testing). File: `kernels/csrc/persistent_moe_dispatch.cu`, lines 363–383 — replace BF16 read-modify-write with float workspace + atomic accumulate.
