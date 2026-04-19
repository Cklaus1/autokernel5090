# Mega-Graph v7: Real Qwen3/Gemma4 MoE + FP4 Integration (Design Only)

**Tag:** `W5_D1_mega_graph_v7_design`
**Date:** 2026-04-18
**Status:** Design spec only. No `.cu` modifications. Framed as hypothesis per `plans/KILL_PATTERNS.md` §P11.
**Predecessors:** `kernels/csrc/mega_graph_gemma4_30layer_v5a.cu` (dense BF16 baseline, 1.12× vs eager), `kernels/csrc/persistent_moe_dispatch.cu` (routing primitive, atomicAdd race at lines 376–383), `plans/persistent_moe_fp4_feasibility.md` (prior DEFER, now recalibrated).

---

## 0. Recalibration vs. Prior DEFER

The 2026-04-18 DEFER verdict in `plans/persistent_moe_fp4_feasibility.md` used **15 µs/barrier** (copied from v5a.1 folklore). The 2026-04-19 v5b audit established the real cost as **~3 µs/barrier** on SM120a cooperative kernels (`plans/KILL_PATTERNS.md` §1). Re-running the arithmetic with the correct constant:

- v5a dense: 5 barriers/layer × 30 layers × 3 µs = **450 µs** (not 2,250 µs).
- v7 projected: 7 barriers/layer × 30 × 3 µs = **630 µs** (not 2,700 µs).
- Δ over v5a: **+180 µs**, not +2,700 µs.

Combined with the observation that the dense H=2048 proxy saturates only **26%** of HBM (`plans/KILL_PATTERNS.md` §1), the key insight changes: real Qwen3-30B-A3B MoE at H=2048 × 128 experts × top_k=8 moves **~24× more weight bytes per layer** than the dense proxy (8 active experts × 3 projections × ~3 MB ≈ 72 MB/layer vs 3 MB/layer dense). That regime is HBM-bound, and FP4 weight compression finally has something to compress. The v6 KILL (0.50×) was a dense-proxy artifact — P11 Category 2 (recalibration) + Category 3 (cross-apply from dense to sparse MoE). Treat v7 as **hypothesis**, not prediction.

---

## 1. Architecture Summary

v7 extends the v5a cooperative kernel with: (a) real MoE routing executed **inside** the persistent kernel between attention and expert GEMMs; (b) FP4 expert weights loaded inline using the Option A dequant-then-WMMA pattern from `plans/mega_graph_v4b_fp4_spec.md`. The 188-SM cooperative kernel uses **expert-grid dispatch** rather than v5a's contiguous row-stripe: per layer, after `grid.sync()` [2/7] delivers `expert_offsets` and `a_map`, SMs re-bind to (expert × N-tile) pairs so token-sorted activations and expert-owned FP4 weights flow to the same set of warps. Attention stays BF16 per the v4b checkpoint-format audit (attention weights are dequantized on disk; see `plans/mega_graph_v4b_checkpoint_format.md` §1, §10).

Per-layer phase order (7 barriers):

1. **QKV + RMSNorm fused** — v5a pattern preserved (`mega_graph_gemma4_30layer_v5a.cu:281–359`). `grid.sync()` [1/7].
2. **Attention core (BF16)** — v5a WMMA path (`:385–545`). `grid.sync()` [2/7].
3. **O-proj + residual + post-attn RMSNorm + Router GEMM** — fused into one stripe-owned stage. Router is a BF16 linear [H, E=128] producing logits → top_k=8 (sorting done in registers + warp-shuffle, no extra barrier). This stage also emits `expert_counts[E]` via `atomicAdd` and writes `a_map` partially (per-SM slices). `grid.sync()` [3/7].
4. **Prefix-sum + scatter (routing settle)** — SM 0 runs the E=128 prefix sum as in `persistent_moe_dispatch.cu:139–152`. All SMs then scatter using `phase1c_scatter` pattern (`:155–191`). Unlike the standalone kernel, scatter writes **BF16** sorted activations (not FP4) — per-expert FP4 quantization happens inside the fused GEMM1 prologue so the global activation scale can piggyback on router weights already in registers. `grid.sync()` [4/7].
5. **Expert GEMM1 (gate+up, FP4→BF16 dequant → WMMA)** — each SM binds to (expert e, N-tile n) pairs drawn from a work list of size `E × INTER_DIM/N_WARP = 128 × (hidden_dim_per_expert/16)`. Tokens per expert vary (ragged). At M=1 decode with top_k=8, exactly 8 experts are active; each active expert has exactly 1 routed token. Warps within an SM cooperate on the K-loop for that expert's slice using the Option A inner loop from `plans/mega_graph_v4b_fp4_spec.md` §4 Phase 2. Inactive experts short-circuit (length-0 `a_map` slice). `grid.sync()` [5/7].
6. **SiLU/GELU + Expert GEMM2 (down, FP4→BF16 dequant → WMMA)** — fused activation written into the BF16 dequant staging buffer; GEMM2 reuses the same expert work-list. `grid.sync()` [6/7].
7. **Weighted combine + residual** — tokens combine from sorted space back to original order, applying `topk_weights`. Uses a **float accumulator workspace** (fixes the atomicAdd race at `persistent_moe_dispatch.cu:376–383`); final cast to BF16 before write to `hidden`. `grid.sync()` [7/7].

The router MLP lives inside stage 3 (before scatter) specifically so that it can share the post-attention RMSNorm smem view and reuse the small-GEMV tile path that v5a already ships (`:550–608`, the O-proj residual stage). No new smem shape is required for the router.

---

## 2. MoE Routing Integration Details

### 2.1 Router placement

Router weights are **BF16** `[H=2048, E=128]` — 512 KB total. They fit in the L2 cache (128 MB on SM120a) and are recomputed per-SM redundantly with zero HBM cost after the first layer, mirroring v5a's RMSNorm redundancy pattern (`:215–229`). Top_k=8 selection uses a warp-parallel top-k on the 128 logits: 4 warps × 32 lanes cover all 128 lanes; each lane holds one logit; warp-shuffle reductions yield top-8 in 3 reductions. No cross-warp sync needed (log2(128)=7, all within one warp via 32-lane shuffle + cooperative masking).

### 2.2 Reuse vs. re-implement the dispatch

**Reuse the algorithmic pattern from `persistent_moe_dispatch.cu`, re-implement inline.** The existing file is a standalone kernel with its own cooperative launch; v7 needs it as a *phase* within the mega-graph. Copy these device functions verbatim into the v7 translation unit with only the signature adapted:

- `phase1_route_and_shuffle` → stage-3b (emit expert_counts). Source: `:111–136`.
- `phase1b_prefix_sum` → stage-4a (SM 0, E=128 scalar scan). Source: `:139–152`.
- `phase1c_scatter` → stage-4b (BF16 sorted activations). Source: `:155–191`.
- `phase6_unshuffle` → stage-7 **with the race fix**: replace the BF16 RMW at `:378–382` with an FP32 atomic workspace (`atomicAdd` on `float*`) followed by a final BF16 cast pass after `grid.sync()` [7/7].

### 2.3 Expert dispatch across SMs (the Amdahl problem)

At M=1 decode, top_k=8 ⇒ only 8 experts active ⇒ only 8 unique weight-stripes matter. With 188 SMs available, the naive "one SM per active expert" leaves 180 SMs idle. Mitigation: **expert × N-tile sharding**. Each expert's GEMM is decomposed into `INTER_DIM_PER_EXPERT / N_WARP` tiles (for Qwen3-30B-A3B, `INTER_DIM_PER_EXPERT ≈ 768` per expert, N_WARP=16 ⇒ 48 N-tiles per expert). 8 active experts × 48 N-tiles = 384 work items. Distribute across 188 SMs × 8 warps = 1504 warps → **~4 warps per work item**. Each work item reduces to 1 WMMA instruction with a K=2048 inner loop (128 K-steps). Utilization goes from 4.3% (per-expert only) to ~25% (expert × N-tile).

This is the same Amdahl structure as `plans/persistent_moe_fp4_feasibility.md` §5 but the HBM-bound regime masks the utilization gap: the 180 "idle" SMs are already idle at M=1 dense too (the dense proxy only had enough work for ~25% of SMs). The test is whether HBM bandwidth — not SM utilization — dominates; that is the hypothesis being validated.

### 2.4 Ragged dispatch robustness

If routing variance produces one expert with many tokens (M>1 future), the N-tile decomposition degenerates gracefully: work items become (expert, M-tile, N-tile) triples. For M=1 this stays flat. No ragged-dispatch gymnastics required for the single-user decode target.

---

## 3. FP4 Expert Weight Integration

### 3.1 Checkpoint format (already characterized)

Per `plans/mega_graph_v4b_checkpoint_format.md` §3–7:

- Per expert per projection: `.weight` (uint8, `[N, K/2]`), `.weight_scale` (fp8_e4m3fn, `[N, K/16]`), `.weight_scale_2` (fp32 scalar). Layout is row-major on disk (not CUTLASS swizzled).
- For Qwen3-30B-A3B: 128 experts × (gate, up, down) × 30 layers. Each matrix is small (~768 × 2048 at ~0.5 MB FP4 each).
- **No pre-swizzling required** for the cooperative path. Each SM accesses its assigned (expert, row-stripe) in row-major order — the `fused_shuffle_quant.cu:72` swizzle formula is CUTLASS-specific and **not** needed for v7. (Retain it documented in the FP4 spec for cross-checking correctness against `cutlass_fp4_moe_mm` reference output.)

### 3.2 Inline dequant (Option A)

Dequant formula per element (exact as `plans/mega_graph_v4b_fp4_spec.md` §4):

```
dequant = FP4_E2M1_LUT[nibble] * fp8_decode(weight_scale[row, blk]) * weight_scale_2
```

Two 16-element FP4 blocks fit in 16 bytes (one 32-bit word pair). Fetch → unpack via bit ops → LUT → FP32 mul by fp8-decoded per-block scale → FP32 mul by fp32 global scale → `__float2bfloat16`. Write to the 16×16 BF16 B-tile in smem. Feed to `wmma::load_matrix_sync`.

**Why dequant-then-WMMA is mandatory**: SM120a has **no native FP4 MMA** (`plans/KILL_PATTERNS.md` §1) — `mma.sync.e2m1` is unsupported by ptxas for sm_120. The native-FP4 hardware path is closed until at least CUDA 13.x + a future consumer Blackwell SKU.

### 3.3 Why v6 KILLed and v7 should not

The v6 FP4 KILL (0.50× regression) happened on the **dense H=2048 single-expert proxy** which saturates only 26% of HBM. In that regime the dequant smem roundtrip adds ~15 cycles per WMMA step to save ~0 µs of HBM wait (there was no wait to hide). For real MoE:

- HBM weight traffic per layer: 8 experts × 3 projections × 0.5 MB ≈ **12 MB/layer (FP4)** vs. 48 MB (BF16).
- At 1792 GB/s peak × 60% realized = 1075 GB/s effective, that's **11.2 µs/layer** FP4 vs 44.6 µs BF16.
- Over 30 layers: ~336 µs FP4, ~1,340 µs BF16 weight-read-only. The ~1 ms FP4 savings dwarfs the dequant overhead (estimated <200 µs across 30 layers from `plans/mega_graph_v4b_fp4_spec.md` §7).

This is explicitly the regime where the v5a→v6 KILL does **not** transfer. P11 Category 3 — treat as hypothesis.

### 3.4 Scale decode micro-detail

`fp8_e4m3fn` decode is one `__nv_fp8_e4m3::__x_to_float` call (or equivalent ~6 cycles of shifts+add for the exponent rebias). Per block (16 elements) that's 6 cycles amortized over 16 elements = 0.375 cycles per element — negligible vs 16 WMMA cycles per 16×16×16 step.

---

## 4. Barrier Budget

| Phase | Barrier | Cost (3 µs each) |
|---|---|---|
| 1. QKV + RMSNorm fused | [1/7] after QKV | 3 µs |
| 2. Attention core (BF16) | [2/7] after attn | 3 µs |
| 3. O-proj + post-norm + Router + topk + partial scatter | [3/7] after routing emit | 3 µs |
| 4. Prefix sum + full scatter | [4/7] after scatter | 3 µs |
| 5. Expert GEMM1 (FP4→BF16→WMMA) | [5/7] after GEMM1 | 3 µs |
| 6. SiLU + Expert GEMM2 (FP4→BF16→WMMA) | [6/7] after GEMM2 | 3 µs |
| 7. Weighted combine + residual | [7/7] after combine | 3 µs |
| **Per layer** | 7 × 3 = **21 µs** | |
| **30 layers** | | **630 µs** |

vs. v5a: 5 barriers × 30 × 3 µs = 450 µs. **+180 µs** over v5a. This is 1.8% of the projected 10 ms total — trivial.

The 4/7 barrier (post-prefix-sum) can potentially be elided by using atomicAdd-based counters (as `persistent_moe_dispatch.cu` does with `expert_counts` reset and reused) so that scatter and prefix sum overlap across experts. That's a v7.1 optimization — out of scope here.

---

## 5. Smem Budget (228 KB/SM cap)

| Region | v5a | v7 delta | v7 total |
|---|---|---|---|
| SMEM_X activation staging (aliased w/ Q16/P16) | 16,384 B | 0 | 16,384 B |
| SMEM_A WMMA A-tile pool (8 warps × 512 B) | 4,096 B | 0 | 4,096 B |
| SMEM_C WMMA C-scratch (8 warps × 1 KB) | 8,192 B | 0 | 8,192 B |
| SMEM_RED block-reduce | 128 B | 0 | 128 B |
| SMEM_INV bookkeeping | 64 B | 0 | 64 B |
| FP4 B-tile staging (Option A, 16-row × 8 B) | — | +128 B | 128 B |
| FP8 scale staging (16 rows × 1 B) | — | +16 B | 16 B |
| BF16 dequant output tile (union with SMEM_C) | — | +0 (aliased) | 0 |
| Expert routing tables (expert_counts[128] + offsets[129]) | — | +1,028 B | 1,028 B |
| a_map slice (M×top_k×4 B, M=1, top_k=8) | — | +32 B | 32 B |
| Router logits (128 fp32) | — | +512 B | 512 B |
| Top-k scratch (8 pairs of {idx, weight}) | — | +64 B | 64 B |
| FP32 combine accumulator (HIDDEN × 4 B, alias with SMEM_X post-scatter) | — | +0 (aliased) | 0 |
| **Total** | **~29 KB** | **+1.8 KB** | **~31 KB** |

The 5 KB FP4 tile estimate in the original prompt was conservative; actual staging is <200 B per active tile because each K-step processes only 16×16 elements. With ping-pong double-buffering (v4a pattern), add another ~300 B for the second FP4 stage. Final estimate: **~32 KB**, well under the 228 KB cap and under the ~100 KB practical cooperative limit.

Budget PASSES with >6× headroom.

---

## 6. Projected Performance (Hypothesis per P11)

Inputs (best available estimates):
- Qwen3-30B-A3B active params per token: ~3 B (8 of 128 experts × ~375 M each).
- FP4 expert weights read per layer: ~12 MB. Over 30 layers: ~360 MB.
- HBM peak: 1792 GB/s. Realized ceiling on cooperative kernels with mixed workload: ~60% = 1075 GB/s.
- Attention BF16 KV-cache traffic (seq=256, head_dim=128, 16 heads, 2 for K+V): ~2 MB/layer × 30 = 60 MB.
- Router BF16: 0.5 MB × 30 = 15 MB.
- Residual reads/writes: ~0.25 MB/layer × 30 = 8 MB.

HBM lower bound: `(360 + 60 + 15 + 8) MB / 1075 GB/s = 412 µs`. Add barrier overhead (630 µs), SM compute (WMMA + dequant + attention softmax estimated at ~2 ms), kernel launch + graph capture (~100 µs), stream-ordered residuals (~100 µs). **Lower-bound total: ~3.2 ms**.

Realistic (factoring HBM-under-realized + unmodelled latency + WMMA stalls that don't fully hide): **5–7 ms/step**.

Eager Qwen3-30B-A3B NVFP4 at M=1 decode: ~12 ms/step (prompt estimate; should re-measure on this silicon before banking).

**Hypothesis range: 1.7×–2.4× over eager.** Treat as upper-bound projection per P11 Category 3. The actual number could be anywhere from 0.8× (if WMMA dequant serialization re-bites as in v6) to 2.5× (if HBM genuinely dominates and the bound is tight).

---

## 7. Integration Surface

### 7.1 vLLM hook point

Monkey-patch the Qwen3MoE block forward — specifically `Qwen3MoeSparseMoeBlock.forward` in vLLM. Symbol path: `vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock` (cite for implementation; path verified via `plans/t2n_ceiling_analysis.md` and `plans/moe_shuffle_fusion_analysis.md`). The patched forward:

1. Check `M ≤ small-batch threshold` (default 2; configurable). Otherwise fall through to `cutlass_fp4_moe_mm` path — same mechanism as `patches/fused_shuffle_quant_wrapper.py:444–463`.
2. Check `os.environ.get("AUTOKERNEL_MEGAGRAPH_V7", "0") == "1"`.
3. Check `hasattr(self.experts.quant_method, "backend")` per P1 in `plans/KILL_PATTERNS.md` §2 (silent-None guard).
4. Check attention block uses BF16 (not NVFP4 attention; per v4b audit).
5. On all checks pass: launch v7 mega-graph via `cudaLaunchCooperativeKernel` with shared pointer state (routing tables, workspaces) carried in a persistent buffer allocated at patch-install time.

### 7.2 Fall-through semantics

Any check failure → call original `Qwen3MoeSparseMoeBlock.forward`. Log on first fallthrough with class name + reason (P1 mitigation idiom from `plans/KILL_PATTERNS.md` §2).

### 7.3 CUDA graph capture

v7 runs as a single cooperative launch per decode step — replay-safe by construction (no intra-graph stream sync, one DMA). Aligns with the mega-graph thesis in `plans/mega_graph_cooperative_kernel.md` §1.3.

### 7.4 Gating env vars

- `AUTOKERNEL_MEGAGRAPH_V7=1` — master switch.
- `AUTOKERNEL_MEGAGRAPH_V7_M_MAX=2` — batch threshold.
- `AUTOKERNEL_MEGAGRAPH_V7_DEBUG=1` — enable per-phase `cudaEvent` recording for profiler.

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Dequant serialization regresses like v6** | HIGH | Run a real-MoE microbench (1 layer, 8 active experts, FP4 weights) BEFORE full 30-layer wiring. Measure HBM realized BW with `ncu --metrics dram__throughput`. If realized BW ≥ 50% of peak on a single layer, HBM is the binding constraint and dequant hides. If <30%, the v6 pathology applies — reconsider. |
| **Expert routing variance (single expert hot)** | MED | At M=1 this cannot occur (top_k=8 means 8 experts active with 1 token each). Bound-check assertion in scatter phase that `expert_counts[e] <= top_k` at M=1. For future M>1, add ragged-dispatch grid.sync partitioning (v7.1). |
| **atomicAdd race in combine stage** | HIGH (correctness) | Use FP32 workspace + `atomicAdd(float*)`, finalize to BF16 after barrier. Explicitly replaces `persistent_moe_dispatch.cu:376–383`. Validate correctness cos ≥ 0.999 vs `cutlass_fp4_moe_mm` reference per `plans/KILL_PATTERNS.md` §3 pre-KILL checklist. |
| **WGMMA unavailable on SM120a** | Known | Use `nvcuda::wmma` 16×16×16 BF16 (v5a-proven `:437, :449, :526, :530`). Do NOT port any `cute::SM90_...` selectors. |
| **Checkpoint scale layout swizzle mismatch** | LOW | Access row-major from disk (`plans/mega_graph_v4b_checkpoint_format.md` §5). Do **not** apply the `fused_shuffle_quant.cu:90–102` swizzle in v7 — swizzle is exclusively for CUTLASS. Validate dequant output matches CUTLASS path at load time. |
| **P11 cross-apply bias** | MED | Project 5–7 ms as **hypothesis**, not prediction. Pre-commit KILL criteria (§10) before benching. Halve projected confidence on any cross-model claims. |
| **Silent plugin inactivity (P2)** | MED | First forward asserts `sum(1 for l in model.layers if l._v7_enabled) == 30`. Log count. |
| **Per-SM smem attribute raise fails** | LOW | v7 needs ~32 KB, no `cudaFuncAttributeMaxDynamicSharedMemorySize` raise required (default ~48 KB suffices). Kept for symmetry with v5a (`mega_graph_gemma4_30layer_v5a.cu:890–894`). |

---

## 9. Implementation Checklist (1-page)

Order of build. Each step has a single testable deliverable. KILL on any step failure per pre-KILL checklist (`plans/KILL_PATTERNS.md` §3).

1. **Correctness harness** — Python reference: load Qwen3-30B-A3B one-layer FP4 expert weights, drive through HF eager reference, dump golden `hidden_out` tensor. ~1 day.
2. **Standalone FP4 dequant micro-kernel** — bit-compare per-element vs Python reference on 768×2048 FP4 expert weight. Validates `FP4_E2M1_LUT` agreement and FP8 scale decode. ~1 day.
3. **Router + top-k phase** — BF16 linear `[H, E]` → top-8 indices + weights. Bit-compare vs PyTorch `torch.topk(router_logits, k=8)`. ~1 day.
4. **Routing tables + scatter (BF16-only, no FP4 yet)** — copy `phase1*` from `persistent_moe_dispatch.cu` into v7 TU. Verify `a_map`, `expert_offsets` vs reference. Fix phase-6 race (FP32 accum). ~2 days.
5. **Expert GEMM1 with inline FP4 dequant (Option A) — single layer, single expert** — run at M=1. Validate output cos ≥ 0.999 vs FP32 reference. ~2 days.
6. **Expert GEMM2 + SiLU fusion** — extend to both GEMMs, one full MoE layer. ~1 day.
7. **Stitch into 30-layer cooperative kernel** — replicate v5a's layer loop; insert MoE stages at the right barriers. End-to-end correctness (cos ≥ 0.999 vs HF eager for 30 layers, seq=256). ~2 days.
8. **vLLM monkey-patch + env-gated activation** — `patches/wire_megagraph_v7_qwen3.py`. First-forward assertion of layer count. ~1 day.
9. **Benchmark + decision** — run against eager Qwen3-30B-A3B NVFP4 at M=1, C=1. Apply §10 gate. ~1 day.

**Total: ~11 days (8 days core + buffer).**

---

## 10. Gate Criteria (Pre-Committed)

Measured vs eager Qwen3-30B-A3B NVFP4 decode at M=1, seq=256, C=1 on RTX PRO 6000 Blackwell (SM120a). Correctness: max_abs ≤ 5e-2 vs HF eager FP32 reference, cos ≥ 0.999 over 30-layer hidden.

| Verdict | Criterion | Action |
|---|---|---|
| **BIG_WIN** | ≥ 2.0× over eager | Ship; upstream patch submission; promote to default when `AUTOKERNEL_MEGAGRAPH_V7=1` |
| **PASS** | ≥ 1.5× over eager | Merge as opt-in env var; document on-disk checkpoint constraints; plan v7.1 |
| **MARGINAL** | 1.2×–1.5× | Profile HBM realized BW and dequant serialization. If HBM ≥ 70% realized, diminishing-returns; freeze v7 and pursue attention optimizations. If <50%, re-examine dequant path. |
| **KILL** | < 1.2× over eager | Apply `plans/KILL_PATTERNS.md` §3 full checklist before declaring KILL. Report which bottleneck binds (HBM, dequant, barriers, SM utilization). Re-audit P11 classification. |

---

## 11. Effort Estimate

**Engineering: 1–2 weeks (lean to 11 days per §9).** Gating risks that could extend:

- If step 5 (FP4 dequant inline) reveals register spill or WMMA fragment contention with the dequant scratch, budget +3 days for tile-shape tuning.
- If step 7 reveals unexpected HBM contention (e.g., router + attention K/V + expert weights competing for memory controllers), budget +2 days for scheduling tweaks.

Realistic range: **10–18 days**, median 13.

---

## 12. Confidence Level (P11 Classification)

**Mixed Category 2 + Category 3** per `plans/KILL_PATTERNS.md` §P11:

- **Category 2 (recalibration) component — MEDIUM-HIGH confidence:** the 15 µs → 3 µs barrier correction is a specific code-grounded fix. The prior DEFER math is invalidated; the barrier budget is clearly affordable now.
- **Category 3 (cross-apply) component — LOW-MEDIUM confidence:** projecting dense-proxy behavior onto real MoE shape. The v6 KILL explicitly failed to transfer because dense H=2048 is not HBM-bound. We are claiming real H=2048 × 128 experts × top_k=8 **is** HBM-bound. This is a load-bearing assumption that must be validated empirically at step 5 (single-expert-single-layer microbench) before banking the 30-layer plan.

**Halve projected confidence per P11 guidance on cross-model extrapolation.** Treat "5–7 ms target" as **worth testing, not expected to land**. Pre-commit to the §10 gate. Do not announce a win until the benchmark passes on this silicon.

---

## 13. Files This Design References

- `kernels/csrc/mega_graph_gemma4_30layer_v5a.cu` — baseline cooperative kernel (lines cited: 281–359 QKV, 385–545 attn, 550–608 O-proj, 618–697 MLP fused gate/up, 702–760 MLP down, 789–851 main kernel 5-barrier loop, 890–899 launcher).
- `kernels/csrc/persistent_moe_dispatch.cu` — routing primitives (lines cited: 111–191 routing phases, 337–384 unshuffle with race, 588–644 grid.sync microbench).
- `kernels/csrc/fused_shuffle_quant.cu:72–102` — CUTLASS swizzle formula (documented only; NOT applied in v7).
- `patches/fused_shuffle_quant_wrapper.py:444–463` — fall-through reference for vLLM patching pattern.
- `plans/mega_graph_v4b_checkpoint_format.md` — FP4 checkpoint layout authority.
- `plans/mega_graph_v4b_fp4_spec.md` — Option A dequant spec.
- `plans/persistent_moe_fp4_feasibility.md` — prior DEFER (superseded by recalibration above).
- `plans/mega_graph_cooperative_kernel.md` — cooperative kernel rationale.
- `plans/KILL_PATTERNS.md` §1 calibration, §2 P1/P2/P11, §3 pre-KILL checklist.

---

*End of design. No `.cu` modifications made in producing this document.*
