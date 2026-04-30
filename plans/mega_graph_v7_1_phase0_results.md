# Mega-Graph v7.1 Phase 0 Results — K-gang Dequant Decision Gate

**Tag:** `W7_v7_1_k_gang_dequant_phase0`
**Date:** 2026-04-18
**Status:** **KILL — Do not proceed to Phase 1.** Pivot to other tracks.
**GPU:** RTX PRO 6000 Blackwell (SM120a), GPU index 1.
**Files:**
- Kernel: `kernels/csrc/mega_graph_v7_1_phase0_microbench.cu`
- Build:  `kernels/csrc/build_mega_graph_v7_1_phase0.py`
- Driver: `kernels/csrc/test_mega_graph_v7_1_phase0.py`

---

## 1. Measured Outcome

| Metric | v7 Phase 0 | **v7.1 Phase 0 (this run)** |
|---|---|---|
| Correctness (cos vs BF16 ref) | 1.0000 | **1.0000 per expert** — PASS |
| max_abs vs reference | 4.77e-7 | **1.19e-7** (numerically cleaner than v7) |
| Wall-time per MoE iter (gate+up + down) | 254 µs | **154.6 µs** |
| Weights read per iter | 21.23 MB | 21.23 MB |
| Effective HBM throughput | 83.6 GB/s (4.7%) | **137.4 GB/s (7.7%)** |
| Speed-up vs v7 Phase 0 | — | **1.64×** |
| **Decision** | KILL | **KILL** (7.7% < 15% MARGINAL gate) |

## 2. K-gang design summary (what was implemented)

Per-block work item = one `(expert, N-tile=16)` pair. All 8 warps in the block
cooperatively dequant a `[16 N, 128 K]` BF16 block in shared memory per outer
K-step, then each warp processes one 16-K sub-WMMA. After all K-gangs are
done, cross-warp reduction across 8 partial accumulators produces the 16-element
output.

- Smem pool: 17.4 KB (x:4 KB, per-warp a:4 KB, b_gate:4 KB, b_up:4 KB,
  reduction:1 KB). Well under 228 KB SM cap.
- Registers: 85/thread, 1 barrier, 8 B spill — healthy.
- Correctness: cos=1.0000 on all 8 experts, max_abs=1.19e-7 (better than v7's
  4.77e-7, likely because BF16 accumulation is done after cross-warp reduction
  at FP32 fidelity vs v7's per-warp FP32 accumulator).

**Cache-line amortization delivered:** the 1.64× speedup over v7 matches the
design's low-end expectation (prompt projected 1.6× improvement based on
cache-line utilization rising from 6% → ~10-12%, below the 50-100% hoped-for
best case).

## 3. Root cause — why 7.7% still KILLs

The prompt projected ~28% HBM at 50% cache-line utilization. Actual 7.7% =
~2× the v7 ceiling (~4.7%), which corresponds to cache-line utilization
roughly doubling from 6% to ~12%, not rising to 50%.

Two mechanical explanations, both consistent with the measurement:

**(a) Per N-row cache-line coverage is still sparse.** With KGANG_K=128 K
elements = 64 FP4 bytes per N-row, a single 128-B L1 cache line is at most
50% filled by the in-gang fetch. The next gang loads a DIFFERENT 128 K window
(128 K-elements later), which maps to the NEXT 64-byte half of the same
cache line... **only if the L1 retains the line across the two gangs**. In
practice, 16 N-rows × (4 KB b_gate + 4 KB b_up smem churn) per gang × grid-
wide concurrency evicts L1 cache lines before the paired gang returns. The
cache does not amortize at the steady-state grid scale.

**(b) Per-block, not per-warp, wastes the 8× cooperation benefit.** A single
block now touches 16 N-rows × (HIDDEN/KGANG_K = 16 gangs) × 2 projections =
512 line-touches per tile. Spread across 188 SMs × 1 block/SM × 1 tile-stride
work, the total HBM demand is comparable to v7's per-warp scheme, but the
coalesced 4-byte-per-thread 256-thread global read at each gang does NOT
amortize across gangs because each gang moves K-offset by +128 = +64 bytes
on the same N-rows. Those bytes are the OTHER HALF of the same cache line,
so in theory free — but L1 eviction under concurrent pressure defeats this.

The net effect: 2× the v7 effective BW, not 6×.

## 4. Pre-KILL Checklist (per `plans/KILL_PATTERNS.md` §3)

- [x] **Clean GPU state:** GPU 1 shows 0 MiB used, no peer compute apps before+after.
- [x] **Single-container isolation:** no docker/vllm concurrent on GPU 1.
- [x] **Correctness pass:** cos=1.0000 all 8 experts, max_abs=1.19e-7 (well under 1e-3 gate).
- [x] **Warm run:** 3 warmup launches discarded, 50×8=400 iters benched.
- [x] **Calibration constants:** `grid.sync` cost ~3 µs/barrier (§1); 2 barriers per iter = 6 µs, negligible of 154.6 µs (4%).
- [x] **Shape regime named:** Qwen3 HIDDEN=2048, INTER=768, TOP_K=8, M=1, SM120a 188 SMs.
- [x] **Specific failure mode identified:** K-gang cache-line amortization delivers ~2× not ~6×, because per-gang K-stride of 64 B and concurrent 16-N-row churn defeats L1 retention of adjacent cache-line halves.
- [x] **Pattern sweep:** P11 Category 3b applied — v7 Phase 0 already proved the assumption could fail; v7.1 halved-P prediction (0.10-0.15) was borne out.

All pre-KILL gates satisfied. KILL is valid.

## 5. P11 confidence recalibration

**Before dispatch:** P(v7.1 lands PASS at ≥28%) was halved from the design
hypothesis's ~0.3 to **~0.15** per `plans/mega_graph_v7_phase0_results.md §9`
(P11 Cat 3b, regime/layout cross-apply).

**After this result:**
- Actual outcome (7.7% KILL) is worse than the P~0.15 PASS hypothesis and
  worse than the P~0.50 MARGINAL (15-28%) hypothesis. The K-gang reduced
  cache-line waste but the mechanical layout floor is higher than 6% — it's
  more like "how much of the second half of each cache line can L1 retain
  under concurrent pressure?" Measured answer: ~12%, giving 7.7% effective.
- **P(v7 full-layer Phase 2 lands at PASS gate) post-Phase 0: ~0.03–0.05.**
  To reach even 28% HBM, v7.1 would need ~3.6× more BW than measured, which
  would require an additional 4× cache-line utilization gain. The only
  candidate is §6.2 from the v7 results doc (offline re-layout to `[K, N/2]`),
  which trades correctness/compatibility for speed. Not recommended as a
  near-term investment.
- **Generalization:** halving projected confidence per P11 Cat 3b was
  insufficient for this case. Layout-mechanical ceilings compound across
  redesign attempts; subsequent v7.2/v7.3 proposals on the same `[N, K/2]`
  Qwen3 layout should start at P~0.05 unless a fundamentally different
  access pattern is proposed (e.g., TMA multicast, weight pre-swizzle).

## 6. Impact on v7 Phase 1 plan

**Phase 1 (cooperative kernel full-layer integration) SKIPPED per dispatch instructions.**
Phase 0 KILL triggered the "do NOT proceed to Phase 1" branch. Build
artifacts remain at `/tmp/build_mega_graph_v7_1_phase0/` for reproducibility.

## 7. Recommendation — pivot tracks

Stop investing agent-days in cooperative FP4 kernels for Qwen3 `[N, K/2]`
layout on SM120a. Higher-leverage alternatives:

1. **CUTLASS `cutlass_fp4_moe_mm`** already handles this layout with its own
   128×4 swizzle and delivers ~1.7× over scalar (per prior audit). v7's
   cooperative fusion cannot beat that without a layout change.
2. **LMCache disaggregation** (W6 track) — protocol-correctness validated,
   projected e2e gains from KV reuse independent of the kernel layout wall.
3. **Rank 2 / Rank 3 speculation tracks** — unchanged recommendation; they
   attack the autoregressive decode bottleneck rather than per-layer HBM.
4. **Attention optimization (v5a is 1.12× vs eager)** remains a cleaner
   leverage point on dense shapes.

**Offline re-layout to `[K_in, N_out/2]` (§6.2 from v7 results)** remains the
only known path to break past 30% HBM on this model, but requires a 14 GB
resave, breaks vLLM's cutlass fallback, and carries P11 Category 4
architectural-lock-in risk. Not recommended.

## 8. Discovered asymmetry — the 1.64× net win is still real

The K-gang kernel delivered 154.6 µs/iter vs v7's 254 µs/iter at bit-perfect
correctness. This is a 1.64× improvement at fixed workload. If a cooperative
kernel were ever to be shipped (e.g., as part of an unrelated v8 design), the
K-gang loader pattern is the better baseline. Preserve the code and design
notes; do not re-derive.

---

## Appendix: reproduction commands

```bash
cd /home/cklaus/projects/autokernel
python3 kernels/csrc/build_mega_graph_v7_1_phase0.py
# Pin GPU 1 (script defaults CUDA_VISIBLE_DEVICES=1):
python3 kernels/csrc/test_mega_graph_v7_1_phase0.py
```

Expected output:
```
[COR] expert_out cos per expert: ['1.0000', ...]  — PASS
[COR] mlp max_abs=7.45e-09   out max_abs=1.19e-07
[BW] per-iter: 154.59 us
[BW] effective HBM throughput: 137.4 GB/s (7.7% of 1792.0 peak)
[DECISION] KILL (7.7% < 15%) — K-gang blocked too. v7 track DEAD.
```

---

*End of v7.1 Phase 0 results. v7 track declared DEAD per dispatch rule. Phase 1 skipped.*
