# Mega-Graph v7 Phase 0 Results — Real-MoE FP4 Decision Gate

**Tag:** `W6_v7_phase0_microbench`
**Date:** 2026-04-18
**Status:** **KILL — Do not proceed to Phase 1.**
**GPU:** RTX PRO 6000 Blackwell (SM120a), GPU index 1.
**Files:**
- Kernel: `kernels/csrc/mega_graph_v7_phase0_microbench.cu`
- Build: `kernels/csrc/build_mega_graph_v7_phase0.py`
- Driver: `kernels/csrc/test_mega_graph_v7_phase0.py`

---

## 1. Measured Outcome

| Metric | Value |
|---|---|
| Correctness (cos vs BF16 PyTorch ref) | **1.0000 per expert** — PASS |
| max_abs vs reference | 4.77e-7 (numerically exact) |
| Wall-time per MoE iter (gate+up + down) | **254 µs** |
| Weights read per iter (8 experts × (2 × gate/up + down)) | 21.23 MB |
| Effective HBM throughput | **83.6 GB/s = 4.7% of 1,792 GB/s peak** |
| Gate threshold for PASS | ≥ 50% (896 GB/s) |
| Gate threshold for MARGINAL | 30–50% |
| **Decision** | **KILL** (4.7% is well below 30%) |

## 2. Interpretation

The v6 pathology (§1 KILL_PATTERNS calibration: "FP4 dequant-in-WMMA-loop at H=2048 dense proxy — 2× slower, serial smem roundtrip dominates") **repeats with worse severity** in the Qwen3 real-MoE regime. Correctness is bit-perfect (cos=1.0000), so the kernel is not broken — it's simply HBM-access-pattern-bound in a way that dequant-then-WMMA cannot hide.

## 3. Root cause: Qwen3 FP4 weight layout forces 16 L1 cache lines per warp per K-step

Qwen3-30B-A3B-NVFP4 stores FP4 weights as `[N_out, K_in/2]` row-major (verified from safetensors header inspection — shapes `(768, 1024)` for gate/up and `(2048, 384)` for down). Each N-row is `K_in/2 = 1024 bytes` apart for gate/up.

The natural dequant tile for `wmma::load_matrix_sync(b_frag, b_dq_smem, ldb=16)` requires a `[16 K, 16 N]` BF16 tile (256 BF16 = 512 B) in smem. To populate it from the on-disk `[N, K/2]` layout, 32 warp lanes must access 16 unique N-rows (one per output column), each 1,024 bytes apart. Each warp-wide load generates **16 L1 cache line loads** (one per N-row), of which only 8 bytes per line are consumed.

At 128 K-steps per tile × 2 dequants (gate+up) per k-step × ~2 tiles/SM × ~48–128 busy SMs, the total L1 miss count dominates. The WMMA itself is cheap; the memory-subsystem serialization on these 6%-utilized cache lines costs 82 GB/s effective vs 1,792 GB/s peak.

## 4. Why the v7 design hypothesis (HBM-bound regime) failed

The v7 design argued that real MoE has 24× more weight traffic per layer than the dense-H=2048 proxy → HBM bound → FP4 pays off. That arithmetic is correct: **peak-capable** HBM traffic would make this regime bound. But the Qwen3 FP4 **layout** maps to an access pattern that cannot achieve peak HBM — each 16×16 BF16 tile pulls only 8/128 = 6% of each cache line's payload. The effective ceiling is not 1,792 GB/s; it's ~112 GB/s (6% × 1,792). Measured 83.6 GB/s is actually 75% of that layout-limited ceiling — we are very close to the wall imposed by the FP4 layout, and the wall is far below the 50% gate.

**This is a classic P11 Category 3b failure** — cross-apply reasoning from "dense H=2048 not HBM-bound" to "real MoE IS HBM-bound" missed the layout interaction. The regime did flip toward memory-bound, but the access pattern pinned the realizable bandwidth much lower than 50% of peak.

## 5. Pre-KILL Checklist (per `plans/KILL_PATTERNS.md` §3)

- [x] **Clean GPU state:** GPU 1 shows 0 MiB used, no peer compute apps.
- [x] **Single-container isolation:** no docker/vllm concurrent.
- [x] **Correctness pass:** cos=1.0000 across all 8 experts for both mlp_scratch and expert_out, max_abs=4.77e-7.
- [x] **Warm run:** 3 warmup launches discarded before timing; 50×8=400 iters in bench.
- [x] **Calibration constants up-to-date:** §1 FP4 dequant-Option-A = "2× slower at H=2048 dense proxy" directly matches this result class (now +worse at MoE shape).
- [x] **Shape regime named:** Qwen3 H=2048, INTER_PER_EXP=768, TOP_K=8, M=1 decode, 8 active experts. Same silicon (SM120a 188 SMs).
- [x] **Specific failure mode identified:** 16 L1 cache lines per warp per K-step from Qwen3 `[N, K/2]` layout vs `[K, N/16]` dequant-tile tile shape → 6% cache-line utilization → ceiling ~6% of peak HBM.
- [x] **Pattern sweep (P1–P10):** no silent-None (direct ctypes launch, not plugin); not launcher env issue; no graph capture; warm; not cross-apply.

All pre-KILL gates satisfied. KILL is valid.

## 6. What would need to change for v7.1

Three candidate redesigns — all strictly more expensive than original v7:

### 6.1 K-gang dequant (medium risk, +3 days)

Have 8 warps cooperatively dequant a `[16 N × 128 K]` tile (128 B per row = 1 cache line per row). This gives 8 K-steps worth of B-tile per load, amortizing cache-line waste 8×. Requires:
- Restructure inner loop: dequant once, WMMA 8 times.
- Additional smem: `8 × 512 B = 4 KB` per warp (vs current 512 B).
- Expected improvement: 6× faster dequant → ~42 µs/iter → 28% HBM, still MARGINAL.

### 6.2 Weight pre-swizzle to `[K, N/2]` layout (high risk, requires offline prep)

Offline-convert Qwen3 checkpoint to `[K_in, N_out/2]` layout. Natural dequant access then reads one N-row per warp = 1 cache line per K-step. Expected ~50–70% of peak. But:
- Requires a 14 GB offline resave (one-time).
- Breaks vLLM's `cutlass_fp4_moe_mm` fallback (scale tensor would also need transpose/re-swizzle).
- Makes checkpoint incompatible with stock vLLM → any fallthrough needs on-the-fly re-layout.
- P11 Category 4 risk: architectural lock-in.

### 6.3 Wider per-warp tile (16 N × 32 K) (low-medium risk, +2 days)

Load two K-tiles of 16 N × 8 bytes = 16 N × 32 K values per warp-wide load, dequant in a larger smem staging buffer, then issue 2 WMMAs per dequant-load. Amortizes cache-line waste 2×. Expected ~10% HBM → still KILL.

**None of the three redesigns clearly lands ≥ 50%.** The layout-limited ceiling at 6% cache-line utilization × 1,792 GB/s = 112 GB/s stays at the 6–7% range with any 16×16 tile shape.

## 7. Impact on v7 Phase 1 plan

**Phase 1 (cooperative kernel skeleton) SKIPPED per dispatch instructions.** No Phase 1 code was compiled. Phase 0 build artifacts remain in `/tmp/build_mega_graph_v7_phase0/` for reproducibility.

## 8. Recommendation

- **Do not proceed to v7 cooperative kernel** as-designed.
- **Consider v7.1 redesign** using K-gang dequant (§6.1) only if a partial win (e.g., MARGINAL 1.2× vs eager, not BIG_WIN) is acceptable. The layout-limited ceiling makes it unlikely to reach PASS (≥1.5×).
- **Higher-leverage alternatives** to revisit instead:
  1. CUTLASS-native `cutlass_fp4_moe_mm` already handles the Qwen3 layout with its own 128×4 swizzle; it's 1.7× over scalar reference per prior audit. v7's cooperative fusion wouldn't beat that without a layout change.
  2. Attention optimization (v5a is 1.12× vs eager at real Gemma4 dense) is a cleaner leverage point.
  3. Batching at M>1 (prefill or higher concurrency) gives compute-bound regime where FP4 IS useful; current M=1 decode is fundamentally layout-bound.

## 9. P11 Confidence Recalibration

Before dispatch: P(Phase 0 passes) ~0.35–0.50 (halved per P11 Cat 2+3b). 

After result: actual outcome (KILL at 4.7%) is worse than even the 30-50% MARGINAL hypothetical. The v7 design team underestimated the **layout×tile-shape interaction** — a specific mechanical effect, not just a calibration error. This is NEW information:
- P11 Cat 3b is even weaker than the P~0.15 base rate suggested; the specific failure mode (Qwen3 N-row stride × 16×16 tile = 6% cache utilization) was not anticipated in any of the preparation docs (`v4b_fp4_spec`, `v4b_checkpoint_format`, `v7_real_moe_design`).
- Future FP4 cross-model transfer proposals should add a layout-×-tile-shape dimension to their pre-check.

**P(v7.1 redesign lands at PASS gate) after this data: ~0.10–0.15.** The layout ceiling is mechanical; only §6.2 (offline re-layout) would break past 50%, and that trades correctness/compatibility for speed. Not recommended as a near-term investment.

---

## Appendix: reproduction commands

```bash
cd /home/cklaus/projects/autokernel
python3 kernels/csrc/build_mega_graph_v7_phase0.py
# Pin GPU 1 to avoid conflicts with other agents
CUDA_VISIBLE_DEVICES=1 python3 kernels/csrc/test_mega_graph_v7_phase0.py
```

Expected output:
```
[COR] expert_out cos per expert: ['1.0000', ...]  — PASS
[BW] effective HBM throughput: 83.6 GB/s (4.7% of 1792.0 peak)
[DECISION] KILL (4.7% < 30%) — v6 pathology repeating, STOP.
```

---

*End of Phase 0 results. Phase 1 skipped per decision-gate dispatch rule.*
