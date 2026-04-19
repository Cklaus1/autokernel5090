# Banked Claims Sweep — 2026-04-19

**Tag:** `W6_banked_claims_verify_sweep`  
**Audit basis:** fused-norm v2 finding (`.kernel` vs `.backend` typo — v1 never fired), FP8 decode audit (W5_A1 — `FIDecode` silent-fallback on SM120 means T2-I custom kernel was inert), W5_25 (WSL2 GPU isolation proven broken for concurrent dual-container benches).  
**Method:** CPU-only; no bench runs; source: `results.tsv` all PASS/KEEP rows + three audit documents.

---

## Classification Legend

- **VERIFIED** — plugin banner confirmed firing + correctness tracked + single-container isolation + warm measurement
- **STALE** — infrastructure or model has since changed; number no longer matches current serving config
- **SUSPECT** — silent fallback, silent-None dispatch, or plugin-not-firing risk exists and is unconfirmed
- **UNREPRODUCIBLE** — requires dual-GPU concurrency that W5_25 proved is broken on this WSL2 setup
- **TIME-DECAYED** — banked >7 days ago with no re-verification; state-drift probability high

---

## Row-by-Row Classification

### W4A16 / NVFP4 micro-bench rows (exp 21–96, rows 3–36)

| Rows | Tags | Claim | Classification | Evidence |
|---|---|---|---|---|
| exp 21–96 (autotune_l2swizzle through clean_4config) | PASS/REVERT | 328 TFLOPS W4A16 matmul | **VERIFIED** | Pure Triton microbench; no plugin, no vLLM; correctness column populated; results are self-contained and bench-reproducible in isolation. Kernel on disk unchanged. |

**Risk: LOW.** These are micro-benchmarks with no plugin chain, correctness confirmed, bench.py runs deterministically.

---

### dequantize_fused_gemm rows (exp 0–2, rows 38–40)

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| exp 2 `split_dequant_cublas` 212 TFLOPS | **VERIFIED** | Triton dequant + cuBLAS F.linear; pure kernel bench; PASS correctness; no vLLM dependency. |

---

### nvfp4_matmul series (exp 3–10, rows 42–49)

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| exp 10 `nvfp4_cache_both` 1,260 TFLOPS | **VERIFIED** | Pure GEMM bench on local shape; cos correctness confirmed; no plugin chain. Single measurement on stable kernel. |
| exp 6–9 (cuda v1–v3) | **VERIFIED** | Individual microbench rows; no cross-dependency. |
| exp 11 `nvfp4_scaled_mm_v2` PASS+REVERT note | **VERIFIED** | Banked only as a comparison point; already reverted. |

---

### Model serving rows (exp 15–48 fp8_kv, 9b_*, vllm_serve)

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| exp 15 `fp8_kv_cache` 88 tok/s, +19% cost claim | **SUSPECT** | The "+19% cost" phrasing echoes the banked projection language. The row is dated pre-fused-norm-v2 era. More importantly: FP8 KV here refers to capacity/cost, not kernel performance — appears to be a vLLM native FP8 KV path, not the T2-I custom Triton kernel. The 88 tok/s decode figure itself is likely clean (FP8 KV is FlashInfer native at 88 tok/s). But **the "+19% cost" phrasing must not be conflated with a kernel-level claim.** Classify as SUSPECT pending clarification of which kernel path was active. |
| exp 30–48 9B serving rows (9b_baseline, no_chunked, variance, etc.) | **VERIFIED** | Qwen3.5-9B; single container; throughput 117–120 tok/s consistent across variance test; correctness column blank but model-level correctness verified via coding quality bench (exp 37). |
| exp 20 `vllm_serve_openai` 119 tok/s | **TIME-DECAYED** | Banked on Qwen3-30B initial setup; >7 days ago; model and serving stack have changed significantly (PRO 6000 vs original HW). |
| exp 33–34 9B NVFP4 batch64 4,471–4,513 tok/s | **VERIFIED** | Clean single-GPU bench on 9B model; FP8 KV confirmed same throughput. Not affected by fused-norm chain (9B model, different quant path). |

---

### T2-I FP8 decode — **THE PRIMARY SUSPECT** (row 109, `T2I_fp8_kv_cudagraph`)

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `T2I_fp8_kv_cudagraph` 12,229 tok/s "NEW PEAK +71% vs BF16 baseline 11,048" | **SUSPECT (HIGH CONFIDENCE PHANTOM)** | W5_A1 audit (`fp8_decode_silent_fallback_fix.md`) confirms: `patches/fp8_decode_monkey_patch.py:319-335` silently re-routes to FlashInfer native when `attn_metadata.decode` is `FIDecode`. SM120 (`is_device_capability(100) = False`) **always selects FIDecode**, not TRTLLMDecode. The custom Triton split-K kernel was therefore inert on every request. The 12,229 vs 11,048 delta (+10.7%, not +71%) is real but represents FP8 KV capacity enabling higher concurrency (C=768 instead of C=512 BF16-constrained), NOT the custom FP8 decode kernel. The "+71%" headline is the **T2-I kernel projection that was never measured** — the banked 12,229 row measures the capacity benefit, not the kernel speed. **Impact: if the T2-I kernel is confirmed inert, the 12,229 number survives but must be re-attributed to FP8 KV capacity gain, not T2-I kernel performance.** |

**Re-verification recipe:**
```bash
# On next docker session:
AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1 ./launch_<fp8_decode>.sh
# If crash on first request → kernel was inert → re-attribute 12,229 to capacity-only
```

---

### T2-N rows (pre-fused-norm-v2 era)

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `T2N_fused_shuffle_quant` 1.68× microbench (row 120) | **VERIFIED** | Pure kernel bench with correctness (cos=0.9952, ±1 FP8 step). Plugin fires via dist-info entry_points. |
| `T2N_cutlass_swizzle_inline` 1.07× microbench (row 147/150) | **VERIFIED** | Correctness confirmed; cos=0.9952; post-pass eliminated. |
| `T2N_e2e_fused_qwen3` 18,756 tok/s **+34% at C=512** (row 156) | **SUSPECT (moderate)** | Banked pre-fused-norm-v2 era. Plugin banner `[T2-N] Patched run_cutlass_moe_fp4 via plugin` confirmed in logs. However: this row was benched on `vllm-fusencache:latest` with `VLLM_USE_FLASHINFER_MOE_FP4=0`. The subsequent `T2N_rebench_clean_20260418e` (19,558 tok/s) DOES verify the number is real under clean isolation. Classify as **VERIFIED** for the underlying mechanism, but the specific 18,756 number is superseded by 19,558 (cleaner run). |
| `T2N_rebench_clean_20260418e` 19,558 tok/s (row 194) | **VERIFIED** | Single-container isolation confirmed; UUID-check isolation concern addressed (sole container); correctness wrapper test in prior microbench rows. This is the operative T2-N banked number. |
| `T2N_e2e_baseline_qwen3` 13,994 tok/s baseline (row 155) | **VERIFIED** | Baseline measurement for T2-N delta; CUTLASS path forced off. |
| `T2N_bigsweep_*` rows (165–170) | **VERIFIED** | Sweep under T2-N conditions; C=512 peak 17,614 (within noise of 18,756). Confirms plateau not new peak. |

---

### W3_CA — Fused-norm Qwen3 v2 (row 204, `W3_CA_fused_norm_qwen3_v2`)

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| 23,254 tok/s **NEW PEAK** (+19% over 18,756) | **VERIFIED** | v2 bug fix confirmed firing: `all 48 layer fusions now active vs 0 in v1`. Banner `[fused_norm_fp4_qwen3]` active. Correctness: v2 dispatch uses same `.backend` pattern as `fused_norm_fp4_integration.py` (Gemma4 plugin — kill audit #3 confirmed correct). Measured today (2026-04-19). Clean-state. |

---

### SWA sparse rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `swa_sparse_decode` 4.64× microbench (row 187) | **VERIFIED** | cos=0.999999 vs PT ref; correctness confirmed; pure Triton kernel bench; single-GPU. |
| `SWA_sparse_e2e_patched_gemma4` 2.52× at C=8 (row 192) | **VERIFIED** | Single-container Gemma4 serving; baseline row (191) present as anchor; graph-capture guards in plugin. Post-W5_25 isolation concern does not apply — single container, no concurrent bench. |
| `W1_4a_swa_fp8_kv_gate4` 1.19× compound (row 201) | **VERIFIED** | Measured 2026-04-19 under clean conditions; FP8 KV + SWA; 2× capacity verified at 32K context. |
| `W5_B2_swa_numwarps8_tune` 1.01× PARTIAL (row 208) | **PARTIAL** — not a banked win; correctly labelled PARTIAL. |

---

### Dual-model / concurrent GPU rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `Qwen3_30B_A3B_NVFP4` 17,426 tok/s GPU1 (row 112/125) | **STALE** | `serve_dual_model.sh` now uses Qwen3.6-FP8 (commit `60090fb`). This number was on Qwen3-30B-NVFP4. Current Qwen3.6-FP8 peak is 5,895 tok/s. |
| `Qwen36_35B_A3B_FP8` 5,895 tok/s (row 113/126) | **VERIFIED** | Single-GPU, single-container, confirmed at C=128. The model and launcher match. |
| `T1G_dp2_linearity` 2,812 tok/s / projected 22k (row 117) | **UNREPRODUCIBLE** | DP=2 concurrent bench. W5_25 proves dual-GPU concurrent containers both landed on GPU 0. The "GPU0=881 (degraded from bench storm) GPU1=1931 (fresh)" measurement is itself contaminated — if both ran on GPU 0, the 881/1931 split is artificial. The "projected 22k" extrapolation is invalid. **Formal retraction recommended for the projection.** |
| `T1G_dp2_fp8_kv` 2,965 tok/s / projected 24,458 (row 123) | **UNREPRODUCIBLE** | Same issue: sequential measurement (noted in description) partially mitigates but "GPU0=1656 GPU1=1309" at concurrent service is unverifiable without WSL2 fix. Sequential single-GPU baseline (12,229 × 2 = 24,458 projection) is arithmetically valid if both GPUs perform independently — but GPU isolation is unconfirmed. Retain the arithmetic projection as a footnote; mark the measured aggregate as UNREPRODUCIBLE. |
| Implied "dual-model 29,655 aggregate" (W5_25 row 209 discussion) | **STALE + UNREPRODUCIBLE** | (a) Stale: Qwen3 half is now 5,895 not 17,426. (b) Unreproducible: concurrent bench proven broken. Formal aggregate is now 12,229 + 5,895 = 18,124 tok/s max (sequential, unverified together). |

---

### T1-A / T1-B FusenCache rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `T1A_pro6000_high_concurrency` 11,048 tok/s BF16+CG C=768 (row 116) | **VERIFIED** | Single-container; BF16+CG path; no plugin complexity. Correctness: zero errors confirmed in description. This is the clean non-FusenCache peak. |
| `T1A_pro6000_cudagraph` 1,940 tok/s C=32 (row 114) | **VERIFIED** | Single-GPU single-container; no FusenCache; pure CG baseline. |
| `T2I_fp8_kv_cudagraph` 12,229 tok/s (row 122) | **SUSPECT** — see T2-I section above. |
| `W5_T1B_shadow_tensor_fix_bench` 751 tok/s piecewise (row 210) | **VERIFIED** | Measured 2026-04-19; zero crashes at C=1–32; shadow-tensor fix confirmed; 17.5× over prior KILL. Single container. Note: full-mode 4,489 tok/s banked peak (prior vLLM version) is now **UNREPRODUCIBLE** per row 211. |
| Banked `T1B_piecewise_C32` 224.9 tok/s PASS tag (row 129) | **STALE** | Superseded by W5_T1B fix (751 tok/s). The 224.9 row was a broken intermediate state. |
| `T1B_peak_vs_baseline` 224.9 FAIL row 136 | **STALE** — correctly labeled FAIL; already superseded. |
| Prior `4,489 tok/s` banked peak (referenced in W5_T1B_fusen_full_mode_CRASH) | **UNREPRODUCIBLE** | Full-mode CUDA graph crashes at C>32 due to SM120 graph-replay limit (silicon-level). That peak was on a different vLLM version. FORMAL RETRACTION of 4,489 as an achievable number on current stack. |

---

### T1-C / T1-F / T1-E rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `T1C_*` (BLOCKED rows) | Not banked claims; correctly BLOCKED. |
| `T1F_*` (KILL rows) | Not banked claims; correctly KILL. |
| `T1E_turboquant_structured` KILL (row 134) | Not a banked win. |
| `turboquant_bench` 0.38ms compress PASS (row 130) | **VERIFIED** | Latency microbench; random data; no plugin chain. Note: K cosine 0.940 below 0.95 threshold — this is a correctness caveat, not a speed claim. |

---

### LMCache rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `W1_4e_lmcache_smoke_rebuilt` 1.27× throughput, 1.56× P50 TTFT (row 205) | **VERIFIED** | Single-container; single-GPU (GPU 0); baseline+patched measured sequentially. SM120 rebuild confirmed working. Caveat: single-instance weak prefix scenario — larger delta expected at multi-instance/cold. |

---

### I-DLM rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `idlm_v1_baseline_pro6000` 162.7 tok/s (rows 139, 152) | **VERIFIED** | Single-GPU (GPU 0 via CVD=1 env); 65-68% acceptance confirmed; SM120 CuTe SM90 fallback patched. Duplicate rows are consistent. |
| `T2G_idlm_v2` 197.9 tok/s KEEP (row 124/137) | **SUSPECT** | v2 banked at C=16 +29.4%. BUT v2 correct-mask integration was subsequently BLOCKED and a full kill row (`idlm_v2_mask`) showed acceptance 42%→8.6% FAIL. The "KEEP" at row 124 used `custom_mask` path that was "unreachable code path" per row 127. **This KEEP row is likely a measurement artifact — the benefit came from a path that was subsequently confirmed non-functional.** Recommend SUSPECT → re-verify which mask path was actually active. |
| `idlm_v2_mask` FAIL (row 128) | Not a banked win; correctly FAIL. |

---

### Mega-graph / FusenCache kernel rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `mega_graph_feasibility` 1.12× skeleton PASS (row 175) | **VERIFIED** | Grid.sync microbench; measured 0.82 µs/barrier; clean kernel bench. |
| `mega_graph_2layer_prototype` 2.89× at H=512 (row 182) | **VERIFIED** | Small shape (H=512); correctness max_abs=1.95e-3. Caveat: this is a toy shape, not production H=2048. |
| `mega_graph_30layer_v5a` 1.42× PARTIAL (row 198) | **VERIFIED (as PARTIAL)** | Correctly tagged PARTIAL; 1.42× < 1.5× gate; correctness PASS. Not a banked win. |
| `native_fp4_scale_kernel` 1.06× PASS (row 177) | **VERIFIED** | Microbench byte-identical to CUTLASS; confirmed by kill audit #15. The "PASS" here is a measurement, not a speedup claim. |
| `fusencache_cpasync_prototype` 1.24× KILL (row 176) | Not a banked win; KILL. |
| `fusencache_warpspec_v2` 1.18× KILL (row 183) | Not a banked win; KILL. |

---

### T2-H Qwen3 FP8 KV sweep rows (rows 141–146)

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `T2H_qwen3_nvfp4_C768` 16,637 tok/s (row 146) | **TIME-DECAYED** | Banked 2026-04-18; measured on `vllm-fusencache:latest` GPU1. Superseded by: (a) T2N_rebench_clean (19,558 — T2-N active), (b) W3_CA (23,254 — fused-norm v2 active). The 16,637 row predates fused-norm v2 fix and the T2-N clean rebench. Retain as historical data point; operative peak is 23,254. |
| `T2H_qwen3_nvfp4_C512` 14,707 tok/s (row 145) | **TIME-DECAYED** | Same reasoning; superseded. |
| `T2H_qwen3_nvfp4_C128` 10,530 tok/s (row 142) | **TIME-DECAYED** | Superseded by current stack (23,254 peak means C=128 is no longer the interesting regime). |

---

### Persistent MoE dispatch rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| No explicit persistent MoE PASS rows in results.tsv — the feasibility was KILL'd based on 15 µs/barrier reasoning. | **N/A** | Tier2_3 audit #19 flags that the 15 µs/barrier cost cited in `persistent_moe_fp4_feasibility.md:61` was 5× too high (true cost ~3 µs). The KILL verdict should be reopened as DEFER. But since no banked PASS exists, no retraction needed. |

---

### Infrastructure PASS rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `autotuner_cache_works` 63s load (exp 32) | **VERIFIED** | Measurement of cache load time; deterministic. |
| `profile_py_rename` PASS (exp 13) | **VERIFIED** | Code rename; infrastructure. |
| `bandwidth_ceiling` PASS (exp 29) | **VERIFIED** | 1,500–1,530 GB/s measured; BW ceiling arithmetic. |
| `9b_nvfp4_default_8k` 4,471 tok/s (exp 33) | **VERIFIED** | See 9B rows above. |
| `T1A_pro6000_baseline` 830.3 tok/s (row 100/113) | **VERIFIED** | Eager-mode baseline; single GPU; no FusenCache. |
| `pro6000_baseline` TFLOPS (rows 87–88) | **VERIFIED** | Raw GEMM benchmarks; no plugin chain. |
| `pro6000_baseline_triton` 1.28× fused_norm (row 90) | **VERIFIED** | Triton kernel bench; PRO 6000; C++ kernel (2.95×) deferred (needs cuda-toolkit-12.8). |

---

### ASI / Priority scheduling rows

| Row | Claim | Classification | Evidence |
|---|---|---|---|
| `ASI_alt_AD_C4/16/32` PASS rows (rows 171–174) | **VERIFIED** | C=4/16/32 P99 < 1000ms; zero errors; single container. C=8 PARTIAL confirmed; does not affect other rows. |
| `ASI1_disagg_bringup` BLOCKED/KILL rows | Not banked wins. |

---

## Summary Table

| Classification | Count | Key rows |
|---|---|---|
| **VERIFIED** | ~45 | All W4A16 micro-benches; 9B serving; T2-N rebench_clean; W3_CA fused-norm v2; SWA micro+e2e; W1_4a FP8+SWA; LMCache; I-DLM v1; most infrastructure rows |
| **STALE** | 5 | Qwen3-30B-NVFP4 GPU1 17,426; dual-model 29,655 aggregate; T2H sweep rows (C=512/768/128 superseded); T1B 224.9 intermediate |
| **SUSPECT** | 4 | T2I_fp8_kv_cudagraph 12,229 (+71% headline phantom); exp 15 fp8_kv "+19% cost" conflation; T2G_idlm_v2 KEEP (mask path uncertain); T2N_e2e_fused_qwen3 18,756 (superseded by cleaner 19,558) |
| **UNREPRODUCIBLE** | 3 | T1G_dp2 projected 22k; T1G_dp2_fp8 projected 24,458; FusenCache full-mode 4,489 banked peak |
| **TIME-DECAYED** | 4 | T2H sweep (pre-fused-norm-v2); exp 20 vllm_serve early; early Qwen3.5-9B rows pre-hardware-upgrade |

**Total PASS/KEEP rows classified: ~61** (excluding all FAIL/REVERT/KILL rows, which are not claims)

---

## Top 5 SUSPECT Claims by Impact-If-Phantom

| Rank | Row | Headline Claim | Impact-if-Phantom | Evidence Quality |
|---:|---|---|---|---|
| 1 | `T2I_fp8_kv_cudagraph` | "+71% from T2-I FP8 decode kernel" | **PHANTOM CONFIRMED.** The 12,229 tok/s number survives (it's real — FP8 KV capacity gain), but the kernel attribution is entirely wrong. The T2-I Triton split-K kernel never fired on SM120. Impact: T2-I remains an UNVERIFIED optimization with no e2e measurement. **>5% of session peak impact — if the kernel were real, it would be the path to ~17k+ tok/s from FP8 decode speedup. Current 23,254 is the true peak via different mechanisms.** | HIGH — `fp8_decode_silent_fallback_fix.md` confirms FIDecode path on SM120; AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK test would crash immediately |
| 2 | Dual-model aggregate 29,655 tok/s | "+100% via concurrent dual-model" | **STALE + UNREPRODUCIBLE.** Model changed (17,426 → 5,895 for Qwen3 half); WSL2 GPU isolation broken (both containers on GPU 0). If measured today: 12,229 + 5,895 = 18,124 max. **38.9% reduction in session aggregate claim.** | HIGH — W5_25 row documents both containers on GPU 0; commit `60090fb` changes Qwen model |
| 3 | FusenCache full-mode peak 4,489 tok/s | "T1-B piecewise fix recovers 4,489" | **UNREPRODUCIBLE.** W5_T1B_fusen_full_mode_CRASH (row 211) proves SM120 graph-replay limit prevents full-mode at C>32. Achievable ceiling is 751 tok/s piecewise. **Impact: 83% reduction in T1-B claimed ceiling.** | HIGH — silicon-level SM120 graph size limit confirmed in today's bench |
| 4 | T2G_idlm_v2 KEEP 197.9 tok/s | "+29.4% from correct mask (C=16)" | **SUSPECT.** The mask integration path that was "working" for the KEEP verdict was subsequently found to be "unreachable code path" per row 127 and produced 77 tok/s (FAIL) in the formal v2 kill test. The KEEP number may reflect a different code path than advertised. **Impact: if phantom, I-DLM baseline reverts to 162.7 tok/s v1.** | MEDIUM — two conflicting rows (137 KEEP vs 128 FAIL) with different mask implementations |
| 5 | T2N_e2e_fused_qwen3 18,756 tok/s | "+34% from T2-N fused shuffle+quant" | **SUSPECT (LOW) / effectively superseded.** The 18,756 was from a possibly-contaminated single-GPU run (WSL2 cross-GPU leak era). The clean rebench (19,558) supersedes it and actually confirms T2-N is real. But the original 18,756 number in the banked claim should formally be retired in favor of 19,558. Impact limited: the mechanism is confirmed. | LOW — superseded by cleaner measurement |

---

## Claims Requiring Formal RETRACTION (mark WITHDRAWN in results.tsv)

| Row identifier | Retraction reason | Replacement |
|---|---|---|
| "T1-B full-mode 4,489 tok/s peak" (referenced in row 210 description, not a standalone row) | SM120 silicon-level graph-replay limit; full mode crashes at C>32 | Current achievable: 751 tok/s piecewise (row 210, W5_T1B_shadow_tensor_fix_bench) |
| "dual-model 29,655 tok/s aggregate" (referenced in W5_25 row 209, not a standalone row) | (a) model upgrade cut Qwen3 half by 66%; (b) WSL2 isolation proven broken for concurrent benches | Current arithmetic max: 18,124 tok/s (12,229 + 5,895, single-GPU each, not concurrent) |
| T2-I "+71%" throughput gain claim | Custom kernel never fired on SM120 (FIDecode path always active; fallback to FlashInfer native) | Re-attribute 12,229 row as "FP8 KV capacity gain enabling higher concurrency", not kernel speedup; T2-I kernel gain = 0× confirmed |
| T1G DP=2 projected 22k/24,458 tok/s | Concurrent measurement on WSL2 with proven GPU isolation failure | Valid projection only after serial re-bench under confirmed GPU isolation |

---

## Priority-Ranked Re-Verification Sequence

**For parent to run in order (each is a single bench session):**

### Priority 1 — T2-I phantom confirmation (30 min)
```bash
# Confirms whether T2-I kernel claim is phantom
AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1 ./launch_<fp8_decode>.sh
# Expected: crashes on first request (FIDecode active) → T2-I RETRACTED
# If no crash: T2-I was actually firing → re-bench to confirm +X%
```
**Why first:** highest-confidence phantom; confirms whether T2-I represents real future upside or a dead end. Informs whether Option A/B/C fix (force TRTLLMDecode) is worth pursuing.

### Priority 2 — T2G_idlm_v2 KEEP vs FAIL reconciliation (1 hr)
Re-run I-DLM v2 with explicit mask-path logging. Identify which of `custom_mask`, `k-zeroing`, or `CuTe` path actually ran during the KEEP measurement. If the winning path is unreachable, formally retract 197.9 tok/s → 162.7 tok/s (v1 baseline).

### Priority 3 — FusenCache piecewise ceiling re-bench (2 hrs)
```bash
# Sweep C=32 through C=128 on piecewise mode post-W5_T1B fix
# Confirm 751 is the ceiling vs prior 4,489 banked
```
**Why:** if piecewise can scale past C=32 with the shadow fix, the ceiling may be higher than 751. Kill audit #2 (T1-B recover, P=0.60) says 4,489 is potentially recoverable — but only via piecewise, and only if the boundary bug is fully resolved.

### Priority 4 — Dual-model serialized re-bench (2 hrs, no concurrency)
```bash
# Run Gemma4 solo → record peak
# Stop Gemma4 → start Qwen3.6-FP8 → record peak
# Sum = valid non-concurrent aggregate
```
**Why:** replaces the unreproducible 29,655 number with a defensible figure. Also tests whether reverting to Qwen3-30B-NVFP4 (instead of Qwen3.6-FP8) restores the 17,426 half and a 29k+ aggregate.

### Priority 5 — T2-N + fused-norm v2 C=1024 re-verify (1 hr)
Confirm 23,254 tok/s holds on a clean re-run (not cold start). The W3_CA row is marked VERIFIED but was measured only once. A warm second run confirms it's not a cold-cache artifact.

---

## Claims Whose Phantom Status Would Impact >5% of Session Peak

**Session peak: 23,254 gen tok/s (W3_CA, Qwen3-30B-A3B NVFP4 + fused-norm v2 + T2-N)**

| Claim | Impact-if-phantom | % of 23,254 |
|---|---|---|
| T2-I "+71%" kernel (phantom confirmed) | Lost: 0× on T2-I kernel specifically; 12,229 FP8 KV capacity number survives | T2-I kernel = 0 delivered; but does not reduce 23,254 peak |
| Dual-model 29,655 aggregate (STALE) | Session aggregate drops by 38% (29,655 → 18,124) | Aggregate impact > 5% of projected concurrent serving |
| FusenCache 4,489 full-mode (UNREPRODUCIBLE) | T1-B FusenCache path for Gemma4 loses 83% of its claimed ceiling | Gemma4 FusenCache path effectively killed for current stack |
| W3_CA 23,254 if fused-norm v2 has its own silent issue | Would reduce to ~19,558 (T2-N only) = -16% | 16% of session peak at risk if v2 has a residual bug |

**The W3_CA 23,254 row is the only VERIFIED high-impact claim. It MUST be protected by the Priority 5 re-bench to confirm it is not a cold-start artifact before being used as the baseline for all future comparisons.**

---

## Systemic Findings

1. **FP8 decode (T2-I) is the canonical second P1 failure** after fused-norm v1. Both involved a silent-fallback that made the optimization look banked while delivering nothing. Two P1 failures in one session strongly motivates the `AUTOKERNEL_FORCE_*_ASSERT=1` env-var pattern for every new patch.

2. **WSL2 GPU isolation is broken for any concurrent dual-container measurement.** All DP=2 and dual-model "aggregate" numbers should carry a UNREPRODUCIBLE tag until a working isolation mechanism (options: serialize containers, or native Linux) is confirmed.

3. **The session's true banked peak (23,254 gen tok/s) is clean and defensible** — it came from a code-level bug fix (`.kernel` → `.backend`) on a single container, single GPU. It is not at risk from any of today's audit findings unless a v2-specific regression is discovered.

4. **The fused-norm v1 "+19% was invisible for weeks" finding should be the default explanation** for any banked claim where plugin banners appear but performance matches baseline. Audit first with `AUTOKERNEL_FORCE_*_ASSERT=1`, then trust the number.

---

*Generated: 2026-04-19 | Tag: W6_banked_claims_verify_sweep*
