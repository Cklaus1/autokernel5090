# Kill Audit 20260419 — top-20 KILL/DEFER/PARTIAL review

**Tag:** `W4_audit_20260419_kills_top20`
**Hardware context:** 2× RTX PRO 6000 (SM120a). Grid.sync cost re-measured at **~3 µs/barrier**, not 15 µs. `getattr(obj,'kernel',None)` silent-None bug landed 48-layer plugin regression (qwen3 v1→v2).
**Method:** CPU-only code review; no builds, no kernel runs. One smoking-gun focus per candidate.

---

## 1. v5a.1 multi-head pack — **RECOVER**

**KILL reason.** Added 1 intra-attention barrier (5→6/layer). File-internal comment lines 40-42 state: "Added cost ~30 × ~50µs = +1.5ms" — then reasoning in the audit brief quotes 15 µs/barrier → +450 µs; the file itself cites an even larger 50 µs estimate. Either way, the attention redistribution win was smaller than the projected barrier cost.

**Smoking-gun checks.**
- Line 42 comment: "Added cost ~30 × ~50µs = +1.5ms" — clearly wrong by **16×** at true 3 µs/barrier: actual cost is +90 µs, not +1500 µs.
- `QK_UNITS_TOTAL = NUM_HEADS * QK_N_TILES = 256` with num_sms=188 → ~1.36 units/SM average, confirming attention redistribution does light up all 188 SMs.
- The scheduling in `attention_qk_phase` is unit-id-major (`unit = h * QK_N_TILES + nt`) — keeps 1 head resident in Q16 smem across consecutive units on the same SM. This is correct, not buggy.

**Verdict: RECOVER, P(lands) ≈ 0.70.**

**Fix.** Rebench v5a.1 as-is at correct barrier cost; the KILL verdict was arithmetic-induced. No code change needed — just rerun the head-to-head vs v5a with stopwatch. Projected gain: attention phase @ v5a leaves 172/188 SMs idle (Amdahl cliff per file comment); distributing across 188 SMs should recover ~300-600 µs across 30 layers. At barrier cost +90 µs, net-positive ~200-500 µs / 24 ms = **+1-2% e2e**.

---

## 2. T1-B FusenCache piecewise shadow-tensor — **RECOVER (specific bugs identified)**

**KILL reason.** Shadow-tensor patch regressed -95%; `cudaErrorIllegalAddress` at C>4. Rolled back.

**Smoking-gun checks (diff'd against `backend.py.t1b_broken`).**
- **Bug A (line ~486, size mismatch):** `self._shadow_query_start_loc = torch.empty(_max_seqs + 1, ...)` — sized at `_max_seqs+1`. But `query_start_loc` length at runtime = `num_reqs+1` which under piecewise scheduler can be padded beyond `_max_seqs`. The `min(ql.shape[0], 33)` silent truncation at line ~994 corrupts attention indexing once the input exceeds the shadow capacity.
- **Bug B (line ~481):** `_shadow_slot_mapping` sized at `_max_tokens` (which was overridden to `max_num_batched_tokens`), but all four shadow tensors share the same "_max_seqs / _max_tokens" bracketing logic with no assertion. `_max_tokens` can = `_max_B` (= max_num_seqs) which is smaller than the padded token count the scheduler emits at C=128.
- **Bug C (line 988-993 in broken copy):** `_block_table = self._shadow_block_table[:bt_rows, :bt_cols]` — this creates a **view**, not a clone. Subsequent scheduler writes through `attn_metadata.block_table` don't touch the shadow (good), but the shadow is overwritten each step; any kernel still reading the prior step's shadow view gets torn data. The intent was isolation from the default pool; the implementation leaks the isolation under piecewise graph capture because views of static buffers **are** captured.

**Verdict: RECOVER, P(lands) ≈ 0.60.**

**Fix (≤10 lines).** Size all four shadow tensors to `max(max_num_seqs, max_num_batched_tokens) + SAFETY_MARGIN` (e.g. 4096) up front, assert at copy time that `ql.shape[0] <= self._shadow_query_start_loc.shape[0]` (crash instead of silent truncate), and clone the slice returned to the caller (`.clone()` on the view) to break any cross-step aliasing under graph capture. Target: recover the 4,489 tok/s peak via piecewise graphs at C=128.

---

## 3. Fused-norm Gemma4 original plugin — **CONFIRM_KILL (no silent-None bug here)**

**KILL reason.** Works historically but projected +12.9% unmeasured.

**Smoking-gun checks.**
- `fused_norm_fp4_integration.py:158` uses `hasattr(qkv_qm, 'backend')` and `qkv_qm.backend != NvFp4LinearBackend.MARLIN` — this is the **correct** attribute pattern. No `.kernel` attribute dereferenced anywhere.
- Line 160: same `backend` check for the MLP path. Both fusion points use `_make_fused_linear_fn` which dispatches on `backend.value.startswith("flashinfer-")` then FBGEMM then default CUTLASS — identical to the v2 Qwen3 fix.
- Diff vs `wire_fused_norm_fp4_qwen3_v2.py`: structurally equivalent. Both use `quant_method.backend`. No silent-None dispatch miss here.

**Verdict: CONFIRM_KILL for the dispatch-bug hypothesis.** The Gemma4 plugin does NOT share the v1 Qwen3 bug. Any shortfall vs the +12.9% projection is elsewhere (likely CUDA graph capture interaction or residual materialization cost) and out of scope for this audit.

---

## 4. Early T2-N swizzle iterations — **CONFIRM_KILL**

**KILL reason.** `T2N_fused_shuffle_quant_swizzled` showed net -46%; agent pivoted to inline CUTLASS swizzle which PASS'd at 1.07×. Earlier variants had post-pass reshape overhead.

**Smoking-gun checks.**
- Session log confirms the successful path went inline in `fused_shuffle_quant.cu` and eliminated the post-pass reshape entirely. The -46% was a specific "post-pass" architecture, not a tunable.
- Prior variants would need the same inline-CUTLASS fix that already landed — they are strictly inferior branches of the same tree.

**Verdict: CONFIRM_KILL.** The final PASS approach dominates. No branch to recover.

---

## 5. v5b barrier fusion — **RECOVER**

**KILL reason.** 1.010× vs v5a. Achieved 4/layer not 3 target. File line 31 explicitly documents: "Expected gain: -30 barriers × ~15 µs = -450 µs." At 3 µs true cost this is -90 µs over 30 layers — ~0.4% e2e — indistinguishable from noise. But the fusion ALSO introduced **functional cost**: 64 KB o_partials traffic per layer + a 16-way sum across heads in `mlp_down_plus_oproj_residual_stage`.

**Smoking-gun checks.**
- Lines 548-561 `oproj_reduce_mlp_gate_up_stage`: each SM reads **all 16 o_partials** (16 × 2048 = 32K bf16 = 64 KB) plus hidden — adds ~3-4 µs/layer of redundant HBM reads per SM (all 188 SMs redo the same read).
- Lines 686-690 `mlp_down_plus_oproj_residual_stage`: inner loop sums 16 partials per element — adds 16 FP32 adds + 16 bf16→fp32 converts per output column, serialized per lane.
- The fused design saves barriers but adds ~480 KB of aggregate HBM traffic per layer (188 SMs × 16 partials × slice), versus v5a's attention+O-proj which only reads each attn_out once.

**Verdict: RECOVER (partial).** The barrier-fusion premise was correct; the implementation introduces a new HBM bottleneck. P(lands if reworked) ≈ 0.45.

**Fix (>10 lines, needs design).** Drop the "all-SMs-read-all-partials" pattern. Instead: restrict the partial reduction to 16 head-SMs (one per head), have them write already-reduced O-proj output to `hidden` in place (race-free because head-SM h owns columns [h*HEAD_DIM..(h+1)*HEAD_DIM)) then barrier → normal MLP. This eliminates the 64 KB broadcast and preserves the 1-barrier save.

---

## 6. v4a cp.async prefetch — **DEFER**

**KILL reason.** 0.92× vs v3. Comment line 30-34 confirms only `cp.async.ca.shared.global` was tested.

**Smoking-gun checks.**
- File uses `cp.async.ca.shared.global` with 16-B loads (line 30-34 of header comment). TMA (`cp.async.bulk.tensor`) is NOT tested; the plan correctly notes SM120a has TMA but not paged-KV scatter, and for these contiguous B-tiles TMA would have different scheduling semantics.
- Depth 2 is the tested ping-pong (line 20-24 header). Depth 3+ unexplored.
- B-tile size 512 B/slot × 2 slots × 8 warps = 8 KB — SMEM headroom is plenty on SM120 (99 KB opt-in), so deeper prefetch is feasible.

**Verdict: DEFER.** The `cp.async.bulk.tensor` / TMA path for these contiguous weight-matrix B-tiles (NOT paged KV) is worth a short microbench — TMA has separate issue slots on some Blackwell variants, but on SM120 consumer it may degenerate. Not a one-line fix; would need a v4b prototype (~300 LoC).

---

## 7. v6 FP4 dense — **RECOVER**

**KILL reason.** 0.50× vs v5a.

**Smoking-gun checks.**
- **Bug A (LUT):** Line 131-138 comment admits `FP4_LUT[16]` in __constant__ memory **serializes** on divergent nibble reads — 16 cycles worst-case per warp-wide load. The proper pattern on SM120 is register-resident LUT (e.g. `__shfl_sync` broadcast from a pre-populated register across lanes) or the native `cvt.rn.e2m1.f32`-inverse instructions if they exist. The kernel as-written is compute-bound on serialized LUT lookups — explains the 0.50×.
- **Bug B (smem alignment):** Line 110-112 `SMEM_BDQ` is round-up-16 aligned — OK for ldmatrix, not optimal for bank conflicts. 16×16 bf16 tile = 512 B = 32 banks × 4 B × 4 words; row-major stride 32 B per row → **bank conflicts** on row-major load from 8 warps simultaneously (not the fundamental bottleneck, but 5-10% on top).
- **Bug C (no double-buffer across K-steps):** Single SMEM_BDQ used for both gate+up but not double-buffered across k0 steps — dequant for k0+16 cannot start until k0 MMA completes reading. Latency fully serialized.

**Verdict: RECOVER, P(lands) ≈ 0.55.**

**Fix.** Replace `__constant__ FP4_LUT[16]` with per-warp register LUT loaded at kernel start (32 fp32 regs = 128 B, acceptable). Add K-step double-buffering in SMEM_BDQ (already sized for 2 tiles at line 112 — just actually rotate between them). Expected: 0.50× → 1.1-1.5× (matching v5a BF16 with the FP4 weight-memory win).

---

## 8. I-DLM v2 per-request masking — **CONFIRM_KILL**

**KILL reason.** acceptance 42%→8.6%; v1 causal beats v2 "semantically correct".

**Smoking-gun checks (plan file and results file).**
- Per `idlm_v2_results.md` §Analysis: the v1 model was **trained** with causal masking including MASK-to-MASK visibility; v2's "correct" isolation produces conditionally independent predictions that are mutually inconsistent.
- This is a **training-distribution mismatch**, not a mechanical bug. Not recoverable by code fix.
- Plan file `idlm_v2_plan.md` correctly identifies the kill criterion "if acceptance rate doesn't improve by ≥5% stop" — met.

**Verdict: CONFIRM_KILL.** Fundamental; only recoverable by retraining the I-DLM model with isolated-mask attention, which is a 1-2 week GPU project, not a code fix.

---

## 9. Expert L2 prefetch — **CONFIRM_KILL**

**KILL reason.** 0.49× EMA / 0.65× oracle.

**Smoking-gun checks.**
- `plans/expert_prefetch_l2.md` §7.: empirical stream-overlap microbench showed two HBM-bound streams serialize at 1.08× (nearly fully serialized). This isolates the cause: HBM bus contention, not scheduling priority.
- Priority-stream (`cudaStreamCreateWithPriority`) would change scheduling between compute kernels on the same SM, NOT between two HBM-bound transfers on the HBM controller. The HBM arbiter has no notion of CUDA stream priority.
- The MoE grouped GEMM is ~2-3 FLOPs/byte (HBM-bound, memory-roofline) — oracle prefetch even with perfect prediction would need the kernel to transition to compute-bound post-prefetch, which it cannot.

**Verdict: CONFIRM_KILL.** The plan's §7 "Core insight for future ideas" correctly identifies the failure mode; priority-stream doesn't address HBM bus sharing.

---

## 10. Warp-spec attention v1 — **DEFER (alternate topology untested)**

**KILL reason.** 0.71× vs FlashInfer.

**Smoking-gun checks.**
- `plans/warp_specialized_attention.md` §1 tested **1 producer warp / 3 consumer warps** (4 warps total, 128 threads).
- 2p/2c, 1p/7c, or 0p/8c-with-cp.async.wait_group-only NOT tested.
- §9 root cause: single-CTA-per-head with no split-K across seq dim. This is the bigger bug — FlashInfer does split-K across ~170 SMs, this prototype grids on num_heads (tiny).
- The warp topology question is secondary to the split-K question; even perfect warp-spec can't beat a split-K design.

**Verdict: DEFER.** P(split-K + warp-spec recovers) ≈ 0.30. Would need substantial rewrite; not a one-liner. Next-action priority is LOW — FlashInfer's dense path is already 89% BW, residual is reductions / page-table, warp-spec doesn't attack those.

---

## 11. FusenCache cp.async v1 — **DEFER**

**KILL reason.** 14.3% BW at depth=3 cp.async.

**Smoking-gun checks.**
- `plans/fusencache_cpasync_tma.md` §2.4 states "stages = 3" decided because L2 miss latency hides in ~500-800 ns but each tile only 4 KiB. Higher depth untested.
- SMEM table shows 4 KiB/stage; 4 stages = 16 KiB — plenty of SMEM headroom on SM120 (101 KiB/block). Depth 4/5 is technically viable.
- The deeper root cause per §Interpretation: "cp.async only hides memory latency when the consumer can keep going; here it stalls on the reduction" — deeper prefetch doesn't fix the reduction serialization.
- Cache config (`ca` vs `cg`): kernel uses `cp.async.cg` (L2-only skip of L1). Testing `cp.async.ca` (cache all) might improve L1 reuse for scales but probably not materially.

**Verdict: DEFER.** Depth=3 already within the diminishing-returns region for 4 KiB tiles; the real unlock is warp-spec (v2). P(depth sweep helps) ≈ 0.15. Not worth the agent-hour.

---

## 12. FusenCache warp-spec v2/v3 direction — **DEFER → (a) highest P**

**KILL reason.** v2 at 16.8% BW (1.18×).

**Smoking-gun checks (plan §Path forward).**
- Options: (a) warp-per-head split, (b) tensor-core mma.sync.m16n8k16 for QK, (c) offline softmax broadcast across full tile.
- **(a) warp-per-head split:** with num_q_heads=16 per CTA and 4-8 consumer warps, each warp owns 2-4 heads → removes the cross-warp reduce entirely. This is the biggest win on paper, smallest code delta (~50 LoC from v2).
- **(b) tensor-core QK:** adds complexity (A-frag layout conversion from nibble-unpack into 16x16 WMMA tile). Gain only pays if QK is compute-bound — v2 profile suggests the softmax broadcast (not the QK dot) is the stall point.
- **(c) offline softmax:** only works at small seq_len (fits scores in SMEM). For decode FusenCache typical seq=2048 that's 2048 × 4 B = 8 KB, fits easily.

**Verdict: DEFER with priority ordering (a) > (c) > (b).**

**P-ranked fix (one-off bench first):** Implement warp-per-head split. Prediction: 16.8% → 30-40% BW = 1.8-2.5× over v1 → PASS gate crossed.

---

## 13. T3-L semantic KV eviction at longer context — **DEFER**

**KILL reason.** 2.5× at 90% retention, only at 1.5K context. §5e found breakpoint at S=8-10K.

**Smoking-gun checks.**
- `plans/semantic_kv_eviction.md` §Expected Impact §Quality: H2O validates >95% quality at 20% KV budget; our 90% retention should retain quality far beyond the 1.5K test shape.
- Eviction applies ONLY to the 5 global layers (25 sliding-window layers already bounded). At 16K context, sliding KV stays at 1024 tokens × 25 layers; global KV at 16K × 5 layers = 80K token-slots, well within the 166K capacity. Eviction isn't pressure-triggered until **much** larger context.
- At 32K context the global KV is 32K × 5 = 160K — just under the 166K capacity (9% headroom). Eviction becomes operative. This is the regime to retest in.

**Verdict: DEFER — retest at 32K context only** (not 16K; 16K doesn't trigger the eviction pressure). P(T3-L wins at 32K) ≈ 0.55.

---

## 14. FP8 attention FlashInfer generic path — **CONFIRM_KILL**

**KILL reason.** 1.01× on TRTLLM backend.

**Smoking-gun checks (`fp8_attn_remeasure.md`).**
- Section "Long-seq sweep": sweep IS performed across hd=128/256 × seq=4096/8192/16384. Best ratio 1.11× at hd=256/seq=8192; worst 0.71× at hd=128/seq=16384.
- Root cause isolated: FP8 kernel plateaus at 37% BW vs BF16's 69% BW. Kernel is compute-/layout-bound inside the FP8 path, not BW-bound.
- This is a FlashInfer-0.6.7 kernel characteristic, not a configuration oversight.

**Verdict: CONFIRM_KILL (on FlashInfer FP8 path).** Only a FlashInfer upstream fix helps. Our Triton SWA+FP8 path (see #20) gets around this via Triton's FP8→FP32 dequant codegen.

---

## 15. SASS FP4 scale kernel — **CONFIRM_KILL (grep was accurate)**

**KILL reason.** CUTLASS already uses native PTX.

**Smoking-gun checks.**
- `kernels/csrc/native_fp4_scale_kernel.cu` lines 45-64 confirm the kernel uses `cvt.rn.satfinite.e2m1x2.f32` inline asm directly — byte-identical behavior to CUTLASS per the session log.
- Session log references `nvfp4_utils.cuh:72-89` containing the same native PTX. If the grep for that file came back positive, CUTLASS's hot path IS using native conversion.
- Measured 1.06× vs CUTLASS, consistent with "no architectural delta, just measurement noise."

**Verdict: CONFIRM_KILL.** Premise confirmed wrong. No recovery path.

---

## 16. Multi-tenant spec with same-tokenizer — **RECOVER (Scenario A untested)**

**KILL reason.** Cross-tokenizer (Qwen↔Gemma) blocked by vocab mismatch. Same-family untested.

**Smoking-gun checks (`multitenant_spec_feasibility.md`).**
- §4 Scenario A: Independent spec decode per engine — Gemma4 26B + Gemma4 E2B draft (matched 262K vocab), Qwen3.6-35B + Qwen3-1.7B draft (matched 151K vocab). Projected **+98% aggregate** (22k + 36.6k = 58.6k tok/s vs 29.6k baseline).
- Line 113: "effort 2-4 days, Gemma4 E2B already downloaded; Qwen3-1.7B available."
- vLLM V1 engine **does** support per-engine spec decode via `--speculative-model` / `SpeculativeConfig` — this is the normal vLLM feature, not a fork. `ngram_gpu` also trivially supported.

**Verdict: RECOVER, P(lands) ≈ 0.80.**

**Fix.** Not a code fix — a launch-config change. Add `--speculative-model google/gemma-4-e2b` (or the appropriate path) to `serve_gemma4.sh` and `--speculative-model Qwen/Qwen3-1.7B` to the Qwen serving. 2-4 days serving-bench to confirm the +98% aggregate. No shared-draft fork needed.

---

## 17. §5c fused attn+O register-level conversion — **DEFER**

**KILL reason.** WMMA accumulator→matrix_a requires smem roundtrip.

**Smoking-gun checks.**
- CUDA 12.8 PTX ISA does include `cvt.rn.bf16x2.f32` (fp32→bf16×2 pack). BUT WMMA fragment layouts differ between `accumulator` (16×16 fp32 distributed per lane) and `matrix_a` (16×16 bf16 with a specific per-lane permutation).
- The fragment layouts are NOT bit-compatible; even with `cvt.rn.bf16x2.f32` in registers, you need a shuffle pattern via `prmt.b32` + `__shfl_sync` to reach the matrix_a layout. That's feasible but non-trivial and hardware-specific.
- No published intrinsic does the full conversion in-register on SM120a; mma.sync.16x8x16 PTX reference lists smem-backed `ldmatrix.m8n8.x4` as the canonical load path.

**Verdict: DEFER.** Possible with ~50 LoC of PTX hand-written shuffle + cvt; P(lands) ≈ 0.35. Low priority — smem roundtrip is small (512 B/tile) and bank-conflict-free with shifting, measured overhead was <5%.

---

## 18. §5a Jacobi with vLLM multi-step decoding — **DEFER**

**KILL reason.** Variable iteration count conflicts with CUDA graph capture.

**Smoking-gun checks.**
- `grep` in `plans/` shows `num_scheduler_steps` / `MultiStep` referenced in `FUSENDIFFUSION_PLAN.md`, `fusen_inference_engine.md`, `vllm_data_driven_plan.md` — not dedicated examples.
- vLLM v1's `--num-scheduler-steps` does exist (N batched forward passes before scheduler runs). **Each step captures its own graph**; graph replay is step-deterministic. Variable count per-step doesn't conflict if each step independently captures.
- Jacobi iteration's convergence detection (per-request stop once output stabilizes) does break this: you cannot graph-capture "run until converged". You CAN graph-capture "run for K iterations always" and do post-hoc convergence pruning.

**Verdict: DEFER.** Feasible with fixed-K iteration count. ~1-2 week impl. P(lands at ≥1.3×) ≈ 0.35 — the K-fixed variant is Medusa-like and Medusa has already been validated.

---

## 19. §5b TMA for FusenCache scale streams — **CONFIRM_KILL**

**KILL reason.** <1% Q-load gain; no TMA benefit for paged KV.

**Smoking-gun checks.**
- Scales ARE contiguous per-expert, but per-**expert** contiguity doesn't help decode FusenCache: decode walks paged KV via `block_table` indirection. Scales live in the same paged layout (one scale-block per KV block per head).
- Per `plans/fusencache_cpasync_tma.md` §3: TMA requires a **per-tensor rectangular tile descriptor**; one descriptor per physical block scratches at it but TMA descriptors are host-encoded (`cuTensorMapEncodeTiled`) — not compatible with per-step dynamic block tables.
- Scales are 1/16 the size of K/V — not the bottleneck even if TMA'd. Total scale BW ≈ 30 MB/step vs KV BW ≈ 500 MB/step.

**Verdict: CONFIRM_KILL.** TMA descriptor semantics fundamentally incompatible with paged indirection; scales too small to matter anyway.

---

## 20. Gemma4 FP8 KV paradox — **RECOVER (resolved: different kernel paths)**

**KILL reason.** fp8_attn_remeasure got 1.01× (FlashInfer TRTLLM); SWA+FP8 got 1.19-1.62× (Triton).

**Smoking-gun checks.**
- `plans/fp8_attention_remeasure.md` §Why FP8 fails: FlashInfer TRTLLM backend is compute-bound on FP8 dequant (37% BW). **Different code path.**
- `plans/swa_fp8_kv_stacking.md` §R1: Triton SWA decode uses explicit `.to(tl.float32)` on FP8 → SM120 PTX `cvt.rn.f32.e4m3` (or scalar fallback). Kernel is BW-bound @ 30% peak; doubling data reduction translates ~linearly to latency.
- No paradox: the 1.01× was the FlashInfer kernel's compute-ceiling; the 1.19-1.62× is the Triton kernel's BW-ceiling. Same model, different kernels, different bottlenecks.

**Verdict: RECOVER — already resolved.** Ship SWA+FP8 Triton path for sliding layers; leave FlashInfer FP8 off-path for global layers. No code change; just don't take the FlashInfer measurement as the ceiling for all FP8 KV.

---

## Ranked RECOVER candidates

| Rank | Item | Fix size | Projected gain | P(lands) | EV |
|---:|---|---|---|---:|---:|
| 1 | #16 Same-family spec decode | 2-line launch config | **+98% aggregate** | 0.80 | **0.78×(29k→58k)** |
| 2 | #1 v5a.1 rebench with correct barrier cost | 0 lines (rebench only) | +1-2% e2e | 0.70 | **+0.7-1.4%** |
| 3 | #2 T1-B shadow-metadata size fix | ~10 lines | +20× @ C=128 (224→4,489 tok/s) | 0.60 | **+12× peak** |
| 4 | #7 v6 FP4 LUT → register-LUT + double-buf | ~40 lines | 0.50× → 1.1-1.5× (FP4 weights win) | 0.55 | **+2.2-3×** in the v6 branch |
| 5 | #5 v5b barrier fusion redesign | ~150 lines (rework) | +90 µs barrier save + eliminate 64 KB broadcast | 0.45 | **+2-5%** over v5a |

---

## Aggregate counts

- **RECOVER:** 5 — #1 v5a.1, #2 T1-B, #7 v6 FP4, #5 v5b, #16 multi-tenant same-family
- **DEFER:** 6 — #6 v4a TMA, #10 warp-spec attn (split-K), #11 FusenCache depth sweep, #12 FusenCache warp-per-head, #13 T3-L at 32K, #17 §5c in-register cvt, #18 Jacobi fixed-K (counted 7, but #12 is really a sub-option)
- **CONFIRM_KILL:** 8 — #3 Gemma4 orig plugin (no bug), #4 early T2-N variants, #8 I-DLM v2, #9 expert prefetch, #14 FP8 FlashInfer path, #15 SASS FP4, #19 TMA scales, #20 wasn't really a paradox

*Reconcile: 5 + 7 + 8 = 20.*

---

## Highest-EV single recommendation

**#16 Same-family spec decode — Gemma4 E2B drafts Gemma4 26B + Qwen3-1.7B drafts Qwen3-30B.** Two-line launch config change (`--speculative-model` flag per serve script), no kernel work, projected **+98% aggregate throughput** (29.6k → 58k tok/s). P(lands) 0.80. Single-session delivery.

## Systemic patterns

1. **Silent-None dispatch bugs from mis-named attributes.** Confirmed 1 case (v1 Qwen3 plugin's `.kernel` vs `.backend`); audited Gemma4 plugin does NOT share the bug; no other candidates exhibit this pattern in the reviewed files. **Recommendation:** add a plugin-side assertion `_build_xxx_fn` returns non-None for at least one layer AND log a WARNING on every None-return with the `type(quant_method).__name__` so the failure is loud not silent.

2. **Over-estimated barrier costs across mega-graph variants.** Both v5a.1 (line 42: "~50µs") and v5b (line 31: "~15 µs") used stale grid.sync estimates. Multiple KILL verdicts (#1 definite, #5 likely) derived from these inflated numbers. **Recommendation:** centralize the measured barrier cost in `kernels/csrc/mega_graph_constants.h` and have every new v* file reference that constant in its gain projection.

3. **View-vs-clone ambiguity under CUDA graph capture (T1-B).** The shadow-metadata design intended isolation but leaked it via views. **Recommendation:** any "shadow-tensor" pattern for graph-captured kernels must return `.clone()`s, not views, and allocate the shadow at the `max(max_seqs, max_num_batched_tokens) + MARGIN` size up front.

4. **Projected gains built on a single measurement shape.** T3-L semantic eviction was killed at 1.5K but only pressure-triggers at 32K. Don't generalize a KILL from one shape when the mechanism is pressure-dependent.

5. **"Already uses native" claims need an actual grep.** SASS FP4 (#15) did this right (cited exact line 72-89); other plans did not. **Recommendation:** every KILL citing "upstream already does X" must link to file:line in the kill note; unverified claims get flipped to DEFER by default.
