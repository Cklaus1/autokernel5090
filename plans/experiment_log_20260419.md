# Experiment Log — 2026-04-19

## Hardware / Context Entering Session

- 2× RTX PRO 6000 Blackwell (SM120a), 96 GB GDDR7 each, 1,792 GB/s HBM peak, 188 SMs
- PCIe-only (no NVLink), AMD 9950X3D host, WSL2
- **Peak banked entering session:** T2-N Qwen3 18,756 gen tok/s @ C=512 (vLLM 0.17 patched stack)
- **Major infra wins today:**
  - `.claude/settings.json` sub-agent allowlist (15-entry; docker/nvidia-smi/curl/ss/pip/nvcc/python3)
  - `unset NAME` patched into 7 launchers (defensive NAME=gpumaster guard)
  - `vllm-0.17.0-patched-backup/` + `vllm-0.18.1-patched-backup/` moved to `_archive/` (2.8 GB freed)

---

## Completed Experiments

### Banked Wins

| ID | Experiment | Result | Headline | Source rows |
|---|---|---|---|---|
| **W3_CA** | Fused-norm Qwen3 v2 | **PASS** | **23,254 gen tok/s @ C=1024 (+19% over 18,756 banked)** | `W3_CA_fused_norm_qwen3_v2` |
| **W1_4a** | SWA + FP8 KV Gate #4 | **PASS** | C=4 421 / C=8 532 / C=16 996 / C=32 1,342 gen tok/s vs BF16 baseline 260/487/772/1,127. **2× KV capacity at 32K max-len, zero OOM** | `W1_4a_swa_fp8_kv_gate4` |
| **T2-N clean rerun** | Single-container baseline verify | **PASS** | C=512 = 19,558 gen / 28,114 total tok/s (+4.3% over banked 18,756). Earlier 9,942 run was WSL2 cross-GPU leak, not regression | see roadmap |
| **W1_4e** | LMCache SM120 rebuild + bench | **PASS** | P50 159→102 ms (1.56×), P99 160→110 ms (1.45×), throughput 1,061→1,347 tok/s. Rebuild recipe: git clone v0.4.3 + torch `_check_cuda_version` monkey-patch + `--no-deps` + sortedcontainers/nvtx/aiofile | `W1_4e_lmcache_smoke_rebuilt` |

### Fused-norm Qwen3 v2 — single-line bug fix detail

v1 plugin (`wire_fused_norm_fp4_qwen3.py`) used `getattr(quant_method, "kernel", None)` which always returned `None` because the attribute is `.backend`, not `.kernel`. All 48 per-layer fusions were silently disabled for weeks. v2 fix: replace with `hasattr(quant_method, "backend")` check + `.backend` dispatch. Recovered the dead fusions and pushed peak from 18,756 → 23,254 gen tok/s (+19%). This is now the canonical silent-None dispatch example in `KILL_PATTERNS.md §P1`.

### SWA + FP8 KV Gate #4 — graph-capture guards detail

`patches/swa_gemma4_plugin.py` + `swa_sparse_plugin.py` patched with `torch.cuda.is_current_stream_capturing()` guards. 24K prompt served at 32K max-len with zero OOM — 2× effective KV capacity verified. All four concurrencies PASS correctness.

---

## KILLs

### Gate Tests / Kills

| ID | Experiment | Verdict | Measured | Root Cause |
|---|---|---|---|---|
| **W1_3a** | v5b barrier fusion (4/layer) | **KILL 1.010×** | 8,726 µs vs v5a 8,811 µs | grid.sync = **~3 µs/barrier on SM120** (not 15/50/278 µs). 30 barriers = 90 µs saved. Below noise floor. Roadmap v5b-v5a.2 compound killed. Real bottleneck: WMMA B-frag hidden-strided load latency at N=16 tiles |
| **W3_P1** | v6 FP4 dense proxy | **KILL 0.50×** | 17.7 ms vs v5a 8.8 ms | Inline dequant serial smem roundtrip adds ~460 µs/layer. v5a not BW-bound at 26% HBM — FP4 compression cuts a ceiling that isn't binding. `mma.sync.e2m1` NOT supported on sm_120 |
| **v5a.1** | Multi-head pack (+1 barrier) | **KILL 0.95×** | Prior session | +1 barrier attributed as +450 µs (used 15 µs estimate). At calibrated 3 µs/barrier, actual cost was +90 µs. Worth rebench; KILL may be wrong |
| **v4a** | cp.async B-prefetch | **KILL 0.92×** | Prior session | B-load hypothesis REJECTED. `wmma::load_matrix_sync` already overlaps gmem→reg via warp-ILP. Adding smem roundtrip costs more than it saves |
| **W4_T5.6** | T2-N → Gemma4 cross-apply | **KILL 0.62×** | 1,113 gen vs 1,798 baseline | Tile shapes and dispatch tuned for Qwen3-30B-A3B MoE don't transfer to Gemma4-26B-A4B. Audit projected +34% (P=0.70) — delivered **regression**. P11 validated |
| **W4_5a** | Qwen3 ngram-GPU spec | **KILL 0.46-0.85×** | C=1→128 all regressed | Math-bench prompts have low repeating n-gram rate. Draft+verify overhead dominates. Audit projected 1.2-1.5× — delivered regression at all concurrencies |
| **W2_5c** | Fused attn+O-proj | **DEFER** | Feasibility only | WMMA `accumulator<float>` → `matrix_a<bf16>` has no in-register conversion path; smem roundtrip required. Cross-SM barrier integration also problematic |

---

## 3-Wave Kill Audit (CPU-Only Code Review)

### Top-20 Audit (`plans/kill_audit_20260419.md`)

**Tag:** `W4_audit_20260419_kills_top20`

| Verdict | Count | Notable |
|---|---|---|
| RECOVER | 5 | #16 same-family spec (+98% aggregate, P=0.80), #1 v5a.1 rebench (0-line fix), #2 T1-B shadow-tensor (10-line fix), #7 v6 FP4 LUT (40-line fix), #5 v5b redesign (150-line) |
| DEFER | 7 | v4a TMA, warp-spec attn split-K, FusenCache depth sweep, warp-per-head, T3-L at 32K, §5c in-register cvt, Jacobi fixed-K |
| CONFIRM_KILL | 8 | Gemma4 orig plugin (no bug), early T2-N variants, I-DLM v2, expert prefetch, FP8 FlashInfer path, SASS FP4, TMA scales, FP8 paradox resolved |

**Highest-EV item:** #16 same-family spec decode — `--speculative-model google/gemma-4-e2b` on Gemma4 + `Qwen/Qwen3-1.7B` on Qwen3. 2-line launch config, projected **+98% aggregate** (29.6k → 58k tok/s), P=0.80.

### Tier 2+3 Audit (`plans/tier2_3_audit_20260419.md`)

**Tag:** `W4_audit_tier23_20260419`

| Verdict | Count | Notable |
|---|---|---|
| RECOVER | 6 | #6 LMCache disagg (~15-line shell, 10× P99 TTFT), #5 FP8 decode silent-fallback (3-line warn → assert), #17 SWA BW headroom (4-line hints + num_warps), #25 WSL2 GPU isolation (4-line env), #23 dead-code patches audit, #19 persistent MoE revisit at 3 µs |
| DEFER | 7 | DDTree GPU top-k, TriAttention, ASI A+D C=8, C>4 patches, m16n8k16 shape, v4a depth 3+, T2-N ceiling |
| CONFIRM_KILL | 6 | T3-M router prediction (MI 170× below gate), T3-O L2 pinning (aux-loss enforced uniform), I-DLM graphs (arch), Qwen3.6 upgrade path (throughput trade), v3 variants, dual-model 29k stale |
| ENDORSE | 5 | v5a fused rmsnorm firing confirmed, FusenCache C++ baseline realistic, RMSNorm+FP4 TMA no headroom, EAGLE3 pipeline correct, I-DLM v1 +17% delta explained |

**Key finding:** FP8 decode silent-fallback bug — `patches/fp8_decode_monkey_patch.py:319-335` silently re-routes to FlashInfer on `FIDecode` metadata path (SM120 default). Banked T2-I 1.3-1.8× may be unmeasured. Promote debug → warning at line 324.

**Stale number flagged:** dual-model 29,655 tok/s aggregate is stale post `60090fb` upgrade to Qwen3.6-FP8. Current aggregate: 12,229 + 5,895 = **18,124 tok/s**.

### Tier 4+5+6 Audit (`plans/tier456_audit_20260419.md`)

**Tag:** `W4_audit_tier456_20260419`

| Verdict | Count | Notable |
|---|---|---|
| RECOVER | 5 | T4.3 sub-agent Docker allowlist (5-line settings), T5.6 T2-N→Gemma4 (2-line env — TESTED, regressed 0.62×), T4.1 `unset NAME` (8-line hygiene), T4.4 UUID GPU-check boot line, T6.10+T6.11 archive |
| DEFER | 5 | SWA on other models (no SWA targets locally), FP8 KV matrix sweep, upstream patch drift, vllm_patches 2026-04-07 prose |
| CONFIRM_KILL | 2 | T4.5 settings conflict (no conflict; sub-agents inherit wrong file), T5.7 Gemma4 plugin correct |
| ARCHIVE | 4 | vllm-0.17.0-patched-backup (1.2 GB, zero refs), vllm-0.18.1-patched-backup (1.6 GB, zero refs), fusen_solver/ (no callers), fusencache/ top-level (superseded) |
| KEEP | 1 | flashinfer-0.6.4-patched-backup (52 MB, cited in plan doc) |

**Infrastructure shipped from this audit:**
- `.claude/settings.json`: 15-entry allowlist with `Bash(docker*)`, `Bash(nvidia-smi*)`, `Bash(curl*)`, `Bash(ss*)`, `Bash(pip*)`, `Bash(nvcc*)`, `Bash(python3*)` — unblocks future sub-agent dispatches
- `unset NAME` added to 7 launchers
- 2.8 GB archived (`_archive/` or removed)

### Audit yield summary (3 waves, 60 candidates)

| Wave | RECOVER | DEFER | CONFIRM_KILL | ENDORSE/ARCHIVE | Total |
|---|---|---|---|---|---|
| Top-20 | 5 | 7 | 8 | — | 20 |
| Tier 2+3 | 6 | 7 | 6 | 5 | 24 (+ 1 overlap) |
| Tier 4+5+6 | 5 | 5 | 2 | 5 | 17 (infra) |
| **Total** | **16** | **19** | **16** | **10** | **~60** |

**Tested from audit PROCEEDs:** 3 tested, 1 validated (fused-norm v2), 2 regressed (T5.6 + §5a). ~33% validation rate on audit PROCEED verdicts.

---

## Key Insights

1. **grid.sync = ~3 µs/barrier on SM120a** (not 15 µs from v5a.1 comment, not 50 µs from v5b reasoning, not 278 µs from Discovery #35). The 278 µs number was SM work imbalance, not barrier mechanism. At 3 µs × 30 barriers saved = 90 µs — below noise on a 9 ms step. Invalidates v5b as a primary lever. KILL_PATTERNS.md §1 updated.

2. **Audit PROCEED verdicts are proposals, not predictions (P11).** Bug-fix category (specific line + specific error) validates ~80%. Cross-apply and literature categories validate ~30%. T5.6 T2-N→Gemma4 (audit P=0.70, projected +34%) delivered **0.62× regression** — tile shapes tuned for Qwen3 don't transfer. §5a ngram spec (audit P=0.80, projected 1.2-1.5×) delivered **0.46-0.85× at all concurrencies** — math-bench prompts have low n-gram repeat rate. KILL_PATTERNS.md §P11 added.

3. **Silent-None dispatch (P1) is a repeat bug class.** Fused-norm v1 `.kernel` typo is the canonical example. Same pattern found in `fp8_decode_monkey_patch.py:319-335` — FIDecode path silently bypasses our FP8 kernel, routing back to FlashInfer. Debug log at line 324 provides no signal. Rule: every monkey-patch with conditional dispatch must log at WARNING when falling back AND every launcher must assert the patch banner appeared.

4. **WSL2 `--gpus 'device=N'` isolation leaks** — confirmed again. Both containers see both GPUs unless `CUDA_VISIBLE_DEVICES=N` is also set. The earlier T2-N "regression" to 9,942 was cross-GPU leak from a parallel container. T2-N clean rerun: 19,558 gen tok/s (+4.3% over banked 18,756). All multi-GPU benches need serial execution or explicit `CUDA_VISIBLE_DEVICES`.

5. **LMCache requires SM120 source rebuild.** PyPI `lmcache==0.4.3` `c_ops.so` has no SM120 cubin. Fix: git clone + torch `_check_cuda_version` monkey-patch + `pip install . --no-build-isolation --no-deps` + add sortedcontainers/nvtx/aiofile. Delivers 1.56× P50 TTFT, 1.27× aggregate on shared-prefix workload. Delta vs 7.6× projection: vLLM native prefix cache already handles warm hits; LMCache gap is largest at cold/multi-instance/longer-prefix.

6. **FP4 compression (v6) is not the right lever at H=2048.** Dense proxy runs at 26% HBM BW — not BW-bound. FP4 cuts data volume but the ceiling is compute/latency, not HBM. At real Gemma4 H=4096 × 128-expert MoE structure the FP4 win re-emerges, but the dense-proxy measurement methodology was too optimistic.

7. **dual-model 29,655 tok/s aggregate is stale** post `60090fb` Qwen3.6-FP8 upgrade. Current: 18,124 tok/s. Plans referencing 29k should be flagged. The 60090fb upgrade traded ~39% throughput for quality.

---

## Session Peak

**23,254 gen tok/s / 33,428 total tok/s** @ Qwen3-30B-A3B NVFP4 C=1024 (fused-norm v2 + T2-N stack, GPU 1, single-container clean state).

---

## Meta-Lessons → `plans/KILL_PATTERNS.md`

Updates logged to KILL_PATTERNS.md:

- **§1 Calibration:** grid.sync = 3 µs (invalidates 15/50/278 µs citations in older plans)
- **§P11 (new):** Audit PROCEED verdicts are proposals. Bug-fix category ~80% realized; cross-apply/literature ~30% realized. Half the projected P for categories 3+4.
- **Repeat bug class P1:** FP8 decode silent-fallback in `fp8_decode_monkey_patch.py:319-335` — same architectural pattern as fused-norm v1 typo. Second confirmed instance.
- **§1:** `lmcache==0.4.3` PyPI wheel has no SM120 cubin. Source rebuild required.
- **§P10:** new checkpoint architecture mismatch pattern from LMCache smoke (`SDARForCausalLM` unsupported).

---

## Untested RECOVER Candidates (Queued for Next Wave)

Priority-ranked by expected value:

| Rank | ID | Fix | P(lands) | EV |
|---:|---|---|---:|---|
| 1 | **#16 Same-family spec decode** | Add `--speculative-model` to serve scripts (2 lines) | 0.80 | **+98% aggregate** (29.6k → ~58k tok/s) |
| 2 | **#6 LMCache disagg** | `serve_disaggregated_lmcache.sh` swapping P2pNcclConnector → LMCacheConnectorV1 (15 lines) | 0.75 | **10× P99 TTFT** (WSL2-compatible ASI-1 substitute) |
| 3 | **T1-B FusenCache shadow-tensor** | Size shadows at `max(max_seqs, max_num_batched_tokens)+MARGIN`; `.clone()` not view; assert at copy (10 lines) | 0.60 | Recover 4,489 tok/s @ C=128 |
| 4 | **#5 FP8 decode silent-fallback** | Promote `logger.debug` → `logger.warning` at `fp8_decode_monkey_patch.py:324`; add assert-mode env var (3 lines) | 0.55 | Unlock banked 1.3-1.8× if kernel was inert |
| 5 | **#17 SWA BW headroom** | `num_warps=4→8` + `tl.multiple_of(k_addrs, 16)` in `swa_decode.py` (4 lines) | 0.50 | +15-25% decode tok/s |
| 6 | **#22 Dual-model re-bench** | Revert `serve_dual_model.sh` model path to Qwen3-30B-NVFP4 (2 lines) OR re-bench current config to know true aggregate | 0.85 | Validates ±39% throughput delta |

---

## Infrastructure Shipped

### `.claude/settings.json` allowlist
15-entry allowlist including `Bash(docker*)`, `Bash(nvidia-smi*)`, `Bash(curl*)`, `Bash(ss*)`, `Bash(pip*)`, `Bash(nvcc*)`, `Bash(python3*)`. Unblocks future sub-agent dispatches. Prior blocker: T2-H Sonnet sub-agent died on Docker permission denial 2026-04-18.

### `unset NAME` in launchers
Added to 7 launchers: `launch_qwen3_fused_norm_fp4.sh`, `launch_qwen3_fused_t2n.sh`, `launch_gemma4_swa.sh`, `launch_gemma4_swa_fp8.sh`, `launch_prefix_aware_sched.sh`, `launch_gemma4_lmcache_hierarchy.sh`, `launch_lmcache_smoke_sm120.sh`. Prevents `NAME=gpumaster` parent-shell clobber from silently destroying running containers.

### Archive cleanup
`vllm-0.17.0-patched-backup/` (1.2 GB) + `vllm-0.18.1-patched-backup/` (1.6 GB) moved to `_archive/`. Zero live references in launch scripts. Live images (`vllm-fusencache:latest`, `vllm-fusencache-gemma4fix:latest`, `vllm-built:latest`) carry their own internal `/build/vllm/` and do not bind-mount the backups.

---

## Result Rows Appended

`results.tsv` W1/W2/W3/W4 rows confirmed:

| Row tag | Experiment | Verdict | Headline |
|---|---|---|---|
| `W1_3a_v5b_barrier_fusion` | v5b 4-barrier mega-graph | KILL 1.010× | grid.sync = 3 µs, not 15/50 µs |
| `W1_4a_swa_fp8_kv_gate4` | SWA + FP8 KV Gate #4 | PASS | C=32 = 1,342 gen tok/s; 2× capacity |
| `W1_4e_lmcache_smoke_rebuilt` | LMCache SM120 rebuild | PASS | 1.56× P50, 1.27× throughput |
| `W2_5c_fused_attn_oproj` | Fused attn+O-proj | DEFER | No in-register acc→matrix_a path |
| `W3_P1_v6_fp4_mlp_dense` | v6 FP4 dense proxy | KILL 0.50× | Serial smem dequant; non-BW-bound |
| `W3_CA_fused_norm_qwen3_v2` | Fused-norm v2 stack | PASS | **23,254 gen tok/s @ C=1024** |
| `W4_T5_6_t2n_gemma4_crossapply` | T2-N → Gemma4 | KILL 0.62× | Tile shapes don't transfer |
| `W4_5a_qwen3_ngram_spec` | Qwen3 ngram-GPU spec | KILL 0.46-0.85× | Low n-gram rate on math prompts |

---

## Session Mechanics

- 3-wave CPU-only audit dispatched covering 60 candidates (Tiers 1-6)
- 2 GPU experiments validated from audit PROCEEDs: T5.6 (KILL), §5a (KILL)
- 1 bug-fix validated: fused-norm v2 (PASS, +19%, session peak)
- Infrastructure wave: settings.json, launcher hygiene, archive cleanup
- No commits. GPUs idle at session close. Audit files: `plans/kill_audit_20260419.md`, `plans/tier2_3_audit_20260419.md`, `plans/tier456_audit_20260419.md`.

---

## Next-Session Priority Queue

1. **Same-family spec decode** — `--speculative-model` flag per serve script, no kernel work. Highest single-action EV in the backlog (+98% aggregate projected, P=0.80). One session.
2. **LMCache disagg bench** — WSL2-compatible ASI-1 substitute. `serve_disaggregated_lmcache.sh` already designed; needs GPU smoke (blocked this session by audit scope). P99 TTFT target: 100 ms @ C=8.
3. **T1-B FusenCache shadow-tensor fix** — 10-line fix with 3 identified bugs (size mismatch line ~486, `_max_tokens` undersize line ~481, view-vs-clone line 988). Target: recover 4,489 tok/s @ C=128.
4. **FP8 decode silent-fallback confirm** — 3-line warning + bench to verify T2-I kernel actually fired during "14/14 PASS" bench. If inert, the 1.3-1.8× is still on the table.
5. **SWA BW headroom** — `num_warps=4→8` + `tl.multiple_of` hints in `swa_decode.py`. 4-line change, P=0.50, +15-25% decode.
