# KILL_PATTERNS.md — Experiment Hygiene + Silent-Bug Catalog

**Purpose:** prevent mis-attributed KILL verdicts and catch silent bugs before they cost weeks. Every experiment dispatch must consult this file; every KILL verdict must verify against the pre-KILL checklist.

**Provenance:** built from autokernel session 2026-04-18/19 recoveries where a single-line `getattr()` typo killed a +71% throughput win for weeks, and barrier-cost miscalibration by 5-17× invalidated multiple downstream KILLs.

---

## 1. Calibration constants (measured truths on this silicon)

These values override any assumption in older plans, code comments, or agent reasoning. Cite this file when they're used.

| Constant | Measured | Source | Invalidates |
|---|---|---|---|
| `grid.sync` cost on SM120a cooperative kernel | **~3 µs/barrier** | v5b audit 2026-04-19 | Discovery #35 "278 µs", v5a.1 "50 µs", v5b original "15 µs" |
| WSL2 `--gpus 'device=N'` isolation | **LEAKS** — pid resident on all GPUs | 2026-04-18 T2-N regression false alarm | All "bench in parallel on 2 GPUs" plans |
| SM120a WMMA B-fragment load | **Already overlaps gmem→reg via LSU + warp-ILP** | v4a cp.async KILL 2026-04-18 | All cp.async-prefetch optimization proposals at small N=16 tiles |
| SM120a tensor-core coverage | **NO WGMMA, NO FP8 MMA, NO native FP4 MMA** | FA3 port 2026-04-18 + v6 FP4 KILL | Any port from Hopper `mma_sm90` code |
| FP4 dequant-in-WMMA-loop (Option A) at H=2048 dense proxy | **2× SLOWER (0.50×)** — serial smem roundtrip dominates | v6 FP4 2026-04-19 | "FP4 will compound 4× on any shape" claims |
| SM120a PTX `cvt.rn.satfinite.e2m1x2.f32` | ENCODE ONLY (fp32→fp4×2) | v4b prereq audit | Assumed bidirectional encode/decode symmetry |
| `mma.sync.e2m1` (native FP4 MMA) on sm_120 | **NOT SUPPORTED** by ptxas | v6 FP4 agent 2026-04-19 | CUDA 12.9+ fp4 MMA path proposals |
| Mega-graph dense-proxy H=2048 HBM saturation | **26% (not BW-bound)** | v5a + v5b 2026-04-19 | FP4/compression wins at this shape |
| vLLM main `gemma4_mm.py:1257 is_multimodal.to()` | **Race crashes C>4** | T1-B 2026-04-18 | All "main vllm works" assumptions for Gemma4 |
| `lmcache==0.4.3` PyPI wheel on SM120 | **No cubin** — rebuild from source required | 2026-04-18 smoke KILL | PyPI install works on SM120 |
| `grid.sync` spread: WMMA-idle vs WMMA-active vs cp.async-active | **PENDING** — run bench1 | W6 calib bench 2026-04-18 | Mega-graph §2.2 barrier budget table (uses single-condition 3 µs) |
| `wmma::load` vs `cp.async` crossover B-tile | **PENDING** — run bench2 sweep | W6 calib bench 2026-04-18 | v4a KILL scope: "valid at N=16" — does NOT extend to B-tile ≥64 until verified |
| `mma.sync.m16n8k16` BF16 TOPS on SM120a | **PENDING** — run bench3 | W6 calib bench 2026-04-18 | All TFLOPS projections citing "Hopper-class ~1000 TOPS" |
| `cvt.rn.satfinite.e2m1x2.f32` cycles/conversion | **PENDING** — run bench4 | W6 calib bench 2026-04-18 | v6 FP4 KILL "serial smem roundtrip" root-cause attribution |
| `__shfl_sync` vs `atomicAdd` smem on SM120a | **PENDING** — run bench5 | W6 calib bench 2026-04-18 | persistent_moe_dispatch.cu W5_D1 race fix cost model |
| HBM sustained BW at M=1 decode (strided access) | **PENDING** — run bench6 | W6 calib bench 2026-04-18 | "26% BW is kernel-specific headroom" assumption; FP4 compression win projections |
| CUDA graph crash node count on SM120a | **PENDING** — run bench7 | W6 calib bench 2026-04-18 | Discovery #56-58 "~450 nodes" estimate; persistent-kernel motivation |
| TMA `cp.async.bulk.tensor` on SM120a | **PENDING** — run bench8 (compile test first) | W6 calib bench 2026-04-18 | W5_5b "guarded by arch>=900, passes for 1200" conclusion |

---

## 2. Silent-bug detection patterns

### P1 — Silent-None dispatch ("the Qwen3 fused-norm bug")

**Symptom:** plugin registers cleanly, startup banners appear, but the fast-path never fires. Measured throughput matches BASELINE not PATCHED.

**Example:** `patches/wire_fused_norm_fp4_qwen3.py` (fixed in v2):
```python
kernel = getattr(quant_method, "kernel", None)  # always None; attribute is .backend
if kernel is None:
    return None  # silently disables all 48 per-layer fusions
```

**Detection rule:** every `getattr(obj, "attr", default)` in plugin/dispatch code MUST either:
- Assert the attribute exists (`assert hasattr(obj, "attr"), f"missing .attr on {type(obj).__name__}"`)
- Log the resolved class name AND resolution result on first call
- Document the fallback semantics in a comment with the expected class

**Mitigation pattern (use this idiom):**
```python
if not hasattr(quant_method, "backend"):
    logger.warning(f"[plugin] fallthrough on {type(quant_method).__name__} — expected .backend attr")
    return None  # fallthrough expected for non-NVFP4 quant paths
backend = quant_method.backend
```

---

### P2 — Plugin banner ≠ fusion active

**Symptom:** `[PLUGIN] Patched foo_bar via plugin: active` appears in logs, but no per-layer callable is actually built. Easy to confuse "module registered" with "hot path covered".

**Detection:** after first forward, assert `sum(1 for layer in model.layers if layer._fused_fn is not None) == expected_count`. Log the count at first forward.

---

### P3 — Parent-shell env var override (`NAME=gpumaster`)

**Symptom:** two launcher invocations produce containers with the same name (from parent shell). Second `docker rm -f "${NAME}"` at launcher start destroys the first container.

**Detection:** every launcher that uses `NAME="${NAME:-default}"` MUST either:
- Explicitly `unset NAME` at top of script, OR
- `docker ps` after launch and assert the intended `default` name appears

**Example bug (2026-04-18):** SWA baseline launcher and fused-norm launcher both ran with `NAME=gpumaster` inherited from parent shell; the second killed the first silently.

---

### P4 — Cross-GPU contention (WSL2 `--gpus device=N` leak)

**Symptom:** measured throughput varies wildly across runs (9,942 → 6,681 → 3,979 on repeated T2-N benches 2026-04-18) even with identical config. Root cause: vLLM process bleeds across GPUs despite docker isolation.

**Detection before benching:**
```bash
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F, '{print $1}' | sort | uniq -c | awk '$1 > 1 {print "WARN: pid " $2 " on " $1 " GPUs"}'
```
If any pid appears on >1 GPU UUID, tear down ALL containers and rerun in strict serial.

**"Durable workaround" INVALIDATED 2026-04-19 evening:** combining `--gpus 'device=N'` + `-e NVIDIA_VISIBLE_DEVICES=N` + `-e CUDA_VISIBLE_DEVICES=0` **DOES NOT work on our WSL2 setup.** Tested via dual-model bench: both containers (Gemma4 on declared GPU 0, Qwen3.6 on declared GPU 1) collapsed onto GPU 0 (96.7 GB used), GPU 1 empty (0 MiB). `docker inspect` confirmed DeviceIDs `[0]` and `[1]` were set on each — runtime ignored them. UUID check banners returned the same UUID for both containers. See `W5_25_wsl2_isolation_FAIL` row in results.tsv.

**Root cause (W6 investigation 2026-04-18):** WSL2 sets `no-cgroups = true` in `/etc/nvidia-container-runtime/config.toml` because the WSL2 kernel does not expose cgroup device controllers. This means `--gpus 'device=N'` affects library injection only — it cannot restrict access to `/dev/nvidia*` device files. Both containers see all GPUs. The specific bug: `serve_dual_model.sh` used `CUDA_VISIBLE_DEVICES=0` in BOTH containers. When cgroup isolation is absent, CUDA_VISIBLE_DEVICES=0 maps to host GPU 0 in both containers → both land on GPU 0. See `plans/wsl2_gpu_isolation_investigation.md` for full analysis.

**Corrected workaround (W6, pending verification — run `test_wsl2_gpu_isolation.sh e` first):**

When `no-cgroups = true`, use the host GPU index in `CUDA_VISIBLE_DEVICES`, NOT the container-internal re-mapped index:

```bash
# GPU 0 container:
--gpus '"device=0"' -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0

# GPU 1 container:  ← KEY FIX: CUDA_VISIBLE_DEVICES=1 not 0
--gpus '"device=1"' -e NVIDIA_VISIBLE_DEVICES=1 -e CUDA_VISIBLE_DEVICES=1
```

**Why:** with no cgroup restriction, both containers see all GPUs. `CUDA_VISIBLE_DEVICES=1` tells CUDA "use the second device in host enumeration order" → host GPU 1. This uses CUDA's own device selection as the isolation mechanism. This change is already staged in `serve_dual_model.sh` fix template; run `test_wsl2_gpu_isolation.sh e` to confirm distinct UUIDs before any dual-model bench.

**Fix template index update:**

| Pattern | Template | Location |
|---|---|---|
| WSL2 GPU isolation (P4) — **CORRECTED** | `--gpus '"device=N"' -e NVIDIA_VISIBLE_DEVICES=N -e CUDA_VISIBLE_DEVICES=N` (N = HOST index, same value in all three) | any docker run targeting a specific GPU on WSL2 |

**Fallback if CUDA_VISIBLE_DEVICES=N still fails:** apply `accept-nvidia-visible-devices-envvar-when-unprivileged = true` in `/etc/nvidia-container-runtime/config.toml` + `sudo systemctl restart docker`. See `plans/wsl2_gpu_isolation_investigation.md §Patch A`.

**Until verified, treat any dual-GPU concurrent bench as suspect.** Single-container benches remain valid (no GPU choice needed). Run `test_wsl2_gpu_isolation.sh e` to confirm fix before dual-model bench.

---

### P5 — Inflated barrier-cost reasoning

**Symptom:** a kernel KILL verdict is attributed to "+N µs from added grid.sync(s)". Value is copied from a plan file, not measured.

**Detection:** if any KILL reasoning cites a barrier cost, the cost MUST come from a differential measurement on this silicon (ideally within the last 3 days). Use the §1 calibration constant (`~3 µs`). Recompute the verdict with the correct constant before finalizing KILL.

**Example:** v5a.1 KILL used 50 µs × 30 layers = +1500 µs penalty → KILL. At 3 µs × 30 = +90 µs, the Amdahl fix might have been net-positive.

---

### P6 — "Upstream already does X" without file:line verification

**Symptom:** SASS FP4 kernel KILL'd because "CUTLASS already uses native PTX `cvt.rn.satfinite.e2m1x2.f32`". Claim was verified via file:line grep (`nvfp4_utils.cuh:72-89`) — good practice.

**Detection rule:** any KILL verdict that includes "upstream already does X" MUST cite a specific file:line and verify the code is in the hot path (not a fallback branch). If the reviewer can't grep the exact citation, reopen the KILL.

---

### P7 — Single-shape KILL over-generalization

**Symptom:** experiment KILL'd at one workload shape; verdict applied to ALL shapes.

**Example:** T3-L semantic KV eviction KILL'd at 1.5K context (MARGINAL 2.5×). Breakpoint analysis (§5e) showed the approach pays off at 8-10K+ and compounds to +13-17% at 32K. KILL verdict incorrectly generalized.

**Detection:** every KILL must name the tested shape explicitly AND either (a) test a second regime shape, or (b) document why the KILL generalizes to all shapes with math.

---

### P8 — View-vs-clone under CUDA graph capture

**Symptom:** tensor views aliased across graph capture iterations corrupt state (T1-B shadow-tensor `.copy_()` bug; gemma4_mm.py:1257 `is_multimodal.to()` race).

**Detection:** inside any `torch.cuda.is_current_stream_capturing()` path, any tensor operation that would write to a shared buffer MUST use `.clone()` or a pre-allocated per-iter scratch, not a view.

---

### P9 — Cold-start / warmup artifact

**Symptom:** first bench run at a concurrency level is anomalously low (C=512 first = 10,393; second = 19,558).

**Detection:** benches at new concurrency MUST warm up (1-2 prior runs) before recording. Never trust the first bench at a new C.

---

### P11 — Audit-PROCEED verdicts are proposals, not truth

**Symptom:** a KILL-audit agent flags an experiment as RECOVER/PROCEED with high P (0.70+), projected +X% gain. In reality, when empirically tested, the "fix" regresses.

**Examples (confirmed 2026-04-18/19):**
- **T5.6 / W5_CA_gemma4_t2n_postmortem — T2-N on Gemma4** — projected +34% (based on Qwen3 C=512 result). Reality: **0.62× regression at C=32.** Root cause: Qwen3 benched at C=512 (M_sorted=4,096); Gemma4 benched at C=32 (M_sorted=256) — **16× smaller batch, identical kernel overhead.** Fixed overhead of `torch.empty([40960, 44] int32)` = 7 MB alloc per call dominates at 256 rows. Additionally K=2,816 (Gemma4) vs K=2,048 (Qwen3) = 38% more work/row with no offsetting gain at small batch. **This is a standard Category 3 failure from regime mismatch, not an incompatible architecture.** Fix: add `FUSED_MIN_TOKENS` threshold + persistent scale buffer. Gemma4 at C≥256 projected +15–25% after fix. See `plans/gemma4_t2n_postmortem.md`.
- **§5a ngram-GPU spec on Qwen3** — audit P=0.80, projected 1.2-1.5× single-user. Reality: **0.46-0.85× regression at all concurrencies.** Math-bench prompts have low repeating-n-gram rate; spec overhead > savings.

**Counter-example that worked:** fused-norm Qwen3 v2 — audit identified single-line `.kernel` vs `.backend` typo, fix delivered +19% real. The difference: code-level bug identification is reliable; cross-model kernel transfer and literature-based acceptance-rate projections are NOT.

**Detection rule:** 
- Audit PROCEED verdicts based on **specific code bug identification** (wrong attribute, missing env var, typo): treat as high-confidence.
- Audit PROCEED verdicts based on **cross-model extrapolation** or **literature projections**: treat as HYPOTHESIS, not prediction. P should be halved. Always bench before banking.
- **CRITICAL: When cross-applying a kernel optimized at concurrency C_A to a new model benched at C_B, check C_A/C_B ratio. If ratio > 4×, the batch regime is incompatible — regime mismatch is the dominant effect.** State the target model's planned bench concurrency explicitly in the dispatch prompt, not the source model's.

**Categorize audit PROCEEDs by evidence type:**
1. **Bug-fix PROCEED** (P~0.8 realized): specific line + specific error
2. **Recalibration PROCEED** (P~0.6 realized): reasoning used wrong constant (e.g., barrier cost)
3. **Cross-apply PROCEED** (P~0.3 realized): "works on model X, should work on model Y"
   - Sub-case **3a — regime-matched cross-apply** (same C, similar K): P~0.5
   - Sub-case **3b — regime-mismatched cross-apply** (C differs >4×, or K differs >30%): P~0.15
4. **Literature PROCEED** (P~0.3 realized): paper claims N× speedup, should apply here

Only categories (1) and (2) deserve the projected confidence. Categories (3) and (4) should be framed as "worth testing" not "expected to land." Sub-case 3b requires the target bench to use the SAME concurrency as the source bench, or the projection is meaningless.

---

### P12 — Missing pre-launch verify_isolation before dual-GPU work

**Symptom:** dual-GPU bench launched (or P4 triple-env applied) without verifying that physical GPU isolation actually occurred. On this WSL2 setup, `--gpus 'device=N'` is silently ignored by the nvidia-container-runtime (confirmed W5_25_wsl2_isolation_FAIL, 2026-04-19). Both containers land on GPU 0 and compete for the same device, but all banners, `docker inspect` DeviceIDs, and container env vars appear correct. The only reliable check is in-container `nvidia-smi --query-gpu=uuid`.

**Why this is silent:** `docker inspect` shows `DeviceIDs: ["0"]` and `["1"]` correctly. `nvidia-smi` on the host shows both containers' PIDs. But inside each container, `nvidia-smi --query-gpu=uuid` returns the SAME UUID (GPU 0) for both. The runtime accepted the request but did not honor it.

**Detection rule:**
```bash
# After container_name is launched and /health is ready:
CONTAINER_UUID=$(docker exec container_name \
    nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')
HOST_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i N | tr -d ' ')
if [ "$CONTAINER_UUID" != "$HOST_UUID" ]; then
    echo "ISOLATION FAILURE: container sees $CONTAINER_UUID, expected $HOST_UUID (GPU N)"
fi
```

**Template:** `serve_disaggregated_lmcache_v2.sh:verify_isolation()` — call after each container's `wait_healthy()` before launching the next container.

**Never rely on:**
- `docker inspect` DeviceIDs as proof of isolation
- `NVIDIA_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` env vars in container as proof
- Host-side `nvidia-smi` process listing as proof (PIDs appear on all GPUs on WSL2)

**Fallback:** if isolation fails, use `ISOLATION=disabled` pattern — launch both containers on GPU 0 sequentially. Validates protocol correctness without dual-GPU speedup. See `serve_disaggregated_lmcache_v2.sh` and `plans/lmcache_disagg_hardened.md §fallback_mode`.

---

### P10 — Checkpoint-architecture mismatch

**Symptom:** model load fails with `Model architectures ['SDARForCausalLM'] are not supported` (LMCache smoke 2026-04-19).

**Detection:** before launching any new model, check it's in vLLM's supported arch list via `docker run --rm IMAGE python3 -c "from vllm.model_executor.models import ModelRegistry; print(list(ModelRegistry.get_supported_archs()))"`.

---

## 3. Pre-KILL checklist

Must verify ALL before declaring KILL on any experiment. If any fail, the KILL is invalid and should be rerun.

- [ ] **Clean GPU state:** `nvidia-smi` shows ≤1 GiB residual and 0 peer compute-apps on the target GPU before launch
- [ ] **Single-container isolation:** no other vLLM/test containers running during measurement
- [ ] **Plugin banners verified:** every expected plugin's `[TAG]` line appears in `docker logs CONTAINER` AND EngineCore subprocess
- [ ] **Per-layer fusion count asserted** (for plugin-based optimizations): sum over layers matches expected, not just registration success
- [ ] **Correctness pass:** max_abs ≤ tolerance vs PyTorch reference; cos ≥ 0.999 where applicable
- [ ] **Warm run, not cold:** measurement is 2nd+ run at each concurrency level
- [ ] **Env vars verified at runtime:** `docker exec NAME env | grep -E 'EXPECTED_VARS'` matches launcher intent
- [ ] **Calibration constants up-to-date:** any reasoning citing barrier cost / BW ceiling / SM instruction cost references §1 of this file
- [ ] **Shape regime named:** KILL verdict explicitly names the shape (M, H, K, seq_len, concurrency, context length) where it was measured; does NOT claim to generalize without math
- [ ] **Specific failure mode identified:** "doesn't help" is not sufficient. Must identify the line/mechanism (e.g., "dequant smem roundtrip adds 460 µs/layer per profiling") — not just "too slow".
- [ ] **Pattern sweep:** check against §2 patterns P1-P12. If any applies, re-audit before KILL.
- [ ] **Dual-GPU launcher:** if launching two containers on separate GPUs, verify_isolation() MUST be called after each /health check before proceeding (P12).

---

## 4. Fix-template index

Recurring fixes for recurring bugs:

| Pattern | Template | Location |
|---|---|---|
| Silent-None dispatch (P1) | `hasattr(obj, "attr")` + warning log | any `patches/wire_*.py` plugin |
| Inherited NAME env (P3) | `unset NAME` at launcher top | any `launch_*.sh` |
| WSL2 GPU isolation (P4) — INVALIDATED 2026-04-19 | P4 triple-env DOES NOT work on this WSL2 setup (W5_25_wsl2_isolation_FAIL). Use `verify_isolation()` from `serve_disaggregated_lmcache_v2.sh` to detect; use `ISOLATION=disabled` for serial fallback. |
| Pre-launch isolation check (P12) | Call `verify_isolation(GPU, container, expected_uuid)` after each container /health. Template: `serve_disaggregated_lmcache_v2.sh:verify_isolation()` | any dual-GPU launcher |
| Graph-capture view aliasing (P8) | `torch.cuda.is_current_stream_capturing()` gate + `.clone()` | any plugin hot path that calls `.item()`/`.max()` |
| LMCache SM120 rebuild | git clone v0.4.3 + torch `_check_cuda_version` monkey-patch + `pip install . --no-build-isolation --no-deps` + `pip install sortedcontainers nvtx aiofile aiofiles` | `launch_lmcache_smoke_sm120.sh` |

---

## 5. Known-good idioms (approved patterns)

Use these without re-deriving:

- **Per-layer plugin dispatch:** `patches/fused_norm_fp4_integration.py` (Gemma4) — uses `hasattr(qm, 'backend')`, lazy builds AFTER `process_weights_after_loading`, per-layer callable cached in dict keyed by layer id
- **CUTLASS 128×4 swizzled scale formula:** `kernels/csrc/fused_shuffle_quant.cu:72` — validated bit-exact against CUTLASS reference via `test_swizzle_unpacker.py`
- **Cooperative kernel launch with grid.sync:** `kernels/csrc/mega_graph_gemma4_30layer_v5a.cu` — stripe ownership per SM, 5 barriers/layer (working baseline)
- **SWA sparse decode kernel:** `kernels/triton/swa_decode.py` — banked 2.52× e2e @ C=8, correctness cos=0.999999 vs FP32 reference
- **FP8 KV stacking:** `patches/swa_gemma4_plugin.py` + `swa_sparse_plugin.py` — graph-capture safe via `is_current_stream_capturing` gate

---

## 6. Usage in dispatch prompts

Every experiment dispatch prompt (human or agent) should begin with:

> Before any KILL verdict, verify against `plans/KILL_PATTERNS.md` §3 (pre-KILL checklist). Reason about silent-bug classes P1-P10 before concluding. Use §1 calibration constants — do NOT cite older barrier/BW numbers.

---

*Last updated: 2026-04-19 — W6_lmcache_disagg_harden: added P12 (pre-launch verify_isolation). Updated P4 entry to INVALIDATED — triple-env does not isolate on this WSL2 setup (W5_25_wsl2_isolation_FAIL). Template: `serve_disaggregated_lmcache_v2.sh:verify_isolation()`. Pre-KILL checklist updated to P1-P12.*

*2026-04-18 — W5 Gemma4 T2-N postmortem: confirmed §P11 Category 3b (regime-mismatched cross-apply) with C=512→32 (16×) + K=2048→2816 (38%) simultaneously. Added sub-cases 3a/3b and C-ratio detection rule. See `plans/gemma4_t2n_postmortem.md` for full analysis.*
