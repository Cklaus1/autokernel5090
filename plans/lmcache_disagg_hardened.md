# LMCache Disaggregated — W6 Hardening Plan
## serve_disaggregated_lmcache_v2.sh

**Date:** 2026-04-18  
**Tag:** W6_lmcache_disagg_harden  
**Trigger:** W5_25_wsl2_isolation_FAIL — P4 triple-env (NVIDIA_VISIBLE_DEVICES + CUDA_VISIBLE_DEVICES + --gpus device=N) confirmed to NOT isolate on this WSL2 setup. Both containers collapse onto GPU 0 regardless of DeviceIDs in docker inspect.  
**Predecessor:** W5_6 (`serve_disaggregated_lmcache.sh`)

---

## 1. Changes from W5_6

### 1a. verify_isolation() — pre- and post-launch UUID check

New function added. Called twice per launch:

1. **After prefill /health**: `docker exec vllm-disagg-lmc-prefill nvidia-smi --query-gpu=uuid --format=csv,noheader` — compares returned UUID against host `nvidia-smi -i 0` UUID. If mismatch, aborts before decode container is launched.

2. **After decode /health**: Same check against GPU 1 UUID. If mismatch (container sees GPU 0 UUID instead of GPU 1), aborts with remediation menu.

**Pid bleed sub-check:** also runs `nvidia-smi --query-compute-apps=pid,gpu_uuid` and flags any pid appearing on >1 UUID — the exact symptom from W5_25_wsl2_isolation_FAIL.

Loud error includes:
- Which UUID was expected vs. seen
- Reference to `W5_25_wsl2_isolation_FAIL` in results.tsv
- Plans/KILL_PATTERNS.md §P4 remediation options
- Exact command to re-run in fallback mode

### 1b. Serial startup — prefill first, verify, then decode

W5_6 launched both containers and then health-checked both. V2 changes the order:

```
docker run prefill → wait_healthy(prefill) → verify_isolation(GPU 0) → 
docker run decode → wait_healthy(decode) → verify_isolation(GPU 1)
```

This prevents the WSL2 race where concurrent cudaDeviceInit from two containers can both claim GPU 0 during initialization. The prefill container is fully loaded and running before the decode container's CUDA context is initialized.

Any failure at either verify step aborts the entire launch.

### 1c. Fallback mode — serial on GPU 0

Activated by:
- `ISOLATION=disabled ./serve_disaggregated_lmcache_v2.sh`
- Auto-triggered when GPU 1 is not present

**Behavior:**
- Prefill container: GPU 0, `--gpus device=0`, port 8100 (unchanged)
- Decode container: GPU 0, `--gpus device=0`, port 8101 (remapped from GPU 1)
- Proxy: unchanged — still does Step1(prefill:8100) → Step2(decode:8101)
- LMCache host-RAM pool: still active — `kv_producer` writes, `kv_consumer` reads same 20 GB pool

**What fallback measures:**
- KV transfer correctness: does the producer→consumer host-RAM round-trip work?
- TTFT structure: does skipping prefill in the consumer actually save time?
- LMCache protocol correctness under GPU-collocated conditions

**What fallback does NOT measure:**
- Dual-GPU parallelism speedup (prefill and decode compete for same GPU)
- Decode throughput isolation (GPU 1 never stalls while GPU 0 prefills)

**Tagged in bench output:** `MODE=fallback ISOLATION=disabled`

**Bench targets for fallback:**
| Metric | Expected | Rationale |
|---|---|---|
| KV transfer correctness | PASS | Protocol independent of GPU assignment |
| C=8 P99 TTFT | 250-400ms | Serialized on one GPU, no isolation benefit |
| vs W5_6 normal | ~2-3× slower | No GPU 1 decode isolation |

### 1d. LMCache-specific diagnostics

New `lmcache_diagnostics()` function. Called after each container's /health check. Scrapes container logs for:

- KV buffer pool address lines (`pinned`, `pool allocated`, `kv_buffer`, `local_cpu`)
- Producer/consumer stats (`cache_hit_rate`, `put_count`, `get_count`, `kv_transfer`)

**Background stats loop:** `periodic_lmcache_stats()` launched as background process for both containers. Every 30 seconds, scrapes last 31 seconds of logs for LMCache activity. Output format: `[lmcache_stats:prefill] <line>`.

---

## 2. Backward Compatibility

V2 is backward-compatible with W5_6's bench plan:

| Feature | W5_6 | V2 |
|---|---|---|
| `stop` subcommand | yes | yes |
| `bench` subcommand | yes | yes (same logic) |
| Container names | same | same |
| Port assignments | same | same |
| KV config (kv_ip, kv_port, kv_buffer_size) | same | same |
| LMCache SM120 rebuild | same | same |
| Proxy logic | same | same |
| `ISOLATION=disabled` mode | not supported | new |
| `bench_fallback` subcommand | not supported | new |
| `verify_isolation()` | not present | new |

The W5_6 bench plan (`lmcache_disagg_migration.md §5`) runs unchanged against V2 in normal mode.

---

## 3. Bench Plan Update

### Path A — "with isolation" (normal mode)

**Precondition:** `verify_isolation()` returns 0 for both containers.

This path is currently BLOCKED by WSL2 isolation failure (W5_25_wsl2_isolation_FAIL). Available when:
- WSL2 isolation fix lands (W6_wsl2_isolation_rootcause findings applied), OR
- Host boots native Linux

```bash
./serve_disaggregated_lmcache_v2.sh   # will pass verify_isolation if fixed
./serve_disaggregated_lmcache_v2.sh bench
```

**Targets (from lmcache_disagg_migration.md §5c):**
- C=8 P99 TTFT: ~110-130ms
- Decode tok/s: ~55 tok/s (GPU 1 never stalls)
- Kill criterion: P99 TTFT at C=64 must be ≥1.5× better than DP=2

**TSV tag:** `W6_lmcache_disagg_dual_gpu`

### Path B — "fallback serial" (ISOLATION=disabled)

**Available NOW.** Does not require isolation fix.

```bash
ISOLATION=disabled ./serve_disaggregated_lmcache_v2.sh
./serve_disaggregated_lmcache_v2.sh bench_fallback
```

**Targets:**
- KV transfer: PASS (required — validates the LMCache producer→consumer protocol)
- C=8 P99 TTFT: 250-400ms (single GPU, serialized)
- C=8 P99 TTFT vs W5_6 collocated (~640ms): should be ~1.6-2.5× better (LMCache overhead amortized)

**TSV tag:** `W6_lmcache_disagg_fallback_serial`

### Bench sequence

1. Run Path B first (available now):
   ```bash
   ISOLATION=disabled ./serve_disaggregated_lmcache_v2.sh
   ./serve_disaggregated_lmcache_v2.sh bench_fallback
   ```
   Record `W6_lmcache_disagg_fallback_serial` rows.

2. Once isolation fix lands, re-run Path A:
   ```bash
   ./serve_disaggregated_lmcache_v2.sh   # auto — verify_isolation must pass
   ./serve_disaggregated_lmcache_v2.sh bench
   ```
   Record `W6_lmcache_disagg_dual_gpu` rows.

3. Compare Path A vs Path B: the delta = the value of actual dual-GPU isolation.

---

## 4. KILL_PATTERNS.md additions

The following pattern should be added to KILL_PATTERNS.md if not already present:

### P12 — Pre-launch verify_isolation before dual-GPU work

**Symptom:** dual-GPU bench launched without verifying physical isolation. On WSL2, `--gpus device=N` is silently ignored by the runtime (W5_25_wsl2_isolation_FAIL). All results attributed to dual-GPU speedup are actually single-GPU results with interference from a second container.

**Detection rule:** any launcher that declares separate GPU assignments MUST call `verify_isolation()` after each container's /health check before proceeding to the next container or to benchmarking. Specifically:
1. After `wait_healthy()` for each container, run `docker exec CONTAINER nvidia-smi --query-gpu=uuid --format=csv,noheader` and compare against host UUID.
2. If UUIDs don't match, refuse to proceed. Loud error with fallback instructions.
3. After both containers are up, check pid bleed: `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F, '{...} count[p]>1 {...}'`.

**Template:** `serve_disaggregated_lmcache_v2.sh:verify_isolation()` — copy this function to any future dual-GPU launcher.

**Never rely on:** `docker inspect` DeviceIDs as proof of isolation. Only trust in-container `nvidia-smi --query-gpu=uuid`.

---

## 5. Forward-compat notes

If WSL2 isolation fix lands (alternative docker runtime config, kernel upgrade, etc.):

- V2 in auto mode will call `verify_isolation()`, get matching UUIDs, and proceed normally.
- No code changes needed in V2 — the fix is transparent to the launcher.
- `ISOLATION=disabled` path remains available as a regression-test path.

If W6_wsl2_isolation_rootcause agent finds that `--runtime=nvidia --gpus all + CUDA_VISIBLE_DEVICES=N (host index)` works:

Update `DECODE_GPU` assignment logic in V2 to use this pattern and re-run `verify_isolation()`. The function already handles the UUID comparison independently of how the GPU was selected.

---

## 6. Files

| File | Role |
|---|---|
| `serve_disaggregated_lmcache_v2.sh` | Hardened launcher (W6) |
| `serve_disaggregated_lmcache.sh` | W5_6 original (preserved, BC reference) |
| `plans/lmcache_disagg_migration.md` | W5_6 architecture + bench plan (unchanged) |
| `plans/lmcache_disagg_hardened.md` | This file — W6 changes + fallback bench plan |
| `plans/KILL_PATTERNS.md §P4` | INVALIDATED annotation (updated 2026-04-19) |
| `results.tsv row W5_25_wsl2_isolation_FAIL` | Confirmed WSL2 isolation failure evidence |
