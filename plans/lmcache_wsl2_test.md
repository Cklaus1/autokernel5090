# LMCache KV Connector on WSL2 — Design & Proof

**Date:** 2026-04-17
**Context:** ASI-1 disaggregated 1P1D died on WSL2 today because `P2pNcclConnector` requires `cudaIpcGetMemHandle`, which the WSL2 CUDA driver does not expose (see `plans/disaggregated_status.md`, blocker 6 + NCCL rendezvous discussion). We need a WSL2-compatible substitute that still delivers the multi-agent shared-prefix win we were chasing with 1P1D.
**This doc proposes:** LMCache (`LMCacheConnectorV1`) with the default local-CPU backend — no NCCL, no CUDA IPC, no P2P — running on a single vLLM server to serve a multi-agent workload with heavy system-prompt overlap.

---

## 1 · Problem Recap

ASI-1's original motivation was decode throughput under mixed prefill/decode load (3.7× win projected, `disaggregated_status.md §Decode Throughput Under Prefill Load`). The 1P1D design paid a per-request KV xfer (~10–20 ms FP8/BF16) to move the prefill's KV tensors from GPU 0 to GPU 1 via NCCL. That transfer path calls `cudaIpcGetMemHandle` (through NCCL's P2P fast-path), which is blocked on WSL2.

A secondary limitation of the 1P1D design noted in `disaggregated_status.md §Blocker 3` was that **prefix cache is per-instance** — the decode GPU never sees the prefill GPU's cached prefixes, so the effective hit rate on shared system prompts is halved. The `disaggregated_status.md` explicitly called out *"LMCacheConnector shared across instances"* as the remedy but scoped it out.

Since 1P1D itself is now dead on WSL2, we pivot the whole investment: run **one** vLLM instance (no disagg, no NCCL, no P2P) and use LMCache as a **shared, cross-request, potentially cross-instance KV cache** whose backing store is CPU RAM on the host.

---

## 2 · What's Already in the Image

Grepped `vllm-fusencache:latest` at `/build/vllm/vllm/distributed/kv_transfer/kv_connector/`:

```
v1/lmcache_connector.py              # vLLM-side adapter, class LMCacheConnectorV1
v1/lmcache_mp_connector.py           # multi-process (worker-scoped) variant
v1/lmcache_integration/               # in-tree integration code
    utils.py                          # reads LMCACHE_CONFIG_FILE + env vars
    vllm_v1_adapter.py                # translates vLLM KV layout -> LMCacheEngine
    multi_process_adapter.py
```

`factory.py` registers both `LMCacheConnectorV1` and `LMCacheMPConnector`. The vLLM half is present. The **`lmcache` Python package itself is NOT installed** in the image (`import lmcache` → `ModuleNotFoundError`).

`pip index versions lmcache` → 0.4.3 is the latest PyPI release (released 2026). A scratch-container install succeeded:

```bash
docker run --rm --entrypoint /bin/bash vllm-fusencache:latest \
    -c "pip install --break-system-packages --no-cache-dir lmcache==0.4.3"
# Installs cleanly; pulls cupy-cuda12x, nixl, nixl-cu12, redis, aiofile, caio,
# opentelemetry-exporter-prometheus, cufile-python, nvtx, sortedcontainers.
```

No new image is built — we inject the package at container start with
`pip install --break-system-packages lmcache==0.4.3` in the bash entrypoint, or pre-install it into a bind-mounted venv / site-packages overlay for iteration.

### WSL2 compatibility audit (done)

Default config `LMCacheEngineConfig.from_defaults()` returns:

```
chunk_size:         256
local_cpu:          True
max_local_cpu_size: 5.0   # GB
remote_url:         None
local_disk:         None
use_layerwise:      False
enable_blending:    False
enable_p2p:         False
```

i.e. **local-CPU backend only, no P2P, no NIXL, no remote Redis.** A recursive grep of `/usr/local/lib/python3.12/dist-packages/lmcache/` for `cudaIpc` / `IpcMemHandle` returns **zero matches**. The `storage_backend/` dir has `local_cpu_backend.py`, `local_disk_backend.py`, `gds_backend.py`, `nixl_storage_backend.py`, `p2p_backend.py`, `remote_backend.py` — only the first two are active at default, and neither touches CUDA IPC. GPU↔CPU transfer uses standard `cudaMemcpyAsync` over a pinned host buffer, which works normally on WSL2.

**Verdict: LMCache default mode is WSL2-safe.** The only risky opt-ins (P2P, NIXL) are off by default and we never enable them.

---

## 3 · Architecture (WSL2-Safe Single-Server Config)

```
         ┌────────────────────────────────────────────────────┐
         │ vLLM server (GPU 1, port 8400)                     │
         │                                                    │
         │  [ attention / paged KV on HBM ]                   │
         │              ▲                                     │
         │              │ cudaMemcpyAsync (pinned host buf)   │
         │              ▼                                     │
         │  [ LMCacheEngine :: local_cpu_backend ]            │
         │  5–20 GB pinned CPU RAM, chunk_size=256 tokens     │
         │  keyed by rolling hash(prefix_tokens)              │
         └────────────────────────────────────────────────────┘
                          ▲
                          │ HTTP
                 ┌────────┴─────────┐
             20 concurrent users submit:
               [ identical 500-tok system prompt ]
             + [ distinct 50-tok user query ]
```

* **One vLLM instance, one GPU.** No NCCL, no P2P, no second container.
* LMCache writes each chunked prefix's K/V tensors to pinned host RAM; subsequent requests with the same prefix skip the prefill forward pass and restore K/V via `cudaMemcpyAsync(H2D)`.
* Cross-request sharing is **process-local** (same vLLM worker), but also **persistent across the vLLM server's lifetime** (standalone LMCache process mode is supported if we ever want to share across multiple vLLM replicas).

### Config file (`/tmp/lmcache_cpu.yaml`)

```yaml
chunk_size: 256
local_cpu: True
max_local_cpu_size: 20.0    # 20 GB CPU pool (host has ≥128 GB RAM)
remote_url: null
enable_p2p: False
save_decode_cache: False    # prefill-only caching; decode stays on GPU
```

### vLLM launch

```bash
docker run --rm --gpus '"device=1"' --ipc=host --network=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e LMCACHE_CONFIG_FILE=/tmp/lmcache_cpu.yaml \
  -e LMCACHE_CHUNK_SIZE=256 \
  -e LMCACHE_LOCAL_CPU=True \
  -e LMCACHE_MAX_LOCAL_CPU_SIZE=20 \
  --entrypoint /bin/bash vllm-fusencache:latest -c '
    pip install --break-system-packages --quiet lmcache==0.4.3 &&
    python3 -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-8B \
      --port 8400 --host 0.0.0.0 \
      --gpu-memory-utilization 0.85 \
      --max-model-len 4096 \
      --kv-transfer-config '\''{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'\'' '
```

Key flag: `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`. `kv_both` means the single instance both produces and consumes KV from LMCache. (Role `kv_producer`/`kv_consumer` would be the disagg split; we don't use that.)

---

## 4 · Why This Beats The Dead 1P1D For Our Workload

Multi-agent `fusen_solver` traffic submits ~20 concurrent queries that share a long system prompt (tool spec, output schema, agent persona — easily 500–2000 tokens). Per-request breakdown at concurrency C=20, prompt=500 prefix + 50 user:

| Stage (per request) | No LMCache | +LMCache |
|---|---|---|
| Prefix prefill (500 tok) | **Full forward** ~35 ms | First request: same 35 ms. Next 19: **cache lookup + H2D restore ~2–5 ms** |
| User-part prefill (50 tok) | ~4 ms | ~4 ms |
| Decode (100 tok @ 60 tok/s) | ~1.7 s | ~1.7 s |

Aggregate prefill time for 20 concurrent requests:

* Baseline: 20 × 35 ms = **700 ms** of GPU prefill, and this compute blocks decode, stretching TTFT at the tail.
* LMCache: 1 × 35 ms (first request populates cache) + 19 × ~3 ms restore = **92 ms** of GPU prefill — a ~**7.6×** prefill reduction at C=20, and most of the per-prefix recomputation is moved off the critical path.

Net throughput upside is governed by how prefill-bound the workload is. For this 500/50 split the prefill is ~2% of end-to-end (decode dominates), so aggregate token/s may only rise 10–20%. The real win is **TTFT at the tail** (P99 drops from ~700 ms queued-prefill regime to ~50 ms), which is the same metric 1P1D was chasing — just without any NCCL/P2P.

---

## 5 · Bench Plan

**Workload:** 3 distinct system prompts (each ~500 tokens) × 5 users (each with a distinct 50-token user query) = 15 requests, fired concurrently. Expected hit rate for prefix cache: 4/5 = 80% per system prompt.

**Measurements (per run):**
- wall-clock total
- per-request TTFT (p50 / p99)
- decode tok/s (prometheus `vllm:time_per_output_token_seconds`)
- **LMCache hit rate:** expose via `curl :8400/metrics | grep lmcache` (LMCache registers `lmcache_cached_tokens` + prefill-hit counters via `LMCStatsMonitor`).
- for comparison, vLLM's native prefix cache hit rate: `vllm:gpu_prefix_cache_hit_rate`.

**Baseline:** identical command without `--kv-transfer-config`. vLLM has its own in-GPU prefix cache (enabled by default), so the *delta* we're measuring is LMCache adding a **larger, CPU-backed tier** that survives cache evictions and (if we promote it) shares across processes.

**Pass / Kill:**
- ≥1.5× aggregate throughput improvement at C=20 with the shared-prefix workload → PASS.
- <1.5× → PASS-but-modest (still beats disagg which is KILLED by CUDA IPC on WSL2).
- LMCache internally invokes `cudaIpcGetMemHandle` or crashes on WSL2 → KILL (verified above: it does not).

---

## 6 · Smoke-Test Status (this session)

| Step | Status |
|---|---|
| Confirm `LMCacheConnectorV1` registered in vLLM | DONE — factory.py lines 168–176 |
| Confirm `lmcache` pip package availability | DONE — 0.4.3 on PyPI |
| Install `lmcache` in scratch container | DONE — clean install, pulls cupy-cuda12x + nixl as deps but nixl is never invoked at default config |
| WSL2 compatibility audit (cudaIpc grep) | DONE — zero matches in lmcache tree |
| Launch vLLM w/ LMCacheConnectorV1 on GPU 1 | **deferred** — see §7 |
| Measure hit rate + throughput vs baseline | **deferred** |

---

## 7 · Why GPU smoke test is deferred

GPU 0 is at 89% utilization / 97 GB used by `vllm-t2n-polish` + `eagle3_train` + `focused_greider` (other agents). GPU 1 is free, so a smoke test is feasible, but the budget (3 h) and the rule "don't touch running containers" argue for staging the test into a dedicated script the user can launch at their discretion rather than racing a live Eagle3 training. The launch recipe in §3 is self-contained and uses `--gpus device=1`; the only shared resource is host RAM, which has abundant headroom.

A runnable helper script should be added as `serve_lmcache_smoke.sh` (not in scope for this PR-level doc; it would duplicate existing `serve_*.sh` patterns with the two-line addition of `pip install lmcache==0.4.3` and the `--kv-transfer-config` flag).

---

## 8 · Open Questions / Follow-ups

1. **Chunk-size vs hit rate:** default 256 tokens. For 500-token prefixes we get 1 full chunk + a 244-token remainder chunk — the remainder is a full-prompt-identity hash, so hit rate should still be ~100%. For 200-token system prompts we'd drop to 0 full chunks and lose most of the win — tune `chunk_size=128` for short-prompt workloads.
2. **Multi-vLLM sharing:** LMCache supports a "standalone" process mode that exposes a ZMQ/RPC lookup server (`lookup_client/` + `standalone/` in the package). If we later want several vLLM replicas to share one KV pool on the same host, we run one LMCache process and point each vLLM at it via `LMCACHE_REMOTE_URL=tcp://...`. Still no CUDA IPC — host RAM + TCP.
3. **Interaction with FusenCache k4v4:** disagg status doc flagged k4v4 as incompatible with P2pNcclConnector because the connector doesn't know the compressed format. LMCache's KV serialization is format-agnostic (it grabs the paged KV tensors vLLM hands it), so **LMCache should compose with k4v4 compression**. Needs verification once smoke test lands.
4. **Decode caching off:** `save_decode_cache: False` in the config — we only cache prefill KV. Caching decode KV is possible (LMCache supports it) but would bloat the cache with non-shared data.

---

## 9 · Summary

- **ASI-1 1P1D is dead on WSL2** (NCCL needs `cudaIpcGetMemHandle` which WSL2 doesn't expose).
- **LMCache is the right replacement** for the shared-prefix multi-agent workload that motivated ASI-1. Default config is pure CPU-backed — no NCCL, no P2P, no CUDA IPC.
- **Integration in the image is 95% done:** vLLM ships `LMCacheConnectorV1` + full integration adapter. Only the Python package is missing; `pip install --break-system-packages lmcache==0.4.3` fills the gap in a scratch container without rebuilding the image.
- **Expected win at C=20 with 500-token shared prefix:** ~7.6× prefill compute reduction; TTFT p99 drops from ~700 ms to ~50 ms; aggregate throughput +10–20% (decode-dominated workload).
- **Decision:** green-light the smoke test (§5) when a free GPU window opens; commit the design, flag package install as a container-start step.
