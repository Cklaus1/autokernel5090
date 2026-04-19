# LMCache Disaggregated Serving Migration
## P2pNcclConnector → LMCacheConnectorV1

**Date:** 2026-04-18  
**Tag:** W5_6_lmcache_disagg_swap  
**Hardware:** 2× RTX PRO 6000 (SM120, Blackwell), WSL2  
**Trigger:** ASI-1 KILL — `P2pNcclConnector` calls `cudaIpcGetMemHandle`, which WSL2 does not implement (SIGKILL on exec). Replacement confirmed WSL2-compatible in `experiment_log_20260418.md §ASI-1` line 227.

---

## 1. Before — P2pNcclConnector Architecture

```
Client
  │
  ▼
Proxy :8200
  │  builds request_id = "___prefill_addr_127.0.0.1:14579___decode_addr_127.0.0.1:14580_<UUID>"
  │
  ├──[Step 1: max_tokens=1]──► GPU 0 :8100  (kv_producer, kv_rank=0, kv_parallel_size=2)
  │                               │  P2pNcclConnector: cudaIpcGetMemHandle → !! SIGKILL on WSL2
  │                               │  NCCL broadcast of KV tensors to GPU 1 via PCIe
  │                               ▼
  │                            ZMQ ROUTER :14579
  │                               │
  │                            ZMQ ROUTER :14580
  │                               ▼
  └──[Step 2: full decode]──►  GPU 1 :8101  (kv_consumer, kv_rank=1, kv_parallel_size=2)
                                  P2pNcclConnector: receives KV via NCCL recv, runs decode
```

**Key config fields (prefill):**
```json
{
  "kv_connector":     "P2pNcclConnector",
  "kv_role":          "kv_producer",
  "kv_rank":          0,
  "kv_parallel_size": 2,
  "kv_ip":            "127.0.0.1",
  "kv_port":          14579,
  "kv_buffer_size":   2000000000
}
```

**Why it died on WSL2:** `cudaIpcGetMemHandle` is a device-to-device IPC primitive that requires the CUDA IPC subsystem, which WSL2's CUDA driver does not expose. Every attempt to form the NCCL KV group terminates with SIGKILL.

---

## 2. After — LMCacheConnectorV1 Architecture

```
Client
  │
  ▼
Proxy :8200
  │  assigns request_id = plain UUID (hex, no address embedding)
  │
  ├──[Step 1: max_tokens=1]──► GPU 0 :8100  (kv_producer)
  │                               │  Forward pass runs normally (prefill)
  │                               │  LMCacheConnectorV1.save(req_id, kv_tensors)
  │                               │    → cudaMemcpyAsync GPU→pinned_host_RAM
  │                               │    → 20 GB local_cpu pool keyed by req_id
  │                               │  (no NCCL, no IPC, no ZMQ address required)
  │                               ▼
  │                          [host RAM pool — both containers share via --network=host]
  │                               ▼
  └──[Step 2: full decode]──►  GPU 1 :8101  (kv_consumer)
                                  LMCacheConnectorV1.load(req_id)
                                    → pinned_host_RAM→GPU cudaMemcpyAsync
                                    → KV tensors restored; prefill skipped
                                  Autoregressive decode only
```

**Key config fields (prefill):**
```json
{
  "kv_connector":   "LMCacheConnectorV1",
  "kv_role":        "kv_producer",
  "kv_buffer_size": 20000000000,
  "kv_ip":          "127.0.0.1",
  "kv_port":        14579
}
```

**Key config fields (decode):**
```json
{
  "kv_connector":   "LMCacheConnectorV1",
  "kv_role":        "kv_consumer",
  "kv_buffer_size": 20000000000,
  "kv_ip":          "127.0.0.1",
  "kv_port":        14579
}
```

**What changed vs P2pNccl:**
| Field | P2pNccl | LMCache |
|---|---|---|
| `kv_rank` | Required (0 / 1) | Removed |
| `kv_parallel_size` | Required (2) | Removed |
| `kv_port` | Two separate ports (14579 / 14580) | Single shared port (14579) |
| `kv_buffer_size` | 2 GB | 20 GB (matches lmcache_cpu.yaml) |
| Request-id format | `___prefill_addr_IP:PORT___decode_addr_IP:PORT_UUID` | Plain UUID hex |
| NCCL group | Yes (requires IPC) | No |
| `cudaIpcGetMemHandle` | Yes → WSL2 KILL | No → WSL2 safe |
| Transport | NCCL over PCIe | cudaMemcpyAsync + pinned host RAM |

---

## 3. LMCache SM120 Rebuild

The PyPI `lmcache==0.4.3` wheel has no SM120 cubin (`c_ops.so` built without `arch=sm_120`). Both containers rebuild from source at startup using the same recipe validated in `launch_lmcache_smoke_sm120.sh`:

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
# Suppress vLLM CUDA version mismatch check
sed -i 's/raise RuntimeError(CUDA_MISMATCH_MESSAGE/pass  #/' \
    /usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py
git clone -q https://github.com/LMCache/LMCache.git /tmp/LMCache
cd /tmp/LMCache && git checkout v0.4.3
pip install . --break-system-packages --no-build-isolation --force-reinstall --no-deps
pip install --break-system-packages sortedcontainers nvtx aiofile aiofiles
```

Build time: ~5 minutes per container (runs in parallel with model weight loading, so no wall-time penalty beyond the first ~5 min of the 10-12 min startup window).

---

## 4. KILL_PATTERNS Applied

| Pattern | Where applied |
|---|---|
| P3 — `unset NAME` | Line 44 of launcher: prevents inherited `NAME` from parent shell clobbering containers |
| P4 — GPU-leak triple | Each `docker run`: `--gpus 'device=N'` + `-e NVIDIA_VISIBLE_DEVICES=N` + `-e CUDA_VISIBLE_DEVICES=0` |
| P4 — UUID check | `check_gpu_uuids()` function called at startup; `check_gpu_isolation()` called before bench |

---

## 5. Bench Plan

### 5a. Smoke test — KV transfer confirmed

```bash
# After both instances healthy:
curl http://localhost:8200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-26B-A4B-it-NVFP4","messages":[{"role":"user","content":"Hello"}],"max_tokens":20}'
```

Expected: response in ~200-300 ms with valid text. If decode instance returns empty or errors, check:
1. `docker logs vllm-disagg-lmc-prefill | grep -i lmcache` — confirm producer wrote KV
2. `docker logs vllm-disagg-lmc-decode | grep -i lmcache` — confirm consumer loaded KV

### 5b. Prefill-heavy workload — KV cache hit rate

Send 20 concurrent requests with long prompts (4K-tok context, 32-tok generation). These should:
- All hit the prefill instance (GPU 0) → GPU 0 compute at ~100% for prefill duration
- KV written to host-RAM pool (~19 MB per request at 28 layers BF16 GQA)
- GPU 1 loads KV → autoregressive decode only → observe decode throughput unconstrained

```python
# Expected decode behavior: GPU 1 tok/s stable even while GPU 0 is saturated with prefill
# Check: nvidia-smi dmon -s u -d 1 (GPU utilization per GPU over time)
```

### 5c. P99 TTFT comparison

Run `./serve_disaggregated_lmcache.sh bench` and compare against:

| Baseline | Expected P99 TTFT (C=8 bimodal) | Source |
|---|---|---|
| Stock vLLM collocated (DP=1) | ~640 ms | disaggregated_serving.md §6 |
| T1-only LMCache baseline (kv_both, prefix cache) | ~160 ms | smoke bench 2026-04-18 |
| **This config (disaggregated LMCache)** | **~110-130 ms** | projected: prefill isolated, no decode stall |
| Kill threshold | < 1.5× vs DP=2 | ASI-1 kill criterion |

Kill criterion inherited from ASI-1: if P99 TTFT at C=64 is less than 1.5× better than `serve_gemma4_dp2.sh`, revert to DP=2.

### 5d. Extended bench (C=16, 32, 64)

```bash
# Modify PROXY_PORT in the bench function or use bench_serving.py directly:
python3 bench_serving.py --port 8200 --model gemma-4-26B-A4B-it-NVFP4 \
    --num-prompts 64 --request-rate 8 --max-tokens 64
```

Observe:
- GPU 0 steady-state prefill throughput (TTFT contribution)
- GPU 1 steady-state decode throughput (tok/s)
- Host-RAM pool eviction rate (if pool fills at 20 GB, oldest entries evicted — watch for decode-side cache miss)

---

## 6. Known Risks

### R1 — WSL2 cross-GPU pid bleed (P4)
Despite `--gpus 'device=N'` isolation, WSL2's CUDA driver can register a vLLM process on all GPU UUIDs simultaneously. The `check_gpu_isolation()` function will warn if detected. If bleed occurs, throughput variance of 3-4× is expected (same symptom as the T2-N false alarm 2026-04-18).

**Mitigation:** run prefill and decode containers sequentially (decode only after prefill is `/health`-ready) to reduce window of concurrent GPU initialization. Already implemented via `wait_healthy()`.

### R2 — 20 GB host-RAM pool exhaustion
At 28 transformer layers, BF16 KV per token ≈ 2 × 28 × 8 × 128 × 2 bytes = 115 KB/token. For a 28672-token max-len request: ~3.3 GB. With 20 GB pool, this supports ~6 simultaneous large requests in flight. At C=8 with 4K-tok prompts (~460 MB each), the pool holds ~43 concurrent requests — well within budget.

**Risk:** if requests are slow to complete (e.g., very long generation), the pool may not evict fast enough. LMCache's LRU eviction is synchronous — a full pool stalls the producer. Monitor pool utilization via `docker exec vllm-disagg-lmc-prefill env | grep LMCACHE`.

### R3 — LMCache rendezvous port collision
Both containers share `--network=host` and bind to `127.0.0.1:14579`. The LMCache server/coordinator must bind exactly once. If both containers try to bind the same port simultaneously, one will fail. LMCache's `LMCacheConnectorV1` documentation indicates the `kv_producer` binds the port as server; the `kv_consumer` connects as client. Verify by checking startup logs:

```bash
docker logs vllm-disagg-lmc-prefill | grep -i "14579"
docker logs vllm-disagg-lmc-decode  | grep -i "14579"
```

If both show "binding" (not one bind + one connect), set `kv_port` to different values and file a LMCache issue.

### R4 — CUDA_VISIBLE_DEVICES remapping in container
The decode container sets `CUDA_VISIBLE_DEVICES=0` (container's internal view of its single device=1). If vLLM or LMCache uses `torch.cuda.device_count()` and expects index 0, this is correct. If any plugin uses the host UUID directly (e.g., NCCL group formation), it may see the wrong device. With LMCacheConnectorV1 there is no NCCL group — this risk is theoretical.

### R5 — kv_producer must complete before kv_consumer reads
The proxy's Step 1 → Step 2 sequential call ensures this. If the prefill instance has a slow path (e.g., CUDA graph capture miss on first request), Step 2 may arrive at the decode instance before the KV is written. LMCache will return a cache miss → decode runs full prefill redundantly (correct but slow on first request). Warm-up the proxy with 1-2 requests before measuring.

---

## 7. Files

| File | Role |
|---|---|
| `/home/cklaus/projects/autokernel/serve_disaggregated_lmcache.sh` | New launcher (this migration) |
| `/home/cklaus/projects/autokernel/serve_disaggregated.sh` | Original P2pNccl launcher (preserved, not modified) |
| `/home/cklaus/projects/autokernel/lmcache_cpu.yaml` | LMCache config (local_cpu=True, 20 GB pool, chunk 256) |
| `/home/cklaus/projects/autokernel/launch_lmcache_smoke_sm120.sh` | Reference SM120 rebuild recipe |
| `/home/cklaus/projects/autokernel/plans/experiment_log_20260418.md §ASI-1` | WSL2 compatibility confirmation |
| `/home/cklaus/projects/autokernel/plans/KILL_PATTERNS.md §4` | P3/P4 fix templates applied |
