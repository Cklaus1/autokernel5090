# LMCache Hierarchy Feasibility: Per-Tenant Session + Remote Tier
## Tag: W2_5g_lmcache_hierarchy_feasibility

**Date:** 2026-04-18  
**Context:** Base LMCache (local_cpu=True, 20 GB, chunk_size=256) projects 7.6× prefill reduction
for 20-user shared-prefix workloads (P99 TTFT 700→50 ms, per `lmcache_wsl2_test.md §4`).
ASI-1 disaggregated serving is dead on WSL2 (`disaggregated_serving.md`). This doc evaluates
extending to a 3-tier hierarchy: GPU HBM (implicit) → Local CPU (current) → Per-Tenant Session
(new) → Remote Shared (new).

---

## 1 · Three-Tier Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Tier 0 — GPU HBM (implicit)                                               │
│   vLLM block manager native KV pages; ~10-20 GB VRAM after weights        │
│   No config change. Hot working set lives here.                           │
└─────────────────────────────┬─────────────────────────────────────────────┘
                              │ cudaMemcpyAsync (H2D / D2H, pinned buffer)
┌─────────────────────────────▼─────────────────────────────────────────────┐
│ Tier 1 — Local CPU (current, 20 GB)                                       │
│   LMCacheEngine: local_cpu=True, chunk_size=256                           │
│   Role: cross-request shared prefix cache (system prompts, RAG context)   │
│   Key: rolling hash(prefix_tokens)                                        │
│   Eviction: LRU — large volume, many distinct prefixes                    │
└─────────────────────────────┬─────────────────────────────────────────────┘
                              │ in-process namespace lookup (kv_namespace)
┌─────────────────────────────▼─────────────────────────────────────────────┐
│ Tier 2 — Per-Tenant Session (new, 40 GB)                                  │
│   LMCacheEngine: local_cpu=True, max_local_cpu_size=40.0                  │
│   Role: multi-turn dialog KV keyed by session_id                          │
│   Key: hash(session_id + turn_index + prefix_tokens)                      │
│   Eviction: TTL-based (1 hr idle expiry) + LRU fallback                   │
│   Mechanism: LMCache kv_namespace per session_id (supported in 0.4.x)    │
└─────────────────────────────┬─────────────────────────────────────────────┘
                              │ TCP / Unix socket (standalone LMCache process)
┌─────────────────────────────▼─────────────────────────────────────────────┐
│ Tier 3 — Remote Shared (new, 60 GB)                                       │
│   LMCache standalone server: remote_url=tcp://127.0.0.1:8200              │
│   Role: popular system prompts + RAG context shared across vLLM replicas  │
│   Key: canonical hash(system_prompt_tokens)                               │
│   Eviction: LFU with a 24-hr TTL floor for ultra-hot prompts             │
│   Backend: lmcache built-in remote_backend.py (Redis-compatible ZMQ/RPC) │
│   RDMA/nixl: nixl is bundled in lmcache 0.4.3 dep tree, but...           │
│     PRO 6000 exposes PCIe Gen5 x16, not InfiniBand — nixl RDMA path       │
│     requires UCX + IB or RoCE HCA; neither is present on a single         │
│     workstation. nixl can fall back to TCP via UCX_TLS=tcp, but that      │
│     offers no latency advantage over the built-in TCP remote backend.     │
│     Verdict: skip nixl/RDMA for single-node; use TCP remote backend only. │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2 · Eviction Policy Per Tier

| Tier | Policy | Rationale |
|---|---|---|
| T1 Local CPU (shared prefix) | **LRU**, no TTL | Many distinct system prompts; recency dominates |
| T2 Per-Tenant Session | **TTL=1 hr** (idle) + LRU fallback | Sessions have natural life; stale KV consumes RAM with no reuse |
| T3 Remote Shared | **LFU** + 24-hr TTL floor | Hot system prompts reused thousands of times; frequency beats recency |

LMCache 0.4.x exposes eviction via `eviction_policy: lru|lfu` in config and TTL via
`kv_ttl_seconds`. Both are supported in the storage backend layer.

---

## 3 · Sizing on a Single PRO 6000 Node (128 GB host RAM)

| Allocation | GB | Notes |
|---|---|---|
| OS + CUDA driver + misc | 8 | headroom |
| T1 Local CPU (current, shared prefix) | 20 | existing baseline |
| T2 Per-Tenant Session (new) | 40 | ~80 concurrent sessions × 500 MB avg session KV |
| T3 Remote Shared (new) | 60 | popular system prompts + RAG chunks, all serving nodes |
| **Total host RAM used** | **128** | fits exactly |

Session KV budget sanity: Qwen3-8B, 32 layers, GQA 8 heads, head_dim=128, 2-turn dialog
~2000 tokens, BF16 → 32 × 8 × 128 × 2000 × 2 × 2 ≈ 524 MB / session.
40 GB / 524 MB ≈ 76 concurrent sessions before LRU eviction — adequate for a 30-user chat
workload with multi-turn depth ≤ 5.

---

## 4 · Per-Tenant Session Cache: `kv_namespace` Support

LMCache 0.4.x includes `kv_namespace` as a first-class config key. Setting
`kv_namespace: "session:{session_id}"` per-request isolates lookup scope so turn-2 prefill
finds turn-1's KV in T2 without polluting T1 shared prefix space.

The vLLM adapter (`lmcache_integration/vllm_v1_adapter.py`) passes per-request metadata to
`LMCacheEngine.retrieve()`; injecting `session_id` from request headers/extra body fields
is ~10 LoC in the connector, or via a per-request `kv_namespace` override if the public API
exposes it (needs verification against 0.4.3 source). Worst case: instantiate one
`LMCacheEngine` per active session (bounded by T2 pool size).

---

## 5 · Projected TTFT Reduction

### Base assumption (from `lmcache_wsl2_test.md §4`)
- Baseline: P99 TTFT 700 ms at C=20 (shared prefix, no cache)
- Base LMCache (T1 only): P99 TTFT ~50 ms → **7.6× reduction**

### (a) Chat multi-turn (30% of traffic, 5-turn dialogs)

Each turn reuses all previous turns' KV + appends new user tokens (~50 tok).

| Turn | Without T2 session cache | With T2 session cache |
|---|---|---|
| Turn 1 | Shared-prefix hit via T1 → 50 ms TTFT | Same → 50 ms |
| Turn 2 | Re-prefill full history (turn 1 = ~600 tok) → ~120 ms TTFT | Append 50 tok only → **~5 ms TTFT** |
| Turn 3+ | Re-prefill cumulative context → 150–300 ms | Incremental append → **~5–8 ms TTFT** |

Effective compounding: base 7.6× (T1) × 10–24× additional (T2, turns 2+) = **76–182×
compound TTFT reduction** vs cold baseline for multi-turn turns ≥ 2.

For the 30% multi-turn fraction, weighted average TTFT improvement at C=20:
~70% single-turn (T1 only, 50 ms) + 30% multi-turn (mostly ~5–8 ms) → blended P99 ~38 ms
vs baseline 700 ms → **~18× effective reduction** across full workload.

### (b) RAG workloads

RAG typically injects 2–8k tokens of retrieved context per request. Without T3 remote tier,
each vLLM replica re-prefills the same retrieved chunks. With T3 remote shared cache:

- First replica to receive a query for chunk C pays full prefill cost (~140 ms for 2k tok)
- All subsequent replicas (same or different nodes) restore from T3 → ~10 ms H2D restore
- Hit rate depends on document popularity; for a 100-doc corpus with Zipf distribution,
  top 10% of chunks absorb ~70% of queries → ~0.70 hit rate on T3
- Effective TTFT for RAG: 0.70 × 10 ms + 0.30 × 140 ms = **49 ms** vs 140 ms baseline
  → **~2.9× reduction** on RAG queries (on top of T1 system-prompt hit)

### (c) Code completion

Code completion has short system prompts (~200 tok) and high per-user context locality
(user's file buffer). T2 session cache captures the file buffer KV; T1 captures any shared
tool/schema prefix.

- Without hierarchy: each keystroke-triggered completion re-prefills ~1000-tok file buffer
- With T2: append ~20 tok diff only → TTFT 200 ms → **~8 ms** → **25× per-keystroke reduction**
- T3 provides minimal lift (code context is user-specific, not globally shared)

---

## 6 · Config Sketch

### T1 (existing, unchanged)
```yaml
# /tmp/lmcache_cpu.yaml  (Tier 1 — shared prefix)
chunk_size: 256
local_cpu: True
max_local_cpu_size: 20.0
eviction_policy: lru
remote_url: null
enable_p2p: False
save_decode_cache: False
```

### T2 (per-tenant session — new LMCacheEngine instance)
```yaml
# /tmp/lmcache_session.yaml  (Tier 2 — per-tenant session)
chunk_size: 256
local_cpu: True
max_local_cpu_size: 40.0
eviction_policy: lru
kv_ttl_seconds: 3600           # 1-hr TTL for idle sessions
enable_p2p: False
save_decode_cache: True        # keep decode KV for multi-turn
```

### T3 (remote shared — standalone LMCache process)
```yaml
# /tmp/lmcache_remote.yaml  (Tier 3 — shared remote)
chunk_size: 256
local_cpu: True
max_local_cpu_size: 60.0
eviction_policy: lfu
kv_ttl_seconds: 86400          # 24-hr floor for hot prompts
enable_p2p: False
```
```bash
# Launch standalone Tier 3 server (before vLLM)
python -m lmcache.server --config /tmp/lmcache_remote.yaml --port 8200
```

T1 vLLM config adds `remote_url: "tcp://127.0.0.1:8200"` for write-through to T3.

---

## 7 · Integration Estimate

| Task | Effort |
|---|---|
| T2 session cache: mount second LMCache engine, wire session_id routing | 1 day |
| T3 standalone server: launch script + T1 write-through config | 0.5 day |
| TTL + LFU eviction config verification (against lmcache 0.4.3 API) | 0.5 day |
| Bench harness: multi-turn TTFT measurement, session hit rate metric | 1 day |
| Integration test: 3-tier warm path, cold path, TTL expiry | 1 day |
| **Total** | **~4 days** |

No new dependencies. nixl/RDMA: not applicable on single-node PRO 6000 (no IB/RoCE HCA).
TCP remote backend is sufficient and already bundled.

---

## 8 · Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `kv_namespace` per-request override not in 0.4.3 public API | Medium | Fallback: one LMCacheEngine per active session (bounded pool) |
| T2 `save_decode_cache=True` bloats session store with diverged KV | Low | Decode KV is session-local by definition; chunk_size=256 amortizes waste |
| T3 remote server becomes SPOF for all vLLM replicas | Medium | Single-node: restart is fast; multi-node: add a hot standby T3 |
| 128 GB RAM leaves 0 headroom | Low | Reduce T3 to 50 GB (lose 10 GB remote capacity); OS uses only 5–6 GB actively |
| nixl RDMA needed for multi-node T3 | Low-future | nixl bundled but needs UCX + IB HCA; out of scope for PRO 6000 single node |

---

## 9 · Verdict

**PROCEED**

- Per-tenant session tier (T2) is a high-value, low-risk addition for chat workloads.
  Multi-turn TTFT gain (10–24× on top of base 7.6×) is the dominant user-visible improvement
  on the fusen_solver / chat product surface.
- Remote shared tier (T3) is straightforward — lmcache standalone process mode is already
  implemented; config change + launch script only. Primary value is RAG deduplication (~2.9×
  per-query, compounding with T1) and future multi-replica sharing.
- Total integration ~4 days. No new hardware. No CUDA IPC. No NCCL.
- Recommended sequencing: land T2 session cache first (highest impact, 2 days), then T3
  remote tier (2 days), then tune eviction policies against real traffic profiles.
- nixl/RDMA: defer until multi-node PRO 6000 cluster exists with IB or RoCE fabric.
