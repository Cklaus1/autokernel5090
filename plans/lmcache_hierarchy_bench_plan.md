# LMCache Hierarchy Bench Plan
## Tag: W3_P3_lmcache_hierarchy_staged

**Date:** 2026-04-18  
**Handoff:** Artifacts staged. Parent bench agent executes. CPU-only staging — no processes launched here.

---

## 0 · Verification Checklist (run before benching)

| Check | Command | Expected |
|---|---|---|
| T2 session config present | `cat /home/cklaus/projects/autokernel/configs/lmcache_t2_session.yaml` | File exists, `max_local_cpu_size: 40.0` |
| T3 config present | `cat /home/cklaus/projects/autokernel/configs/lmcache_t3_remote.yaml` | File exists, `port: 8200`, `eviction_policy: lfu` |
| T3 server listening | `ss -tlnp \| grep 8200` | `LISTEN` on `127.0.0.1:8200` |
| T3 HEARTBEAT in log | `grep -i "heartbeat\|ready\|listening" /tmp/lmcache_t3.log` | At least one match |
| T3 KV count grows | `grep -c "store\|put\|write" /tmp/lmcache_t3.log` (after 5 requests) | Count increases |
| No CUDA IPC errors | `docker logs vllm-lmcache-hierarchy-patched 2>&1 \| grep -i cudaIpc` | 0 matches |
| vLLM health (patched) | `curl -sf http://localhost:8005/health` | HTTP 200 |
| Session shim fronting | `curl -sf http://localhost:8005/v1/models` | Lists gemma model |

---

## 1 · Workload Definitions

### W1 — Shared Prefix (20 users, single-turn)
- **Users:** 20 concurrent
- **Prompt:** 500-token shared system prompt (FusenSolver prompt from `bench_lmcache_smoke.py`) + 50-token distinct user query
- **Generation:** 32 tokens max
- **Metric:** P50 / P95 / P99 TTFT (ms), measured via streaming first-chunk arrival
- **Script:** `bench_lmcache_smoke.py` (existing; reuse unchanged)

### W2 — Multi-turn Session (20 sessions × 3 turns)
- **Sessions:** 20 distinct `session_id` values (`user-00` … `user-19`)
- **Per session:** 3-turn dialog. Turn 1: 500-tok system prompt + 50-tok query. Turn 2: same system + turn-1 KV + 50-tok query. Turn 3: + turn-2 KV.
- **Generation:** 32 tokens per turn
- **Metric:** TTFT per turn (T1 vs T2 vs T3), broken down by turn number. Expected: Turn 1 ~50 ms; Turn 2+ ~5–8 ms with T2 session hit.
- **Session_id injection:** Pass `"extra_body": {"session_id": "user-{N}"}` in each request.

### W3 — RAG / T3 Cross-Instance (2 vLLM instances × 10 users)
- **Setup:** Launch two `MODE=patched` containers (ports 8005 and 8007) sharing the same T3 server at `tcp://127.0.0.1:8200`.
- **Phase 1:** Instance A, 10 users, W1 workload (warm T3 with the shared system prompt).
- **Phase 2:** Instance B, 10 users, W1 workload. Measure T3 hit rate: TTFT should drop to ~10 ms for shared-prompt tokens already in T3.
- **Metric:** Cross-instance TTFT reduction. Instance B Turn-1 TTFT vs Instance A Turn-1 TTFT (first run cold, second run T3-warm).
- **T3 key count:** `grep -c "store\|put" /tmp/lmcache_t3.log` before/after Phase 1 — count should grow ~(500/256) = 2 chunks per user request.

### W4 — Code Completion (10 users, 2K-tok file buffer)
- **Users:** 10 concurrent
- **Prompt:** 2000-token synthetic file buffer (stable across keystrokes) + 50-token prefix (the "current line" — distinct per request to simulate keystroke events) + 32-token gen
- **Session isolation:** Each user has a fixed `session_id` so the file buffer KV is cached in T2 across keystroke requests.
- **Metric:** Per-keystroke TTFT (ms). Without T2: ~200 ms re-prefill of 2K buffer. With T2: ~8 ms incremental append.

---

## 2 · Configuration Matrix

| Run | MODE | T1 | T2 | T3 | Port |
|---|---|---|---|---|---|
| `baseline` | `baseline` | off | off | off | 8005 |
| `t1_only` | `t1_only` | on (20 GB LRU) | off | off | 8005 |
| `t1_t2` | `patched` | on | on (shim, 40 GB) | off (set `remote_url: null` in T1 config for this run) | 8005 |
| `t1_t2_t3` | `patched` | on | on | on (60 GB LFU) | 8005 + 8007 |

For `t1_t2` run: temporarily set `remote_url: null` in `lmcache_cpu.yaml` (or set env `LMCACHE_REMOTE_URL=""`) to isolate T2 from T3 contribution.

---

## 3 · Launch Sequence

```bash
# Step 0: Verify lmcache 0.4.3 installable
docker run --rm vllm-fusencache-gemma4fix:latest bash -c \
  "pip install --break-system-packages --quiet lmcache==0.4.3 && python3 -c 'import lmcache; print(lmcache.__version__)'"

# Step 1: Start T3 remote tier (before any vLLM instance)
./launch_lmcache_remote.sh
# Verify: ss -tlnp | grep 8200

# Step 2a: Baseline run (no LMCache)
MODE=baseline PORT=8005 ./launch_gemma4_lmcache_hierarchy.sh
# Wait for health: until curl -sf http://localhost:8005/health; do sleep 5; done
python3 bench_lmcache_smoke.py --port 8005 --tag baseline_no_lmcache
docker rm -f vllm-lmcache-hierarchy-baseline

# Step 2b: T1-only run
MODE=t1_only PORT=8005 ./launch_gemma4_lmcache_hierarchy.sh
# Wait for health
python3 bench_lmcache_smoke.py --port 8005 --tag t1_only_20gb
docker rm -f vllm-lmcache-hierarchy-t1_only

# Step 2c: T1+T2 run (T3 remote_url=null — set in env or temp-patch lmcache_cpu.yaml)
# Edit lmcache_cpu.yaml: remote_url: null  (already null in the base file)
MODE=patched PORT=8005 ./launch_gemma4_lmcache_hierarchy.sh
# Wait for health (shim on 8005, vLLM on 8006)
# Run multi-turn bench (W2):
python3 bench_lmcache_smoke.py --port 8005 --tag t1_t2_multiturn --workload multiturn
docker rm -f vllm-lmcache-hierarchy-patched

# Step 2d: T1+T2+T3 cross-instance run (W3)
# Instance A:
MODE=patched PORT=8005 NAME=vllm-hierarchy-A ./launch_gemma4_lmcache_hierarchy.sh
# Instance B (different port, same T3):
MODE=patched PORT=8007 VLLM_INTERNAL_PORT=8008 NAME=vllm-hierarchy-B \
  ./launch_gemma4_lmcache_hierarchy.sh
# Wait for both health checks
# Phase 1 (warm T3 via instance A):
python3 bench_lmcache_smoke.py --port 8005 --tag t3_warm_phase1 --concurrency 10
# Phase 2 (measure cross-instance T3 hit via instance B):
python3 bench_lmcache_smoke.py --port 8007 --tag t3_crossinstance_phase2 --concurrency 10
# Step 2e: Code completion bench (W4) — against instance A (T1+T2+T3 still running)
python3 bench_lmcache_smoke.py --port 8005 --tag code_completion \
  --workload code_completion --users 10
docker rm -f vllm-hierarchy-A vllm-hierarchy-B

# Step 3: Stop T3
pkill -f 'lmcache.server' || true
```

---

## 4 · Metrics to Collect

| Metric | Workload | Target |
|---|---|---|
| P50 / P95 / P99 TTFT (ms) | All | Baseline ~700 ms cold; T1 ~50 ms; T1+T2 turn-2+ ~5–8 ms |
| Turn-1 / Turn-2 / Turn-3 TTFT | W2 multi-turn | Turn-2 ≥10× better than turn-1 for T1+T2 |
| Cross-instance TTFT (Phase 2 vs Phase 1) | W3 T3 cross | Phase 2 ≤20 ms (T3 warm hit) vs Phase 1 ~50 ms |
| Keystroke TTFT | W4 code completion | ≤10 ms with T2 session hit |
| T3 chunk count growth | W3 | `grep -c put /tmp/lmcache_t3.log` grows ~2 chunks/request |
| LMCache log banner fired | All patched runs | `grep "LMCache\|lmcache.*init\|LocalCPU" <docker logs>` |
| No cudaIpcGetMemHandle errors | All | Zero matches in docker logs |

---

## 5 · Expected Results Summary

| Config | Workload | TTFT Target | vs Baseline |
|---|---|---|---|
| baseline | shared prefix | ~700 ms P99 | 1× |
| t1_only | shared prefix | ~50 ms P99 | ~14× |
| t1+t2 | multi-turn turn 2 | ~5–8 ms | ~88–140× |
| t1+t2+t3 | RAG cross-instance | ~10 ms P99 | ~70× |
| t1+t2+t3 | code completion | ~8 ms / keystroke | ~88× |

Blended across workload mix (70% single-turn + 30% multi-turn): **~18× P99 TTFT reduction** vs cold baseline.

---

## 6 · Results TSV Row

Log the following row to `results.tsv` upon completion of the full bench:

```
W3_P3	W3_P3_lmcache_hierarchy_staged	lmcache_hierarchy	—	—	—	—	STAGED	—	T1+T2+T3 hierarchy configs + launchers staged; bench pending parent agent execution
```
