# ASI-1 Disaggregated Prefill/Decode — Deployment Status

**Date:** 2026-04-16
**Script:** `serve_disaggregated.sh`
**Spec:** `plans/disaggregated_serving.md`
**Plan entry:** `plans/rtx_pro6000_experiments.md` ASI-1 (lines 35-56)

---

## What 1P1D Optimizes For

**Primary metric: P99 TTFT under mixed load.**

In a collocated setup (single GPU or DP=2 where each GPU still runs both prefill and decode), a long prefill (4K-token context) stalls the active decode batch for 200-500 ms. Under concurrent traffic this compounds: P99 TTFT scales linearly with request concurrency because each prefill queues behind all preceding prefills and all in-flight decode steps.

Disaggregating across two dedicated GPUs breaks the interference entirely:
- GPU 0 runs prefill continuously, never blocked by decode steps.
- GPU 1 runs decode continuously, never interrupted by prefill compute.
- TTFT = prefill compute time only, no queuing behind decode.
- Decode throughput is unaffected by prefill traffic volume.

This topology is strictly better than DP=2 when:
1. The workload has **mixed prompt lengths** (bimodal short+long is the worst case for collocated serving), AND
2. A **latency SLA exists** (interactive multi-turn chat, multi-agent fusen_solver swarms where TTFT gates agent iteration rate).

DP=2 remains better for pure aggregate throughput benchmarks at C >= 64 with uniform short prompts, because it doubles effective KV capacity without the per-request KV transfer overhead.

---

## Expected Metrics (from disaggregated_serving.md §6)

### P99 TTFT

| Concurrency | Collocated | Disaggregated | Improvement |
|-------------|-----------|---------------|-------------|
| C=1 (idle)  | ~80 ms    | ~100 ms (+20 ms KV xfer) | -25% (worse at idle) |
| C=4         | ~320 ms   | ~100 ms       | 3.2x better |
| C=8         | ~640 ms   | ~120 ms       | **5.3x better** |

The idle-server regression is expected and acceptable: the +20 ms KV transfer overhead (FP8 KV: ~10 ms) only appears when the spec says "no queuing." At any real concurrent load the disaggregated path wins decisively.

### Decode Throughput Under Prefill Load

| Prefill traffic | Collocated | Disaggregated |
|-----------------|-----------|---------------|
| None            | 60 tok/s  | 60 tok/s      |
| 50% prefill mix | ~35 tok/s | ~58 tok/s     |
| Heavy prefill   | ~15 tok/s | ~55 tok/s     |

The 15 → 55 tok/s figure (3.7x) is the headline improvement cited in rtx_pro6000_experiments.md ASI-1.

### KV Transfer Overhead

- BF16 KV (26B GQA, 2048-token prompt): 536 MB @ 28 GB/s PCIe 4.0 = ~19 ms
- FP8 KV (halved): ~268 MB = **~10 ms**
- Script uses `--kv-cache-dtype fp8` to stay at the 10 ms figure.

---

## Kill Criterion

From rtx_pro6000_experiments.md ASI-1:

> If P99 TTFT under mixed load is **< 1.5x better than DP=2 at C=64**, the KV transfer overhead dominates — stick with DP=2.

Benchmark procedure (implemented in `./serve_disaggregated.sh bench`):
- Run `bench_serving.py` at C in {4, 8, 16, 64, 128}
- Use bimodal prompt distribution: 50% short (256-tok), 50% long (4K-tok)
- Compare P50/P99 TTFT and decode tok/s vs T1-G (DP=2) results

---

## Blockers Identified

### 1. KV Transfer Compatibility With FP8 KV Cache (verified OK)
The spec table in rtx_pro6000_experiments.md ASI-3 marks "Disaggregated 1P1D + FP8 KV (native)" as checkmark-compatible. This is the configuration the script uses. No blocker here.

### 2. FusenCache k4v4 Is Incompatible With 1P1D (do not stack)
The same ASI-3 compatibility table marks "Disaggregated 1P1D + FusenCache k4v4" as "warning: KV transfer must handle compressed format." The P2pNcclConnector transfers raw KV tensors; it has no knowledge of FusenCache's compressed slot format. **Do not enable FusenCache on either instance while running disaggregated serving.** This is a hard blocker for any experiment that tries to combine the two.

### 3. Prefix Cache Is Per-Instance (cache hit rate halved)
From disaggregated_serving.md §8: prefix caches are not shared between the prefill and decode instances. Requests with shared prefixes get cache hits on GPU 0's prefix cache, but GPU 1 never sees them. Under multi-agent fusen_solver traffic (which has high system-prompt overlap), this means effective prefix cache hit rate is cut roughly in half vs a collocated setup. The workaround (LMCacheConnector shared across instances) is not in scope for this deployment.

### 4. No Automatic Failover
From disaggregated_serving.md §8: if the prefill instance (GPU 0) crashes, the decode instance stops receiving new requests with no graceful degradation. The script has no watchdog process. For production use, add a supervisor (systemd, supervisord, or a Docker restart policy `--restart unless-stopped`) to each container. Currently the script uses bare `docker run -d` with no restart policy.

### 5. Both GPUs Load Full Model Weights
Gemma4 26B NVFP4 is ~16 GB in NVF4 weights (per disaggregated_serving.md §7). Both GPUs hold independent copies. On the PRO 6000 (96 GB GDDR7 each), this is not a VRAM blocker, but it is 32 GB total model weight across the two GPUs that DP=2 also pays. No asymmetric weight savings possible with 1P1D.

### 6. Port Mapping Mismatch in docker run (requires verification)
The spec's reference instances use `--port 8100` and `--port 8101` inside the container. The Docker `-p` flags map those to the host: `-p 8100:8100` and `-p 8101:8101`. The kv_port (14579) for NCCL rendezvous is used for inter-container communication and is bound to `127.0.0.1` (loopback). If the two containers are isolated in separate Docker networks rather than sharing the host network, NCCL P2P over localhost will not work. The script uses the default Docker bridge network; if containers cannot reach each other via `127.0.0.1:14579`, both must be launched with `--network=host`.

**Recommendation:** for the first run, add `--network=host` to both `docker run` invocations and adjust the host-side port flags accordingly. This is the lowest-friction path for same-machine NCCL P2P.

---

## Next Steps

1. Verify P2P access: `nvidia-smi topo -m` — confirm `PIX` or better between GPU 0 and GPU 1.
2. Run `nccl-tests p2p_bw --minbytes 512M --maxbytes 512M -g 2` — confirm >= 20 GB/s.
3. Consider adding `--network=host` if NCCL rendezvous fails on default bridge network (see blocker 6).
4. Start: `./serve_disaggregated.sh`
5. Smoke test: single request to port 8100, verify token stream arrives.
6. Benchmark: `./serve_disaggregated.sh bench` at C={4,8,16,64}.
7. Compare P99 TTFT vs T1-G (DP=2) results. Apply kill criterion at C=64.
8. If keeping: deploy as Config B default (`rtx_pro6000_experiments.md` ASI-3).
