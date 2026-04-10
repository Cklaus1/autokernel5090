# Disaggregated Prefill/Decode Serving Design
## PRO 6000 2-GPU Workstation Configuration

**Date:** 2026-04-09  
**Hardware target:** 2x PRO 6000 GPUs (GPU 0 = prefill, GPU 1 = decode)

---

## 1. What This Solves

On a single-GPU or collocated dual-GPU setup, prefill and decode compete for the same compute resources. A long prefill (e.g., 4k-token context) can stall the decode batch for 200-500 ms, spiking P99 TTFT and reducing decode throughput. Disaggregating across two dedicated GPUs eliminates this interference entirely.

**Current collocated problem:**
- Decode running at 60 tok/s → prefill arrives → decode stalls for prefill duration
- TTFT = queuing wait + prefill compute (both are unpredictable under load)
- P99 TTFT is dominated by worst-case decode stall, not just prefill cost

**Disaggregated outcome:**
- GPU 0 runs prefill continuously, never blocked by decode batches
- GPU 1 runs decode continuously, never interrupted by prefill
- TTFT = prefill compute only (no queuing behind decode)
- Decode throughput = uninterrupted, no memory bandwidth contention from prefill attention

---

## 2. vLLM Native Support: Confirmed

vLLM 0.18.1 has a first-class disaggregated prefill/decode API built into the KV transfer subsystem (`vllm/distributed/kv_transfer/`). No workarounds needed.

### Key concepts

| Concept | Role |
|---------|------|
| `kv_role = "kv_producer"` | Prefill instance — runs forward pass, produces KV cache, ships it |
| `kv_role = "kv_consumer"` | Decode instance — receives KV cache, resumes generation |
| `kv_role = "kv_both"` | Single-instance collocated (current mode) |
| `kv_rank` | 0 = prefill, 1 = decode |
| `kv_parallel_size` | Total instances in the group (2 for 1P1D) |
| `kv_connector` | Transport backend (see Section 3) |

The config is passed as a JSON blob to `--kv-transfer-config`. Both instances must share the same `engine_id` to form a transfer group.

### Supported transport backends (in order of preference for same-machine)

| Connector | Transport | Notes |
|-----------|-----------|-------|
| `P2pNcclConnector` | NCCL over PCIe/NVLink | Best for same-node, no extra deps |
| `NixlConnector` | UCX/RDMA | Lowest latency but needs UCX stack |
| `LMCacheConnectorV1` | Redis or shared memory | Cross-node friendly |

For the PRO 6000 workstation (two GPUs in the same box), `P2pNcclConnector` is the right choice — it uses NCCL directly over the PCIe interconnect with zero external dependencies and a 32 GB memory pool by default.

---

## 3. Concrete Launch Configuration

### Instance 0 — Prefill GPU (GPU 0)

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model <model-path> \
  --port 8100 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --kv-transfer-config '{
      "kv_connector":      "P2pNcclConnector",
      "kv_role":           "kv_producer",
      "kv_rank":           0,
      "kv_parallel_size":  2,
      "kv_ip":             "127.0.0.1",
      "kv_port":           14579,
      "kv_buffer_size":    2000000000
  }'
```

**Tuning rationale:**
- `max-model-len 32768` — prefill GPU handles long contexts; no wasted VRAM on decode side
- `gpu-memory-utilization 0.90` — maximize KV cache for prompt storage
- `kv_buffer_size 2e9` (2 GB) — enough to buffer KV for several large requests in-flight

### Instance 1 — Decode GPU (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model <model-path> \
  --port 8101 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --kv-transfer-config '{
      "kv_connector":      "P2pNcclConnector",
      "kv_role":           "kv_consumer",
      "kv_rank":           1,
      "kv_parallel_size":  2,
      "kv_ip":             "127.0.0.1",
      "kv_port":           14579,
      "kv_buffer_size":    2000000000
  }'
```

**Tuning rationale:**
- `max-model-len 8192` — decode GPU only needs to hold active generation context, not full prompt replay
- `max-num-seqs 128` — optimize for high batch decode throughput
- `gpu-memory-utilization 0.85` — leave headroom for incoming KV cache DMA buffers

### Important: Both instances must load the same model weights

Both processes load the full model. This is expected — disaggregation saves GPU *compute time*, not model memory. Both GPUs hold a copy of the weights.

---

## 4. Proxy / Request Router

vLLM does not ship a built-in proxy for 1P1D routing. Two approaches:

### Option A: Nginx upstream (simplest)

```nginx
upstream prefill { server 127.0.0.1:8100; }
upstream decode  { server 127.0.0.1:8101; }

server {
    location /v1/completions {
        # All requests go to prefill first via kv_transfer_params
        proxy_pass http://prefill;
    }
}
```

This only works if vLLM's internal KV transfer protocol handles the hand-off automatically after prefill completes. With `P2pNcclConnector`, the prefill instance sends the KV to the decode instance and the decode instance streams tokens back to the client. The client connects to the prefill instance once.

### Option B: Custom proxy (production)

A lightweight FastAPI proxy:
1. Receives client request
2. POSTs to GPU 0 (prefill) at port 8100 — this instance runs the forward pass, transfers KV to GPU 1, and returns the completed token stream
3. GPU 1 (decode) runs autoregressive generation, streams tokens back through GPU 0's connection

The `kv_transfer_params` field in the request body can carry routing metadata:

```json
{
  "model": "...",
  "messages": [...],
  "kv_transfer_params": {
    "do_remote_decode": true,
    "do_remote_prefill": false
  }
}
```

This is already defined in `vllm/entrypoints/openai/chat_completion/protocol.py` and `vllm/entrypoints/serve/disagg/protocol.py`.

---

## 5. KV Cache Transfer: What Actually Moves

After GPU 0 completes prefill for a request, it transfers the KV cache tensors for all layers to GPU 1 via NCCL P2P copy over PCIe. For a 7B-class model:

| Parameter | Estimate |
|-----------|----------|
| Layers | 32 |
| KV heads per layer | 8 (GQA) |
| Head dim | 128 |
| Tokens in prompt | 2048 |
| KV dtype | BF16 (2 bytes) |
| **Total KV size** | **32 × 8 × 128 × 2048 × 2 × 2 = ~536 MB** |

PCIe 4.0 x16 delivers ~28 GB/s device-to-device. Transfer time ≈ 536 MB / 28 GB/s ≈ **19 ms**. This is added to TTFT but is fixed overhead regardless of prompt length growth.

For FP8 KV cache (enabled via `--kv-cache-dtype fp8`), this halves to ~268 MB → ~10 ms.

---

## 6. Expected Performance Gains

### TTFT (Time to First Token)

| Scenario | Collocated | Disaggregated | Improvement |
|----------|-----------|---------------|-------------|
| No queuing (idle server) | ~80 ms | ~80 ms + 20 ms KV xfer | -25% (worse) |
| 4 concurrent requests | ~320 ms P99 | ~100 ms P99 | **3x better** |
| 8 concurrent requests | ~640 ms P99 | ~120 ms P99 | **5x better** |

*Collocated P99 TTFT scales linearly with concurrent requests because prefill requests queue behind each other and behind decode batches. Disaggregated TTFT is bounded by max(prefill_queue_depth × prefill_time).*

### Decode throughput

| Scenario | Collocated | Disaggregated |
|----------|-----------|---------------|
| Decode tok/s (no prefill load) | 60 tok/s | 60 tok/s |
| Decode tok/s (50% prefill traffic) | ~35 tok/s | ~58 tok/s |
| Decode tok/s (heavy prefill load) | ~15 tok/s | ~55 tok/s |

*Disaggregated decode is almost entirely unaffected by prefill load because the prefill GPU never borrows decode GPU compute.*

**Overall expected improvement:** 20-40% better mean latency under moderate load, 2-5x better P99 TTFT under heavy mixed workloads.

---

## 7. Memory Allocation Strategy

| | GPU 0 (Prefill) | GPU 1 (Decode) |
|--|-----------------|----------------|
| Model weights | ~14 GB (7B BF16) | ~14 GB (7B BF16) |
| KV cache (active) | Large — needs room for full prompts | Smaller — only holds generated seqs |
| KV transfer buffer | 2 GB (outbound) | 2 GB (inbound) |
| VRAM target | 90% utilization | 85% utilization |

For a 31B model (e.g., Gemma 4 31B NVFP4), weights are ~16 GB in NVF4. Adjust `gpu-memory-utilization` down if OOM occurs during KV buffer allocation.

---

## 8. Limitations and Constraints

1. **1P1D only currently supported.** vLLM's `KVTransferConfig` doc states: "Currently only 1P1D is supported." No 2P1D or 1P2D without custom work.

2. **Both GPUs load full model weights.** No weight sharding across prefill/decode. Each GPU needs enough VRAM for the model independently.

3. **PCIe bandwidth is the KV transfer bottleneck.** If PRO 6000 GPUs are connected via PCIe (not NVLink), 28 GB/s is the ceiling. At high QPS with long contexts this can become a queue.

4. **Decode GPU must also load model to process speculative tokens.** If using MTP speculative decoding (from exp93), both instances must run the same model including the speculative heads.

5. **Prefix caching is per-instance.** The prefill GPU has its own prefix cache; the decode GPU does not see it. Requests with shared prefixes still get cache hits on GPU 0, but the cache cannot be shared across instances without LMCache.

6. **No automatic failover.** If the prefill instance crashes, decode stops receiving new requests. Need a watchdog or supervisor process.

---

## 9. Implementation Checklist

- [ ] Verify PRO 6000 P2P access: `nvidia-smi topo -m` — confirm GPUs can peer-transfer
- [ ] Enable P2P: `nvidia-smi nvlink -s` or check PCIe P2P status
- [ ] Test NCCL P2P bandwidth: `nccl-tests p2p_bw --minbytes 512M --maxbytes 512M -g 2`
- [ ] Launch prefill instance (GPU 0, port 8100) with producer config above
- [ ] Launch decode instance (GPU 1, port 8101) with consumer config above
- [ ] Verify both instances are healthy: `curl http://localhost:8100/health && curl http://localhost:8101/health`
- [ ] Send test request to prefill instance, verify decode GPU generates tokens
- [ ] Measure baseline TTFT collocated vs. disaggregated at 1, 4, 8 concurrent requests
- [ ] Tune `kv_buffer_size` if KV transfer stalls appear in logs
- [ ] Optionally enable FP8 KV cache (`--kv-cache-dtype fp8`) to halve transfer overhead

---

## 10. Alternative: Two Fully Independent Instances

If KV transfer is too complex or adds latency, an alternative is two completely independent vLLM instances with a stateless proxy:

- GPU 0 (port 8100): handles all requests, no disaggregation
- GPU 1 (port 8101): handles all requests, no disaggregation
- Nginx/HAProxy round-robins between them

This doubles throughput but does NOT solve the prefill-decode interference problem on each GPU. It is a simpler operational model for workloads where P99 TTFT is not critical.

**Recommendation:** Use disaggregated P2pNcclConnector for latency-sensitive workloads. Use independent round-robin for maximum total throughput with simple operations.
