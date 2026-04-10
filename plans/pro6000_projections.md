# RTX PRO 6000 Performance Projections: TP=2 and DP=2

**Date:** 2026-04-09
**Hardware arriving:** 1x RTX PRO 6000 (96GB) + 1x RTX PRO 6000 Max-Q (96GB), Blackwell SM120
**Total VRAM:** 192 GB (96 GB × 2)
**Interconnect:** PCIe only — NVLink is NOT present
**Reference:** RTX 5090 (32GB) single-GPU measured data

> **Recommendation: Use DP=2, not TP=2.**
> PCIe AllReduce costs 50-100 µs per all-reduce × 60 decoder layers = 3-6 ms per token.
> DP=2 has zero communication overhead and scales perfectly for throughput workloads.
> See the [DP=2 section](#dp2-data-parallel-two-independent-servers) below.

---

## Hardware Comparison

| Spec              | RTX 5090 (current)  | RTX PRO 6000 Max-Q (x2, DP=2)      |
|-------------------|---------------------|-------------------------------------|
| VRAM              | 32 GB               | 192 GB total (96 GB × 2)            |
| Architecture      | Blackwell SM120     | Blackwell SM120 (same)              |
| HBM bandwidth     | 1792 GB/s           | 1792 GB/s per GPU (independent)     |
| SM count          | 170 per GPU         | ~170 per GPU (to verify)            |
| TDP               | 575 W               | ~200 W each (Max-Q variant)         |
| Parallelism       | 1                   | DP=2 (two independent servers)      |
| Interconnect      | N/A                 | PCIe only (no NVLink)               |
| Inter-GPU comm    | N/A                 | None for DP=2 (zero overhead)       |

**Max-Q TDP note:** Both GPUs are the Max-Q variant (~200 W vs full PRO 6000's 300 W).
For DP=2 this is symmetric — both GPUs operate at the same load, so neither throttles
the other. Expect ~10-15% lower throughput vs full PRO 6000, but no bottleneck imbalance.

For TP=2 (hypothetical), asymmetric full/Max-Q TDP would gate throughput on the Max-Q.
This is an additional argument against TP=2 on this specific hardware.

---

## Single-GPU Measured Data (RTX 5090, BF16 KV, 4K context)

| Concurrency | tok/s  | Notes                          |
|-------------|--------|--------------------------------|
| C=1         | 89     | Decode-bound, small batch      |
| C=4         | 201    |                                |
| C=16        | 305    |                                |
| C=32        | 1,738  | Batching kicks in              |
| C=64        | 3,193  |                                |
| C=128       | 4,982  |                                |
| C=192       | 4,915  |                                |
| **C=256**   | **6,615** | **Peak**                    |
| C=384       | 5,863  | Scheduler overhead             |
| C=512       | 6,173  | Plateau                        |

**KV cache:** 43,760 tokens at 4K context → ~10x concurrency headroom before KV pressure.

---

## Throughput Projections — TP=2

### Methodology

TP=2 scaling is **not 2x**. Each decode step requires an AllReduce across GPUs (NCCL),
adding ~2-5 ms of communication overhead per step. This overhead:
- Is fixed regardless of batch size
- Amortizes at high concurrency (C=128+)
- Is painful at low concurrency (C=1: ~15-20% overhead, C=256: ~5% overhead)

Measured TP=2 scaling in comparable settings (vLLM, Llama-class models):
- Typical range: **1.5x to 1.9x** vs single GPU
- For MoE models with high compute intensity: closer to **1.7-1.8x**
- For small-batch (C<32): closer to **1.3-1.5x** (NCCL dominates)

### Projected tok/s by Concurrency (4K context, BF16 KV)

| Concurrency | TP=1 ref | TP=2 low (1.5x) | TP=2 mid (1.7x) | TP=2 high (1.9x) |
|-------------|----------|-----------------|-----------------|-----------------|
| C=1         | 89       | 100             | 110             | 130             |
| C=16        | 305      | 400             | 520             | 580             |
| C=32        | 1,738    | 2,100           | 2,500           | 3,300           |
| C=64        | 3,193    | 4,200           | 5,400           | 6,100           |
| C=128       | 4,982    | 6,500           | 8,500           | 9,500           |
| C=256       | 6,615    | 8,200           | **10,000**      | **12,600**      |
| C=512       | 6,173    | 8,500           | **10,500**      | **11,700**      |
| C=1024      | ~6,000   | **9,000**       | **11,000**      | **13,000**      |

**Best estimate:** 10,000-12,000 tok/s peak at C=512-1024, 4K context.

### Why C=1024 May Beat C=256 (Unlike Single GPU)

On the RTX 5090, peak throughput was at C=256 — above that, KV pressure caused
scheduler inefficiency (only 43,760 tokens of KV, 10x concurrency at 4K).

On the PRO 6000 with 192GB, KV capacity is ~480,000 tokens (BF16). At 4K context,
this supports **~120x concurrency** before KV pressure. So C=512 and C=1024 should
be fine and may yield higher throughput as batch size grows.

---

## KV Cache Capacity Analysis

### VRAM Budget (per GPU with TP=2)

```
Total per GPU:          96.0 GB
Model weights (TP=2):  - 8.5 GB   (17 GB total, halved by TP sharding)
CUDA graphs:           - 1.0 GB
NCCL buffers:          - 0.5 GB
KV available:          ~86.0 GB per GPU
Total KV (2x GPU):     ~172 GB
```

### Capacity by KV Dtype

| KV Dtype       | Compression | Effective KV  | Tokens (4K ctx) | Concurrency (4K) |
|----------------|-------------|---------------|-----------------|-----------------|
| BF16 (default) | 1.0x        | 172 GB        | ~480,000        | ~120x           |
| FP8            | 2.0x        | 344 GB equiv  | ~960,000        | ~240x           |
| FusenCache FP8+int4 | 2.67x  | 459 GB equiv  | ~1,280,000      | ~320x           |

**With FusenCache:** 192 GB × 4x effective compression = **768 GB equivalent KV capacity**.
(The 4x figure assumes FP8+int4 for K/V at 2.67x plus FusenCache's selective attention
reducing actual attention computation further.)

### Capacity vs Context Length (BF16 KV)

| Context   | Tokens available | Max concurrency |
|-----------|-----------------|-----------------|
| 4K        | ~480,000        | ~120x           |
| 8K        | ~480,000        | ~60x            |
| 16K       | ~480,000        | ~30x            |
| 32K       | ~480,000        | ~15x            |
| 64K       | ~480,000        | ~7.5x           |
| 128K      | ~480,000        | ~3.75x          |

At 128K context, we're still limited to 3-4 concurrent requests — but that's 128K context
each, which is the full Gemma4 window. The single GPU couldn't do this at all.

---

## Context-Length Scaling Projections

### TP=2 tok/s at C=128 (fixed concurrency, varying context)

| Context | Projected tok/s | Reasoning |
|---------|-----------------|-----------|
| 4K      | ~8,500          | Attention cheap, MoE dominates |
| 8K      | ~7,000          | Attention grows, MoE same |
| 16K     | ~5,000          | Attention ~2x, KV bandwidth doubles |
| 32K     | ~3,500          | Attention ~4x the 4K cost |

**Memory bandwidth is the ceiling.** Each decode step loads:
- Model weights (fixed): 17 GB × some fraction per step
- KV cache (grows with context): C × ctx_len × 2 × head_dim × num_layers bytes

At 32K context, C=128: KV load ≈ 128 × 32768 × 256 × 64 × 2 bytes ≈ 34 GB per step.
At 1792 GB/s bandwidth: 19 ms just for KV reads — that's the bottleneck.

---

## FusenCache on TP=2

FusenCache (FP8 K + int4 V) reduces KV memory by 2.67x, but TP=2 adds a new consideration:
**each GPU holds the full KV cache** (KV is not sharded across GPUs in standard TP).

This means FusenCache's benefit is **independent of TP** — you get 2.67x more tokens
per GPU regardless of tensor parallelism degree.

Expected behavior:
- Same throughput per token as single-GPU FusenCache
- 2.67x more concurrent requests before KV pressure
- At 32K context: ~40x concurrency (vs ~15x with BF16 KV)

**If FusenCache's batch sizes beat the TP NCCL overhead:** this is where TP=2 + FusenCache
becomes particularly powerful — both GPUs process very large batches efficiently.

---

## Day-One Benchmarking Plan

### Phase 1: Baseline (first 30 minutes)

1. Launch TP=2 with `./serve_gemma4_tp2.sh 32768` (32K context, BF16 KV)
2. Wait for health check (~120-180s for TP=2 CUDA graph capture)
3. Run quick sweep: `python bench_tp2.py --quick`
   - This tests C=1,32,64,128,256,512 at 4K context
   - Takes ~15 minutes
4. Record peak tok/s, peak concurrency, P50/P95 latency

### Phase 2: Context Scaling (next 30 minutes)

5. Run context sweep: `python bench_tp2.py --context-sweep`
   - C=128 fixed, tests 4K/8K/16K/32K contexts
   - Takes ~20 minutes
6. Record how throughput drops with context length

### Phase 3: TP=1 Comparison (if second GPU can run standalone)

7. Launch TP=1 on port 8001: `./serve_gemma4.sh serving 4096 8001`
8. Run comparison: `python bench_tp2.py --tp1-port 8001 --tp2-port 8000 --quick`
9. Measure actual TP=2 scaling efficiency vs 2x ideal

### Phase 4: FusenCache (if installed in Docker image)

10. Launch with FusenCache: `./serve_gemma4_tp2.sh 32768 8002 fusen`
11. Run: `python bench_tp2.py --tp2-port 8002 --full`
12. Compare peak concurrency vs BF16 baseline

### Key Metrics to Record

- Peak tok/s (absolute)
- Peak concurrency where tok/s is highest
- P50/P95 latency at C=1 and peak
- KV cache token count (from vLLM startup log)
- Scaling efficiency: actual_tp2 / (2 × tp1_reference)
- NCCL overhead: observed C=1 latency vs single-GPU baseline

---

## What Could Change These Projections

### Upside risks (better than projected)

- **NVLink bandwidth**: If PRO 6000 uses NVLink instead of PCIe, AllReduce is ~5x faster,
  scaling efficiency could reach 1.85-1.95x at all batch sizes.
- **Blackwell fused kernels**: vLLM's Blackwell path may include additional fused kernels
  not available on 5090 (fuse_norm_quant, better grouped GEMM) — could give 10-20% bonus.
- **Higher TDP consistency**: If both GPUs sustain 300W, throughput will be higher
  than if the Max-Q throttles at 200W.
- **Larger FP4 GEMM efficiency**: With 2x the batch from TP=2, each expert GEMM
  handles more tokens → better arithmetic intensity → FP4 tensor cores more efficient.
  The 59% bandwidth gap on 5090 should narrow.

### Downside risks (worse than projected)

- **PCIe interconnect**: PCIe AllReduce is ~10-15% slower than NVLink, hurting C=1
  latency significantly and reducing peak scaling efficiency.
- **Max-Q throttling**: If the Max-Q runs at 200W and the full PRO 6000 at 300W,
  TP=2 throughput is gated by the slower GPU, potentially losing 15-20%.
- **CUDA graph capture time**: TP=2 CUDA graph capture takes ~2x longer than TP=1.
  With 32K context, expect 180-300s startup time.
- **vLLM TP=2 stability**: Tensor parallelism adds complexity; may hit edge cases
  with Gemma4's heterogeneous head dims (256 sliding / 512 global attention).

---

## Summary: Expected Day-One Numbers

| Metric                     | Conservative   | Expected        | Optimistic      |
|----------------------------|----------------|-----------------|-----------------|
| Peak tok/s (4K ctx, BF16)  | 9,000          | **11,000**      | 13,500          |
| Peak concurrency           | C=256          | **C=512**       | C=1024          |
| Scaling vs TP=1            | 1.5x           | **1.7x**        | 1.9x            |
| C=1 latency                | 12 ms/tok      | **10 ms/tok**   | 8 ms/tok        |
| KV tokens (BF16)           | 400K           | **480K**        | 520K            |
| KV tokens (FusenCache)     | 1.0M           | **1.28M**       | 1.5M            |
| 32K context peak C         | 10x            | **15x**         | 20x             |
| 32K context tok/s at C=15  | 2,500          | **3,500**       | 4,500           |

**Single biggest win over RTX 5090 (TP=2):** Not throughput (1.7x), but **context and concurrency**.
With 6x the VRAM, we can run 32K context at 15x concurrency that was impossible before,
and 128K context becomes feasible for the first time.

**However, DP=2 is recommended for this specific hardware. See section below.**

---

## DP=2: Data Parallel — Two Independent Servers

### Why DP=2 > TP=2 on PCIe

The hardware has PCIe interconnect only — no NVLink. This changes the TP=2 calculus entirely.

**TP=2 communication cost on PCIe:**

Each decode step in a Transformer requires an AllReduce after every attention and MLP layer.
Gemma4 has 62 decoder layers, each needing one AllReduce:

```
AllReduce latency (PCIe):  50-100 µs per all-reduce
Layers per step:           62 (attention + MLP combined)
Total overhead per token:  50-100 µs × 62 ≈ 3.1-6.2 ms
```

At single-token decode (C=1), the step time is ~11 ms. Adding 3-6 ms of PCIe overhead
is a **30-55% throughput penalty** that cannot be amortized away — it is proportional
to batch size, not fixed.

**DP=2 communication cost:**

Zero. Each GPU runs an independent vLLM server. Requests are distributed by fusen_solver
using round-robin. No NCCL, no AllReduce, no coordination.

### DP=2 Architecture

```
Client requests
      │
      ▼
fusen_solver (round-robin router)
  ├── GPU 0 → vllm-gpu0 (port 8000, CUDA_VISIBLE_DEVICES=0)
  │           Full model (17 GB), 96 GB VRAM, full KV cache
  └── GPU 1 → vllm-gpu1 (port 8001, CUDA_VISIBLE_DEVICES=1)
              Full model (17 GB), 96 GB VRAM, full KV cache
```

Each GPU runs the full model independently. Requests that arrive together are
served by whichever GPU is less loaded (or strict round-robin). The fusen_solver's
`MultiBackend` mode handles failover, session affinity, and per-backend stats.

### VRAM Budget per GPU (DP=2)

```
Total per GPU:             96.0 GB
Model weights (full):     -17.0 GB   (NOT halved — each GPU runs full model)
CUDA graphs:               -1.0 GB
KV available:             ~78.0 GB
```

**Note vs TP=2:** TP=2 halved the weight cost to 8.5 GB/GPU, freeing ~8.5 GB more for KV.
DP=2 loses that 8.5 GB/GPU but gains zero communication overhead. The tradeoff clearly
favors DP=2 on PCIe for throughput workloads.

### KV Cache Capacity (DP=2)

| KV Dtype            | Per GPU     | Total (2 GPUs) | Concurrency at 4K | Concurrency at 32K |
|---------------------|-------------|----------------|-------------------|--------------------|
| BF16 (default)      | ~300K tok   | ~600K tok      | ~75x per GPU      | ~9x per GPU        |
| FP8                 | ~600K tok   | ~1.2M tok      | ~150x per GPU     | ~19x per GPU       |
| FusenCache FP8+int4 | ~500K tok   | ~1M tok        | ~125x per GPU     | ~16x per GPU       |

FusenCache wins on capacity vs raw FP8 because its k4v4b64 format achieves 2.67×
compression relative to BF16, and the per-GPU model-weight overhead (17 GB) reduces
the BF16 baseline slightly compared to TP=2's sharded weights.

### Throughput Projections (DP=2 Aggregate, 4K context, BF16 KV)

DP=2 scales at exactly 2× single-GPU throughput because there is zero communication
overhead. The only deviation is Max-Q TDP limiting the per-GPU ceiling.

**Per-GPU (single server, 96 GB, 4K ctx):**

| Concurrency | RTX 5090 ref | PRO 6000 projection | Notes                                |
|-------------|-------------|---------------------|--------------------------------------|
| C=1         | 89 tok/s    | ~90-95 tok/s        | Same decode speed (same SM count)    |
| C=32        | 1,738 tok/s | ~1,700-1,900 tok/s  | Similar                              |
| C=128       | 4,982 tok/s | ~5,000-5,500 tok/s  | PRO 6000: no KV pressure yet         |
| C=256       | 6,615 tok/s | ~6,200-6,800 tok/s  | 5090 was KV-limited here; PRO 6000 is not |
| C=512       | 6,173 tok/s | ~6,500-7,000 tok/s  | PRO 6000: 300K tokens → no KV pressure |
| C=1024      | ~5,500 tok/s| ~6,000-6,500 tok/s  | PRO 6000 KV still comfortable at 4K ctx |

**DP=2 aggregate (both GPUs, zero overhead):**

| C per GPU | C total | DP=2 tok/s (conservative) | DP=2 tok/s (expected) | DP=2 tok/s (optimistic) |
|-----------|---------|---------------------------|------------------------|--------------------------|
| C=64      | C=128   | ~10,000                   | ~11,000                | ~12,000                 |
| C=128     | C=256   | ~10,400                   | ~11,500                | ~13,000                 |
| C=256     | C=512   | ~12,000                   | **~13,500**            | **~14,500**             |
| C=512     | C=1024  | **~12,500**               | **~14,000**            | **~15,000**             |

**Best estimate:** 12,000-14,000 tok/s aggregate at C=256/GPU (C=512 total).

**Max-Q impact:** Both GPUs throttle at ~200 W vs 300 W for full PRO 6000. Expect
~10-15% lower throughput than a full PRO 6000 pair, so reduce estimates accordingly:
~10,000-12,000 tok/s is the realistic range.

### DP=2 + Parallel Solver: Perfect Match

The fusen_solver is designed for exactly this topology. With 8 agents split 4/4:

```yaml
# fusen_solver_dp2_config.yaml
backends:
  gpu0: {url: http://localhost:8000/v1, model: gemma4-nvfp4}
  gpu1: {url: http://localhost:8001/v1, model: gemma4-nvfp4}
strategy:
  default_n: 8       # 4 agents per GPU
  routing: round_robin
  session_affinity: true   # long sessions stay on same GPU (prefix cache)
```

The solver provides:
- Round-robin distribution (equal load)
- Least-loaded fallback (if one GPU is slower due to Max-Q throttle)
- Session affinity (coding assistant sessions stay on one GPU for prefix cache hits)
- Health monitoring (removes a backend if /health fails 3×)
- Per-backend throughput reporting every 5 minutes

For coding workloads with long context, session affinity is important: consecutive
turns in a conversation re-use the same prompt prefix, and keeping them on the same
GPU allows vLLM's prefix cache to avoid re-computing the KV for repeated context.

### Context Length Scaling (DP=2 Aggregate, C=128 per GPU)

| Context | DP=2 projected tok/s | Bottleneck                              |
|---------|---------------------|------------------------------------------|
| 4K      | ~11,000             | MoE GEMM (compute-bound)                |
| 8K      | ~8,000              | Attention growing (KV bandwidth)         |
| 16K     | ~5,500              | KV bandwidth ~2× the 4K load            |
| 32K     | ~3,500              | KV bandwidth ~4× the 4K load            |

At 32K context, C=128 per GPU: KV load ≈ 128 × 32768 × head_dim × layers × 2 bytes
≈ 17 GB per step per GPU. At 1792 GB/s bandwidth: ~9 ms just for KV reads.

### When TP=2 Would Be Better

TP=2 is only preferable when:
1. A **single request** requires more KV than one GPU can hold (> ~78 GB of KV)
2. This happens at context lengths > **~65K tokens** (78 GB / bytes-per-token)
3. Or with many concurrent very long requests approaching the 78 GB KV limit

At 32K context, the PRO 6000 holds ~9 concurrent requests per GPU before KV
pressure. For most workloads (C=1 to C=256 at 32K), DP=2 is never KV-limited.

**Summary:** TP=2 on NVLink would be competitive; TP=2 on PCIe is not. Use DP=2.

### Day-One Benchmarking Plan (DP=2)

#### Phase 1: Both servers up, quick sweep (20 minutes)

```bash
./serve_gemma4_dp2.sh 32768           # BF16, 32K ctx, ports 8000/8001
python bench_dp2.py --quick            # C=[1,32,64,128,256,512] at 4K
```

#### Phase 2: Compare single GPU vs DP=2 combined (40 minutes)

```bash
python bench_dp2.py --compare-single --context-lengths 4096,32768
```

This runs GPU 0 alone, GPU 1 alone, then both together. Records actual scaling
efficiency and fills `PRO6000_SINGLE_REFERENCE` for efficiency calculations.

#### Phase 3: Context scaling (20 minutes)

```bash
python bench_dp2.py --context-sweep   # C=128/GPU, 4K→8K→16K→32K
```

#### Phase 4: FusenCache (if installed)

```bash
./serve_gemma4_dp2.sh 32768 fusen fusen     # both GPUs with FusenCache
python bench_dp2.py --full --output bench_dp2_fusen.json
```

#### Phase 5: fusen_solver routing validation

```bash
# Start solver (after both vLLM servers are up)
python -m fusen_solver --config fusen_solver_dp2_config.yaml --port 9000 &
python bench_dp2.py --solver-port 9000 --quick
```

### DP=2 Summary Table

| Metric                        | Conservative  | Expected       | Optimistic     |
|-------------------------------|---------------|----------------|----------------|
| Peak tok/s aggregate (4K BF16)| 10,000        | **12,000**     | 14,500         |
| Peak C per GPU                | C=256         | **C=512**      | C=1024         |
| DP=2 scaling efficiency       | 1.9x          | **~2.0x**      | 2.0x           |
| C=1 latency (per GPU)         | 11 ms/tok     | **10 ms/tok**  | 9 ms/tok       |
| KV tokens per GPU (BF16)      | 260K          | **300K**       | 340K           |
| KV tokens per GPU (FusenCache)| 450K          | **500K**       | 560K           |
| KV tokens total (FusenCache)  | 900K          | **1M**         | 1.1M           |
| 32K ctx peak C per GPU        | 7x            | **9x**         | 12x            |
| 32K ctx tok/s at C=9/GPU      | 2,800         | **3,500**      | 4,200          |

**Single biggest win over RTX 5090 with DP=2:** Same as TP=2 — context and concurrency.
96 GB per GPU lets each server handle 32K context at 9× concurrency independently,
and 128K+ context becomes feasible. The aggregate benefit is **2× that of TP=2**
because there is zero PCIe overhead tax on throughput.
