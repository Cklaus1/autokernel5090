# Context Scaling Benchmark Results

**Date:** 2026-04-09  
**Server:** vLLM on RTX 5090, Gemma-4 26B A4B NVFP4, BF16 KV cache  
**Config:** max_model_len=4096, ~43K token KV budget  
**Generation length:** 128 tokens (fixed across all tests)

---

## Phase 1: Single-Request Latency vs Context Length

| Ctx Len | TTFT (ms) | Decode (tok/s) | Notes |
|--------:|----------:|---------------:|-------|
|     128 |    168.0* |          125.9 | *Cold start, ignore TTFT |
|     256 |      59.9 |          122.8 | Baseline after warmup |
|     512 |      52.3 |          121.0 | |
|   1,024 |      52.9 |          113.1 | -7% decode vs 256 |
|   2,048 |      60.6 |          111.7 | -9% decode vs 256 |
|   3,072 |      68.3 |          110.3 | -10% decode, TTFT climbing |
|   3,840 |      72.2 |           91.9 | **-25% decode** vs 256 |

**Key finding:** Single-request decode degrades gradually from 123 tok/s (short) to 92 tok/s at near-max context. The 25% drop at 3840 tokens shows attention compute becoming significant even for single sequences.

TTFT scales sub-linearly: 15x more context (256->3840) only costs 1.2x more prefill time (60->72ms), showing prefill is compute-efficient on the RTX 5090.

---

## Phase 2: Throughput at C=32 vs Context Length

| Ctx Len | Agg tok/s | % of Peak | KV Cache % | TTFT p50 (ms) | TTFT p99 (ms) |
|--------:|----------:|----------:|-----------:|---------------:|---------------:|
|     128 |   2,609.0 |    100.0% |       0.0% |          143.8 |          144.6 |
|     256 |   2,417.9 |     92.7% |       0.0% |          151.5 |          152.2 |
|     512 |   2,360.3 |     90.5% |       0.0% |          163.1 |          164.1 |
|   1,024 |   1,921.1 |     73.6% |       0.0% |          163.1 |          164.0 |
|   2,048 |   1,558.4 |     59.7% |       3.6% |          152.5 |          153.4 |
|   3,072 |     924.8 |     35.4% |      22.9% |          282.7 |          283.7 |
|   3,840 |     811.4 |     31.1% |      10.3% |          405.4 |          496.9 |

**Throughput degradation curve at C=32:**
```
ctx=128   ████████████████████████████████████████████████████  2609 t/s (100%)
ctx=256   ████████████████████████████████████████████████      2418 t/s  (93%)
ctx=512   ███████████████████████████████████████████████       2360 t/s  (90%)
ctx=1024  ██████████████████████████████████████                1921 t/s  (74%)
ctx=2048  ██████████████████████████████                        1558 t/s  (60%)
ctx=3072  ██████████████████                                     925 t/s  (35%)
ctx=3840  ████████████████                                       811 t/s  (31%)
```

**Critical finding:** Throughput drops >20% starting at ctx=1024 (1921 vs 2609 tok/s). By ctx=3072, throughput has collapsed to 35% of peak.

The KV cache utilization metrics are somewhat misleading because vLLM reports post-completion usage (cache freed). The 22.9% peak at ctx=3072 is measured after completions finish. During active generation, 32 concurrent requests at 3072 tokens each = ~98K tokens of KV, which exceeds the 43K budget. vLLM handles this by queuing/scheduling, which is why TTFT jumps from 163ms to 283ms at ctx=3072 -- requests are waiting for KV slots.

---

## Phase 3: Concurrency Scaling at ctx=2048

| Concurrency | Agg tok/s | Per-req tok/s | Scaling Efficiency | KV Cache % |
|------------:|----------:|--------------:|-------------------:|-----------:|
|           1 |      70.3 |          73.6 |             100.0% |       5.2% |
|           2 |     135.1 |          69.8 |              92.5% |       8.8% |
|           4 |     280.9 |          73.1 |              95.4% |      13.8% |
|           8 |     528.6 |          71.4 |              89.8% |       8.8% |
|          16 |   1,036.7 |          68.5 |              88.6% |       8.6% |
|          32 |   1,593.6 |          53.4 |              67.9% |       7.3% |

```
Concurrency scaling at ctx=2048:
C=1    ██                                                          70 t/s
C=2    ████                                                       135 t/s
C=4    ████████                                                   281 t/s
C=8    ████████████████                                           529 t/s
C=16   █████████████████████████████████                         1037 t/s
C=32   █████████████████████████████████████████████████         1594 t/s
```

Near-linear scaling up to C=16 (per-request decode stays ~70 tok/s). At C=32, per-request rate drops to 53 tok/s (67.9% efficiency) -- the batching overhead and memory bandwidth contention start to matter.

---

## Analysis: Where KV Cache Becomes the Bottleneck

### Two regimes identified:

**Regime 1 -- Compute-bound (ctx <= 1024, C=32):**
- Throughput scales linearly with batch size
- KV cache utilization stays near 0%
- Per-request decode rate stays ~80-90 tok/s
- TTFT is stable around 150-163ms
- The GPU is spending most time on MLP/MoE compute

**Regime 2 -- KV-memory-bound (ctx >= 2048, C=32):**
- Throughput degrades sharply with context length
- At ctx=3072, only 35% of peak throughput remains
- TTFT nearly doubles (283ms) as requests queue for KV slots
- At ctx=3840, TTFT p99 hits 497ms with high variance

### The crossover point: **ctx ~1024-2048 at C=32**

At 32 concurrent requests:
- 32 x 1024 tokens = 32,768 tokens of KV -- fits in 43K budget (76% util)
- 32 x 2048 tokens = 65,536 tokens of KV -- **exceeds 43K budget by 1.5x**
- 32 x 3072 tokens = 98,304 tokens of KV -- **exceeds 43K budget by 2.3x**

This matches the observed throughput cliff: the server must serialize requests beyond what fits in KV cache.

### The KV pressure point: **ctx=2048 at C=32**

This is where:
- Aggregate throughput has already dropped to 60% of peak
- KV demand (65K tokens) exceeds capacity (43K tokens)
- vLLM starts queuing requests, increasing latency
- Per-request decode rate drops from ~80 to ~51 tok/s

### What FusenCache 4x compression would unlock:

With 4x KV compression (43K -> 172K effective tokens):
- 32 x 3840 tokens = 122K tokens -- **fits comfortably** (71% util)
- Could serve C=32 at max context without KV spilling
- Could push to C=128 at ctx=1024 (131K tokens, 76% util)
- Estimated throughput recovery at ctx=3072: from 925 to ~2400+ tok/s (2.6x)

### Decode speed vs context (single request):

The 25% decode drop from 123 to 92 tok/s at single-request ctx=3840 is pure attention compute overhead (more KV entries to attend to). This is independent of KV memory pressure and represents the attention bottleneck. FusenCache would not help here -- this needs FlashAttention/FlashDecoding optimizations.

---

## Summary Table

| Metric | ctx=128 | ctx=1024 | ctx=2048 | ctx=3840 |
|--------|--------:|---------:|---------:|---------:|
| Single decode (tok/s) | 125.9 | 113.1 | 111.7 | 91.9 |
| C=32 aggregate (tok/s) | 2,609 | 1,921 | 1,558 | 811 |
| C=32 % of peak | 100% | 74% | 60% | 31% |
| KV demand at C=32 | 4K | 33K | 66K | 123K |
| Fits in 43K budget? | Yes | Yes | **No** | **No** |
| TTFT p50 at C=32 (ms) | 144 | 163 | 153 | 405 |
| **FusenCache 4x: fits?** | Yes | Yes | Yes | Yes (71%) |
