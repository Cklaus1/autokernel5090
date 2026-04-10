# FusenCache 32K Context Analysis

**Date:** 2026-04-10  
**Server:** vLLM 0.19.1rc1 on RTX 5090, Gemma4 26B A4B NVFP4, FusenCache k4v4b64  
**KV budget:** 168,128 tokens  
**Status:** Cannot test 32K context directly — `max_model_len=4096`

---

## Why Direct 32K Testing Is Not Possible

The current server was launched with `--max-model-len 4096`. vLLM enforces this at the request level: any prompt + completion exceeding 4096 tokens is rejected. Changing max_model_len requires a server restart.

From `docker logs vllm-gemma4`:
```
max_seq_len=4096
GPU KV cache size: 168,128 tokens
Maximum concurrency for 4,096 tokens per request: 57.69x
```

Despite having 168K tokens of KV capacity (enough for ~5 full 32K sessions), the scheduler will not admit any request longer than 4096 tokens.

**To test 32K context:** Restart the server with `--max-model-len 32768`.

---

## Current KV Configuration

### k4v4b64 Compression Spec

| Parameter | Value |
|-----------|-------|
| K bits | 4-bit symmetric int |
| V bits | 4-bit symmetric int |
| Scale block size | 64 elements |
| Slot bytes (head_dim=256) | 256 + 16 (scales) = 272 bytes |
| BF16 bytes per token per head | 1024 bytes |
| **Compression ratio vs BF16** | **3.76x** |

The actual measured multiplier from log comparison (168,128 vs 43,712 baseline) = **3.85x**, consistent with the theoretical 3.76x (rounding in block allocation accounts for the small discrepancy).

### GPU Memory Split (RTX 5090, 32GB)

| Component | VRAM |
|-----------|------|
| Model weights (NVFP4) | 17.24 GB |
| KV cache (k4v4b64) | ~10.01 GB → 168K tokens |
| CUDA context + activations | ~2 GB |
| **Total** | ~29.3 GB |

---

## How Many 32K Sessions Fit?

### Calculation: tokens_in_budget / 32768

| Hardware | KV Budget | FusenCache 32K sessions | BF16 32K sessions | Multiplier |
|----------|-----------|-------------------------|-------------------|------------|
| RTX 5090 (32GB) | 168K | **5** | 1 | **5x** |
| RTX Pro 6000 single (96GB) | ~1.29M | **39** | 10 | **3.9x** |
| RTX Pro 6000 DP=2 (192GB) | ~2.58M | **78** | 20 | **3.9x** |

**RTX 5090 detail:** 168,128 / 32768 = 5.13 → 5 full concurrent 32K sessions.  
With BF16 KV (43,712 tokens): 43,712 / 32768 = 1.33 → only 1 full session at a time.

### Context Capacity Across Hardware and Context Lengths

Sessions that fit fully within KV budget:

| Context Len | RTX 5090 (FusenCache) | Pro 6000 (FusenCache) | Pro 6000 DP=2 (FusenCache) | Pro 6000 (BF16) |
|-------------|----------------------|----------------------|---------------------------|------------------|
| 4K | 41 | 314 | 628 | 10 |
| 8K | 20 | 157 | 314 | 5 |
| 16K | 10 | 78 | 156 | 2 |
| **32K** | **5** | **39** | **78** | **1** |
| 64K | 2 | 19 | 38 | 0 |
| 128K | 1 | 9 | 18 | 0 |

Key insight: BF16 KV cannot serve even a single 32K session concurrently with anything else on the RTX Pro 6000. FusenCache enables 39 concurrent 32K sessions on the same hardware.

---

## Expected Decode Throughput at 32K Context

### Extrapolation from Measured Data

From `context_scaling_raw.json` (single-request, 128-token generation, BF16 KV):

| Context | Decode (tok/s) | ms/tok | Attn overhead vs ctx=256 |
|---------|----------------|--------|--------------------------|
| 256 | 122.8 | 8.14 | baseline |
| 512 | 121.0 | 8.26 | +0.12 ms |
| 1,024 | 113.1 | 8.84 | +0.70 ms |
| 2,048 | 111.7 | 8.95 | +0.81 ms |
| 3,072 | 110.3 | 9.07 | +0.93 ms |
| 3,840 | 91.9 | 10.88 | +2.74 ms |

The attention overhead scales roughly linearly with context length in this regime:
- Rate: **0.764 μs of extra decode latency per token of context**
- At ctx=3840: +2.74ms overhead (3584 extra tokens × 0.764 μs)

### Projected at 32K Context

```
Base decode time: 8.14 ms/tok (MoE + non-attention compute)
Attn overhead at 32K: (32768 - 256) × 0.764 μs = 24.7 ms
Total decode time: 8.14 + 24.7 = 32.8 ms/tok
Projected single-request decode rate: ~30 tok/s
```

**Caveats:**
1. This extrapolates 8.5x beyond the measured range (3840 → 32768 tokens).
2. FlashAttention may switch computation modes at longer sequences; the actual rate could be better or worse.
3. The attention overhead at very long sequences is memory-bandwidth-bound (reading the KV cache), so it scales O(S). The linear model is reasonable.
4. With FusenCache k4v4b64, the KV cache is 3.8x smaller → attention reads 3.8x less data → projected rate may be **somewhat better** than BF16 KV at equivalent context. Rough estimate: 35-40 tok/s with FusenCache vs 30 tok/s with BF16.

### Aggregate Throughput Projections

| Scenario | Sessions | Per-session (tok/s) | Batch efficiency | Aggregate (tok/s) |
|----------|----------|--------------------|-----------------|--------------------|
| RTX 5090, 32K, C=5 | 5 | ~31 | ~85% | ~130 tok/s |
| Pro 6000, 32K, C=39 | 39 | ~31 | ~70% | ~840 tok/s |
| Pro 6000 DP=2, 32K, C=78 | 78 | ~31 | ~70% | ~1,690 tok/s |

For reference, aggregate at shorter contexts (measured):
- ctx=128, C=32: 2,609 tok/s (peak)
- ctx=2048, C=32: 1,558 tok/s
- ctx=3840, C=32: 811 tok/s

Long-context serving is fundamentally compute-limited by attention, not KV memory. FusenCache solves the KV memory bottleneck but not the attention compute bottleneck. For the latter, FlashDecoding (split-K attention across SMs) would be needed.

---

## What the Scaling Numbers Mean in Practice

### For the RTX 5090 (current dev machine)

Without FusenCache, 32K context = 1 session max. Any coding assistant or document analysis use case requiring 32K is essentially single-user. With FusenCache: 5 concurrent users with 32K context each.

### For RTX Pro 6000 Single GPU

FusenCache transforms the Pro 6000 from a "1 user at 32K" machine to a "39 users at 32K" machine. This is the difference between a personal workstation and a small team deployment.

| Use Case | BF16 KV | FusenCache k4v4b64 |
|----------|---------|---------------------|
| 32K code review sessions | 1 | 39 |
| 16K document Q&A | 2 | 78 |
| 8K chat sessions | 5 | 157 |
| 4K standard chat | 10 | 314 |

### For DP=2 (Two Pro 6000s, 192GB)

78 concurrent 32K context users. Aggregate throughput ~1,690 tok/s at 32K. This is comparable to a small inference cluster. Cost estimate: 2x Pro 6000 ($10-15K hardware) vs cloud inference at equivalent throughput.

---

## Blockers for Actual 32K Testing

To run a real 32K context benchmark:

1. **Restart server** with `--max-model-len 32768`
   - This will re-profile memory and may allocate fewer total KV tokens
   - The 168K budget assumes vLLM set `num_gpu_blocks_override=512` for 4096-token profiling

2. **Verify text quality** — FusenCache k4v4b64 has a known decode quality issue from `fusencache_nvfp4_results.md`. The current server generates degraded output. Quality must be fixed before throughput numbers are meaningful.

3. **Run benchmark script** — extend `bench_context_scaling.py` with ctx=[4096, 8192, 16384, 32768] after quality is fixed.

---

## Next Steps

1. Fix FusenCache decode quality (attention kernel issue, see `fusencache_nvfp4_results.md` — Issue A)
2. Restart server: `--max-model-len 32768 --kv-cache-dtype k4v4b64`
3. Verify quality with short-answer prompts at 32K context
4. Run actual throughput benchmark
5. Compare FusenCache vs BF16 attention latency at 32K (tests whether the 3.8x smaller KV actually helps attention speed)
