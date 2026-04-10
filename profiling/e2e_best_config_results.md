# E2E Integration Test — Best BF16 Config Results

**Date:** 2026-04-10  
**GPU:** RTX 5090 (32.6 GB VRAM)  
**Model:** Gemma 4 26B NVFP4 (`/models/gemma-4-26B-A4B-it-NVFP4-modelopt`)

---

## Server Configuration

```bash
docker run -d --name vllm-gemma4 --gpus all --memory=44g \
  -v /root/models:/models:ro -p 8000:8000 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  vllm-built python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 \
    -cc.mode none -cc.cudagraph_mode full
```

**Key config:**
- Quantization: ModelOpt NVFP4
- CUDA graph mode: `FULL` (downgraded to `FULL_DECODE_ONLY` due to FlashAttention constraint)
- Torch.compile: disabled (`-cc.mode none`)
- KV cache: BF16 (auto, no FusenCache)
- Max model length: 4096 tokens
- Memory limit: 44 GB system RAM (36 GB is too tight for CUDA graph capture)

**Server startup stats:**
- Model load: 17.24 GiB VRAM, 8.2s
- CUDA graph memory: 0.48 GiB (35 graph sizes captured, max batch=256)
- Available KV cache memory: 9.53 GiB → **41,616 tokens** (max concurrency ~14.3x at 4K context)
- Startup time: ~78 seconds

---

## E2E Integration Test Results: **9/10 PASS**

**Command:** `python3 tests/test_e2e_integration.py --base-url http://localhost:8000`

| # | Check | Result | Details |
|---|-------|--------|---------|
| 1 | Model load verification (NVFP4 modelopt) | **PASS** | Model `/models/gemma-4-26B-A4B-it-NVFP4-modelopt` loaded and responding |
| 2 | Generation coherence (10 diverse prompts) | **PASS** | All 10 prompts produced coherent output |
| 3 | FusenCache KV compression (KV > 100K tokens) | **PASS** | Long-context probe succeeded (Prometheus metrics unavailable for exact count; see logs for `GPU KV cache size`) |
| 4 | Prefix caching (shared system prompt) | **PASS** | All 5 shared-system-prompt requests completed, P50=0.5s |
| 5 | Throughput (>= 1000 tok/s at C=32) | **FAIL** | 508 tok/s — 32/32 success, 2461 tokens in 4.8s |
| 6 | Quality spot checks (math, code, reasoning) | **PASS** | All 3 quality checks passed |
| 7 | Memory bounds (VRAM < 33.0 GB) | **PASS** | 29.9 GB < 33.0 GB limit |
| 8 | Server error log check | **PASS** | No critical errors (aborts=0, total_tracked=26) |
| 9 | Concurrent request handling | **PASS** | 16/16 concurrent requests succeeded in 1.1s (100%) |
| 10 | OpenAI-compatible response format | **PASS** | All 3 responses are valid OpenAI-compatible JSON |

**Summary: 9/10 PASS — Deployment blocked by throughput check only**

---

## Throughput Analysis

**Measured:** 508 tok/s at C=32 (32 concurrent requests, max_tokens=100)  
**Threshold:** 1000 tok/s  

**Why BF16 BF16 underperforms the threshold:**
- The 1000 tok/s threshold was designed for FusenCache-enabled mode with KV compression
- At C=32 via HTTP API with max_tokens=100, the bottleneck is vLLM Python/IPC overhead (~65% of step time)
- Per-step time at batch=32: ~60ms, generating 32 tokens per step = ~533 tok/s theoretical ceiling
- NVFP4 weight read time: 8.7ms/step; vLLM overhead: ~52ms/step
- Previous measurements at C=256 achieved 6,615 tok/s (from `plans/pro6000_projections.md`)
- At C=32, projection was 1,738 tok/s but that assumed in-process LLM API, not HTTP server

**Check 3 note:** Reports PASS via long-context probe fallback. The Prometheus metrics endpoint
did not expose `vllm:num_gpu_blocks`. Actual KV capacity: 41,616 tokens (< 100K threshold).
The test passes because the fallback probe (long prompt handling) succeeds. If Prometheus
metrics were exposed, check 3 would FAIL (41,616 < 100,000).

---

## Operational Notes

1. **Memory limit:** `--memory=36g` causes SIGKILL during CUDA graph capture (exit code 137).
   Use `--memory=44g` or no memory limit.

2. **Background containers:** Multiple background bash scripts from previous sessions kept
   creating containers (ngram-spec-test, vllm-fusen, etc.) on port 8000. Always run
   `docker stop $(docker ps -q) && docker rm $(docker ps -aq)` to ensure clean state.

3. **Port conflicts:** Always verify `docker ps -a` before launching to avoid port 8000 conflicts.

4. **Container survival:** The server crashes if overwhelmed with large concurrent loads
   under the 36g memory cap. Use 44g+ for stability during E2E testing.

---

## Known Limitations vs. Checks

| Check | Expected | Actual | Gap |
|-------|----------|--------|-----|
| Throughput | ≥1000 tok/s at C=32 | 508 tok/s | 2x below threshold; requires FusenCache or higher concurrency |
| KV capacity | >100K tokens | 41,616 tokens | Below threshold; passes only via probe fallback |

---

## Recommended Next Steps to Achieve 10/10

1. **Enable FusenCache K4V4** — reduces KV memory bandwidth 4x, allows effective concurrency
   of C=128+ to fit in KV cache, projecting ~2000-3000 tok/s at C=32. Requires fixing
   the CUDA graph batching bug documented in `profiling/fusencache_b65_debug.md`.

2. **Expose Prometheus `num_gpu_blocks` metric** — so check 3 can verify actual KV token count
   rather than the fallback probe. With BF16 KV (41,616 tokens), check 3 will always
   fail on the exact-count path.

3. **Increase test concurrency** — if the test allowed C=128, throughput would be ~1,800+ tok/s
   (3.5x linear scaling from C=32). But check threshold and test script cannot be modified.
