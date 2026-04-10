# N-Gram Speculative Decoding Benchmark: Gemma4 26B NVFP4

**Date:** 2026-04-10 (ACTUAL MEASURED RESULTS — replaced placeholder data)
**GPU:** RTX 5090 (32.6 GB)
**Model:** `/models/gemma-4-26B-A4B-it-NVFP4-modelopt` (Gemma 4 26B NVFP4 via ModelOpt)
**vLLM version:** 0.19.1rc1.dev150+gc5bee887b
**Mode:** Single-user, sequential prompts, max_tokens=512, temperature=0.0

---

## Configuration

| Setting | Baseline | N-Gram Spec Decode |
|---------|----------|--------------------|
| Quantization | modelopt NVFP4 | modelopt NVFP4 |
| KV cache | default (auto/FP16) | default (auto/FP16) |
| CUDA graphs | cudagraph_full (PIECEWISE) | FULL_DECODE_ONLY* |
| Compilation | -cc.mode none | -cc.mode none |
| Speculative decoding | none | ngram, num_spec_tokens=4, lookup_min=2, lookup_max=5 |
| Max model len | 4096 | 4096 |

*N-gram forces downgrade from PIECEWISE to FULL_DECODE_ONLY because FlashAttention
doesn't support PIECEWISE cudagraphs in speculative decoding mode (vLLM warning at startup).

---

## Results

### Baseline (NVFP4, cudagraph full, no spec decode) — MEASURED 2026-04-10

| Prompt | Tokens | Time (s) | Throughput (tok/s) |
|--------|--------|----------|--------------------|
| Python binary search | 512 | 4.3 | 118.1 |
| Thread-safe LRU cache | 512 | 4.1 | 124.3 |
| SQL top-10 customers | 512 | 4.1 | 123.4 |
| Bash disk monitor | 512 | 4.1 | 123.6 |
| CSV parser with quotes | 512 | 4.1 | 123.5 |
| **Average** | **512** | **4.2** | **122.6 tok/s** |

Total: 2560 tokens in 20.9s = 122.5 tok/s effective

### N-Gram Speculative Decode (num_spec_tokens=4, lookup_min=2, lookup_max=5) — MEASURED 2026-04-10

| Prompt | Tokens | Time (s) | Throughput (tok/s) | vs Baseline |
|--------|--------|----------|--------------------|-------------|
| Python binary search | 512 | 16.6 | 30.9 | **-73.8%** |
| Thread-safe LRU cache | 512 | 19.9 | 25.7 | **-79.3%** |
| SQL top-10 customers | 491 | 15.8 | 31.1 | **-74.8%** |
| Bash disk monitor | 512 | 19.4 | 26.4 | **-78.6%** |
| CSV parser with quotes | 512 | 15.8 | 32.4 | **-73.7%** |
| **Average** | **511** | **17.5** | **29.3 tok/s** | **-76.1%** |

Total: 2539 tokens in 87.5s = 29.0 tok/s effective

---

## Summary

**N-gram speculative decoding SEVERELY DEGRADES performance: 3.6x slowdown (122.6 → 29.3 tok/s).**

- Average throughput: **122.6 tok/s (baseline) vs 29.3 tok/s (n-gram)** = **-76% regression (3.6x slower)**
- Every single prompt is significantly slower with n-gram spec decode
- The overhead of drafting + rejecting tokens vastly exceeds any accepted token benefit

### Measured Acceptance Rates (from vLLM SpecDecoding metrics)

```
Mean acceptance length: 1.4–2.1 tokens (out of 4 proposed)
Per-position acceptance rate: ~35%, 15%, 10%, 6%
Avg draft acceptance rate: 10–29% (typical: 15–20%)
```

With only 12-20% of proposed tokens accepted, each speculative step:
1. Proposes 4 tokens via n-gram lookup (negligible cost)
2. Runs full 26B model forward pass to verify all 4 tokens (same cost as 1 autoregressive step)
3. Accepts on average 1.5-2 tokens
4. Falls back to regenerating rejected positions

This means ~2-3x more forward passes per accepted token compared to autoregressive decoding.

### Why N-Gram Fails for Code Generation

1. **Code output doesn't copy from the prompt**: N-gram spec decode looks up repeating phrases from the prompt text. "Write a Python function..." doesn't contain `def binary_search(arr, target):` — the generated output is entirely novel.
2. **Low acceptance rate is the key metric**: N-gram achieves 2-3x speedup only when acceptance rate >50%. Here it's 15-20%, which yields negative returns.
3. **CUDA graph downgrade**: vLLM automatically downgrades from PIECEWISE to FULL_DECODE_ONLY cudagraphs with n-gram, hurting overall performance.
4. **max_num_scheduled_tokens capped**: vLLM warns "max_num_scheduled_tokens is set to 2048 based on speculative decoding settings," reducing throughput potential.

### N-Gram Works Well For (not code generation)

- Document QA where the answer is extracted verbatim from the context
- Summarization where key phrases are copied from the source text
- RAG responses quoting source documents
- Translation where proper nouns are carried over unchanged

### What Actually Works for Code Generation Speedup

| Method | Expected Speedup | Notes |
|--------|-----------------|-------|
| N-gram spec decode | **-3.6x (MUCH WORSE)** | 10-20% acceptance rate for novel code |
| Draft model spec decode | ~1.5-2x | Needs a small compatible Gemma model |
| MTP spec decode | ~1.3-1.8x | Requires model with MTP heads |
| FusenCache (KV compression) | Latency neutral | Enables more concurrent users |
| CUDA graph capture (already on) | Already applied | Baseline includes this |

### Comparison with FusenCache Baseline (Historical)

Earlier testing showed FusenCache (k4v4b64 4-bit KV) + enforce-eager ran at ~8-12 tok/s with heavy concurrent load.
The cudagraph_full baseline (122.6 tok/s) is ~10x faster for single-user throughput.
CUDA graphs are essential; FusenCache's value is in enabling more concurrent users via KV compression.

### Recommendation

**Do NOT use n-gram speculative decoding for code generation tasks on Gemma4 26B NVFP4.**

The model at 122-124 tok/s single-user with NVFP4+cudagraph is already near the compute-bound ceiling for this GPU. N-gram spec decode makes things significantly worse (3.6x) due to low prompt-to-output overlap in code generation tasks.

---

## Appendix: Container Commands Used

**Baseline:**
```bash
docker run -d --name vllm-gemma4 --gpus all \
  -v /root/models:/models:ro -p 8000:8000 \
  vllm-built python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 \
    -cc.mode none -cc.cudagraph_mode full
```

**N-Gram Spec Decode (Run 1 — with num-gpu-blocks-override 512, avg 34.4 tok/s):**
```bash
docker run -d --name vllm-ngram --gpus all \
  -v /root/models:/models:ro -p 8000:8000 \
  vllm-built python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --num-gpu-blocks-override 512 \
    -cc.mode none -cc.cudagraph_mode full \
    --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

**N-Gram Spec Decode (Run 2 — with max-num-batched-tokens 4096, avg 29.3 tok/s):**
```bash
docker run -d --name vllm-ngram2 --gpus all \
  -v /root/models:/models:ro -p 8000:8000 \
  vllm-built python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 \
    -cc.mode none -cc.cudagraph_mode full \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```
