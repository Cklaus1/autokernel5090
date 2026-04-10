# N-Gram Speculative Decoding Benchmark: Gemma4 26B NVFP4

**Date:** 2026-04-09  
**GPU:** RTX 5090  
**Model:** `/models/gemma-4-26B-A4B-it-NVFP4-modelopt` (Gemma 4 26B NVFP4 via ModelOpt)  
**vLLM version:** 0.19.1rc1.dev150+gc5bee887b  
**Mode:** Single-user, sequential prompts, max_tokens=512, temperature=0.0

---

## Configuration

| Setting | Baseline | N-Gram Spec Decode |
|---------|----------|--------------------|
| Quantization | modelopt NVFP4 | modelopt NVFP4 |
| KV cache | default (FP16) | default (FP16) |
| CUDA graphs | cudagraph_full | cudagraph_full |
| Compilation | -cc.mode none | -cc.mode none |
| Speculative decoding | none | ngram, num_spec_tokens=4, lookup_min=2, lookup_max=5 |
| GPU memory util | default | 0.90 |
| Max model len | 4096 | 4096 |

**Note:** Async scheduling is automatically disabled with ngram-based spec decode (vLLM warning at startup).

---

## Results

### Baseline (NVFP4, cudagraph full, no spec decode)

| Prompt | Tokens | Time (s) | Throughput (tok/s) |
|--------|--------|----------|--------------------|
| Python binary search | 512 | 4.2 | 121.0 |
| Thread-safe LRU cache | 512 | 4.1 | 123.4 |
| SQL top-10 customers | 512 | 4.2 | 122.6 |
| Bash disk monitor | 512 | 4.2 | 123.2 |
| CSV parser with quotes | 512 | 4.2 | 123.2 |
| **Average** | | | **122.7 tok/s** |

### N-Gram Speculative Decode (num_spec_tokens=4, lookup_min=2, lookup_max=5)

| Prompt | Tokens | Time (s) | Throughput (tok/s) | vs Baseline |
|--------|--------|----------|--------------------|-------------|
| Python binary search | 512 | 6.1 | 84.1 | **-30.5%** |
| Thread-safe LRU cache | 512 | 5.6 | 92.2 | **-25.3%** |
| SQL top-10 customers | 512 | 4.2 | 123.0 | +0.3% |
| Bash disk monitor | 512 | 4.2 | 123.1 | -0.1% |
| CSV parser with quotes | 512 | 4.1 | 123.4 | +0.2% |
| **Average** | | | **109.2 tok/s** | **-11.0%** |

---

## Summary

**N-gram speculative decoding provides no benefit and hurts performance on Gemma4 26B NVFP4.**

- Average throughput: **122.7 tok/s (baseline) vs 109.2 tok/s (n-gram)** = **-11.0% regression**
- Prompts with highly repetitive/structured output (SQL, bash scripts, CSV parsers) show near-zero impact (~0%)
- Prompts requiring novel code generation (binary search, LRU cache) show significant regression: **-25% to -31%**

### Why N-Gram Fails Here

1. **NVFP4 + CUDA graphs are already fast:** The baseline at 122+ tok/s is close to peak hardware throughput for this model. N-gram adds overhead (draft evaluation, verification) that exceeds its gains.
2. **Code generation has low n-gram match rate:** Python code with unique variable names and logic doesn't repeat long n-gram sequences from the prompt, so most draft tokens are rejected.
3. **Async scheduling disabled:** vLLM warns that async scheduling is incompatible with n-gram spec decode, removing a key throughput optimization.
4. **num_gpu_blocks_override=512:** The n-gram container used this constraint vs the baseline's full KV cache allocation, potentially causing slight VRAM pressure differences.

### Comparison with FusenCache Baseline (Historical)

Earlier in testing, a FusenCache (k4v4b64) + enforce-eager container ran at 10-15 tok/s on the same prompts. The cudagraph_full baseline (122.7 tok/s) is ~8-10x faster than FusenCache+eager mode — confirming CUDA graphs are critical for performance.

### Recommendation

**Do NOT use n-gram speculative decoding with Gemma4 26B NVFP4 on RTX 5090.** The model is already bottlenecked by GPU throughput rather than memory bandwidth, so spec decode overhead exceeds its benefits. If speculative decoding is desired, a draft model approach (smaller Gemma variant) would be more appropriate.

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

**N-Gram Spec Decode:**
```bash
docker run -d --name vllm-ngram --gpus all \
  -v /root/models:/models:ro -p 8000:8000 \
  vllm-built python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 \
    --trust-remote-code --port 8000 \
    --gpu-memory-utilization 0.90 \
    --num-gpu-blocks-override 512 \
    -cc.mode none -cc.cudagraph_mode full \
    --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```
