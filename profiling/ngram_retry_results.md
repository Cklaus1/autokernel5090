# N-gram Speculative Decoding Retry Results

**Date:** 2026-04-10  
**Model:** Gemma-4 26B A4B NVFP4 (modelopt)  
**Hardware:** RTX 5090  
**Server config:** BF16, no inductor (`-cc.mode none`), full CUDA graphs (`-cc.cudagraph_mode full`), `num-gpu-blocks-override 512`

## Motivation

Previous test showed `num_speculative_tokens=4` gave -11% on code. Hypothesis:
1. 4 spec tokens may have too much overhead per step
2. We tested on novel code (low repetition) — n-gram should work better on repetitive output

This retry tests `n=2` and `n=1` on a mix of repetitive and novel content.

---

## Results

### Baseline (no speculative decoding) — warm server

| Prompt | Type | Throughput |
|--------|------|-----------|
| JSON schema 20 fields | repetitive | 121.9 tok/s |
| 10 dataclass definitions | repetitive | 123.8 tok/s |
| HTML table 15 rows | repetitive | 124.6 tok/s |
| Python binary search function | novel | 123.8 tok/s |
| SQL query joining 3 tables | novel | 123.9 tok/s |

**Repetitive avg: 123.4 tok/s | Novel avg: 123.9 tok/s | Overall avg: 123.6 tok/s**

> Note: Cold-start first request (CUDA graph warmup) was ~54 tok/s on the very first call.
> Stable throughput is ~124 tok/s after warmup.

---

### Spec n=2 (num_speculative_tokens=2, prompt_lookup_min=1, prompt_lookup_max=3)

| Prompt | Type | Throughput | vs Baseline |
|--------|------|-----------|-------------|
| JSON schema 20 fields | repetitive | 56.8 tok/s | -53.4% |
| 10 dataclass definitions | repetitive | 68.8 tok/s | -44.5% |
| HTML table 15 rows | repetitive | 64.4 tok/s | -48.3% |
| Python binary search function | novel | 67.3 tok/s | -45.7% |
| SQL query joining 3 tables | novel | 60.3 tok/s | -51.3% |

**Repetitive avg: 63.3 tok/s | Novel avg: 63.8 tok/s | Overall avg: 63.5 tok/s**  
**Change vs baseline: -48.7%**

---

### Spec n=1 (num_speculative_tokens=1, prompt_lookup_min=1, prompt_lookup_max=3)

| Prompt | Type | Throughput | vs Baseline |
|--------|------|-----------|-------------|
| JSON schema 20 fields | repetitive | 55.4 tok/s | -54.6% |
| 10 dataclass definitions | repetitive | 66.3 tok/s | -46.5% |
| HTML table 15 rows | repetitive | 62.1 tok/s | -50.2% |
| Python binary search function | novel | 68.3 tok/s | -44.9% |
| SQL query joining 3 tables | novel | 62.3 tok/s | -49.7% |

**Repetitive avg: 61.2 tok/s | Novel avg: 65.3 tok/s | Overall avg: 62.9 tok/s**  
**Change vs baseline: -49.1%**

---

## Summary Table

| Config | Repetitive avg | Novel avg | Overall avg | vs Baseline |
|--------|---------------|-----------|-------------|-------------|
| Baseline (no spec) | 123.4 tok/s | 123.9 tok/s | 123.6 tok/s | — |
| n=2, lookup 1-3 | 63.3 tok/s | 63.8 tok/s | 63.5 tok/s | **-48.7%** |
| n=1, lookup 1-3 | 61.2 tok/s | 65.3 tok/s | 62.9 tok/s | **-49.1%** |

---

## Key Findings

### 1. N-gram speculation is consistently harmful — 49-54% regression across all configs

Even with `n=1` (minimum possible speculative tokens), every prompt slows down by ~50%. The regression is **not** related to content type — repetitive and novel prompts perform equally badly.

### 2. Content type doesn't matter — repetitive vs novel makes no difference

The original hypothesis was that repetitive content (JSON schemas, HTML tables, dataclasses) would benefit from n-gram speculation. This is **false**: repetitive content shows the same ~50% regression as novel code. The n-gram lookup itself isn't the bottleneck — the speculative decoding execution framework is.

### 3. Fewer spec tokens doesn't help — n=1 is as bad as n=2

Reducing from n=2 to n=1 had negligible effect (-49.1% vs -48.7% vs baseline). This means the overhead is in the spec decode execution loop itself, not in verification of wrong guesses.

### 4. All prompts converge to ~62-68 tok/s with spec decode

The baseline shows uniform ~124 tok/s across all prompts. Spec decode brings everything down to a tight band of 55-68 tok/s regardless of prompt. This suggests a fixed overhead per decoding step that dominates throughput.

---

## Root Cause Analysis

The RTX 5090 with this model (NVFP4, CUDA graphs, no inductor) runs at ~124 tok/s baseline without spec decode. N-gram speculative decoding requires:

1. **Candidate proposal**: Scan prompt for n-gram matches (fast, negligible)
2. **Parallel verification**: Run the full model on speculated tokens AND the next token simultaneously — this is the key overhead
3. **Accept/reject step**: Additional logic per batch

The vLLM implementation of n-gram spec decode appears to run speculative and verification passes in ways that don't leverage the CUDA graph optimization well. The CUDA graph mode (`-cc.cudagraph_mode full`) is optimized for standard decode shapes, and speculation introduces variable batch sizes that likely break CUDA graph reuse.

Also notable: the spec decode mode shows uniform slowdown regardless of n-gram lookup success rate (repetitive vs novel content performs identically), confirming the overhead is structural, not lookup-hit-rate dependent.

---

## Conclusion: N-gram Speculative Decoding is Not Viable for This Setup

**All three tests (n=4 from previous experiment, n=2, n=1) show 11-54% regression.**

The degradation is fundamental to the current vLLM spec decode implementation with CUDA graphs + NVFP4 quantization on RTX 5090. The spec decode framework adds overhead that outweighs any speculation benefit.

**Recommendation:** Do not use n-gram speculative decoding with this server configuration. The baseline (no spec decode) at 123.6 tok/s is optimal.

---

## Experiment Metadata

```
Server: vLLM 0.19.1rc1.dev150+gc5bee887b
Model: /models/gemma-4-26B-A4B-it-NVFP4-modelopt
Quantization: modelopt (NVFP4)
GPU: NVIDIA RTX 5090 (32607 MiB)
Context: 4096 tokens
KV blocks: 512 (override)
Spec config tested:
  - {"method":"ngram","num_speculative_tokens":2,"prompt_lookup_min":1,"prompt_lookup_max":3}
  - {"method":"ngram","num_speculative_tokens":1,"prompt_lookup_min":1,"prompt_lookup_max":3}
  - (previous session: num_speculative_tokens=4, code prompts, -11%)
Prompts: 5 prompts, 512 max_tokens, temperature=0.0
Measurement: single-user sequential requests
```
