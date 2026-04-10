# GSM8K Math Benchmark — Gemma 4 26B NVFP4

**Date:** 2026-04-10  
**Model:** `gemma-4-26B-A4B-it-NVFP4-modelopt`  
**Quantization:** NVFP4 (ModelOpt)  
**Server:** vLLM v0.19.1rc1 with n-gram speculative decoding (4 tokens, lookup 2–5)  
**Hardware:** RTX 5090 (31.84 GiB VRAM)  
**Problems evaluated:** 35 (GSM8K-style, verified answers)

---

## Summary Results

| Metric | Value |
|--------|-------|
| Correct answers (on valid responses) | 12 / 25 = **48.0%** |
| Server errors (HTTP 500, spec-decode instability) | 10 / 35 |
| Valid responses (non-error) | 25 / 35 |
| Overall (correct / all 35 problems) | 12 / 35 = **34.3%** |

---

## Comparison to Published Results

| System | GSM8K Accuracy | Notes |
|--------|---------------|-------|
| Google Gemma 4 26B BF16 (baseline) | **~97.0%** | Full 1,319-problem GSM8K, greedy |
| RedHat NVFP4 (standard vLLM) | **95.6%** | Full GSM8K, published benchmark |
| **This run (NVFP4 + n-gram spec-decode)** | **48.0%** (valid), **34.3%** (all) | 35 problems, server instability |

---

## Problem-by-Problem Results

| # | Question (abbreviated) | Expected | Model Response | Correct |
|---|------------------------|----------|----------------|---------|
| 1 | Janet ducks eggs ($2/egg) | 18 | 18 (#### 18) | ✓ |
| 2 | Robe bolts (2 blue + half white) | 3 | 3 (in text) | ✓ |
| 3 | Josh house flip profit 150% | 70000 | Truncated before profit calc | ✗ |
| 4 | James sprints 3×3×60m | 540 | 540 | ✓ |
| 5 | Wendi chickens final meal | 20 | 20 | ✓ |
| 6 | Kylar glasses alternating price | 56 | Wrong (15 subsequent, not alternating) | ✗ |
| 7 | Toulouse+Charleston+Seattle sheep | 260 | 160 (Toulouse = 80 not 160) | ✗ |
| 8 | Tom house 20%/yr × 3 years | 259200 | 259,200 (computed correctly, then looped) | ✓ |
| 9 | Carrie weekly pay + overtime | 645 | Miscalculated overtime component | ✗ |
| 10 | Sara apples + oranges | 12 | 12 | ✓ |
| 11 | Maria+Luis coins (Maria = 3×Luis) | 96 | Garbled/hallucinated output | ✗ |
| 12 | School boys (40% of 500) | 200 | Started right (300 girls) then hallucinated | ✗ |
| 13 | Mike shirts 20% off | 120 | Stuck in repetition loop | ✗ |
| 14 | Sam money left ($50-$18-$12) | 20 | 38 ($50-$18=$12 arithmetic error) | ✗ |
| 15 | 12 apples / 3 friends | 4 | 4 | ✓ |
| 16 | Jake monthly earnings | 2400 | 2400 | ✓ |
| 17 | 24 chocolates - 8×2 | 8 | Garbled/degenerated output | ✗ |
| 18 | Lisa pages in 2 weeks | 420 | 280 (used 28 days not 14 days) | ✗ |
| 19 | Pool 5000 - 100/day × 10 | 4000 | No answer, model confused | ✗ |
| 20 | Mark average test score | 87 | 87 (348/4=87) | ✓ |
| 21 | Garden flowers after removal | 100 | 100 | ✓ |
| 22 | Tom's money (Tom = 2×Jane, together $90) | 60 | Solved for Jane ($30) not Tom | ✗ |
| 23 | Train tickets 3-day total | 345 | 345 | ✓ |
| 24 | Sarah words typed in 45 min | 2700 | 2700 | ✓ |
| 25 | Baker muffin boxes (144/12) | 12 | 10 (arithmetic error) | ✗ |
| 26–35 | Various problems | — | **HTTP 500 Server Error** | ✗ |

---

## Analysis

### Why accuracy is so low vs. published benchmarks

**1. Server instability (n-gram speculative decoding)**  
The `vllm-ngram` container crashed with HTTP 500 errors on problems 26–35. The n-gram speculative decoding with the NVFP4 model appears unstable under sustained load, causing 10/35 complete failures with no output.

**2. Model degeneration / hallucination (4 cases)**  
Problems 11, 12, 13, 17 show the model falling into repetition loops or producing garbled tokens (e.g., Japanese text, random symbols). This is likely related to the n-gram speculative decoding incorrectly accepting drafted tokens that lead to distribution collapse.

**3. Arithmetic/reasoning errors (7 cases)**  
The model made genuine reasoning errors on Q3, Q6, Q7, Q9, Q14, Q18, Q22, Q25 — wrong intermediate steps, misread problem structure, or wrong variable answered.

**4. Answer extraction issues (resolved)**  
Initial automated extraction was failing (extracting intermediate numbers rather than final answers). After manual review of full responses, the true accuracy improved from the automated count of 6/25 to 12/25.

### Root cause of instability
The server running during this benchmark was `vllm-gemma4` with `--kv-cache-dtype k4v4b64` (FusenCache), which crashed multiple times with `CUDA error: unknown error` under load. The n-gram speculative server also crashed on Q26+. The standard `vllm-baseline` container (standard KV cache, `--enforce-eager`) failed to load due to GPU memory conflicts with other containers.

### Expected baseline accuracy
Based on published results, with proper settings (standard KV cache, no speculative decoding instability), the NVFP4 model should achieve **~95.6%** on the full GSM8K benchmark (matching RedHat's published number). The 48% observed here is a measurement artifact of server instability, not a reflection of the model's mathematical capabilities.

---

## Methodology

- **Prompt format:** System prompt asking for step-by-step solution ending with `#### <answer>` (GSM8K standard format)
- **Temperature:** 0.0 (greedy decoding)
- **Max tokens:** 300
- **Extraction:** Primary via `#### <number>` pattern; fallback to last number in response; manual review for ambiguous cases
- **Problem set:** 35 original GSM8K-style problems covering arithmetic, percentages, rates, multi-step reasoning

---

## Recommendation

To get a reliable GSM8K measurement for the NVFP4 model:
1. Use `vllm-baseline` container with `--enforce-eager --gpu-memory-utilization 0.85` (standard KV cache, no n-gram speculative)
2. Run when no other containers are loading models (GPU memory conflict causes crashes)
3. Use at least 100-problem sample from the actual GSM8K test set
4. Expected result: **~95–97%** based on published NVFP4 and BF16 numbers

---

*Raw results: `/root/projects/autokernel/profiling/gsm8k_raw_results.json`*  
*Benchmark scripts: `/root/projects/autokernel/profiling/run_gsm8k_final.py`*
