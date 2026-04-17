#!/usr/bin/env python3
"""
Experiments 150+: All optimizations applied together.
1. FP8 e5m2 KV cache (unblocked)
2. gpu_memory_utilization=0.92
3. Disable prefix caching
4. No chunked prefill
5. DFlash draft=6

Run as batch sweep to find peak throughput.
"""
import sys, time, numpy as np
from vllm import LLM, SamplingParams

PROMPTS = [
    "<|im_start|>user\nWrite quicksort in Python.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain how TCP works.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a linked list class.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nHow does garbage collection work?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain binary search trees.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a merge sort implementation.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nExplain hash tables and collisions.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWrite a graph BFS traversal.<|im_end|>\n<|im_start|>assistant\n",
] * 64

MODE = sys.argv[1] if len(sys.argv) > 1 else "baseline"

if __name__ == '__main__':
    kwargs = dict(
        model='Kbenkhaled/Qwen3.5-9B-NVFP4',
        dtype='bfloat16',
        trust_remote_code=True,
        max_model_len=2048,
        language_model_only=True,
    )

    if MODE == "baseline":
        kwargs['gpu_memory_utilization'] = 0.90
        label = "Baseline (auto KV, 0.90)"
    elif MODE == "fp8kv":
        kwargs['gpu_memory_utilization'] = 0.90
        kwargs['kv_cache_dtype'] = 'fp8_e5m2'
        label = "FP8 e5m2 KV, 0.90"
    elif MODE == "fp8kv_92":
        kwargs['gpu_memory_utilization'] = 0.92
        kwargs['kv_cache_dtype'] = 'fp8_e5m2'
        label = "FP8 e5m2 KV, 0.92"
    elif MODE == "fp8kv_noprefix":
        kwargs['gpu_memory_utilization'] = 0.92
        kwargs['kv_cache_dtype'] = 'fp8_e5m2'
        kwargs['enable_prefix_caching'] = False
        label = "FP8 e5m2 KV, 0.92, no prefix cache"
    elif MODE == "dflash_fp8kv":
        kwargs['gpu_memory_utilization'] = 0.90
        kwargs['kv_cache_dtype'] = 'fp8_e5m2'
        kwargs['speculative_config'] = {
            'method': 'dflash',
            'model': 'z-lab/Qwen3.5-9B-DFlash',
            'num_speculative_tokens': 6,
        }
        label = "DFlash draft=6 + FP8 KV"
    elif MODE == "dflash_decode":
        kwargs['gpu_memory_utilization'] = 0.90
        kwargs['speculative_config'] = {
            'method': 'dflash',
            'model': 'z-lab/Qwen3.5-9B-DFlash',
            'num_speculative_tokens': 6,
        }
        label = "DFlash draft=6 (decode focus)"
    else:
        print(f"Unknown mode: {MODE}")
        sys.exit(1)

    print(f"=== {label} ===")
    llm = LLM(**kwargs)

    # Warmup
    llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=5))
    llm.generate(PROMPTS[:4], SamplingParams(temperature=0.0, max_tokens=50))

    # Single decode
    results = []
    for i in range(5):
        t0 = time.perf_counter()
        out = llm.generate(
            ['<|im_start|>user\nWrite a BST in Python.<|im_end|>\n<|im_start|>assistant\n'],
            SamplingParams(temperature=0.0, max_tokens=256))
        elapsed = time.perf_counter() - t0
        tok = len(out[0].outputs[0].token_ids)
        results.append(tok / elapsed)
    steady_decode = np.mean(results[1:])  # skip first run
    print(f"Decode: {steady_decode:.1f} tok/s (steady)")

    # Batch sweep
    print(f"{'BS':>5} | {'Total tok/s':>12} | {'Per-user':>10}")
    print("-" * 40)
    peak_bs, peak_tps = 0, 0
    for bs in [1, 8, 32, 64, 96, 120, 128, 160, 192, 224, 256, 288, 320]:
        batch = PROMPTS[:bs]
        try:
            t0 = time.perf_counter()
            outs = llm.generate(batch, SamplingParams(temperature=0.0, max_tokens=128))
            elapsed = time.perf_counter() - t0
            total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            tok_s = total_tok / elapsed
            print(f"{bs:>5} | {tok_s:>12.0f} | {tok_s/bs:>10.1f}")
            if tok_s > peak_tps:
                peak_tps = tok_s
                peak_bs = bs
        except Exception as e:
            print(f"{bs:>5} | FAILED: {str(e)[:60]}")
            break

    print(f"\nRESULT [{MODE}]: decode={steady_decode:.1f} peak_batch={peak_tps:.0f}@bs{peak_bs}")
