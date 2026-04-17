#!/usr/bin/env python3
"""Batch throughput sweep for vLLM + DFlash + NVFP4."""
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
] * 64  # 512 prompts

if __name__ == '__main__':
    use_dflash = '--dflash' in sys.argv
    label = "DFlash" if use_dflash else "Baseline"

    kwargs = dict(
        model='Kbenkhaled/Qwen3.5-9B-NVFP4',
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        language_model_only=True,
    )
    if use_dflash:
        kwargs['speculative_config'] = {
            'method': 'dflash',
            'model': 'z-lab/Qwen3.5-9B-DFlash',
            'num_speculative_tokens': 6,
        }

    print(f"=== {label} Batch Sweep ===")
    llm = LLM(**kwargs)

    # Warmup
    llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=5))
    llm.generate(PROMPTS[:4], SamplingParams(temperature=0.0, max_tokens=50))

    batch_sizes = [1, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
    print(f"{'BS':>5} | {'Total tok/s':>12} | {'Per-user':>10} | {'Tokens':>8} | {'Time':>8}")
    print("-" * 60)

    for bs in batch_sizes:
        batch = PROMPTS[:bs]
        try:
            t0 = time.perf_counter()
            outs = llm.generate(batch, SamplingParams(temperature=0.0, max_tokens=128))
            elapsed = time.perf_counter() - t0
            total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            tok_s = total_tok / elapsed
            per_user = tok_s / bs
            print(f"{bs:>5} | {tok_s:>12.0f} | {per_user:>10.1f} | {total_tok:>8} | {elapsed:>7.1f}s")
        except Exception as e:
            print(f"{bs:>5} | FAILED: {str(e)[:50]}")
            break

    print(f"\n{label} sweep complete.")
