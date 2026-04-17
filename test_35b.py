#!/usr/bin/env python3
"""Test Qwen3.5-35B-A3B NVFP4 on RTX 5090."""
import sys, time, numpy as np
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    print("Loading Qwen3.5-35B-A3B NVFP4...")
    llm = LLM(
        model='Sehyo/Qwen3.5-35B-A3B-NVFP4',
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=0.95,  # Need max memory for 25GB model
        max_model_len=2048,
        language_model_only=True,
        enforce_eager=True,  # Start simple
    )

    # Quality check
    print("Quality check...")
    out = llm.generate(
        ['<|im_start|>user\nExplain TCP in 50 words.<|im_end|>\n<|im_start|>assistant\n'],
        SamplingParams(temperature=0.0, max_tokens=100))
    text = out[0].outputs[0].text
    quality = 'OK' if len(set(text[:50])) >= 5 else 'GARBAGE'
    print(f'Quality: {quality}')
    print(f'Output: {text[:200]}')

    # Decode speed
    print("\nDecode benchmark:")
    results = []
    for i in range(5):
        t0 = time.perf_counter()
        out = llm.generate(
            ['<|im_start|>user\nWrite quicksort in Python.<|im_end|>\n<|im_start|>assistant\n'],
            SamplingParams(temperature=0.0, max_tokens=256))
        elapsed = time.perf_counter() - t0
        tok = len(out[0].outputs[0].token_ids)
        results.append(tok / elapsed)
        print(f'  Run {i+1}: {tok/elapsed:.1f} tok/s ({tok} tokens)')

    # Batch test
    print("\nBatch test:")
    PROMPTS = ['<|im_start|>user\nWrite quicksort.<|im_end|>\n<|im_start|>assistant\n'] * 64
    for bs in [1, 4, 8, 16, 32]:
        try:
            t0 = time.perf_counter()
            outs = llm.generate(PROMPTS[:bs], SamplingParams(temperature=0.0, max_tokens=128))
            elapsed = time.perf_counter() - t0
            total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            print(f'  bs={bs}: {total_tok/elapsed:.0f} tok/s')
        except Exception as e:
            print(f'  bs={bs}: FAILED ({str(e)[:50]})')
            break

    avg = np.mean(results[1:])
    print(f'\nRESULT 35B: decode={avg:.1f} quality={quality}')
