#!/usr/bin/env python3
"""Benchmark vLLM + DFlash with NVFP4 - env vars set first."""
import os
os.environ['CC'] = '/usr/bin/gcc-12'
os.environ['CXX'] = '/usr/bin/g++-12'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9'
os.environ['PATH'] = '/usr/local/cuda-12.9/bin:' + os.environ.get('PATH', '')

# These must be set before multiprocessing spawn
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import sys
import time
import numpy as np
from vllm import LLM, SamplingParams

DRAFT_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 16

print(f"Config: draft_tokens={DRAFT_TOKENS}")

llm = LLM(
    model='Kbenkhaled/Qwen3.5-9B-NVFP4',
    speculative_config={
        'method': 'dflash',
        'model': 'z-lab/Qwen3.5-9B-DFlash',
        'num_speculative_tokens': DRAFT_TOKENS,
    },
    dtype='bfloat16',
    trust_remote_code=True,
    gpu_memory_utilization=0.85,
    max_model_len=2048,
    language_model_only=True,
)

llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=10))

results = []
for i in range(5):
    t0 = time.perf_counter()
    out = llm.generate(
        ['<|im_start|>user\nWrite a binary search tree in Python.<|im_end|>\n<|im_start|>assistant\n'],
        SamplingParams(temperature=0.0, max_tokens=256)
    )
    elapsed = time.perf_counter() - t0
    tok = len(out[0].outputs[0].token_ids)
    results.append(tok / elapsed)
    print(f"  Run {i+1}: {tok/elapsed:.1f} tok/s")

out = llm.generate(
    ['<|im_start|>user\nExplain TCP in 50 words.<|im_end|>\n<|im_start|>assistant\n'],
    SamplingParams(temperature=0.0, max_tokens=100)
)
text = out[0].outputs[0].text
quality = 'GARBAGE' if len(set(text[:50])) < 5 else 'OK'

avg = np.mean(results)
print(f"\nRESULT: draft={DRAFT_TOKENS} avg={avg:.1f} tok/s quality={quality}")
