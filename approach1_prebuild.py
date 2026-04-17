#!/usr/bin/env python3
"""Approach 1: Pre-build FlashInfer cache via baseline, then run DFlash with CUDA graphs"""
import os
os.environ['CC'] = '/usr/bin/gcc-12'
os.environ['CXX'] = '/usr/bin/g++-12'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9'

import time, numpy as np
from vllm import LLM, SamplingParams

# Step 1: Build cache with baseline (CUDA graphs enabled, no DFlash)
print("Step 1: Building FlashInfer cache with baseline...")
llm = LLM(model='Kbenkhaled/Qwen3.5-9B-NVFP4', dtype='bfloat16', trust_remote_code=True,
          gpu_memory_utilization=0.85, max_model_len=2048, language_model_only=True)
out = llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=10))
print(f"Baseline OK: {out[0].outputs[0].text[:50]}")
del llm

import gc, torch
gc.collect(); torch.cuda.empty_cache()
time.sleep(5)

# Step 2: DFlash with CUDA graphs (using pre-built cache)
print("\nStep 2: DFlash with CUDA graphs...")
llm = LLM(
    model='Kbenkhaled/Qwen3.5-9B-NVFP4',
    speculative_config={'method':'dflash','model':'z-lab/Qwen3.5-9B-DFlash','num_speculative_tokens':16},
    dtype='bfloat16', trust_remote_code=True, gpu_memory_utilization=0.85,
    max_model_len=2048, language_model_only=True,
)
llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=5))

out = llm.generate(['<|im_start|>user\nExplain TCP in 50 words.<|im_end|>\n<|im_start|>assistant\n'],
                   SamplingParams(temperature=0.0, max_tokens=100))
text = out[0].outputs[0].text
quality = 'OK' if len(set(text[:50])) >= 5 else 'GARBAGE'
print(f'Quality: {quality}')

results = []
for i in range(5):
    t0 = time.perf_counter()
    out = llm.generate(['<|im_start|>user\nWrite a BST in Python.<|im_end|>\n<|im_start|>assistant\n'],
                       SamplingParams(temperature=0.0, max_tokens=256))
    elapsed = time.perf_counter() - t0
    tok = len(out[0].outputs[0].token_ids)
    results.append(tok / elapsed)
    print(f'  Run {i+1}: {tok/elapsed:.1f} tok/s')

print(f'RESULT approach1: avg={np.mean(results):.1f} std={np.std(results):.1f} quality={quality}')
