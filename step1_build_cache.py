#!/usr/bin/env python3
"""Step 1: Build FlashInfer cache with baseline (no DFlash, no -c flag)"""
import os, time
os.environ.setdefault('CC', '/usr/bin/gcc-12')
os.environ.setdefault('CXX', '/usr/bin/g++-12')
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.9')

from vllm import LLM, SamplingParams

llm = LLM(model='Kbenkhaled/Qwen3.5-9B-NVFP4', dtype='bfloat16', trust_remote_code=True,
          gpu_memory_utilization=0.85, max_model_len=2048, language_model_only=True)
out = llm.generate(['<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'],
                   SamplingParams(temperature=0.0, max_tokens=20))
print(f'CACHE_OK: {out[0].outputs[0].text[:100]}')
