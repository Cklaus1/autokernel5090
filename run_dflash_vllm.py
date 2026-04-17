#!/usr/bin/env python3
"""Run vLLM + DFlash + NVFP4. usercustomize.py handles FlashInfer JIT patching."""
import sys, time, numpy as np
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

llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=5))

out = llm.generate(
    ['<|im_start|>user\nExplain TCP in 50 words.<|im_end|>\n<|im_start|>assistant\n'],
    SamplingParams(temperature=0.0, max_tokens=100),
)
text = out[0].outputs[0].text
quality = 'OK' if len(set(text[:50])) >= 5 else 'GARBAGE'
print(f'Quality: {quality}')
print(f'Output: {text[:200]}')

results = []
for i in range(5):
    t0 = time.perf_counter()
    out = llm.generate(
        ['<|im_start|>user\nWrite a BST in Python with insert and search.<|im_end|>\n<|im_start|>assistant\n'],
        SamplingParams(temperature=0.0, max_tokens=256),
    )
    elapsed = time.perf_counter() - t0
    tok = len(out[0].outputs[0].token_ids)
    results.append(tok / elapsed)
    print(f'  Run {i+1}: {tok/elapsed:.1f} tok/s ({tok} tokens)')

avg = np.mean(results)
print(f'\nRESULT: draft={DRAFT_TOKENS} avg={avg:.1f} std={np.std(results):.1f} quality={quality}')
