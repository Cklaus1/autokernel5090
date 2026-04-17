#!/usr/bin/env python3
"""Test baseline vLLM without DFlash."""
from vllm import LLM, SamplingParams

if __name__ == '__main__':
    llm = LLM(model='Kbenkhaled/Qwen3.5-9B-NVFP4', dtype='bfloat16', trust_remote_code=True,
              gpu_memory_utilization=0.85, max_model_len=2048, language_model_only=True, enforce_eager=True)
    out = llm.generate(['<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n'],
                       SamplingParams(temperature=0.0, max_tokens=20))
    print(f'BASELINE: {out[0].outputs[0].text[:100]}')
