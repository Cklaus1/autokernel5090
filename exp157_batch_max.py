#!/usr/bin/env python3
"""Batch optimization experiments: push toward 15K tok/s ceiling."""
import sys, time, numpy as np
from vllm import LLM, SamplingParams

PROMPTS = ["<|im_start|>user\nWrite quicksort.<|im_end|>\n<|im_start|>assistant\n"] * 512
MODE = sys.argv[1] if len(sys.argv) > 1 else "mamba_fp16_fp8kv"

if __name__ == '__main__':
    kwargs = dict(
        model='Kbenkhaled/Qwen3.5-9B-NVFP4',
        dtype='bfloat16', trust_remote_code=True,
        max_model_len=2048, language_model_only=True,
        kv_cache_dtype='fp8_e5m2', mamba_ssm_cache_dtype='float16',
    )

    if MODE == "gpu92":
        kwargs['gpu_memory_utilization'] = 0.92
        label = "Mamba FP16 + FP8 KV + gpu=0.92"
    elif MODE == "noprefix":
        kwargs['gpu_memory_utilization'] = 0.90
        kwargs['enable_prefix_caching'] = False
        label = "Mamba FP16 + FP8 KV + no prefix"
    elif MODE == "ctx1024":
        kwargs['gpu_memory_utilization'] = 0.90
        kwargs['max_model_len'] = 1024
        label = "Mamba FP16 + FP8 KV + ctx=1024"
    elif MODE == "ctx1024_noprefix":
        kwargs['gpu_memory_utilization'] = 0.90
        kwargs['max_model_len'] = 1024
        kwargs['enable_prefix_caching'] = False
        label = "Mamba FP16 + FP8 KV + ctx=1024 + no prefix"
    elif MODE == "dflash_combo":
        kwargs['gpu_memory_utilization'] = 0.90
        kwargs['speculative_config'] = {
            'method': 'dflash',
            'model': 'z-lab/Qwen3.5-9B-DFlash',
            'num_speculative_tokens': 6,
        }
        label = "DFlash + Mamba FP16 + FP8 KV"
    else:
        kwargs['gpu_memory_utilization'] = 0.90
        label = f"Custom: {MODE}"

    print(f"=== {label} ===")
    llm = LLM(**kwargs)
    llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=5))
    llm.generate(PROMPTS[:8], SamplingParams(temperature=0.0, max_tokens=50))

    # Decode
    results = []
    for i in range(5):
        t0 = time.perf_counter()
        out = llm.generate(['<|im_start|>user\nWrite a BST.<|im_end|>\n<|im_start|>assistant\n'],
                           SamplingParams(temperature=0.0, max_tokens=256))
        elapsed = time.perf_counter() - t0
        tok = len(out[0].outputs[0].token_ids)
        results.append(tok / elapsed)
    decode = np.mean(results[1:])
    print(f"Decode: {decode:.1f} tok/s")

    # Batch sweep
    peak_bs, peak_tps = 0, 0
    for bs in [64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]:
        try:
            t0 = time.perf_counter()
            outs = llm.generate(PROMPTS[:bs], SamplingParams(temperature=0.0, max_tokens=128))
            elapsed = time.perf_counter() - t0
            total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
            tok_s = total_tok / elapsed
            print(f"  bs={bs}: {tok_s:.0f} tok/s")
            if tok_s > peak_tps:
                peak_tps, peak_bs = tok_s, bs
        except:
            print(f"  bs={bs}: FAILED")
            break

    print(f"RESULT [{MODE}]: decode={decode:.1f} peak={peak_tps:.0f}@bs{peak_bs}")
