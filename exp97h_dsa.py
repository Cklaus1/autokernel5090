#!/usr/bin/env python3
"""Debug with device-side assertions to find the exact crashing kernel."""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"


def main():
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
        speculative_config={
            "num_speculative_tokens": 3,
            "method": "qwen3_5_mtp",
        },
    )

    sp = SamplingParams(max_tokens=100, temperature=0.0)

    print("Warmup...")
    llm.generate([PROMPT], sp)
    llm.generate([PROMPT], sp)

    print("batch=2...")
    llm.generate([PROMPT]*2, sp)
    print("  OK")

    print("batch=4...")
    try:
        llm.generate([PROMPT]*4, sp)
        print("  OK")
    except Exception as e:
        print(f"  CRASHED: {e}")

    print("batch=8...")
    try:
        llm.generate([PROMPT]*8, sp)
        print("  OK")
    except Exception as e:
        print(f"  CRASHED: {e}")


if __name__ == "__main__":
    main()
