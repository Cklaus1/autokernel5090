#!/usr/bin/env python3
"""Minimal crash reproducer: does it crash on first batch=4 or on continuation?"""

import os, time, gc, traceback

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "DEBUG"

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

    sp_short = SamplingParams(max_tokens=5, temperature=0.0)
    sp_medium = SamplingParams(max_tokens=30, temperature=0.0)

    # Warmup single
    print("\n[1] Single request warmup...")
    llm.generate([PROMPT], sp_short)
    llm.generate([PROMPT], sp_short)
    print("  OK")

    # batch=2 short
    print("[2] batch=2, 5 tokens...")
    outputs = llm.generate([PROMPT]*2, sp_short)
    print(f"  OK: {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")

    # batch=4 very short (5 tokens) — does it crash even with very few decode steps?
    print("[3] batch=4, 5 tokens...")
    try:
        outputs = llm.generate([PROMPT]*4, sp_short)
        print(f"  OK: {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")
    except Exception as e:
        print(f"  CRASHED: {str(e)[:100]}")
        return

    # batch=4 medium (30 tokens)
    print("[4] batch=4, 30 tokens...")
    try:
        outputs = llm.generate([PROMPT]*4, sp_medium)
        print(f"  OK: {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")
    except Exception as e:
        print(f"  CRASHED: {str(e)[:100]}")
        return

    # batch=8 short
    print("[5] batch=8, 5 tokens...")
    try:
        outputs = llm.generate([PROMPT]*8, sp_short)
        print(f"  OK: {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")
    except Exception as e:
        print(f"  CRASHED: {str(e)[:100]}")
        return


if __name__ == "__main__":
    main()
