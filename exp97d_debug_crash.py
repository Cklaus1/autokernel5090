#!/usr/bin/env python3
"""Debug: MTP 3 batch=4 crash with CUDA_LAUNCH_BLOCKING=1."""

import os, time, gc, traceback

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"
MAX_TOKENS = 50  # Shorter to speed up


def main():
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

    sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

    # Warmup with single request
    print("Warmup single...")
    llm.generate([PROMPT], sp)
    llm.generate([PROMPT], sp)

    # Test batch=2 (should work)
    print("Testing batch=2...")
    try:
        outputs = llm.generate([PROMPT] * 2, sp)
        print(f"  batch=2 OK: {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")
    except Exception as e:
        print(f"  batch=2 CRASHED: {e}")
        traceback.print_exc()
        return

    # Test batch=3
    print("Testing batch=3...")
    try:
        outputs = llm.generate([PROMPT] * 3, sp)
        print(f"  batch=3 OK: {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")
    except Exception as e:
        print(f"  batch=3 CRASHED: {e}")
        traceback.print_exc()
        return

    # Test batch=4
    print("Testing batch=4...")
    try:
        outputs = llm.generate([PROMPT] * 4, sp)
        print(f"  batch=4 OK: {sum(len(o.outputs[0].token_ids) for o in outputs)} tokens")
    except Exception as e:
        print(f"  batch=4 CRASHED: {e}")
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
