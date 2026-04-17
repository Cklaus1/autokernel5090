#!/usr/bin/env python3
"""Debug: MTP 3 race condition — test async vs sync execution.

CUDA_LAUNCH_BLOCKING=1 makes batch=4 work. Without it, crashes at batch>=4.
This is a race condition in async scheduling/execution.

Test hypothesis: the crash is in the speculator's draft generation running
concurrently with the next step's input preparation.
"""

import os, time, gc, traceback

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"
# NO CUDA_LAUNCH_BLOCKING — we want the race condition

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"
MAX_TOKENS = 50


def main():
    import torch
    from vllm import LLM, SamplingParams

    print("=== Test 1: MTP 3 with enforce_eager=True ===")
    print("(Disables CUDA graphs, keeps async scheduling)")
    try:
        llm = LLM(
            model=MODEL,
            gpu_memory_utilization=0.90,
            max_num_seqs=128,
            max_model_len=4096,
            enable_chunked_prefill=False,
            enforce_eager=True,
            speculative_config={
                "num_speculative_tokens": 3,
                "method": "qwen3_5_mtp",
            },
        )
        sp = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)
        llm.generate([PROMPT], sp)  # warmup
        llm.generate([PROMPT], sp)  # warmup

        for bs in [2, 3, 4, 8]:
            try:
                outputs = llm.generate([PROMPT] * bs, sp)
                ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
                print(f"  batch={bs}: OK ({ntok} tokens)")
            except Exception as e:
                print(f"  batch={bs}: CRASHED — {str(e)[:80]}")
                break

        del llm
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(3)
    except Exception as e:
        print(f"  Init failed: {e}")

    print("\n=== Test 2: MTP 3 with CUDA graphs (default) ===")
    print("(This should crash around batch=4)")
    try:
        llm2 = LLM(
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
        llm2.generate([PROMPT], sp)  # warmup
        llm2.generate([PROMPT], sp)

        for bs in [2, 3, 4, 8]:
            try:
                outputs = llm2.generate([PROMPT] * bs, sp)
                ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
                print(f"  batch={bs}: OK ({ntok} tokens)")
            except Exception as e:
                print(f"  batch={bs}: CRASHED — {str(e)[:80]}")
                break
    except Exception as e:
        print(f"  Init failed: {e}")


if __name__ == "__main__":
    main()
