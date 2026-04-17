#!/usr/bin/env python3
"""Test: enforce_eager with sync points."""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"

from vllm import LLM, SamplingParams
llm = LLM(
    model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False, enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=100, temperature=0.0)
llm.generate([PROMPT], sp)
llm.generate([PROMPT], sp)
for bs in [4, 8, 16, 32]:
    try:
        outputs = llm.generate([PROMPT]*bs, sp)
        ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"batch={bs}: OK ({ntok} tokens)")
    except Exception as e:
        print(f"batch={bs}: CRASHED")
        break
