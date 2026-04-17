#!/usr/bin/env python3
"""Run only the first pass of propose() — skip loop."""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

import torch
import vllm.v1.spec_decode.eagle as eagle_mod
orig = eagle_mod.EagleProposer.propose

def first_pass_only(self, *args, **kwargs):
    save = self.num_speculative_tokens
    self.num_speculative_tokens = 1  # triggers early return at line 473
    result = orig(self, *args, **kwargs)
    self.num_speculative_tokens = save
    # Pad to expected shape
    if result.shape[1] < save:
        padding = result[:, -1:].expand(-1, save - result.shape[1])
        result = torch.cat([result, padding], dim=1)
    return result

eagle_mod.EagleProposer.propose = first_pass_only
print("[PATCH] First pass only (early return)")

from vllm import LLM, SamplingParams
llm = LLM(
    model="Kbenkhaled/Qwen3.5-9B-NVFP4",
    gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=100, temperature=0.0)
llm.generate(["Write merge sort:"], sp)
llm.generate(["Write merge sort:"], sp)
for bs in [4, 8, 16, 32]:
    try:
        outputs = llm.generate(["Write merge sort:"] * bs, sp)
        ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"batch={bs}: OK ({ntok} tokens)")
    except Exception as e:
        print(f"batch={bs}: CRASHED")
        break
