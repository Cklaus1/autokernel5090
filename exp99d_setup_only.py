#!/usr/bin/env python3
"""Run first pass + setup code but not the loop body."""
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

import torch
import vllm.v1.spec_decode.eagle as eagle_mod
orig = eagle_mod.EagleProposer.propose

def setup_no_loop(self, *args, **kwargs):
    """Run first pass, then setup, then skip loop."""
    save = self.num_speculative_tokens
    # Run first pass
    self.num_speculative_tokens = 1
    result = orig(self, *args, **kwargs)
    self.num_speculative_tokens = save

    # Now do the setup that normally happens before the loop
    cad = args[5] if len(args) > 5 else kwargs['common_attn_metadata']
    batch_size = cad.batch_size()

    # These are the lines 519-524 that modify common_attn_metadata
    cad.num_actual_tokens = batch_size
    cad.max_query_len = 1
    cad.query_start_loc = self.arange[: batch_size + 1]

    # Pad result
    if result.shape[1] < save:
        padding = result[:, -1:].expand(-1, save - result.shape[1])
        result = torch.cat([result, padding], dim=1)
    return result

eagle_mod.EagleProposer.propose = setup_no_loop
print("[PATCH] First pass + setup (no loop)")

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
