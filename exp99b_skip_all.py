#!/usr/bin/env python3
"""Skip ALL of propose() — return dummy draft tokens."""
import os, gc, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

import torch
import vllm.v1.spec_decode.eagle as eagle_mod
orig = eagle_mod.EagleProposer.propose

def noop_propose(self, *args, **kwargs):
    cad = args[5] if len(args) > 5 else kwargs['common_attn_metadata']
    bs = cad.batch_size()
    return torch.zeros(bs, self.num_speculative_tokens, dtype=torch.int64, device='cuda')

eagle_mod.EagleProposer.propose = noop_propose
print("[PATCH] Propose returns dummy tokens (no model forward at all)")

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
