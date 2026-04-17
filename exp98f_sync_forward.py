#!/usr/bin/env python3
"""Add torch.cuda.synchronize() after each model forward in the drafting loop."""

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"

import torch


def patch_sync_in_loop():
    """Add sync after each model forward in the drafting loop to catch the exact failure."""
    import vllm.v1.spec_decode.eagle as m

    orig_propose = m.EagleProposer.propose

    _step_count = [0]

    # Patch the model's __call__ to add sync
    def patched(self, *args, **kwargs):
        result = orig_propose(self, *args, **kwargs)
        return result

    # Instead, patch at the level of set_forward_context
    from vllm import forward_context as fc
    orig_set = fc.set_forward_context

    def synced_set(*args, **kwargs):
        ctx = orig_set(*args, **kwargs)
        class SyncedCtx:
            def __enter__(self_ctx):
                return ctx.__enter__()
            def __exit__(self_ctx, *exc_args):
                result = ctx.__exit__(*exc_args)
                # After every forward context exit, sync
                torch.cuda.synchronize()
                _step_count[0] += 1
                return result
        return SyncedCtx()

    fc.set_forward_context = synced_set
    m.set_forward_context = synced_set
    print("[PATCH] Sync after every model forward")


patch_sync_in_loop()
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=30, temperature=0.0)
llm.generate([PROMPT], sp)
llm.generate([PROMPT], sp)

for bs in [4, 8, 16, 32]:
    try:
        outputs = llm.generate([PROMPT]*bs, sp)
        ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
        print(f"batch={bs}: OK ({ntok} tokens)")
    except Exception as e:
        print(f"batch={bs}: CRASHED — {str(e)[:60]}")
        break
