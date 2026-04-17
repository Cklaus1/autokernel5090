#!/usr/bin/env python3
"""Isolate the crash: skip MTP drafting loop, only return first draft token.

If crash goes away → bug is in the drafting loop.
If crash persists → bug is in the target model forward with MTP config.
"""

import os, gc, time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"


def patch_skip_drafting_loop():
    """Monkey-patch EagleProposer to skip the multi-step drafting loop."""
    import vllm.v1.spec_decode.eagle as eagle_mod
    original_propose = eagle_mod.EagleProposer.propose

    def patched_propose(self, *args, **kwargs):
        # Temporarily pretend we only have 1 speculative token
        orig = self.num_speculative_tokens
        self.num_speculative_tokens = 1
        result = original_propose(self, *args, **kwargs)
        self.num_speculative_tokens = orig
        # Pad result to expected shape [batch, num_spec_tokens]
        if result.dim() == 2 and result.shape[1] < orig:
            import torch
            padding = result[:, -1:].expand(-1, orig - result.shape[1])
            result = torch.cat([result, padding], dim=1)
        return result

    eagle_mod.EagleProposer.propose = patched_propose
    print("[PATCH] Skipping MTP drafting loop (single draft token only)")


def main():
    import torch
    from vllm import LLM, SamplingParams

    print("=== Test A: WITH drafting loop (should crash at batch=4) ===")
    llm = LLM(
        model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
        max_model_len=4096, enable_chunked_prefill=False,
        speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
    )
    sp = SamplingParams(max_tokens=50, temperature=0.0)
    llm.generate([PROMPT], sp)
    llm.generate([PROMPT], sp)

    for bs in [2, 4, 8]:
        try:
            outputs = llm.generate([PROMPT]*bs, sp)
            ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
            print(f"  batch={bs}: OK ({ntok} tokens)")
        except Exception as e:
            print(f"  batch={bs}: CRASHED — {str(e)[:60]}")
            break

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)

    print("\n=== Test B: WITHOUT drafting loop (skip to single draft token) ===")
    patch_skip_drafting_loop()

    llm2 = LLM(
        model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
        max_model_len=4096, enable_chunked_prefill=False,
        speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
    )
    llm2.generate([PROMPT], sp)
    llm2.generate([PROMPT], sp)

    for bs in [2, 4, 8, 16, 32]:
        try:
            outputs = llm2.generate([PROMPT]*bs, sp)
            ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
            print(f"  batch={bs}: OK ({ntok} tokens)")
        except Exception as e:
            print(f"  batch={bs}: CRASHED — {str(e)[:60]}")
            break


if __name__ == "__main__":
    main()
