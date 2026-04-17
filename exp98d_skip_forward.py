#!/usr/bin/env python3
"""Skip the model forward in the drafting loop — only do metadata updates."""

import os, gc, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"

import torch

def patch_skip_loop_forward():
    """Patch: run the loop metadata updates but skip the actual model forward."""
    import vllm.v1.spec_decode.eagle as m

    original_propose = m.EagleProposer.propose

    def patched(self, target_token_ids, target_positions, target_hidden_states,
                next_token_ids, token_indices_to_sample, common_attn_metadata,
                sampling_metadata, mm_embed_inputs=None,
                num_rejected_tokens_gpu=None, slot_mappings=None):
        # Run first pass normally
        orig_n = self.num_speculative_tokens
        self.num_speculative_tokens = 1
        result = original_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu,
            slot_mappings,
        )
        self.num_speculative_tokens = orig_n

        # Now simulate the metadata updates from the loop WITHOUT running model
        batch_size = common_attn_metadata.batch_size()
        if self.uses_mrope:
            positions = self.mrope_positions[:, :batch_size]
        else:
            positions = self.positions[:batch_size]

        for token_index in range(orig_n - 1):
            positions = positions + 1  # NOT in-place
            common_attn_metadata.seq_lens = common_attn_metadata.seq_lens + 1
            # Simulate slot_mapping recompute
            block_size = self.draft_attn_groups[0].kv_cache_spec.block_size
            if self.uses_mrope:
                block_numbers = positions[0] // block_size
            else:
                block_numbers = positions // block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            common_attn_metadata.slot_mapping = (
                block_ids * block_size + positions % block_size
            )

        # Pad result
        if result.dim() == 2 and result.shape[1] < orig_n:
            padding = result[:, -1:].expand(-1, orig_n - result.shape[1])
            result = torch.cat([result, padding], dim=1)
        return result

    m.EagleProposer.propose = patched
    print("[PATCH] Skip model forward in drafting loop (metadata updates only)")


patch_skip_loop_forward()
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
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
