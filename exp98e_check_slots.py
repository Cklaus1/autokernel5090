#!/usr/bin/env python3
"""Check if the MTP drafting loop writes to out-of-bounds KV cache slots."""

import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # sync to catch exact error

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"

import torch

def patch_check_slots():
    """Instrument the drafting loop to validate slot_mapping values."""
    import vllm.v1.spec_decode.eagle as m

    _orig_model_call = None
    _call_count = [0]

    orig_propose = m.EagleProposer.propose

    def patched(self, target_token_ids, target_positions, target_hidden_states,
                next_token_ids, token_indices_to_sample, common_attn_metadata,
                sampling_metadata, mm_embed_inputs=None,
                num_rejected_tokens_gpu=None, slot_mappings=None):
        # Capture the kv_cache shape info
        batch_size = common_attn_metadata.batch_size()
        seq_lens = common_attn_metadata.seq_lens[:batch_size].clone()
        block_table = common_attn_metadata.block_table_tensor[:batch_size].clone()

        print(f"\n[SLOT CHECK] batch={batch_size}, seq_lens={seq_lens.tolist()[:4]}..., "
              f"block_table shape={block_table.shape}, "
              f"block_table[:4]:\n{block_table[:4].tolist()}")

        # Check what slots the drafting loop would write to
        if self.num_speculative_tokens > 1:
            if hasattr(self, 'draft_attn_groups') and self.draft_attn_groups:
                block_size = self.draft_attn_groups[0].kv_cache_spec.block_size
                print(f"[SLOT CHECK] block_size={block_size}, max_model_len={self.max_model_len}")

                # Simulate what the loop does
                if self.uses_mrope:
                    positions = self.mrope_positions[:, :batch_size]
                else:
                    positions_val = target_positions
                    if token_indices_to_sample is not None:
                        positions_val = target_positions[token_indices_to_sample]
                    else:
                        positions_val = target_positions[-batch_size:]

                print(f"[SLOT CHECK] current positions (last {batch_size})={positions_val.tolist()[:4]}...")

                for step in range(self.num_speculative_tokens - 1):
                    positions_val = positions_val + 1
                    block_numbers = positions_val // block_size
                    max_block_col = block_table.shape[1] - 1
                    if (block_numbers > max_block_col).any():
                        print(f"[SLOT CHECK] *** OOB! step={step}, block_numbers={block_numbers.tolist()[:4]}..., "
                              f"max_col={max_block_col}")
                    else:
                        block_ids = block_table.gather(dim=1, index=block_numbers.view(-1, 1)).view(-1)
                        slots = block_ids * block_size + positions_val % block_size
                        print(f"[SLOT CHECK] step={step}, positions={positions_val.tolist()[:4]}..., "
                              f"block_nums={block_numbers.tolist()[:4]}, "
                              f"slots={slots.tolist()[:4]}...")

        return orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu,
            slot_mappings,
        )

    m.EagleProposer.propose = patched
    print("[PATCH] Slot checking enabled")


patch_check_slots()
from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=30, temperature=0.0)

# Minimal warmup
llm.generate([PROMPT], sp)

print("\n\n=== CRITICAL: batch=4 with 30 tokens ===")
try:
    outputs = llm.generate([PROMPT]*4, sp)
    print("batch=4: OK")
except Exception as e:
    print(f"batch=4: CRASHED — {str(e)[:60]}")
