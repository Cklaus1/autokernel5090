#!/usr/bin/env python3
"""Skip only the MTP head forward, do everything else."""
import os, gc, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

import torch
import vllm.v1.spec_decode.eagle as eagle_mod

_orig_propose = eagle_mod.EagleProposer.propose

def _skip_mtp_forward(self, target_token_ids, target_positions, target_hidden_states,
                      next_token_ids, token_indices_to_sample, common_attn_metadata,
                      sampling_metadata, mm_embed_inputs=None,
                      num_rejected_tokens_gpu=None, slot_mappings=None):
    """Skip the model.forward() in propose, return dummy tokens."""
    # Save and patch model to be a no-op
    real_model_call = self.model.__call__
    batch_size = common_attn_metadata.batch_size()
    hidden_size = self.hidden_size

    def fake_forward(**kwargs):
        n = kwargs.get('input_ids')
        nt = n.shape[0] if n is not None else kwargs['inputs_embeds'].shape[0]
        return torch.zeros(nt, hidden_size, device='cuda', dtype=torch.bfloat16)

    self.model.__call__ = fake_forward
    try:
        result = _orig_propose(
            self, target_token_ids, target_positions, target_hidden_states,
            next_token_ids, token_indices_to_sample, common_attn_metadata,
            sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    finally:
        self.model.__call__ = real_model_call
    return result

eagle_mod.EagleProposer.propose = _skip_mtp_forward
print("[FIX] Skipping MTP head forward (dummy hidden states)")

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
