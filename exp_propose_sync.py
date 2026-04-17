#!/usr/bin/env python3
"""Add syncs inside propose() to find exact crash point."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

# No-op target model Mamba kernels
import vllm.model_executor.layers.mamba.short_conv as sc
import vllm.model_executor.models.qwen3_next as qn
def _noop_conv(x, *a, **kw): return torch.zeros_like(x)
sc.causal_conv1d_update = _noop_conv
qn.causal_conv1d_update = _noop_conv
_orig_fr = qn.fused_recurrent_gated_delta_rule
def _noop_fr(**kwargs):
    v = kwargs['v']; ist = kwargs.get('initial_state')
    return torch.zeros_like(v), ist
qn.fused_recurrent_gated_delta_rule = lambda *a, **kw: _noop_fr(**kw)

# Patch propose to add sync at every stage
import vllm.v1.spec_decode.eagle as eagle_mod
_orig_propose = eagle_mod.EagleProposer.propose

def _synced_propose(self, target_token_ids, target_positions, target_hidden_states,
                    next_token_ids, token_indices_to_sample, common_attn_metadata,
                    sampling_metadata, mm_embed_inputs=None,
                    num_rejected_tokens_gpu=None, slot_mappings=None):
    # Sync before
    torch.cuda.synchronize()
    result = _orig_propose(self, target_token_ids, target_positions, target_hidden_states,
                           next_token_ids, token_indices_to_sample, common_attn_metadata,
                           sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
    # Sync after
    torch.cuda.synchronize()
    return result

eagle_mod.EagleProposer.propose = _synced_propose

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Hi"], sp, use_tqdm=False)
for bs in [4, 8, 16, 32]:
    try:
        llm.generate(["Hi"] * bs, sp, use_tqdm=False)
        print(f"batch={bs}: OK")
    except:
        print(f"batch={bs}: CRASH")
        break
