#!/usr/bin/env python3
"""Pinpoint: is the crash in propose()'s FIRST PASS or DRAFTING LOOP?"""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

# No-op Mamba kernels in target model
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

# Instrument propose() to sync after first pass
import vllm.v1.spec_decode.eagle as eagle_mod
_orig = eagle_mod.EagleProposer.propose

def _instrumented(self, target_token_ids, target_positions, target_hidden_states,
                  next_token_ids, token_indices_to_sample, common_attn_metadata,
                  sampling_metadata, mm_embed_inputs=None,
                  num_rejected_tokens_gpu=None, slot_mappings=None):
    # Run with num_spec=1 to isolate first pass only
    save = self.num_speculative_tokens
    self.num_speculative_tokens = 1
    try:
        result = _orig(self, target_token_ids, target_positions, target_hidden_states,
                       next_token_ids, token_indices_to_sample, common_attn_metadata,
                       sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)
        # If first pass succeeds, sync to catch async errors
        torch.cuda.synchronize()
        print(".", end="", flush=True)
    except Exception as e:
        print(f"\n[FIRST PASS CRASHED: {str(e)[:40]}]")
        raise
    finally:
        self.num_speculative_tokens = save

    if result.shape[1] < save:
        padding = result[:, -1:].expand(-1, save - result.shape[1])
        result = torch.cat([result, padding], dim=1)
    return result

eagle_mod.EagleProposer.propose = _instrumented

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Hi"], sp, use_tqdm=False)
print("batch=8: ", end="")
try:
    llm.generate(["Hi"] * 8, sp, use_tqdm=False)
    print(" OK")
except:
    print(" CRASHED")
