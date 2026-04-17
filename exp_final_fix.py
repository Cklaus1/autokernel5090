#!/usr/bin/env python3
"""The REAL fix: shallow-copy common_attn_metadata in propose() + disable async."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

from copy import copy as shallow_copy
import vllm.v1.spec_decode.eagle as eagle_mod

_orig = eagle_mod.EagleProposer.propose

def _fixed(self, target_token_ids, target_positions, target_hidden_states,
           next_token_ids, token_indices_to_sample, common_attn_metadata,
           sampling_metadata, mm_embed_inputs=None,
           num_rejected_tokens_gpu=None, slot_mappings=None):
    if self.num_speculative_tokens > 1:
        # Deep copy ALL tensor fields to prevent ANY mutation of the original
        cad = type(common_attn_metadata)(
            query_start_loc=common_attn_metadata.query_start_loc.clone(),
            query_start_loc_cpu=common_attn_metadata.query_start_loc_cpu.clone()
                if common_attn_metadata.query_start_loc_cpu is not None else None,
            seq_lens=common_attn_metadata.seq_lens.clone(),
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu.clone()
                if common_attn_metadata._seq_lens_cpu is not None else None,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu.clone()
                if common_attn_metadata._num_computed_tokens_cpu is not None else None,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping.clone()
                if common_attn_metadata.slot_mapping is not None else None,
            causal=common_attn_metadata.causal,
            logits_indices_padded=common_attn_metadata.logits_indices_padded,
            num_logits_indices=common_attn_metadata.num_logits_indices,
        )
        common_attn_metadata = cad

    return _orig(self, target_token_ids, target_positions, target_hidden_states,
                 next_token_ids, token_indices_to_sample, common_attn_metadata,
                 sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)

eagle_mod.EagleProposer.propose = _fixed

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=200, temperature=0.0)
llm.generate(["Hi"], sp, use_tqdm=False)
import gc, time
for bs in [4, 8, 16, 32]:
    try:
        llm.generate(["Hi"] * bs, sp, use_tqdm=False)
        gc.collect()
        t0 = time.perf_counter()
        out = llm.generate(["Hi"] * bs, sp, use_tqdm=False)
        t1 = time.perf_counter()
        ntok = sum(len(o.outputs[0].token_ids) for o in out)
        tps = ntok / (t1 - t0)
        print(f"batch={bs}: OK ({tps:.0f} tok/s)")
    except:
        print(f"batch={bs}: CRASH")
        break
