#!/usr/bin/env python3
"""Monkey-patch fix: shallow-copy common_attn_metadata in propose() to prevent corruption."""

import os, gc, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"

from copy import copy as shallow_copy
import torch
import vllm.v1.spec_decode.eagle as eagle_mod

_original_propose = eagle_mod.EagleProposer.propose

def _fixed_propose(self, target_token_ids, target_positions, target_hidden_states,
                   next_token_ids, token_indices_to_sample, common_attn_metadata,
                   sampling_metadata, mm_embed_inputs=None,
                   num_rejected_tokens_gpu=None, slot_mappings=None):
    """Wrap propose() to deep-copy common_attn_metadata, preventing
    in-place corruption that causes OOB in subsequent engine steps."""
    if self.num_speculative_tokens > 1:
        # Deep copy the metadata to isolate from main model
        cad_copy = shallow_copy(common_attn_metadata)
        # Clone all mutable tensors
        cad_copy.seq_lens = common_attn_metadata.seq_lens.clone()
        cad_copy.query_start_loc = common_attn_metadata.query_start_loc.clone()
        if common_attn_metadata.slot_mapping is not None:
            cad_copy.slot_mapping = common_attn_metadata.slot_mapping.clone()
        cad_copy._seq_lens_cpu = None
        cad_copy._num_computed_tokens_cpu = None
        common_attn_metadata = cad_copy

    return _original_propose(
        self, target_token_ids, target_positions, target_hidden_states,
        next_token_ids, token_indices_to_sample, common_attn_metadata,
        sampling_metadata, mm_embed_inputs, num_rejected_tokens_gpu, slot_mappings)

eagle_mod.EagleProposer.propose = _fixed_propose
print("[FIX] Patched EagleProposer.propose with metadata isolation")

from vllm import LLM, SamplingParams
llm = LLM(
    model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=200, temperature=0.0)
llm.generate([PROMPT], sp)
llm.generate([PROMPT], sp)

# Decode benchmark
times = []
for _ in range(3):
    gc.collect()
    t0 = time.perf_counter()
    outputs = llm.generate([PROMPT], sp)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    ntok = len(outputs[0].outputs[0].token_ids)
tok_s = ntok / (sum(times) / len(times))
print(f"Decode: {tok_s:.1f} tok/s")

# Batch sweep
for bs in [2, 4, 8, 16, 32]:
    try:
        llm.generate([PROMPT]*bs, sp)
        gc.collect()
        t0 = time.perf_counter()
        outputs = llm.generate([PROMPT]*bs, sp)
        t1 = time.perf_counter()
        ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = ntok / (t1 - t0)
        print(f"Batch {bs:3d}: {tps:7.0f} tok/s total ({tps/bs:6.1f}/user)")
    except Exception as e:
        print(f"Batch {bs:3d}: CRASHED")
        break

print(f"\nDecode speedup vs no-MTP: {tok_s/120.8:.2f}x")
