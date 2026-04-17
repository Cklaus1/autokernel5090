#!/usr/bin/env python3
"""Check if propose()'s slot_mapping values go OOB in the KV cache."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

# Patch the FlashAttention forward to check slot_mapping bounds
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
_orig_kv_update = FlashAttentionImpl.do_kv_cache_update
_check_count = [0]

def _checked_kv_update(self, layer, key, value, kv_cache, slot_mapping):
    _check_count[0] += 1
    num_blocks = kv_cache.size(0)
    block_size = kv_cache.size(2) if kv_cache.dim() == 5 else kv_cache.size(1)
    max_valid_slot = num_blocks * block_size - 1

    # Check slot_mapping bounds (ignore padding=-1)
    valid_slots = slot_mapping[slot_mapping >= 0]
    if valid_slots.numel() > 0:
        max_slot = valid_slots.max().item()
        if max_slot > max_valid_slot:
            print(f"\n*** KV CACHE OOB #{_check_count[0]} ***")
            print(f"  max_slot={max_slot}, max_valid={max_valid_slot}")
            print(f"  kv_cache.shape={tuple(kv_cache.shape)}")
            print(f"  slot_mapping.shape={tuple(slot_mapping.shape)}")
            print(f"  num_valid_slots={valid_slots.numel()}")
            return  # Skip the write

    return _orig_kv_update(self, layer, key, value, kv_cache, slot_mapping)

FlashAttentionImpl.do_kv_cache_update = _checked_kv_update

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Hi"], sp, use_tqdm=False)
_check_count[0] = 0
print("batch=8: ", end="")
try:
    llm.generate(["Hi"] * 8, sp, use_tqdm=False)
    print(f"OK ({_check_count[0]} KV updates)")
except:
    print(f"CRASH ({_check_count[0]} KV updates)")
