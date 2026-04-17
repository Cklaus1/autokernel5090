#!/usr/bin/env python3
"""Find the EXACT layer and operation that crashes."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

# Monkey-patch causal_conv1d_update to add validation
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _orig_conv1d_update

_call_count = [0]

def _checked_conv1d_update(x, conv_state, weight, bias=None, activation=None,
                           conv_state_indices=None, **kwargs):
    _call_count[0] += 1
    call_id = _call_count[0]

    num_cache_lines = conv_state.size(0)

    # Check conv_state_indices bounds
    if conv_state_indices is not None:
        flat_idx = conv_state_indices.reshape(-1)
        max_idx = flat_idx.max().item()
        min_idx = flat_idx.min().item()
        # PAD_SLOT_ID is typically -1 or a large negative
        valid = flat_idx[flat_idx >= 0]
        if valid.numel() > 0:
            max_valid = valid.max().item()
            if max_valid >= num_cache_lines:
                print(f"[CRASH #{call_id}] OOB conv_state_indices! "
                      f"max_valid_idx={max_valid} >= num_cache_lines={num_cache_lines}, "
                      f"indices_shape={conv_state_indices.shape}, "
                      f"x_shape={x.shape}, "
                      f"query_start_loc={'yes' if kwargs.get('query_start_loc') is not None else 'no'}")
                # Don't call the kernel - it would crash
                return torch.zeros_like(x) if x.dim() == 2 else torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)

    try:
        result = _orig_conv1d_update(x, conv_state, weight, bias, activation,
                                     conv_state_indices=conv_state_indices, **kwargs)
        return result
    except Exception as e:
        print(f"[CRASH #{call_id}] causal_conv1d_update failed: {e}")
        raise

import vllm.model_executor.layers.mamba.ops.causal_conv1d as conv_mod
conv_mod.causal_conv1d_update = _checked_conv1d_update

# Also patch the imports in short_conv and qwen3_next
import vllm.model_executor.layers.mamba.short_conv as sc_mod
sc_mod.causal_conv1d_update = _checked_conv1d_update
import vllm.model_executor.models.qwen3_next as qn_mod
qn_mod.causal_conv1d_update = _checked_conv1d_update

# Also check fused_recurrent_gated_delta_rule
_orig_fused_recurrent = qn_mod.fused_recurrent_gated_delta_rule

def _checked_fused_recurrent(*args, **kwargs):
    ssm_state_indices = kwargs.get('ssm_state_indices')
    initial_state = kwargs.get('initial_state')
    if initial_state is None and len(args) > 6:
        initial_state = args[6]

    if ssm_state_indices is not None and initial_state is not None:
        flat_idx = ssm_state_indices.reshape(-1)
        valid = flat_idx[flat_idx >= 0]
        if valid.numel() > 0:
            max_valid = valid.max().item()
            num_states = initial_state.size(0)
            if max_valid >= num_states:
                print(f"[CRASH fused_recurrent] OOB ssm_state_indices! "
                      f"max_valid_idx={max_valid} >= num_states={num_states}, "
                      f"indices_shape={ssm_state_indices.shape}")

    return _orig_fused_recurrent(*args, **kwargs)

qn_mod.fused_recurrent_gated_delta_rule = _checked_fused_recurrent

print("[INSTRUMENTED] All conv1d_update and fused_recurrent calls checked for OOB")

from vllm import LLM, SamplingParams
llm = LLM(
    model="Kbenkhaled/Qwen3.5-9B-NVFP4",
    gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Write merge sort:"], sp)
llm.generate(["Write merge sort:"], sp)

print("\n=== Testing batch=4 ===")
try:
    outputs = llm.generate(["Write merge sort:"] * 4, sp)
    print(f"batch=4: OK ({sum(len(o.outputs[0].token_ids) for o in outputs)} tokens)")
except Exception as e:
    print(f"batch=4: CRASHED — {str(e)[:60]}")

print(f"\nTotal conv1d_update calls: {_call_count[0]}")
