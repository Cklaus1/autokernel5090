#!/usr/bin/env python3
"""Find crash with minimal-overhead GPU-side bounds checking."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

PAD_SLOT_ID = -1

# Patch causal_conv1d_update to check bounds on GPU
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _orig

def _checked(x, conv_state, weight, bias=None, activation=None,
             conv_state_indices=None, **kwargs):
    if conv_state_indices is not None:
        num_cache_lines = conv_state.size(0)
        flat = conv_state_indices.reshape(-1)
        # GPU-side check without sync
        valid_mask = flat >= 0
        if valid_mask.any():
            max_valid = flat[valid_mask].max()
            # This creates a boolean tensor on GPU, no sync yet
            oob = max_valid >= num_cache_lines
            # Only sync if we suspect OOB
            if oob.item():  # This syncs
                print(f"[OOB] conv_state_indices max={max_valid.item()}, "
                      f"num_cache_lines={num_cache_lines}, "
                      f"shape={conv_state_indices.shape}, x_shape={x.shape}, "
                      f"ql={'yes' if kwargs.get('query_start_loc') is not None else 'no'}")
                import traceback; traceback.print_stack()
                return torch.zeros_like(x) if x.dim() == 2 else x.new_zeros(*x.shape)

    return _orig(x, conv_state, weight, bias, activation,
                 conv_state_indices=conv_state_indices, **kwargs)

import vllm.model_executor.layers.mamba.short_conv as sc
import vllm.model_executor.models.qwen3_next as qn
sc.causal_conv1d_update = _checked
qn.causal_conv1d_update = _checked

# Similarly check fused_recurrent
_orig_fr = qn.fused_recurrent_gated_delta_rule
def _checked_fr(*args, **kwargs):
    ssm_indices = kwargs.get('ssm_state_indices')
    initial_state = kwargs.get('initial_state')
    if initial_state is None and len(args) > 6:
        initial_state = args[6]
    if ssm_indices is not None and initial_state is not None:
        num_states = initial_state.size(0)
        flat = ssm_indices.reshape(-1)
        valid = flat[flat >= 0]
        if valid.numel() > 0:
            max_v = valid.max()
            if (max_v >= num_states).item():
                print(f"[OOB] ssm_state_indices max={max_v.item()}, "
                      f"num_states={num_states}, shape={ssm_indices.shape}")
                import traceback; traceback.print_stack()
    return _orig_fr(*args, **kwargs)
qn.fused_recurrent_gated_delta_rule = _checked_fr

print("[ARMED]")
from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Write merge sort:"], sp)
llm.generate(["Write merge sort:"], sp)
# Run batch=4 many times to trigger the non-deterministic crash
for trial in range(10):
    try:
        outputs = llm.generate(["Write merge sort:"] * 4, sp)
        print(f"trial {trial}: OK")
    except Exception as e:
        print(f"trial {trial}: CRASHED — {str(e)[:60]}")
        break
