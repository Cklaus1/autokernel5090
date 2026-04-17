#!/usr/bin/env python3
"""Sync after EVERY causal_conv1d_update to find the exact crashing call."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _orig
_call_id = [0]

def _synced(x, conv_state, weight, bias=None, activation=None,
            conv_state_indices=None, **kwargs):
    _call_id[0] += 1
    cid = _call_id[0]
    result = _orig(x, conv_state, weight, bias, activation,
                   conv_state_indices=conv_state_indices, **kwargs)
    try:
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[CRASH at call #{cid}] x_shape={tuple(x.shape)}, "
              f"csi_shape={tuple(conv_state_indices.shape) if conv_state_indices is not None else None}, "
              f"conv_state_shape={tuple(conv_state.shape)}, "
              f"qsl={'yes' if kwargs.get('query_start_loc') is not None else 'no'}, "
              f"nat={'yes' if kwargs.get('num_accepted_tokens') is not None else 'no'}")
        raise
    return result

import vllm.model_executor.layers.mamba.short_conv as sc
import vllm.model_executor.models.qwen3_next as qn
sc.causal_conv1d_update = _synced
qn.causal_conv1d_update = _synced

# Also sync after fused_recurrent
_orig_fr = qn.fused_recurrent_gated_delta_rule
def _synced_fr(*args, **kwargs):
    _call_id[0] += 1
    cid = _call_id[0]
    result = _orig_fr(*args, **kwargs)
    try:
        torch.cuda.synchronize()
    except Exception as e:
        ssm_idx = kwargs.get('ssm_state_indices')
        init = kwargs.get('initial_state')
        print(f"[CRASH fused_recurrent #{cid}] "
              f"ssm_indices_shape={tuple(ssm_idx.shape) if ssm_idx is not None else None}, "
              f"init_state_shape={tuple(init.shape) if init is not None else None}")
        raise
    return result
qn.fused_recurrent_gated_delta_rule = _synced_fr

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=30, temperature=0.0)
llm.generate(["Hi"], sp)
_call_id[0] = 0

print("=== batch=4 ===")
try:
    outputs = llm.generate(["Hi"] * 4, sp)
    print(f"OK ({sum(len(o.outputs[0].token_ids) for o in outputs)} tokens, {_call_id[0]} calls)")
except Exception as e:
    print(f"Total calls before crash: {_call_id[0]}")
