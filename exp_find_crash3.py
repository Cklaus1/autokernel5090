#!/usr/bin/env python3
"""Log exact shapes passed to causal_conv1d_update."""
import os, torch, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _orig

_log = []

def _logging_update(x, conv_state, weight, bias=None, activation=None,
                    conv_state_indices=None, **kwargs):
    entry = {
        'x_shape': tuple(x.shape),
        'conv_state_shape': tuple(conv_state.shape),
        'csi_shape': tuple(conv_state_indices.shape) if conv_state_indices is not None else None,
        'has_qsl': kwargs.get('query_start_loc') is not None,
        'has_nat': kwargs.get('num_accepted_tokens') is not None,
        'max_ql': kwargs.get('max_query_len', -1),
    }
    _log.append(entry)
    return _orig(x, conv_state, weight, bias, activation,
                 conv_state_indices=conv_state_indices, **kwargs)

import vllm.model_executor.layers.mamba.short_conv as sc
import vllm.model_executor.models.qwen3_next as qn
sc.causal_conv1d_update = _logging_update
qn.causal_conv1d_update = _logging_update

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=20, temperature=0.0)
llm.generate(["Hi"], sp)  # warmup
_log.clear()

print("=== Single request (decode) ===")
llm.generate(["Hi"], sp)
for i, e in enumerate(_log):
    if e['x_shape'][0] > 0:
        print(f"  call {i}: x={e['x_shape']}, conv_state={e['conv_state_shape']}, "
              f"csi={e['csi_shape']}, qsl={e['has_qsl']}, nat={e['has_nat']}")
        break  # Just show first non-zero

_log.clear()
print("\n=== Batch=4 (first generate) ===")
try:
    outputs = llm.generate(["Hi"] * 4, sp)
    # Show unique shapes
    seen = set()
    for i, e in enumerate(_log):
        key = (e['x_shape'], e['csi_shape'], e['has_qsl'], e['has_nat'])
        if key not in seen and e['x_shape'][0] > 0:
            seen.add(key)
            print(f"  call {i}: x={e['x_shape']}, csi={e['csi_shape']}, "
                  f"qsl={e['has_qsl']}, nat={e['has_nat']}, max_ql={e['max_ql']}")
    print(f"  Total calls: {len(_log)}")
    print("  OK")
except Exception as e:
    # Show the last few calls before crash
    print(f"  CRASHED after {len(_log)} calls")
    seen = set()
    for i, entry in enumerate(_log[-20:]):
        key = (entry['x_shape'], entry['csi_shape'], entry['has_qsl'])
        if key not in seen:
            seen.add(key)
            idx = len(_log) - 20 + i
            print(f"  call {idx}: x={entry['x_shape']}, csi={entry['csi_shape']}, "
                  f"qsl={entry['has_qsl']}, nat={entry['has_nat']}, max_ql={entry['max_ql']}")
