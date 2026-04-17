#!/usr/bin/env python3
"""Skip ALL causal_conv1d_update calls — return zeros."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

import vllm.model_executor.layers.mamba.short_conv as sc
import vllm.model_executor.models.qwen3_next as qn

def _noop(x, conv_state, weight, bias=None, activation=None,
          conv_state_indices=None, **kwargs):
    return torch.zeros_like(x)

sc.causal_conv1d_update = _noop
qn.causal_conv1d_update = _noop

# Also noop fused_recurrent
_orig_fr = qn.fused_recurrent_gated_delta_rule
def _noop_fr(*args, **kwargs):
    q = kwargs.get('q')
    if q is None: q = args[0]
    v = kwargs.get('v')
    if v is None: v = args[2]
    o = torch.zeros_like(v)
    ist = kwargs.get('initial_state')
    if ist is None and len(args) > 6:
        ist = args[6]
    return o, ist
qn.fused_recurrent_gated_delta_rule = _noop_fr

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
