#!/usr/bin/env python3
"""Mamba no-op'd, but propose() runs normally. Where does it crash?"""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

# No-op Mamba kernels
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

# Add sync inside propose to find WHERE it crashes
import vllm.v1.spec_decode.eagle as eagle_mod
_orig_propose = eagle_mod.EagleProposer.propose
_propose_count = [0]

def _traced_propose(self, *args, **kwargs):
    _propose_count[0] += 1
    pc = _propose_count[0]
    try:
        result = _orig_propose(self, *args, **kwargs)
        return result
    except Exception as e:
        print(f"[PROPOSE #{pc} CRASHED: {str(e)[:60]}]")
        raise

eagle_mod.EagleProposer.propose = _traced_propose

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Hi"], sp, use_tqdm=False)
_propose_count[0] = 0
print("=== batch=8 ===")
try:
    llm.generate(["Hi"] * 8, sp, use_tqdm=False)
    print(f"OK after {_propose_count[0]} propose calls")
except:
    print(f"CRASHED after {_propose_count[0]} propose calls")
