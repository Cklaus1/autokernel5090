#!/usr/bin/env python3
"""Skip propose() AND no-op Mamba kernels — isolate to FA layers."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

# No-op Mamba kernels (like before)
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

# Skip propose()
import vllm.v1.spec_decode.eagle as eagle_mod
_orig_propose = eagle_mod.EagleProposer.propose
def _skip_propose(self, *args, **kwargs):
    # Set to 1 to take early return
    save = self.num_speculative_tokens
    self.num_speculative_tokens = 1
    result = _orig_propose(self, *args, **kwargs)
    self.num_speculative_tokens = save
    if result.shape[1] < save:
        padding = result[:, -1:].expand(-1, save - result.shape[1])
        result = torch.cat([result, padding], dim=1)
    return result
eagle_mod.EagleProposer.propose = _skip_propose

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
