#!/usr/bin/env python3
"""Test: sync ONLY after fused_recurrent (not after conv1d)."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _orig_conv
import vllm.model_executor.models.qwen3_next as qn
_orig_fr = qn.fused_recurrent_gated_delta_rule

# Only sync after fused_recurrent
def _synced_fr(*args, **kwargs):
    result = _orig_fr(*args, **kwargs)
    torch.cuda.synchronize()
    return result
qn.fused_recurrent_gated_delta_rule = _synced_fr

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Hi"], sp)

for trial in range(5):
    try:
        outputs = llm.generate(["Hi"] * 4, sp)
        print(f"trial {trial}: OK")
    except:
        print(f"trial {trial}: CRASH")
        break
