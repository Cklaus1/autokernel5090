#!/usr/bin/env python3
"""Prove the exact OOB: conv_state_indices stride vs batch mismatch."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _orig
import vllm.model_executor.layers.mamba.short_conv as sc
import vllm.model_executor.models.qwen3_next as qn

def _check(x, conv_state, weight, bias=None, activation=None,
           conv_state_indices=None, **kwargs):
    if conv_state_indices is not None and kwargs.get('query_start_loc') is None:
        # Without query_start_loc, kernel uses: batch = x.size(0)
        # and accesses: conv_state_indices[batch_idx * stride(0)]
        batch = x.size(0)
        stride0 = conv_state_indices.stride(0) if conv_state_indices.dim() > 0 else 1
        numel = conv_state_indices.numel()
        max_offset = (batch - 1) * stride0
        if max_offset >= numel:
            print(f"\n*** OOB DETECTED ***")
            print(f"  x.shape={tuple(x.shape)} → batch={batch}")
            print(f"  conv_state_indices.shape={tuple(conv_state_indices.shape)}")
            print(f"  conv_state_indices.stride={conv_state_indices.stride()}")
            print(f"  stride(0)={stride0}")
            print(f"  max access offset: (batch-1)*stride(0) = {max_offset}")
            print(f"  tensor numel: {numel}")
            print(f"  OVERFLOW BY {max_offset - numel + 1} ELEMENTS")
            print(f"  query_start_loc={kwargs.get('query_start_loc')}")
            print(f"  num_accepted_tokens={kwargs.get('num_accepted_tokens')}")
            # Return zeros to avoid crash
            return torch.zeros_like(x)
    return _orig(x, conv_state, weight, bias, activation,
                 conv_state_indices=conv_state_indices, **kwargs)

sc.causal_conv1d_update = _check
qn.causal_conv1d_update = _check

from vllm import LLM, SamplingParams
llm = LLM(model="Kbenkhaled/Qwen3.5-9B-NVFP4", gpu_memory_utilization=0.90,
    max_num_seqs=128, max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"})
sp = SamplingParams(max_tokens=30, temperature=0.0)
llm.generate(["Hi"], sp, use_tqdm=False)
print("\n=== batch=4 ===")
llm.generate(["Hi"] * 4, sp, use_tqdm=False)
print("\n=== batch=8 ===")
llm.generate(["Hi"] * 8, sp, use_tqdm=False)
print("\nDONE")
