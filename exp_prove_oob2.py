#!/usr/bin/env python3
"""Correct OOB check using storage size, not numel."""
import os, torch
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "ERROR"

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _orig
import vllm.model_executor.layers.mamba.short_conv as sc
import vllm.model_executor.models.qwen3_next as qn

_call_count = [0]
_oob_count = [0]

def _check(x, conv_state, weight, bias=None, activation=None,
           conv_state_indices=None, **kwargs):
    _call_count[0] += 1
    if conv_state_indices is not None and kwargs.get('query_start_loc') is None:
        batch = x.size(0)
        stride0 = conv_state_indices.stride(0) if conv_state_indices.dim() > 0 else 1

        # Correct check: max offset vs storage size from the tensor's storage_offset
        max_byte_offset = (batch - 1) * stride0
        # The tensor's storage extends from storage_offset to storage_offset + total_accessible
        storage_offset = conv_state_indices.storage_offset()
        total_storage = conv_state_indices.untyped_storage().size() // conv_state_indices.element_size()
        max_accessible = total_storage - storage_offset
        actual_max_access = max_byte_offset  # element offset from tensor start

        if actual_max_access >= max_accessible:
            _oob_count[0] += 1
            if _oob_count[0] <= 3:
                print(f"\n*** REAL OOB #{_oob_count[0]} (call {_call_count[0]}) ***")
                print(f"  x.shape={tuple(x.shape)} → batch={batch}")
                print(f"  csi.shape={tuple(conv_state_indices.shape)} stride={conv_state_indices.stride()}")
                print(f"  storage_offset={storage_offset}, total_storage={total_storage}")
                print(f"  max_accessible={max_accessible}")
                print(f"  actual_max_access={actual_max_access}")
                print(f"  OVERFLOW: accessing element {actual_max_access} but only {max_accessible} accessible")
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
sp = SamplingParams(max_tokens=50, temperature=0.0)
llm.generate(["Hi"], sp, use_tqdm=False)
_call_count[0] = 0; _oob_count[0] = 0

print("=== batch=4 ===")
llm.generate(["Hi"] * 4, sp, use_tqdm=False)
print(f"  {_call_count[0]} calls, {_oob_count[0]} OOBs")

_call_count[0] = 0; _oob_count[0] = 0
print("\n=== batch=8 ===")
llm.generate(["Hi"] * 8, sp, use_tqdm=False)
print(f"  {_call_count[0]} calls, {_oob_count[0]} OOBs")
print("DONE")
