# PR: Support NVFP4 per-expert weight loading for Gemma 4 MoE

## Summary

Gemma 4 26B-A4B NVFP4-quantized models (e.g. `RedHatAI/gemma-4-26B-A4B-it-NVFP4`) fail to load with:

```
KeyError: 'layers.0.moe.experts.0.down_proj.input_global_scale'
```

The model checkpoint stores per-expert NVFP4 quantization parameters (`weight_packed`, `weight_scale`, `weight_global_scale`, `input_global_scale`) but Gemma 4's `_weight_iterator()` and `expert_params_mapping` only handle base weight tensors — not NVFP4 scale parameters.

## Root Cause

Two issues in `vllm/model_executor/models/gemma4.py`:

**1. `_weight_iterator()` doesn't remap per-expert NVFP4 names**

The checkpoint has per-expert names like:
```
model.language_model.layers.0.experts.0.gate_proj.weight_packed
model.language_model.layers.0.experts.0.gate_proj.weight_scale
model.language_model.layers.0.experts.0.gate_proj.input_global_scale
```

The existing code (line 1169-1178) only handles fused names (`experts.gate_up_proj`, `experts.down_proj`). Per-expert names with numeric IDs (`experts.0.gate_proj`) pass through without the `experts.` → `moe.experts.` remapping.

**2. `expert_params_mapping` missing NVFP4 suffixes**

The mapping (line 937-953) only maps base weight names:
```python
("experts.w2_weight", f"experts.{expert_id}.down_proj", expert_id, shard_id)
```

NVFP4 parameters like `experts.0.down_proj.weight_packed`, `experts.0.down_proj.input_global_scale` have no mapping entry, so `AutoWeightsLoader` can't find the corresponding model parameter.

## Fix

**1. Add per-expert name remapping to `_weight_iterator()`:**

```python
# Per-expert NVFP4 weights: checkpoint has
# experts.{id}.{proj}.{param} — remap to
# moe.experts.{id}.{proj}.{param} for FusedMoE.
expert_match = re.match(
    r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(.*)",
    name,
)
if expert_match:
    prefix, eid, proj, suffix = expert_match.groups()
    name = f"{prefix}.moe.experts.{eid}.{proj}.{suffix}"
    yield name, weight
    continue
```

**2. Add NVFP4 suffix mappings to `expert_params_mapping`:**

```python
nvfp4_suffixes = [
    "weight_packed", "weight_scale",
    "weight_global_scale", "input_global_scale",
]
for expert_id in range(num_experts):
    for shard_id, proj_name in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
        for suffix in nvfp4_suffixes:
            param_base = ("experts.w13_" if proj_name in ["gate_proj", "up_proj"]
                          else "experts.w2_")
            expert_params_mapping.append((
                param_base + suffix,
                f"experts.{expert_id}.{proj_name}.{suffix}",
                expert_id, shard_id,
            ))
```

**3. Update weight_loader call to handle non-2D scale tensors:**

```python
# Determine weight_name suffix for FusedMoE loader
wn_for_loader = weight_name + ".weight"
for _sfx in nvfp4_suffixes:
    if weight_name.endswith("." + _sfx):
        wn_for_loader = weight_name.split(".", 1)[-1]
        break
else:
    assert loaded_weight.dim() == 2, ...
```

## Impact

Tested with `RedHatAI/gemma-4-26B-A4B-it-NVFP4`:

```
Model loading took 15.76 GiB memory
GPU KV cache size: 54,256 tokens
Output: "The capital of France is **Paris**."  ✓
```

## Files Changed

- `vllm/model_executor/models/gemma4.py`: ~40 lines added to `load_weights()` and `_weight_iterator()`

## Test

```python
from vllm import LLM, SamplingParams

# Previously raised KeyError: 'layers.0.moe.experts.0.down_proj.input_global_scale'
llm = LLM(
    model="RedHatAI/gemma-4-26B-A4B-it-NVFP4",
    max_model_len=8192,
    trust_remote_code=True,
    enforce_eager=True,
)
out = llm.generate(["What is the capital of France?"], SamplingParams(max_tokens=32))
print(out[0].outputs[0].text)  # "The capital of France is **Paris**."
```

## Notes

- This fix only addresses the weight loading path. The NVFP4 quantization kernels themselves are already supported in vLLM via `CompressedTensorsW4A4Nvfp4MoEMethod`.
- The fix is backwards-compatible — AWQ and other fused-format checkpoints continue to work via the existing code path.
- The per-expert unfused format is how `nvidia-modelopt` exports NVFP4 quantized MoE models. Multiple community quants (RedHatAI, bg-digitalservices) use this format.
