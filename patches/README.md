# AutoKernel vLLM Patches

Patches applied to vLLM 0.17.0 + FlashInfer 0.6.4 for RTX 5090 (Blackwell SM120) 
with Qwen3.5-9B NVFP4 + DFlash speculative decoding.

## Patch Inventory

### vLLM Patches (7 files)

| Patch | File | Lines Changed | Purpose |
|-------|------|--------------|---------|
| `v1_spec_decode_dflash.py.patch` | `v1/spec_decode/dflash.py` | 290 (new file) | DFlash speculative decoding proposer for Qwen3.5 hybrid architecture |
| `v1_spec_decode_eagle.py.patch` | `v1/spec_decode/eagle.py` | 104 | DFlash integration into Eagle proposer, aux_hidden_state_layers, metadata cloning |
| `v1_worker_gpu_model_runner.py.patch` | `v1/worker/gpu_model_runner.py` | 36 | slot_mapping fix, M-RoPE fix, DFlash model loading support |
| `model_executor_models_qwen3_next.py.patch` | `model_executor/models/qwen3_next.py` | 54 | SupportsEagle3 mixin, aux layer methods, collection in forward() |
| `model_executor_models_qwen3_5.py.patch` | `model_executor/models/qwen3_5.py` | 10 | Qwen3.5 GDN model Eagle3 support |
| `model_executor_models_registry.py.patch` | `model_executor/models/registry.py` | 2 | Register DFlash proposer |
| `transformers_utils_configs_eagle.py.patch` | `transformers_utils/configs/eagle.py` | 10 | Eagle config for DFlash |

### FlashInfer Patches (1 file)

| Patch | File | Lines Changed | Purpose |
|-------|------|--------------|---------|
| `flashinfer_jit_cpp_ext.py.patch` | `jit/cpp_ext.py` | 12 | Force CUDA 12.9 for SM120 FP4, gcc-12 compatibility |

### Already Upstream (no patch needed)

| Fix | Status |
|-----|--------|
| MTP Batch Fix (flash_attn.py MultipleOf(16)) | Already in vLLM 0.17.0 |

## How to Apply After Upgrade

```bash
# After pip install vllm==NEW_VERSION
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
cd $VLLM_PATH

# Apply each patch (will warn on conflicts)
for p in /root/projects/autokernel/patches/*.patch; do
  echo "Applying: $p"
  patch -p0 --dry-run < "$p" && patch -p0 < "$p" || echo "CONFLICT: manual resolution needed for $p"
done
```

## Verification After Patching

```bash
# 1. Basic import
python -c "from vllm.v1.spec_decode.dflash import DFlashProposer; print('DFlash OK')"

# 2. MTP batch stability (the critical test)
python -c "
from vllm import LLM, SamplingParams
llm = LLM('Qwen/Qwen3.5-9B-NVFP4', enforce_eager=True, num_speculative_tokens=3)
out = llm.generate(['Hello world'] * 8, SamplingParams(max_tokens=50))
print(f'Batch=8 OK: {sum(len(o.outputs[0].token_ids) for o in out)} total tokens')
"

# 3. DFlash decode speed (should be ~170 tok/s single, ~4900 tok/s batch)
python /root/projects/autokernel/bench.py --quick
```

## Created

Generated from installed vLLM 0.17.0 on $(date +%Y-%m-%d).
Base: vLLM 0.17.0 + FlashInfer 0.6.4
Target: RTX 5090 (SM120), CUDA 12.8/12.9, Qwen3.5-9B-NVFP4
