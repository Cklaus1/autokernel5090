# DFlash on vLLM 0.17.0 + Qwen3.5-9B NVFP4

## Working configuration
- vLLM 0.17.0, torch 2.10.0+cu128, FlashInfer 0.6.4
- Target: Kbenkhaled/Qwen3.5-9B-NVFP4 (compressed-tensors)
- Draft: z-lab/Qwen3.5-9B-DFlash (5 layers, block_size=16)
- RTX 5090, nvcc 12.9, gcc-12

## Result: 94.6 tok/s with CUDA graphs, correct output

## Patches applied

### FlashInfer patches (3 files)
1. `flashinfer/jit/cpp_ext.py:get_cuda_path()` — Force CUDA 12.9 for SM120 FP4 support
2. `flashinfer/jit/cpp_ext.py:build_cuda_cflags()` — Use gcc-12 for -ccbin (nvcc 12.9 rejects gcc-13)
3. `flashinfer/jit/cpp_ext.py` — Use g++-12 as default CXX compiler

### vLLM patches (8 files)
1. `model_executor/models/registry.py` — Add DFlashDraftModel to _SPECULATIVE_DECODING_MODELS
2. `config/speculative.py` — Add dflash method, use_dflash(), parallel_drafting, aux_hidden_states
3. `model_executor/models/interfaces.py` — Add _get_default_eagle3_aux_hidden_state_layers() reading dflash_config
4. `model_executor/models/qwen3.py` — Store vllm_config, delegate to default aux layers helper
5. `transformers_utils/configs/eagle.py` — Handle DFlash architecture naming (avoid double-prefix)
6. `v1/spec_decode/eagle.py` — Refactor into overridable methods for DFlash
7. `v1/worker/gpu_model_runner.py` — Add DFlashProposer, fix tuple unpacking, read dflash_config target_layer_ids
8. `v1/spec_decode/dflash.py` — New file: DFlashProposer with M-RoPE fix, slot_mapping fix, load_model override

### Qwen3.5 model patches (2 files)
9. `model_executor/models/qwen3_next.py` — Add aux_hidden_state_layers + collection in forward()
10. `model_executor/models/qwen3_5.py` — Add SupportsEagle3 mixin, aux layer methods, aux_hidden_state_layers init
