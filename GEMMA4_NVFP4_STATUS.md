# Gemma 4 26B-A4B NVFP4 on vLLM — Status Notes

**Date:** 2026-04-09
**Hardware:** RTX 5090 (Blackwell SM120), 32GB VRAM, 48GB system RAM
**Goal:** Run Gemma 4 26B-A4B-it with NVFP4 quantization on vLLM for fast inference

---

## Current State

### Checkpoint: READY
- **Location:** `/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/` (17 GB)
- **Format:** modelopt (matching NVIDIA's Gemma-4-31B-IT-NVFP4 pattern)
- **Attention:** Dequantized to BF16 (115 projections across 30 layers)
- **MoE/MLP:** NVFP4 (weight=uint8 packed, weight_scale=FP8, weight_scale_2=FP32, input_scale=FP32)
- **Config:** `quant_method: modelopt`, `quant_algo: NVFP4`, `group_size: 16`
- **Created by:** Converting RedHat's compressed-tensors checkpoint → modelopt format via `convert_ct_to_modelopt.py`

### vLLM Docker Image: READY
- **Image:** `vllm-built:latest` (28.6 GB)
- **Version:** vLLM 0.19.1rc1.dev150
- **Patches applied:**
  - PR #38891 — Per-layer attention backend (Gemma4 heterogeneous head dims: 256 sliding / 512 global)
  - PR #39084 — Fix NVFP4 expert scale suffix mapping
  - PR #39406 — Robust quantized MoE expert weight loading

### WORKING (Apr 9, 2026)
- Model loads in 12s, 17.24 GiB VRAM
- Inference produces coherent, high-quality output
- ~18 tok/s generation (single request, enforce_eager, no CUDA graphs)
- 10 GiB KV cache, ~43K token capacity, ~15x concurrency at 4096 ctx
- Backends: FlashAttention v2 (sliding), Triton (global), FlashInfer+Cutlass (NVFP4 linear), VLLM_CUTLASS (NVFP4 MoE)

### NOT YET TESTED
- CUDA graphs (remove --enforce-eager)
- FP8 KV cache
- Batch throughput benchmarking
- Perplexity / quality eval vs BF16 original

---

## Why RedHat's Model Doesn't Work

**RedHat's `gemma-4-26B-A4B-it-NVFP4`** (compressed-tensors format) has two problems:

1. **Attention is quantized to NVFP4** — RedHat's ignore list only excludes `router.proj`, `vision_tower`, `embed_vision`, and `lm_head`. All 30 language model `self_attn` layers (q/k/v/o_proj) are NVFP4-quantized. NVIDIA's reference 31B properly keeps attention in BF16.

2. **Compressed-tensors format has broken MoE scale handling on SM120** — The CT path in vLLM doesn't correctly handle per-expert NVFP4 scales for the Cutlass MoE kernel on Blackwell.

### Why attention must stay BF16
- vLLM fuses QKV projections and takes `max()` of their global scales
- When scales differ significantly (which they do — q_proj vs k_proj vs v_proj have different distributions), the smaller scales underflow after fusion
- NVIDIA avoids this entirely by keeping attention in BF16
- The attention layers are small relative to MoE (128 experts per layer) — BF16 attention adds only ~1.6 GB

---

## What We Built

### Conversion Script: `convert_ct_to_modelopt.py`
Converts RedHat's compressed-tensors checkpoint to modelopt format:
- Renames: `weight_packed` → `weight`, `weight_global_scale` → `weight_scale_2` (inverted), `input_global_scale` → `input_scale` (inverted)
- Dequantizes all `self_attn` q/k/v/o projections back to BF16 using FP4-E2M1 lookup table
- Updates `config.json` with modelopt `quant_method` and proper `exclude_modules` list
- Lightweight: runs on CPU, ~16 GB RAM, no GPU needed

### MoE Per-Expert Loop Patch: `patches/apply_moe_fix.py`
Workaround for SM120 grouped GEMM bug in Cutlass MoE:
- Adds `_run_cutlass_moe_fp4_loop()` that processes experts individually
- Enabled via `VLLM_NVFP4_MOE_LOOP=1` environment variable
- Slower than grouped GEMM but avoids the crash

### Other Scripts (attempted, not needed now)
- `quantize_gemma4_modelopt.py` — Direct quantization from BF16 original (OOM'd at 42GB RAM)
- `fix_nvfp4_scales.py` — Scale rescaling for vLLM QKV fusion (unnecessary if attention stays BF16)
- `fix_nvfp4_attn_to_bf16.py` — Earlier version of attention dequantization

---

## Models on Disk

| Directory | Size | Format | Status |
|-----------|------|--------|--------|
| `gemma-4-26B-A4B-it-original` | 51.6 GB | BF16 unquantized | Complete |
| `gemma-4-26B-A4B-it-NVFP4-redhat` | 16.4 GB | compressed-tensors | Complete (broken attn) |
| `gemma-4-26B-A4B-it-NVFP4-modelopt` | 17.0 GB | modelopt | Complete (our version) |
| `gemma-4-26B-A4B-it-NVFP4-ours` | incomplete | modelopt | OOM'd during calibration |
| `gemma-4-26B-A4B-it-AWQ-4bit` | ~16 GB | AWQ | Older attempt |

---

## Next Steps

### Immediate: Test in Docker
```bash
# Launch vLLM with our modelopt checkpoint
docker run -d --name vllm-gemma4 --gpus all \
  --memory=36g \
  -v /root/models:/models:ro \
  -p 8000:8000 \
  vllm-built \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --max-model-len 4096

# If grouped GEMM crashes, add: -e VLLM_NVFP4_MOE_LOOP=1

# Test inference
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-26B-A4B-it-NVFP4-modelopt","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### If It Works
1. Remove `--enforce-eager` and test CUDA graphs
2. Benchmark tok/s at various batch sizes
3. Compare quality vs BF16 original (perplexity eval)
4. Try FP8 KV cache (`--kv-cache-dtype fp8`)

### If Weight Loading Fails
- Check if vLLM's modelopt loader expects `weight` or `weight_packed` for NVFP4
- May need to adjust tensor naming in the checkpoint
- The `model.safetensors.index.json` from RedHat is stale — may need regeneration

### If MoE Crashes
- Enable per-expert loop: `VLLM_NVFP4_MOE_LOOP=1`
- Apply `patches/apply_moe_fix.py` inside the Docker container

---

## Key Insight: NVIDIA's Pattern

NVIDIA's Gemma-4-31B-IT-NVFP4 uses `NVFP4_MLP_ONLY_CFG` from modelopt:
- **Quantized:** All MLP/MoE expert layers (gate_proj, up_proj, down_proj) → NVFP4
- **BF16:** All self_attn (q/k/v/o_proj), router.proj, lm_head, vision_tower, embed_vision
- **Format:** modelopt with `weight`, `weight_scale`, `weight_scale_2`, `input_scale` naming
- **Why:** Attention quantization causes quality degradation and triggers vLLM's scale fusion bug

The 26B-A4B has 128 MoE experts per layer × 30 layers = 3,840 expert modules. These dominate the parameter count and benefit most from quantization. Attention is relatively small.
