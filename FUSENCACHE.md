# FusenCache — Rank-Preserving KV Cache Compression for vLLM

## What It Is

FusenCache is a vLLM attention backend that compresses the KV cache during LLM inference. It replaces TurboQuant (which has [3 critical bugs](https://github.com/vllm-project/vllm/pull/38479#issuecomment-4184609256)) with a simpler, correct design.

**Core principle:** Keys and values should not be treated the same. Keys control attention routing (fragile), values get weighted-summed (forgiving).

## Current Status

### v0 — FP8 K + FP8 V (DONE)

- **Compression:** 2.0x vs FP16
- **KV tokens:** 17,600 (2x baseline 8,800)
- **Throughput:** 15.8 tok/s (baseline 14.2)
- **Quality:** No visible degradation
- **Usage:** `--kv-cache-dtype fusen`

Tested on Gemma 4 31B AWQ (RTX 5090 32GB). All test prompts produce coherent, correct output.

### v1 — FP8 K + int4 V (DONE)

- **Compression:** 2.67x vs FP16
- **KV tokens:** 23,472 (2.67x baseline 8,800)
- **Throughput:** 3.5 tok/s (Python decode — Triton kernel would be faster)
- **Quality:** No visible degradation
- **Key features:**
  - FP8 E4M3 keys (1 byte per element)
  - int4 symmetric values with per-head scale (0.5 bytes per element)
  - V scales stored in separate lazy layer buffer (avoids padding waste)
  - Automatic v0 fallback for layers with incompatible head_dim

### v2 — Attention-Aware Tiering (PLANNED)

- Hot/cold token tiers based on recency
- TWA (Token-Wise Attention) promotion/demotion based on cumulative attention mass
- Pinned tokens: system prompt, BOS, high-attention anchors
- Hot tier: higher precision, Cold tier: more aggressive compression

### v3 — Hybrid Selective Mode (DONE)

- **Enabled via:** `FUSEN_SELECTIVE=1`
- Dense attention for hot window (last 512 tokens)
- Chunk landmarks: mean of FP8 keys per 32-token chunk
- Top-16 chunk selection via Q @ landmarks scoring
- Only fetch K/V for selected chunks + hot window
- At 8K context: attend to ~1024 tokens instead of 8192 (8x less work)
- Configurable: `FUSEN_HOT_WINDOW`, `FUSEN_CHUNK_SIZE`, `FUSEN_TOP_M`

## Architecture

### Files (installed in vLLM)

```
vllm/
  fusencache/
    __init__.py
    config.py              # FusenCacheConfig — slot layout, effective_head_size
  v1/attention/backends/
    fusencache_attn.py     # Main backend: Backend, Metadata, Builder, Impl
```

### Modified vLLM Files

```
vllm/config/cache.py                              # "fusen" in CacheDType
vllm/v1/attention/backends/registry.py             # FUSENCACHE enum
vllm/platforms/cuda.py                             # routing for "fusen" dtype
vllm/model_executor/layers/attention/attention.py  # get_kv_cache_spec for "fusen"
vllm/utils/torch_utils.py                          # "fusen" -> torch.uint8
```

### Cache Layout (v0)

Combined K+V cache, dtype=uint8:
```
Shape: (num_blocks, block_size, num_kv_heads, 2 * head_dim)

Per token per head:
  [k_fp8 (D bytes) | v_fp8 (D bytes)]
```

No separate K/V tensors. No leading dimension of 2. Each slot is exactly `2D` bytes.

### How It Works

**Prefill:** Standard flash_attn or SDPA on uncompressed Q/K/V. After attention, K/V are cast to FP8 and stored in the compressed cache.

**Decode:** Gather compressed cache slots for the sequence, view bytes as FP8, cast to float32, run standard Q@K^T attention with softmax, weighted sum of values.

**Store path:**
```python
k_fp8 = key.to(torch.float8_e4m3fn)
v_fp8 = value.to(torch.float8_e4m3fn)
packed = torch.cat([k_fp8.view(uint8), v_fp8.view(uint8)], dim=-1)
kv_cache[blk_idx, blk_off] = packed
```

**Decode path:**
```python
slots = kv_cache[blk_idx, blk_off]       # (S, Hk, 2D) uint8
k = slots[:, :, :D].view(float8).float()  # dequant K
v = slots[:, :, D:].view(float8).float()  # dequant V
scores = einsum('hd,shd->hs', q, k) * scale
attn_w = softmax(scores, dim=-1)
out = einsum('hs,shd->hd', attn_w, v)
```

## Why Not TurboQuant

We found and reported 3 critical bugs in TurboQuant (vllm-project/vllm#38479):

1. **Triton kernels hardcoded for 2-bit MSE** — silently corrupts tq4 (3-bit) output
2. **Store/load head_dim mismatch** — all tq3 configs produce garbage on all models
3. **Fundamental quality issue** — 33-75% K reconstruction error, 56% attention top-1 agreement

FusenCache avoids all three by design:
- No bitpacking (FP8 is byte-aligned)
- No effective_head_size != real_head_dim (they're equal)
- No random rotations or centroid codebooks (just dtype cast)

## Design Principles

1. **Correctness first.** Simple quantization that is easy to verify. `dequant(quant(x)) ≈ x` with standard FP8 behavior.
2. **Asymmetric K/V.** Keys get more bits than values because attention ranking is fragile.
3. **No magic.** No rotations, centroids, QJL projections, or residual estimators. Just block quantization.
4. **Debuggable failures.** When something goes wrong, you can isolate K error, V error, and attention pattern divergence independently.
5. **Evaluate on ranking, not MSE.** Attention top-k overlap and KL divergence matter more than vector reconstruction error.

## Comparison

| Method | Compression | Quality | Complexity | Status |
|--------|------------|---------|------------|--------|
| FP16 KV (baseline) | 1.0x | Perfect | None | Built-in |
| FP8 KV (vLLM built-in) | 2.0x | Excellent | Low | Built-in |
| FusenCache v0 | 2.0x | Excellent | Low | **Working** |
| TurboQuant tq3 | ~3x claimed | Broken | Very high | Buggy |
| TurboQuant tq4 | ~4x claimed | Broken | Very high | Buggy |
| FusenCache v1 | 2.67x | Excellent | Medium | **Working** |

## Requirements

- vLLM 0.19.0+
- NVIDIA GPU with FP8 support (SM89+: Ada, Hopper, Blackwell)
- Any transformer model supported by vLLM

## Usage

```bash
# Serve with FusenCache
vllm serve your-model --kv-cache-dtype fusen --enforce-eager

# Python API
from vllm import LLM
llm = LLM(model="your-model", kv_cache_dtype="fusen", enforce_eager=True)
```

## Background

FusenCache was developed while testing Gemma 4 31B on an RTX 5090 (32GB). The model's 19.6GB weights left only 8GB for KV cache — enough for ~8,800 tokens with FP16. FusenCache doubles this to 17,600 tokens with no quality loss, making 8K+ context practical on a single consumer GPU.

The project started by attempting to use TurboQuant (Google's ICLR 2026 paper), which led to discovering 3 critical implementation bugs and designing a replacement from scratch. The name "FusenCache" reflects the fusion of rank-preserving compression with attention-aware caching.
