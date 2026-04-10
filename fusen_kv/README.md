# fusen-kv

4x KV cache compression for vLLM via custom quantized attention. Stores K and V tensors in 2–8 bit with per-block scales, cutting KV cache VRAM by up to 4x with minimal quality loss. Plugs into vLLM's general plugin system — no vLLM source modifications required.

## Install

```bash
pip install fusen-kv
# or with vLLM as a declared dependency:
pip install "fusen-kv[vllm]"
```

## Quick start

vLLM auto-discovers the plugin via the entry-points mechanism. No `import` needed.

```bash
# Auto-select best compression spec
vllm serve meta-llama/Llama-3-8B-Instruct --kv-cache-dtype fusen

# Specific format: 4-bit K, 4-bit V, block size 64
vllm serve meta-llama/Llama-3-8B-Instruct --kv-cache-dtype k4v4b64

# 8-bit K, 4-bit V — better quality, 3x compression
vllm serve meta-llama/Llama-3-8B-Instruct --kv-cache-dtype k8v4b32
```

## Supported dtype strings

| dtype | K bits | V bits | Block | Compression |
|---|---|---|---|---|
| `fusen` | auto | auto | auto | best for your GPU |
| `k4v4b64` | 4 | 4 | 64 | ~4x |
| `k4v4b32` | 4 | 4 | 32 | ~4x |
| `k4v4b16` | 4 | 4 | 16 | ~4x |
| `k8v4b32` | 8 | 4 | 32 | ~3x |
| `k8v8b32` | 8 | 8 | 32 | ~2x |
| `k4v2b16` | 4 | 2 | 16 | ~6x |

All formats store quantization scales per block in a separate FP16 tensor.

## Explicit registration (Python API)

```python
import fusen_kv  # triggers register() on import

# Or explicitly:
from fusen_kv import register
register()
```

After registration, any `--kv-cache-dtype` string starting with `fusen`, `k4v`, or `k8v` routes to the FusenKV attention backend.

## Compatibility

- vLLM v0.19+ (v1 attention backend)
- PyTorch 2.1+
- Triton 3.0+ (Triton kernel) or CUDA 12+ (optional C++ kernel)
- Compatible with NVFP4 weight quantization (orthogonal systems — weights and KV cache are quantized independently)

## Debug mode

```bash
FUSEN_DEBUG=1 vllm serve ... --kv-cache-dtype k4v4b32
```

Enables bounds-checking assertions inside the attention kernel. Zero overhead when disabled.

## How it works

1. **Plugin registration** — `fusen_kv.plugin:register` patches vLLM's `CacheDType` Literal to accept FusenKV dtype strings, then registers `FusenKVBackend` as the attention backend for those dtypes.
2. **KV layout** — K and V are quantized to N-bit integers with per-block FP16 scales and packed into a single `uint8` cache tensor. Block size is configurable (16/32/64 tokens).
3. **Decode attention** — a Triton kernel dequantizes K/V on-the-fly during the attention computation, avoiding storing a full FP16 cache. An optional pre-compiled CUDA C++ kernel is used if available (`/tmp/build_fusencache/fusencache_decode.so`).
4. **Prefill attention** — standard FlashAttention via `flash_attn_varlen_func`; tokens are quantized when written to the cache.

## License

Apache 2.0
