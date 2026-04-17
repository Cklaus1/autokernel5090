# TurboQuant Setup Plan — RTX PRO 6000 (T1-E)

Source: `plans/rtx_pro6000_experiments.md` lines 268-274,
repo: `/home/cklaus/projects/aigpu/turboquant-gpu/`.

---

## What TurboQuant Does

TurboQuant achieves **5.02x KV cache compression** through two mathematically
motivated steps applied per attention head, per token:

1. **Random orthogonal rotation.** A fixed random rotation matrix Pi (shape
   `[head_dim, head_dim]`) is applied to the normalized key/value vector. After
   rotation, each coordinate is approximately Gaussian regardless of the original
   distribution. This makes Lloyd-Max quantization near-optimal.

2. **Lloyd-Max quantization.** Centroids and boundaries are pre-fitted to the
   Gaussian distribution offline. Keys use 2-bit MSE quantization (4 centroids)
   plus a separate 1-bit QJL sign-correction on the residual (projected via a
   second random matrix S). Values use 3-bit MSE (8 centroids), no QJL. The
   fused encoding stores: `K_Indices` (uint8), `K_Signs` (int8), `K_Norms`
   (fp16), `K_RNorms` (fp16), `V_Indices` (uint8), `V_Norms` (fp16).

**Compression ratio vs FP16 baseline:**
- 2-bit K + 1-bit QJL: ~3 bits effective per key element
- 3-bit V: 3 bits per value element
- vs FP16 (16 bits): 5.02x aggregate compression
- Cosine similarity of reconstructed vectors: 0.98 (K), ≥0.98 (V) at head_dim=128

**Five cuTile kernels** (`compress.py:14-283`):
| Kernel | Purpose |
|--------|---------|
| `turboquant_compress_kv_3bit` | Fused K+V in one launch (2-bit K + 1-bit QJL, 3-bit V) |
| `turboquant_compress_2bit` | Key-only, 4 centroids + QJL |
| `turboquant_compress_3bit` | Key-only, 8 centroids + QJL |
| `turboquant_compress_values_3bit` | Value-only, 8 centroids |
| `turboquant_compress_values_2bit` | Value-only, 4 centroids |

**Host engine** (`host.py:27-229`): `TurboQuantEngine` wraps rotation matrices,
Lloyd-Max codebooks, and `auto_tune()` which benchmarks cuTile vs PyTorch
fallback and 2-bit vs 3-bit, selecting the fastest config that clears a
cosine-similarity threshold (default 0.85).

---

## Setup on RTX PRO 6000 (SM120 / Blackwell)

The RTX PRO 6000 is SM120. cuTile availability depends on the `tileiras` wheel
for SM120; if unavailable the PyTorch fallback is always functional.

```bash
# 1. Core install
pip install turboquant-gpu

# 2. Attempt cuTile acceleration (CUDA 13.0+ driver required for SM120)
pip install cuda-tile[tileiras] --extra-index-url https://pypi.nvidia.com

# 3. Verify
python - <<'EOF'
from turboquant_gpu import TurboQuantEngine
import torch
engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cuda")
engine.auto_tune(seq_len=512)
EOF
```

If `auto_tune` prints `cutile kernels: available` cuTile is active. If not,
PyTorch fallback will be used — still correct, slower by ~1.5-2x on compression
overhead (which is not on the critical decode path).

---

## Wiring into vLLM as Custom Attention Backend

Target model: **Qwen3-30B-MoE** (homogeneous head_dim=128, no sliding window,
no FP8 attention penalty — ideal for this technique).

### Step 1: Implement `TurboQuantAttentionBackend`

Create `workspace/turboquant_attn_backend.py`:

```python
# workspace/turboquant_attn_backend.py
import torch
from turboquant_gpu import TurboQuantEngine
from vllm.attention.backends.abstract import (
    AttentionBackend, AttentionImpl, AttentionMetadata
)

_ENGINE: dict[int, TurboQuantEngine] = {}  # head_dim -> engine

def _get_engine(head_dim: int, device: str) -> TurboQuantEngine:
    if head_dim not in _ENGINE:
        e = TurboQuantEngine(head_dim=head_dim, total_bits=3, device=device)
        e.auto_tune(seq_len=512)
        _ENGINE[head_dim] = e
    return _ENGINE[head_dim]

class TurboQuantAttentionImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, scale, **kwargs):
        self.engine = _get_engine(head_size, "cuda")

    def forward(self, query, key, value, kv_cache, attn_metadata, ...):
        # Prefill: compress KV before writing to cache
        if attn_metadata.is_prompt:
            for h in range(key.shape[1]):
                ck, cv = self.engine._compress_kv_fused(
                    key[0, h].half().contiguous(),
                    value[0, h].half().contiguous()
                )
                # write compressed repr into kv_cache slots
                ...
        # Decode: decompress on read, run standard sdpa
        ...
```

### Step 2: Register the backend with vLLM

```python
# In vllm serving config or monkey-patch at startup:
import vllm.attention
vllm.attention.ATTENTION_BACKEND_MAP["turboquant"] = TurboQuantAttentionBackend

# Launch:
# vllm serve Qwen/Qwen3-30B-A3B \
#   --attention-backend turboquant \
#   --max-model-len 131072 \
#   --gpu-memory-utilization 0.90
```

### Step 3: Validate correctness before benchmarking

```python
engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cuda")
K = torch.randn(4096, 128, device="cuda", dtype=torch.float16)
V = torch.randn(4096, 128, device="cuda", dtype=torch.float16)
ck, cv = engine._compress_kv_fused(K, V)
K_hat = ck["k_mse"]
V_hat = engine._decompress_values(cv)
import torch.nn.functional as F
assert F.cosine_similarity(K.flatten(), K_hat.flatten(), dim=0) > 0.95
assert F.cosine_similarity(V.flatten(), V_hat.flatten(), dim=0) > 0.95
```

---

## Benchmark Plan vs FusenCache k4v4b64

Model: Qwen3-30B-MoE, dtype BF16 weights, BS=1 (latency) and BS=8 (throughput).

| Axis | Values |
|------|--------|
| Context length | 4K, 32K, 128K |
| Concurrency C | 64, 256, 512 |
| KV scheme | BF16 baseline, FusenCache k4v4b64, TurboQuant 3-bit |

**Primary metrics:**
- `tok/s` (decode throughput)
- KV footprint in VRAM (GB)
- PPL delta vs BF16 KV on wikitext-2 (max acceptable: +0.3 PPL)
- Cosine similarity of reconstructed K, V at each layer

**Measurement script outline (`workspace/bench_turboquant.py`):**
```bash
for ctx in 4096 32768 131072; do
  for scheme in baseline fusencache turboquant; do
    python workspace/bench_turboquant.py \
      --model Qwen/Qwen3-30B-A3B --ctx $ctx \
      --scheme $scheme --concurrency 64 256 512 \
      >> results_kv_compression.tsv
  done
done
```

Log columns: `scheme`, `ctx_len`, `concurrency`, `tok_s`, `kv_vram_gb`,
`ppl_delta`, `k_cos_mean`, `v_cos_mean`.

---

## Kill Criterion

Abandon TurboQuant integration if ANY of the following:

1. **Quality fails:** mean cosine sim < 0.95 OR PPL delta > +0.5 on wikitext-2
   at 32K ctx. (FusenCache k4v4b64 achieves 0.991 cosine sim — TurboQuant must
   come within 0.03 of that to be competitive.)

2. **No throughput gain:** TurboQuant tok/s ≤ FusenCache tok/s at all three
   context lengths. TurboQuant needs to win at ≥ 1 ctx point by ≥ 5% to justify
   the integration complexity.

3. **Compression integration too expensive:** compress+decompress overhead on
   the decode path adds > 10% latency vs FusenCache at C=512, 4K ctx (the
   latency-sensitive regime where FusenCache is already fast).

4. **cuTile unavailable AND PyTorch fallback breaks parity:** if SM120 cuTile
   wheels are absent and PyTorch fallback is > 2x slower than FusenCache, park
   until NVIDIA ships SM120 tileiras wheels.

If TurboQuant passes at 128K ctx but not at 4K, still useful as a long-context
specialist: document as "TurboQuant preferred ctx > 64K, FusenCache preferred
ctx < 64K" and combine in serving config accordingly.
