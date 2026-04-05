# FusenCache Experiment Log

All experiments run on RTX 5090 (32GB), Gemma 4 31B AWQ-4bit (19.6GB weights).
5-prompt smoke test: "2+2", "capital of Japan", "haiku mountains", "telephone inventor", "speed of light".

## Performance Progression

| # | Config | KV Tokens | tok/s | Notes |
|---|--------|-----------|-------|-------|
| 1 | FP16 baseline (eager) | 8,800 | 14.2 | enforce_eager=True |
| 2 | FP16 baseline (CUDA graphs) | 8,688 | **146.3** | The real baseline |
| 3 | FusenCache v0 FP8+FP8 (eager) | 17,600 | 15.8 | 2.0x compression |
| 4 | FusenCache v1 FP8+int4 (eager, Python) | 23,472 | 3.5 | 2.67x, Python decode |
| 5 | FusenCache v1 (eager, Triton) | 23,472 | 12.5 | First Triton kernel |
| 6 | FusenCache v1 (eager, warm Triton) | — | 26.3 | After JIT warmup |
| 7 | FusenCache v1 (CUDA graphs, first kernel) | 22,320 | 46.4 | Graphs enabled |
| 8 | FusenCache v1 (CUDA graphs, optimized) | 23,168 | 59.9 | Cleaned up kernel |
| 9 | FusenCache v1 (CUDA graphs, tl.dot+BLOCK_H) | 23,168 | **74.3** | Head-batched tensor core |
| 10 | FusenCache v1 (BLOCK_KV=64 SPLITS=32) | 23,168 | 32.4 | **Worse** — register pressure |
| 10b | FusenCache v1 (BLOCK_KV=64 SPLITS=32) | 23,168 | 32.4 | **Worse** — register pressure |
| 11 | FusenCache v3.1 selective (Python) | 23,472 | 0.1-0.2 | 3.1x token reduction |
| 12 | FusenCache v1 + n-gram spec (eager) | 21,120 | **36.8** | 1.4x over eager baseline |
| 13 | FusenCache v1 + n-gram spec (graphs) | — | FAILED | Graph capture .item() |
| 14 | FusenCache v1 + spec + graphs (FIXED) | 21K | **38.0** | Slower than dense! |
| 15 | FP16 + spec + graphs (baseline) | 8.7K | **58.7** | Also slower than dense! |
| 16 | FP16 + graphs (no spec) | 8.7K | **146.3** | Best FP16 baseline |
| 17 | FusenCache v1 + graphs (no spec) | 23.2K | **74.3** | Best custom kernel |
| 18 | Native FP8 KV + graphs | 17.4K | **145.2** | Same speed as FP16! |
| 19 | FusenCache v4 FP8+FP8 + graphs | — | FAILED | Graph capture .item() in decode |
| 20 | FusenCache v4 FP8+FP8 eager (Python) | 17.6K | 7.1 | Works! Python decode slow |
| 21 | FusenCache v4 eager + v4 Triton | 17.6K | 3.3 | Triton JIT overhead in eager |
| 22 | FusenCache v4 + CUDA graphs | — | FAILED | Mixed prefill graph capture |
| 23 | v4 Triton kernel graph test (isolated) | — | PASS | Kernel IS graph-safe |
| 24 | **v4c: Native FP8 + selective hook** | **17.4K** | **145.3** | **FULL SPEED! Option C** |
| 25 | v4c short prompts (3x) | 17.4K | 91.9 | Native speed maintained |
| 26 | v4c long prompt (1631 tok) | 17.4K | 17.0 | "STARFISH-42" correct |
| 27 | v4c monkey-patch TritonAttnImpl | — | FAILED | Breaks torch.compile + graphs |
| 28 | v4c patch disabled, native FP8 | 17.4K | 145+ | Confirms import is safe |
| 29 | **Serving: C=16 FP8 KV** | 17.4K | **544** | Continuous batching |
| 30 | **Serving: C=32 FP8 KV** | 17.4K | **839** | Sweet spot |
| 31 | **Serving: C=64 FP8 KV** | 17.4K | **961** | Peak throughput |
| 32 | Serving: C=192 sweep | 17.4K | 853 | Second peak at high C |
| 33 | **Prefix cache: C=32** | 17.4K | **1,217** | **+52% from prefix cache!** |
| 34 | Prefix cache: C=1 | 17.4K | 64 | +68%, P50: 2.6→1.6s |
| 35 | Prefix cache + standard: C=32 | 17.4K | **1,021** | Confirmed with both benchmarks |
| 36 | E2B draft spec decode | — | FAILED | OOM + architecture mismatch |
| 37 | #3: FP4 weights | — | SKIPPED | NVFP4 31B doesn't fit (30.4GB) |
| 38 | E2B text-only download + weight fix | 9.3GB | OK | Fixed prefix + architecture |
| 39 | vLLM multimodal bypass patch | — | Fixed | Allow spec decode for MM targets |
| 40 | vLLM image_token_index patch | — | Fixed | Add Gemma4 to known model list |
| 41 | E2B BF16 draft (0.88 util) | — | OOM | 19.6+9.3+1.6=30.5>27GB budget |
| 42 | E2B BF16 draft (0.95 util) | — | OOM | CUDA ctx 1.6GB blocks 0.95 |
| 43 | AutoAWQ E2B quant | — | FAILED | gemma4_text not supported |
| 44 | llm-compressor E2B | — | FAILED | transformers version conflict |
| 45 | FP8 on-the-fly draft | — | FAILED | vLLM validation error |
| 46 | Manual FP8 weights | 4.6GB | Created | But vLLM can't load raw FP8 |

## Key Insights

### What worked
- **FP8 for both K+V (v0)**: trivial implementation, 2x compression, zero quality loss
- **int4 V with per-head scale in side tensor**: 2.67x compression, good quality at short context
- **CUDA graphs**: 46→74 tok/s just from removing enforce_eager. Free performance.
- **tl.dot + BLOCK_H head batching**: 59.9→74.3 tok/s. Tensor cores matter.
- **Persistent landmarks**: no cold-K scan at decode, correct chunk selection

### ASI Insight (experiment 18)
- **Native FP8 KV gives 145 tok/s with 2x compression — zero custom code**
- Custom int4 V (FusenCache v1) gives 2.67x but costs 50% throughput (74 vs 145)
- **The real contribution is selective attention, not the codec**
- v4 design: use native FP8 speed + selective landmarks for long context

### What didn't help
- **N-gram speculative decoding**: HURTS performance on short Q&A (-49% for FP16, -49% for FusenCache). N-gram spec only helps on repetitive/long-form text, not diverse short answers.

### What didn't work
- **BLOCK_KV=64**: worse than BLOCK_KV=32 due to register pressure from combined K+V unpack
- **TurboQuant**: 3 critical bugs, garbage output on all models (reported to vLLM)
- **int4 V at long context**: needle retrieval drops to 20% at 4K context (early positions degraded)

### Key bottlenecks identified
- **2.0x gap to FP16**: combined K+V layout + int4 unpack overhead. Fundamental to format.
- **vLLM per-layer Python dispatch**: ~5ms/layer × 60 layers. Limits selective path to 0.1-0.2 tok/s.
- **Selective decode compute is fast**: 0.9ms/layer in isolation (1100 tok/s theoretical)

## TurboQuant Bugs Found

1. **Triton kernels hardcoded 2-bit MSE** — breaks tq4 silently
2. **Store/load head_dim mismatch** — breaks all tq3 configs
3. **Fundamental quality issue** — 33-75% K reconstruction error
- Posted: vllm-project/vllm#38479 comment 4184609256

## Benchmark Results (formal)

### Factual Accuracy (20 questions)
| Config | Accuracy |
|--------|----------|
| FP16 baseline | 70% (14/20) |
| FusenCache v1 | 70% (14/20) |

### Needle-in-a-Haystack
| Context | FP16 | FusenCache v1 |
|---------|------|---------------|
| 512 | 5/5 (100%) | 5/5 (100%) |
| 1024 | 5/5 (100%) | 4/5 (80%) |
| 2048 | 5/5 (100%) | 3/5 (60%) |
| 4096 | N/A (OOM) | 1/5 (20%) |

**Insight**: int4 V loses precision at early positions in long context. Hot/cold tiering or V outlier protection needed.

## Architecture

### Files created
- `vllm/fusencache/__init__.py` — package
- `vllm/fusencache/config.py` — FusenCacheConfig + FusenCacheSelectiveConfig
- `vllm/v1/attention/backends/fusencache_attn.py` — full backend (~700 lines)
- `vllm/v1/attention/ops/triton_fusencache_decode.py` — Triton kernels (~350 lines)

### Files modified
- `vllm/config/cache.py` — "fusen" CacheDType
- `vllm/v1/attention/backends/registry.py` — FUSENCACHE enum
- `vllm/platforms/cuda.py` — routing
- `vllm/model_executor/layers/attention/attention.py` — get_kv_cache_spec
- `vllm/utils/torch_utils.py` — dtype mapping

### Cache Layout
```
Per token per head: [k_fp8 (D bytes) | v_int4 (D/2 bytes)]
Slot size = 1.5D bytes
V scales: separate tensor per layer (layer._fc_v_scales)
Landmarks: separate tensor per layer (layer._fc_landmarks)
```
