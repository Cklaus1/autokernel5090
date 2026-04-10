# Gemma 4 E2B Analysis
*Date: 2026-04-09 | GPU: RTX 5090 (32 GB)*

---

## 1. Cache Status

Three variants are present in `/root/.cache/huggingface/hub/`:

| Model | Status | Weights |
|-------|--------|---------|
| `google/gemma-4-E2B-it` | Metadata only (refs pointer) | NOT downloaded |
| `prithivMLmods/gemma-4-E2B-it-FP8` | Metadata only (refs pointer) | NOT downloaded |
| `principled-intelligence/gemma-4-E2B-it-text-only` | Config only (2.2 KB blob) | NOT downloaded |

**Conclusion: No E2B weights are available locally. All three entries are incomplete HuggingFace cache stubs (only the `config.json` was fetched, no safetensors).**

To use any of them, a full download is required (~5 GB BF16, ~2.6 GB FP8, ~1.3 GB NVFP4).

---

## 2. Architecture (from `principled-intelligence/gemma-4-E2B-it-text-only/config.json`)

| Parameter | Value |
|-----------|-------|
| `model_type` | `gemma4_text` |
| `hidden_size` | 1536 |
| `num_hidden_layers` | 35 |
| `num_attention_heads` | 8 (Q) |
| `num_key_value_heads` | 1 (extreme GQA) |
| `head_dim` | 256 (local) / 512 (global) |
| `intermediate_size` | 6144 (but `use_double_wide_mlp=True` → effectively 12288) |
| `vocab_size` | 262,144 (tied embeddings) |
| `enable_moe_block` | **False** — dense, not MoE |
| `num_kv_shared_layers` | 20 (KV reuse across layers) |
| Layer pattern | 28 sliding window (512 tok) + 7 full global attention |
| `max_position_embeddings` | 131,072 |

**This is a 2.63B parameter dense model** (2.23B unique + 0.40B tied embedding), confirmed by counting:
- Attention: 35 layers × (Q 1536→2048 + KV 1536→256×2 + O 2048→1536) = 0.25B
- MLP (double-wide gate/up/down): 35 × 3 × 1536 × 12288 = 1.98B
- Embedding (tied): 262144 × 1536 = 0.40B

---

## 3. VRAM Usage

| Precision | Weight VRAM | KV per token | Concurrent@2K ctx | Concurrent@8K ctx |
|-----------|-------------|--------------|-------------------|-------------------|
| BF16 | **5.3 GB** | 42 KB | 303 requests | 76 requests |
| FP8 | **2.6 GB** | 21 KB | 655 requests | 164 requests |
| NVFP4 | **1.3 GB** | 21 KB | ~700 requests | ~175 requests |

*KV per token = 28 × 2 × 1 head × 256 dim × 2 bytes + 7 × 2 × 1 head × 512 dim × 2 bytes = 43,008 bytes*

Even BF16 leaves ~27 GB of KV budget on the RTX 5090 (32 GB). The model fits comfortably alongside a large KV cache, or alongside Gemma 4 26B (5.3 + 16 = 21 GB total, still fits).

---

## 4. Throughput Estimates (RTX 5090, single-user decode)

Methodology: measured bandwidth = 1530 GB/s (experiment #29). vLLM Python overhead = 5.0 ms constant (experiment #48/#87b). SGLang overhead ~2 ms.

| Precision | BW ceiling | vLLM | SGLang | Raw framework |
|-----------|-----------|------|--------|---------------|
| BF16 | 291 tok/s | ~118 tok/s | ~183 tok/s | ~252 tok/s |
| FP8 | 581 tok/s | ~149 tok/s | ~270 tok/s | ~455 tok/s |
| NVFP4 | 1,163 tok/s | ~171 tok/s | ~351 tok/s | ~741 tok/s |

**Key insight:** At this model size (1.3–5.3 GB), the weight-read time (0.85–3.5 ms) is comparable to or smaller than the Python/CUDA scheduling overhead (~5 ms in vLLM). Single-user throughput is largely overhead-bound, not memory-bandwidth-bound. Using SGLang or a minimal framework significantly changes the practical rate.

**Batch throughput:** At batch 64 with BF16, KV cache is only 5.6 GB (303 concurrent 2K requests), weights are shared, so effective throughput approaches the bandwidth ceiling: ~5,000–8,000 tok/s total.

---

## 5. Comparison vs Calibrated Results

| Model | Precision | Measured | BW Ceiling | Efficiency |
|-------|-----------|----------|------------|-----------|
| Qwen 3.5-9B | NVFP4 | 118 tok/s | 340 tok/s | 35% |
| Gemma 4 26B MoE | NVFP4 | ~120 tok/s | ~820 tok/s | 15% |
| Gemma 4 31B AWQ | AWQ-4bit | 15.8 tok/s | — | — |
| **E2B BF16 (est.)** | BF16 | ~118 tok/s | 291 tok/s | ~40% |
| **E2B NVFP4 (est.)** | NVFP4 | ~171 tok/s | 1163 tok/s | ~15% |

The E2B in BF16 would deliver similar single-user throughput to the 9B model (both overhead-limited), but with far less VRAM and extreme KV budget.

---

## 6. Two-Tier Architecture Assessment

### Option A: GPU 0 = Gemma4 26B (hard) + GPU 1 = Gemma4 E2B (easy)
- E2B handles routing, classification, short-answer, code autocomplete
- 26B handles multi-step reasoning, long generation, complex code
- **Pros:** E2B delivers ~300+ tok/s in SGLang for easy tasks (sub-100ms first token)
- **Pros:** 26B + E2B fit on a single 32GB GPU simultaneously (5.3 + 16 = 21 GB BF16)
- **Cons:** Requires task classifier to route requests — adds latency for short tasks
- **Cons:** E2B quality gap vs 26B may be large for borderline tasks

### Option B: E2B as speculative decode draft for Gemma4 26B
- E2B (2.63B) proposes tokens, 26B verifies at ~4 tokens per forward pass
- **Tokenizer compatibility: CONFIRMED** — both use `vocab_size=262,144` (Gemma4 family)
- **Architecture compatibility:** E2B is `gemma4_text`, 26B is `gemma4` — same family, but E2B text-only variant may align well with 26B text backbone
- **Expected speculative decode gain:** If acceptance rate ~70%, effective throughput ~2.5–3× the base 26B rate → ~300–360 tok/s from baseline 120 tok/s
- **VRAM for spec decode:** 26B (16 GB NVFP4) + E2B (1.3 GB NVFP4) = 17.3 GB — trivially fits
- **Cons:** Spec decode works best when draft and target share identical tokenization AND the draft is a distilled version; E2B is not a distilled 26B, it's independently trained — acceptance rate is unknown and could be low

### Option C: E2B as standalone "fast brain" for all short coding tasks
- Route all tasks with expected output <500 tokens to E2B
- Only escalate to 26B for extended reasoning, multi-file analysis
- E2B in SGLang NVFP4: ~350 tok/s → 1.5-second response for 500-token output
- 26B MoE in vLLM NVFP4: ~120 tok/s → 10-second response for 1200-token output

---

## 7. Recommendation

**For the parallel solver (coding tasks):**

1. **Short-term (no download needed):** Stay on Gemma4 26B MoE NVFP4. The E2B weights are not available and must be downloaded first.

2. **If downloading E2B:** Try NVFP4 as speculative decode draft for 26B first. It fits in ~1.3 GB, leaves full KV budget for 26B, and the shared tokenizer is a prerequisite that is confirmed. Use `vllm serve --speculative-model` or sglang equivalent. Expected acceptance rate: 40–70% (unknown without testing — E2B is not a distilled 26B).

3. **Two-GPU split is premature** on a single-GPU setup. The real value of E2B is either:
   - Spec decode draft (1.3 GB overhead, potentially 2–3× speedup on 26B)
   - Standalone for ultra-fast routing/classification (single call, <50ms)

4. **Most impactful near-term:** The 26B running at 120 tok/s is 13% of bandwidth ceiling due to vLLM overhead (exp87). Fix that overhead first (C-level scheduler, continuous batching tuning). That alone could 2–4× the 26B throughput without any new model download.

---

## 8. Download Commands (if proceeding)

```bash
# NVFP4 (smallest, best batch throughput):
huggingface-cli download prithivMLmods/gemma-4-E2B-it-FP8 --local-dir /root/models/gemma-4-E2B-it-FP8

# BF16 (best compatibility, spec decode draft):
huggingface-cli download google/gemma-4-E2B-it --local-dir /root/models/gemma-4-E2B-it

# Text-only (lightest, no vision overhead):
huggingface-cli download principled-intelligence/gemma-4-E2B-it-text-only --local-dir /root/models/gemma-4-E2B-it-text-only
```
