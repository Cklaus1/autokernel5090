# Block-Size and Chunked Prefill Analysis

**Model:** Gemma 4 26B NVFP4 (RTX 5090, SM120, 32 GB VRAM)  
**vLLM version:** V1 engine (0.19.1rc1+)  
**max-model-len:** 4096

---

## 1. Theoretical Analysis: `--block-size`

### What a block is

vLLM stores KV cache in *paged* blocks — fixed-size token slots analogous to OS virtual memory pages.  Each block holds `block_size` tokens of KV cache for a single sequence.  A sequence occupies `ceil(seq_len / block_size)` blocks.

### Memory arithmetic for Gemma 4 26B

Gemma 4 26B has:
- 62 layers, 8 KV heads, head dim 256
- KV per token (BF16): 2 × 8 × 256 × 2 bytes = **8 192 bytes/token**
- KV per token (FP8):  2 × 8 × 256 × 1 byte  = **4 096 bytes/token**

| block_size | bytes/block (FP8) | blocks per seq (max 4096 tok) | page-table entries per seq |
|:----------:|:-----------------:|:-----------------------------:|:--------------------------:|
| 16         | 65 536            | 256                           | 256                        |
| 32         | 131 072           | 128                           | 128                        |
| 64         | 262 144           | 64                            | 64                         |

At `max_model_len=4096`, going from block_size=16 to block_size=32 halves the page-table size per sequence.  This reduces per-step scheduler overhead and can improve cache-line utilization during KV reads.

### Internal fragmentation

The worst case is a sequence whose last block is only 1 token full.  Expected waste is `block_size / 2` tokens per sequence.  With 256 concurrent sequences:

| block_size | Expected wasted tokens (256 seqs) | Wasted FP8 bytes |
|:----------:|:---------------------------------:|:----------------:|
| 16         | 2 048                             | ~8 MB            |
| 32         | 4 096                             | ~16 MB           |
| 64         | 8 192                             | ~32 MB           |

On 32 GB VRAM this is negligible.  Wasted capacity never exceeds ~0.1% of total VRAM at any of these sizes.

### Recommended value: `--block-size 32`

- Halves page-table overhead vs. the default 16.
- Fragmentation cost is trivial (<0.1% VRAM).
- block_size=64 shows diminishing returns and only benefits uniform, very long outputs.

---

## 2. FusenCache Interaction: `--kv-cache-dtype k4v4b64`

FusenCache stores KV values at 4 bits each in a block of 64 tokens (`b64` in the dtype name).  The "64" suffix refers to the internal compression block, not the vLLM page block.

### How vLLM block_size interacts with FusenCache b64

FusenCache aligns its quantization groups to 64-token boundaries within each vLLM block.  This creates a constraint:

```
vllm_block_size must be a multiple of fusen_quant_block (64)
OR the overhead of partial quant blocks per page must be acceptable.
```

| vLLM block_size | Fusen quant blocks per vLLM block | Partial group possible? |
|:---------------:|:---------------------------------:|:-----------------------:|
| 16              | 0.25 (fractional)                 | yes — extra bookkeeping |
| 32              | 0.50 (fractional)                 | yes — extra bookkeeping |
| 64              | 1.00 (exact fit)                  | no — optimal            |
| 128             | 2.00 (exact fit)                  | no — optimal            |

**For FusenCache k4v4b64, `--block-size 64` is the cleanest alignment.**  With block_size=32, each vLLM page spans half a FusenCache quant block, meaning the boundary-alignment metadata is stored per-block (not per-quant-group), which adds minor overhead but does not cause correctness issues.

### Memory comparison: FP8 vs FusenCache k4v4b64

At `block_size=32`:

| KV dtype     | bytes/token | bytes/block (32 tok) | blocks in 15 GB KV pool |
|:------------:|:-----------:|:--------------------:|:-----------------------:|
| BF16 (auto)  | 8 192       | 262 144              | ~57 000                 |
| FP8          | 4 096       | 131 072              | ~114 000                |
| FusenCache k4v4b64 | ~1 024  | ~32 768              | ~458 000 (4.5x FP8)     |

FusenCache at 4-bit K + 4-bit V achieves roughly 4x the block count of FP8 and 8x the block count of BF16, allowing 4–8x more concurrent sequences before evictions.

---

## 3. Chunked Prefill Parameters

### `--max-num-batched-tokens`

Controls the total tokens (prefill + decode) per scheduler step.  The RTX 5090 (32 GB) defaults to 2048; raising to 8192 fills the GPU more efficiently.

Impact at high concurrency (C=32+):
- 2048 → 8192: expected +50–150% decode throughput (more sequences batched per step)
- 8192 → 16384: diminishing returns; watch for memory pressure

### `--max-num-partial-prefills`

Allows multiple requests to chunk-prefill simultaneously.  Default=1 means only one long prompt can be mid-prefill at a time.

| Setting | Effect |
|:-------:|:-------|
| 1       | One long prefill at a time; others wait in queue |
| 2       | Two long prefills chunk concurrently; short requests can slip in between |
| 4       | Four concurrent; each gets 1/4 of the prefill token budget per step |

Raising beyond 2 only helps when you have many simultaneous long-prompt requests.  For interactive chat workloads, 2 is the sweet spot.

---

## 4. Recommended Test Matrix

The `bench_block_size.py` script covers the following matrix.  Each cell is
tested at C=32 (high throughput) and C=128 (saturation stress), with context
lengths of 256 (short/chat) and 1024 (medium/doc) tokens.

### Priority 1 — Block size sweep (fix batched_tokens=8192, partial_prefills=2)

| block_size | Expected outcome |
|:----------:|:-----------------|
| 16         | Baseline (vLLM default); 256 pages/seq |
| **32**     | Recommended; 128 pages/seq, ~5–10% lower scheduler overhead |
| 64         | Best for FusenCache k4v4b64 alignment; may waste 32 tok/seq end |

### Priority 2 — Batched tokens sweep (fix block_size=32, partial_prefills=2)

| max_batched_tokens | Expected outcome |
|:------------------:|:-----------------|
| 2048               | Default for <80 GB GPUs; decode-limited at C>8 |
| **4096**           | Intermediate; doubles step budget vs default |
| **8192**           | Recommended; 4x default; fills GPU well at C=32 |

### Priority 3 — Partial prefill sweep (fix block_size=32, batched_tokens=8192)

| max_partial_prefills | Expected outcome |
|:--------------------:|:-----------------|
| 1                    | Default; single long prefill serializes others |
| **2**                | Recommended for mixed workloads; halves P99 TTFT for short requests when 1 long prompt is active |
| 4                    | Useful only if >4 concurrent long prompts are common |

### Full 3×3×3 sweep

The sweep covers 27 configurations.  At ~5 min startup per container + ~3 min benchmark, expect ~3.5–4 hours total for the full sweep in `--mode sweep`.

For a quicker run, use the two high-priority configs only:

```
bs16_bt8192_pp2   (current best guess, but default block size)
bs32_bt8192_pp2   (recommended target)
bs64_bt8192_pp2   (FusenCache alignment check)
bs32_bt2048_pp2   (baseline throughput)
bs32_bt4096_pp2   (intermediate)
```

---

## 5. How to Run

### Prerequisite

The production server runs on port 8000.  Use port 8001 for all tests.  Do NOT restart or interfere with the production server.

### Option A — Manual (one config at a time)

```bash
# Start test container with desired config
docker run --rm --gpus all --memory=36g \
  -v /root/models:/models:ro -p 8001:8000 --name vllm-test \
  vllm-built python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 \
    -cc.mode none -cc.cudagraph_mode full \
    --block-size 32 \
    --max-num-batched-tokens 8192 \
    --max-num-partial-prefills 2 \
    --kv-cache-dtype fp8_e4m3 &

# Wait for ready, then benchmark
python profiling/bench_block_size.py --port 8001 --mode single

# Stop test container
docker rm -f vllm-test
```

### Option B — FusenCache k4v4b64

```bash
docker run --rm --gpus all --memory=36g \
  -v /root/models:/models:ro \
  -v /root/projects/autokernel/fusen_kv:/fusen/fusen_kv:ro \
  -p 8001:8000 --name vllm-test \
  -e PYTHONPATH=/fusen \
  vllm-built python3 /fusen/fusen_kv/launch_vllm.py \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt --max-model-len 4096 \
    -cc.mode none -cc.cudagraph_mode full \
    --block-size 64 \
    --max-num-batched-tokens 8192 \
    --max-num-partial-prefills 2 \
    --kv-cache-dtype k4v4b64 &

python profiling/bench_block_size.py --port 8001 --mode single --kv-type k4v4b64

docker rm -f vllm-test
```

### Option C — Automated sweep (GPU must be free)

```bash
# CAUTION: stops/starts containers; do not run while production is active
python profiling/bench_block_size.py --mode sweep --kv-type fp8_e4m3
```

Results are written to `profiling/block_size_results.json`.

---

## 6. Expected Findings

Based on theory and prior experiments:

| Metric | block_size=16 (baseline) | block_size=32 | block_size=64 |
|:------:|:------------------------:|:-------------:|:-------------:|
| Decode throughput (C=32) | 100% | +5–10% | +3–5% |
| P99 TTFT (C=128)         | 100% | -5–10% | -3–8% |
| KV fragmentation waste   | ~8 MB | ~16 MB | ~32 MB |
| Prefill overhead (per step) | high | medium | low |

With `max_batched_tokens=8192` vs 2048:
- Decode throughput at C=32: +50–150%
- P99 TTFT at C=128: -20–40% (more decode tokens per step = fewer steps to completion)

With `max_partial_prefills=2` vs 1:
- P99 TTFT for short requests (256 tok) when mixed with long prompts: -20–40%
- Total aggregate throughput: neutral (same total GPU work, reordered)

---

## 7. Key Constraints and Gotchas

1. **block_size is baked into CUDA graphs.**  Changing `--block-size` requires stopping and restarting the server with a fresh graph capture.  Allow 3–5 min for Gemma 4 26B NVFP4.

2. **block_size must divide evenly into max_model_len** (4096 / 16, 32, 64 — all OK).

3. **FusenCache k4v4b64 and block_size=32:**  Works correctly but the quant group (64 tok) spans two vLLM blocks.  The plugin handles this via the `spec_resolver.py` boundary logic.  For maximum simplicity and performance, use `block_size=64` with FusenCache.

4. **max_num_partial_prefills=4 with small batched_tokens budget:**  If `max_batched_tokens=2048` and 4 partial prefills share it, each gets only 512 tokens/step.  This can *increase* TTFT vs pp=1 at low token budgets.  Only raise pp when batched_tokens >= 4096.

5. **RTX 5090 quirk:**  The SM120 / Blackwell architecture with `-cc.cudagraph_mode full` captures graphs at batch sizes that fit within `max_num_batched_tokens`.  Increasing this parameter increases capture time but eliminates GPU-idle padding at large batch sizes.
