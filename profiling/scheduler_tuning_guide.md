# vLLM Scheduler Tuning Guide

**vLLM version:** 0.19.1rc1.dev150+gc5bee887b (V1 engine, built from source)  
**Model:** Gemma 4 26B NVFP4 (Blackwell RTX 5090, SM120, 32GB VRAM)  
**Current launch:** `serve_gemma4_fused.sh` — max-model-len 4096, modelopt quantization

> NOTE: vLLM V1 (the default since ~0.8) has a substantially different scheduler
> from V0. Several parameters from older guides (`--scheduler-delay-factor`,
> `--preemption-mode`, `--swap-space`, `--num-scheduler-steps`) **no longer
> exist** in V1. This guide covers only parameters that actually appear in the
> V1 codebase.

---

## 1. `--max-num-batched-tokens`

**What it does:** Hard cap on the total number of tokens (prefill + decode) that
can enter the GPU in a single forward pass iteration. This is the primary knob
that controls both throughput and TTFT (time to first token) latency.

**Default:**
- H100/H200/MI300x (>70 GiB): **8192** (API server context)
- A100 and smaller GPUs: **2048** (API server context)
- RTX 5090 (32 GiB VRAM) falls into the "else" branch → **2048** default

**Our recommended value:** `8192` or `16384`

A 32 GB GPU with 4096 max_model_len can comfortably process 8192 tokens/step
without memory issues. Increasing from 2048 to 8192 roughly 4x the decode
throughput by amortizing the attention overhead across a larger batch. At 16384
you may hit memory pressure — monitor with `docker stats`.

**Expected impact:**
- Throughput: +50–200% at high concurrency (more sequences batched per step)
- TTFT: increases proportionally if a prefill fills the whole budget alone
- P99 decode latency: lower (fewer steps per sequence overall)

**Example:**
```bash
--max-num-batched-tokens 8192
```

---

## 2. `--max-num-seqs`

**What it does:** Maximum number of distinct request sequences that can be
scheduled in a single iteration. Acts as a concurrency cap.

**Default:**
- H100/H200: **1024** (API server)
- RTX 5090 (32 GiB, "else" branch): **256** (API server)

**Our recommended value:** `512`

With 4096 max_model_len and 8192 batched tokens, 256 seqs is a reasonable cap.
Increasing to 512 allows more simultaneous requests to be active (good for
throughput), but each gets fewer tokens per step on average. For P99 latency
improvement, keep this at or below `max_num_batched_tokens / avg_output_tokens`.

**Constraint:** Must satisfy `max_num_batched_tokens >= max_num_seqs`.

**Expected impact:**
- Throughput: +20–40% at sustained concurrency
- TTFT/TPOT: slightly higher per-request due to more competition for tokens

**Example:**
```bash
--max-num-seqs 512
```

---

## 3. `--enable-chunked-prefill`

**What it does:** Allows a long prefill to be split (chunked) across multiple
scheduling steps, interleaved with decode steps. Prevents long prefills from
monopolizing the GPU and causing decode latency spikes for other requests.

**Default:** Auto-enabled for all standard generate models (including Gemma 4).

**Our recommended value:** Leave enabled (default). This is already on.

Without chunked prefill, a 4096-token prompt would consume a full step and block
all running decode requests. With it, the prefill is split into `max_num_batched_tokens`-sized chunks and decode continues in parallel.

**Expected impact (when already enabled):** Baseline behavior — no change needed.

**Example:**
```bash
--enable-chunked-prefill  # (default, no need to add explicitly)
```

---

## 4. `--max-num-partial-prefills`

**What it does:** Maximum number of requests whose prefill can be *in-progress*
(partially chunked) at the same time. Default is **1**.

Setting to >1 allows multiple long prompts to prefill concurrently, but each
gets a smaller chunk of `max_num_batched_tokens` per step.

**Default:** `1`

**Our recommended value:** `2` or `3` for mixed workloads with long prompts.

With `max_num_partial_prefills=2` and `long_prefill_token_threshold` auto-set
(4% of max_model_len = ~163 tokens for 4096), shorter prompts can skip ahead
of a long one being chunked, reducing P99 TTFT for short requests.

**Expected impact:**
- P99 TTFT for short requests: -20 to -40% when mixed with long prompts
- Total throughput: roughly neutral (same GPU work, just reordered)

**Example:**
```bash
--max-num-partial-prefills 2
```

---

## 5. `--max-long-partial-prefills`

**What it does:** Within `max_num_partial_prefills`, how many can be "long"
prompts (above `long_prefill_token_threshold`). Short prompts can jump the queue
ahead of the remaining slots.

**Default:** `1`

**Our recommended value:** `1` (keep at default unless you raise
`max_num_partial_prefills` to 3+)

With `max_num_partial_prefills=2, max_long_partial_prefills=1`, one long prompt
chunks while one slot stays open for short prompts. This directly reduces P99
TTFT for interactive short queries mixed with batch long prompts.

**Example:**
```bash
--max-num-partial-prefills 2 --max-long-partial-prefills 1
```

---

## 6. `--long-prefill-token-threshold`

**What it does:** A request is classified as "long" (subject to
`max_long_partial_prefills` limits) if its prompt exceeds this token count.

**Default:** Auto-computed as `4% * max_model_len` when
`max_num_partial_prefills > 1`. At `max_model_len=4096` this is **~163 tokens**.

**Our recommended value:** `256` — explicit control is better. This separates
"conversational" (< 256 tokens) from "document processing" (> 256 tokens).

**Example:**
```bash
--long-prefill-token-threshold 256
```

---

## 7. `--scheduling-policy`

**What it does:** Queue ordering policy. `fcfs` = first-come-first-served.
`priority` = by explicit request priority field (useful for SLO tiers).

**Default:** `fcfs`

**Our recommended value:** `fcfs` for uniform workloads. Use `priority` only if
you have multi-tier SLOs (e.g., premium vs. standard users) and set the
`priority` field in requests.

**Expected impact:** Neutral for single-tier workloads.

---

## 8. `--scheduler-reserve-full-isl`

**What it does:** When admitting a new request, checks whether the *full* prompt
length fits in the KV cache before admission (not just the first chunk). Prevents
over-admission where a request starts prefilling but stalls mid-way because the
KV cache fills up.

**Default:** `True` (enabled)

**Our recommended value:** Keep `True`. Disabling it risks KV cache thrashing
(sequences get evicted and recomputed), which destroys P99 latency.

Only disable if you observe excessive request rejections with very long prompts
and are willing to handle recomputation.

**Example:**
```bash
--scheduler-reserve-full-isl  # (default, no need to add)
```

---

## 9. `--async-scheduling`

**What it does:** Enables the AsyncScheduler, which overlaps the scheduling CPU
work with the previous GPU step. Eliminates the CPU-GPU synchronization gap
between iterations, filling what would otherwise be GPU idle time.

**Default:** Auto-enabled when the executor backend supports it (standard single-
GPU and ray backends). It is the default in V1 for most configurations.

**Our recommended value:** Let it auto-enable (default). If using MTP/Eagle spec
decode, it is already required. Do **not** disable it.

If you see scheduler bottleneck in profiling (CPU latency > 1ms between steps),
confirm it is enabled with `docker logs | grep "Asynchronous scheduling"`.

**Expected impact:**
- Decode step latency: -5 to -15% (fills GPU gaps)
- Throughput: +5 to +10%

**Example:**
```bash
# Do not add --async-scheduling=false unless debugging
```

---

## 10. `--stream-interval`

**What it does:** Buffer size (in tokens) before streaming a response chunk to
the client. `1` = send each token immediately. `10` = buffer 10 tokens before
flushing. Larger values reduce host-side I/O overhead.

**Default:** `1`

**Our recommended value:** `1` for interactive/chat workloads (smooth UX).
`5` or `10` for batch/offline inference where streaming overhead matters.

**Expected impact:**
- With `stream_interval=1`: slightly higher Python overhead per response, but
  users see tokens arrive immediately
- With `stream_interval=10`: reduces per-step overhead by ~5–10% but introduces
  noticeable token buffering

---

## 11. `--block-size`

**What it does:** Size of a KV cache page in tokens. Smaller blocks = better
memory utilization (less internal fragmentation), worse GPU memory locality.
Larger blocks = less metadata overhead, better cache line utilization.

**Default:** `16` tokens per block (from `CacheConfig.DEFAULT_BLOCK_SIZE`)

**Our recommended value:** `32` for Gemma 4 with max_model_len=4096.

With block_size=32 and 4096 max_model_len, there are 128 blocks per sequence
max. At block_size=16 there are 256 blocks, doubling the page table overhead.
Going to 64 reduces overhead further but increases wasted memory per partially-
filled sequence (up to 63 tokens wasted at sequence end).

**Expected impact:**
- block_size=32: ~5–10% better KV cache throughput vs. 16, same memory usage
- block_size=64: another small gain but only useful for long uniform outputs

**Note:** Block size affects CUDA graph capture sizes — changing it requires
a full restart.

**Example:**
```bash
--block-size 32
```

---

## 12. `--gpu-memory-utilization`

**What it does:** Fraction of GPU VRAM reserved for vLLM (model weights + KV
cache). The profiler allocates `total_vram * utilization` minus model weight
memory for KV cache blocks.

**Default:** `0.9` (90%)

**Our recommended value:** `0.92` for RTX 5090 (32 GB).

Gemma 4 26B NVFP4 weights are ~13–14 GB. At 90% of 32 GB = 28.8 GB available,
leaving ~15 GB for KV cache. At 92% = 29.4 GB, gaining ~0.6 GB more KV cache
(~400–800 extra blocks at block_size=32 with FP8 KV). Worth the marginal risk.

**Expected impact:**
- More KV cache blocks → higher sustained concurrency without evictions
- P99 latency: -5 to -15% at peak load (fewer evictions/recomputes)

**Example:**
```bash
--gpu-memory-utilization 0.92
```

---

## 13. `--enable-prefix-caching`

**What it does:** Reuses KV cache blocks for identical prompt prefixes across
requests. Essential for chatbots (system prompts), RAG (document context), and
code completion (file prefix).

**Default:** Auto-enabled for standard generate models (including Gemma 4).

**Our recommended value:** Keep enabled. With consistent system prompts, expect
30–70% cache hit rate, which directly eliminates prefill latency for those tokens.

**Expected impact:**
- TTFT for cache-hit requests: near-zero (only decode the new tokens)
- Throughput: significant improvement when prompts share common prefixes

---

## 14. `--kv-cache-dtype`

**What it does:** Data type for KV cache storage. `auto` uses model dtype. `fp8`
halves KV cache memory at the cost of some numerical precision.

**Default:** `auto` → bfloat16 for Gemma 4 NVFP4 (unless overridden by model)

**Our recommended value:** `fp8` (specifically `fp8_e4m3`)

With FP8 KV cache, each block uses half the memory of BF16. For Gemma 4 26B:
- BF16 KV: ~2 bytes * 2 (K+V) * num_heads * head_dim * seq_len
- FP8 KV: ~1 byte * 2 * num_heads * head_dim * seq_len

This roughly doubles the number of KV cache blocks available, supporting 2x
more concurrent requests before evictions occur.

**Expected impact:**
- Concurrent capacity: ~2x
- Throughput at high load: +40–80%
- Quality: negligible degradation for most tasks (within 0.1–0.5% on benchmarks)

**Example:**
```bash
--kv-cache-dtype fp8_e4m3
```

---

## 15. `--kv-offloading-size` (replaces deprecated `--swap-space`)

**What it does:** Size in GiB of CPU DRAM buffer for KV cache offloading. When
the GPU KV cache is full, blocks are offloaded to CPU instead of being evicted
(which would require recomputation). This is the V1 replacement for `--swap-space`.

**Default:** `None` (disabled)

**Our recommended value:** `16.0` GiB (use available CPU RAM)

With 64+ GB system RAM typical on a workstation with RTX 5090, allocating 16 GiB
for KV offloading adds a significant overflow buffer. Offloading has latency
(PCIe bandwidth ~32 GB/s on PCIe 5.0), but is much cheaper than full recompute.

**Expected impact:**
- P99 tail latency at overload: -30 to -60% (offload vs. recompute)
- Throughput at sustained high concurrency: +10–20%

**Example:**
```bash
--kv-offloading-size 16.0
--kv-offloading-backend native
```

---

## 16. `--performance-mode`

**What it does:** High-level preset that adjusts CUDA graph capture granularity
and kernel selection:
- `balanced` (default): captures graphs at sizes 1,2,4,8,16,...,256,272,...
- `interactivity`: captures every batch size 1–32 individually (no padding waste
  at small batches, best P99 at low concurrency)
- `throughput`: same as balanced but arg_utils doubles `max_num_batched_tokens`
  and `max_num_seqs` defaults if not set explicitly

**Default:** `balanced`

**Our recommended value:**
- For interactive chat (P99 focus): `interactivity`
- For batch serving / high concurrency: `throughput`

`interactivity` mode captures CUDA graphs for batch sizes 1 through 32 (instead
of jumping 1,2,4,8,16), eliminating the padding overhead when serving 3, 5, 7,
etc. simultaneous requests. This directly reduces P99 token latency at low
concurrency.

**Expected impact (`interactivity`):**
- P99 decode latency at low load (1–8 concurrent): -10 to -25% (less padding)
- Startup time: slightly longer (more graphs to capture)

**Expected impact (`throughput`):**
- Throughput at high concurrency: +20–40% (larger default batch budgets)

**Example:**
```bash
--performance-mode interactivity   # for chat/interactive
# or
--performance-mode throughput      # for batch serving
```

---

## Removed Parameters (V0 only, do NOT use with V1)

| Parameter | Status | Replacement |
|-----------|--------|-------------|
| `--scheduler-delay-factor` | **Removed in V1** | No direct replacement; async scheduling handles gaps automatically |
| `--preemption-mode swap/recompute` | **Removed in V1** | V1 uses `--kv-offloading-size` for CPU spill; recompute is the fallback |
| `--swap-space` | **Removed in V1** | Use `--kv-offloading-size` (GiB float) |
| `--num-scheduler-steps` | **Removed in V1** | Multi-step scheduling was removed; V1 always does single-step with async overlap |
| `--max-num-partial-prefills` (V0 name) | Exists in V1 but different semantics | V1 uses chunked prefill with the same flag name |

---

## Recommended Configuration for Our Setup

Target: Gemma 4 26B NVFP4, RTX 5090 32GB, max_model_len=4096, port 8000.

### For Interactive Chat (minimize P99 latency)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt \
    --max-model-len 4096 \
    --port 8000 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --max-num-partial-prefills 2 \
    --max-long-partial-prefills 1 \
    --long-prefill-token-threshold 256 \
    --block-size 32 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --kv-offloading-size 16.0 \
    --performance-mode interactivity \
    --served-model-name gemma-4-26B-A4B-it-NVFP4
```

### For High-Throughput Batch Serving

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
    --quantization modelopt \
    --max-model-len 4096 \
    --port 8000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 512 \
    --max-num-partial-prefills 3 \
    --max-long-partial-prefills 2 \
    --long-prefill-token-threshold 512 \
    --block-size 32 \
    --gpu-memory-utilization 0.92 \
    --kv-cache-dtype fp8_e4m3 \
    --enable-prefix-caching \
    --kv-offloading-size 16.0 \
    --performance-mode throughput \
    --served-model-name gemma-4-26B-A4B-it-NVFP4
```

---

## Parameter Summary Table

| Parameter | Default (our GPU) | Interactive | Throughput | Impact |
|-----------|-------------------|-------------|------------|--------|
| `--max-num-batched-tokens` | 2048 | 8192 | 16384 | High |
| `--max-num-seqs` | 256 | 256 | 512 | High |
| `--max-num-partial-prefills` | 1 | 2 | 3 | Medium |
| `--max-long-partial-prefills` | 1 | 1 | 2 | Medium |
| `--long-prefill-token-threshold` | auto (~163) | 256 | 512 | Low–Medium |
| `--block-size` | 16 | 32 | 32 | Medium |
| `--gpu-memory-utilization` | 0.90 | 0.92 | 0.92 | Medium |
| `--kv-cache-dtype` | auto (bf16) | fp8_e4m3 | fp8_e4m3 | High |
| `--enable-prefix-caching` | on | on | on | High (with shared prefixes) |
| `--kv-offloading-size` | off | 16.0 | 16.0 | Medium (at overload) |
| `--performance-mode` | balanced | interactivity | throughput | Medium |
| `--async-scheduling` | auto (on) | (leave default) | (leave default) | High |
| `--stream-interval` | 1 | 1 | 5 | Low |
| `--scheduling-policy` | fcfs | fcfs | fcfs | Neutral |
| `--scheduler-reserve-full-isl` | true | true | true | Protective |

---

## Notes on Integration with Our Fused Kernel

The `serve_gemma4_fused.sh` script uses `launch_fused_vllm.py` (via PYTHONPATH
patches) rather than `vllm.entrypoints.openai.api_server` directly. The
scheduler parameters above apply identically — they are parsed by
`EngineArgs.add_cli_args()` regardless of the entry point. Add them to the
`python3 /patches/launch_fused_vllm.py` call in `serve_gemma4_fused.sh`.

Current compile args in the script use `-cc.mode none -cc.cudagraph_mode full`.
With `--performance-mode interactivity`, the cudagraph capture logic changes.
These flags may conflict — test `--performance-mode interactivity` without the
explicit `-cc.cudagraph_mode full` override first.
