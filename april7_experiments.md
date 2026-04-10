# April 7, 2025 — Experiment Log

## Summary

Switched from Gemma 4 31B dense to Gemma 4 26B-A4B MoE. Fixed 3 vLLM bugs. Achieved 128K context with FP8 KV cache at 1,816 tok/s serving throughput on RTX 5090. Built and tested FusenCache K4V4B32 (3.5x KV compression) with working Triton kernel and CUDA graphs.

---

## Experiments

### 1. Sliding Window KV Allocation Fix (FullAttentionSpec)
- **Goal:** Enable 128K context on Gemma 4 31B AWQ
- **Bug:** `FullAttentionSpec.max_memory_usage_bytes()` ignores `sliding_window` parameter after `unify_hybrid_kv_cache_specs()` converts SlidingWindowSpec → FullAttentionSpec
- **Fix:** Added `if self.sliding_window is not None: max_model_len = min(self.sliding_window, max_model_len)` in `kv_cache_interface.py`
- **Result:** Raised max context from 64K → 93K on 31B model. 128K still impossible — 10 full-attention layers alone need 10 GiB, only 8 GiB available
- **File:** `/usr/local/lib/python3.12/dist-packages/vllm/v1/kv_cache_interface.py`

### 2. 93K Context Serving on 31B (BF16 KV)
- **Config:** `max_model_len=95360, disable_hybrid_kv_cache_manager=True`
- **KV tokens:** 9,600
- **Results:** Peak 654 tok/s at C=30, collapsed at C=60+ (KV exhaustion)
- **Insight:** 9.6K KV tokens is too few for high concurrency

### 3. Switch to Gemma 4 26B-A4B MoE
- **Why:** Half the layers (30 vs 60), half the KV heads (8 vs 16), smaller weights (~13 GiB AWQ vs 19.6 GiB)
- **Result:** 128K context loads with 54,656 KV tokens (BF16)
- **Serving:** 3,178 tok/s at C=150, zero errors, P50=4.1s — still climbing at C=150
- **Insight:** MoE architecture is dramatically better for single-GPU serving — less weight memory = more KV headroom

### 4. NVFP4 Model Attempts
- **bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4:** Weight naming bug (`experts.X` vs `moe.experts.X`). Renamed weights but hit deeper issue — vLLM's compressed-tensors loader doesn't route `nvfp4-pack-quantized` format through its NVFP4 MoE quant class for Gemma 4
- **RedHatAI/gemma-4-26B-A4B-it-NVFP4:** Same naming bug plus per-expert unfused weights vs vLLM's fused `w13`/`w2` format. Fixed with patch to `gemma4.py` `_weight_iterator()` — model loads and produces correct output. But weights are 15.8 GiB (larger than AWQ's 13 GiB due to scale tensor overhead)
- **Insight:** NVFP4 quants give worse KV headroom than AWQ 4-bit for this model. Not worth pursuing.
- **vLLM bug fixed:** `gemma4.py` `_weight_iterator()` now handles per-expert NVFP4 scale parameters

### 5. FP8 KV Cache + AWQ Fix
- **Bug 1:** `fp8_e5m2 kv-cache is not supported with fp8 checkpoints` — false positive. `_init_kv_cache_quant()` blocks FP8 KV for ALL `BaseKVCacheMethod` instances, including `CompressedTensorsKVCacheMethod` (AWQ). Should only block for actual FP8 checkpoint methods.
- **Fix 1:** Added `not isinstance(quant_method, CompressedTensorsKVCacheMethod)` check in `attention.py:165`
- **Bug 2:** `assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}` — missing `fp8_e5m2`
- **Fix 2:** Added `"fp8_e5m2"` to the assertion set in `attention.py:466`
- **Result:** FP8 KV now works with AWQ models. 109,328 KV tokens (2x BF16). All answers correct.
- **Files:** `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py`

### 6. FP8 KV Concurrency Sweep (26B MoE)
- **Config:** 128K context, FP8 KV, AWQ 4-bit weights
- **KV tokens:** 109,328
- **Wide sweep:** C=1 to C=300, all zero errors
- **Fine sweep:** C=35-45 at 1-step granularity
- **Peak:** C=39 at 1,816 tok/s, P50=2.2s
- **Sweet spot:** C=37-40
- **After C=42:** throughput degrades, P50 climbs to 3s+
- **Second peak at C=200-300:** ~1,700 tok/s but P50=11-14s (batch mode)

### 7. FusenCache K4V4B32 (v6)
- **Design:** Both K and V at 4-bit symmetric, per-block-32 scales. Cache = D bytes/head/token (~3.5x compression)
- **KV tokens:** 218,656 (4x BF16)
- **Quality:** Coherent output on all 5 test prompts
- **Triton kernel:** Built v6 kernel with even/odd dim splitting for 4-bit K dequant in QK^T
- **CUDA graphs:** Fixed by making decode path unconditional (no try/except, no hasattr checks)
- **Speed:** 26.5 tok/s with CUDA graphs (70% of native), 2.9 tok/s eager
- **Serving:** Crashes at C=8+ — persistent buffer shape mismatch under varying batch sizes during graph replay
- **Files:** `fusencache_attn.py`, `triton_fusencache_v6.py`, `fusencache/config.py`

---

## Lessons Learned

### Architecture Matters More Than Compression
The 26B MoE model at BF16 KV (54K tokens) massively outperforms the 31B dense model at any compression level. Choosing the right model architecture is worth more than any amount of kernel engineering.

### FP8 KV Is Free Performance
`kv_cache_dtype='fp8_e5m2'` doubles KV capacity at zero speed cost — no custom kernels, no quality tradeoffs, native vLLM support. It was blocked by two trivial bugs, not any fundamental limitation.

### vLLM's Hybrid KV Manager Hurts More Than It Helps (for Memory Checks)
With `disable_hybrid_kv_cache_manager=True`, the memory check uses sum-of-per-layer (correct). With hybrid enabled, it uses group_size × page_size × sum_of_blocks (over-allocates by 1.7x for Gemma 4). The hybrid manager's pool-sharing model wastes memory in the check even though runtime allocation is fine.

### CUDA Graph Compatibility Checklist
To make a custom attention backend graph-safe:
1. No `hasattr()` checks in forward/decode paths
2. No `try/except` around kernel calls
3. No dynamic `torch.zeros()`/`torch.empty()` allocation (or use persistent buffers)
4. No `.item()` calls on tensors
5. Scale tensors must be pre-allocated before graph capture
6. Triton kernels themselves are graph-safe — it's the Python wrapper that breaks

### Community NVFP4 Quants Are Broken for MoE
Both tested NVFP4 models have weight naming mismatches with vLLM's Gemma 4 loader. The per-expert unfused format (`experts.0.gate_proj.weight_packed`) doesn't match vLLM's expected fused format (`w13_weight_packed`). Fixed in gemma4.py but the models themselves are larger than AWQ due to scale overhead.

### Concurrency Sweet Spots Are Sharp
The FP8 KV setup peaks at exactly C=39 (1,816 tok/s). C=35 is 3% slower, C=45 is 18% slower. Production configs should be tuned precisely, not guessed.

---

## What Worked

| Achievement | Detail |
|-------------|--------|
| 128K context on single RTX 5090 | First time — requires 26B MoE + sliding window fix |
| 109K KV tokens with FP8 KV | 2x BF16, zero quality/speed cost |
| 1,816 tok/s serving | C=39, FP8 KV, 26B MoE, zero errors |
| 3 vLLM bugs fixed | Sliding window alloc, FP8+AWQ block, fp8_e5m2 assertion |
| NVFP4 MoE loading fixed | gemma4.py per-expert weight remapping |
| FusenCache K4V4B32 | 218K KV tokens, working Triton kernel, CUDA graphs |

## What Failed

| Failure | Root Cause |
|---------|-----------|
| 128K on 31B model | 10 full-attention layers × 128K tokens > 8 GiB available |
| NVFP4 smaller than AWQ | Per-expert scale tensors add 2.8 GiB overhead |
| K4V4 serving at C>4 | CUDA graph replay with varying batch sizes corrupts Triton kernel buffers |
| K4V4 speed (26 tok/s) | 4-bit K dequant requires even/odd dim splitting, can't use tl.dot directly on packed data |

---

## Future Steps

### Immediate (Production)
1. **Deploy FP8 KV config** — C=39, 128K context, 26B MoE AWQ. Production-ready now.
2. **Prefix caching test** — Got +52% on 31B before. Should boost 26B MoE further.
3. **24-hour stability test** — Run at C=39 for 24h to verify no memory leaks or crashes.

### Short-term (Performance)
4. **CUDA graphs for K4V4 serving** — Fix persistent buffer management for varying batch sizes. Would unlock 218K tokens at ~26 tok/s per user.
5. **K8V4B16 on 26B MoE** — Better K quality (0.5% error vs ~3% for K4) at 2.3x compression. May be better quality/speed tradeoff.
6. **FP8 KV + prefix caching combined sweep** — Could push throughput past 2,500 tok/s.

### Medium-term (Research)
7. **Triton K4V4 kernel optimization** — Current kernel splits Q into even/odd dims. A kernel that operates on packed 4-bit data directly (without interleaving) could be 2x faster.
8. **K4V4 CUDA graph fix** — Need fixed-size intermediate buffers that handle all batch sizes via masking rather than resizing.
9. **Submit vLLM PRs** — Three bugs fixed locally, should upstream:
   - FullAttentionSpec sliding_window awareness
   - FP8 KV + AWQ compatibility
   - Gemma 4 NVFP4 MoE weight loading

### Validated Dead Ends
- NVFP4 quants for 26B MoE — larger than AWQ, no benefit
- 128K on 31B dense — physically impossible on 32GB GPU
- FusenCache K4V4 serving — crashes under concurrent load (graph buffer issue)
