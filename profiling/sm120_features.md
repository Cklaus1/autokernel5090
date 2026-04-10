# SM120 Feature Investigation: XQA MLA Kernel & L2 Cache Persistence

RTX 5090 (SM120, compute capability 12.0), FlashInfer 0.6.6, CUDA 12.8

## 1. FlashInfer XQA MLA SM120 Kernel

### What exists

FlashInfer 0.6.6 ships two XQA decode paths:
- `flashinfer.xqa.xqa()` — Standard MHA/GQA paged decode (TMA-based, supports BF16/FP16/FP8, head_dim 16-256)
- `flashinfer.xqa.xqa_mla()` — DeepSeek-style MLA decode (FP8-only, head_dim=qk_nope=128+qk_rope=64, kv_lora_rank=512)

Both are compiled from NVIDIA's TensorRT-LLM XQA kernel source, using CUDA cubins downloaded from NVIDIA's artifactory at runtime.

### SM120 dispatch status

**The XQA path is gated out on SM120.** Three separate gates block it:

1. **vLLM `supports_trtllm_attention()`** (`/build/vllm/vllm/utils/flashinfer.py:298`):
   ```python
   return current_platform.is_device_capability(100) and has_nvidia_artifactory()
   ```
   SM120 != SM100, so this returns False. The XQA decode path is never entered during normal vLLM operation.

2. **FlashInfer `determine_attention_backend()`**: Only checks for SM90a (Hopper FA3). SM120 falls back to FA2 (FlashAttention-2).

3. **FlashInfer `is_sm90a_supported()`**: `major == 9` check, so SM120 (major=12) fails.

### Can we force the XQA path?

**Yes, the kernel works on SM120.** Direct calls to `flashinfer.xqa.xqa()` succeed:
```python
from flashinfer.xqa import xqa
xqa(q=query, k_cache=k, v_cache=v, page_table=bt, seq_lens=sl, ...)
# Works with head_dim=256, BF16, GQA 8:1
```

### Performance: XQA vs FA2 on SM120

XQA is **dramatically slower** than FA2 on RTX 5090 for Gemma4's attention pattern (GQA 8:1):

| Batch | Seq Len | XQA (us) | FA2 (us) | XQA/FA2 |
|-------|---------|----------|----------|---------|
| 1     | 1024    | 66.8     | 21.5     | 3.1x slower |
| 1     | 4096    | 46.4     | 28.2     | 1.6x slower |
| 4     | 4096    | 238.4    | 24.3     | 9.8x slower |
| 16    | 4096    | 697.2    | 37.0     | 18.8x slower |
| 64    | 4096    | 2959.0   | 175.4    | 16.9x slower |
| 64    | 8192    | 12005.3  | 633.0    | 19.0x slower |

**XQA scales terribly with batch size** on SM120, while FA2 scales gracefully.

### head_dim=512 (Gemma4 global attention)

**Not supported.** XQA throws: `Invalid head_dim: 512, must be divisible by 16 and in range [16, 256]`

### XQA MLA (DeepSeek-style)

Requires FP8 (float8_e4m3fn) for both query and KV cache, with fixed dimensions (qk_nope=128, kv_lora_rank=512, qk_rope=64). Not applicable to Gemma4.

### Conclusion

**XQA is not exploitable on SM120 for Gemma4.** The kernel was designed for NVIDIA datacenter GPUs (B200/B100) with different SM and memory hierarchy. vLLM's gate is correct — enabling XQA on SM120 would hurt performance.

The FA2 backend that vLLM currently uses is already optimal for SM120 GQA decode.

---

## 2. L2 Cache Persistence API

### API status

The CUDA runtime L2 persistence API is **fully functional** on SM120:

```
Total L2 cache: 96 MB
Maximum persisting capacity: 60 MB (set via cudaDeviceSetLimit)
cudaStreamSetAttribute with accessPolicyWindow: returns 0 (success)
```

### Benchmark results

Four test scenarios were run. **L2 persistence provides no meaningful benefit on RTX 5090.**

#### Test 1: Raw memory access (copy + mul)

| Size | No Persist | Persist | Speedup |
|------|-----------|---------|---------|
| 1 MB | 0.0056 ms | 0.0146 ms | 0.39x (slower!) |
| 4 MB | 0.0165 ms | 0.0166 ms | 1.00x |
| 16 MB | 0.0220 ms | 0.0202 ms | 1.09x |
| 32 MB | 0.0378 ms | 0.0527 ms | 0.72x (slower!) |
| 60 MB | 0.0712 ms | 0.1280 ms | 0.56x (slower!) |

#### Test 2: Expert weight matmul (simulating MoE reuse)

| Expert Size | BS | No Persist | Persist | Speedup |
|------------|-----|-----------|---------|---------|
| 8 MB | 1 | 0.0123 ms | 0.0120 ms | 1.03x |
| 8 MB | 64 | 0.0152 ms | 0.0166 ms | 0.91x |
| 32 MB | 1 | 0.0162 ms | 0.0172 ms | 0.94x |
| 32 MB | 64 | 0.0291 ms | 0.0268 ms | 1.08x |

#### Test 3: Expert weight pinned while activations stream through

```
Weight: 8 MB, 50 batches cycling (25 MB activations thrashing L2)
NO persistence:   0.0195 ms/iter
WITH persistence: 0.0186 ms/iter
Speedup: 1.044x  (4.4%, within noise)
```

#### Test 4: Competing kernels (weight A pinned, B+C interference)

```
Pattern: A -> B -> C -> A -> B -> C -> ...
NO persistence:   0.0115 ms
WITH persistence: 0.0114 ms
Speedup: 1.006x  (noise)
```

### Why L2 persistence doesn't help on RTX 5090

1. **96 MB L2 is already large.** The working sets in our benchmarks fit comfortably, so the default LRU replacement policy handles it well.

2. **cuBLAS already manages L2 internally.** NVIDIA's matmul kernels already optimize their L2 usage patterns. Adding a persistence hint on top doesn't improve anything.

3. **The RTX 5090's memory subsystem has high bandwidth (1792 GB/s).** Even L2 cache misses are cheap. The benefit window for L2 persistence is narrow.

4. **The API adds overhead.** Setting access policy windows involves driver calls that can actually slow down small kernels (see 1MB result: 0.39x).

### When L2 persistence MIGHT help

- Systems with smaller L2 (e.g., older GPUs with 6-12 MB L2)
- Workloads with extreme cache thrashing from many concurrent streams
- Custom CUDA kernels designed around the persistence API (not PyTorch ops)

### Conclusion

**L2 cache persistence is not exploitable for vLLM on RTX 5090.** The hardware's default cache management is already near-optimal for our workload patterns.

---

## Overall Assessment

Neither SM120-specific feature provides actionable performance improvements for Gemma4 27B inference on RTX 5090:

| Feature | Status | Benefit |
|---------|--------|---------|
| XQA MLA kernel | Works but 2-19x slower than FA2 | None (negative) |
| XQA head_dim=512 | Not supported | N/A |
| L2 persistence | API works, no perf gain | None |

The current FlashInfer FA2 decode + standard L2 management is already the best configuration for SM120.

## Files

- `xqa_sm120_bench.py` — XQA vs FA2 decode benchmark
- `l2_persist_bench.py` — L2 persistence basic benchmark
- `l2_persist_v2.py` — L2 persistence cache-thrashing benchmark
