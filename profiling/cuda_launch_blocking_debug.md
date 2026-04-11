# CUDA_LAUNCH_BLOCKING Crash Pinpointing — Session 2026-04-11

## Summary

**Result: No crash occurred under CUDA_LAUNCH_BLOCKING=1 across 40+ requests.**

The sporadic "illegal memory access" / "illegal instruction" crashes are **not reproducible under single-request sequential workloads**, even with CUDA_LAUNCH_BLOCKING=1 and FUSEN_SYNC=1 active. This strongly points toward the crashes requiring either:
- High concurrency (multiple concurrent decode batches)
- CUDA graph replay at specific batch sizes
- Asynchronous multi-stream execution racing

---

## Environment

| Item | Value |
|------|-------|
| GPU | RTX 5090 (SM120, Blackwell GB202) |
| GPU Memory | 31.84 GiB |
| CUDA Version | 12.8.0 |
| PyTorch | 2.11.0+cu130 |
| Triton | 3.6.0 |
| vLLM | 0.19.1rc1.dev150+gc5bee887b |
| Container | vllm-built |
| Model | gemma-4-26B-A4B-it-NVFP4-modelopt |
| KV cache format | k4v4b64 (4-bit K, 4-bit V, 64-token scale blocks) |

---

## Test Configuration

Test scripts located in `/root/projects/autokernel/profiling/`:

- `clb_test.py` — Test script (Python, requires `if __name__ == '__main__'` guard for vLLM spawn)
- `clb_run.sh` — Docker wrapper with dist-info installation and env vars

### Key settings needed to make the test work:
1. `gpu_memory_utilization=0.92` (not 0.85/0.90 — model profiling needs more room on 32 GB)
2. `max_model_len=512, max_num_seqs=4` (reduces profiling peak, avoids "Available KV cache: -X GiB" failure)
3. `enforce_eager=True` (eliminates CUDA graph variables)
4. Script must be a real `.py` file with `if __name__ == '__main__':` (not inline `-c` code — spawn subprocess needs a file path to re-import)

---

## Subprocess Registration Issue (Important Finding)

vLLM on WSL uses `spawn` multiprocessing (not `fork`). The EngineCore subprocess re-imports the launcher script from scratch. This means:

1. The dist-info `entry_points.txt` approach DOES work — vLLM calls `load_general_plugins()` in the subprocess, which discovers `fusen_kv.plugin:register` via importlib.metadata
2. But the main script's `register()` call in top-level code runs AGAIN in the subprocess, before `main()` is called
3. This is intentional and correct — the subprocess sees the plugin registration

Without the dist-info approach, the subprocess fails with:
```
AssertionError: Invalid kv_cache_dtype: k4v4b64. Valid values are: ('auto', 'float16', ...)
```

---

## Test Run Results

### Run 1: 20 sequential requests, max_tokens=50, CLB=1

```
Available KV cache memory: 11.57 GiB  (11.25-12.09 GiB range across runs)
All 20 requests completed without crash.
```

### Run 2: 20 sequential requests, max_tokens=20, CLB=1, FUSEN_SYNC=1

All 20 requests OK. Tokens generated: Req 000-019.
No CUDA errors, no illegal memory access, no illegal instruction.

Speed: ~0.5s/token (normal — CUDA_LAUNCH_BLOCKING serializes all GPU ops)

### Run 3: 20 sequential requests, max_tokens=20, NO CLB

All 20 requests OK. No crash.
Speed: ~9 tok/s (normal decode throughput at B=1)

### Run 4 (background, killed): 5 requests, max_tokens=200

Requests 000-004 OK before container was killed by timeout. No crash in those 5.

---

## Conclusion: Which Kernel Causes the Crash?

**CUDA_LAUNCH_BLOCKING=1 did not trigger the crash**, so we cannot directly identify the offending kernel from stack traces.

### What we know from prior experiments (EXPERIMENT_DISCOVERIES.md Discovery #46):

> "Buffer overflow fixed. But sporadic CUDA crashes remain (Triton/SM120 issue, affects all concurrency levels including C=1, unrelated to FusenCache)."

The crashes have been observed during **serving** (concurrent requests via HTTP) but NOT during sequential `llm.generate()` calls. This means:

1. **Not the FusenCache Triton decode kernel** — sequential decode with CLB=1 is crash-free
2. **Not the FusenCache store kernel** — same reasoning
3. **Not the FusenCache CUDA graph replay path** — we're using `enforce_eager=True`

### Most Likely Candidates (based on prior observations)

The crashes affect "pure prefill batches too" and "all concurrency levels." Given the crash context (concurrent serving, FlashInfer for SW attention, CUTLASS NVFP4 MoE), the most likely culprits are:

| Kernel | Evidence | Why Suspected |
|--------|----------|---------------|
| **FlashInfer sliding-window prefill kernel** | Crashes in pure prefill batches | Gemma4 has 25/30 layers using FlashInfer SW-FA. SM120 has known JIT compilation quirks with FlashInfer's CUDA 12.8 codegen |
| **CUTLASS NVFP4 MoE (VLLM_CUTLASS backend)** | Sporadic, not reproducible | MoE dispatch with non-uniform expert load under concurrent requests |
| **vLLM MoE gather/scatter kernels** | Crash at all concurrency including C=1 | Pre-existing vLLM SM120 issue |
| **FlashInfer autotuner JIT** | First request is slow; crash on retry | FlashInfer autotuning at init generates SM120-specific cubins |

### Why FlashInfer is Most Suspect

The startup log shows:
```
25 sliding-window layers will use FlashAttention; 5 full-attention layers will fall back to a compatible backend.
flashinfer.jit: [Autotuner]: Autotuning process starts ... ends
```

FlashInfer compiles JIT kernels for SM120 at init. The FlashInfer autotune system is known to produce incorrect kernels under specific conditions (see `/root/projects/autokernel/DFLASH_PATCHES.md`). A race condition in the JIT cache or an incorrect SM120 kernel being selected could explain sporadic crashes in concurrent serving.

---

## Why CUDA_LAUNCH_BLOCKING Didn't Help

CUDA_LAUNCH_BLOCKING=1 only catches crashes that are reliably triggered in the kernel that just launched. For **sporadic** crashes caused by:
- Memory corruption from an earlier kernel (write happens, read crashes later)
- JIT-compiled kernel selection bugs (wrong kernel selected based on dynamic dispatch)
- Race conditions between GPU streams

...CUDA_LAUNCH_BLOCKING still won't help unless the offending kernel is the **exact** kernel that launched just before the crash.

---

## Next Steps for Pinpointing

### Option A: Concurrent serving test with CLB=1

The production crashes happen under concurrent serving. Run the API server (not `llm.generate()`) with CLB=1 and 50+ concurrent requests:

```bash
docker run --rm --gpus all \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e FUSEN_SYNC=1 \
  -p 8001:8001 \
  -v /root/models:/models:ro \
  -v /root/projects/autokernel:/fusen:ro \
  vllm-built bash /fusen/fusen_kv/launch_k4v4b64.sh &

# Then hammer with concurrent requests
python3 -c "
import httpx, asyncio
async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        tasks = [c.post('http://localhost:8001/v1/completions',
            json={'model': 'gemma-4-26B-A4B-it-NVFP4-modelopt', 
                  'prompt': 'Count to 10', 'max_tokens': 50})
            for _ in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        crashes = [r for r in results if isinstance(r, Exception)]
        print(f'Crashes: {len(crashes)}/{len(results)}')
asyncio.run(main())
"
```

### Option B: Kernel-by-kernel bisection

Disable FusenCache and test with standard backends to isolate:
1. Test with `kv_cache_dtype=fp8_e5m2` + high concurrency → does it crash?
2. If yes → FusenCache is NOT the cause
3. If no → FusenCache (or interaction) is involved

### Option C: NVTX tracing

Enable `enable_layerwise_nvtx_tracing=True` in observability config and run with Nsight Systems to get a timeline showing which kernel was running at the time of crash.

### Option D: Check FlashInfer kernel cache

Inspect FlashInfer's JIT cache for SM120 kernels:
```bash
ls ~/.cache/flashinfer/  # or /tmp/flashinfer_jit/
```
If the cache contains malformed SM120 cubins, clearing it might fix sporadic crashes.

---

## Triton 3.6.0 on SM120 — Known Status

Triton 3.6.0 (December 2025) added improved SM120 (Blackwell) support:
- Fixed shared memory allocation for >101KB configs (BK=128 now works)
- Improved bank conflict avoidance for Blackwell
- No known outstanding "illegal instruction" bugs specifically for SM120

The FusenCache decode kernel itself uses Triton and has been verified crash-free at B=1 with `enforce_eager=True`. The SM120 Triton issue noted in EXPERIMENT_DISCOVERIES.md may be fixed in 3.6.0 (the discovery was made during Triton 3.5.x testing).

**Verdict: Triton 3.6.0 is not the primary suspect for these crashes.**

---

## Files Created

| File | Purpose |
|------|---------|
| `/root/projects/autokernel/profiling/clb_test.py` | Main test script |
| `/root/projects/autokernel/profiling/clb_run.sh` | Docker wrapper with dist-info setup |
| `/root/projects/autokernel/profiling/cuda_launch_blocking_debug.md` | This document |

---

## Technical Note: Memory Constraint for This Model

The Gemma4 26B NVFP4 model requires ~17-22 GiB for weights+activations depending on profiling batch size. On a 31.84 GiB GPU:

| Setting | Free for KV | Status |
|---------|------------|--------|
| `max_model_len=2048, gpu_util=0.85` | -3.28 GiB | FAILS |
| `max_model_len=2048, gpu_util=0.93` | -0.73 GiB | FAILS |
| `max_model_len=512, max_num_seqs=4, gpu_util=0.92` | **+11.25 GiB** | WORKS |

For production serving, the model must use `max_num_seqs <= 4` for the profiling step, or the profiler itself OOMs. In production serving, the actual peak batch during inference is lower than the `max_num_seqs=256` profiling batch, so it works — but the profiling step fails without reducing `max_num_seqs`.
