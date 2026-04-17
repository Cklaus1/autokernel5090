# FusenCache on vLLM Main — Session Notes (2026-04-11)

## Starting Point
- **OS crash**: Disk full (920G/1007G) → OOM killer → WSL restart. Freed 305G (smartrm trash 258G, logs 22G).
- **FusenCache regression**: vLLM 0.19.1rc1 → main (0.1.dev100+g92feb99) caused C=128 throughput drop from 4,489 → 1,290 tok/s.

---

## Root Causes Found

### 1. CUDA Graph Mode Auto-Downgrade (commit `8f121f7` in vLLM main)
New vLLM `resolve_cudagraph_mode_and_sizes()` checks backend's `_cudagraph_support` level. FusenCache declares `UNIFORM_SINGLE_TOKEN_DECODE` (not `ALWAYS`), so vLLM auto-downgrades `FULL` → `FULL_DECODE_ONLY`. This changes scheduler behavior: mixed batches lose CUDA graph coverage, causing 3.4x regression at C=128.

**Fix**: `_patch_cudagraph_mode_override()` in plugin.py restores FULL mode after downgrade. However, FULL mode with mixed-batch graphs crashes on SM120 (see #3), so we accept FULL_DECODE_ONLY and compensate with larger capture sizes.

### 2. `slot_mapping` int64/int32 Type Mismatch
C++ store kernel expects int32 but vLLM passes int64 → RuntimeError at startup.

**Fix**: `slot_mapping.int()` cast in `_store_fn` dispatch.

### 3. SM120 CUDA Graph Complexity Limit
RTX 5090 (Blackwell SM120) cannot replay large CUDA graphs:
- **FULL mode (ALWAYS)**: Mixed-batch graphs crash at capture sizes ≥48 with `cudaErrorIllegalInstruction`
- **FULL mode (ALWAYS)**: Capture sizes ≤32 work but are slower (batches >32 fall to eager)
- **FULL_DECODE_ONLY**: Decode-only graphs work up to size 64 (simpler graph structure)
- **ALWAYS mode is verified correct**: `CUDA_LAUNCH_BLOCKING=1` passes at C=128/256

### 4. `supports_update_block_table` Stale Metadata (compile+piecewise crash)
With `supports_update_block_table=True`, vLLM caches metadata via `update_block_table()` which updates block_table/slot_mapping but reuses **stale `query_start_loc`** from a prior step. When batch size changes, `searchsorted` reads past the end of the smaller cached allocation → `cudaErrorIllegalAddress`.

**Fix**: Set `supports_update_block_table=False` to force `build()` every step.

### 5. Piecewise CUDA Graph Race Condition
With torch.compile + piecewise graphs, FusenCache runs eagerly between graph pieces. The `.clone()` calls in metadata cloning allocate from PyTorch's caching allocator, which conflicts with the graph's private memory pool. Also, vLLM's metadata uses `non_blocking=True` H2D copies that may not be visible on the compute stream.

**Status**: Verified correct with `CUDA_LAUNCH_BLOCKING=1`. Fix in progress — removing cloning (no-clone approach) and adding stream wait.

### 6. Gemma4 Backend Override
New vLLM forces `TRITON_ATTN` for Gemma4's heterogeneous head dims (256/512). FusenCache supports all head sizes natively.

**Fix**: `_patch_gemma4_backend_override()` in plugin.py intercepts `verify_and_update_config()`.

### 7. Memory Budget Explosion
With 30 Gemma4 layers, `num_kv_splits=32` → 8.1 GB shared buffers. Capping at 8 splits → 2.0 GB.

**Fix**: `_MAX_SHARED_SPLITS = 8` in backend.py.

---

## Commits Made

| Commit | Description | Impact |
|--------|-------------|--------|
| `97b17f6` | Memory cap, C++ store dispatch, Gemma4 backend override | Foundation fixes |
| `8e64244` | slot_mapping.int() cast, mode override patch (kept but disabled) | Startup crash fixed |
| `2b5d935` | Stable config: FULL_DECODE_ONLY + max_num_seqs=64 + captures [1..64] | 3,551 tok/s at C=64 |
| `7e27333` | Fix compile+piecewise crash: supports_update_block_table=False | Unblocks piecewise mode |
| `9cdfdc8` | Enable async scheduling, remove --no-async-scheduling | 4,029 tok/s at C=64 |

---

## Throughput Progression

| Config | C=32 | C=64 | C=128 | C=256 | Notes |
|--------|------|------|-------|-------|-------|
| **Baseline (broken)** | 1,195 | 1,290 | FAIL | FAIL | FULL_DECODE_ONLY, captures [1..32] |
| **+ capture sizes [1..64]** | 1,827 | 3,551 | 1,867* | - | max_num_seqs=64, --no-async |
| **+ async scheduling** | 2,017 | 4,029 | FAIL | - | Best stable config |
| **ALWAYS mode, captures [1..32]** | 2,016 | 1,511 | 1,438 | 1,487 | Works but slow (eager fallback >32) |
| **Piecewise (compile)** | CRASH | - | - | - | Race condition (correct with CLB=1) |
| **Eager (no graphs)** | 568 | 748 | 855 | 828 | Baseline without CUDA graphs |

\* staggered requests required

---

## Historical Peaks (vLLM 0.19.1rc1)

| Config | Peak | Notes |
|--------|------|-------|
| Stock NVFP4 BF16 KV, mode=none, CUDA graphs | 6,615 tok/s C=256 | No FusenCache |
| FusenCache eager, older vLLM | 6,685 tok/s | No longer reproducible |
| FusenCache + VLLM_COMPILE piecewise | 4,489 tok/s C=128 | With C++ kernels |
| Theoretical bandwidth ceiling | 8,000-10,000 tok/s | Model-specific estimate |

---

## Key Architecture Insights

### FusenCache Plugin Patches (plugin.py, 8 steps)
1. Register CUSTOM attention backend
2. Patch CacheDType to accept k4v4/k8v8 strings
3. Patch backend selection for FusenKV dtypes
4. Patch dtype string → torch.dtype mapping
5. Patch KV cache spec for correct page sizing
6. Patch Gemma4 verify_and_update_config (TRITON_ATTN override)
7. (Disabled) Patch cudagraph mode resolution
8. Startup hooks

### Dual Kernel Paths
- **C++ CUDA kernels** (`fusencache_decode.so`): Graph-safe, no Triton JIT. Used when available.
- **Triton kernels**: Portable fallback. Crash on SM120 inside large CUDA graphs.

### Universal Decode Path (backend.py lines 964-1048)
Treats every prefill token as a single-token decode using pure tensor ops (searchsorted, arange, clamp). Designed for CUDA graph safety. Logic is correct (verified with CUDA_LAUNCH_BLOCKING=1) but blocked by SM120 graph size limits at larger batch sizes.

### Async Race Protection
- Metadata cloning: prevents CPU from modifying block_table/seq_lens while GPU reads
- CUDA event fence: GPU-side ordering for shared buffer reuse (~5us)
- Stream sync: needed for piecewise mode (non_blocking H2D copy race)

---

## Remaining Work

### Path to 6,685 tok/s
Fix the piecewise CUDA graph race condition. The `supports_update_block_table=False` fix is committed. The remaining issue is `.clone()` allocations conflicting with graph memory pool. Approach: skip cloning in piecewise mode (metadata is stable during graph replay).

### Path to 11,000 tok/s
1. Piecewise graphs working (MLP/norm/MoE in graphs, attention eager)
2. Raise max_num_seqs to 256+ (requires fixing C++ decode kernel batch limit)
3. Async scheduling enabled
4. High concurrency (C=256-512)
5. Possibly DP=2 across multiple GPUs

### C++ Decode Kernel Batch Limit
Crashes at batch ≥96 in eager mode. Under investigation — may be head group mapping bug or block_table OOB at large batch.

---

## Docker Environment

| Image | vLLM Version | CUDA | Status |
|-------|-------------|------|--------|
| `vllm-built:latest` | 0.19.1rc1.dev150 (cu128) | 12.8 | Proven stack, C++ kernels missing |
| `vllm-cu132:latest` | 0.1.dev100 (cu132) | 13.2 | Current dev, C++ kernels available |
| `vllm-cu132:gemma4-ready` | 0.1.dev100 (cu132) | 13.2 | Same as above, tagged |

**Container**: `vllm-gemma4` uses `vllm-cu132:gemma4-ready`
**Mounts**: `/root/projects/autokernel → /workspace`, `/root/models → /models`
**Symlink**: `/fusen → /workspace` (inside container)
**Port**: 8001 (container) → not mapped to host; use container IP 172.17.0.2:8001

---

## Launch Script Reference

```bash
# Current best config (4,029 tok/s at C=64)
python3 /fusen/fusen_kv/launch_vllm.py \
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
  --quantization modelopt \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --trust-remote-code \
  --port 8001 \
  --kv-cache-dtype k4v4b64 \
  -cc.mode none \
  -cc.cudagraph_mode full \
  -cc.cudagraph_capture_sizes '[1,2,4,8,16,24,32,48,64]' \
  -cc.max_cudagraph_capture_size 64

# Piecewise config (target: 6,685+ tok/s, currently crashes without CLB=1)
python3 /fusen/fusen_kv/launch_vllm.py \
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
  --quantization modelopt \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --trust-remote-code \
  --port 8001 \
  --kv-cache-dtype k4v4b64 \
  --no-async-scheduling \
  '-cc.custom_ops=["all"]' \
  -cc.cudagraph_mode piecewise
```
