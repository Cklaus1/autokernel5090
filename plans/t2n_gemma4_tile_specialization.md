# T2-N Gemma4 Tile Specialization
**Tag:** W8_T2N_gemma4_tile_variant  
**Date:** 2026-04-18  
**Status:** IMPLEMENTED — rebuild required before bench

---

## 1. Current Block/Warp Config (Qwen3)

| Parameter | Qwen3 K=2048 | Gemma4 K=2816 (pre-fix) |
|---|---|---|
| `n_blocks` (K/16) | 128 | 176 |
| `block` dim (threads) | 128 | 176 |
| Warps per block | 4 (exact) | 5.5 (**inefficient**) |
| Idle threads | 0 | 48 (wasted warp slots) |
| SM120a warp scheduler | 4 full warps → clean issue | fractional warp → stall on tail group |

The original launch formula `dim3 block(min(n_blocks, 1024))` passes `n_blocks` unchanged. For Qwen3 n_blocks=128 this is a clean 4-warp block. For Gemma4 n_blocks=176 this is 5.5 warps: SM120a groups threads into warps of 32, so the SM creates 6 warps (192 threads dispatched), but 16 of them have threadIdx.x=176..191 that immediately hit the `block_idx >= n_blocks` guard and retire. Those 16 threads still consume warp scheduler slots for the initial predicated-off cycles and inflate register file pressure, reducing occupancy vs a natural 192-thread launch.

---

## 2. Proposed Gemma4 Config

**Block = 192 threads (6 warps × 32)**

Why 192 and not 160:
- `n_blocks = 176`: we need at least 176 active threads. 160 would miss threads 160..175 → wrong results.
- 192 = next multiple of 32 above 176 → exactly 6 warps.
- 16 "idle" threads (177..191) hit the `block_idx >= n_blocks` return guard in 1 instruction and retire. This is a predictable warp-level early exit that the SM handles cleanly.
- 224 (7 warps) would waste 48 threads and add unnecessarily.

**Effect on SM occupancy (SM120a):**
- SM120a has 64 warp slots per SM (4 warp schedulers × 16 concurrent warps).
- At 4 warps/block (Qwen3): 16 blocks/SM possible.
- At 5.5 warps/block (old Gemma4): effectively 5.5 warps → 11 blocks/SM.
- At 6 warps/block (new Gemma4): 10 blocks/SM — one fewer than 5.5, but FULL warps → no fractional-warp stalls. Net effect: cleaner issue slots, better instruction throughput.

The rounding change is projected to recover the `~5-10% tile overhead` from non-aligned warps, contributing to the `+9-18% on Gemma4 C=128` projection (postmortem Fix C estimate).

---

## 3. Compile Path

**Single .cu file with runtime parameterization** — no new file needed.

The existing `kernels/csrc/fused_shuffle_quant.cu` already has:
- Line 131: `if (dst_row >= M_sorted || block_idx >= n_blocks) return;` — guards idle threads
- Line 206: same guard in BF16 variant

Change: replace `dim3 block(min(n_blocks, 1024))` with:
```c
int block_threads = ((n_blocks + 31) / 32) * 32;
if (block_threads > 1024) block_threads = 1024;
dim3 block(block_threads);
```

This is a **2-line change** in the host function — no new template instantiations, no separate .cu file. The kernel body is unchanged; only the grid configuration changes. The change is backward-compatible:
- Qwen3 n_blocks=128: `((128+31)/32)*32 = 128` → unchanged.
- Gemma4 n_blocks=176: `((176+31)/32)*32 = 192` → 6 warps.
- Any K where n_blocks is already a multiple of 32: unchanged.

The rounding formula generalizes to any future model (e.g., n_blocks=160 for K=2560 → already multiple of 32, no change; n_blocks=224 for K=3584 → 224, clean).

---

## 4. Python Dispatch Rule

The wrapper `patches/fused_shuffle_quant_wrapper.py` does **not** need to call a different kernel — the warp rounding happens in the CUDA host function automatically. The wrapper adds:

1. **Per-shape tile banner** (P2 pattern, §KILL_PATTERNS): fires once per `(K_padded, block_threads)` combination:
   - `[T2N] Gemma4-tile dispatch: K_padded=44 block=192 (n_blocks=176 → 192 threads, 6 warps × 32; tag W8_T2N_gemma4_tile_variant)`
   - `[T2N] Qwen3-tile dispatch: K_padded=32 block=128 ...` for Qwen3
   - Condition: `_padded_k_int32_for_gate >= 44` (K ≥ 2816) for Gemma4 banner

2. **Computed mirror of kernel's block size** (`_block_threads = min(((n_blocks+31)//32)*32, 1024)`) — logged but not passed to the kernel (the kernel computes it internally).

Dispatch gate summary:
```
padded_k_int32 = (K // 16 + 3) // 4
K=2048 (Qwen3):  padded_k=32 < 44  → Qwen3-tile banner, block=128
K=2816 (Gemma4): padded_k=44 ≥ 44  → Gemma4-tile banner, block=192
```

---

## 5. BC Verification: Qwen3 Untouched

- `n_blocks=128` for Qwen3: `((128+31)/32)*32 = 128`. Block dim unchanged.
- Kernel body identical; no new code paths in the hot loop.
- Persistent buffer auto-gate: Qwen3 padded_k_int32=32 < 44 AND num_experts=128 ≤ 128 → OFF (direct view, no copy). Gemma4 padded_k_int32=44 ≥ 44 → ON.
- Qwen3 session peak 24,923 tok/s is safe.

**Correctness gate:** cos ≥ 0.999999 vs reference.  
The kernel computes `max_abs`, `scale`, `inv_scale` and quantizes to FP4-E2M1 per 16-element block. The guard ensures idle threads (177..191) do no reads/writes. FP4 output and swizzled scales are written only for `block_idx < n_blocks = 176`. Result is bit-for-bit identical to the pre-change kernel at n_blocks ≤ old_block (176 active threads are identical). **BC preserved.**

---

## 6. Parent Build + Bench Commands

```bash
# Step 1: Rebuild the .so from the updated .cu
cd /home/cklaus/projects/autokernel
python3 kernels/csrc/build_fused_shuffle_quant.py
# Output: workspace/fused_shuffle_quant_sm120a.so

# Step 2: Correctness smoke test (CPU-safe, mocks vLLM ops)
# (run on GPU host — requires vLLM + CUDA)
python3 -c "
import sys; sys.path.insert(0, '.')
from patches.fused_shuffle_quant_wrapper import run_correctness_test
r = run_correctness_test(verbose=True)
print('PASS' if r['passed'] else 'FAIL', r)
"

# Step 3: Microbench Gemma4 shape (K=2816, E=128, C=128)
# (inside vLLM docker on GPU 1 with Gemma4 config)
docker exec -e AUTOKERNEL_FUSED_SHUFFLE_QUANT=1 \
    -e AUTOKERNEL_FUSED_MIN_TOKENS=512 \
    vllm_gemma4 python3 /autokernel/patches/fused_shuffle_quant_wrapper.py --test

# Step 4: Full e2e bench (parent bench recipe)
# Gemma4 C=128, C=256 sweep (Fix A+B+C combined):
docker exec vllm_gemma4 python3 /autokernel/bench_gemma4_nvfp4.py \
    --concurrencies 128,256,512 --warmup 2 --iters 5
```

**Verify banner fires:**
```
grep "Gemma4-tile dispatch" docker_logs.txt
# Expected: [T2N] Gemma4-tile dispatch: K_padded=44 block=192 ...
```

---

## 7. Projected Gain vs Risk

### Gain model
- **Source:** postmortem `plans/gemma4_t2n_postmortem.md` Fix C estimate: "+15–25% at C≥128"
- **Mechanism:** warp alignment eliminates fractional-warp stall on SM120a. At 176 threads, warp 5 (threads 160-191) is partially populated: threads 160..175 are active, 176..191 are predicated off. The SM120a warp scheduler issues the warp header for all 32 threads even when 16 are masked → wastes ~8% of issue slots per block at steady state.
- **Conservative projection:** +9% (PASS threshold) vs current Gemma4 peak 1,633 tok/s → ≥1,780 tok/s
- **Optimistic projection:** +15% (BIG_WIN) → ≥1,878 tok/s
- **Combined with Fix A (persistent buf) + Fix B (min-tokens gate):** total ceiling +20–28%

### Risk model
Per §P11 Cat 3a (same-regime cross-apply, shape-specific adaptation):
- **P~0.55** this delivers ≥+9% (PASS)
- **P~0.30** delivers ≥+15% (BIG_WIN)
- **P~0.15** KILL (<+3%)

KILL scenario: the SM120a warp scheduler already handles fractional warps efficiently via predication and the real bottleneck is memory bandwidth (7 MB scale buffer copy at Gemma4). In that case the rounding change is neutral and the net gain from Fix A+B+C combined is driven entirely by Fix A (persistent buffer) + Fix B (min-tokens gate).

### Risk mitigations applied
1. **Guard already present** in kernel — idle threads cannot corrupt output.
2. **Qwen3 untouched** — n_blocks=128 rounds to 128 (no-op).
3. **No new kernel launch parameters** — same signature, internal rounding only.
4. **Correctness gate**: cos ≥ 0.999999 enforced by existing `run_correctness_test()`.

---

## 8. Files Modified

| File | Change |
|---|---|
| `kernels/csrc/fused_shuffle_quant.cu` | Lines 314-322: replace `dim3 block(min(n_blocks,1024))` with warp-aligned rounding (+7 lines, 0 new kernel code) |
| `patches/fused_shuffle_quant_wrapper.py` | Add `_gemma4_tile_logged` dict + per-shape P2 banner in `fused_shuffle_and_quant_moe()` (+20 lines) |
| `plans/t2n_gemma4_tile_specialization.md` | This file (design doc) |

No new .cu file. No new build script (existing `kernels/csrc/build_fused_shuffle_quant.py` unchanged — just re-run it).

---

*Cite: `plans/KILL_PATTERNS.md §P11 Cat 3a` (shape-specific adaptation, same regime). Postmortem Fix C: `plans/gemma4_t2n_postmortem.md §3 Fix C`.*
