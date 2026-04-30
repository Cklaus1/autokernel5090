# T2-N Gemma4 Fix A+B Implementation
**Tag:** W7_T2N_gemma4_fix_AB  
**Date:** 2026-04-18  
**File edited:** `patches/fused_shuffle_quant_wrapper.py`

---

## Diff Summary

### New module-level flags (lines ~79-95)

```python
_PERSISTENT_BUF_ENABLED = os.environ.get("AUTOKERNEL_FUSED_PERSISTENT_BUF", "1") != "0"
_FUSED_MIN_TOKENS: int = int(os.environ.get("AUTOKERNEL_FUSED_MIN_TOKENS", "512"))
_min_tokens_warned_once: bool = False
_fused_active_logged_once: bool = False
```

### Fix B — FUSED_MIN_TOKENS threshold (in `fused_shuffle_and_quant_moe`)

**Inserted after** `M_sorted = a_map.shape[0]`, **before** the kernel call:

```python
if M_sorted < _FUSED_MIN_TOKENS:
    if not _min_tokens_warned_once:
        logger.warning("[T2N] ... M_sorted=%d < threshold=%d — using two-op baseline ...",
                       __name__, M_sorted, _FUSED_MIN_TOKENS, M_sorted)
        _min_tokens_warned_once = True
    sorted_a = ops.shuffle_rows(a, a_map)
    return ops.scaled_fp4_experts_quant(sorted_a, ...)
```

- Default threshold: 512 (configurable via `AUTOKERNEL_FUSED_MIN_TOKENS`)
- Warning logs **once per process** (rate-limited by `_min_tokens_warned_once`)

### Fix A — Persistent buffer as DEFAULT (replacing `AUTOKERNEL_FUSED_RESHAPE_SCALES=1` opt-in)

**Was:** `if os.environ.get("AUTOKERNEL_FUSED_RESHAPE_SCALES", "0") == "1": ... else: direct view`

**Now:** `if _PERSISTENT_BUF_ENABLED:  # default True ... else: direct view`

- `_PERSISTENT_BUF_ENABLED` defaults to `True`; set `AUTOKERNEL_FUSED_PERSISTENT_BUF=0` to revert to direct view
- First allocation logs a P2 banner: `[T2N] ... persistent scale buf allocated: shape=(...) ...`
- Cache key is `(device_index, target_rows, padded_k_int32)` — one slab per device/shape combo

### P1/P2 hygiene (KILL_PATTERNS)

- **P1:** threshold fallthrough log includes `__name__` (class-equiv for module-level code)
- **P2:** first fused-path activation logs `[T2N] ... fused path active: M_sorted=... K=... persistent_buf=... min_tokens_threshold=...`
- **P2:** first persistent-buf allocation logs shape, dtype, device, and disable hint

---

## BC Verification

| Model | C | M_sorted | threshold=512 | Path |
|---|---|---|---|---|
| Qwen3-30B-A3B | 512 | 4,096 | 4096 >= 512 | **fused (unchanged)** |
| Gemma4-26B-A4B | 32 | 256 | 256 < 512 | **two-op fallthrough (fix)** |
| Gemma4-26B-A4B | 128 | 1,024 | 1024 >= 512 | fused |
| Gemma4-26B-A4B | 256 | 2,048 | 2048 >= 512 | fused |

Qwen3 at C=512 still hits the fused path. No behavior change for the banked +34% result.

---

## Parent Bench Recipe

```bash
# 1. Ensure the patch is live in the running container
# 2. Launch Gemma4 T2-N sweep
./launch_gemma4_t2n.sh  # C=32, C=128, C=256, C=512

# 3. Compare against regression baseline
# Regression baseline: C=32 → 1,113 tok/s (0.62×); two-op baseline: 1,798 tok/s
# Expected post-fix:
#   C=32  → ~1,750-1,800 tok/s (threshold fires, two-op path, no regression)
#   C=128 → fused fires, break-even or slight gain
#   C=256 → +9-18% vs Gemma4 baseline (6,615 tok/s → ~7,200-7,800)
#   C=512 → +21-33% projected

# 4. Verify Qwen3 not regressed
./launch_qwen3_t2n.sh  # C=512; expect ≥23,254 tok/s (banked peak)

# 5. Correctness gate
# max_abs ≤ 1e-3 vs stock-vLLM Gemma4 MoE output
```

---

## Performance Gates

| Concurrency | Metric | PASS | BIG WIN | KILL |
|---|---|---|---|---|
| C=32 | vs two-op baseline | ≥ -2% (neutral) | n/a | any regression |
| C=256+ | vs Gemma4 baseline | ≥ +9% | ≥ +15% | < +5% OR regression vs C=32 |

---

## Risks

1. **Persistent-buf at C=256+ still pays memcpy cost.** The slab allocation is eliminated, but the `copy_()` of `small_rows * padded_k_int32` int32s still happens each call. At M_sorted=2048, K=2816: 2048 × 44 × 4 = 360 KB copy per MoE layer. Expected ~2-5 µs overhead. Should be dominated by kernel compute at C≥128.

2. **Fix A has no effect at C=32 because Fix B fires first.** This is the intended behavior — Fix B short-circuits before Fix A is reached.

3. **Threshold of 512 may be too conservative at M_sorted=512 (C=64).** At the threshold boundary (M_sorted exactly 512), the fused kernel may be marginally slower or equal. If bench shows regression at C=64, lower threshold to 256 via `AUTOKERNEL_FUSED_MIN_TOKENS=256`.

4. **Qwen3 BC.** Fix A changes the default buffer strategy from "direct view" to "persistent buf + copy." On the first call the slab is allocated; subsequent calls pay only the copy. At M_sorted=4096, K=2048: 4096 × 32 × 4 = 524 KB per call. The historical "direct view" had zero copy cost. If Qwen3 throughput drops >2%, set `AUTOKERNEL_FUSED_PERSISTENT_BUF=0` to revert Fix A for Qwen3 while keeping Fix B.
