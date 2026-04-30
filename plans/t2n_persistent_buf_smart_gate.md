# T2-N Persistent-Buffer Smart Gate
**Tag:** W8_T2N_persistent_buf_smart_gate
**Date:** 2026-04-18
**File edited:** `patches/fused_shuffle_quant_wrapper.py`

---

## Root Cause

`AUTOKERNEL_FUSED_PERSISTENT_BUF=1` (Fix A default from W7_T2N_gemma4_fix_AB) caused a
Qwen3 regression: 23,254 → 17,928 tok/s (0.77×). Fix A adds a per-call `.copy_()` of the
scale buffer that was absent in the historical direct-view path. The copy is necessary for
Gemma4 (small buffer → persistent slab needed), but harmful for Qwen3 (large buffer already
fits the .so's allocation; direct view was safe and zero-copy).

Setting `=0` restores Qwen3 but loses Fix A's Gemma4 benefit (1,633 vs 0.62× KILL).

---

## Fix: Shape-Aware Smart Gate

The persistent-buffer path is activated **only when the tensor shape indicates a model that
benefits from it**. The activation heuristic is based on `padded_k_int32` (the int32 column
count of `scales_i32` from the .so) and `num_experts`:

```python
use_persistent_buf = (
    padded_k_int32 >= 44   # K > 2048 threshold
    or num_experts > 128
)
```

### Why K_padded = 44?

`padded_k_int32 = ceil(K // 16, 4)` — the number of int32 columns in the scale buffer.

| Model  | K    | n_blocks = K//16 | padded_k_int32 = ceil(n_blocks,4) | Gate decision |
|--------|------|-----------------|-----------------------------------|---------------|
| Qwen3  | 2048 | 128             | 32                                | 32 >= 44? **NO** → direct view |
| Gemma4 | 2816 | 176             | 44                                | 44 >= 44? **YES** → persistent |

The threshold 44 is the smallest integer that separates Qwen3 (32) from Gemma4 (44).
Any future K >= 2816 will also satisfy the gate and get persistent buffer automatically.

---

## New Env Var Semantics

| `AUTOKERNEL_FUSED_PERSISTENT_BUF` | Behavior |
|---|---|
| `auto` **(new default)** | Shape-gated: persistent only when padded_k_int32 >= 44 OR num_experts > 128 |
| `1` | Always persistent (Fix A original) — for debug/isolation |
| `0` | Always direct view (pre-Fix-A) — safety valve |

---

## Diff Summary

1. Replaced `_PERSISTENT_BUF_ENABLED = ... != "0"` with:
   - `_PERSISTENT_BUF_RAW` + `_PERSISTENT_BUF_MODE` ("auto"/"always"/"never")
   - `_persistent_buf_auto_logged: set` for P1 one-shot logging per shape combo

2. Inside `fused_shuffle_and_quant_moe`, after kernel call:
   - Extract `padded_k_int32 = scales_i32.shape[1]` and `num_experts = expert_offsets.shape[0] - 1`
   - Resolve `use_persistent_buf` from mode + heuristic
   - P1 log `[T2N] auto persistent_buf=ON/OFF: K_padded=X num_experts=Y` on first call per shape

3. Updated fused-active banner to say `persistent_buf_mode=auto/always/never`
   and updated tag to `W8_T2N_persistent_buf_smart_gate`.

---

## BC Verification Matrix

| Model | C | M_sorted | K | padded_k_int32 | E | auto gate | Expected path | Expected result |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 1024 | 8192 | 2048 | 32 | 128 | 32>=44? N, 128>128? N → **OFF** | direct view | ≥23,254 tok/s (banked peak restored) |
| Gemma4-26B-A4B | 128 | 1024 | 2816 | 44 | 64 | 44>=44? Y → **ON** | persistent buf | ≥1,633 tok/s (Fix A intact) |
| Gemma4-26B-A4B | 32 | 256 | 2816 | 44 | 64 | M_sorted=256 < 512 → Fix B fires first | two-op fallthrough | ≥1,798 tok/s (no regression) |
| Gemma4-26B-A4B | 256 | 2048 | 2816 | 44 | 64 | 44>=44? Y → **ON** | persistent buf | ~1,800+ tok/s |

---

## Parent Bench Recipe

```bash
# 1. Qwen3 C=1024 with auto (default) — expect ≥23,254 tok/s
AUTOKERNEL_FUSED_PERSISTENT_BUF=auto ./launch_qwen3_t2n.sh  # C=1024

# 2. Gemma4 C=256+ with auto — expect ≥1,800 tok/s (Fix A intact)
AUTOKERNEL_FUSED_PERSISTENT_BUF=auto ./launch_gemma4_t2n.sh  # C=128,C=256

# 3. Gemma4 C=32 with auto — expect ≥1,798 tok/s (two-op fallthrough)
AUTOKERNEL_FUSED_PERSISTENT_BUF=auto ./launch_gemma4_t2n.sh  # C=32

# Debug: force always-on to reproduce original Fix A behavior
AUTOKERNEL_FUSED_PERSISTENT_BUF=1 ./launch_qwen3_t2n.sh  # C=1024 → expect ~17,928
```

---

## P1 Log Evidence (expected at first call per shape)

```
[T2N] auto persistent_buf=OFF: K_padded=32 num_experts=128 (tag W8_T2N_persistent_buf_smart_gate)
[T2N] auto persistent_buf=ON:  K_padded=44 num_experts=64  (tag W8_T2N_persistent_buf_smart_gate)
```

---

## Risks

1. **padded_k_int32=44 is exact boundary for Gemma4.** Using `>= 44` (not `> 44`) ensures
   Gemma4 K=2816 fires. If a future model has K=2816 but benefits from direct view, it
   will get persistent buffer — acceptable since persistent buf is only harmful when the
   direct view is sufficient AND the copy cost exceeds allocation savings.

2. **num_experts > 128 guard is future-proofing.** No current model has E>128 with
   small K. If such a model appears, this gate would activate persistent buf which is
   conservative/safe (never wrong, may be slightly suboptimal vs direct view).

3. **Qwen3 E=128 exactly equals threshold (> 128 is False).** Intentional: Qwen3 uses
   direct view, which is the validated optimal path for K=2048, E=128.
