# SWA + FP8 KV Stacking — Implementation-Ready Design Spec

**Status:** design-only. Gated on SWA vLLM wiring landing + PASS ≥1.3× e2e on BF16 KV.
**Author note:** CPU/code-reading only. No GPU runs.

## Why stack FP8 on top of SWA

Today's BF16 SWA sparse decode (`kernels/triton/swa_decode.py`) delivers **4.64× @ seq=8192** and **20× @ seq=16384** vs FlashInfer dense+mask on the Gemma4 sliding shape. Measured effective bandwidth at ~30% of HBM peak on PRO 6000 → **70% BW headroom** remains, and the loop is bytes-read-from-KV dominated.

FP8 KV cuts KV bytes in half. Discovery #37 re-confirmed today shows FlashInfer's FP8 path does NOT translate that into latency on SM120 because it is compute/layout-bound (37% BW). Our SWA kernel is in a different regime: BW-bound, not compute-bound, so halving KV bytes should convert into an estimated **5–20% extra latency win** on top of the sparsity win.

The larger, certain benefit is **2× KV capacity**: at max_model_len=16384 today, FP8 KV can take us to 32768 at the same VRAM, or double max concurrent sequences.

---

## 1. Source analysis

### 1a. `kernels/triton/swa_decode.py` — KV load sites

Lines referenced below are the current file.

| Line | Site | Current dtype path |
|------|------|---------------------|
| 89 | Q load | `tl.load(Q_ptr + ...).to(tl.float32)` — stays BF16 |
| 105-109 | **Stage-1 K load (main)** | `k_vals = tl.load(K_cache_ptr + ...).to(tl.float32)` — direct to FP32, no scale |
| 111 | QK matmul | `tl.dot(q, tl.trans(k_vals)) * sm_scale` — sm_scale folded here |
| 121-125 | **Stage-1 V load (main)** | `v_vals = tl.load(V_cache_ptr + ...).to(tl.float32)` — direct to FP32, no scale |
| 130 | PV matmul / accumulate | `acc = acc * re_scale[:, None] + tl.dot(p.to(tl.float32), v_vals)` — FP32 accum |
| 143-145 | Stage-1 write mid_out | FP32 partials, no scale involvement |
| 180-191 | Stage-2 merge | Operates on FP32 mid_out, no scale involvement |

**Key takeaway:** only lines 109 and 125 need to change to consume FP8 + apply scales. Lines 111, 130, 180-191 are already FP32 and need no edit. This is surgical.

Stride assertions / assumptions in the wrapper (lines 194-292):
- Line 211: `assert k_cache.dtype in (torch.bfloat16, torch.float16)` — **must relax for FP8**.
- Line 213: `assert q.dtype == k_cache.dtype` — **must relax**: q stays BF16, kv is FP8.

### 1b. vLLM FP8 paged KV layout (inside `vllm-fusencache:latest`)

Sources: `/build/vllm/vllm/v1/attention/backends/flashinfer.py`, `/build/vllm/vllm/model_executor/layers/quantization/kv_cache.py`, `/build/vllm/vllm/v1/attention/backends/utils.py`.

- **vLLM paged KV shape:** `(num_blocks, 2, block_size, num_kv_heads, head_size)` — the "2" is K/V packed. `flashinfer.py:357-379` confirms the stride_order `(0,1,2,3,4)` for NHD layout, `(0,1,3,2,4)` for HND.
- **Our SWA kernel expects `(num_blocks, page_size, num_kv_heads, head_dim)` per separate K and V tensors.** The plugin already slices `kv_cache[..., 0, ...]` / `kv_cache[..., 1, ...]` in the BF16 path — **this slicing stays identical for FP8**; only the element dtype changes.
- **FP8 dtype:** either `torch.float8_e4m3fn` (default, `kv_cache_dtype="fp8"` or `"fp8_e4m3"`) or `torch.float8_e5m2` (`kv_cache_dtype="fp8_e5m2"`). `flashinfer.py:383-389` selects via `FlashInferBackend.get_fp8_dtype_for_flashinfer()`.
- **Scale granularity — confirmed PER-TENSOR PER-LAYER (not per-token, not per-head, not per-block):**
  - `kv_cache.py:118-121` stores `layer._k_scale` (tensor) and `layer._k_scale_float` (Python float scalar). `if not isinstance(k_scale, float)...raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")` — vLLM hard-asserts this at line 103-106.
  - `attention.py:564` sets `self._k_scale_float = self._k_scale.item()` → plain Python scalar at forward time.
- **How existing FP8 paths read scales at forward():**
  - `flashinfer.py:291`: `k_scale=layer._k_scale_float` (prefill wrapper).
  - `flashinfer.py:1325-1330`: `bmm1_scale *= layer._q_scale_float * layer._k_scale_float`, `bmm2_scale *= layer._v_scale_float` (TRTLLM path — scales folded into sm_scale).
  - `flashinfer.py:1459,1569,1584`: `k_scale=layer._k_scale_float, v_scale=layer._v_scale_float` — float args.

**Conclusion:** k_scale and v_scale are Python floats (or 1-element tensors). Our kernel takes scalars. One scalar load per program. Minimal change.

### 1c. `kernels/fp8_decode_attention.py` — reference for dequant path

Structure used:
- Lines 94-103: per-tensor OR per-head scale load. `k_scale = tl.load(K_scale_ptr).to(tl.float32)`.
- Line 103: `qk_scale = sm_scale * k_scale` — k_scale pre-folded into sm_scale to save one mul per K element. **This is the reference pattern to copy.**
- Lines 131-133: `k_fp8 = tl.load(K_cache_ptr + k_addrs, mask=k_load_mask)`; `k_vals = k_fp8.to(tl.float32)`; `k_vals = tl.where(k_load_mask, k_vals, 0.0)`. NOTE: no `.to(tl.float32)` intermediate is written back to the same variable — the FP8-to-FP32 cast is the full path. The `other=0.0` in tl.load is omitted because FP8 `other=0.0` may not be representable cleanly — the explicit `tl.where` after cast is used instead.
- Lines 152-154: mirror path for V, `v_vals = v_fp8.to(tl.float32) * v_scale`. Note v_scale is applied here (cannot be folded elsewhere since V is multiplied by softmax probs, not a pre-scaled query).
- Line 136: `qk = tl.dot(q, tl.trans(k_vals)) * qk_scale` — qk_scale already includes k_scale.

---

## 2. Proposed changes to `swa_decode.py`

Total: **~10 line changes** in the JIT body + **signature extensions** in the wrapper.

### Change 2.1 — Stage-1 signature: add scale pointers & PER_HEAD_SCALE flag

```
Line 33-35 (current signature has no scale args):
  Before:
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Mid_out_ptr,
  After:
    Q_ptr,
    K_cache_ptr,
    V_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    K_scale_ptr,       # NEW: scalar or [num_kv_heads] float32
    V_scale_ptr,       # NEW
    Mid_out_ptr,
```

```
Line 48-50 (add constexpr flags at end of constexpr block):
  Add after NUM_Q_HEADS:
    LOGITS_SOFT_CAP: tl.constexpr = 0.0,
    PER_HEAD_SCALE: tl.constexpr = 0,  # NEW: 0=per-tensor, 1=per-head
    FP8_KV: tl.constexpr = 0,          # NEW: 0=BF16/FP16, 1=FP8 dequant path
```

### Change 2.2 — Stage-1 K load (line 109)

```
Line 109 (K load + dequant):
  Before:
    k_vals = tl.load(K_cache_ptr + k_addrs, mask=k_load_mask, other=0.0).to(tl.float32)
  After:
    if FP8_KV:
        k_fp8 = tl.load(K_cache_ptr + k_addrs, mask=k_load_mask)
        k_vals = k_fp8.to(tl.float32)
        k_vals = tl.where(k_load_mask, k_vals, 0.0)
    else:
        k_vals = tl.load(K_cache_ptr + k_addrs, mask=k_load_mask, other=0.0).to(tl.float32)
```

### Change 2.3 — Scale loads (new block before main loop, after line 93)

```
After Q load (line 89-93), add:
  # Load FP8 scales (only consumed if FP8_KV=1; compiler DCEs for BF16)
  if FP8_KV:
      if PER_HEAD_SCALE:
          k_scale = tl.load(K_scale_ptr + cur_kv_head).to(tl.float32)
          v_scale = tl.load(V_scale_ptr + cur_kv_head).to(tl.float32)
      else:
          k_scale = tl.load(K_scale_ptr).to(tl.float32)
          v_scale = tl.load(V_scale_ptr).to(tl.float32)
      # Fold k_scale into sm_scale — saves one mul per K element
      qk_scale = sm_scale * k_scale
  else:
      qk_scale = sm_scale
      v_scale = 1.0  # no-op
```

### Change 2.4 — QK scale application (line 111)

```
Line 111:
  Before:
    qk = tl.dot(q, tl.trans(k_vals)) * sm_scale
  After:
    qk = tl.dot(q, tl.trans(k_vals)) * qk_scale
```
(qk_scale equals sm_scale in the BF16 path; equals sm_scale*k_scale in the FP8 path.)

### Change 2.5 — Stage-1 V load (line 125)

```
Line 125 (V load + dequant):
  Before:
    v_vals = tl.load(V_cache_ptr + v_addrs, mask=v_load_mask, other=0.0).to(tl.float32)
  After:
    if FP8_KV:
        v_fp8 = tl.load(V_cache_ptr + v_addrs, mask=v_load_mask)
        v_vals = v_fp8.to(tl.float32) * v_scale
        v_vals = tl.where(v_load_mask, v_vals, 0.0)
    else:
        v_vals = tl.load(V_cache_ptr + v_addrs, mask=v_load_mask, other=0.0).to(tl.float32)
```

### Change 2.6 — Stage-2 kernel: NO CHANGE

Stage-2 (lines 148-191) operates purely on FP32 mid_out. Scales are already baked into stage-1 partial outputs. **Zero edits needed.**

### Change 2.7 — Wrapper assertions (lines 211-213)

```
Line 211-213:
  Before:
    assert k_cache.dtype in (torch.bfloat16, torch.float16)
    assert v_cache.dtype == k_cache.dtype
    assert q.dtype == k_cache.dtype
  After:
    _FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
    assert k_cache.dtype in (torch.bfloat16, torch.float16) + _FP8_DTYPES
    assert v_cache.dtype == k_cache.dtype
    if k_cache.dtype in _FP8_DTYPES:
        assert q.dtype in (torch.bfloat16, torch.float16)
        assert k_scale is not None and v_scale is not None, "FP8 KV requires k_scale, v_scale"
    else:
        assert q.dtype == k_cache.dtype
```

### Change 2.8 — Wrapper signature + kernel launch

```
Line 194 (wrapper signature):
  Before:
    def swa_decode_attention(
        q, k_cache, v_cache, block_table, seq_lens, window,
        sm_scale=None, logits_soft_cap=0.0, num_kv_splits=0,
    ):
  After:
    def swa_decode_attention(
        q, k_cache, v_cache, block_table, seq_lens, window,
        sm_scale=None, logits_soft_cap=0.0, num_kv_splits=0,
        k_scale=None, v_scale=None,  # NEW: FP32 scalar or [num_kv_heads] tensor
    ):
```

```
Line 253 (_swa_decode_stage1 call, add scale args + constexprs):
  - Insert K_scale_ptr and V_scale_ptr args after Seq_lens, before Mid_out_ptr.
  - Handle None for BF16 path: pass a dummy 1-element float32 tensor (kernel won't read it when FP8_KV=0).
  - Compute PER_HEAD_SCALE = 1 if (k_scale is not None and k_scale.numel() > 1) else 0.
  - Compute FP8_KV = 1 if k_cache.dtype in _FP8_DTYPES else 0.
  - Pass both as constexpr kwargs.

  Concrete insertion:
    _dummy_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    k_scale_t = k_scale if k_scale is not None else _dummy_scale
    v_scale_t = v_scale if v_scale is not None else _dummy_scale
    fp8_kv = 1 if k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 0
    per_head_scale = 1 if (k_scale is not None and k_scale.numel() > 1) else 0

    _swa_decode_stage1[grid_stage1](
        q, k_cache, v_cache, block_table, seq_lens,
        k_scale_t, v_scale_t,       # NEW
        mid_out,
        ... strides unchanged ...
        sm_scale,
        WINDOW=window, HEAD_DIM=head_dim, ...,
        PER_HEAD_SCALE=per_head_scale,    # NEW
        FP8_KV=fp8_kv,                    # NEW
        ...
    )
```

**Total diff:** 2 JIT signature lines, 2 constexpr flags, 1 scale-load block (~8 lines), 2 KV load blocks (~5 lines each), 1 QK line, 2 wrapper assertion lines, 2 wrapper signature lines, 1 launch site (~5 new lines). Around **20 lines modified / 30 added** end-to-end. The `FP8_KV=0` path compiles to byte-identical code as today's kernel (Triton DCEs the dead branches).

---

## 3. Wrapper signature extension

**Before:**
```python
swa_decode_attention(query, kv_cache, block_tables, seq_lens, window_size,
                    sm_scale=None, logits_soft_cap=0.0, num_kv_splits=0)
```

**After:**
```python
swa_decode_attention(query, kv_cache, block_tables, seq_lens, window_size,
                    sm_scale=None, logits_soft_cap=0.0, num_kv_splits=0,
                    k_scale=None, v_scale=None)
```

**How the plugin gets these values from vLLM at forward time:**

From `flashinfer.py:1325` / `flashinfer.py:1459`:
```python
# layer is the Attention module; vLLM sets these in model_executor/layers/quantization/kv_cache.py
k_scale_f = layer._k_scale_float  # Python float, per-layer per-tensor
v_scale_f = layer._v_scale_float
```

Then the plugin wraps them as 1-element tensors:
```python
k_scale_t = torch.tensor([k_scale_f], dtype=torch.float32, device=query.device)
v_scale_t = torch.tensor([v_scale_f], dtype=torch.float32, device=query.device)
```

These can be cached on the layer (same constant every decode step — scales are stored at model load time, not recomputed) — build them once in the plugin wrapper's `__init__` or first-forward-cache-then-reuse to avoid a `torch.tensor(...)` allocation per step.

Note: NVFP4 models typically use a single global scale per weight (per-tensor), so `k_scale.numel() == 1` — we take the non-per-head branch. Hybrid per-head scales are supported by the PER_HEAD_SCALE flag but are not currently emitted by vLLM's FP8 KV path (it raises `ValueError` at kv_cache.py:104 if a non-scalar k_scale is provided).

---

## 4. Plugin diff — `patches/swa_gemma4_plugin.py`

**State at time of writing:** the BF16 SWA plugin is being wired by a separate agent. The exact line numbers below are indicative — adapt to final layout at implementation time. The pattern is:

```python
# Inside the SWA plugin's impl.forward(...)

# Near top of method, after extracting kv_cache and block tables:
kv_cache_dtype_str = getattr(self, "kv_cache_dtype", "auto")
is_fp8_kv = kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2")

# Optional env-var gate for A/B testing (so we can ship FP8 KV off-by-default
# and flip it independently of the BF16 SWA path):
import os
swa_fp8_enabled = os.environ.get("AUTOKERNEL_SWA_SPARSE_FP8", "0") == "1"
use_fp8_path = is_fp8_kv and swa_fp8_enabled

# Build scale tensors (cache on first forward; per-layer constants)
if use_fp8_path and not hasattr(self, "_swa_k_scale_cached"):
    self._swa_k_scale_cached = torch.tensor(
        [layer._k_scale_float], dtype=torch.float32, device=query.device)
    self._swa_v_scale_cached = torch.tensor(
        [layer._v_scale_float], dtype=torch.float32, device=query.device)

# Slice K/V halves from vLLM's packed layout.
# vLLM shape: (num_blocks, 2, block_size, num_kv_heads, head_size), dtype=fp8_e4m3fn for FP8 KV.
# If we're on the FP8 path, the cache tensor will already be viewed as float8_e4m3fn by the flashinfer path at line ~1383 of flashinfer.py — the SWA plugin must mirror that view:
if use_fp8_path:
    torch_fp8 = (torch.float8_e4m3fn if kv_cache_dtype_str in ("fp8", "fp8_e4m3")
                 else torch.float8_e5m2)
    kv_cache = kv_cache.view(torch_fp8)

k_cache_view = kv_cache[:, 0]  # (num_blocks, block_size, num_kv_heads, head_size)
v_cache_view = kv_cache[:, 1]

if use_fp8_path:
    out = swa_decode_attention(
        query, k_cache_view, v_cache_view,
        block_tables, seq_lens, window_size,
        sm_scale=self.scale, logits_soft_cap=self.logits_soft_cap,
        k_scale=self._swa_k_scale_cached,
        v_scale=self._swa_v_scale_cached,
    )
else:
    # Existing BF16 path — unchanged signature, no scales
    out = swa_decode_attention(
        query, k_cache_view, v_cache_view,
        block_tables, seq_lens, window_size,
        sm_scale=self.scale, logits_soft_cap=self.logits_soft_cap,
    )
```

**Guard rationale:** `AUTOKERNEL_SWA_SPARSE_FP8=1` keeps FP8 SWA off-by-default. `AUTOKERNEL_SWA_SPARSE=1` activates BF16 SWA (existing gate). The two gates compose: user opts into SWA and FP8 independently. If FP8 gate is off but `kv_cache_dtype=fp8`, the plugin should **not** silently route FP8 tensors into the BF16 kernel — instead fall back to FlashInfer (return `None`/raise a `NotImplementedError` depending on the plugin's fallback contract).

---

## 5. Expected behavior + verification plan

### Correctness

- **Target:** cos ≥ 0.995 vs BF16 SWA on identical bf16 input (after fp8 roundtrip of the KV tensor with `scale=0.125`).
- **Rationale:** FP8 E4M3 has ~3 mantissa bits → relative error ~2^-3 ≈ 12.5% per element BUT bounded within one ULP of the representable ladder. Across a sliding window of 4096 steps, softmax-weighted sum averages those errors — expected cos ≥ 0.9995 in practice (consistent with `kernels/fp8_decode_attention.py` behavior at similar shapes).
- **Must-not-regress:** BF16 path correctness (cos = 0.999999 today) must remain identical when `FP8_KV=0`. Triton's DCE of the dead `if FP8_KV:` branch must produce byte-identical PTX for the BF16 path.

### Latency

- **Prediction:** 5–20% gain over BF16 SWA at seq=8192, window=4096.
- **Math:** today's BF16 SWA measures 30% of 1700 GB/s peak = 510 GB/s effective. KV bytes are the dominant load. Halving them → theoretical 2× of the KV-load-portion of the loop. The remaining ~50% of the time is Q load + softmax ops + writes. Realistic: ~1.10–1.25× overall, so quoting 5–20% is the safe bracket. Any result <5% gain and the experiment goes KILL (means the FP8→FP32 dequant is bottlenecking, not the load).
- **Non-goal:** do NOT expect compounding 2× on top of the 4.64× sparsity win. The two wins combine multiplicatively in bytes saved but sub-linearly in latency because each already leaves less room for the other.

### Capacity

- **Claim:** 2× context-length at fixed VRAM. On PRO 6000 96 GB, Gemma4 26B NVFP4 weights fit in ~13 GB → 83 GB for KV. BF16 KV at the Gemma4 sliding config (25 layers × 2 kv_heads × 128 head_dim × 2 bytes × 16 batch) = 200 KB per token. FP8 halves this → max_model_len 16384 → 32768 is feasible. Global layers (5 × head_dim=256 × 2 bytes) add another ~10 KB per token but don't scale as hard.

### Test harness additions — `kernels/csrc/test_swa_decode.py`

Add a new `run_shape_fp8()` function that mirrors `run_shape()`:

```python
def run_shape_fp8(batch, seq_len, head_dim, num_q_heads, num_kv_heads, window,
                  page_size=16, device="cuda"):
    torch.manual_seed(0)
    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    k_cache_bf16, v_cache_bf16, block_table = _fill_paged_kv(
        batch, seq_len, num_kv_heads, head_dim, page_size, device, torch.bfloat16)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Quantize to FP8 e4m3 with a fixed small scale (matches NVFP4/FP8 checkpoint convention)
    fp8_max = 448.0
    k_amax = k_cache_bf16.float().abs().max().item()
    v_amax = v_cache_bf16.float().abs().max().item()
    k_scale_v = max(k_amax / fp8_max, 1e-6)
    v_scale_v = max(v_amax / fp8_max, 1e-6)
    k_fp8 = (k_cache_bf16.float() / k_scale_v).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    v_fp8 = (v_cache_bf16.float() / v_scale_v).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    k_scale_t = torch.tensor([k_scale_v], dtype=torch.float32, device=device)
    v_scale_t = torch.tensor([v_scale_v], dtype=torch.float32, device=device)

    out_fp8 = swa_decode_attention(
        q, k_fp8, v_fp8, block_table, seq_lens, window=window,
        k_scale=k_scale_t, v_scale=v_scale_t)
    out_bf16 = swa_decode_attention(
        q, k_cache_bf16, v_cache_bf16, block_table, seq_lens, window=window)

    cos = _cos(out_fp8, out_bf16)
    abs_max = (out_fp8.float() - out_bf16.float()).abs().max().item()
    # Benchmark
    t_fp8 = bench(lambda: swa_decode_attention(
        q, k_fp8, v_fp8, block_table, seq_lens, window=window,
        k_scale=k_scale_t, v_scale=v_scale_t))
    t_bf16 = bench(lambda: swa_decode_attention(
        q, k_cache_bf16, v_cache_bf16, block_table, seq_lens, window=window))
    print(f"  FP8 vs BF16: cos={cos:.5f} max_abs={abs_max:.2e} "
          f"t_fp8={t_fp8*1000:.1f}us t_bf16={t_bf16*1000:.1f}us "
          f"speedup={t_bf16/t_fp8:.2f}x")
```

Call it in `main()` after the existing BF16 sweep.

---

## 6. Integration sequencing

1. **[PENDING, separate agent]** BF16 SWA wiring lands in `patches/swa_gemma4_plugin.py` + activated via `AUTOKERNEL_SWA_SPARSE=1`.
2. **Gate 1:** validate SWA e2e at Gemma4 26B, max_model_len=16384, on real serving workload. Requires ≥1.3× throughput or ≥1.3× TTFT vs stock FlashInfer to proceed. If <1.3× the FP8 extension is not worth building — go optimize something else.
3. **Implement FP8 variant per Section 2** (single commit, ~30 lines diff).
4. **Verify in-kernel correctness** with `test_swa_decode.py` extended FP8 path. Gate: cos ≥ 0.995 vs BF16 baseline.
5. **Verify in-kernel latency**: FP8 must be ≥5% faster than BF16 SWA on the same shape (same window, same seq_len). If not, KILL — the capacity benefit alone does not justify the plugin complexity unless it's at least latency-neutral.
6. **E2E bench** at max_model_len=16384 → 32768 with `kv_cache_dtype=fp8` on Gemma4. Measure throughput + verify memory fits. Capacity doubling is the primary deliverable; latency gain is bonus.
7. **Ship** behind `AUTOKERNEL_SWA_SPARSE_FP8=1` gate; keep it opt-in for one session of A/B testing before making it default.

---

## 7. Risks / open questions

### R1. Does SM120 Triton emit efficient FP8→FP32 dequant+FMA?
**Evidence (cited):** `kernels/fp8_decode_attention.py` is the functional equivalent of this design on FP8 KV — it compiles and runs on SM120 with k_fp8→fp32 via `.to(tl.float32)` followed by an FP32 `tl.dot`. Existing code comments at line 2: "Key advantage over FA2: half the memory traffic". No known issue with Triton's codegen for this pattern on SM120. The `fp8_decode_attention` measurements (Discovery #30) show the pattern works; Discovery #37 isolates the "no latency win" problem to the FlashInfer TRTLLM path, not Triton.
**Residual risk:** Triton may not emit the hardware FP8 conversion instructions (`cvt.rn.f32.e4m3`) on SM120 if the backend lacks this path. If so, the conversion falls back to scalar unpack — still correct but slower. Mitigation: verify emitted PTX on first compile; if scalar, we accept the perf hit (still bandwidth-limited so still faster than BF16).

### R2. Where do scales live at forward() time?
**Answer:** `layer._k_scale_float` and `layer._v_scale_float` are Python float scalars set at model load (`/build/vllm/vllm/model_executor/layers/quantization/kv_cache.py:118-121`). They are **not** part of FlashInferMetadata — they are attributes of the attention `layer` module directly. The SWA plugin receives `layer` as an argument (it must, since `self.scale` and other per-layer constants live there). No metadata plumbing needed.

### R3. Does Gemma4's sliding layers use FP8 KV natively?
FP8 KV is an **override**, specified via `--kv-cache-dtype=fp8` at vLLM serve time. It is NOT encoded in Gemma4's `config.json` (Gemma4 is a BF16/NVFP4-weight model; KV dtype is independent). This means:
- The plugin must check `self.kv_cache_dtype` (the runtime override) not the model config.
- Both global layers (head_dim=256) and sliding layers (head_dim=128) would be affected by the FP8 KV flag. The SWA plugin only intercepts sliding layers — global layers continue to go through FlashInfer (which is compute-bound and won't benefit from FP8 per Discovery #37, but won't regress either).

### R4. Shape-edge: what if page_size ≠ 16?
The test and kernel both assume page_size=16. FP8 KV does not change this — it's a vLLM-level choice of allocator. Verified in `flashinfer.py:357` the shape is `(num_blocks, 2, block_size, num_kv_heads, head_size)` with block_size=16 by default. If vLLM is configured with different block_size, the existing kernel already handles it (`PAGE_SIZE: tl.constexpr` is passed at launch from `k_cache.shape[1]`).

### R5. HND vs NHD layout
`flashinfer.py:373-378` returns stride order `(0,1,3,2,4)` for HND. This swaps num_kv_heads and block_size in the physical memory layout. The current SWA wrapper reads `k_cache.shape[1]` for page_size and `k_cache.shape[2]` for num_kv_heads — this assumes NHD. If HND is used, the strides are computed correctly by `k_cache.stride(...)` BUT `shape[1]` and `shape[2]` would be swapped. **Action at implementation time:** add a shape probe or accept only NHD (vLLM's default) and assert.

---

## 8. Go-signal criteria (gating THIS spec's implementation)

Before writing any FP8 SWA code, the following MUST be true:
1. BF16 SWA wiring PR merged / plugin landed.
2. Gemma4 26B + `AUTOKERNEL_SWA_SPARSE=1` produces a measurable e2e win ≥1.3× (throughput or TTFT) at seq_len ≥ 8192.
3. Correctness: end-to-end generation quality shows no regression (BLEU/ROUGE spot check or perplexity within 0.5% on 100 prompts).

If #2 fails → SWA sparsity is not materializing at the serving layer; there's no base to stack FP8 on. Abandon this extension.

If #1 and #2 pass but #3 fails → bug in SWA wiring; fix that first, then revisit FP8.

**Only after all three hold,** implement Section 2's diff in a single focused commit (~30 lines), run test harness (Section 5), and proceed to e2e bench with `kv_cache_dtype=fp8` at doubled max_model_len.
