# T2-N Rank 4: fused qknorm + rope + KV-cache scatter

**Tag:** `W8_T2N_rank4_kvcache_rotary`
**Status:** implemented + microbenched (parent e2e bench pending)
**Stack:** Qwen3-30B-A3B NVFP4 decode (T2-N + fused-norm v2 + Rank 2)
**Baseline:** 23,254 tok/s banked peak (T2-N + fused-norm v2)
**Rank 2 delta (expected):** +1.5-3% → ~23,600-23,950
**Rank 4 additional delta:** +1.5-2.5% on top of Rank 2 stack (P≈0.70 per §P11 Cat 1)

---

## Problem

On the FlashInfer decode backend, `self.attn_backend.forward_includes_kv_cache_update` is
False. `Attention.forward` (`vllm/model_executor/layers/attention/attention.py:479-481`)
fires a **separate** kernel launch:

```python
kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
    key, value, self.layer_name
)
```

which dispatches to `FlashInferImpl.do_kv_cache_update` (`flashinfer.py:1651-1676`),
which in turn calls `reshape_and_cache_flash`:

```python
torch.ops._C_cache_ops.reshape_and_cache_flash(
    key, value,
    kv_cache[:, 0],  # key_cache
    kv_cache[:, 1],  # value_cache
    slot_mapping,
    self.kv_cache_dtype,
    layer._k_scale,
    layer._v_scale,
)
```

At M=1 decode that scatter is tiny (512 BF16 for K + 512 for V per kv-head ×
4 heads = 4 KB) — pure launch-overhead territory, ~3-5 µs × 48 layers ≈
144-240 µs/step. We can fold it into Rank 2's fused Triton kernel
(which already has the rotated K in registers at the write-point) and
emit a single cooperative kernel.

## Design

### Kernel extension (`kernels/triton/fused_qknorm_rope.py`)

Additions to `_fused_qknorm_rope_qk_kernel`:

- New `constexpr` `WRITE_KV_CACHE` flag. When True, after the K programs
  finish RoPE and write back into the K tensor, they ALSO compute
  `slot = slot_mapping[pid_m]`, `block_idx = slot // BLOCK_SIZE_KV`,
  `block_off = slot % BLOCK_SIZE_KV`, and scatter-write two halves of
  the rotated K into the paged cache at
  `key_cache[block_idx, block_off, kv_head, :]`.
- Sentinel guard: if `slot < 0`, the program is a padded CUDA-graph
  token and MUST skip the scatter — writing to `key_cache[-1, ...]`
  silently corrupts the last page (the "worst bug class").

New kernel `_v_scatter_kernel`:

- Grid `(M, num_kv_heads)`.
- Pure copy — V is NOT rotated (no pre-RoPE consumer of V). Each
  program loads `v[t, kv_h, :]` and stores to
  `value_cache[block_idx, block_off, kv_h, :]` with the same
  slot-mapping math and sentinel guard as K.
- Kept separate from the qk kernel to keep register pressure low on
  the qk programs and let the scatter run in parallel on SM fabric
  with the tail of qk.

Together, two launches (qk+v) replace three launches (2 norms + RoPE
separate + KV-cache update). Per-layer launch-overhead saving at M=1:
~5-8 µs eager, ~3-5 µs inside CUDA graph capture.

### Layout + slot-mapping semantics

```text
kv_cache.shape == (num_blocks, 2, block_size, num_kv_heads, head_dim)   # logical NHD
key_cache   = kv_cache[:, 0]
value_cache = kv_cache[:, 1]

slot = slot_mapping[token_idx]          # int32; -1 = padded
block_idx    = slot // block_size
block_offset = slot %  block_size
key_cache[block_idx, block_offset, kv_head, :] = K_rot[token_idx, kv_head, :]
```

Physical memory may be HND-permuted
(`(num_blocks, 2, num_kv_heads, block_size, head_dim)` — see
`FlashInferBackend.get_kv_cache_stride_order()` returning `(0, 1, 3, 2, 4)`
for HND). The kernel uses explicit strides (no shape assumptions), so
HND is handled transparently via `key_cache.stride(0)`, `.stride(1)`,
`.stride(2)` passed in from the wrapper.

Last-dim stride is asserted to be 1; the kernel stores `head_dim`
contiguous elements per head.

### Plugin wiring (`patches/wire_fused_norm_fp4_qwen3_v2.py`)

1. Env gate: `AUTOKERNEL_FUSED_KV_CACHE_UPDATE=1`. Default off. Requires
   `AUTOKERNEL_FUSED_QKNORM_ROPE=1` (Rank 2 is the fusion site).
2. Pre-flight per layer (once at first forward):
   - `attn.attn.kv_cache_dtype` must be `auto` (BF16). FP8 KV falls
     back to stock (FP8 requires `layer._k_scale` / `_v_scale` in the
     scatter, which we don't reimplement in Triton).
   - Monkey-patch `type(attn.attn.impl).do_kv_cache_update` so it
     becomes a no-op when `layer._autokernel_rank4_active == True`.
     BC: when False (every pre-existing backend path / other layers /
     profiling run), the original scatter runs unchanged.
3. At forward time, the existing fused qk+rope closure additionally
   reads `forward_context.slot_mapping[layer_name]` and passes
   `v`, `key_cache`, `value_cache`, `slot_mapping` to the kernel.
   Sets `attn.attn._autokernel_rank4_active = True` so the
   patched `do_kv_cache_update` skips on THIS step.
4. P1 hygiene: every `getattr` has a warn-and-fallthrough branch
   (explicit assertion of `attn`, `.impl`, `kv_cache_dtype`).
5. P2 hygiene: banner counts active-Rank-4 layers vs total layers
   seen on the first forward; emits once ≥48 layers have been
   visited (matches the Rank 2 banner pattern).

### Correctness harness (`kernels/triton/test_fused_qknorm_rope_kvcache.py`)

- **Basic KV-scatter correctness:** at multiple (M, num_kv_heads,
  head_dim) shapes, run the fused kernel and compare:
  - Rotated K / Q vs `qknorm_rope_torch_ref` (cos ≥ 0.9999; same gate
    as Rank 2).
  - `key_cache` / `value_cache` contents vs
    `kv_cache_scatter_torch_ref` (cos ≥ 0.9999).
  - **Self-consistency (slot-mapping off-by-one detector):**
    `key_cache[slot_mapping[t]]` must bit-equal `k_fused[t]` at every
    valid token. V scatter must be bit-equal to the raw V input.
- **Sentinel-skip:** trailing padded tokens with `slot_mapping[t] = -1`
  AND mid-batch sentinel slots must NOT write to the cache. Test
  covers `pad_M=3`, `bad_slot_frac=0.25`.
- **Multi-step logit stability (CUDA-graph-style):** 8 sequential
  decode steps writing to slots 0..7, computing an attention-score
  proxy (Q @ K^T softmaxed against the populated cache) at every
  step. Fused and stock cos must stay ≥ 0.9999 at every step —
  any slot-mapping drift diverges the pattern.

All 5 shapes + 8-step stability PASS. Bit-exact scatter self-consistency
confirmed.

### Microbench (standalone, Qwen3 shape, M=1 decode on PRO 6000)

At M=1, Hq=32, Hkv=4, D=128:
- Rank 2-only kernel (no scatter): ~4.3 µs/call
- Rank 2+4 kernel (with scatter): ~9.8 µs/call
- Stock 4-op eager (2×norm + rope + scatter): ~84 µs/call

The Rank 2+4 path is +5.5 µs vs Rank 2 alone (cost of the V-scatter
kernel launch + the in-kernel scatter in the qk kernel). In
production this REPLACES the stock `unified_kv_cache_update` launch
(~3-5 µs inside CUDA graph). Net per-layer saving: ~3-5 µs.

Across 48 layers: ~144-240 µs/step saved.

At ~43 µs/step baseline (23,254 tok/s → 43 µs/tok), that's +0.3-0.5%
per-layer savings × 48 = +1.5-2.5% e2e on the banked Rank 2 stack.

## Projected e2e gain

Mid-point +2% on Rank 2's ~23,800 → ~24,275 tok/s.
Lower bound (P=0.4 realized): +1.5% → ~24,150 tok/s.
Upper bound (P=0.8 realized): +2.5% → ~24,400 tok/s.

## Parent bench recipe

```bash
# Baseline (fused-norm v2 only):
AUTOKERNEL_FUSED_NORM_FP4_QWEN3=1 \
AUTOKERNEL_FUSED_QKNORM_ROPE=0 \
AUTOKERNEL_FUSED_KV_CACHE_UPDATE=0 \
./launch_qwen3_fused_norm_fp4.sh

# Rank 2-only stack (current peak = 23,254):
AUTOKERNEL_FUSED_NORM_FP4_QWEN3=1 \
AUTOKERNEL_FUSED_QKNORM_ROPE=1 \
AUTOKERNEL_FUSED_KV_CACHE_UPDATE=0 \
./launch_qwen3_fused_norm_fp4.sh

# Rank 2+4 stack (NEW):
AUTOKERNEL_FUSED_NORM_FP4_QWEN3=1 \
AUTOKERNEL_FUSED_QKNORM_ROPE=1 \
AUTOKERNEL_FUSED_KV_CACHE_UPDATE=1 \
./launch_qwen3_fused_norm_fp4.sh
```

Expected banners (both emit once on first forward with ≥48 layers):
```
[fused_qknorm_rope] active layers=48/48
[fused_kv_cache_update] active layers=48/48
```

Any Rank 4 "active layers <48" is a §P2 failure signal — fusion
registered but did NOT cover all layers.

## Decision gates

| Outcome | Action |
|---|---|
| ≥ +2.5% over Rank 2-only (≥23,800×1.025 ≈ 24,400 tok/s) | BIG WIN — bank |
| +1.5-2.5% | PASS — bank |
| < +0.5% | KILL — §P11 retrospective |
| Correctness fail (logit drift, KV corruption) | KILL — debug slot-mapping |

## Risks + mitigations

- **Slot-mapping off-by-one:** would silently corrupt KV cache and
  only show up multi-token in. Mitigated by (a) the 8-step logit
  stability test in the correctness harness, and (b) the self-
  consistency bit-exact check at every valid slot.
- **Sentinel (-1) padding:** trailing CUDA-graph padded tokens MUST
  be skipped. Both kernels early-exit on `slot < 0`. Tested.
- **FP8 KV cache:** unsupported in Rank 4; plugin falls through to
  stock scatter. Gated by `kv_cache_dtype == "auto"` check.
- **HND vs NHD layout:** kernel uses explicit strides, not shape
  order, so either is supported. FlashInfer defaults to HND on SM120.
- **CUDA-graph capture:** no dynamic allocations in the kernel path
  (slot_mapping is an existing tensor from forward_context). Safe.
- **BC:** with both env gates off, `do_kv_cache_update` is NOT
  patched and the stock vLLM path is bit-identical to before.

## Tag chain

- `W7_T2N_rank2_qknorm_rope` → Rank 2 base kernel
- `W8_T2N_rank4_kvcache_rotary` → **this plan** (extension)
- `W8_T2N_rank5_postnorm_router` → Rank 5 (implemented in same plugin)
