# Root cause: Qwen3 fused-norm plugin dispatch attribute mismatch (v1 -> v2)

Tag: W3_CA_fused_norm_qwen3_v2_diff

## Root cause

`wire_fused_norm_fp4_qwen3.py` (`_build_fused_qkv_fn`, lines 101-103) inspects
`quant_method.kernel` to identify the matmul backend:

```python
kernel = getattr(quant_method, "kernel", None)
if kernel is None:
    return None
```

`ModelOptNvFp4LinearMethod` — the class vLLM assigns to Qwen3's `qkv_proj` —
does not expose a `.kernel` attribute.  `getattr` returns `None`, the function
returns `None` immediately, and `self._fused_qkv_fn` is set to `None` on every
decoder layer.  The patched `_patched_decoder_forward` silently falls through to
the unfused path for all 48 layers.  The plugin registers without error, so
nothing in the log signals the failure.

The working Gemma4 plugin (`fused_norm_fp4_integration.py`, lines 85-102)
dispatches on `quant_method.backend` — a `NvFp4LinearBackend` enum that
**is** always present on `ModelOptNvFp4LinearMethod` after weight loading.

## Fix (v2)

Replace the `.kernel`-based dispatch with `.backend`-based dispatch, mirroring
the Gemma4 plugin exactly:

```python
# v1 (broken)
kernel = getattr(quant_method, "kernel", None)
if kernel is None:
    return None
kernel_name = type(kernel).__name__
if kernel_name not in ("CutlassNvFp4LinearKernel", ...):
    return None

# v2 (fixed)
backend = getattr(quant_method, "backend", None)
if backend is None:
    return None
if backend == NvFp4LinearBackend.MARLIN:
    return None
if backend.value.startswith("flashinfer-"):
    sub_backend = backend.value[len("flashinfer-"):]
    # ... flashinfer dispatch
elif backend == NvFp4LinearBackend.FBGEMM:
    # ... fbgemm dispatch
else:
    # cutlass default
```

All other logic (pre-allocated buffers, fused_add_op residual path, lazy build)
is preserved unchanged from v1.

## Projected throughput recovery

48 layers × 1 fused pair each = 48 fusions re-enabled.  Each fusion eliminates
1 BF16 global-memory round-trip (hidden_size=2048) and 1 extra kernel launch.
v1 ran completely unfused; v2 restores the full fusion.  Baseline T2-N peak is
19,558 gen tok/s.  Expected recovery: full return to ~19,558 tok/s baseline, plus
the +3-6% fusion gain = ~20,100-20,700 tok/s (same headroom estimate as the
original wiring plan).

## Secondary divergences (non-blocking)

- v1 adds diagnostic logging only on failure; v2 logs per-layer build success
  to confirm fusion is actually active.
- Fake-tensor stub registration and `.so` load path are identical in both
  versions.
- No Qwen3-specific structural issues found: `q_norm`/`k_norm` inline logic,
  residual handling, and MoE skip-path are all correct in v1 and preserved in v2.
