"""vLLM plugin: fused RMSNorm + FP4-quant wiring for Qwen3 MoE.  (v2 — fixed)

Monkey-patches ``vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer``
so the ``input_layernorm -> self_attn.qkv_proj`` path runs a single fused
kernel (``torch.ops._C.rms_norm_dynamic_fp4_quant`` /
``fused_add_rms_norm_dynamic_fp4_quant``) instead of:

    rms_norm(x[, residual]) -> scaled_fp4_quant -> cutlass_scaled_fp4_mm

The fused kernel is provided by ``workspace/fused_rms_norm_fp4_cu13.so``.

W7 ADDITION (qknorm+rope fusion)
=================================
In addition to the input-layernorm+qkv_proj fusion, this plugin now also
replaces the 3 back-to-back kernels:

    q_by_head = self.q_norm(q.view(..., head_dim))
    k_by_head = self.k_norm(k.view(..., head_dim))
    q, k      = self.rotary_emb(positions, q, k)

with ONE Triton kernel from ``kernels/triton/fused_qknorm_rope.py``. This
saves ~13 us/layer at M=1 decode on SM120 (measured vs vLLM's
``torch.ops._C.rotary_embedding`` + two ``torch.nn.functional.rms_norm``).

Disable with ``AUTOKERNEL_FUSED_QKNORM_ROPE=0``.  Default: on when this
plugin is active.

W8 ADDITION (Rank 4: KV-cache-update fusion)
=============================================
Extends the W7 kernel further: after computing rotated K, the K programs
ALSO scatter-write directly to the paged KV cache at
``slot_mapping[token_idx]``. A small companion kernel scatters V (which
does not pass through RoPE). Together these replace the separate
``torch.ops.vllm.unified_kv_cache_update`` launch that vLLM fires on
FlashInfer decode (where ``forward_includes_kv_cache_update=False``).

On M=1 decode the ``unified_kv_cache_update`` launch is ~3-5 us × 48
layers ≈ 144-240 us/step pure launch overhead; folding into the RoPE
kernel eliminates it entirely.

Enable with ``AUTOKERNEL_FUSED_KV_CACHE_UPDATE=1``.  Default: off
(requires BF16 KV cache; FP8 KV path falls through to stock).
When active, this plugin monkey-patches
``attn.impl.do_kv_cache_update`` on the ``Attention`` instance to a
no-op — so vLLM's ``unified_kv_cache_update`` still fires but becomes
a cheap sentinel, avoiding the double-write.

W8 ADDITION (post_attention_layernorm + router-gate fusion — Rank 5)
====================================================================
This plugin additionally collapses the 3 kernels:

    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual,
    )
    router_logits, _ = self.mlp.gate(hidden_states)
    # (then hidden_states feeds self.mlp.experts and shared expert)

into ONE Triton kernel from
``kernels/triton/fused_post_attn_router.py``. At M=1 decode on SM120
each of the three is launch-overhead-bound (~3-6 us pure overhead for
sub-us compute); fusing them removes 2 of 3 launches per layer × 48
layers per step.

Only the MoE layers (``Qwen3MoeSparseMoeBlock``) are handled here; the
dense-MLP layers (``Qwen3MoeMLP`` — a handful in the 30B-A3B checkpoint)
fall through to the stock 3-kernel path.

Disable with ``AUTOKERNEL_FUSED_POST_ATTN_ROUTER=0``.  Default: on when
this plugin is active.

FIX vs v1
=========
v1 dispatched via ``quant_method.kernel`` (a kernel *object*), which
``ModelOptNvFp4LinearMethod`` does not expose — causing ``_build_fused_qkv_fn``
to return ``None`` for every layer and silently falling back to the unfused path
on all 48 decoder layers.

v2 dispatches via ``quant_method.backend`` (a ``NvFp4LinearBackend`` enum),
matching the pattern used by the working Gemma4 plugin
(``patches/fused_norm_fp4_integration.py``).  The backend enum is always
present on ``ModelOptNvFp4LinearMethod`` after ``process_weights_after_loading``
has run.

Originally only the ``input_layernorm -> qkv_proj`` path was fused.
Fusing the ``post_attention_layernorm -> mlp`` path to FP4 was
rejected (the normed output feeds both the BF16 router gate AND the
FP4 experts — a FP4-only materialisation would require a second
dequant pass, net neutral).  W8 addresses this differently: a BF16-
out fused norm+router-gate kernel (Rank 5 of the T2-N ceiling
analysis).  The normed BF16 output still materialises to gmem for
the downstream FP4 quant path, but the router-gate GEMM is folded
into the norm kernel's epilogue — removing 2 of 3 launches per
layer without doubling FP4 work.

Entry point registration (via dist-info injected by
``launch_qwen3_fused_norm_fp4.sh``):

    [vllm.general_plugins]
    fused_norm_fp4_qwen3 = wire_fused_norm_fp4_qwen3_v2:register

Disable at runtime with ``AUTOKERNEL_FUSED_NORM_FP4_QWEN3=0``.
"""

from __future__ import annotations

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_SO_PATH = os.environ.get(
    "AUTOKERNEL_FUSED_NORM_FP4_SO",
    "/autokernel/workspace/fused_rms_norm_fp4_cu13.so",
)


def _load_shared_lib() -> bool:
    """Load the fused kernel .so into torch.ops._C.

    Returns True on success. Idempotent: safe to call if ops are already
    present.
    """
    if hasattr(torch.ops, "_C") and hasattr(
        torch.ops._C, "rms_norm_dynamic_fp4_quant"
    ):
        return True
    if not os.path.exists(_SO_PATH):
        logger.error("[fused_norm_fp4_qwen3] .so not found at %s", _SO_PATH)
        return False
    try:
        torch.ops.load_library(_SO_PATH)
    except Exception as e:
        logger.error("[fused_norm_fp4_qwen3] load_library(%s) failed: %s",
                     _SO_PATH, e)
        return False
    # Register fake-tensor stubs so torch.compile / piecewise capture still
    # works (inputs/outputs are allocated before the op runs so we can no-op).
    try:
        @torch.library.register_fake("_C::rms_norm_dynamic_fp4_quant")
        def _fake1(result, result_scale, input, weight, input_global_scale,
                   epsilon, is_sf_swizzled_layout):
            return None

        @torch.library.register_fake("_C::fused_add_rms_norm_dynamic_fp4_quant")
        def _fake2(result, result_scale, input, weight, residual,
                   input_global_scale, epsilon, is_sf_swizzled_layout):
            return None
    except Exception:
        # Already registered or torch version without register_fake.
        pass
    return True


def _build_fused_qkv_fn(layer, max_num_tokens: int = 512):
    """Build a per-decoder-layer fused ``input_layernorm -> qkv_proj`` callable.

    Returns None if the qkv_proj is not NVFP4-quantised (e.g. running an
    unquantised checkpoint for debugging).

    ``max_num_tokens`` controls the pre-allocated activation buffer size.  It
    must be >= the largest batch that will ever be run (== max_num_seqs for
    decode-only workloads).  The default 512 matches ``--max-num-seqs 512``.

    FIX (v2): dispatch on ``quant_method.backend`` (NvFp4LinearBackend enum),
    not ``quant_method.kernel`` (kernel object not present on
    ModelOptNvFp4LinearMethod).  Mirrors the dispatch used by the working
    Gemma4 plugin (fused_norm_fp4_integration.py lines 88-102).
    """
    from vllm._custom_ops import create_fp4_output_tensors
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        NvFp4LinearBackend,
        pad_nvfp4_activation_for_cutlass,
    )

    qkv_proj = layer.self_attn.qkv_proj
    quant_method = getattr(qkv_proj, "quant_method", None)
    if quant_method is None:
        logger.debug("[fused_norm_fp4_qwen3] qkv_proj has no quant_method — skipping")
        return None

    # --- FIX: use .backend (NvFp4LinearBackend enum), not .kernel ---
    # v1 checked quant_method.kernel which is absent on ModelOptNvFp4LinearMethod
    # causing silent None return and 100% fallback to unfused path.
    backend = getattr(quant_method, "backend", None)
    if backend is None:
        logger.debug("[fused_norm_fp4_qwen3] quant_method has no .backend — skipping")
        return None

    # Marlin uses a different (shuffled/packed) activation layout — not compatible.
    if backend == NvFp4LinearBackend.MARLIN:
        logger.debug("[fused_norm_fp4_qwen3] Marlin backend — skipping fusion")
        return None

    # Select matmul backend using the same dispatch logic as Gemma4 plugin.
    if backend.value.startswith("flashinfer-"):
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
        # Strip the "flashinfer-" prefix to get the sub-backend name
        # (e.g. "flashinfer-cutlass" -> "cutlass", "flashinfer-cudnn" -> "cudnn").
        sub_backend = backend.value[len("flashinfer-"):]

        def _do_mm(x_fp4, weight, x_sf, weight_scale, alpha, dtype):
            return flashinfer_scaled_fp4_mm(
                x_fp4, weight, x_sf, weight_scale, alpha, dtype,
                backend=sub_backend,
            )
    elif backend == NvFp4LinearBackend.FBGEMM:
        import fbgemm_gpu.experimental.gen_ai  # noqa: F401

        def _do_mm(x_fp4, weight, x_sf, weight_scale, alpha, dtype):
            return torch.ops.fbgemm.f4f4bf16_rowwise(x_fp4, weight, x_sf, weight_scale)
    else:
        # CutlassNvFp4 / default
        from vllm._custom_ops import cutlass_scaled_fp4_mm as _mm

        def _do_mm(x_fp4, weight, x_sf, weight_scale, alpha, dtype):
            return _mm(x_fp4, weight, x_sf, weight_scale, alpha, dtype)

    norm_weight = layer.input_layernorm.weight.data
    eps = layer.input_layernorm.variance_epsilon

    fused_op = torch.ops._C.rms_norm_dynamic_fp4_quant
    fused_add_op = torch.ops._C.fused_add_rms_norm_dynamic_fp4_quant

    # Capture weight tensors and scalars at closure-build time so they are NOT
    # re-fetched from qkv_proj on every forward call.
    _q_gs_inv = qkv_proj.input_global_scale_inv
    _weight_matrix = qkv_proj.weight
    _weight_scale = qkv_proj.weight_scale
    _alpha = qkv_proj.alpha
    _output_size = qkv_proj.output_size_per_partition
    _padding = getattr(qkv_proj, "weights_padding_cols", 0)

    # Pre-allocate activation output buffers for max_num_tokens so that each
    # forward call reuses static GPU memory instead of issuing torch.empty()
    # allocations, which break CUDA graph capture.
    _hidden_size = norm_weight.shape[0]
    _device = norm_weight.device
    _fp4_buf, _sf_buf = create_fp4_output_tensors(
        max_num_tokens, _hidden_size, _device, is_sf_swizzled_layout=True,
    )

    def _fused_norm_and_qkv(hidden_states, residual):
        """Replaces input_layernorm(x[, residual]) + qkv_proj.

        Returns (qkv_output_BF16, new_residual).  ``new_residual`` is the
        post-add residual (== hidden_states input when residual is None, else
        hidden_states + residual).
        """
        output_dtype = hidden_states.dtype

        x_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        output_shape = [*hidden_states.shape[:-1], _output_size]
        m = x_2d.shape[0]

        # Use the pre-allocated static buffers for the common decode path
        # (m <= max_num_tokens).  Fall back to a dynamic allocation for prefill
        # chunks that exceed the pre-allocated size so we never go out of bounds.
        if m <= max_num_tokens:
            sf_bytes = _sf_buf.view(torch.uint8)
            if residual is None:
                fused_op(
                    _fp4_buf, sf_bytes, x_2d, norm_weight, _q_gs_inv, eps,
                    True,
                )
                new_residual = hidden_states
            else:
                res_2d = residual.reshape(-1, residual.shape[-1])
                fused_add_op(
                    _fp4_buf, sf_bytes, x_2d, norm_weight, res_2d,
                    _q_gs_inv, eps, True,
                )
                new_residual = residual
            # Slice valid fp4/scale rows for the matmul.
            sf_rows = ((m + 127) // 128) * 128
            fp4_out = _fp4_buf[:m]
            sf = _sf_buf.view(torch.float8_e4m3fn)[:sf_rows]
        else:
            # Prefill path: allocate fresh buffers sized exactly for this m.
            fp4_dyn, sf_dyn = create_fp4_output_tensors(
                m, _hidden_size, _device, is_sf_swizzled_layout=True,
            )
            sf_bytes = sf_dyn.view(torch.uint8)
            if residual is None:
                fused_op(
                    fp4_dyn, sf_bytes, x_2d, norm_weight, _q_gs_inv, eps,
                    True,
                )
                new_residual = hidden_states
            else:
                res_2d = residual.reshape(-1, residual.shape[-1])
                fused_add_op(
                    fp4_dyn, sf_bytes, x_2d, norm_weight, res_2d,
                    _q_gs_inv, eps, True,
                )
                new_residual = residual
            fp4_out = fp4_dyn
            sf = sf_dyn.view(torch.float8_e4m3fn)

        if _padding > 0:
            fp4_out = pad_nvfp4_activation_for_cutlass(fp4_out, _padding)

        out = _do_mm(
            fp4_out, _weight_matrix, sf, _weight_scale, _alpha, output_dtype,
        )
        out = out[:, :_output_size]
        return out.view(*output_shape), new_residual

    return _fused_norm_and_qkv


def _build_fused_qk_rope_fn(attn):
    """Build a per-attention-module callable that fuses q_norm + k_norm + rotary.

    Returns a ``(q, k) -> (q, k)`` callable, or ``None`` if the expected
    attributes are missing (P1 silent-None detection: warn and fallthrough).

    The closure captures ``q_norm.weight``, ``k_norm.weight``, the rope
    ``cos_sin_cache``, the epsilon, and the head-dim-derived kernel constants
    so there is no per-forward attribute lookup.
    """
    # P1 hygiene: explicitly assert the attributes exist; warn+fallthrough if not.
    for attr in ("q_norm", "k_norm", "rotary_emb", "head_dim"):
        if not hasattr(attn, attr):
            logger.warning(
                "[fused_qknorm_rope] Qwen3MoeAttention missing .%s "
                "(class=%s) — falling back to stock path",
                attr, type(attn).__name__,
            )
            return None

    # Import the kernel.  Resolve via direct file load so it works whether
    # or not this plugin is installed as a package on sys.path.
    try:
        from kernels.triton.fused_qknorm_rope import fused_qknorm_rope
    except ImportError:
        import importlib.util
        import pathlib
        # patches/ sits alongside kernels/ in the autokernel repo; walk up.
        kpath = pathlib.Path(__file__).resolve().parent.parent / \
            "kernels" / "triton" / "fused_qknorm_rope.py"
        if not kpath.exists():
            logger.warning(
                "[fused_qknorm_rope] kernel file not found at %s — fallthrough",
                kpath,
            )
            return None
        spec = importlib.util.spec_from_file_location(
            "fused_qknorm_rope", str(kpath),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fused_qknorm_rope = mod.fused_qknorm_rope

    q_norm = attn.q_norm
    k_norm = attn.k_norm
    rope = attn.rotary_emb

    # Expected NEOX rotary on Qwen3; we only support the neox-style fused path.
    if not getattr(rope, "is_neox_style", True):
        logger.warning(
            "[fused_qknorm_rope] rotary_emb.is_neox_style=False — only "
            "neox supported; falling back",
        )
        return None

    head_dim = attn.head_dim
    if getattr(rope, "rotary_dim", head_dim) != head_dim:
        logger.warning(
            "[fused_qknorm_rope] rotary_dim=%s != head_dim=%s — partial "
            "rotary not supported; falling back",
            getattr(rope, "rotary_dim", None), head_dim,
        )
        return None

    eps = float(q_norm.variance_epsilon)
    # Sanity: q_norm and k_norm should share epsilon (they do in Qwen3).
    if abs(float(k_norm.variance_epsilon) - eps) > 1e-12:
        logger.warning(
            "[fused_qknorm_rope] q_norm.eps=%s != k_norm.eps=%s — "
            "falling back to stock path",
            eps, float(k_norm.variance_epsilon),
        )
        return None

    q_w = q_norm.weight.data
    k_w = k_norm.weight.data
    # cos_sin_cache is populated via register_buffer during RotaryEmbedding
    # __init__.  It may be BF16 or FP32 depending on config; the kernel
    # handles both (it loads to FP32 internally).
    if not hasattr(rope, "cos_sin_cache"):
        logger.warning(
            "[fused_qknorm_rope] rotary_emb has no cos_sin_cache — fallthrough",
        )
        return None
    cos_sin = rope.cos_sin_cache

    # --- Rank 4 extension: resolve KV cache layout + install no-op patch
    # on the Attention.impl class (once) so vLLM's reshape_and_cache_flash
    # becomes a cheap sentinel when our fused kernel fired. ---
    rank4_capable = False
    if _fused_kv_cache_update_enabled[0]:
        attn_layer = getattr(attn, "attn", None)
        if attn_layer is None:
            logger.warning(
                "[fused_kv_cache_update] attn has no .attn attr (class=%s) "
                "— Rank 4 disabled for this layer",
                type(attn).__name__,
            )
        elif not hasattr(attn_layer, "impl"):
            logger.warning(
                "[fused_kv_cache_update] attn.attn has no .impl — Rank 4 "
                "disabled for this layer",
            )
        else:
            kv_dtype = getattr(attn_layer, "kv_cache_dtype", "auto")
            if kv_dtype != "auto" and kv_dtype != "":
                # FP8 KV cache — our Triton kernel is BF16-only; defer to
                # stock path which handles the scale math.
                logger.info(
                    "[fused_kv_cache_update] kv_cache_dtype=%s != auto — "
                    "Rank 4 disabled (FP8 KV path uses stock scatter)",
                    kv_dtype,
                )
            else:
                _install_noop_kv_cache_update(type(attn_layer.impl))
                rank4_capable = True

    def _fused_qk_rope(q, k, positions, v=None, slot_mapping=None):
        """Invoke the fused Triton kernel.

        If ``v`` and ``slot_mapping`` are supplied AND Rank 4 is enabled
        for this layer, the kernel also scatter-writes rotated K and raw
        V into the paged KV cache.

        Returns True iff the Rank 4 path ran (so the caller knows to set
        ``attn.attn._autokernel_rank4_active`` for this step).
        """
        if rank4_capable and v is not None and slot_mapping is not None:
            attn_layer = attn.attn
            kv_cache = attn_layer.kv_cache
            # The cache tensor shape is (num_blocks, 2, block_size,
            # num_kv_heads, head_dim) in logical NHD ordering; physical
            # memory may be HND-permuted (see FlashInferBackend.
            # get_kv_cache_stride_order). Strides on key_cache /
            # value_cache carry the permutation correctly.
            key_cache = kv_cache[:, 0]
            value_cache = kv_cache[:, 1]
            fused_qknorm_rope(
                q, k, positions, q_w, k_w, cos_sin, eps, is_neox=True,
                v=v, key_cache=key_cache, value_cache=value_cache,
                slot_mapping=slot_mapping,
            )
            return True
        fused_qknorm_rope(
            q, k, positions, q_w, k_w, cos_sin, eps, is_neox=True,
        )
        return False

    # Expose rank4 capability on the attn so the decoder forward can
    # decide whether to fetch slot_mapping from forward_context.
    attn._autokernel_rank4_capable = rank4_capable
    return _fused_qk_rope


_fused_qk_rope_layer_count = [0]  # shared counter across layers (list for closure)
_fused_qk_rope_total_layers = [0]
_fused_qk_rope_banner_emitted = [False]
_fused_qk_rope_enabled = [True]

# ---- W8 Rank 4: KV-cache-update fusion ----
_fused_kv_cache_update_enabled = [False]
_fused_kv_cache_active_layer_count = [0]
_fused_kv_cache_total_layers_seen = [0]
_fused_kv_cache_banner_emitted = [False]
# Remember which Attention.impl classes we have patched so do_kv_cache_update
# skips when our Rank 4 path is active on that layer instance.
_patched_impl_classes: set = set()


def _install_noop_kv_cache_update(impl_cls):
    """Replace ``do_kv_cache_update`` on ``impl_cls`` with a method that
    skips the scatter when the attention layer's
    ``_autokernel_rank4_active`` flag is set. Otherwise delegates to the
    original method (BC preserved for other backends/layers).
    """
    if impl_cls in _patched_impl_classes:
        return
    orig = getattr(impl_cls, "do_kv_cache_update", None)
    if orig is None:
        logger.warning(
            "[fused_kv_cache_update] impl %s has no do_kv_cache_update — "
            "Rank 4 not installable on this backend",
            impl_cls.__name__,
        )
        return

    def patched(self, layer, key, value, kv_cache, slot_mapping):
        if getattr(layer, "_autokernel_rank4_active", False):
            # Our fused Triton kernel already wrote the scatter in this
            # step. Return without touching the cache again (avoids
            # double-write and, more importantly, eliminates the
            # reshape_and_cache_flash launch — the actual win).
            return None
        return orig(self, layer, key, value, kv_cache, slot_mapping)

    impl_cls.do_kv_cache_update = patched
    _patched_impl_classes.add(impl_cls)
    logger.info(
        "[fused_kv_cache_update] installed no-op do_kv_cache_update on %s",
        impl_cls.__name__,
    )


# ---- W8: post_attention_layernorm + router-gate fusion (Rank 5) ----
_fused_post_router_layer_count = [0]
_fused_post_router_total_layers = [0]
_fused_post_router_banner_emitted = [False]
_fused_post_router_enabled = [True]

# ---- W8 Rank 6: shared_expert FusedMLP collapse ----
_fused_shared_expert_enabled = [False]
_fused_shared_expert_layer_count = [0]
_fused_shared_expert_total_layers = [0]
_fused_shared_expert_banner_emitted = [False]
# Cache the resolved fused callable (None if unavailable).  Holding a
# single reference at module scope amortises the kernel import.
_fused_shared_expert_kernel = [None]


def _resolve_fused_shared_expert_kernel():
    """Import ``fused_shared_expert_mlp`` (cached).  Returns the callable
    or ``None`` if unavailable (e.g. Triton not installed).  P1 hygiene:
    explicit warn+fallthrough on any failure.
    """
    if _fused_shared_expert_kernel[0] is not None:
        return _fused_shared_expert_kernel[0]
    try:
        from kernels.triton.fused_shared_expert_mlp import (
            fused_shared_expert_mlp,
        )
    except ImportError:
        # Fallback: resolve via file path (plugin may not be on sys.path).
        try:
            import importlib.util
            import pathlib
            kpath = pathlib.Path(__file__).resolve().parent.parent / \
                "kernels" / "triton" / "fused_shared_expert_mlp.py"
            if not kpath.exists():
                logger.warning(
                    "[fused_shared_expert_mlp] kernel file not found at %s "
                    "— fallthrough",
                    kpath,
                )
                return None
            spec = importlib.util.spec_from_file_location(
                "fused_shared_expert_mlp", str(kpath),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fused_shared_expert_mlp = mod.fused_shared_expert_mlp
        except Exception as e:
            logger.warning(
                "[fused_shared_expert_mlp] import failed: %s — fallthrough", e,
            )
            return None
    _fused_shared_expert_kernel[0] = fused_shared_expert_mlp
    return fused_shared_expert_mlp


def _is_fused_shared_expert_capable(mlp) -> bool:
    """P1: check whether this ``Qwen3MoeMLP`` instance can use the fused
    path.  Requires:
      - an ``expert_gate`` attribute (True only on the shared_expert
        variant; stock dense MLPs have ``expert_gate=None`` or no attr);
      - BF16 unquantised gate_up_proj and down_proj weights exposed via
        ``.weight`` attributes.
    Returns ``False`` with a debug log for any missing prerequisite.
    """
    if not hasattr(mlp, "expert_gate") or mlp.expert_gate is None:
        # Not a shared_expert — leave stock path.
        return False
    for attr in ("gate_up_proj", "down_proj"):
        layer = getattr(mlp, attr, None)
        if layer is None:
            logger.debug(
                "[fused_shared_expert_mlp] mlp missing .%s — fallthrough",
                attr,
            )
            return False
        w = getattr(layer, "weight", None)
        if w is None:
            logger.debug(
                "[fused_shared_expert_mlp] %s has no .weight — fallthrough "
                "(quantised path not supported)", attr,
            )
            return False
        if w.dtype != torch.bfloat16:
            logger.debug(
                "[fused_shared_expert_mlp] %s.weight dtype=%s (expected bf16) "
                "— fallthrough", attr, w.dtype,
            )
            return False
    # The expert_gate itself must be a callable sub-module with a weight
    # we can read (for the sigmoid gating).  Stock ReplicatedLinear OK.
    if getattr(mlp.expert_gate, "weight", None) is None:
        logger.debug(
            "[fused_shared_expert_mlp] expert_gate has no .weight — "
            "fallthrough"
        )
        return False
    return True


def _build_fused_shared_expert_forward(mlp):
    """Build a closure replacing ``Qwen3MoeMLP.forward`` for ONE mlp
    instance (the shared_expert).  Returns None if any prerequisite is
    missing (P1 silent-None protection: caller must warn+fallthrough).

    The fused path collapses:
        gate_up = self.gate_up_proj(x)   [3 cuBLAS/launch hops in stock]
        out     = SiluAndMul(gate_up)
        out     = self.down_proj(out)
    into a single Triton kernel producing the [M, INTER] BF16
    intermediate followed by a cuBLAS down GEMM.  The outer
    ``expert_gate`` sigmoid multiply is applied exactly as in the stock
    ``Qwen3MoeMLP.forward`` (captured by-reference from the mlp
    instance).
    """
    kernel_fn = _resolve_fused_shared_expert_kernel()
    if kernel_fn is None:
        return None
    if not _is_fused_shared_expert_capable(mlp):
        return None

    # Weight tensors captured at closure build; read once, not per-call.
    gate_up_weight = mlp.gate_up_proj.weight
    down_weight = mlp.down_proj.weight
    expert_gate = mlp.expert_gate  # keep the live callable

    import torch.nn.functional as _F

    def _fused_forward(x):
        # Shape: x = [M, H] or [M, K, H] (latter reshape-normalised by kernel).
        out = kernel_fn(x, gate_up_weight, down_weight)
        # Apply expert_gate sigmoid multiply (stock path includes this
        # branch when self.expert_gate is not None).  We keep it outside
        # the fused kernel — the gate is a tiny [H, 1] BF16 matmul that
        # runs in parallel with the main kernel on SM120 and is
        # launch-overhead-negligible.  Importantly, BC is preserved even
        # for rare shared_experts where expert_gate might be None (not
        # the Qwen3-30B case, but defensive).
        eg = expert_gate(x)
        # expert_gate may return (tensor, None) from ReplicatedLinear, or
        # a bare tensor from a plain nn.Linear; handle both.
        if isinstance(eg, tuple):
            eg = eg[0]
        out = _F.sigmoid(eg) * out
        return out

    return _fused_forward


def _maybe_patch_shared_expert(mlp):
    """Install the fused forward on this mlp instance if enabled and
    capable.  Idempotent: safe to call on every first-forward invocation.
    P2 hygiene: track layer count + total so the banner distinguishes
    "patch active" from "plugin loaded".
    """
    if not _fused_shared_expert_enabled[0]:
        return
    if getattr(mlp, "_fused_shared_expert_installed", False):
        return
    mlp._fused_shared_expert_installed = True
    _fused_shared_expert_total_layers[0] += 1

    try:
        fused_fn = _build_fused_shared_expert_forward(mlp)
    except Exception as e:
        logger.warning(
            "[fused_shared_expert_mlp] build failed on mlp %s: %s — "
            "fallthrough", id(mlp), e,
        )
        return

    if fused_fn is None:
        # _build_... already logged the reason at debug level.  Leave
        # stock forward intact.
        return

    # Replace the bound forward on THIS instance only (other dense
    # Qwen3MoeMLPs — non-shared-expert — keep the stock path).
    import types
    mlp._orig_forward = mlp.forward
    mlp.forward = types.MethodType(lambda self, x: fused_fn(x), mlp)
    _fused_shared_expert_layer_count[0] += 1

    # P2 banner: emit once when we have seen all MoE layers (Qwen3-30B = 48
    # layers, but some configurations have dense-MLP layers interleaved —
    # total_layers tracks only blocks that have a shared_expert attribute
    # set).  We defer to the same threshold used by other Rank banners.
    if (not _fused_shared_expert_banner_emitted[0]
            and _fused_shared_expert_total_layers[0] >= 48):
        logger.info(
            "[fused_shared_expert_mlp] active layers=%d/%d",
            _fused_shared_expert_layer_count[0],
            _fused_shared_expert_total_layers[0],
        )
        print(
            f"[fused_shared_expert_mlp] active layers="
            f"{_fused_shared_expert_layer_count[0]}/"
            f"{_fused_shared_expert_total_layers[0]}",
            flush=True,
        )
        _fused_shared_expert_banner_emitted[0] = True


def _build_fused_post_attn_router_fn(layer, max_num_tokens: int = 512):
    """Build a per-decoder-layer fused ``post_attention_layernorm + mlp.gate`` callable.

    Returns a ``(hidden_states, residual) -> (hidden_normed, new_residual,
    router_logits)`` callable, or ``None`` when the layer is not an MoE
    block (dense-MLP layers still use the stock path) or when the gate
    weight is unavailable (P1: warn + fallthrough).

    ``max_num_tokens`` controls the size of the pre-allocated output
    buffers so steady-state decode reuses static GPU memory (CUDA graph
    friendly).  Prefill chunks larger than this allocate fresh.
    """
    # Import the kernel (resolving via file path as a fallback).
    try:
        from kernels.triton.fused_post_attn_router import (
            fused_post_attn_norm_router_gate,
        )
    except ImportError:
        import importlib.util
        import pathlib
        kpath = pathlib.Path(__file__).resolve().parent.parent / \
            "kernels" / "triton" / "fused_post_attn_router.py"
        if not kpath.exists():
            logger.warning(
                "[fused_post_attn_router] kernel file not found at %s — fallthrough",
                kpath,
            )
            return None
        spec = importlib.util.spec_from_file_location(
            "fused_post_attn_router", str(kpath),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fused_post_attn_norm_router_gate = mod.fused_post_attn_norm_router_gate

    # Only MoE layers have mlp.gate (ReplicatedLinear router).
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        logger.warning(
            "[fused_post_attn_router] layer missing .mlp — fallthrough",
        )
        return None
    gate = getattr(mlp, "gate", None)
    if gate is None:
        # Dense-MLP layer (no router); stock path applies.
        logger.debug(
            "[fused_post_attn_router] layer has no mlp.gate (dense MLP) — fallthrough",
        )
        return None

    # P1: explicit attribute assertion + class logging.
    if not hasattr(gate, "weight"):
        logger.warning(
            "[fused_post_attn_router] mlp.gate missing .weight "
            "(class=%s) — falling back to stock path",
            type(gate).__name__,
        )
        return None

    post_norm = getattr(layer, "post_attention_layernorm", None)
    if post_norm is None or not hasattr(post_norm, "weight"):
        logger.warning(
            "[fused_post_attn_router] post_attention_layernorm missing "
            "or has no .weight (class=%s) — fallthrough",
            type(post_norm).__name__ if post_norm is not None else None,
        )
        return None

    gate_weight = gate.weight.data
    norm_weight = post_norm.weight.data
    eps = float(post_norm.variance_epsilon)

    # Bias: Qwen3 router gate has bias=False (config-driven). If a bias
    # ever appears, we must add it to router_logits; handle here.
    gate_bias = getattr(gate, "bias", None)
    if gate_bias is not None:
        logger.warning(
            "[fused_post_attn_router] mlp.gate has bias — not supported "
            "by the fused kernel; falling back to stock path",
        )
        return None

    # Sanity: weight shape is [E, H].
    if gate_weight.dim() != 2:
        logger.warning(
            "[fused_post_attn_router] unexpected gate.weight shape %s "
            "— fallthrough", tuple(gate_weight.shape),
        )
        return None
    E, H = gate_weight.shape
    if norm_weight.shape[0] != H:
        logger.warning(
            "[fused_post_attn_router] norm_weight/gate_weight H mismatch "
            "(%s vs %s) — fallthrough", norm_weight.shape[0], H,
        )
        return None

    device = gate_weight.device
    # Pre-allocate output buffers for the steady-state decode regime.
    _hidden_buf = torch.empty(
        (max_num_tokens, H), dtype=torch.bfloat16, device=device,
    )
    _residual_buf = torch.empty(
        (max_num_tokens, H), dtype=torch.bfloat16, device=device,
    )
    _logits_buf = torch.empty(
        (max_num_tokens, E), dtype=torch.bfloat16, device=device,
    )

    def _fused_post_attn_router(hidden_states, residual):
        """Replaces post_attention_layernorm(x, r) + mlp.gate(x).

        Returns (hidden_normed, new_residual, router_logits).
        new_residual == (hidden_states + residual), matching stock
        RMSNorm.forward_static semantics exactly.
        """
        # Flatten to [M, H] for the kernel; caller passes either 2D or
        # 3D (squeezed to 2D inside the MoE block).
        if hidden_states.dim() == 3:
            orig_shape = hidden_states.shape
            x_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            r_2d = residual.reshape(-1, residual.shape[-1])
        else:
            orig_shape = None
            x_2d = hidden_states
            r_2d = residual
        m = x_2d.shape[0]

        if m <= max_num_tokens:
            ho = _hidden_buf[:m]
            ro = _residual_buf[:m]
            lo = _logits_buf[:m]
            fused_post_attn_norm_router_gate(
                x_2d, r_2d, norm_weight, gate_weight, eps,
                router_logits_out=lo,
                hidden_out=ho,
                residual_out=ro,
            )
        else:
            # Prefill chunk larger than the pre-alloc; allocate fresh.
            ho, ro, lo = fused_post_attn_norm_router_gate(
                x_2d, r_2d, norm_weight, gate_weight, eps,
            )

        if orig_shape is not None:
            ho = ho.view(orig_shape)
            ro = ro.view(orig_shape)
        return ho, ro, lo

    return _fused_post_attn_router


def _patched_decoder_forward(self, positions, hidden_states, residual):
    """Drop-in replacement for Qwen3MoeDecoderLayer.forward.

    Fuses ``input_layernorm + qkv_proj``; leaves the MLP/MoE path untouched.
    """
    # Lazy build the fused callable on first forward (weights are live now).
    if not hasattr(self, "_fused_qkv_fn"):
        try:
            _max_toks = int(os.environ.get(
                "AUTOKERNEL_FUSED_NORM_MAX_TOKENS", "512"))
            self._fused_qkv_fn = _build_fused_qkv_fn(self, _max_toks)
            if self._fused_qkv_fn is not None:
                logger.info(
                    "[fused_norm_fp4_qwen3] layer %s: fused callable built OK",
                    id(self),
                )
            else:
                logger.warning(
                    "[fused_norm_fp4_qwen3] layer %s: _build_fused_qkv_fn "
                    "returned None — falling back to unfused path", id(self),
                )
        except Exception as e:
            logger.warning(
                "[fused_norm_fp4_qwen3] _build_fused_qkv_fn failed: %s", e)
            self._fused_qkv_fn = None

    attn = self.self_attn

    # Lazy-build the fused qknorm+rope callable (once per attention module).
    if _fused_qk_rope_enabled[0] and not hasattr(attn, "_fused_qk_rope_fn"):
        try:
            attn._fused_qk_rope_fn = _build_fused_qk_rope_fn(attn)
        except Exception as e:
            logger.warning(
                "[fused_qknorm_rope] build failed on layer %s: %s — fallthrough",
                id(self), e,
            )
            attn._fused_qk_rope_fn = None
        _fused_qk_rope_total_layers[0] += 1
        if attn._fused_qk_rope_fn is not None:
            _fused_qk_rope_layer_count[0] += 1

    if self._fused_qkv_fn is not None:
        qkv, residual = self._fused_qkv_fn(hidden_states, residual)
        q, k, v = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1,
        )

        qk_rope_fn = getattr(attn, "_fused_qk_rope_fn", None)
        if qk_rope_fn is not None:
            # Fused path: single Triton kernel does q_norm + k_norm + rope.
            # Reshape to per-head for the kernel, which mutates in place.
            q_view = q.view(
                *q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim,
            )
            k_view = k.view(
                *k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim,
            )
            # --- Rank 4: fetch slot_mapping for this layer and pass V so
            # the kernel additionally scatter-writes into the paged cache.
            rank4_v = None
            rank4_slot = None
            rank4_attn_layer = None
            if getattr(attn, "_autokernel_rank4_capable", False):
                try:
                    from vllm.forward_context import get_forward_context
                    fctx = get_forward_context()
                    slot_dict = getattr(fctx, "slot_mapping", None)
                    if isinstance(slot_dict, dict):
                        rank4_attn_layer = attn.attn
                        layer_name = getattr(rank4_attn_layer, "layer_name", None)
                        rank4_slot = slot_dict.get(layer_name) if layer_name else None
                    if rank4_slot is not None:
                        # Reshape V like K for the scatter kernel.
                        rank4_v = v.view(
                            *v.shape[:-1],
                            v.shape[-1] // attn.head_dim,
                            attn.head_dim,
                        )
                except Exception as e:
                    logger.debug(
                        "[fused_kv_cache_update] could not fetch slot_mapping "
                        "(layer %s): %s — falling back to stock scatter",
                        id(self), e,
                    )
                    rank4_slot = None
                    rank4_v = None
            did_rank4 = qk_rope_fn(
                q_view, k_view, positions,
                v=rank4_v, slot_mapping=rank4_slot,
            )
            # P2 banner: count active Rank 4 layers on first forward.
            if did_rank4 and not _fused_kv_cache_banner_emitted[0]:
                _fused_kv_cache_active_layer_count[0] += 1
            elif getattr(attn, "_autokernel_rank4_capable", False):
                # Rank 4 was wired but this call didn't fire (e.g.
                # profiling pass with empty slot_mapping).
                pass
            _fused_kv_cache_total_layers_seen[0] += 1
            # Signal the patched impl.do_kv_cache_update to skip on this
            # step if we wrote the scatter already.
            if rank4_attn_layer is not None:
                rank4_attn_layer._autokernel_rank4_active = did_rank4
            # P2 banner emission (once).
            if (_fused_kv_cache_update_enabled[0]
                    and not _fused_kv_cache_banner_emitted[0]
                    and _fused_kv_cache_total_layers_seen[0] >= 48):
                msg = (
                    f"[fused_kv_cache_update] active layers="
                    f"{_fused_kv_cache_active_layer_count[0]}/"
                    f"{_fused_kv_cache_total_layers_seen[0]}"
                )
                logger.info(msg)
                print(msg, flush=True)
                _fused_kv_cache_banner_emitted[0] = True
            # q_view and k_view alias q, k — already updated in place.
        else:
            # Stock path: 3 separate kernels.
            q_by_head = q.view(
                *q.shape[:-1], q.shape[-1] // attn.head_dim, attn.head_dim,
            )
            q_by_head = attn.q_norm(q_by_head)
            q = q_by_head.view(q.shape)

            k_by_head = k.view(
                *k.shape[:-1], k.shape[-1] // attn.head_dim, attn.head_dim,
            )
            k_by_head = attn.k_norm(k_by_head)
            k = k_by_head.view(k.shape)

            q, k = attn.rotary_emb(positions, q, k)

        # P2 banner: emit once after all layers have been visited on the
        # first forward.  This distinguishes "plugin active" (decoder forward
        # patched) from "fusion actually built on every layer" (count /
        # total).
        if (not _fused_qk_rope_banner_emitted[0]
                and _fused_qk_rope_total_layers[0] >= 48):
            logger.info(
                "[fused_qknorm_rope] active layers=%d/%d",
                _fused_qk_rope_layer_count[0],
                _fused_qk_rope_total_layers[0],
            )
            print(
                f"[fused_qknorm_rope] active layers="
                f"{_fused_qk_rope_layer_count[0]}/"
                f"{_fused_qk_rope_total_layers[0]}",
                flush=True,
            )
            _fused_qk_rope_banner_emitted[0] = True

        attn_output = attn.attn(q, k, v)
        hidden_states, _ = attn.o_proj(attn_output)
    else:
        # Fallback to the stock path.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual,
            )
        hidden_states = attn(positions=positions, hidden_states=hidden_states)

    # ---- W8 Rank 5: post_attention_layernorm + router-gate fusion ----
    # Lazy-build the fused callable on first forward (weights are live).
    if (_fused_post_router_enabled[0]
            and not hasattr(self, "_fused_post_router_fn")):
        try:
            _max_toks = int(os.environ.get(
                "AUTOKERNEL_FUSED_NORM_MAX_TOKENS", "512"))
            self._fused_post_router_fn = _build_fused_post_attn_router_fn(
                self, _max_toks,
            )
        except Exception as e:
            logger.warning(
                "[fused_post_attn_router] build failed on layer %s: %s "
                "— fallthrough",
                id(self), e,
            )
            self._fused_post_router_fn = None
        _fused_post_router_total_layers[0] += 1
        if self._fused_post_router_fn is not None:
            _fused_post_router_layer_count[0] += 1

    post_router_fn = getattr(self, "_fused_post_router_fn", None)
    if post_router_fn is not None:
        # Fused path: one Triton kernel produces (hidden_normed, residual,
        # router_logits); hand router_logits to the MoE block via a
        # per-instance ``_prerouted_logits`` attribute so the patched
        # Qwen3MoeSparseMoeBlock.forward skips its own ``self.gate(...)``.
        h_flat = hidden_states
        r_flat = residual
        hidden_states, residual, router_logits = post_router_fn(
            h_flat, r_flat,
        )
        mlp = self.mlp
        # Stash for the patched MoE forward (see _patched_moe_forward).
        mlp._prerouted_logits = router_logits
        try:
            hidden_states = mlp(hidden_states)
        finally:
            # Always clear so a later non-fused call doesn't reuse stale
            # logits (and to avoid retaining the buffer across steps).
            mlp._prerouted_logits = None
    else:
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = self.mlp(hidden_states)

    # P2 banner (post-attn-router): emit once when all layers visited.
    if (_fused_post_router_enabled[0]
            and not _fused_post_router_banner_emitted[0]
            and _fused_post_router_total_layers[0] >= 48):
        logger.info(
            "[fused_post_attn_router] active layers=%d/%d",
            _fused_post_router_layer_count[0],
            _fused_post_router_total_layers[0],
        )
        print(
            f"[fused_post_attn_router] active layers="
            f"{_fused_post_router_layer_count[0]}/"
            f"{_fused_post_router_total_layers[0]}",
            flush=True,
        )
        _fused_post_router_banner_emitted[0] = True

    return hidden_states, residual


def _patched_moe_forward(self, hidden_states):
    """Drop-in replacement for ``Qwen3MoeSparseMoeBlock.forward`` that
    honours a precomputed ``_prerouted_logits`` attribute set by the
    fused post-attn-router path.

    When ``self._prerouted_logits`` is ``None`` this reproduces the
    stock forward verbatim.  When set, it skips the ``self.gate(...)``
    call and uses the stashed logits instead.
    """
    from vllm.distributed import tensor_model_parallel_all_gather
    try:
        from vllm.model_executor.models.utils import sequence_parallel_chunk
    except ImportError:
        sequence_parallel_chunk = None

    assert hidden_states.dim() <= 2, (
        "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
    )
    is_input_1d = hidden_states.dim() == 1
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if self.is_sequence_parallel and sequence_parallel_chunk is not None:
        hidden_states = sequence_parallel_chunk(hidden_states)

    # W8 Rank 6: lazy-install the fused shared_expert forward on first
    # forward.  Idempotent + P2-safe (see _maybe_patch_shared_expert
    # for banner counting).  We install on the mlp-block's
    # ``shared_expert`` attribute — the Qwen3MoeMLP instance that the
    # MoE dispatch calls in parallel with the routed experts.  Safe
    # when shared_expert is None (early return inside the helper).
    if _fused_shared_expert_enabled[0]:
        shared_mlp = getattr(self, "shared_expert", None)
        if shared_mlp is not None:
            _maybe_patch_shared_expert(shared_mlp)

    prerouted = getattr(self, "_prerouted_logits", None)
    if prerouted is not None:
        # Fused path: reuse precomputed router logits. Slice to num_tokens
        # in case the fused kernel wrote a larger pre-allocated buffer.
        router_logits = prerouted[: hidden_states.shape[0]]
    else:
        router_logits, _ = self.gate(hidden_states)

    shared_out, fused_out = self.experts(
        hidden_states=hidden_states, router_logits=router_logits,
    )
    final_hidden_states = (
        shared_out + fused_out if shared_out is not None else fused_out
    )

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(
            final_hidden_states, 0,
        )
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(  # noqa E501
            final_hidden_states,
        )

    return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


_PATCHED = False


def register() -> None:
    """vLLM plugin entry point."""
    global _PATCHED

    if os.environ.get("AUTOKERNEL_FUSED_NORM_FP4_QWEN3", "1") == "0":
        msg = "[fused_norm_fp4_qwen3] disabled via env"
        logger.info(msg)
        print(msg, flush=True)
        return

    # W7: qknorm+rope fusion env gate (default ON when plugin is active).
    if os.environ.get("AUTOKERNEL_FUSED_QKNORM_ROPE", "1") == "0":
        _fused_qk_rope_enabled[0] = False
        msg = "[fused_qknorm_rope] disabled via env — using stock q_norm/k_norm/rotary_emb"
        logger.info(msg)
        print(msg, flush=True)
    else:
        msg = "[fused_qknorm_rope] enabled — Triton kernel will replace q_norm+k_norm+rotary_emb"
        logger.info(msg)
        print(msg, flush=True)

    # W8 Rank 4: KV-cache-update fusion env gate (default OFF; requires Rank 2).
    if os.environ.get("AUTOKERNEL_FUSED_KV_CACHE_UPDATE", "0") == "1":
        if not _fused_qk_rope_enabled[0]:
            msg = (
                "[fused_kv_cache_update] requires AUTOKERNEL_FUSED_QKNORM_ROPE=1 — "
                "disabled"
            )
            logger.warning(msg)
            print(msg, flush=True)
        else:
            _fused_kv_cache_update_enabled[0] = True
            msg = (
                "[fused_kv_cache_update] enabled — rotated K + raw V will be "
                "scatter-written into the paged KV cache by the fused Triton kernel "
                "(unified_kv_cache_update becomes a no-op sentinel)"
            )
            logger.info(msg)
            print(msg, flush=True)
    else:
        msg = "[fused_kv_cache_update] disabled (default) — stock unified_kv_cache_update fires"
        logger.info(msg)
        print(msg, flush=True)

    # W8: post-attn-norm + router-gate fusion env gate (Rank 5, default ON).
    if os.environ.get("AUTOKERNEL_FUSED_POST_ATTN_ROUTER", "1") == "0":
        _fused_post_router_enabled[0] = False
        msg = (
            "[fused_post_attn_router] disabled via env — using stock "
            "post_attention_layernorm + mlp.gate"
        )
        logger.info(msg)
        print(msg, flush=True)
    else:
        msg = (
            "[fused_post_attn_router] enabled — Triton kernel will replace "
            "post_attention_layernorm + mlp.gate (Rank 5)"
        )
        logger.info(msg)
        print(msg, flush=True)

    # W8 Rank 6: shared_expert FusedMLP collapse env gate (default OFF).
    # Default OFF because the microbench gain at M=1 decode is ~0-5%
    # (launch-count savings are partially offset by Triton's eager
    # launch overhead), and the big wins (+10-13%) are at M>=4 where
    # the real stock vLLM path is also cuBLAS-bound.  Enable explicitly
    # for end-to-end banked bench sweeps.
    if os.environ.get("AUTOKERNEL_FUSED_SHARED_EXPERT_MLP", "0") == "1":
        _fused_shared_expert_enabled[0] = True
        msg = (
            "[fused_shared_expert_mlp] enabled — Triton kernel will replace "
            "Qwen3MoeMLP.forward on the shared_expert (Rank 6)"
        )
        logger.info(msg)
        print(msg, flush=True)
    else:
        msg = "[fused_shared_expert_mlp] disabled (default) — stock 3-op path fires"
        logger.info(msg)
        print(msg, flush=True)

    if _PATCHED:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    if not _load_shared_lib():
        msg = "[fused_norm_fp4_qwen3] failed to load .so -- patch NOT applied"
        logger.error(msg)
        print(msg, flush=True)
        return

    try:
        from vllm.model_executor.models.qwen3_moe import (
            Qwen3MoeDecoderLayer,
            Qwen3MoeSparseMoeBlock,
        )
    except Exception as e:
        logger.error(
            "[fused_norm_fp4_qwen3] could not import Qwen3MoeDecoderLayer: %s",
            e,
        )
        return

    if getattr(Qwen3MoeDecoderLayer, "_fused_norm_fp4_patched", False):
        return

    Qwen3MoeDecoderLayer._orig_forward = Qwen3MoeDecoderLayer.forward
    Qwen3MoeDecoderLayer.forward = _patched_decoder_forward
    Qwen3MoeDecoderLayer._fused_norm_fp4_patched = True

    # W8 Rank 5: patch Qwen3MoeSparseMoeBlock.forward so it honours
    # ``_prerouted_logits`` when the fused decoder path precomputes them.
    # Safe even when AUTOKERNEL_FUSED_POST_ATTN_ROUTER=0 — without the
    # attribute set it falls back to stock self.gate(...) verbatim.
    if not getattr(Qwen3MoeSparseMoeBlock, "_fused_post_router_patched", False):
        Qwen3MoeSparseMoeBlock._orig_forward = Qwen3MoeSparseMoeBlock.forward
        Qwen3MoeSparseMoeBlock.forward = _patched_moe_forward
        Qwen3MoeSparseMoeBlock._fused_post_router_patched = True

    _PATCHED = True

    msg = (
        "[fused_norm_fp4_qwen3] Patched Qwen3MoeDecoderLayer.forward + "
        "Qwen3MoeSparseMoeBlock.forward via plugin v2: fused RMSNorm+FP4 "
        "for input_layernorm->qkv_proj active; fused qknorm+rope (W7) "
        "and post_attn_norm+router_gate (W8 Rank 5) active subject to "
        "their env gates"
    )
    logger.info(msg)
    print(msg, flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    register()
