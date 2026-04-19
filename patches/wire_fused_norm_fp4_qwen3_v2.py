"""vLLM plugin: fused RMSNorm + FP4-quant wiring for Qwen3 MoE.  (v2 — fixed)

Monkey-patches ``vllm.model_executor.models.qwen3_moe.Qwen3MoeDecoderLayer``
so the ``input_layernorm -> self_attn.qkv_proj`` path runs a single fused
kernel (``torch.ops._C.rms_norm_dynamic_fp4_quant`` /
``fused_add_rms_norm_dynamic_fp4_quant``) instead of:

    rms_norm(x[, residual]) -> scaled_fp4_quant -> cutlass_scaled_fp4_mm

The fused kernel is provided by ``workspace/fused_rms_norm_fp4_cu13.so``.

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

Only the ``input_layernorm -> qkv_proj`` path is fused.  The
``post_attention_layernorm -> mlp`` path is NOT fused because the normed
output feeds both the BF16 router gate and the FP4 experts — materialising
FP4-only would require a second dequant pass, net neutral.

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

    if self._fused_qkv_fn is not None:
        qkv, residual = self._fused_qkv_fn(hidden_states, residual)
        q, k, v = qkv.split(
            [attn.q_size, attn.kv_size, attn.kv_size], dim=-1,
        )
        # Remainder of Qwen3MoeAttention.forward, inline so we can hand the
        # pre-fused qkv through.
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

    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual,
    )
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


_PATCHED = False


def register() -> None:
    """vLLM plugin entry point."""
    global _PATCHED

    if os.environ.get("AUTOKERNEL_FUSED_NORM_FP4_QWEN3", "1") == "0":
        msg = "[fused_norm_fp4_qwen3] disabled via env"
        logger.info(msg)
        print(msg, flush=True)
        return

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
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
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
    _PATCHED = True

    msg = (
        "[fused_norm_fp4_qwen3] Patched Qwen3MoeDecoderLayer.forward via "
        "plugin v2: fused RMSNorm+FP4 for input_layernorm->qkv_proj active"
    )
    logger.info(msg)
    print(msg, flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    register()
