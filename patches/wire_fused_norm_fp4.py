#!/usr/bin/env python3
"""
Wire fused RMSNorm+FP4Quant Triton kernel into live vLLM Gemma4 server.

TASK A: Fused RMSNorm + FP4 quantization with CUTLASS 128x4 swizzled scale output.
        Our Triton kernel (fused_norm_fp4_swizzled.py) combines norm + FP4 quant +
        scale swizzle in a single kernel launch, eliminating the intermediate bf16
        materialization between norm and quant.

        Microbenchmark: 2.3x speedup (31us fused vs 72us separate at m=1, n=2816).
        ~2% FP4 nibble rounding difference vs separate path (float32-throughout
        in fused vs bf16 intermediate in separate — fused is more precise).

        Integration status: Kernel + CUTLASS-compatible swizzle verified correct.
        End-to-end integration requires either:
        (a) Restructuring model forward to call fused kernel before linear layer, OR
        (b) vLLM compiler fusion pass recognizing FP4 quant patterns (infra added
            in vllm_fp4_norm_quant_fusion.py).
        Neither is production-ready due to CUDA graph compatibility concerns
        with Python-level dispatch. See apply_task_a() for infrastructure.

TASK B: Fused residual addition + RMSNorm using vLLM's built-in fused_add_rms_norm.
        Eliminates 48 separate elementwise addition kernels per forward pass by
        combining `hidden + residual` with the following pre_feedforward_layernorm.

        Measured end-to-end throughput improvement (Gemma4 26B NVFP4, RTX 5090,
        eager mode, no CUDA graphs):
          C=1:   28 tok/s vs 26 baseline (+8%)
          C=32:  543 tok/s vs 539 baseline (+1%)
          C=128: 2070 tok/s vs 2006 baseline (+3%)
          C=256: 3790 tok/s vs 3682 baseline (+3%)

Usage:
  # Apply file-level patch to gemma4.py inside Docker container, then restart:
  docker exec <container> python3 /patches/apply_gemma4_patch.py
  # Or for a fresh start:
  docker run --gpus all -v /patches:/patches vllm-built bash -c "
    python3 /patches/apply_gemma4_patch.py &&
    python3 -m vllm.entrypoints.openai.api_server --model ... --enforce-eager
  "

CUDA-graph safety:
  Task B uses vLLM's built-in C++ fused_add_rms_norm kernel. No Python
  conditionals in the hot path. Safe with CUDA graphs.

Files:
  patches/wire_fused_norm_fp4.py       - This file (documentation + Task A infra)
  patches/apply_gemma4_patch.py        - File-level patch for Task B
  patches/fused_norm_fp4_swizzled.py   - Task A fused Triton kernel
  kernels/fused_norm_fp4.py            - Original fused kernel (non-swizzled)
"""

import sys
import logging
import threading

import torch

logger = logging.getLogger(__name__)


def apply_task_b():
    """
    TASK B: Fuse residual addition into RMSNorm for Gemma4.

    Replaces:
        hidden = post_attention_layernorm(hidden)
        hidden = hidden + residual      # separate add kernel
        residual = hidden
        hidden = pre_feedforward_layernorm(hidden)

    With:
        hidden = post_attention_layernorm(hidden)
        hidden, residual = fused_add_rms_norm(hidden, residual, w, eps)

    The C++ fused_add_rms_norm kernel does:
      residual = x + residual   (in-place)
      x = rms_norm(residual)    (in-place)
    Eliminating the separate add kernel and extra memory traffic.

    This is applied via monkey-patching Gemma4DecoderLayer.forward.
    For production, use apply_gemma4_patch.py to modify the source file.
    """
    from vllm.model_executor.layers.layernorm import fused_add_rms_norm
    from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer

    _orig_forward = Gemma4DecoderLayer.forward

    def _fused_forward(self, positions, hidden_states, residual,
                       per_layer_input=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states, **kwargs,
        )

        # FUSED: post_attn_norm + add + pre_ff_norm
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, residual = fused_add_rms_norm(
            hidden_states, residual,
            self.pre_feedforward_layernorm.weight.data,
            self.pre_feedforward_layernorm.variance_epsilon,
        )

        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            router_logits = self.router(residual)
            hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
            hidden_states_2 = self.moe(hidden_states_2, router_logits)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        if per_layer_input is not None and self.per_layer_input_gate is not None:
            gate = self.per_layer_input_gate(hidden_states)
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
            gated_per_layer = gate * per_layer_input
            per_layer_contribution = self.per_layer_projection(gated_per_layer)
            per_layer_contribution = self.post_per_layer_input_norm(
                per_layer_contribution
            )
            hidden_states = hidden_states + per_layer_contribution

        hidden_states = hidden_states * self.layer_scalar
        return hidden_states, None

    Gemma4DecoderLayer.forward = _fused_forward
    logger.info("[PATCH] Task B applied: fused_add_rms_norm for pre_feedforward")


def apply_task_a():
    """
    TASK A: Infrastructure for fused RMSNorm + FP4 quant.

    Sets up:
    1. Thread-local storage for passing pre-computed FP4 data from norm to linear
    2. Patched ModelOptNvFp4LinearMethod.apply that checks for cached FP4

    The actual fused kernel call must be made from the model forward (not here)
    because the norm weights and global scale live on the model/layer objects.

    Note: This uses Python-level dispatch (thread-local check) which is NOT
    safe for CUDA graph capture. For CUDA graph mode, the model forward must
    be rewritten to unconditionally use the fused kernel.
    """
    sys.path.insert(0, '/patches')
    try:
        from fused_norm_fp4_swizzled import fused_rms_norm_fp4_quant_swizzled
    except ImportError:
        logger.warning("[PATCH] Task A: fused_norm_fp4_swizzled.py not found")
        return

    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        NvFp4LinearBackend,
    )
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptNvFp4LinearMethod,
    )
    from vllm._custom_ops import cutlass_scaled_fp4_mm

    _tls = threading.local()

    def set_cached_fp4(fp4, scales):
        _tls.cached_fp4 = fp4
        _tls.cached_scales = scales

    def get_and_clear_cached_fp4():
        fp4 = getattr(_tls, 'cached_fp4', None)
        scales = getattr(_tls, 'cached_scales', None)
        _tls.cached_fp4 = None
        _tls.cached_scales = None
        return fp4, scales

    _orig_apply = ModelOptNvFp4LinearMethod.apply

    def _patched_apply(self, layer, x, bias=None):
        fp4, scales = get_and_clear_cached_fp4()
        if fp4 is not None:
            weight = layer.weight
            weight_scale = layer.weight_scale
            alpha = layer.alpha
            output_size = layer.output_size_per_partition
            output_shape = [*x.shape[:-1], output_size]

            padding = getattr(layer, 'weights_padding_cols', 0)
            if padding > 0:
                from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
                    pad_nvfp4_activation_for_cutlass,
                )
                fp4 = pad_nvfp4_activation_for_cutlass(fp4, padding)

            out = cutlass_scaled_fp4_mm(
                fp4, weight, scales, weight_scale, alpha, x.dtype,
            )
            out = out[:, :output_size]
            if bias is not None:
                out = out + bias
            return out.view(*output_shape)

        return _orig_apply(self, layer, x, bias)

    ModelOptNvFp4LinearMethod.apply = _patched_apply

    # Export for model forward patches to use
    apply_task_a.fused_kernel = fused_rms_norm_fp4_quant_swizzled
    apply_task_a.set_cached_fp4 = set_cached_fp4

    logger.info("[PATCH] Task A infra applied: cached FP4 dispatch in NvFp4 linear")


def apply_all():
    """Apply all safe patches (Task B only by default)."""
    apply_task_b()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    apply_all()
    print("Patches applied.")
