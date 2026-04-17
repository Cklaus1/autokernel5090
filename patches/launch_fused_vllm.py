#!/usr/bin/env python3
"""
Launch vLLM with fused RMSNorm+FP4 kernel.

This script:
1. Loads the fused kernel .so into the _C namespace
2. Registers fake tensor implementations for torch.compile
3. Patches gemma4.py source to use fused norm+quant
4. Patches apply_nvfp4_linear to support fused path
5. Delegates to vLLM's api_server

The patches are applied at the source level (before import) so they
propagate to all subprocesses (EngineCore, workers).

Usage:
    python3 /patches/launch_fused_vllm.py --model ... [vllm args]
"""

import os
import sys
import logging
import importlib

logger = logging.getLogger("fused_launcher")
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

# ============================================================
# Step 1: Load the fused kernel .so
# ============================================================
SO_PATH = os.environ.get("FUSED_KERNEL_SO", "/tmp/fused_kernel/fused_rms_norm_fp4.so")
FUSED_AVAILABLE = False

if os.path.exists(SO_PATH):
    import torch
    try:
        torch.ops.load_library(SO_PATH)
        FUSED_AVAILABLE = True
        logger.info("Loaded fused kernel from %s", SO_PATH)

        # Register fake tensor impls
        @torch.library.register_fake("_C::rms_norm_dynamic_fp4_quant")
        def _fake1(result, result_scale, input, weight, input_global_scale,
                   epsilon, is_sf_swizzled_layout):
            pass

        @torch.library.register_fake("_C::fused_add_rms_norm_dynamic_fp4_quant")
        def _fake2(result, result_scale, input, weight, residual,
                   input_global_scale, epsilon, is_sf_swizzled_layout):
            pass

        logger.info("Registered fake tensor implementations")
    except Exception as e:
        logger.warning("Failed to load fused kernel: %s", e)
else:
    logger.warning("Fused kernel .so not found at %s", SO_PATH)

# ============================================================
# Step 2: Patch gemma4.py to use fused kernel
# ============================================================
if FUSED_AVAILABLE:
    GEMMA4_PATH = "/build/vllm/vllm/model_executor/models/gemma4.py"

    if os.path.exists(GEMMA4_PATH):
        with open(GEMMA4_PATH) as f:
            source = f.read()

        # Only patch if not already patched
        if "FUSED_NORM_FP4" not in source:
            # Add imports and fused functions after the existing imports
            FUSED_CODE = '''
# === FUSED_NORM_FP4 PATCH START ===
import os as _os_fused
import torch as _torch_fused
from vllm._custom_ops import create_fp4_output_tensors as _create_fp4_out
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass as _pad_fp4,
)

# Load the fused kernel .so if available (needed in EngineCore subprocess)
_FUSED_SO = _os_fused.environ.get("FUSED_KERNEL_SO", "/tmp/fused_kernel/fused_rms_norm_fp4.so")
if _os_fused.path.exists(_FUSED_SO) and not hasattr(_torch_fused.ops._C, "rms_norm_dynamic_fp4_quant"):
    try:
        _torch_fused.ops.load_library(_FUSED_SO)
    except Exception:
        pass
_FUSED_OP = getattr(_torch_fused.ops._C, "rms_norm_dynamic_fp4_quant", None)
if _FUSED_OP is not None:
    try:
        @_torch_fused.library.register_fake("_C::rms_norm_dynamic_fp4_quant")
        def _fake_rnfq(result, result_scale, input, weight, input_global_scale, epsilon, is_sf_swizzled_layout):
            pass
        @_torch_fused.library.register_fake("_C::fused_add_rms_norm_dynamic_fp4_quant")
        def _fake_farnfq(result, result_scale, input, weight, residual, input_global_scale, epsilon, is_sf_swizzled_layout):
            pass
    except Exception:
        pass  # Already registered or not needed

def _fused_norm_quant(x_2d, weight, eps, input_global_scale):
    """Fused RMSNorm + FP4 quantization in a single kernel."""
    m, n = x_2d.shape
    fp4_out, scale_out = _create_fp4_out(m, n, x_2d.device, True)
    scale_bytes = scale_out.view(_torch_fused.uint8)
    _FUSED_OP(fp4_out, scale_bytes, x_2d, weight, input_global_scale, eps, True)
    return fp4_out, scale_out.view(_torch_fused.float8_e4m3fn)


def _fused_linear_forward(linear_layer, x_prenorm, norm_weight, norm_eps):
    """
    Fused norm+quant+matmul: replaces rms_norm(x) -> scaled_fp4_quant -> cutlass_mm
    with: fused_norm_quant(x) -> cutlass_mm
    """
    from vllm._custom_ops import cutlass_scaled_fp4_mm

    weight = linear_layer.weight
    weight_scale = linear_layer.weight_scale
    alpha = linear_layer.alpha
    input_gs_inv = linear_layer.input_global_scale_inv
    output_size = linear_layer.output_size_per_partition
    output_dtype = x_prenorm.dtype
    output_shape = [*x_prenorm.shape[:-1], output_size]

    x_2d = x_prenorm.reshape(-1, x_prenorm.shape[-1])
    x_fp4, x_sf = _fused_norm_quant(x_2d, norm_weight, norm_eps, input_gs_inv)

    padding = getattr(linear_layer, "weights_padding_cols", 0)
    x_fp4 = _pad_fp4(x_fp4, padding)

    out = cutlass_scaled_fp4_mm(x_fp4, weight, x_sf, weight_scale, alpha, output_dtype)
    out = out[:, :output_size]
    return out.view(*output_shape)

# === FUSED_NORM_FP4 PATCH END ===
'''
            # Insert fused code after imports (before first class def)
            insert_pos = source.find("\nclass Gemma4MLP")
            if insert_pos == -1:
                logger.error("Could not find Gemma4MLP class in gemma4.py")
            else:
                source = source[:insert_pos] + FUSED_CODE + source[insert_pos:]

                # Now patch the Gemma4DecoderLayer.forward method
                # Find the original forward and replace the attention path
                OLD_ATTN = '''        residual = hidden_states

        hidden_states = self.input_layernorm(residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )'''

                NEW_ATTN = '''        residual = hidden_states

        if _FUSED_OP is not None and hasattr(self.self_attn.qkv_proj, 'quant_method') and hasattr(self.self_attn.qkv_proj.quant_method, 'backend'):
            # FUSED: input_layernorm + qkv_proj.scaled_fp4_quant in one kernel
            _norm_w = self.input_layernorm.weight.data
            _norm_eps = self.input_layernorm.variance_epsilon
            qkv = _fused_linear_forward(
                self.self_attn.qkv_proj, residual, _norm_w, _norm_eps)
            q, k, v = qkv.split(
                [self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size],
                dim=-1)
            q = q.unflatten(-1, (self.self_attn.num_heads, self.self_attn.head_dim))
            q = self.self_attn.q_norm(q)
            q = q.flatten(-2, -1)
            if not self.self_attn.is_kv_shared_layer:
                k = k.unflatten(-1, (self.self_attn.num_kv_heads, self.self_attn.head_dim))
                k = self.self_attn.k_norm(k)
                k = k.flatten(-2, -1)
                q, k = self.self_attn.rotary_emb(positions, q, k)
                v = v.unflatten(-1, (self.self_attn.num_kv_heads, self.self_attn.head_dim))
                v = self.self_attn.v_norm(v)
                v = v.flatten(-2, -1)
            else:
                q = self.self_attn.rotary_emb(positions, q, k)[0]
            attn_output = self.self_attn.attn(q, k, v)
            hidden_states, _ = self.self_attn.o_proj(attn_output)
        else:
            hidden_states = self.input_layernorm(residual)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                **kwargs,
            )'''

                if OLD_ATTN in source:
                    source = source.replace(OLD_ATTN, NEW_ATTN, 1)
                    logger.info("Patched attention path with fused norm+quant")
                else:
                    logger.warning("Could not find attention path to patch")

                # Patch MLP path
                OLD_MLP = '''        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)'''

                NEW_MLP = '''        if _FUSED_OP is not None and hasattr(self.mlp.gate_up_proj, 'quant_method') and hasattr(self.mlp.gate_up_proj.quant_method, 'backend'):
            # FUSED: pre_feedforward_layernorm + gate_up_proj.scaled_fp4_quant
            _ff_norm_w = self.pre_feedforward_layernorm.weight.data
            _ff_norm_eps = self.pre_feedforward_layernorm.variance_epsilon
            gate_up = _fused_linear_forward(
                self.mlp.gate_up_proj, hidden_states, _ff_norm_w, _ff_norm_eps)
            hidden_states = self.mlp.act_fn(gate_up)
            hidden_states, _ = self.mlp.down_proj(hidden_states)
        else:
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)'''

                if OLD_MLP in source:
                    source = source.replace(OLD_MLP, NEW_MLP, 1)
                    logger.info("Patched MLP path with fused norm+quant")
                else:
                    logger.warning("Could not find MLP path to patch")

                # Write patched source
                with open(GEMMA4_PATH, 'w') as f:
                    f.write(source)

                # Clear bytecode cache
                import glob
                pycache = os.path.join(os.path.dirname(GEMMA4_PATH), "__pycache__")
                for pyc in glob.glob(os.path.join(pycache, "gemma4.cpython-*.pyc")):
                    os.remove(pyc)

                logger.info("Gemma4 source patched successfully")

                # Verify syntax
                try:
                    compile(source, GEMMA4_PATH, 'exec')
                    logger.info("Syntax verification: OK")
                except SyntaxError as e:
                    logger.error("Syntax error in patched source: %s", e)
                    sys.exit(1)
        else:
            logger.info("Gemma4 already patched (FUSED_NORM_FP4 marker found)")
    else:
        logger.warning("gemma4.py not found at %s", GEMMA4_PATH)

# ============================================================
# Step 3: Launch vLLM
# ============================================================
logger.info("Launching vLLM api_server with args: %s", sys.argv[1:])

# Use subprocess.exec to replace the current process with vLLM
# This ensures proper signal handling and sys.argv processing
import subprocess
cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + sys.argv[1:]
logger.info("Executing: %s", " ".join(cmd))
os.execv(sys.executable, cmd)
