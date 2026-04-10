#!/usr/bin/env python3
"""
Diagnostic: Which RMSNorm implementation does vLLM actually use during inference?

=============================================================================
DEFINITIVE FINDING (2026-04-09)
=============================================================================

The warning "Priority not set for op rms_norm, using native implementation"
is COSMETIC for the language model decoder. The vllm_c C++ RMSNorm kernel
IS used for all decoder forward passes (both inference and CUDA graph capture).

However, the warning IS real for the multimodal encoder profiling step:
- embed_multimodal() is called WITHOUT set_forward_context()
- This means _priority_impls is empty, so native Python RMSNorm is used
- This only happens ONCE during startup profiling, not during actual inference
- The affected code is in gpu_model_runner.py profile_run() ~line 5788

ROOT CAUSE:
  vllm/v1/worker/gpu_model_runner.py, profile_run():
    # Line ~5788: No set_forward_context wrapper!
    dummy_encoder_outputs = self.model.embed_multimodal(**batched_dummy_mm_inputs)

  This calls Gemma4's embed_multimodal -> embedding_post_projection_norm (RMSNorm)
  -> ir.ops.rms_norm -> dispatch() with _priority_impls=[] -> fallback to native

TIMELINE (from actual container logs):
  03:25:39 - IR op priority configured as ['vllm_c', 'native']
  03:25:48 - Model loaded, encoder profiling starts
  03:25:56 - WARNING fires (embed_multimodal without forward_context)
  03:25:57 - Decoder profile_run starts (with set_forward_context -> vllm_c used)
  03:26:37 - IR op priority set again for main execution loop

IMPACT:
  - ZERO impact on inference performance
  - The native fallback only runs during the single encoder profiling pass at startup
  - All decoder RMSNorm calls use vllm_c during inference (set_forward_context active)
  - Gemma4's RMSNorm args (variance_size=None, matching dtypes) are fully compatible
    with vllm_c's supports_args check

POTENTIAL FIX (upstream):
  In gpu_model_runner.py profile_run(), wrap embed_multimodal in set_forward_context:
    with set_forward_context(None, self.vllm_config, num_tokens=0):
        dummy_encoder_outputs = self.model.embed_multimodal(...)
  This would silence the warning and ensure vllm_c is used for encoder profiling too.
  However, since it's a one-shot profiling call, the performance impact is negligible.

=============================================================================
"""

import sys
import os
import traceback
import threading
from collections import defaultdict


# ============================================================================
# Verification: Static analysis of vllm_c compatibility with Gemma4
# ============================================================================

def verify_vllm_c_compatibility():
    """Verify that vllm_c kernel supports all Gemma4 RMSNorm argument patterns."""
    import torch
    torch.cuda.init()

    # Must import IR kernels first (normally done by set_priority context manager)
    from vllm.platforms import current_platform
    current_platform.import_ir_kernels()

    from vllm.ir.op import IrOp

    rms_op = IrOp.registry.get("rms_norm")
    if rms_op is None:
        print("[DIAG] ERROR: rms_norm not found in IrOp.registry")
        return False

    print("=" * 70)
    print("vLLM RMSNorm IR Op Static Analysis")
    print("=" * 70)

    # Show registered implementations
    print(f"\nRegistered implementations: {list(rms_op.impls.keys())}")
    print(f"Supported (on this platform): {rms_op.supported_providers()}")
    print(f"Current priority (no forward_context): {rms_op.get_priority()}")

    # Check vllm_c implementation
    vllm_c = rms_op.impls.get("vllm_c")
    if vllm_c is None:
        print("\n[DIAG] vllm_c not registered!")
        return False

    print(f"\nvllm_c supported: {vllm_c.supported}")

    # Test all Gemma4 RMSNorm argument patterns
    x_bf16 = torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda")
    w_bf16 = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6

    patterns = [
        ("Main norms (bf16 x, bf16 w, var=None)", (x_bf16, w_bf16, eps, None)),
        ("Router norm (bf16 x, None w, var=None)", (x_bf16, None, eps, None)),
        ("Q/K norm (bf16 x, bf16 w, var=None)", (x_bf16, w_bf16, eps, None)),
    ]

    print(f"\nGemma4 RMSNorm compatibility with vllm_c:")
    all_ok = True
    for desc, args in patterns:
        ok = vllm_c.supports_args(*args)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {desc}")
        if not ok:
            all_ok = False

    # Test dispatch with priority set (simulating forward_context)
    print(f"\nDispatch with priority=['vllm_c', 'native']:")
    with rms_op.set_priority(["vllm_c", "native"]):
        for desc, args in patterns:
            impl = rms_op.dispatch(*args)
            print(f"  {desc} -> {impl.provider}")

    # Test actual kernel execution
    print(f"\nActual kernel execution test:")
    with rms_op.set_priority(["vllm_c", "native"]):
        x = torch.randn(2, 4096, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
        result = rms_op(x, w, 1e-6, None)
        print(f"  vllm_c output shape: {result.shape}, dtype: {result.dtype}")
        print(f"  vllm_c output sample: {result[0, :4]}")

    # Test native for comparison
    native = rms_op.impls["native"]
    result_native = native.impl_fn(x, w, 1e-6, None)
    diff = (result - result_native).abs().max().item()
    print(f"  Max diff (vllm_c vs native): {diff:.2e}")
    print(f"  Match: {'YES' if diff < 1e-3 else 'NO'}")

    return all_ok


def check_embed_multimodal_context():
    """
    Check whether embed_multimodal runs inside set_forward_context.
    This is the root cause of the warning.
    """
    print("\n" + "=" * 70)
    print("Root Cause Analysis: embed_multimodal missing forward_context")
    print("=" * 70)

    print("""
    In vllm/v1/worker/gpu_model_runner.py, profile_run() method:

    1. Encoder profiling (~line 5788):
       dummy_encoder_outputs = self.model.embed_multimodal(**batched_dummy_mm_inputs)
       ^ NO set_forward_context wrapper -> _priority_impls empty -> native fallback

    2. Decoder profiling (~line 5799):
       hidden_states, last_hidden_states = self._dummy_run(self.max_num_tokens, ...)
       ^ _dummy_run DOES use set_forward_context -> vllm_c properly selected

    The fix would be to wrap embed_multimodal in set_forward_context too.
    But since this is a one-time profiling call, the impact is negligible.

    During actual inference:
    - execute_model() always uses set_forward_context
    - _dummy_run() (warmup/CUDA graphs) always uses set_forward_context
    - All decoder RMSNorm calls dispatch to vllm_c
    """)


def propose_fix():
    """Show the minimal fix to silence the warning."""
    print("=" * 70)
    print("Proposed Fix (optional, cosmetic)")
    print("=" * 70)
    print("""
    In vllm/v1/worker/gpu_model_runner.py, profile_run(), wrap the
    embed_multimodal call:

    BEFORE:
        dummy_encoder_outputs = self.model.embed_multimodal(
            **batched_dummy_mm_inputs
        )

    AFTER:
        with set_forward_context(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            num_tokens=0,
        ):
            dummy_encoder_outputs = self.model.embed_multimodal(
                **batched_dummy_mm_inputs
            )

    This ensures IR op priority is set during encoder profiling too.
    Note: set_forward_context with attn_metadata=None may need testing
    to ensure it doesn't break attention layers in the encoder.

    Alternative: just suppress the warning (it's warning_once so it only
    fires once and doesn't affect performance).
    """)


if __name__ == "__main__":
    print("[DIAG] vLLM RMSNorm Diagnostic for Gemma4")
    print("[DIAG] " + "=" * 50)
    print()

    try:
        all_ok = verify_vllm_c_compatibility()
        check_embed_multimodal_context()
        propose_fix()

        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        if all_ok:
            print("  vllm_c kernel IS compatible with all Gemma4 RMSNorm patterns.")
            print("  vllm_c IS used during actual inference (inside forward_context).")
            print("  The warning fires ONLY during encoder profiling at startup.")
            print("  No fix needed for inference performance.")
        else:
            print("  WARNING: Some Gemma4 patterns are NOT supported by vllm_c!")
            print("  Check the compatibility results above.")
        print("=" * 70)

    except Exception as e:
        print(f"[DIAG] Error: {e}")
        import traceback
        traceback.print_exc()
