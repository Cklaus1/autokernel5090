#!/usr/bin/env python3
"""
Patch: Add FP4 (NVFP4) patterns to vLLM's RMSNorm+Quant fusion pass.

Target: /build/vllm/vllm/compilation/passes/fusion/rms_quant_fusion.py
        inside Docker container vllm-gemma4 (vLLM 0.19.1rc1)

Usage:
  docker cp patches/vllm_fp4_norm_quant_fusion.py vllm-gemma4:/tmp/
  docker exec vllm-gemma4 python3 /tmp/vllm_fp4_norm_quant_fusion.py

============================================================================
WHAT THIS DOES
============================================================================

vLLM's compilation pipeline has a "fuse_norm_quant" pass that detects
  rms_norm(x, w, eps) -> quant(normed)
subgraphs and replaces them with a single fused C++ kernel call,
eliminating the intermediate BF16 materialization through global memory.

For FP8 quant, fused kernels already exist:
  - rms_norm_static_fp8_quant
  - fused_add_rms_norm_static_fp8_quant
  - rms_norm_dynamic_per_token_quant
  - rms_norm_per_block_quant

For FP4 (NVFP4) quant, NO fused kernel exists yet. This patch adds the
pattern-matching infrastructure so that when a fused C++ kernel is added,
the graph-level fusion will activate automatically.

Two pattern classes are added:
  - RMSNormFP4QuantPattern:         rms_norm + scaled_fp4_quant
  - FusedAddRMSNormFP4QuantPattern: fused_add_rms_norm + scaled_fp4_quant

Both classes check for the fused kernel at registration time and silently
skip if it doesn't exist (zero runtime cost).

============================================================================
FP4 vs FP8 KEY DIFFERENCES
============================================================================

FP8 dynamic quant:
  scaled_fp8_quant(input) -> (fp8_output, scale)
  - Computes scale dynamically (absmax / fp8_max)
  - Output: fp8_e4m3fn, same shape as input
  - Scale: float32, per-tensor/token/group

FP4 (NVFP4) quant:
  scaled_fp4_quant.out(input, input_scale, is_sf_swizzled_layout,
                        *, output, output_scale)
  - Uses a PRECOMPUTED global scale (static scalar, not dynamic)
  - Output is PACKED: 2 fp4 values per uint8 -> shape [m, n//2]
  - Block scales: float8_e4m3fn in swizzled 128x4 tile layout
  - Block size is fixed at 16 elements
  - Uses .out variant (pre-allocated output buffers)

============================================================================
WHAT'S NEEDED FOR END-TO-END ACTIVATION
============================================================================

1. C++ CUDA kernel: rms_norm_dynamic_fp4_quant
   Proposed signature:
     rms_norm_dynamic_fp4_quant(
       Tensor! result,              // [m, n//2] uint8 packed FP4
       Tensor! result_scale,        // swizzled block scales (int32)
       Tensor input,                // [m, n] bf16/fp16
       Tensor weight,               // [n] bf16/fp16
       Tensor input_global_scale,   // [1] float32
       float epsilon,
       bool is_sf_swizzled_layout
     ) -> ()

   Implementation: Each thread block processes one row. Compute RMS over
   the row, normalize each element, then quantize to FP4 with per-16
   block scales -- all in registers without writing the BF16 intermediate
   to global memory.

2. Register in csrc/torch_bindings.cpp

3. Register fake tensor impl in vllm/_custom_ops.py

4. (Optional) fused_add_rms_norm_dynamic_fp4_quant variant that also
   fuses the residual connection.

============================================================================
PERFORMANCE ESTIMATE
============================================================================

Target: Gemma4 26B NVFP4, RMSNorm is 26% of decode time (4.1ms/step)
- 60 RMSNorm layers, each followed by scaled_fp4_quant
- Current: 2 kernel launches per layer, BF16 intermediate through GMEM
- Fused: 1 kernel launch, intermediate stays in registers
- Expected: ~1-2ms/step savings -> ~7,500-8,000 tok/s (+13-21%)
"""

import glob
import os
import re
import sys
import textwrap

TARGET = "/build/vllm/vllm/compilation/passes/fusion/rms_quant_fusion.py"


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


# ======================================================================== #
# Code blocks to inject                                                     #
# ======================================================================== #

# 1. FUSED_OPS dict entries -- conditional on C++ kernel availability
FUSED_OPS_ADDITION = textwrap.dedent("""\

# FP4 (NVFP4) fused norm+quant ops.
# These activate when the C++ kernels are added to vLLM.
# Without the kernels, the FP4 pattern classes silently skip registration.
if (current_platform.is_cuda()
        and hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "rms_norm_dynamic_fp4_quant")):
    FUSED_OPS[FusedRMSQuantKey(
        kNvfp4Dynamic, False
    )] = torch.ops._C.rms_norm_dynamic_fp4_quant.default
if (current_platform.is_cuda()
        and hasattr(torch.ops, "_C")
        and hasattr(torch.ops._C, "fused_add_rms_norm_dynamic_fp4_quant")):
    FUSED_OPS[FusedRMSQuantKey(
        kNvfp4Dynamic, True
    )] = torch.ops._C.fused_add_rms_norm_dynamic_fp4_quant.default
""")

# 2. Pattern classes -- placed before RMSNormQuantFusionPass
PATTERN_CLASSES = textwrap.dedent("""\


class RMSNormFP4QuantPattern:
    \"\"\"
    Fuse rms_norm + scaled_fp4_quant.out into a single C++ kernel.

    Pattern (matched in FX graph):
        normed = rms_norm(input, weight, epsilon)
        packed_fp4, block_scale = auto_functionalized(
            scaled_fp4_quant.out, input=normed,
            input_scale=global_scale, ...)

    Replacement (emitted):
        packed_fp4, block_scale = auto_functionalized(
            rms_norm_dynamic_fp4_quant, result=..., result_scale=...,
            input=input, weight=weight,
            input_global_scale=global_scale, epsilon=epsilon, ...)

    FP4-specific differences from FP8 patterns:
    - global_scale is a precomputed static scalar, not computed dynamically
    - Output is packed uint8 [m, n//2] (2 fp4 values per byte)
    - Block scales are float8_e4m3fn in swizzled 128x4 tile layout
    - Uses the .out variant with pre-allocated output buffers
    - Block size is fixed at 16 (every 16 elements share one scale factor)
    \"\"\"

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        config = get_current_vllm_config()
        self.model_dtype = (
            config.model_config.dtype if config.model_config else None
        )
        # Look up the fused kernel -- None if C++ kernel not yet implemented
        self.FUSED_OP = FUSED_OPS.get(
            FusedRMSQuantKey(kNvfp4Dynamic, False), None
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        if self.FUSED_OP is None:
            logger.debug(
                "Skipping RMSNormFP4QuantPattern: fused C++ kernel "
                "(rms_norm_dynamic_fp4_quant) not available"
            )
            return

        fp4_quant_op = QUANT_OPS.get(kNvfp4Dynamic)
        if fp4_quant_op is None:
            return

        epsilon = self.epsilon
        model_dtype = self.model_dtype
        fused_op = self.FUSED_OP

        def pattern(
            output: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Step 1: RMSNorm
            rms_out = vllm.ir.ops.rms_norm(input, weight, epsilon)
            # Step 2: FP4 quant via .out variant
            at = auto_functionalized(
                fp4_quant_op,
                input=rms_out,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
                output=output,
                output_scale=output_scale,
            )
            return at[1], at[2]

        def replacement(
            output: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # Single fused kernel: RMSNorm + FP4 quant in one launch
            if model_dtype is not None:
                input = input.to(dtype=model_dtype)
            at = auto_functionalized(
                fused_op,
                result=output,
                result_scale=output_scale,
                input=input,
                weight=weight,
                input_global_scale=input_global_scale,
                epsilon=epsilon,
                is_sf_swizzled_layout=True,
            )
            # result, result_scale
            return at[1], at[2]

        # Dummy inputs for pattern tracing (shapes must be valid, values
        # don't matter -- used only by torch's pattern matcher to trace)
        m, n = 5, 16
        inputs = [
            torch.empty((m, n // 2), device="cuda", dtype=torch.uint8),   # output (packed FP4)
            torch.empty((128, 1), device="cuda", dtype=torch.int32),      # output_scale (swizzled)
            empty_bf16(m, n),                                              # input
            empty_bf16(n),                                                 # weight
            empty_fp32(1, 1),                                              # input_global_scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class FusedAddRMSNormFP4QuantPattern:
    \"\"\"
    Fuse fused_add_rms_norm + scaled_fp4_quant.out into a single C++ kernel.

    Pattern:
        normed, residual = fused_add_rms_norm(input, weight, residual, eps)
        packed_fp4, block_scale = auto_functionalized(
            scaled_fp4_quant.out, input=normed, ...)

    Replacement:
        packed_fp4, block_scale, residual = auto_functionalized(
            fused_add_rms_norm_dynamic_fp4_quant, ...)

    Same FP4-specific notes as RMSNormFP4QuantPattern, plus:
    - Also fuses the residual add (input + residual -> normalize -> quant)
    - Returns 3 tensors: (fp4_packed, block_scale, updated_residual)
    \"\"\"

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        config = get_current_vllm_config()
        self.model_dtype = (
            config.model_config.dtype if config.model_config else None
        )
        self.rmsnorm_matcher = MatcherFusedAddRMSNorm(epsilon)
        self.FUSED_OP = FUSED_OPS.get(
            FusedRMSQuantKey(kNvfp4Dynamic, True), None
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        if self.FUSED_OP is None:
            logger.debug(
                "Skipping FusedAddRMSNormFP4QuantPattern: fused C++ kernel "
                "(fused_add_rms_norm_dynamic_fp4_quant) not available"
            )
            return

        fp4_quant_op = QUANT_OPS.get(kNvfp4Dynamic)
        if fp4_quant_op is None:
            return

        epsilon = self.epsilon
        model_dtype = self.model_dtype
        fused_op = self.FUSED_OP
        rmsnorm_matcher = self.rmsnorm_matcher

        def pattern(
            output: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Step 1: Fused add + RMSNorm
            rms_out, residual = rmsnorm_matcher(input, weight, residual)
            # Step 2: FP4 quant via .out variant
            at = auto_functionalized(
                fp4_quant_op,
                input=rms_out,
                input_scale=input_global_scale,
                is_sf_swizzled_layout=True,
                output=output,
                output_scale=output_scale,
            )
            return at[1], residual, at[2]

        def replacement(
            output: torch.Tensor,
            output_scale: torch.Tensor,
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            input_global_scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Single fused kernel: residual add + RMSNorm + FP4 quant
            if model_dtype is not None:
                input = input.to(dtype=model_dtype)
            at = auto_functionalized(
                fused_op,
                result=output,
                result_scale=output_scale,
                input=input,
                weight=weight,
                residual=residual,
                input_global_scale=input_global_scale,
                epsilon=epsilon,
                is_sf_swizzled_layout=True,
            )
            # result, result_scale, residual
            return at[1], at[3], at[2]

        m, n = 5, 16
        inputs = [
            torch.empty((m, n // 2), device="cuda", dtype=torch.uint8),   # output
            torch.empty((128, 1), device="cuda", dtype=torch.int32),      # output_scale
            *rmsnorm_matcher.inputs(),                                     # input, weight, residual
            empty_fp32(1, 1),                                              # input_global_scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


""")

# 3. Registration block for __init__
REGISTRATION_BLOCK = (
    "\n"
    "            # ---- FP4 (NVFP4) patterns ----\n"
    "            # Fuse RMSNorm + scaled_fp4_quant into a single kernel.\n"
    "            # These only activate if the fused C++ kernel exists.\n"
    "            # Without it, register() is a silent no-op.\n"
    "            if current_platform.is_cuda() and hasattr(\n"
    '                torch.ops._C, "scaled_fp4_quant"\n'
    "            ):\n"
    "                # Fused add variant first (superset matches before subset)\n"
    "                FusedAddRMSNormFP4QuantPattern(epsilon).register(\n"
    "                    self.patterns\n"
    "                )\n"
    "                # Plain RMSNorm variant\n"
    "                RMSNormFP4QuantPattern(epsilon).register(\n"
    "                    self.patterns\n"
    "                )\n"
    "\n"
)


def remove_old_patch(source: str) -> str:
    """Remove the previous (no-op) FP4 patch if present."""
    if "RMSNormNvfp4QuantPattern" not in source and "RMSNormFP4QuantPattern" not in source:
        return source

    print("[INFO] Removing previous FP4 patch...")

    # Remove old class definitions (both naming conventions)
    for cls_name in ["RMSNormNvfp4QuantPattern", "FusedAddRMSNormNvfp4QuantPattern",
                     "RMSNormFP4QuantPattern", "FusedAddRMSNormFP4QuantPattern"]:
        # Match: class Foo: ... until next class or end of classes section
        pattern = re.compile(
            rf'\n\nclass {cls_name}:.*?(?=\n\nclass |\nclass RMSNormQuantFusionPass)',
            re.DOTALL,
        )
        source = pattern.sub('', source)

    # Remove old registration blocks (any style of FP4 comment + registration)
    source = re.sub(
        r'\n\s*# ---*\s*FP4.*?(?:Nvfp4|FP4)QuantPattern\(epsilon\)\.register\(\s*self\.patterns\s*\)\n',
        '\n',
        source,
        flags=re.DOTALL,
    )
    # Remove any standalone old-style registration lines that survived
    source = re.sub(
        r'\n\s*# Fuse (?:fused_add_)?rms_norm \+ nvfp4 quant\n'
        r'\s*(?:# \(must come before.*\n)?'
        r'\s*(?:FusedAdd)?RMSNormNvfp4QuantPattern\(epsilon\)\.register\(\s*\n'
        r'\s*self\.patterns\s*\n\s*\)\n',
        '\n',
        source,
    )

    # Remove old FUSED_OPS additions
    source = re.sub(
        r'\n# FP4 \(NVFP4\) fused.*?\.default\n',
        '\n',
        source,
        flags=re.DOTALL,
    )

    # Remove old uuid references
    for cls_name in ["RMSNormNvfp4QuantPattern", "FusedAddRMSNormNvfp4QuantPattern",
                     "RMSNormFP4QuantPattern", "FusedAddRMSNormFP4QuantPattern"]:
        source = source.replace(f"\n            {cls_name},", "")

    # Clean up excessive blank lines
    source = re.sub(r'\n{4,}', '\n\n\n', source)

    print("[OK] Previous patch removed.")
    return source


def apply_patch() -> None:
    source = read_file(TARGET)

    # ------------------------------------------------------------------ #
    # 0. Remove any previous FP4 patch                                   #
    # ------------------------------------------------------------------ #
    source = remove_old_patch(source)

    # ------------------------------------------------------------------ #
    # 1. Verify prerequisites                                            #
    # ------------------------------------------------------------------ #
    assert "MatcherFusedAddRMSNorm" in source, "MatcherFusedAddRMSNorm import missing"
    assert "import vllm.ir.ops" in source, "vllm.ir.ops import missing"
    assert "FUSED_OPS" in source, "FUSED_OPS dict missing"

    # ------------------------------------------------------------------ #
    # 2. Add FUSED_OPS entries after the dict definition                 #
    # ------------------------------------------------------------------ #
    fused_dict_match = re.search(
        r'(FUSED_OPS:\s*dict\[FusedRMSQuantKey,\s*OpOverload\]\s*=\s*\{.*?^\})',
        source,
        re.MULTILINE | re.DOTALL,
    )
    if fused_dict_match:
        pos = fused_dict_match.end()
        source = source[:pos] + FUSED_OPS_ADDITION + source[pos:]
        print("[OK] Added FP4 FUSED_OPS entries (conditional on C++ kernel).")
    else:
        print("[ERROR] Could not find FUSED_OPS dict.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Add pattern classes before RMSNormQuantFusionPass               #
    # ------------------------------------------------------------------ #
    anchor = "class RMSNormQuantFusionPass(VllmPatternMatcherPass):"
    assert anchor in source, f"Class anchor not found: {anchor}"
    source = source.replace(anchor, PATTERN_CLASSES + anchor)
    print("[OK] Added RMSNormFP4QuantPattern and FusedAddRMSNormFP4QuantPattern.")

    # ------------------------------------------------------------------ #
    # 4. Add registration in __init__ (inside the epsilon loop, before   #
    #    self.dump_patterns)                                              #
    # ------------------------------------------------------------------ #
    dump_anchor = "        self.dump_patterns(config, self.patterns)"
    assert dump_anchor in source, f"dump_patterns anchor not found"
    source = source.replace(dump_anchor, REGISTRATION_BLOCK + dump_anchor)
    print("[OK] Added FP4 registration in __init__.")

    # ------------------------------------------------------------------ #
    # 5. Update uuid() hash sources                                      #
    # ------------------------------------------------------------------ #
    uuid_anchor = "            FusedAddRMSNormGroupQuantPattern,"
    if uuid_anchor in source:
        source = source.replace(
            uuid_anchor,
            uuid_anchor
            + "\n            RMSNormFP4QuantPattern,"
            + "\n            FusedAddRMSNormFP4QuantPattern,",
            1,
        )
        print("[OK] Updated uuid() hash sources.")
    else:
        print("[WARN] Could not find uuid anchor -- cache invalidation may not work.")

    # ------------------------------------------------------------------ #
    # 6. Write and clear bytecode cache                                  #
    # ------------------------------------------------------------------ #
    write_file(TARGET, source)

    pycache_dir = os.path.join(os.path.dirname(TARGET), "__pycache__")
    basename = os.path.basename(TARGET).replace(".py", "")
    for pyc in glob.glob(os.path.join(pycache_dir, f"{basename}.cpython-*.pyc")):
        os.remove(pyc)
        print(f"  Removed cached bytecode: {pyc}")

    print(f"\n[DONE] Patched {TARGET}")
    print()
    print("NOTE: No fused C++ kernel (rms_norm_dynamic_fp4_quant) exists yet.")
    print("The FP4 patterns will silently skip registration until the kernel")
    print("is added. See the docstring at the top of this file for the proposed")
    print("C++ kernel signature and implementation strategy.")


def verify_patch() -> None:
    """Syntax and structural verification."""
    print("\n--- Verification ---")
    source = read_file(TARGET)

    # Syntax check
    try:
        compile(source, TARGET, "exec")
        print("[OK] Syntax valid")
    except SyntaxError as e:
        print(f"[FAIL] Syntax error: {e}")
        sys.exit(1)

    # Check new classes present
    for cls in ["RMSNormFP4QuantPattern", "FusedAddRMSNormFP4QuantPattern"]:
        if f"class {cls}" in source:
            print(f"[OK] {cls} found")
        else:
            print(f"[FAIL] {cls} NOT found")
            sys.exit(1)

    # Check old classes removed
    for old_cls in ["RMSNormNvfp4QuantPattern", "FusedAddRMSNormNvfp4QuantPattern"]:
        if f"class {old_cls}" in source:
            print(f"[WARN] Old class {old_cls} still present")

    # Check FUSED_OPS
    if "rms_norm_dynamic_fp4_quant" in source:
        print("[OK] FUSED_OPS entries reference rms_norm_dynamic_fp4_quant")
    else:
        print("[FAIL] FUSED_OPS entries for FP4 NOT found")
        sys.exit(1)

    # Check registration
    if "RMSNormFP4QuantPattern(epsilon).register" in source:
        print("[OK] FP4 pattern registration in __init__")
    else:
        print("[FAIL] FP4 pattern registration NOT found")
        sys.exit(1)

    # Check uuid
    if "RMSNormFP4QuantPattern," in source:
        print("[OK] uuid hash sources include FP4 classes")
    else:
        print("[WARN] uuid may not include FP4 classes")

    # Check pattern != replacement (the old no-op bug)
    if source.count("fused_op,") >= 2:
        print("[OK] Replacement calls fused_op (not the same as pattern)")
    else:
        print("[WARN] Could not verify pattern differs from replacement")

    print("\n--- Verification complete ---")


if __name__ == "__main__":
    apply_patch()
    verify_patch()
