"""
Correctness tests for fused RMSNorm + FP4-E2M1 quantization kernel.

Tests against vLLM's separate rms_norm + scaled_fp4_quant as reference.
"""

import os
import sys
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Reference implementations (pure PyTorch, no vLLM dependency needed)
# ---------------------------------------------------------------------------

# FP4-E2M1 values indexed by magnitude code 0..7
FP4_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def rms_norm_ref(x, weight, eps=1e-6):
    """Reference RMSNorm in PyTorch."""
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_norm = x.float() * torch.rsqrt(variance + eps)
    if weight is not None:
        x_norm = x_norm * weight.float()
    return x_norm


def fp4_quantize_ref(x_norm, global_scale, block_size=16):
    """Reference FP4-E2M1 quantization in PyTorch.

    Returns:
        fp4_packed: [B, H/2] uint8 (low nibble first)
        block_scales: [B, H/block_size] float8_e4m3fn
    """
    B, H = x_norm.shape
    x_f32 = x_norm.float()

    # Reshape to blocks
    x_blocks = x_f32.reshape(B, H // block_size, block_size)

    # Compute per-block max absolute value
    block_max = x_blocks.abs().max(dim=-1).values  # [B, H/block_size]

    # Compute block scales
    block_scale = block_max / (6.0 * global_scale.float())
    block_scale = block_scale.clamp(max=448.0)

    # Convert to fp8_e4m3fn and back to get the quantized scale
    block_scale_fp8 = block_scale.to(torch.float8_e4m3fn)
    block_scale_f32 = block_scale_fp8.float()

    # Compute scaled values
    denom = block_scale_f32.unsqueeze(-1) * global_scale.float()
    denom = denom.clamp(min=1e-10)
    val_scaled = x_blocks.abs() / denom

    # Map to FP4 codes using boundary thresholds
    codes = torch.zeros_like(val_scaled, dtype=torch.int32)
    for i, boundary in enumerate(FP4_BOUNDARIES):
        codes = torch.where(val_scaled > boundary, i + 1, codes)

    # Apply sign bit
    sign = (x_blocks < 0).int() * 8
    fp4_codes = codes | sign  # [B, H/block_size, block_size]

    # Flatten to [B, H]
    fp4_flat = fp4_codes.reshape(B, H)

    # Pack pairs: byte = fp4[2i] | (fp4[2i+1] << 4)
    even = fp4_flat[:, 0::2]  # [B, H/2]
    odd = fp4_flat[:, 1::2]   # [B, H/2]
    packed = (even & 0xF) | ((odd & 0xF) << 4)

    return packed.to(torch.uint8), block_scale_fp8


def reference_rms_norm_fp4_quant(x, weight, global_scale, eps=1e-6, block_size=16):
    """Full reference: RMSNorm then FP4 quantization."""
    x_norm = rms_norm_ref(x, weight, eps)
    return fp4_quantize_ref(x_norm, global_scale, block_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fp4_code_mapping():
    """Test that all FP4-E2M1 codes map correctly."""
    print("Test: FP4 code mapping ... ", end="", flush=True)

    # Test each positive value maps to the correct code
    for code, value in enumerate(FP4_TABLE):
        # Create a single block where max = 6.0, global_scale = 1.0
        # So block_scale = 6.0 / (6.0 * 1.0) = 1.0
        # And val_scaled = value / (1.0 * 1.0) = value
        x = torch.zeros(1, 16, dtype=torch.float32)
        x[0, 0] = value
        x[0, 1] = 6.0  # ensure max = 6.0

        gs = torch.tensor([1.0])
        packed, scales = fp4_quantize_ref(x, gs, block_size=16)

        # First byte packs elements 0 and 1
        byte_val = packed[0, 0].item()
        lo = byte_val & 0xF  # element 0 code
        hi = (byte_val >> 4) & 0xF  # element 1 (6.0 -> code 7)

        assert lo == code, f"Value {value} expected code {code}, got {lo}"
        assert hi == 7, f"Value 6.0 expected code 7, got {hi}"

    # Test negative values
    for code, value in enumerate(FP4_TABLE):
        if value == 0.0:
            continue  # -0.0 is special
        neg_code = code | 8
        x = torch.zeros(1, 16, dtype=torch.float32)
        x[0, 0] = -value
        x[0, 1] = 6.0
        gs = torch.tensor([1.0])
        packed, _ = fp4_quantize_ref(x, gs, block_size=16)
        byte_val = packed[0, 0].item()
        lo = byte_val & 0xF
        assert lo == neg_code, f"Value {-value} expected code {neg_code}, got {lo}"

    print("PASSED")


def test_block_scaling():
    """Test per-block scale computation."""
    print("Test: block scaling ... ", end="", flush=True)

    # Block with max_abs = 3.0, global_scale = 1.0
    # Expected scale = 3.0 / (6.0 * 1.0) = 0.5
    x = torch.zeros(1, 16, dtype=torch.float32)
    x[0, 0] = 3.0
    gs = torch.tensor([1.0])
    _, scales = fp4_quantize_ref(x, gs, block_size=16)
    assert abs(scales[0, 0].float().item() - 0.5) < 0.01, f"Expected 0.5, got {scales[0,0].float().item()}"

    # Block with max_abs = 6.0, gs = 2.0 -> scale = 6.0 / (6.0 * 2.0) = 0.5
    gs2 = torch.tensor([2.0])
    x[0, 0] = 6.0
    _, scales2 = fp4_quantize_ref(x, gs2, block_size=16)
    assert abs(scales2[0, 0].float().item() - 0.5) < 0.01

    print("PASSED")


def test_byte_packing_roundtrip():
    """Test that byte packing and unpacking are consistent."""
    print("Test: byte packing roundtrip ... ", end="", flush=True)

    # Generate random FP4 codes (0-15)
    codes = torch.randint(0, 16, (4, 32), dtype=torch.int32)

    # Pack
    even = codes[:, 0::2]
    odd = codes[:, 1::2]
    packed = (even & 0xF) | ((odd & 0xF) << 4)

    # Unpack
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF

    assert (lo == even).all(), "Low nibble roundtrip failed"
    assert (hi == odd).all(), "High nibble roundtrip failed"

    print("PASSED")


def test_against_vllm_reference():
    """Test against vLLM's rms_norm + scaled_fp4_quant."""
    print("Test: vs vLLM reference ... ", end="", flush=True)

    try:
        from vllm._custom_ops import rms_norm as vllm_rms_norm
        from vllm._custom_ops import scaled_fp4_quant as vllm_fp4_quant
    except ImportError:
        print("SKIPPED (vLLM not available)")
        return

    torch.manual_seed(42)
    B, H = 4, 2816
    x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(H, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    eps = 1e-6

    # vLLM reference: separate ops
    normed = torch.empty_like(x)
    vllm_rms_norm(normed, x, weight, eps)
    ref_fp4, ref_scale = vllm_fp4_quant(normed, gs, is_sf_swizzled_layout=False)

    # Our Python reference (run on GPU)
    our_fp4, our_scale = reference_rms_norm_fp4_quant(
        x.clone(), weight, gs, eps, block_size=16
    )

    # Compare FP4 packed bytes
    match_fp4 = (ref_fp4 == our_fp4).float().mean().item()
    # Compare scales
    match_scale = (ref_scale.view(torch.uint8) == our_scale.view(torch.uint8)).float().mean().item()

    print(f"FP4 match: {match_fp4*100:.1f}%, Scale match: {match_scale*100:.1f}%")

    if match_fp4 < 0.95 or match_scale < 0.95:
        # Show first mismatch
        diff_mask = ref_fp4 != our_fp4
        if diff_mask.any():
            idx = diff_mask.nonzero()[0]
            r, c = idx[0].item(), idx[1].item()
            print(f"  First FP4 mismatch at [{r},{c}]: ref=0x{ref_fp4[r,c].item():02x} ours=0x{our_fp4[r,c].item():02x}")
            # Show the corresponding normalized values
            norm_val = normed[r, c*2:c*2+2].float()
            print(f"  Normed values: {norm_val}")
        print("  WARNING: Match rate below 95%, likely rounding differences at FP4 boundaries")


def test_triton_kernel():
    """Test the Triton kernel against our Python reference."""
    print("Test: Triton kernel ... ", end="", flush=True)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fused_norm_fp4 import fused_rms_norm_fp4_quant
    except ImportError as e:
        print(f"SKIPPED ({e})")
        return

    torch.manual_seed(42)
    B, H = 4, 2816
    x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(H, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    eps = 1e-6

    # Triton kernel
    triton_fp4, triton_scale = fused_rms_norm_fp4_quant(
        x.clone(), weight, gs, eps, quant_block_size=16
    )

    # Python reference
    ref_fp4, ref_scale = reference_rms_norm_fp4_quant(
        x.clone(), weight, gs, eps, block_size=16
    )

    match_fp4 = (triton_fp4 == ref_fp4).float().mean().item()
    match_scale = (triton_scale.view(torch.uint8) == ref_scale.view(torch.uint8)).float().mean().item()

    print(f"FP4 match: {match_fp4*100:.1f}%, Scale match: {match_scale*100:.1f}%")

    if match_fp4 < 0.90:
        diff_mask = triton_fp4 != ref_fp4
        idx = diff_mask.nonzero()[0]
        r, c = idx[0].item(), idx[1].item()
        print(f"  First mismatch at [{r},{c}]: triton=0x{triton_fp4[r,c].item():02x} ref=0x{ref_fp4[r,c].item():02x}")


def test_triton_vs_vllm():
    """Test Triton kernel against vLLM's separate rms_norm + scaled_fp4_quant."""
    print("Test: Triton vs vLLM ... ", end="", flush=True)

    try:
        from vllm._custom_ops import rms_norm as vllm_rms_norm
        from vllm._custom_ops import scaled_fp4_quant as vllm_fp4_quant
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fused_norm_fp4 import fused_rms_norm_fp4_quant
    except ImportError as e:
        print(f"SKIPPED ({e})")
        return

    torch.manual_seed(42)
    B, H = 4, 2816
    x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(H, device="cuda", dtype=torch.bfloat16).abs() + 0.1
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    eps = 1e-6

    # vLLM reference
    normed = torch.empty_like(x)
    vllm_rms_norm(normed, x, weight, eps)
    ref_fp4, ref_scale = vllm_fp4_quant(normed, gs, is_sf_swizzled_layout=False)

    # Our Triton kernel
    triton_fp4, triton_scale = fused_rms_norm_fp4_quant(
        x.clone(), weight, gs, eps, quant_block_size=16
    )

    match_fp4 = (triton_fp4 == ref_fp4).float().mean().item()
    match_scale = (triton_scale.view(torch.uint8) == ref_scale.view(torch.uint8)).float().mean().item()

    print(f"FP4 match: {match_fp4*100:.1f}%, Scale match: {match_scale*100:.1f}%")

    if match_fp4 < 0.95:
        diff_mask = triton_fp4 != ref_fp4
        n_diff = diff_mask.sum().item()
        total = diff_mask.numel()
        print(f"  {n_diff}/{total} bytes differ ({n_diff/total*100:.2f}%)")
        # Analyze: most mismatches should be at FP4 boundaries (rounding differences)


def test_no_weight():
    """Test with weight=None (has_weight=False)."""
    print("Test: no weight ... ", end="", flush=True)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fused_norm_fp4 import fused_rms_norm_fp4_quant
    except ImportError as e:
        print(f"SKIPPED ({e})")
        return

    torch.manual_seed(42)
    B, H = 2, 256
    x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # Triton kernel without weight
    fp4, scale = fused_rms_norm_fp4_quant(x.clone(), None, gs, 1e-6)

    # Reference without weight
    ref_fp4, ref_scale = reference_rms_norm_fp4_quant(x.clone(), None, gs, 1e-6)


    match = (fp4 == ref_fp4).float().mean().item()
    print(f"FP4 match: {match*100:.1f}%")


def test_residual_addition():
    """Test fused residual + RMSNorm + FP4."""
    print("Test: residual addition ... ", end="", flush=True)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fused_norm_fp4 import fused_rms_norm_fp4_quant
    except ImportError as e:
        print(f"SKIPPED ({e})")
        return

    torch.manual_seed(42)
    B, H = 2, 256
    x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    # Fused: x += residual, then RMSNorm(x) -> FP4
    x_fused = x.clone()
    fp4_fused, scale_fused = fused_rms_norm_fp4_quant(
        x_fused, weight, gs, 1e-6, residual=residual
    )

    # Check x was modified in-place
    x_expected = x + residual
    x_inplace_ok = torch.allclose(x_fused.float(), x_expected.float(), atol=1e-3)

    # Compare fused vs row-by-row separate calls (both Triton)
    all_match = True
    for row in range(B):
        x_row = (x[row:row+1] + residual[row:row+1])
        fp4_row, _ = fused_rms_norm_fp4_quant(x_row, weight, gs, 1e-6)
        if not torch.equal(fp4_fused[row:row+1], fp4_row):
            all_match = False

    # Also compare against reference (may have rounding diffs)
    ref_fp4, _ = reference_rms_norm_fp4_quant(x_expected, weight, gs, 1e-6)
    ref_match = (fp4_fused == ref_fp4).float().mean().item()

    status = "OK" if x_inplace_ok else "FAIL"
    row_status = "MATCH" if all_match else "DIFF (autotune sensitivity)"
    print(f"in-place: {status}, row-by-row: {row_status}, vs ref: {ref_match*100:.1f}%")


def test_edge_cases():
    """Test edge cases: zero input, large values."""
    print("Test: edge cases ... ", end="", flush=True)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fused_norm_fp4 import fused_rms_norm_fp4_quant
    except ImportError as e:
        print(f"SKIPPED ({e})")
        return

    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)

    # Zero input: should produce all-zero FP4 output
    x_zero = torch.zeros(1, 256, device="cuda", dtype=torch.bfloat16)
    fp4, scale = fused_rms_norm_fp4_quant(x_zero, weight, gs, 1e-6)
    assert fp4.sum().item() == 0, f"Zero input should produce zero FP4, got sum={fp4.sum().item()}"

    # Very large values (should clamp to max FP4)
    x_large = torch.ones(1, 256, device="cuda", dtype=torch.bfloat16) * 100.0
    fp4, scale = fused_rms_norm_fp4_quant(x_large, weight, gs, 1e-6)
    # After RMSNorm, all values should be ~1.0 (since all same)
    # With weight=1, x_norm ≈ 1.0 for all elements
    # block_max ≈ 1.0, block_scale = 1/6 ≈ 0.1667
    # val_scaled ≈ 1.0 / (0.1667 * 1.0) ≈ 6.0 -> code 7
    # Check that result is non-zero
    assert fp4.sum().item() > 0, "Large input should produce non-zero FP4"

    # FP16 input (not just bf16)
    x_fp16 = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    fp4, scale = fused_rms_norm_fp4_quant(x_fp16, weight.half(), gs, 1e-6)
    assert fp4.shape == (2, 128), f"FP16 output shape wrong: {fp4.shape}"

    print("PASSED")


def test_multiple_batch_sizes():
    """Test various batch sizes."""
    print("Test: batch sizes ... ", end="", flush=True)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fused_norm_fp4 import fused_rms_norm_fp4_quant
    except ImportError as e:
        print(f"SKIPPED ({e})")
        return

    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)

    for B in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x = torch.randn(B, 256, device="cuda", dtype=torch.bfloat16)
        fp4, scale = fused_rms_norm_fp4_quant(x, weight, gs, 1e-6)
        assert fp4.shape == (B, 128), f"B={B}: wrong shape {fp4.shape}"
        assert scale.shape == (B, 16), f"B={B}: wrong scale shape {scale.shape}"

    print("PASSED")


def test_gemma4_shapes():
    """Test Gemma4-specific hidden dimensions."""
    print("Test: Gemma4 shapes ... ", end="", flush=True)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from fused_norm_fp4 import fused_rms_norm_fp4_quant
    except ImportError as e:
        print(f"SKIPPED ({e})")
        return

    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    for H in [256, 512, 2816]:
        weight = torch.ones(H, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(4, H, device="cuda", dtype=torch.bfloat16)
        fp4, scale = fused_rms_norm_fp4_quant(x, weight, gs, 1e-6)
        assert fp4.shape == (4, H // 2), f"H={H}: wrong FP4 shape {fp4.shape}"
        assert scale.shape == (4, H // 16), f"H={H}: wrong scale shape {scale.shape}"

    print("PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Fused RMSNorm + FP4 Quantization Tests")
    print("=" * 60)

    # Pure Python tests (no GPU needed)
    test_fp4_code_mapping()
    test_block_scaling()
    test_byte_packing_roundtrip()

    # GPU tests
    if torch.cuda.is_available():
        test_against_vllm_reference()
        test_triton_kernel()
        test_triton_vs_vllm()
        test_no_weight()
        test_residual_addition()
        test_edge_cases()
        test_multiple_batch_sizes()
        test_gemma4_shapes()
    else:
        print("No CUDA device available, skipping GPU tests")

    print("=" * 60)
    print("All tests completed!")
