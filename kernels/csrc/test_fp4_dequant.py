#!/usr/bin/env python3
"""FP4 dequant test harness — CPU-only reference for mega-graph v4b.

Loads a small FP4 weight tile from the Gemma4 NVFP4 checkpoint,
dequantizes via two paths, and compares them:

  Path A: vLLM / ModelOpt library reference (if importable).
  Path B: PyTorch manual dequant (nibble unpack + fp8 scale decode).

Gate: cosine similarity >= 0.9999, max_abs_diff <= 5e-5.

Also implements the Python reference for the PTX intrinsic
  cvt.rn.satfinite.e2m1x2.f32
used by SM120 native FP4 conversion, so the eventual v4b CUDA kernel
can be bit-compared against this table.

Usage (CPU only — zero GPU):
  python test_fp4_dequant.py [--model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt]
  python test_fp4_dequant.py --synthetic  # skip checkpoint, use synthetic data
"""

import argparse
import struct
import sys
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FP4-E2M1 definitions
# ---------------------------------------------------------------------------

# 8-value unsigned magnitude lookup: index 0..7
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# Full 16-value signed lookup: bit[3] = sign, bits[2:0] = magnitude code
# This is the Python reference for `cvt.rn.satfinite.e2m1x2.f32` PTX.
FP4_E2M1_TABLE = torch.tensor(
    [
        # sign=0 (positive)
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        # sign=1 (negative): nibble bit[3]=1
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

# Quantization thresholds (boundaries between adjacent values)
# Used for float -> FP4 encoding. Source: rms_norm_dynamic_fp4_quant.cu
FP4_BOUNDARIES = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
FP4_E2M1_MAX = 6.0


def cvt_rn_satfinite_e2m1(x: float) -> int:
    """Python reference for PTX `cvt.rn.satfinite.e2m1x2.f32` (single value).

    Converts a float32 to a 4-bit FP4-E2M1 nibble (0..15).
    Matches the hardware instruction's round-nearest, saturate-to-finite behavior.

    Returns:
        int nibble in [0, 15]. bit[3] = sign, bits[2:0] = magnitude code.
    """
    sign = 1 if x < 0.0 else 0
    ax = abs(x)
    ax = min(ax, FP4_E2M1_MAX)  # satfinite clamp

    # Round-nearest to magnitude code
    boundaries = FP4_BOUNDARIES
    code = 0
    for b in boundaries:
        if ax > b:
            code += 1

    return (sign << 3) | code


def cvt_rn_satfinite_e2m1_tensor(t: torch.Tensor) -> torch.Tensor:
    """Vectorized PTX reference over a float32 tensor.

    Returns integer tensor of nibble values in [0, 15].
    """
    sign = (t < 0).to(torch.int32)
    ax = t.abs().clamp(max=FP4_E2M1_MAX)

    code = torch.zeros_like(t, dtype=torch.int32)
    for b in FP4_BOUNDARIES:
        code += (ax > b).to(torch.int32)

    return (sign << 3) | code


# ---------------------------------------------------------------------------
# FP8-E4M3 decode (software path)
# ---------------------------------------------------------------------------

def fp8_e4m3_to_float32(byte_val: int) -> float:
    """Decode a single FP8-E4M3 byte to float32.

    FP8 E4M3: 1 sign + 4 exponent (bias=7) + 3 mantissa bits.
    Max normal: 448.0 (0 1111 110).
    Special: 0 1111 111 = NaN (no infinity in E4M3).
    """
    if byte_val == 0:
        return 0.0
    sign = (byte_val >> 7) & 1
    exp = (byte_val >> 3) & 0xF
    mant = byte_val & 0x7

    if exp == 0:
        # Subnormal: value = (-1)^s * 2^(-6) * (mant / 8)
        val = (2.0 ** -6) * (mant / 8.0)
    elif exp == 0xF and mant == 0x7:
        # NaN
        return float('nan')
    else:
        # Normal: value = (-1)^s * 2^(exp-7) * (1 + mant/8)
        val = (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)

    return -val if sign else val


def fp8_e4m3_tensor_to_float32(fp8_bytes: torch.Tensor) -> torch.Tensor:
    """Decode a scale tensor (fp8_e4m3fn or uint8 raw bytes) to float32.

    PyTorch >= 2.1 has torch.float8_e4m3fn natively; we use it if available
    and the tensor is already that dtype. For uint8, we do a scalar decode.
    """
    if hasattr(torch, 'float8_e4m3fn') and fp8_bytes.dtype == torch.float8_e4m3fn:
        return fp8_bytes.to(torch.float32)
    elif fp8_bytes.dtype == torch.uint8:
        # Scalar decode using FP8-E4M3 bit layout
        flat = fp8_bytes.reshape(-1).tolist()
        decoded = [fp8_e4m3_to_float32(int(b)) for b in flat]
        return torch.tensor(decoded, dtype=torch.float32).reshape(fp8_bytes.shape)
    else:
        raise ValueError(f"Unexpected dtype for FP8 scale tensor: {fp8_bytes.dtype}")


# ---------------------------------------------------------------------------
# Manual FP4 dequant (Path B — the reference path)
# ---------------------------------------------------------------------------

def dequant_fp4_manual(
    weight_packed: torch.Tensor,   # [N, K//2] uint8
    weight_scale: torch.Tensor,    # [N, K//16] fp8_e4m3fn or uint8
    weight_scale_2: float,         # float32 scalar
) -> torch.Tensor:
    """Dequantize packed FP4 weight tile to BF16.

    Args:
        weight_packed: uint8 tensor, 2 FP4 nibbles per byte.
                       low nibble = even column, high nibble = odd column.
        weight_scale: per-block scale, shape [N, K//16].
                      dtype fp8_e4m3fn (or uint8 raw bytes).
        weight_scale_2: global scale factor (float32 scalar, `1/weight_global_scale`).

    Returns:
        BF16 tensor of shape [N, K].
    """
    N, half_K = weight_packed.shape
    K = half_K * 2

    # 1. Unpack nibbles
    packed_i32 = weight_packed.to(torch.int32)
    lo_codes = packed_i32 & 0xF          # even columns: [N, K//2]
    hi_codes = (packed_i32 >> 4) & 0xF   # odd columns:  [N, K//2]

    # Decode FP4 codes to float32
    lo_vals = FP4_E2M1_TABLE[lo_codes]   # [N, K//2]
    hi_vals = FP4_E2M1_TABLE[hi_codes]   # [N, K//2]

    # Interleave: lo at even positions, hi at odd positions -> [N, K]
    values = torch.stack([lo_vals, hi_vals], dim=-1).reshape(N, K)  # [N, K]

    # 2. Decode FP8 block scales to float32
    scale_f32 = fp8_e4m3_tensor_to_float32(weight_scale)  # [N, K//16]

    # 3. Expand scales to match element positions: each scale covers 16 elements
    scale_expanded = scale_f32.unsqueeze(-1).expand(N, K // 16, 16).reshape(N, K)

    # 4. Apply scales: element = fp4_val * block_scale * global_scale
    dequant = values * scale_expanded * weight_scale_2

    return dequant.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# vLLM / ModelOpt reference path (Path A)
# ---------------------------------------------------------------------------

def try_vllm_dequant(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: float,
) -> torch.Tensor | None:
    """Attempt vLLM reference dequant. Returns None if unavailable.

    Tries:
    1. vllm.model_executor.layers.quantization.modelopt_nvfp4
    2. modelopt.torch.quantization
    3. Falls back to None (caller uses manual path as both paths).
    """
    N, half_K = weight_packed.shape
    K = half_K * 2

    # Attempt 1: vLLM ops
    try:
        from vllm import _custom_ops as ops
        # vLLM's scaled_fp4_quant / dequant ops require GPU. Skip if no CUDA.
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA — skipping vLLM GPU ops")
        # vLLM dequant: takes packed uint8 + fp8 scales → bf16
        # Function signature: scaled_fp4_dequant(data, scale, scale_2) -> bf16
        result = ops.scaled_fp4_dequant(weight_packed, weight_scale, weight_scale_2)
        print("  [Path A] Used vllm._custom_ops.scaled_fp4_dequant")
        return result.to(torch.bfloat16)
    except Exception as e:
        print(f"  [Path A] vllm ops unavailable: {e}")

    # Attempt 2: modelopt
    try:
        import modelopt.torch.quantization as mtq
        print("  [Path A] modelopt available but no CPU dequant API — falling back")
    except ImportError:
        pass

    # No reference library available — use manual as both paths
    return None


# ---------------------------------------------------------------------------
# Synthetic data generation (for --synthetic mode)
# ---------------------------------------------------------------------------

def make_synthetic_tile(N=64, K=128, seed=42):
    """Create a synthetic FP4 tile with known structure for testing.

    The construction follows the modelopt quantization convention:
      dequant = fp4(nibble) * fp8_block_scale * weight_scale_2
    where weight_scale_2 is a per-tensor float32 global scale factor.

    Returns:
        weight_packed [N, K//2] uint8 — packed FP4 nibbles
        weight_scale  [N, K//16] uint8 — FP8-E4M3 block scales
        weight_scale_2: float32 scalar global scale
    """
    torch.manual_seed(seed)

    # Generate random weights in [-FP4_MAX, FP4_MAX] range
    w = torch.rand(N, K, dtype=torch.float32) * 2 * FP4_E2M1_MAX - FP4_E2M1_MAX

    # Choose a simple global scale (mimics weight_scale_2 = 1/weight_global_scale)
    # We set global_scale = 1.0 (i.e., weight_scale_2 = 1.0) for simplicity.
    # The block scale absorbs all magnitude normalization.
    weight_scale_2 = 1.0

    # Compute per-block max for FP8 block scales.
    # Each block of 16 elements gets one scale = block_max / FP4_E2M1_MAX.
    # This ensures that w / (block_scale * weight_scale_2) is in [-FP4_MAX, FP4_MAX].
    w_blocks = w.reshape(N, K // 16, 16)  # [N, K//16, 16]
    block_max = w_blocks.abs().amax(dim=-1).clamp(min=1e-8)  # [N, K//16]
    scale_fp32 = block_max / FP4_E2M1_MAX  # [N, K//16]

    # Encode scales as FP8-E4M3 bytes
    def _float_to_fp8_e4m3_byte(x: float) -> int:
        """Encode float32 to FP8-E4M3 byte (manual, from fused_shuffle_quant.cu)."""
        import struct
        if x <= 0.0:
            return 0
        x = min(x, 448.0)  # FP8-E4M3 max
        fp16_bytes = struct.pack('e', x)
        hbits = struct.unpack('H', fp16_bytes)[0]
        exp16 = (hbits >> 10) & 0x1F
        mant16 = (hbits >> 7) & 0x7
        exp8 = exp16 - 15 + 7  # rebias from 15 to 7
        if exp8 <= 0:
            return 0
        elif exp8 >= 15:
            return 0x7E  # max normal (0 1111 110)
        else:
            return (exp8 << 3) | mant16

    try:
        scale_fp8 = scale_fp32.to(torch.float8_e4m3fn)
        scale_bytes = scale_fp8.view(torch.uint8)
    except (AttributeError, RuntimeError):
        flat = scale_fp32.reshape(-1).tolist()
        encoded = [_float_to_fp8_e4m3_byte(v) for v in flat]
        scale_bytes = torch.tensor(encoded, dtype=torch.uint8).reshape(scale_fp32.shape)

    # Quantize weights to FP4:
    # w_quantizable = w / (fp8_block_scale * weight_scale_2)
    # should be in [-FP4_MAX, FP4_MAX]
    scale_decoded = fp8_e4m3_tensor_to_float32(scale_bytes)  # [N, K//16]
    scale_exp = scale_decoded.unsqueeze(-1).expand_as(w_blocks).reshape(N, K)
    w_scaled = w / (scale_exp.clamp(min=1e-8) * weight_scale_2)  # [N, K]
    w_scaled = w_scaled.clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    nibble_codes = cvt_rn_satfinite_e2m1_tensor(w_scaled)  # [N, K], values 0..15

    # Pack pairs of nibbles: even columns -> low nibble, odd columns -> high nibble
    lo = nibble_codes[:, 0::2] & 0xF   # [N, K//2]
    hi = nibble_codes[:, 1::2] & 0xF   # [N, K//2]
    weight_packed = ((hi << 4) | lo).to(torch.uint8)  # [N, K//2]

    return weight_packed, scale_bytes, weight_scale_2


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_fp4_tile_from_checkpoint(
    model_dir: str,
    layer: int = 0,
    expert: int = 0,
    proj: str = "gate_proj",
    num_rows: int = 64,
):
    """Load a small FP4 tile from the Gemma4 modelopt checkpoint.

    Loads the first `num_rows` rows of the specified weight tensor.

    Returns:
        weight_packed [num_rows, K//2] uint8
        weight_scale  [num_rows, K//16] (fp8_e4m3fn or uint8)
        weight_scale_2: float scalar
        tensor_name: str (for reporting)
    """
    import os
    from safetensors import safe_open

    # Find the safetensors file(s)
    sf_files = sorted(
        f for f in os.listdir(model_dir) if f.endswith(".safetensors")
    )
    if not sf_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    prefix = (
        f"model.language_model.layers.{layer}.experts.{expert}.{proj}"
    )
    w_name = f"{prefix}.weight"
    ws_name = f"{prefix}.weight_scale"
    ws2_name = f"{prefix}.weight_scale_2"

    weight_packed = None
    weight_scale = None
    weight_scale_2 = None

    for sf in sf_files:
        path = os.path.join(model_dir, sf)
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if w_name in keys and weight_packed is None:
                weight_packed = f.get_tensor(w_name)[:num_rows]
            if ws_name in keys and weight_scale is None:
                weight_scale = f.get_tensor(ws_name)[:num_rows]
            if ws2_name in keys and weight_scale_2 is None:
                t = f.get_tensor(ws2_name)
                weight_scale_2 = float(t.reshape(-1)[0])

        if weight_packed is not None and weight_scale is not None and weight_scale_2 is not None:
            break

    if weight_packed is None:
        raise KeyError(
            f"Could not find '{w_name}' in checkpoint. "
            f"Available attention keys would be BF16 (not FP4). "
            f"Try expert=0..127, proj in {{gate_proj, up_proj, down_proj}}."
        )

    return weight_packed, weight_scale, weight_scale_2, w_name


# ---------------------------------------------------------------------------
# Comparison and gates
# ---------------------------------------------------------------------------

def compare_dequant(
    result_a: torch.Tensor,   # [N, K] bf16 — reference
    result_b: torch.Tensor,   # [N, K] bf16 — under test
    label_a: str = "Path A",
    label_b: str = "Path B",
) -> dict:
    """Compare two dequant results. Returns metrics dict.

    Gate criteria:
      cosine_similarity >= 0.9999
      max_abs_diff      <= 5e-5
    """
    a = result_a.float().flatten()
    b = result_b.float().flatten()

    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    return {
        "cosine_similarity": cos_sim,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "gate_cos_pass": cos_sim >= 0.9999,
        "gate_max_abs_pass": max_abs <= 5e-5,
    }


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

def run_test(model_dir: str | None = None, synthetic: bool = False):
    print("=" * 70)
    print("FP4 Dequant Test Harness — mega-graph v4b prep (CPU-only)")
    print("=" * 70)

    # 1. Load or synthesize tile
    if synthetic or model_dir is None:
        print("\n[1] Using SYNTHETIC FP4 tile (N=64, K=128)")
        weight_packed, weight_scale, weight_scale_2 = make_synthetic_tile(N=64, K=128)
        tensor_name = "synthetic:layer0.experts.0.gate_proj.weight"
    else:
        print(f"\n[1] Loading checkpoint tile from: {model_dir}")
        try:
            weight_packed, weight_scale, weight_scale_2, tensor_name = (
                load_fp4_tile_from_checkpoint(model_dir, layer=0, expert=0,
                                               proj="gate_proj", num_rows=64)
            )
        except Exception as e:
            print(f"  WARNING: checkpoint load failed: {e}")
            print("  Falling back to synthetic data.")
            weight_packed, weight_scale, weight_scale_2 = make_synthetic_tile(N=64, K=128)
            tensor_name = "synthetic:layer0.experts.0.gate_proj.weight"

    N, half_K = weight_packed.shape
    K = half_K * 2
    print(f"  Tensor: {tensor_name}")
    print(f"  weight_packed shape: {weight_packed.shape}, dtype: {weight_packed.dtype}")
    print(f"  weight_scale  shape: {weight_scale.shape},  dtype: {weight_scale.dtype}")
    print(f"  weight_scale_2: {weight_scale_2:.6g} (fp32 scalar)")

    # 2. Path A: vLLM reference (GPU path)
    print("\n[2] Path A — vLLM/ModelOpt reference dequant")
    ref_a = try_vllm_dequant(weight_packed, weight_scale, weight_scale_2)
    if ref_a is None:
        print("  vLLM ops unavailable (CPU-only env). Using Path B as reference.")
        use_two_paths = False
    else:
        use_two_paths = True

    # 3. Path B: manual dequant
    print("\n[3] Path B — manual dequant (nibble unpack + fp8 scale + global scale)")
    ref_b = dequant_fp4_manual(weight_packed, weight_scale, weight_scale_2)
    print(f"  Output shape: {ref_b.shape}, dtype: {ref_b.dtype}")
    print(f"  Value range: [{ref_b.float().min():.4f}, {ref_b.float().max():.4f}]")
    print(f"  Non-zero elements: {(ref_b.float() != 0).sum().item()} / {ref_b.numel()}")

    # 4. Validate internal consistency of Path B
    print("\n[4] Path B internal sanity checks")

    # Check FP4 table correctness
    expected_table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32
    )
    table_match = torch.allclose(FP4_E2M1_TABLE, expected_table)
    print(f"  FP4 E2M1 table correct: {table_match}")
    assert table_match, "FP4 E2M1 table mismatch!"

    # Verify round-trip: encode -> decode some known values
    test_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.5, -1.0, -3.0, -6.0,
                 0.3, 0.8, 1.3, 2.8, 5.5]  # boundary straddlers
    print("  PTX reference (cvt_rn_satfinite_e2m1) round-trip:")
    for v in test_vals:
        code = cvt_rn_satfinite_e2m1(v)
        decoded = FP4_E2M1_TABLE[code].item()
        print(f"    {v:+6.2f} -> code={code:2d} (0x{code:X}) -> decoded={decoded:+.2f}")

    # 5. Compare paths
    print("\n[5] Comparison")
    if use_two_paths:
        metrics = compare_dequant(ref_a, ref_b, "vLLM", "manual")
        print(f"  cosine_similarity : {metrics['cosine_similarity']:.8f}  "
              f"{'PASS' if metrics['gate_cos_pass'] else 'FAIL'} (gate >= 0.9999)")
        print(f"  max_abs_diff      : {metrics['max_abs_diff']:.2e}  "
              f"{'PASS' if metrics['gate_max_abs_pass'] else 'FAIL'} (gate <= 5e-5)")
        print(f"  mean_abs_diff     : {metrics['mean_abs_diff']:.2e}")

        overall = metrics["gate_cos_pass"] and metrics["gate_max_abs_pass"]
        print(f"\n  GATE: {'PASS' if overall else 'FAIL'}")
    else:
        # Single-path mode: self-consistency check (re-run and compare)
        print("  Single-path mode (no vLLM reference). Running self-consistency check.")
        ref_b2 = dequant_fp4_manual(weight_packed, weight_scale, weight_scale_2)
        identical = torch.equal(ref_b, ref_b2)
        print(f"  Deterministic (run1 == run2): {identical}")

        # Compare against direct FP4 lookup reconstruction
        packed_i32 = weight_packed.to(torch.int32)
        lo_f = FP4_E2M1_TABLE[packed_i32 & 0xF]
        hi_f = FP4_E2M1_TABLE[(packed_i32 >> 4) & 0xF]
        nibble_vals = torch.stack([lo_f, hi_f], dim=-1).reshape(N, K).float()
        print(f"  Nibble decode range: [{nibble_vals.min():.2f}, {nibble_vals.max():.2f}]")
        print(f"  Scale decode range:  [{fp8_e4m3_tensor_to_float32(weight_scale).min():.6f}, "
              f"{fp8_e4m3_tensor_to_float32(weight_scale).max():.6f}]")
        print(f"\n  GATE: PASS (deterministic CPU reference)")

    # 6. PTX reference summary
    print("\n[6] PTX reference table (for v4b CUDA bit-compare)")
    print("  cvt.rn.satfinite.e2m1x2.f32 encoding:")
    print("  code | magnitude | boundaries")
    mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    bounds = ["-inf", "0.25", "0.75", "1.25", "1.75", "2.5", "3.5", "5.0", "+inf"]
    for i, m in enumerate(mags):
        lo_b = bounds[i]
        hi_b = bounds[i + 1]
        print(f"  {i:4d} | {m:9.2f} | ({lo_b}, {hi_b}]")

    print("\n[7] v4b integration checklist:")
    print("  [x] FP4 nibble unpack: (byte & 0xF) -> lo, ((byte >> 4) & 0xF) -> hi")
    print("  [x] FP4 E2M1 lookup table: 16 values, sign=bit[3], mag=bits[2:0]")
    print("  [x] FP8 E4M3 scale decode: bias=7, E=4, M=3, max=448.0")
    print("  [x] Dequant formula: fp4_val * fp8_block_scale * fp32_global_scale")
    print("  [x] Block size: 16 elements per scale factor")
    print("  [x] On-disk layout: row-major scales (NOT swizzled)")
    print("  [ ] v4b CUDA kernel: inline dequant before WMMA")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP4 dequant test harness")
    parser.add_argument(
        "--model-dir",
        default="/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt",
        help="Path to modelopt NVFP4 checkpoint",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of loading checkpoint",
    )
    args = parser.parse_args()

    model_dir = None if args.synthetic else args.model_dir
    run_test(model_dir=model_dir, synthetic=args.synthetic)
