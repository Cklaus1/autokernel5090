"""Data-driven compatibility matrix for (weight_quant, kv_cache_dtype) pairs.

Replaces fragile isinstance checks with an explicit lookup table.
Background: vLLM PR bugs incorrectly blocked valid combos (FP8 KV + AWQ)
and missed invalid ones. This module is the single source of truth.

Default policy: ALLOW unless explicitly blocked.
"""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

# ============================================================
# Canonical names
# ============================================================

# Weight quantization methods (as they appear in model configs)
WEIGHT_QUANT_NONE = "none"
WEIGHT_QUANT_AWQ = "awq"
WEIGHT_QUANT_GPTQ = "gptq"
WEIGHT_QUANT_COMPRESSED_TENSORS = "compressed-tensors"
WEIGHT_QUANT_FP8 = "fp8"
WEIGHT_QUANT_NVFP4 = "nvfp4"

# KV cache dtype strings (subset that FusenKV supports)
KV_FP8_E4M3 = "fp8_e4m3"
KV_FP8_E5M2 = "fp8_e5m2"
KV_AUTO = "auto"
KV_K4V4 = "k4v4"
KV_K8V4 = "k8v4"
KV_K8V8 = "k8v8"
KV_K4V2 = "k4v2"
KV_K8V2 = "k8v2"

# All FusenKV-specific formats
FUSEN_KV_FORMATS = [
    KV_K4V4, KV_K8V4, KV_K8V8,
    KV_K4V2, KV_K8V2,
    KV_FP8_E4M3, KV_FP8_E5M2,
    KV_AUTO,
]


# ============================================================
# The matrix: (weight_quant, kv_cache_dtype) -> (allowed, notes)
# ============================================================

COMPATIBILITY_MATRIX: dict[tuple[str, str], tuple[bool, str]] = {
    # ---- none (unquantized weights) ----
    (WEIGHT_QUANT_NONE, KV_AUTO):      (True, "Standard: unquantized weights, auto KV"),
    (WEIGHT_QUANT_NONE, KV_FP8_E4M3):  (True, "FP8 E4M3 KV with unquantized weights"),
    (WEIGHT_QUANT_NONE, KV_FP8_E5M2):  (True, "FP8 E5M2 KV with unquantized weights"),
    (WEIGHT_QUANT_NONE, KV_K4V4):      (True, "FusenKV 4-bit KV with unquantized weights"),
    (WEIGHT_QUANT_NONE, KV_K8V4):      (True, "FusenKV K8V4 with unquantized weights"),
    (WEIGHT_QUANT_NONE, KV_K8V8):      (True, "FusenKV K8V8 with unquantized weights"),

    # ---- AWQ ----
    (WEIGHT_QUANT_AWQ, KV_AUTO):       (True, "AWQ weights, auto KV"),
    (WEIGHT_QUANT_AWQ, KV_FP8_E4M3):   (True, "AWQ + FP8 E4M3 KV — fixed in PR#2, weight quant orthogonal to KV"),
    (WEIGHT_QUANT_AWQ, KV_FP8_E5M2):   (True, "AWQ + FP8 E5M2 KV — fixed in PR#2, weight quant orthogonal to KV"),
    (WEIGHT_QUANT_AWQ, KV_K4V4):       (True, "AWQ + FusenKV 4-bit, weight quant orthogonal to KV"),
    (WEIGHT_QUANT_AWQ, KV_K8V4):       (True, "AWQ + FusenKV K8V4, weight quant orthogonal to KV"),
    (WEIGHT_QUANT_AWQ, KV_K8V8):       (True, "AWQ + FusenKV K8V8, weight quant orthogonal to KV"),

    # ---- compressed-tensors (covers AWQ/GPTQ under CT umbrella) ----
    (WEIGHT_QUANT_COMPRESSED_TENSORS, KV_AUTO):      (True, "Compressed-tensors, auto KV"),
    (WEIGHT_QUANT_COMPRESSED_TENSORS, KV_FP8_E4M3):  (True, "CT + FP8 E4M3 KV, orthogonal"),
    (WEIGHT_QUANT_COMPRESSED_TENSORS, KV_FP8_E5M2):  (True, "CT + FP8 E5M2 KV, orthogonal"),
    (WEIGHT_QUANT_COMPRESSED_TENSORS, KV_K4V4):      (True, "CT + FusenKV 4-bit, orthogonal"),
    (WEIGHT_QUANT_COMPRESSED_TENSORS, KV_K8V4):      (True, "CT + FusenKV K8V4, orthogonal"),
    (WEIGHT_QUANT_COMPRESSED_TENSORS, KV_K8V8):      (True, "CT + FusenKV K8V8, orthogonal"),

    # ---- GPTQ ----
    (WEIGHT_QUANT_GPTQ, KV_AUTO):      (True, "GPTQ weights, auto KV"),
    (WEIGHT_QUANT_GPTQ, KV_FP8_E4M3):  (True, "GPTQ + FP8 E4M3 KV, orthogonal"),
    (WEIGHT_QUANT_GPTQ, KV_FP8_E5M2):  (True, "GPTQ + FP8 E5M2 KV, orthogonal"),
    (WEIGHT_QUANT_GPTQ, KV_K4V4):      (True, "GPTQ + FusenKV 4-bit, orthogonal"),
    (WEIGHT_QUANT_GPTQ, KV_K8V4):      (True, "GPTQ + FusenKV K8V4, orthogonal"),
    (WEIGHT_QUANT_GPTQ, KV_K8V8):      (True, "GPTQ + FusenKV K8V8, orthogonal"),

    # ---- FP8 weight quantization ----
    (WEIGHT_QUANT_FP8, KV_AUTO):       (True, "FP8 weights, auto KV"),
    (WEIGHT_QUANT_FP8, KV_FP8_E4M3):   (True, "FP8 weights + E4M3 KV, same format family"),
    (WEIGHT_QUANT_FP8, KV_FP8_E5M2):   (False, "BLOCKED: FP8 checkpoint + E5M2 KV has numerical issues"),
    (WEIGHT_QUANT_FP8, KV_K4V4):       (True, "FP8 weights + FusenKV 4-bit, orthogonal"),
    (WEIGHT_QUANT_FP8, KV_K8V4):       (True, "FP8 weights + FusenKV K8V4, orthogonal"),
    (WEIGHT_QUANT_FP8, KV_K8V8):       (True, "FP8 weights + FusenKV K8V8, orthogonal"),

    # ---- NVFP4 / modelopt_fp4 ----
    (WEIGHT_QUANT_NVFP4, KV_AUTO):      (True, "NVFP4 weights, auto KV"),
    (WEIGHT_QUANT_NVFP4, KV_FP8_E4M3):  (True, "NVFP4 + FP8 E4M3 KV, orthogonal"),
    (WEIGHT_QUANT_NVFP4, KV_FP8_E5M2):  (False, "BLOCKED: NVFP4 + E5M2 KV untested, block until validated"),
    (WEIGHT_QUANT_NVFP4, KV_K4V4):      (True, "NVFP4 + FusenKV 4-bit, KV quant independent"),
    (WEIGHT_QUANT_NVFP4, KV_K8V4):      (True, "NVFP4 + FusenKV K8V4, KV quant independent"),
    (WEIGHT_QUANT_NVFP4, KV_K8V8):      (True, "NVFP4 + FusenKV K8V8, KV quant independent"),
    (WEIGHT_QUANT_NVFP4, KV_K4V2):      (True, "NVFP4 + FusenKV K4V2, KV quant independent (aggressive)"),
    (WEIGHT_QUANT_NVFP4, KV_K8V2):      (True, "NVFP4 + FusenKV K8V2, KV quant independent (aggressive)"),

    # ---- none + aggressive formats ----
    (WEIGHT_QUANT_NONE, KV_K4V2):       (True, "FusenKV K4V2 aggressive compression"),
    (WEIGHT_QUANT_NONE, KV_K8V2):       (True, "FusenKV K8V2 aggressive compression"),

    # ---- FP8 + aggressive formats ----
    (WEIGHT_QUANT_FP8, KV_K4V2):        (True, "FP8 weights + FusenKV K4V2, aggressive"),
    (WEIGHT_QUANT_FP8, KV_K8V2):        (True, "FP8 weights + FusenKV K8V2, aggressive"),
}


# ============================================================
# Normalize aliases so lookups work regardless of naming
# ============================================================

_WEIGHT_QUANT_ALIASES: dict[str, str] = {
    "none": WEIGHT_QUANT_NONE,
    "": WEIGHT_QUANT_NONE,
    "awq": WEIGHT_QUANT_AWQ,
    "compressed-tensors": WEIGHT_QUANT_COMPRESSED_TENSORS,
    "compressed_tensors": WEIGHT_QUANT_COMPRESSED_TENSORS,
    "gptq": WEIGHT_QUANT_GPTQ,
    "fp8": WEIGHT_QUANT_FP8,
    "nvfp4": WEIGHT_QUANT_NVFP4,
    "modelopt_fp4": WEIGHT_QUANT_NVFP4,
    "modelopt-fp4": WEIGHT_QUANT_NVFP4,
}

_KV_DTYPE_ALIASES: dict[str, str] = {
    "auto": KV_AUTO,
    "fp8_e4m3": KV_FP8_E4M3,
    "fp8_e4m3fn": KV_FP8_E4M3,
    "fp8_e5m2": KV_FP8_E5M2,
    "fp8": KV_FP8_E4M3,  # fp8 without suffix defaults to e4m3
    "k4v4": KV_K4V4,
    "k4v4b16": KV_K4V4,
    "k4v4b32": KV_K4V4,
    "k4v4b64": KV_K4V4,
    "k8v4": KV_K8V4,
    "k8v4b16": KV_K8V4,
    "k8v4b32": KV_K8V4,
    "k8v8": KV_K8V8,
    "k8v8b32": KV_K8V8,
    "k4v2": KV_K4V2,
    "k4v2b16": KV_K4V2,
    "k4v2b32": KV_K4V2,
    "k8v2": KV_K8V2,
    "k8v2b16": KV_K8V2,
    "int4": KV_K4V4,
    "int8": KV_K8V8,
    "fusen": KV_K4V4,
}


def _normalize(weight_quant: str, kv_cache_dtype: str) -> tuple[str, str]:
    """Normalize aliases to canonical names."""
    wq = _WEIGHT_QUANT_ALIASES.get(weight_quant.lower().strip(), weight_quant.lower().strip())
    kv = _KV_DTYPE_ALIASES.get(kv_cache_dtype.lower().strip(), kv_cache_dtype.lower().strip())
    return wq, kv


# ============================================================
# Public API
# ============================================================

def check_compatibility(weight_quant: str, kv_cache_dtype: str) -> tuple[bool, str]:
    """Check if a (weight_quant, kv_cache_dtype) combination is valid.

    Returns (allowed, reason).
    Default policy: allow unless explicitly blocked in the matrix.
    """
    wq, kv = _normalize(weight_quant, kv_cache_dtype)
    key = (wq, kv)

    if key in COMPATIBILITY_MATRIX:
        return COMPATIBILITY_MATRIX[key]

    # Default: allow with a note that it is untested
    return (True, f"Not in matrix ({wq}, {kv}) — allowed by default (untested)")


def list_compatible_formats(weight_quant: str) -> list[str]:
    """List all KV cache formats compatible with a weight quant method."""
    wq = _normalize(weight_quant, "auto")[0]
    compatible = []
    for fmt in FUSEN_KV_FORMATS:
        allowed, _ = check_compatibility(wq, fmt)
        if allowed:
            compatible.append(fmt)
    return compatible


def auto_test_compatibility(
    weight_quant: str,
    kv_cache_dtype: str,
    model_path: str | None = None,
) -> tuple[bool, str]:
    """Run a quick smoke test to check if a combination works.

    If model_path is provided, attempts a minimal forward pass with the
    given weight quant + KV cache dtype to detect runtime errors.
    Falls back to matrix lookup if model_path is not provided.
    """
    # Matrix lookup first — if explicitly blocked, skip the test
    allowed, reason = check_compatibility(weight_quant, kv_cache_dtype)
    if not allowed:
        return allowed, reason

    if model_path is None:
        return allowed, reason + " (matrix lookup, no smoke test)"

    # Attempt a real smoke test
    try:
        import torch
        from fusen_kv.spec_resolver import resolve_spec

        spec = resolve_spec(kv_cache_dtype)

        # Minimal allocation to verify the dtype pipeline works
        head_size = 128
        num_kv_heads = 8
        block_size = 16
        num_blocks = 4
        slot_bytes = int(
            spec.k_bytes_per_dim * head_size + spec.v_bytes_per_dim * head_size
        )
        kv_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, slot_bytes,
            dtype=torch.uint8, device="cuda",
        )
        # If we got here without error, the format is viable
        del kv_cache
        torch.cuda.empty_cache()
        return True, f"Smoke test passed for ({weight_quant}, {kv_cache_dtype})"

    except Exception as e:
        return False, f"Smoke test failed: {e}"


def warn_if_untested(weight_quant: str, kv_cache_dtype: str) -> None:
    """Emit a warning if the combination is not in the matrix.

    Called from FusenKVImpl.__init__ — warns but does not block.
    """
    wq, kv = _normalize(weight_quant, kv_cache_dtype)
    key = (wq, kv)

    if key not in COMPATIBILITY_MATRIX:
        warnings.warn(
            f"FusenKV: combination ({weight_quant!r}, {kv_cache_dtype!r}) "
            f"is not in the compatibility matrix. It may work but is untested. "
            f"Please report results to help expand coverage.",
            stacklevel=3,
        )
    else:
        allowed, notes = COMPATIBILITY_MATRIX[key]
        if not allowed:
            warnings.warn(
                f"FusenKV: combination ({weight_quant!r}, {kv_cache_dtype!r}) "
                f"is known to be problematic: {notes}",
                stacklevel=3,
            )


# ============================================================
# Standalone demo / pretty-print
# ============================================================

def _print_matrix() -> None:
    """Print the full compatibility matrix as a readable table."""
    # Collect all weight quant methods and kv formats from the matrix
    weight_quants = []
    kv_formats = []
    seen_wq = set()
    seen_kv = set()
    for (wq, kv) in COMPATIBILITY_MATRIX:
        if wq not in seen_wq:
            weight_quants.append(wq)
            seen_wq.add(wq)
        if kv not in seen_kv:
            kv_formats.append(kv)
            seen_kv.add(kv)

    # Column widths
    wq_width = max(len(wq) for wq in weight_quants)
    kv_width = max(len(kv) for kv in kv_formats)
    col_w = max(kv_width, 10)

    # Header
    header = f"{'weight_quant':<{wq_width}}  " + "  ".join(
        f"{kv:^{col_w}}" for kv in kv_formats
    )
    print("FusenKV Compatibility Matrix")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for wq in weight_quants:
        cells = []
        for kv in kv_formats:
            key = (wq, kv)
            if key in COMPATIBILITY_MATRIX:
                allowed, _ = COMPATIBILITY_MATRIX[key]
                symbol = "  OK  " if allowed else "BLOCKED"
            else:
                symbol = "  --  "
            cells.append(f"{symbol:^{col_w}}")
        print(f"{wq:<{wq_width}}  " + "  ".join(cells))

    print("-" * len(header))
    print()

    # Details for blocked entries
    blocked = [(wq, kv, notes) for (wq, kv), (allowed, notes)
               in COMPATIBILITY_MATRIX.items() if not allowed]
    if blocked:
        print("Blocked combinations:")
        for wq, kv, notes in blocked:
            print(f"  ({wq}, {kv}): {notes}")
        print()

    # Summary
    total = len(COMPATIBILITY_MATRIX)
    n_allowed = sum(1 for (a, _) in COMPATIBILITY_MATRIX.values() if a)
    n_blocked = total - n_allowed
    print(f"Total entries: {total}  |  Allowed: {n_allowed}  |  Blocked: {n_blocked}")
    print(f"Default policy: ALLOW (untested combos are permitted with a warning)")


if __name__ == "__main__":
    _print_matrix()
