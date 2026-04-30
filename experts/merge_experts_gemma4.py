#!/usr/bin/env python3
"""Merge expert weight pairs in Gemma4-26B-NVFP4 layer 29.

Strategy
--------
For each merge pair (a, b):
  1. Dequantize NVFP4(expert_a, proj) -> BF16  for gate/up/down projections
  2. Dequantize NVFP4(expert_b, proj) -> BF16
  3. merged = 0.5 * (a + b)          (straight weight average)
  4. Requantize merged -> NVFP4 using modelopt layout
  5. Write merged tensors into BOTH expert_a and expert_b slots

Router vector for both experts is also set to the average of the two rows.
This means any token whose top-8 routing picks either slot lands on the same
(merged) compute. Compute is "saved" only when a token's top-8 happens to
contain both slots in the same pair; we honestly document the low probability.

NVFP4 modelopt layout inferred from the existing checkpoint:
  weight         : uint8 [out, in/2]       (2 FP4 nibbles per byte, low nibble = col 0)
  weight_scale   : e4m3fn [out, in/16]     (per-block FP8 scale)
  weight_scale_2 : float32 [1]             (global weight scale)
  input_scale    : float32 [1]             (global activation scale)

Dequant formula (per element):
  fp4_val = decode_e2m1(nibble)
  dequant[m, k] = fp4_val * float(weight_scale[m, k//16]) * float(weight_scale_2)

Requant inverse: we divide by weight_scale_2, pick per-block FP8 scale as
max|.|/6.0, then quantize to the nearest representable NVFP4 magnitude.

We pick the NEW weight_scale_2 as max|merged|/(448*6) so the per-block scale
still fits comfortably in FP8 e4m3 (range ~[0, 448]).

Usage
-----
  python3 experts/merge_experts_gemma4.py                 # default: write to /root/models/gemma-4-26B-merged-layer29
  python3 experts/merge_experts_gemma4.py --dry-run       # just print pairs and numerical merge stats
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SRC_DIR = Path("/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt")
DST_DIR = Path("/root/models/gemma-4-26B-merged-layer29")

LAYER = 29
NUM_EXPERTS = 128
BLOCK_SIZE = 16
FP4_MAX = 6.0
FP8_MAX = 448.0
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


# Top-5 disjoint merge candidates for layer 29 (from
# profiling/expert_similarity_results.json combined_score ranking).
MERGE_PAIRS_LAYER29: List[Tuple[int, int, float]] = [
    (18, 81, 0.8762),
    (78, 111, 0.8736),
    (22, 49, 0.8310),
    (83, 85, 0.8175),
    (54, 101, 0.8172),
]

# NVFP4 representable magnitudes (E2M1, unsigned — sign bit added separately).
NVFP4_MAGS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


# ---------------------------------------------------------------------------
# NVFP4 codec
# ---------------------------------------------------------------------------

def _decode_e2m1(nibble_code: torch.Tensor) -> torch.Tensor:
    """Decode a uint8 tensor of 4-bit E2M1 codes (0..15) -> float32 values."""
    sign = (nibble_code >> 3) & 1
    mag_idx = (nibble_code & 0x7).long()
    mag = NVFP4_MAGS.to(nibble_code.device)[mag_idx]
    return torch.where(sign.bool(), -mag, mag)


def dequantize_nvfp4(weight_u8: torch.Tensor,
                     weight_scale_e4m3: torch.Tensor,
                     weight_scale_2: torch.Tensor) -> torch.Tensor:
    """Dequantize an NVFP4 modelopt-layout weight to float32.

    Args:
        weight_u8       : uint8 [M, K/2]  — 2 nibbles per byte, low nibble = col 0
        weight_scale    : e4m3fn [M, K/16]
        weight_scale_2  : float32 [1]

    Returns:
        dequant : float32 [M, K]
    """
    M, Khalf = weight_u8.shape
    K = Khalf * 2

    # Split byte into two nibbles
    low = weight_u8 & 0x0F          # col 2c
    high = (weight_u8 >> 4) & 0x0F  # col 2c + 1
    # Interleave -> [M, K]
    nibbles = torch.empty((M, K), dtype=torch.uint8, device=weight_u8.device)
    nibbles[:, 0::2] = low
    nibbles[:, 1::2] = high

    fp_vals = _decode_e2m1(nibbles)  # [M, K] float32

    # Apply per-block scale
    num_blocks = K // BLOCK_SIZE
    assert weight_scale_e4m3.shape == (M, num_blocks), (
        f"weight_scale shape {tuple(weight_scale_e4m3.shape)} != expected ({M}, {num_blocks})"
    )
    block_scales = weight_scale_e4m3.to(torch.float32)  # [M, num_blocks]
    block_scales = block_scales.unsqueeze(-1).expand(M, num_blocks, BLOCK_SIZE).reshape(M, K)

    ws2 = weight_scale_2.to(torch.float32).reshape(())
    return fp_vals * block_scales * ws2


def quantize_nvfp4(dequant_f32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a float32 tensor to NVFP4 modelopt layout.

    Returns:
        weight_u8      : uint8 [M, K/2]
        weight_scale   : e4m3fn [M, K/16]
        weight_scale_2 : float32 [1]
    """
    assert dequant_f32.dtype == torch.float32
    M, K = dequant_f32.shape
    assert K % BLOCK_SIZE == 0
    num_blocks = K // BLOCK_SIZE

    # Pick new global scale so per-block scales fit in e4m3 comfortably.
    # Per-block scale we'll want = max_abs_block / FP4_MAX / ws2
    # For that to stay < FP8_MAX: ws2 >= max_abs_block / (FP4_MAX * FP8_MAX)
    abs_max = dequant_f32.abs().amax().clamp_min(1e-12)
    ws2 = (abs_max / (FP4_MAX * FP8_MAX)).clamp_min(1e-12)
    scaled = dequant_f32 / ws2  # "after removing ws2"
    blocks = scaled.reshape(M, num_blocks, BLOCK_SIZE)

    block_max = blocks.abs().amax(dim=-1).clamp_min(1e-12)  # [M, num_blocks]
    # Per-block scale in e4m3 space (stored value)
    block_scale = (block_max / FP4_MAX).clamp(max=FP8_MAX)

    # Quantize per-block scale to e4m3fn (nearest representable value)
    block_scale_e4m3 = block_scale.to(torch.float8_e4m3fn)
    block_scale_rt = block_scale_e4m3.to(torch.float32).clamp_min(1e-12)

    # Per-element scale factor
    per_elem_scale = block_scale_rt.unsqueeze(-1).expand_as(blocks).reshape(M, K)
    x_in_fp4_space = dequant_f32 / (per_elem_scale * ws2)

    # Nearest NVFP4 magnitude
    mags = NVFP4_MAGS.to(x_in_fp4_space.device)
    abs_x = x_in_fp4_space.abs()
    diffs = (abs_x.unsqueeze(-1) - mags.view(1, 1, -1)).abs()
    mag_idx = diffs.argmin(dim=-1).to(torch.uint8)  # [M, K]

    sign_bit = (x_in_fp4_space < 0).to(torch.uint8)
    code = (sign_bit << 3) | mag_idx  # [M, K], uint8, 0..15

    # Pack two nibbles per byte (low nibble = even col)
    low = code[:, 0::2]
    high = code[:, 1::2]
    packed = ((high & 0x0F) << 4) | (low & 0x0F)

    return (
        packed.contiguous(),
        block_scale_e4m3.contiguous(),
        ws2.reshape(1).to(torch.float32).contiguous(),
    )


# ---------------------------------------------------------------------------
# Merge driver
# ---------------------------------------------------------------------------

def load_all_tensors(src: Path) -> dict:
    print(f"Loading {src}/model.safetensors ...")
    tensors = {}
    with safe_open(src / "model.safetensors", framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    print(f"  loaded {len(tensors)} tensors")
    return tensors


def merge_expert_pair(tensors: dict, layer: int, a: int, b: int,
                      dry_run: bool = False) -> dict:
    """Merge expert_b into expert_a by averaging dequantized weights.

    Writes merged NVFP4 tensors into BOTH expert slots.
    Returns per-projection merge stats.
    """
    stats = {}

    for proj in PROJECTIONS:
        pa = f"model.language_model.layers.{layer}.experts.{a}.{proj}"
        pb = f"model.language_model.layers.{layer}.experts.{b}.{proj}"

        wa = tensors[f"{pa}.weight"]
        sa = tensors[f"{pa}.weight_scale"]
        sa2 = tensors[f"{pa}.weight_scale_2"]

        wb = tensors[f"{pb}.weight"]
        sb = tensors[f"{pb}.weight_scale"]
        sb2 = tensors[f"{pb}.weight_scale_2"]

        da = dequantize_nvfp4(wa, sa, sa2)
        db = dequantize_nvfp4(wb, sb, sb2)
        merged = 0.5 * (da + db)

        # Collect merge stats
        rel_err_a = (merged - da).norm() / (da.norm() + 1e-12)
        rel_err_b = (merged - db).norm() / (db.norm() + 1e-12)
        stats[proj] = {
            "shape": tuple(da.shape),
            "rel_err_vs_a": float(rel_err_a),
            "rel_err_vs_b": float(rel_err_b),
            "merged_abs_max": float(merged.abs().amax()),
        }

        if dry_run:
            continue

        w_new, s_new, s2_new = quantize_nvfp4(merged)

        # Measure roundtrip quant error
        roundtrip = dequantize_nvfp4(w_new, s_new, s2_new)
        quant_err = (roundtrip - merged).norm() / (merged.norm() + 1e-12)
        stats[proj]["quant_roundtrip_rel_err"] = float(quant_err)

        # Average activation-side input_scale too, so quant grids line up
        input_scale_a = tensors[f"{pa}.input_scale"]
        input_scale_b = tensors[f"{pb}.input_scale"]
        input_scale_merged = 0.5 * (
            input_scale_a.to(torch.float32) + input_scale_b.to(torch.float32)
        ).to(input_scale_a.dtype)

        for pref in (pa, pb):
            tensors[f"{pref}.weight"] = w_new.clone()
            tensors[f"{pref}.weight_scale"] = s_new.clone()
            tensors[f"{pref}.weight_scale_2"] = s2_new.clone()
            tensors[f"{pref}.input_scale"] = input_scale_merged.clone()

    # Also merge router rows so logits for the two slots are identical.
    if not dry_run:
        router_w_key = f"model.language_model.layers.{layer}.router.proj.weight"
        router_w = tensors[router_w_key]  # [128, 2816] bf16
        row_a = router_w[a].to(torch.float32)
        row_b = router_w[b].to(torch.float32)
        merged_row = 0.5 * (row_a + row_b)
        router_w[a] = merged_row.to(router_w.dtype)
        router_w[b] = merged_row.to(router_w.dtype)
        tensors[router_w_key] = router_w

        pes_key = f"model.language_model.layers.{layer}.router.per_expert_scale"
        pes = tensors[pes_key]  # [128]
        merged_pes = 0.5 * (pes[a].to(torch.float32) + pes[b].to(torch.float32))
        pes[a] = merged_pes.to(pes.dtype)
        pes[b] = merged_pes.to(pes.dtype)
        tensors[pes_key] = pes

    return stats


def copy_non_safetensor_files(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith(".safetensors"):
            continue
        s = src / name
        d = dst / name
        if s.is_file():
            shutil.copy2(s, d)
        elif s.is_dir():
            if d.exists():
                continue
            shutil.copytree(s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC_DIR)
    ap.add_argument("--dst", type=Path, default=DST_DIR)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute merge stats only, do not write output")
    args = ap.parse_args()

    print(f"Merging layer {args.layer} expert pairs:")
    for a, b, score in MERGE_PAIRS_LAYER29:
        print(f"  ({a:3d}, {b:3d})  combined_score={score:.4f}")

    t0 = time.time()
    tensors = load_all_tensors(args.src)

    all_stats = {}
    for a, b, score in MERGE_PAIRS_LAYER29:
        print(f"\nMerging layer {args.layer}: expert {a} <- avg({a},{b}) -> expert {b}")
        stats = merge_expert_pair(tensors, args.layer, a, b, dry_run=args.dry_run)
        for proj, s in stats.items():
            print(f"  {proj:10s} shape={s['shape']} "
                  f"rel_err_vs_a={s['rel_err_vs_a']:.4f} "
                  f"rel_err_vs_b={s['rel_err_vs_b']:.4f} "
                  f"roundtrip_err={s.get('quant_roundtrip_rel_err', float('nan')):.4f}")
        all_stats[f"{a}-{b}"] = stats

    if args.dry_run:
        print("\n[dry-run] stats only, no output written.")
        return

    print(f"\nWriting merged checkpoint to {args.dst} ...")
    copy_non_safetensor_files(args.src, args.dst)

    # Save in a single shard to match source layout
    save_file(tensors, str(args.dst / "model.safetensors"))

    with open(args.dst / "merge_stats.json", "w") as f:
        json.dump({
            "layer": args.layer,
            "pairs": [{"a": a, "b": b, "combined_score": score}
                      for a, b, score in MERGE_PAIRS_LAYER29],
            "stats": all_stats,
        }, f, indent=2)

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
