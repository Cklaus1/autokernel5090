"""
Gentle Expert Zeroing: Zero the weights of the 5 least-active experts in layer 0 only.

Strategy:
- Keep 128 experts (no structural change, no power-of-2 constraint)
- Zero only the weight tensors (packed weights + scales) for 5 experts in ONE layer
- The router still selects them, but their output is zeros → effectively disabled
- No quality loss from the other 29 layers or 123 healthy experts in layer 0

Target: Layer 0 (Gini=0.657, most skewed layer)
Bottom 5 experts by real-token activation frequency:
  Expert 46: 0.0% of uniform rate (freq=0.0000125)
  Expert 75: 0.0% of uniform rate (freq=0.0000125)
  Expert 110: 0.0% of uniform rate (freq=0.0000125)
  Expert   9: 0.0% of uniform rate (freq=0.0000250)
  Expert  24: 0.0% of uniform rate (freq=0.0000250)

These 5 experts collectively account for <0.02% of routing selections.
Zeroing them should have negligible quality impact.

Output: /root/models/gemma4-gentle-zero/ (full 17GB checkpoint copy with 5 experts zeroed)
"""

import json
import os
import shutil
import sys
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE_MODEL = Path("/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt")
OUTPUT_MODEL = Path("/root/models/gemma4-gentle-zero")
WEIGHTS_FILE = "model.safetensors"

# Layer 0, bottom 5 experts by real-token activation frequency
TARGET_LAYER = 0
TARGET_EXPERTS = [46, 75, 110, 9, 24]

# Weight tensors to zero (the actual parameter tensors, not scales)
# We zero the weight AND the scales to ensure zero output
ZERO_SUFFIXES = [
    "gate_proj.weight",
    "gate_proj.weight_scale",
    "gate_proj.weight_scale_2",
    "up_proj.weight",
    "up_proj.weight_scale",
    "up_proj.weight_scale_2",
    "down_proj.weight",
    "down_proj.weight_scale",
    "down_proj.weight_scale_2",
]

def build_zero_key_set(layer_id: int, expert_ids: list[int]) -> set[str]:
    """Build the set of tensor keys to zero."""
    zero_keys = set()
    prefix = f"model.language_model.layers.{layer_id}.experts."
    for eid in expert_ids:
        for suffix in ZERO_SUFFIXES:
            zero_keys.add(f"{prefix}{eid}.{suffix}")
    return zero_keys


def main():
    print("=" * 70)
    print("Gentle Expert Zeroing: Layer 0, Bottom 5 Experts")
    print("=" * 70)
    print(f"Source: {SOURCE_MODEL}")
    print(f"Output: {OUTPUT_MODEL}")
    print(f"Target layer: {TARGET_LAYER}")
    print(f"Target experts: {TARGET_EXPERTS}")
    print()

    # Create output directory
    OUTPUT_MODEL.mkdir(parents=True, exist_ok=True)

    # Copy all non-weight files first
    print("Step 1: Copying config files...")
    for fname in SOURCE_MODEL.iterdir():
        if fname.name != WEIGHTS_FILE:
            dst = OUTPUT_MODEL / fname.name
            if fname.is_file():
                shutil.copy2(fname, dst)
                print(f"  Copied {fname.name}")

    # Build the set of keys to zero
    zero_keys = build_zero_key_set(TARGET_LAYER, TARGET_EXPERTS)
    print(f"\nStep 2: Will zero {len(zero_keys)} tensor keys")
    for k in sorted(zero_keys):
        print(f"  {k}")

    # Process the weights file
    src_weights = SOURCE_MODEL / WEIGHTS_FILE
    dst_weights = OUTPUT_MODEL / WEIGHTS_FILE

    print(f"\nStep 3: Loading weights from {src_weights} ({src_weights.stat().st_size / 1e9:.1f} GB)...")
    print("  This will take a few minutes...")

    tensors = {}
    zeroed_count = 0
    total_count = 0

    with safe_open(str(src_weights), framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        total_count = len(all_keys)
        print(f"  Total tensors: {total_count}")

        for i, key in enumerate(all_keys):
            if i % 500 == 0:
                print(f"  Progress: {i}/{total_count} ({100*i/total_count:.1f}%)...", flush=True)

            tensor = f.get_tensor(key)

            if key in zero_keys:
                # Zero out this tensor
                original_dtype = tensor.dtype
                original_shape = tensor.shape
                tensor = torch.zeros_like(tensor)
                zeroed_count += 1
                print(f"  ZEROED: {key} shape={original_shape} dtype={original_dtype}")

            tensors[key] = tensor

    print(f"\n  Loaded {total_count} tensors, zeroed {zeroed_count}")

    # Verify we zeroed the right number
    expected_zero = len(zero_keys)
    if zeroed_count != expected_zero:
        print(f"WARNING: Expected to zero {expected_zero} tensors, but zeroed {zeroed_count}")
        print("Missing keys:")
        for k in zero_keys:
            if k not in tensors:
                print(f"  NOT IN CHECKPOINT: {k}")

    # Save the modified checkpoint
    print(f"\nStep 4: Saving modified weights to {dst_weights}...")
    print("  This will take several minutes (17 GB file)...")
    save_file(tensors, str(dst_weights), metadata={"format": "pt"})

    saved_size = dst_weights.stat().st_size / 1e9
    print(f"  Saved: {saved_size:.2f} GB")

    # Update the index file to point to the single weights file
    src_index = SOURCE_MODEL / "model.safetensors.index.json"
    if src_index.exists():
        # The index already points to model.safetensors, just copy it
        # (already copied in step 1 above)
        pass

    # Write a summary of what was done
    summary = {
        "method": "gentle_expert_zeroing",
        "source_model": str(SOURCE_MODEL),
        "target_layer": TARGET_LAYER,
        "target_experts": TARGET_EXPERTS,
        "zeroed_tensor_count": zeroed_count,
        "total_tensor_count": total_count,
        "expert_activation_data": "expert_activation_real_results.json",
        "activation_frequencies": {
            "46": 0.0000125,
            "75": 0.0000125,
            "110": 0.0000125,
            "9": 0.0000250,
            "24": 0.0000250,
        },
        "uniform_rate": 8 / 128,
        "activation_as_pct_of_uniform": {
            "46": "0.020%",
            "75": "0.020%",
            "110": "0.020%",
            "9": "0.040%",
            "24": "0.040%",
        },
        "notes": [
            "Expert count stays at 128 (no structural change)",
            "Router still selects these experts but output is zeros",
            "Only layer 0 is modified (most skewed: Gini=0.657)",
            "5 experts collectively handle <0.02% of routing selections",
        ]
    }

    with open(OUTPUT_MODEL / "zeroing_info.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("DONE!")
    print(f"Output: {OUTPUT_MODEL}")
    print(f"Zeroed {zeroed_count} tensors for experts {TARGET_EXPERTS} in layer {TARGET_LAYER}")
    print("=" * 70)


if __name__ == "__main__":
    main()
