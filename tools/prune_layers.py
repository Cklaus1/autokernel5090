#!/usr/bin/env python3
"""
Create pruned Gemma 4 26B checkpoints by removing specified layers.

Produces a new checkpoint directory with:
  - Updated config.json (num_hidden_layers, layer_types, exclude_modules)
  - Reindexed safetensors weights (layers renumbered contiguously)

Usage:
  python prune_layers.py --remove 2,4,8       # Remove specific layers
  python prune_layers.py --tier 1             # Remove tier 1 (1 layer)
  python prune_layers.py --tier 2             # Remove tier 2 (3 layers)
  python prune_layers.py --tier 3             # Remove tier 3 (5 layers)
  python prune_layers.py --remove 2,4,8 --dry-run  # Show what would happen
"""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from collections import OrderedDict

MODEL_DIR = "/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
OUTPUT_BASE = "/root/models"

# Pruning tiers from analyze_layers.py results
TIERS = {
    1: [2],
    2: [2, 4, 8],
    3: [2, 4, 6, 8, 20],
}


def load_importance_data():
    """Load pre-computed importance scores."""
    path = "/root/projects/autokernel/profiling/layer_importance.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def compute_layer_mapping(n_layers, remove_layers):
    """Compute old->new layer index mapping after removal."""
    remove_set = set(remove_layers)
    mapping = {}  # old_idx -> new_idx
    new_idx = 0
    for old_idx in range(n_layers):
        if old_idx not in remove_set:
            mapping[old_idx] = new_idx
            new_idx += 1
    return mapping, new_idx


def update_config(config, remove_layers, mapping, new_n_layers):
    """Create updated config with removed layers."""
    config = json.loads(json.dumps(config))  # deep copy

    old_layer_types = config["text_config"]["layer_types"]
    remove_set = set(remove_layers)

    # Update layer_types
    new_layer_types = [t for i, t in enumerate(old_layer_types) if i not in remove_set]
    config["text_config"]["layer_types"] = new_layer_types
    config["text_config"]["num_hidden_layers"] = new_n_layers

    # Update quantization exclude_modules with new layer indices
    quant_cfg = config.get("quantization_config", {}).get("quantization", {})
    old_excludes = quant_cfg.get("exclude_modules", [])

    new_excludes = []
    layer_exclude_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.(.*)")

    for exc in old_excludes:
        m = layer_exclude_pattern.match(exc)
        if m:
            old_idx = int(m.group(1))
            suffix = m.group(2)
            if old_idx in mapping:
                new_idx = mapping[old_idx]
                new_excludes.append(f"model.language_model.layers.{new_idx}.{suffix}")
        else:
            new_excludes.append(exc)  # non-layer excludes stay

    if "exclude_modules" in quant_cfg:
        config["quantization_config"]["quantization"]["exclude_modules"] = new_excludes

    return config


def prune_checkpoint(remove_layers, output_dir=None, dry_run=False):
    """Create pruned checkpoint."""
    # Load config
    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)

    n_layers = config["text_config"]["num_hidden_layers"]
    layer_types = config["text_config"]["layer_types"]

    # Validate
    remove_set = set(remove_layers)
    global_attn = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    removing_global = remove_set & global_attn
    if removing_global:
        print(f"WARNING: Removing global attention layers {removing_global}!")
        print("  This will likely severely degrade quality.")

    mapping, new_n_layers = compute_layer_mapping(n_layers, remove_layers)

    print(f"Pruning plan:")
    print(f"  Original layers: {n_layers}")
    print(f"  Removing: {sorted(remove_layers)}")
    print(f"  Remaining: {new_n_layers}")
    print(f"  Layer mapping (old -> new):")
    for old, new in sorted(mapping.items()):
        lt = layer_types[old]
        print(f"    {old:2d} ({lt:>18}) -> {new:2d}")

    new_config = update_config(config, remove_layers, mapping, new_n_layers)

    # Show new layer type distribution
    new_types = new_config["text_config"]["layer_types"]
    n_sliding = sum(1 for t in new_types if t == "sliding_attention")
    n_full = sum(1 for t in new_types if t == "full_attention")
    full_positions = [i for i, t in enumerate(new_types) if t == "full_attention"]
    print(f"  New distribution: {n_sliding} sliding + {n_full} full_attention")
    print(f"  Full attention at new positions: {full_positions}")

    if dry_run:
        print("\n[DRY RUN] Would create checkpoint at:", output_dir or "auto-named dir")
        return

    # Determine output directory
    if output_dir is None:
        removed_str = "_".join(str(l) for l in sorted(remove_layers))
        output_dir = f"{OUTPUT_BASE}/gemma-4-26B-pruned-{new_n_layers}L-rm{removed_str}"

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Save updated config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(new_config, f, indent=2)
    print("  Saved config.json")

    # Copy non-weight files
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "processor_config.json", "chat_template.jinja", "hf_quant_config.json",
                   "recipe.yaml"]:
        src = f"{MODEL_DIR}/{fname}"
        if os.path.exists(src):
            shutil.copy2(src, f"{output_dir}/{fname}")
            print(f"  Copied {fname}")

    # Process safetensors weights
    print("  Processing weights...")
    t0 = time.time()

    from safetensors import safe_open
    from safetensors.torch import save_file

    sf = safe_open(f"{MODEL_DIR}/model.safetensors", framework="pt")

    layer_pattern = re.compile(r"(model\.language_model\.layers\.)(\d+)\.(.*)")
    new_tensors = OrderedDict()

    for key in sf.keys():
        m = layer_pattern.match(key)
        if m:
            prefix = m.group(1)
            old_idx = int(m.group(2))
            suffix = m.group(3)

            if old_idx in remove_set:
                continue  # skip removed layer weights

            new_idx = mapping[old_idx]
            new_key = f"{prefix}{new_idx}.{suffix}"
            new_tensors[new_key] = sf.get_tensor(key)
        else:
            # Non-layer weights (embeddings, final norm, etc.)
            new_tensors[key] = sf.get_tensor(key)

    # Save
    out_path = f"{output_dir}/model.safetensors"
    save_file(new_tensors, out_path)
    elapsed = time.time() - t0

    # Create index file
    weight_map = {k: "model.safetensors" for k in new_tensors.keys()}
    total_size = sum(t.numel() * t.element_size() for t in new_tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(f"{output_dir}/model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Saved {len(new_tensors)} tensors in {elapsed:.1f}s")
    print(f"  Checkpoint size: {os.path.getsize(out_path) / 1e9:.2f} GB")
    print(f"\nPruned checkpoint ready at: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prune layers from Gemma 4 26B checkpoint")
    parser.add_argument("--remove", type=str, help="Comma-separated layer indices to remove")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Use predefined pruning tier")
    parser.add_argument("--output", type=str, help="Output directory (auto-generated if not set)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without creating checkpoint")
    args = parser.parse_args()

    if args.tier:
        remove_layers = TIERS[args.tier]
        print(f"Using Tier {args.tier}: removing layers {remove_layers}")
    elif args.remove:
        remove_layers = [int(x) for x in args.remove.split(",")]
    else:
        parser.error("Must specify --remove or --tier")

    prune_checkpoint(remove_layers, output_dir=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
