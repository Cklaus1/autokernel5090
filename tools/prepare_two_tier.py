#!/usr/bin/env python3
"""
Prepare Two-Tier Brain Checkpoints.

Takes a full (unpruned) model checkpoint and a pruning specification, then
splits it into:
    1. Fast brain checkpoint  - pruned model (experts re-indexed, layers removed)
    2. Slow brain experts     - pruned expert weights + manifest
    3. Slow brain layers      - pruned layer weights + manifest

This is the "surgery" step that runs once.  After this, the TwoTierModel
class in two_tier_brain.py can load both halves for inference.

Usage:
    # Split a Gemma 4 26B NVFP4 checkpoint with 30% expert pruning + 3 layer pruning
    python tools/prepare_two_tier.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --output-dir /root/models/gemma4-two-tier-30pct/ \
        --expert-prune-pct 30 \
        --remove-layers 2,4,8

    # Expert pruning only (no layer pruning)
    python tools/prepare_two_tier.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --output-dir /root/models/gemma4-two-tier-experts-only/ \
        --expert-prune-pct 30

    # Dry run: just show what would be pruned
    python tools/prepare_two_tier.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --expert-prune-pct 30 \
        --remove-layers 2,4,8 \
        --dry-run

Requires:
    pip install safetensors torch numpy
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


# ─────────────────────────────────────────────────────────────────────────────
# Constants (Gemma 4 26B defaults)
# ─────────────────────────────────────────────────────────────────────────────

NUM_LAYERS = 30
NUM_EXPERTS = 128
TOP_K = 8
EXPERT_PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]
EXPERT_TENSOR_SUFFIXES = ["weight", "weight_scale", "weight_scale_2", "input_scale"]


# ─────────────────────────────────────────────────────────────────────────────
# Expert Importance (reuses logic from prune_experts.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_expert_importance(model_path: str, num_layers: int = NUM_LAYERS) -> dict:
    """Compute per-layer, per-expert composite importance scores.

    Returns dict: {layer: {expert: composite_score}}
    """
    f = safe_open(model_path, framework="pt", device="cpu")
    composite = {}

    for layer in range(num_layers):
        # Router analysis
        router_proj_key = f"model.language_model.layers.{layer}.router.proj.weight"
        router_proj = f.get_tensor(router_proj_key).float()

        per_expert_scale_key = f"model.language_model.layers.{layer}.router.per_expert_scale"
        per_expert_scale = f.get_tensor(per_expert_scale_key).float()

        router_bias = {}
        router_scale = {}
        for expert in range(NUM_EXPERTS):
            router_bias[expert] = router_proj[expert].norm().item()
            router_scale[expert] = per_expert_scale[expert].item()

        # Weight norm analysis
        expert_norms = {}
        for expert in range(NUM_EXPERTS):
            total_norm_sq = 0.0
            for proj in EXPERT_PROJ_NAMES:
                w_key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.weight"
                try:
                    w = f.get_tensor(w_key).float()
                    total_norm_sq += w.norm().item() ** 2
                except Exception:
                    pass
                ws_key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.weight_scale"
                try:
                    ws = f.get_tensor(ws_key).float()
                    total_norm_sq += ws.norm().item() ** 2
                except Exception:
                    pass
            expert_norms[expert] = total_norm_sq ** 0.5

        # Variance from mean
        norms_arr = np.array([expert_norms[e] for e in range(NUM_EXPERTS)])
        mean_norm = norms_arr.mean()
        weight_var = {e: abs(expert_norms[e] - mean_norm) for e in range(NUM_EXPERTS)}

        # Normalize and combine
        def normalize(d):
            vals = np.array([d[e] for e in range(NUM_EXPERTS)])
            mn, mx = vals.min(), vals.max()
            if mx - mn < 1e-12:
                return {e: 0.5 for e in range(NUM_EXPERTS)}
            return {e: (d[e] - mn) / (mx - mn) for e in range(NUM_EXPERTS)}

        n_l2 = normalize(expert_norms)
        n_var = normalize(weight_var)
        n_rb = normalize(router_bias)
        n_rs = normalize(router_scale)

        composite[layer] = {}
        for expert in range(NUM_EXPERTS):
            composite[layer][expert] = (
                0.40 * n_l2[expert]
                + 0.30 * n_rb[expert]
                + 0.20 * n_var[expert]
                + 0.10 * n_rs[expert]
            )

    return composite


def get_experts_to_prune(importance: dict, prune_pct: float) -> dict[int, list[int]]:
    """For each layer, return sorted list of expert IDs to prune."""
    num_to_prune = int(NUM_EXPERTS * prune_pct / 100)
    prune_map = {}
    for layer in importance:
        scores = sorted(importance[layer].items(), key=lambda x: x[1])
        prune_map[int(layer)] = [int(e) for e, _ in scores[:num_to_prune]]
    return prune_map


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Extract Slow Brain Experts
# ─────────────────────────────────────────────────────────────────────────────

def extract_slow_experts(
    model_path: str,
    prune_map: dict[int, list[int]],
    output_dir: str,
) -> tuple[int, float]:
    """Extract pruned expert weights into the slow brain directory.

    Saves a single safetensors file with naming convention:
        layer_{L}.expert_{E}.{proj}.{suffix}

    Plus manifest.json mapping (layer, expert) -> metadata.

    Returns (num_experts_extracted, size_gb).
    """
    os.makedirs(output_dir, exist_ok=True)
    f = safe_open(model_path, framework="pt", device="cpu")

    slow_tensors = {}
    manifest_experts = {}
    count = 0

    for layer, experts in prune_map.items():
        for expert in experts:
            expert_key_prefix = f"model.language_model.layers.{layer}.experts.{expert}"
            slow_key_prefix = f"layer_{layer}.expert_{expert}"

            found_any = False
            for proj in EXPERT_PROJ_NAMES:
                for suffix in EXPERT_TENSOR_SUFFIXES:
                    orig_key = f"{expert_key_prefix}.{proj}.{suffix}"
                    try:
                        tensor = f.get_tensor(orig_key)
                        slow_key = f"{slow_key_prefix}.{proj}.{suffix}"
                        slow_tensors[slow_key] = tensor
                        found_any = True
                    except Exception:
                        pass  # Not all suffixes exist for all experts

            if found_any:
                manifest_experts[f"{layer},{expert}"] = {
                    "original_layer": layer,
                    "original_expert": expert,
                }
                count += 1

    # Save slow expert tensors
    st_path = os.path.join(output_dir, "slow_experts.safetensors")
    print(f"  Saving {len(slow_tensors)} tensors for {count} slow experts...")
    save_file(slow_tensors, st_path)
    size_gb = os.path.getsize(st_path) / 1e9

    # Save manifest
    manifest = {
        "type": "slow_brain_experts",
        "num_experts": count,
        "prune_map": {str(k): v for k, v in prune_map.items()},
        "experts": manifest_experts,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    print(f"  Extracted {count} slow experts ({size_gb:.2f} GB)")
    return count, size_gb


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Extract Slow Brain Layers
# ─────────────────────────────────────────────────────────────────────────────

def extract_slow_layers(
    model_path: str,
    remove_layers: list[int],
    num_original_layers: int,
    output_dir: str,
) -> tuple[int, float]:
    """Extract pruned layer weights into the slow brain directory.

    Each removed layer gets its own safetensors file:
        layer_{original_idx}.safetensors

    Returns (num_layers_extracted, total_size_gb).
    """
    os.makedirs(output_dir, exist_ok=True)
    f = safe_open(model_path, framework="pt", device="cpu")

    remove_set = set(remove_layers)
    total_size = 0

    # Build layer index mapping
    layer_map = {}  # original -> new fast brain index
    new_idx = 0
    for orig in range(num_original_layers):
        if orig not in remove_set:
            layer_map[orig] = new_idx
            new_idx += 1

    for layer_idx in remove_layers:
        prefix = f"model.language_model.layers.{layer_idx}."
        layer_tensors = {}

        for key in f.keys():
            if key.startswith(prefix):
                # Store with relative key (strip the layer prefix)
                relative_key = key[len(prefix):]
                layer_tensors[relative_key] = f.get_tensor(key)

        if layer_tensors:
            st_path = os.path.join(output_dir, f"layer_{layer_idx}.safetensors")
            save_file(layer_tensors, st_path)
            size = os.path.getsize(st_path)
            total_size += size
            print(f"  Layer {layer_idx}: {len(layer_tensors)} tensors, {size / 1e6:.1f} MB")

    # Save manifest
    manifest = {
        "type": "slow_brain_layers",
        "pruned_layers": remove_layers,
        "num_original_layers": num_original_layers,
        "num_fast_layers": new_idx,
        "layer_map": {str(k): v for k, v in layer_map.items()},
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)

    total_gb = total_size / 1e9
    print(f"  Extracted {len(remove_layers)} slow layers ({total_gb:.2f} GB)")
    return len(remove_layers), total_gb


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Create Fast Brain Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def create_fast_brain(
    model_dir: str,
    prune_map: dict[int, list[int]],
    remove_layers: list[int],
    output_dir: str,
) -> tuple[int, float]:
    """Create the fast brain checkpoint (pruned experts re-indexed, layers removed).

    This is essentially the same as prune_experts.py + prune_layers.py combined,
    but in a single pass.

    Returns (num_remaining_experts, size_gb).
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.safetensors")
    f = safe_open(model_path, framework="pt", device="cpu")
    all_keys = list(f.keys())

    remove_layer_set = set(remove_layers)

    # Layer re-indexing for removed layers
    with open(os.path.join(model_dir, "config.json")) as cf:
        config = json.load(cf)

    num_original = config.get("text_config", {}).get("num_hidden_layers", NUM_LAYERS)
    layer_remap = {}
    new_layer_idx = 0
    for orig in range(num_original):
        if orig not in remove_layer_set:
            layer_remap[orig] = new_layer_idx
            new_layer_idx += 1

    num_pruned_per_layer = len(next(iter(prune_map.values()))) if prune_map else 0
    num_remaining = NUM_EXPERTS - num_pruned_per_layer

    new_tensors = {}
    skipped = 0

    for key in all_keys:
        # Parse layer index
        parts = key.split(".")
        if "layers" in parts:
            layer_pos = parts.index("layers") + 1
            orig_layer = int(parts[layer_pos])

            # Skip removed layers entirely
            if orig_layer in remove_layer_set:
                skipped += 1
                continue

            new_layer = layer_remap[orig_layer]
            parts[layer_pos] = str(new_layer)

            # Handle expert re-indexing
            if "experts" in parts:
                expert_pos = parts.index("experts") + 1
                orig_expert = int(parts[expert_pos])
                pruned_set = set(prune_map.get(orig_layer, []))

                if orig_expert in pruned_set:
                    skipped += 1
                    continue  # This expert goes to slow brain

                # Re-index: count how many pruned experts have lower IDs
                new_expert = orig_expert - sum(1 for p in pruned_set if p < orig_expert)
                parts[expert_pos] = str(new_expert)

            # Handle router tensor slicing
            elif ".router.proj.weight" in key:
                router_w = f.get_tensor(key)
                pruned_set = set(prune_map.get(orig_layer, []))
                keep_mask = torch.ones(NUM_EXPERTS, dtype=torch.bool)
                for e in pruned_set:
                    keep_mask[e] = False
                new_key = ".".join(parts)
                new_tensors[new_key] = router_w[keep_mask]
                continue

            elif ".router.per_expert_scale" in key:
                scale = f.get_tensor(key)
                pruned_set = set(prune_map.get(orig_layer, []))
                keep_mask = torch.ones(NUM_EXPERTS, dtype=torch.bool)
                for e in pruned_set:
                    keep_mask[e] = False
                new_key = ".".join(parts)
                new_tensors[new_key] = scale[keep_mask]
                continue

            new_key = ".".join(parts)
            new_tensors[new_key] = f.get_tensor(key)
        else:
            new_tensors[key] = f.get_tensor(key)

    # Save fast brain checkpoint
    st_path = os.path.join(output_dir, "model.safetensors")
    print(f"  Saving fast brain ({len(new_tensors)} tensors, skipped {skipped})...")
    save_file(new_tensors, st_path)
    size_gb = os.path.getsize(st_path) / 1e9

    # Update config
    config["text_config"]["num_experts"] = num_remaining
    config["text_config"]["num_hidden_layers"] = new_layer_idx
    if config["text_config"].get("top_k_experts", TOP_K) > num_remaining:
        config["text_config"]["top_k_experts"] = num_remaining

    # Update layer_types if present
    if "layer_types" in config.get("text_config", {}):
        old_types = config["text_config"]["layer_types"]
        config["text_config"]["layer_types"] = [
            t for i, t in enumerate(old_types) if i not in remove_layer_set
        ]

    # Add two-tier metadata
    config["two_tier"] = {
        "fast_brain": True,
        "expert_prune_pct": num_pruned_per_layer / NUM_EXPERTS * 100,
        "removed_layers": remove_layers,
        "original_num_experts": NUM_EXPERTS,
        "original_num_layers": num_original,
    }

    with open(os.path.join(output_dir, "config.json"), "w") as cf:
        json.dump(config, cf, indent=2)

    # Copy auxiliary files
    for fname in [
        "tokenizer.json", "tokenizer_config.json", "generation_config.json",
        "processor_config.json", "chat_template.jinja", "hf_quant_config.json",
        "recipe.yaml",
    ]:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, output_dir)

    # Create safetensors index
    index = {
        "metadata": {"total_size": os.path.getsize(st_path)},
        "weight_map": {k: "model.safetensors" for k in new_tensors.keys()},
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as fi:
        json.dump(index, fi)

    print(f"  Fast brain: {num_remaining} experts/layer, {new_layer_idx} layers, {size_gb:.2f} GB")
    return num_remaining, size_gb


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Two-Tier Brain checkpoints from a full model"
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to full (unpruned) model directory")
    parser.add_argument("--output-dir", default="",
                        help="Output base directory (default: {model-dir}-two-tier/)")
    parser.add_argument("--expert-prune-pct", type=float, default=30.0,
                        help="Percentage of experts to prune per layer (default: 30)")
    parser.add_argument("--remove-layers", type=str, default="",
                        help="Comma-separated list of layer indices to remove (e.g. 2,4,8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing files")
    parser.add_argument("--skip-fast-brain", action="store_true",
                        help="Only extract slow brain components (fast brain already exists)")
    parser.add_argument("--skip-slow-experts", action="store_true",
                        help="Skip slow expert extraction")
    parser.add_argument("--skip-slow-layers", action="store_true",
                        help="Skip slow layer extraction")
    args = parser.parse_args()

    model_dir = args.model_dir.rstrip("/")
    if not args.output_dir:
        output_dir = f"{model_dir}-two-tier"
    else:
        output_dir = args.output_dir.rstrip("/")

    fast_dir = os.path.join(output_dir, "fast_brain")
    slow_expert_dir = os.path.join(output_dir, "slow_experts")
    slow_layer_dir = os.path.join(output_dir, "slow_layers")

    remove_layers = []
    if args.remove_layers:
        remove_layers = [int(x.strip()) for x in args.remove_layers.split(",")]

    model_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    print("=" * 70)
    print("TWO-TIER BRAIN CHECKPOINT PREPARATION")
    print("=" * 70)
    print(f"  Source model:     {model_dir}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Expert prune %:   {args.expert_prune_pct}%")
    print(f"  Remove layers:    {remove_layers or 'none'}")
    print()

    # Step 1: Compute expert importance
    print("[1/4] Computing expert importance scores...")
    t0 = time.time()
    importance = compute_expert_importance(model_path)
    prune_map = get_experts_to_prune(importance, args.expert_prune_pct)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    num_pruned = len(next(iter(prune_map.values())))
    total_pruned = sum(len(v) for v in prune_map.values())
    print(f"  Pruning {num_pruned} experts/layer ({total_pruned} total across {len(prune_map)} layers)")

    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        print(f"  Fast brain:    {fast_dir}/")
        print(f"    - {NUM_EXPERTS - num_pruned} experts/layer")
        print(f"    - {NUM_LAYERS - len(remove_layers)} layers")
        print(f"  Slow experts:  {slow_expert_dir}/")
        print(f"    - {total_pruned} expert weight sets")
        if remove_layers:
            print(f"  Slow layers:   {slow_layer_dir}/")
            print(f"    - {len(remove_layers)} layer weight sets")
        print("\nSample prune map (layer 0):")
        for layer in sorted(prune_map.keys())[:3]:
            print(f"  Layer {layer}: prune experts {prune_map[layer][:10]}...")
        return

    # Step 2: Extract slow brain experts
    if not args.skip_slow_experts:
        print(f"\n[2/4] Extracting slow brain experts to {slow_expert_dir}/")
        n_exp, exp_gb = extract_slow_experts(model_path, prune_map, slow_expert_dir)
    else:
        print("\n[2/4] Skipping slow expert extraction")
        n_exp, exp_gb = 0, 0.0

    # Step 3: Extract slow brain layers
    if remove_layers and not args.skip_slow_layers:
        print(f"\n[3/4] Extracting slow brain layers to {slow_layer_dir}/")
        n_lay, lay_gb = extract_slow_layers(model_path, remove_layers, NUM_LAYERS, slow_layer_dir)
    else:
        print("\n[3/4] Skipping slow layer extraction (no layers to remove)")
        n_lay, lay_gb = 0, 0.0

    # Step 4: Create fast brain checkpoint
    if not args.skip_fast_brain:
        print(f"\n[4/4] Creating fast brain checkpoint at {fast_dir}/")
        n_remaining, fast_gb = create_fast_brain(model_dir, prune_map, remove_layers, fast_dir)
    else:
        print("\n[4/4] Skipping fast brain creation")
        n_remaining, fast_gb = NUM_EXPERTS - num_pruned, 0.0

    # Summary
    print("\n" + "=" * 70)
    print("TWO-TIER BRAIN PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  Fast brain (GPU):  {fast_dir}/")
    print(f"    Experts:  {n_remaining}/layer  ({fast_gb:.2f} GB)")
    print(f"    Layers:   {NUM_LAYERS - len(remove_layers)}")
    print(f"  Slow experts (CPU): {slow_expert_dir}/")
    print(f"    Experts:  {n_exp} total  ({exp_gb:.2f} GB)")
    print(f"  Slow layers (CPU):  {slow_layer_dir}/")
    print(f"    Layers:   {n_lay}  ({lay_gb:.2f} GB)")
    print(f"  Total size: {fast_gb + exp_gb + lay_gb:.2f} GB")
    print(f"    (original: ~14.5 GB)")
    print()
    print("Next steps:")
    print(f"  1. Load fast brain:  vllm serve {fast_dir} ...")
    print(f"  2. Attach slow brain in code:")
    print(f"     from tools.two_tier_brain import TwoTierModel")
    print(f"     model = TwoTierModel.from_checkpoints(")
    print(f"         fast_model_dir='{fast_dir}',")
    print(f"         slow_expert_dir='{slow_expert_dir}',")
    print(f"         slow_layer_dir='{slow_layer_dir}',")
    print(f"     )")


if __name__ == "__main__":
    main()
