#!/usr/bin/env python3
"""
Cross-Layer KV Cache Sharing Analysis for Gemma4 26B NVFP4

Analyzes whether adjacent attention layers produce similar K/V projections
(measured via weight-space cosine similarity) that could allow sharing a
single KV cache entry across a group of 2-5 layers.

Architecture facts (from config.json):
  - 30 language model layers total
  - 25 sliding-window attention layers: head_dim=256, num_kv_heads=8
      → k_proj/v_proj shape: [2048, 2816]  (8 heads * 256 dim, hidden=2816)
  - 5 global (full) attention layers at positions [5,11,17,23,29]: head_dim=512, num_kv_heads=2
      → k_proj shape: [1024, 2816]  (2 heads * 512 dim)
      → attention_k_eq_v=True → no separate v_proj (K=V weights tied)
  - Sliding attention: rope_theta=10000, window=1024
  - Global attention: rope_theta=1e6, partial_rotary_factor=0.25

Usage:
  python3 cross_layer_kv_similarity.py [--model-path PATH] [--threshold FLOAT]

Output:
  Prints full similarity matrix and grouping recommendations.
  Optionally saves JSON with raw similarity values.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

MODEL_PATH = "/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/model.safetensors"
NUM_LAYERS = 30
GLOBAL_LAYERS = {5, 11, 17, 23, 29}
SLIDING_LAYERS = [i for i in range(NUM_LAYERS) if i not in GLOBAL_LAYERS]

# Layer type info
LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]


def cosine_similarity_matrix(weight_a: torch.Tensor, weight_b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two weight matrices by flattening to 1D vectors.

    This measures the angular similarity of the projection subspaces in weight space.
    High similarity (> ~0.90) suggests the two projections capture similar features,
    implying their KV outputs will be correlated across typical inputs.
    """
    a = weight_a.float().flatten()
    b = weight_b.float().flatten()
    # Must be same shape for direct comparison
    if a.shape != b.shape:
        return float("nan")
    sim = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)
    return sim.item()


def row_wise_cosine_mean(weight_a: torch.Tensor, weight_b: torch.Tensor) -> float:
    """
    Row-wise cosine similarity (each output neuron = one row) then averaged.

    More meaningful than flat cosine when we care about per-head alignment:
    each row corresponds to a key/value head dimension.
    """
    if weight_a.shape != weight_b.shape:
        return float("nan")
    a = weight_a.float()
    b = weight_b.float()
    # Normalize rows
    a_norm = a / (torch.norm(a, dim=1, keepdim=True) + 1e-8)
    b_norm = b / (torch.norm(b, dim=1, keepdim=True) + 1e-8)
    row_sims = (a_norm * b_norm).sum(dim=1)  # [out_dim]
    return row_sims.mean().item()


def load_weights(model_path: str) -> dict:
    """Load all k_proj and v_proj weights for all 30 language model layers."""
    print(f"Loading weights from {model_path} ...", flush=True)
    f = safe_open(model_path, framework="pt", device="cpu")
    weights = {}
    for layer in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{layer}.self_attn"
        k_key = f"{prefix}.k_proj.weight"
        v_key = f"{prefix}.v_proj.weight"
        weights[layer] = {}
        weights[layer]["k"] = f.get_tensor(k_key)
        if layer not in GLOBAL_LAYERS:
            weights[layer]["v"] = f.get_tensor(v_key)
        else:
            # Global layers: k=v (attention_k_eq_v=True, no separate v_proj)
            weights[layer]["v"] = None
        layer_type = "global" if layer in GLOBAL_LAYERS else "sliding"
        k_shape = weights[layer]["k"].shape
        v_str = str(weights[layer]["v"].shape) if weights[layer]["v"] is not None else "tied to k"
        print(f"  Layer {layer:2d} [{layer_type:7s}]: k={k_shape}, v={v_str}")
    print()
    return weights


def compute_similarity_matrices(weights: dict):
    """
    Compute pairwise cosine similarity for all adjacent and nearby layer pairs.
    Returns dict with keys 'k_flat', 'v_flat', 'k_rowwise', 'v_rowwise',
    each being a (30x30) numpy matrix.
    """
    n = NUM_LAYERS
    k_flat = np.full((n, n), np.nan)
    v_flat = np.full((n, n), np.nan)
    k_row = np.full((n, n), np.nan)
    v_row = np.full((n, n), np.nan)

    # Only compute for same-type pairs (sliding vs sliding, global vs global)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for i, j in all_pairs:
        i_global = i in GLOBAL_LAYERS
        j_global = j in GLOBAL_LAYERS
        if i_global != j_global:
            continue  # skip cross-type pairs (incompatible shapes)

        # K similarity
        k_flat[i, j] = k_flat[j, i] = cosine_similarity_matrix(weights[i]["k"], weights[j]["k"])
        k_row[i, j] = k_row[j, i] = row_wise_cosine_mean(weights[i]["k"], weights[j]["k"])

        # V similarity (None for global layers)
        if weights[i]["v"] is not None and weights[j]["v"] is not None:
            v_flat[i, j] = v_flat[j, i] = cosine_similarity_matrix(weights[i]["v"], weights[j]["v"])
            v_row[i, j] = v_row[j, i] = row_wise_cosine_mean(weights[i]["v"], weights[j]["v"])

    return {"k_flat": k_flat, "v_flat": v_flat, "k_rowwise": k_row, "v_rowwise": v_row}


def print_similarity_table(sims: dict):
    """Print adjacent-layer similarity for quick inspection."""
    k_flat = sims["k_flat"]
    v_flat = sims["v_flat"]
    k_row = sims["k_rowwise"]
    v_row = sims["v_rowwise"]

    print("=" * 80)
    print("ADJACENT-LAYER KV WEIGHT COSINE SIMILARITY")
    print("(flat = whole-matrix cosine, rowwise = per-output-neuron cosine mean)")
    print("=" * 80)
    print(f"{'Pair':12s}  {'Type':16s}  {'K flat':>8s}  {'V flat':>8s}  {'K rowwise':>10s}  {'V rowwise':>10s}")
    print("-" * 80)

    for i in range(NUM_LAYERS - 1):
        j = i + 1
        i_type = "global" if i in GLOBAL_LAYERS else "sliding"
        j_type = "global" if j in GLOBAL_LAYERS else "sliding"
        pair_type = f"{i_type[:3]}-{j_type[:3]}"
        kf = k_flat[i, j]
        vf = v_flat[i, j]
        kr = k_row[i, j]
        vr = v_row[i, j]

        def fmt(x):
            return f"{x:8.4f}" if not np.isnan(x) else "     N/A"

        print(f"L{i:02d}-L{j:02d}      {pair_type:16s}  {fmt(kf)}  {fmt(vf)}  {fmt(kr):>10s}  {fmt(vr):>10s}")

    print()


def print_extended_similarity(sims: dict, max_gap: int = 5):
    """Print similarity for all pairs within max_gap layers."""
    k_row = sims["k_rowwise"]
    v_row = sims["v_rowwise"]

    print("=" * 80)
    print(f"WITHIN-GROUP SIMILARITY (gaps 2..{max_gap}, row-wise cosine)")
    print("=" * 80)
    print(f"{'Pair':12s}  {'Gap':>4s}  {'Type':16s}  {'K rowwise':>10s}  {'V rowwise':>10s}")
    print("-" * 80)

    for gap in range(2, max_gap + 1):
        for i in range(NUM_LAYERS - gap):
            j = i + gap
            if np.isnan(k_row[i, j]):
                continue
            i_type = "global" if i in GLOBAL_LAYERS else "sliding"
            j_type = "global" if j in GLOBAL_LAYERS else "sliding"
            pair_type = f"{i_type[:3]}-{j_type[:3]}"

            def fmt(x):
                return f"{x:10.4f}" if not np.isnan(x) else "       N/A"

            print(f"L{i:02d}-L{j:02d}      {gap:4d}  {pair_type:16s}  {fmt(k_row[i,j])}  {fmt(v_row[i,j])}")
    print()


def find_sharing_groups(sims: dict, thresholds=(0.95, 0.90, 0.85, 0.80)):
    """
    Greedily cluster adjacent layers whose combined KV weight similarity
    exceeds the threshold. Returns grouping info per threshold.
    """
    k_row = sims["k_rowwise"]
    v_row = sims["v_rowwise"]

    print("=" * 80)
    print("KV SHARING GROUP ANALYSIS")
    print("=" * 80)

    results = {}
    for thresh in thresholds:
        groups = []
        used = [False] * NUM_LAYERS
        current_group = []

        for layer in range(NUM_LAYERS):
            if used[layer]:
                continue
            if not current_group:
                current_group = [layer]
            else:
                prev = current_group[-1]
                # Check similarity between this layer and previous
                k_sim = k_row[prev, layer]
                v_sim = v_row[prev, layer]
                # For global layers k=v, so only check k
                if prev in GLOBAL_LAYERS or layer in GLOBAL_LAYERS:
                    can_share = not np.isnan(k_sim) and k_sim >= thresh
                else:
                    can_share = (
                        not np.isnan(k_sim)
                        and not np.isnan(v_sim)
                        and k_sim >= thresh
                        and v_sim >= thresh
                    )
                if can_share:
                    current_group.append(layer)
                else:
                    groups.append(current_group)
                    current_group = [layer]

        if current_group:
            groups.append(current_group)

        # Compute stats
        total_kv_slots_original = 0
        total_kv_slots_shared = 0
        for layer in range(NUM_LAYERS):
            if layer in GLOBAL_LAYERS:
                # global: 2 KV heads × 512 head_dim = 1024 per token
                total_kv_slots_original += 1024
            else:
                # sliding: 8 KV heads × 256 head_dim = 2048 per token
                total_kv_slots_original += 2048

        for group in groups:
            # representative layer stores one slot, others share
            rep = group[0]
            if rep in GLOBAL_LAYERS:
                slot = 1024
            else:
                slot = 2048
            total_kv_slots_shared += slot

        reduction = 1.0 - total_kv_slots_shared / total_kv_slots_original
        num_groups = len(groups)
        avg_size = NUM_LAYERS / num_groups

        print(f"\nThreshold {thresh:.2f}:")
        print(f"  Groups: {num_groups}, avg size: {avg_size:.2f}")
        print(f"  KV memory reduction: {reduction*100:.1f}%")
        print(f"  Effective compression: {1/(1-reduction):.2f}x")
        print(f"  Groups breakdown:")
        for g in groups:
            types = ["G" if x in GLOBAL_LAYERS else "S" for x in g]
            type_str = "/".join(types)
            rep_layer = g[0]
            if len(g) > 1:
                last = g[-1]
                k_sims = [f"{k_row[g[ii], g[ii+1]]:.3f}" for ii in range(len(g)-1)]
                v_sims = []
                for ii in range(len(g) - 1):
                    a, b = g[ii], g[ii+1]
                    if a not in GLOBAL_LAYERS and b not in GLOBAL_LAYERS:
                        v_sims.append(f"{v_row[a,b]:.3f}")
                    else:
                        v_sims.append("k=v")
                k_str = ", ".join(k_sims)
                v_str = ", ".join(v_sims)
                print(f"    [{', '.join(f'L{x}' for x in g)}] ({type_str}) k-sims=[{k_str}] v-sims=[{v_str}]")
            else:
                print(f"    [L{g[0]}] ({type_str}) singleton")

        results[thresh] = {
            "groups": groups,
            "num_groups": num_groups,
            "avg_group_size": avg_size,
            "kv_reduction_pct": reduction * 100,
            "compression_factor": 1 / (1 - reduction) if reduction < 1 else float("inf"),
        }

    return results


def analyze_global_layers(sims: dict):
    """Check if the 5 global layers form their own natural sharing cluster."""
    k_row = sims["k_rowwise"]
    global_list = sorted(GLOBAL_LAYERS)

    print("=" * 80)
    print("GLOBAL ATTENTION LAYER ANALYSIS (layers 5, 11, 17, 23, 29)")
    print("head_dim=512, num_kv_heads=2, attention_k_eq_v=True")
    print("=" * 80)
    print(f"{'Pair':12s}  {'K rowwise':>10s}")
    print("-" * 40)
    for i in range(len(global_list)):
        for j in range(i + 1, len(global_list)):
            a, b = global_list[i], global_list[j]
            sim = k_row[a, b]
            gap = b - a
            print(f"L{a:02d}-L{b:02d} (gap {gap:2d})  {sim:10.4f}")
    print()

    # Check if all global layers are mutually similar
    sims_vals = [k_row[a, b] for i, a in enumerate(global_list) for b in global_list[i+1:]]
    sims_arr = [s for s in sims_vals if not np.isnan(s)]
    if sims_arr:
        print(f"Global-layer K similarity: min={min(sims_arr):.4f}, max={max(sims_arr):.4f}, mean={np.mean(sims_arr):.4f}")
    print()


def kv_memory_breakdown(groups_by_threshold: dict):
    """Estimate KV memory savings for a 32k-token context."""
    print("=" * 80)
    print("KV MEMORY ESTIMATES (32k token context, bfloat16)")
    print("=" * 80)

    # Without sharing
    # Sliding: 25 layers × 8 heads × 256 dim × 2 (K+V) × 2 bytes
    # Global:   5 layers × 2 heads × 512 dim × 1 (K=V) × 2 bytes
    seq_len = 32768
    bytes_per_elem = 2  # bfloat16

    sliding_kv_per_token = 25 * 8 * 256 * 2 * bytes_per_elem  # K and V
    global_kv_per_token = 5 * 2 * 512 * 1 * bytes_per_elem  # K=V tied
    total_kv_bytes = (sliding_kv_per_token + global_kv_per_token) * seq_len

    print(f"\nBaseline (no sharing):")
    print(f"  Sliding attention (25 layers): {sliding_kv_per_token} bytes/token")
    print(f"  Global attention  ( 5 layers): {global_kv_per_token} bytes/token")
    print(f"  Total at seq_len={seq_len//1024}k: {total_kv_bytes / 1024**3:.3f} GB")

    for thresh, info in sorted(groups_by_threshold.items(), reverse=True):
        groups = info["groups"]
        # Compute shared bytes
        shared_bytes = 0
        for group in groups:
            rep = group[0]
            if rep in GLOBAL_LAYERS:
                shared_bytes += 2 * 512 * 1 * bytes_per_elem * seq_len
            else:
                shared_bytes += 8 * 256 * 2 * bytes_per_elem * seq_len

        print(f"\n  Threshold {thresh:.2f} ({info['num_groups']} groups, {info['avg_group_size']:.1f}x avg):")
        print(f"    Shared KV memory: {shared_bytes / 1024**3:.3f} GB")
        print(f"    Savings: {(total_kv_bytes - shared_bytes) / 1024**3:.3f} GB ({info['kv_reduction_pct']:.1f}%)")
        print(f"    Effective compression: {info['compression_factor']:.2f}x")

    print()


def save_json(sims: dict, groups: dict, output_path: str):
    """Save raw similarity data and grouping results to JSON."""

    def nan_to_null(arr):
        return [[None if np.isnan(x) else round(float(x), 6) for x in row] for row in arr]

    data = {
        "model": "gemma-4-26B-A4B-it-NVFP4-modelopt",
        "num_layers": NUM_LAYERS,
        "global_layers": sorted(GLOBAL_LAYERS),
        "sliding_layers": SLIDING_LAYERS,
        "similarity_matrices": {
            "k_flat_cosine": nan_to_null(sims["k_flat"]),
            "v_flat_cosine": nan_to_null(sims["v_flat"]),
            "k_rowwise_cosine": nan_to_null(sims["k_rowwise"]),
            "v_rowwise_cosine": nan_to_null(sims["v_rowwise"]),
        },
        "sharing_groups": {
            str(thresh): {
                "groups": info["groups"],
                "num_groups": info["num_groups"],
                "avg_group_size": round(info["avg_group_size"], 3),
                "kv_reduction_pct": round(info["kv_reduction_pct"], 2),
                "compression_factor": round(info["compression_factor"], 3),
            }
            for thresh, info in groups.items()
        },
    }
    with open(output_path, "w") as fp:
        json.dump(data, fp, indent=2)
    print(f"Saved raw results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to model.safetensors")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Single threshold for grouping (default: sweep 0.80-0.95)")
    parser.add_argument("--save-json", default=None, help="Path to save JSON results")
    args = parser.parse_args()

    weights = load_weights(args.model_path)
    sims = compute_similarity_matrices(weights)

    print_similarity_table(sims)
    print_extended_similarity(sims, max_gap=5)
    analyze_global_layers(sims)

    thresholds = [args.threshold] if args.threshold else [0.95, 0.90, 0.85, 0.80]
    groups = find_sharing_groups(sims, thresholds=thresholds)
    kv_memory_breakdown(groups)

    if args.save_json:
        save_json(sims, groups, args.save_json)


if __name__ == "__main__":
    main()
