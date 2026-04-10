#!/usr/bin/env python3
"""
Expert Pairwise Similarity Analysis for Gemma4 26B MoE.

Computes cosine similarity between expert weight matrices to identify
merging candidates. Experts with high similarity can potentially be merged
with minimal quality loss.

Note: Expert weights are NVFP4 quantized. We dequantize to compute similarity.
For the router weight vectors (bf16), we compare directly.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from pathlib import Path
from itertools import combinations
import time

MODEL_PATH = "/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/model.safetensors"
NUM_LAYERS = 30
NUM_EXPERTS = 128
TOP_K = 8


def compute_router_vector_similarity(sf):
    """
    Compare router weight vectors (row of the router matrix per expert).
    This is fast and tells us which experts the router treats as similar.
    """
    print("Computing router vector similarities...")
    results = {}

    for layer_idx in range(NUM_LAYERS):
        w = sf.get_tensor(f"model.language_model.layers.{layer_idx}.router.proj.weight").float()
        # w: [128, 2816] - each row is an expert's routing vector

        # Normalize for cosine similarity
        w_norm = F.normalize(w, dim=1)  # [128, 2816]

        # Pairwise cosine similarity
        sim_matrix = (w_norm @ w_norm.T).numpy()  # [128, 128]

        # Find top merge candidates (highest similarity pairs)
        # Zero out diagonal
        np.fill_diagonal(sim_matrix, 0)

        # Get top-20 most similar pairs
        flat_idx = np.argsort(sim_matrix.flatten())[::-1]
        top_pairs = []
        seen = set()
        for idx in flat_idx:
            i, j = divmod(idx, NUM_EXPERTS)
            if i >= j:
                continue
            pair = (int(i), int(j))
            if pair not in seen:
                seen.add(pair)
                top_pairs.append((int(i), int(j), float(sim_matrix[i, j])))
            if len(top_pairs) >= 20:
                break

        # Stats
        upper_tri = sim_matrix[np.triu_indices(NUM_EXPERTS, k=1)]
        results[layer_idx] = {
            "sim_matrix": sim_matrix,
            "top_pairs": top_pairs,
            "mean_sim": float(np.mean(upper_tri)),
            "max_sim": float(np.max(upper_tri)),
            "min_sim": float(np.min(upper_tri)),
            "std_sim": float(np.std(upper_tri)),
            "pairs_above_0.9": int(np.sum(upper_tri > 0.9)),
            "pairs_above_0.8": int(np.sum(upper_tri > 0.8)),
            "pairs_above_0.7": int(np.sum(upper_tri > 0.7)),
        }

    return results


def compute_expert_weight_fingerprint_similarity(sf):
    """
    Compare experts by their weight matrix fingerprints.
    Since full dequant of 128*30 experts is expensive, we use the
    weight_scale tensors as fingerprints (they capture the magnitude structure).
    """
    print("Computing expert weight fingerprint similarities...")
    results = {}

    for layer_idx in range(NUM_LAYERS):
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}/{NUM_LAYERS}...")

        fingerprints = []
        for expert_idx in range(NUM_EXPERTS):
            prefix = f"model.language_model.layers.{layer_idx}.experts.{expert_idx}"
            # Use weight scales as fingerprint (they encode magnitude distribution)
            gate_scale = sf.get_tensor(f"{prefix}.gate_proj.weight_scale").float().flatten()
            up_scale = sf.get_tensor(f"{prefix}.up_proj.weight_scale").float().flatten()
            down_scale = sf.get_tensor(f"{prefix}.down_proj.weight_scale").float().flatten()
            # Concatenate into a single fingerprint vector
            fp = torch.cat([gate_scale, up_scale, down_scale])
            fingerprints.append(fp)

        fingerprints = torch.stack(fingerprints)  # [128, D]
        fp_norm = F.normalize(fingerprints, dim=1)
        sim_matrix = (fp_norm @ fp_norm.T).numpy()
        np.fill_diagonal(sim_matrix, 0)

        upper_tri = sim_matrix[np.triu_indices(NUM_EXPERTS, k=1)]

        # Top merge candidates
        flat_idx = np.argsort(sim_matrix.flatten())[::-1]
        top_pairs = []
        seen = set()
        for idx in flat_idx:
            i, j = divmod(idx, NUM_EXPERTS)
            if i >= j:
                continue
            pair = (int(i), int(j))
            if pair not in seen:
                seen.add(pair)
                top_pairs.append((int(i), int(j), float(sim_matrix[i, j])))
            if len(top_pairs) >= 20:
                break

        results[layer_idx] = {
            "sim_matrix": sim_matrix,
            "top_pairs": top_pairs,
            "mean_sim": float(np.mean(upper_tri)),
            "max_sim": float(np.max(upper_tri)),
            "min_sim": float(np.min(upper_tri)),
            "std_sim": float(np.std(upper_tri)),
            "pairs_above_0.95": int(np.sum(upper_tri > 0.95)),
            "pairs_above_0.9": int(np.sum(upper_tri > 0.9)),
            "pairs_above_0.8": int(np.sum(upper_tri > 0.8)),
        }

    return results


def find_merge_candidates(router_sim, weight_sim, activation_results=None):
    """
    Combine router similarity and weight similarity to find best merge candidates.
    Experts that are both routed-similarly AND have similar weights are best candidates.
    """
    print("\nFinding merge candidates (combined analysis)...")
    candidates_per_layer = {}

    for layer_idx in range(NUM_LAYERS):
        r_sim = router_sim[layer_idx]["sim_matrix"]
        w_sim = weight_sim[layer_idx]["sim_matrix"]

        # Combined score: geometric mean of router and weight similarity
        combined = np.sqrt(np.maximum(r_sim, 0) * np.maximum(w_sim, 0))
        np.fill_diagonal(combined, 0)

        # Top pairs by combined score
        flat_idx = np.argsort(combined.flatten())[::-1]
        top_pairs = []
        seen = set()
        for idx in flat_idx:
            i, j = divmod(idx, NUM_EXPERTS)
            if i >= j:
                continue
            pair = (int(i), int(j))
            if pair not in seen:
                seen.add(pair)
                top_pairs.append({
                    "expert_a": int(i),
                    "expert_b": int(j),
                    "combined_score": float(combined[i, j]),
                    "router_sim": float(r_sim[i, j]),
                    "weight_sim": float(w_sim[i, j]),
                })
            if len(top_pairs) >= 10:
                break

        candidates_per_layer[layer_idx] = top_pairs

    return candidates_per_layer


def main():
    print("=" * 80)
    print("Expert Pairwise Similarity Analysis: Gemma4 26B MoE")
    print("=" * 80)

    t0 = time.time()
    sf = safe_open(MODEL_PATH, framework="pt", device="cpu")

    # Router vector similarity
    print("\n[1/3] Router vector similarity...")
    t1 = time.time()
    router_sim = compute_router_vector_similarity(sf)
    print(f"  Done in {time.time()-t1:.1f}s")

    # Weight fingerprint similarity
    print("\n[2/3] Expert weight fingerprint similarity...")
    t1 = time.time()
    weight_sim = compute_expert_weight_fingerprint_similarity(sf)
    print(f"  Done in {time.time()-t1:.1f}s")

    # Combined merge candidates
    print("\n[3/3] Combined merge candidate analysis...")
    merge_candidates = find_merge_candidates(router_sim, weight_sim)

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # ==================== REPORT ====================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Router similarity summary
    print("\n--- Router Vector Similarity (per layer) ---")
    print(f"{'Layer':>5} {'Mean':>7} {'Max':>7} {'Std':>7} {'>0.9':>6} {'>0.8':>6} {'>0.7':>6} {'Top Pair':>20}")
    print("-" * 75)
    for layer_idx in range(NUM_LAYERS):
        r = router_sim[layer_idx]
        tp = r["top_pairs"][0] if r["top_pairs"] else (0, 0, 0)
        print(f"{layer_idx:>5} {r['mean_sim']:>7.3f} {r['max_sim']:>7.3f} {r['std_sim']:>7.3f} "
              f"{r['pairs_above_0.9']:>6} {r['pairs_above_0.8']:>6} {r['pairs_above_0.7']:>6} "
              f"E{tp[0]:>3}-E{tp[1]:>3} ({tp[2]:.3f})")

    # Weight fingerprint similarity summary
    print("\n--- Weight Fingerprint Similarity (per layer) ---")
    print(f"{'Layer':>5} {'Mean':>7} {'Max':>7} {'Std':>7} {'>0.95':>6} {'>0.9':>6} {'>0.8':>6} {'Top Pair':>20}")
    print("-" * 75)
    for layer_idx in range(NUM_LAYERS):
        w = weight_sim[layer_idx]
        tp = w["top_pairs"][0] if w["top_pairs"] else (0, 0, 0)
        print(f"{layer_idx:>5} {w['mean_sim']:>7.3f} {w['max_sim']:>7.3f} {w['std_sim']:>7.3f} "
              f"{w['pairs_above_0.95']:>6} {w['pairs_above_0.9']:>6} {w['pairs_above_0.8']:>6} "
              f"E{tp[0]:>3}-E{tp[1]:>3} ({tp[2]:.3f})")

    # Top merge candidates across all layers
    print("\n--- Top Merge Candidates (Combined Score) ---")
    all_candidates = []
    for layer_idx in range(NUM_LAYERS):
        for c in merge_candidates[layer_idx]:
            c["layer"] = layer_idx
            all_candidates.append(c)

    all_candidates.sort(key=lambda x: x["combined_score"], reverse=True)
    print(f"{'Layer':>5} {'Expert A':>9} {'Expert B':>9} {'Combined':>9} {'Router':>8} {'Weight':>8}")
    print("-" * 55)
    for c in all_candidates[:30]:
        print(f"{c['layer']:>5} E{c['expert_a']:>7} E{c['expert_b']:>7} "
              f"{c['combined_score']:>9.4f} {c['router_sim']:>8.4f} {c['weight_sim']:>8.4f}")

    # Save results (without large matrices)
    output = {
        "router_similarity": {},
        "weight_similarity": {},
        "merge_candidates": {},
    }
    for layer_idx in range(NUM_LAYERS):
        output["router_similarity"][str(layer_idx)] = {
            k: v for k, v in router_sim[layer_idx].items() if k != "sim_matrix"
        }
        output["weight_similarity"][str(layer_idx)] = {
            k: v for k, v in weight_sim[layer_idx].items() if k != "sim_matrix"
        }
        output["merge_candidates"][str(layer_idx)] = merge_candidates[layer_idx]

    out_path = Path(__file__).parent / "expert_similarity_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")

    return output


if __name__ == "__main__":
    results = main()
