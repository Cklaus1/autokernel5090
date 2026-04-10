#!/usr/bin/env python3
"""
Expert Activation Profiling for Gemma4 26B MoE (128 experts, top-8, 30 layers).

Loads router weights from checkpoint, simulates routing with diverse hidden states,
and produces per-layer activation histograms + pruning candidates.
"""

import json
import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from pathlib import Path
import time

MODEL_PATH = "/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/model.safetensors"
CONFIG_PATH = "/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/config.json"
NUM_LAYERS = 30
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_SIZE = 2816

# Number of simulated tokens for routing analysis
NUM_TOKENS = 10000


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_router_weights(sf):
    """Load all router weights into memory."""
    routers = {}
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{layer_idx}.router"
        w = sf.get_tensor(f"{prefix}.proj.weight").float()       # [128, 2816]
        scale = sf.get_tensor(f"{prefix}.scale").float()          # [2816]
        per_expert_scale = sf.get_tensor(f"{prefix}.per_expert_scale").float()  # [128]
        routers[layer_idx] = {
            "weight": w,
            "scale": scale,
            "per_expert_scale": per_expert_scale,
        }
    return routers


def simulate_routing(routers, hidden_states):
    """
    Simulate top-k routing for all layers.

    Gemma4 router: logits = (hidden * scale) @ W^T * per_expert_scale
    Then top-k selection with softmax over selected experts.

    Args:
        routers: dict of router weights per layer
        hidden_states: [num_tokens, hidden_size] tensor

    Returns:
        activations: dict[layer_idx] -> [num_tokens, top_k] expert indices
        all_logits: dict[layer_idx] -> [num_tokens, num_experts] raw logits
    """
    activations = {}
    all_logits = {}

    for layer_idx in range(NUM_LAYERS):
        r = routers[layer_idx]
        w = r["weight"]           # [128, 2816]
        scale = r["scale"]        # [2816]
        pes = r["per_expert_scale"]  # [128]

        # Apply input scaling (RMSNorm-like)
        h_scaled = hidden_states * scale.unsqueeze(0)  # [N, 2816]

        # Router logits
        logits = h_scaled @ w.T  # [N, 128]

        # Per-expert scaling
        logits = logits * pes.unsqueeze(0)  # [N, 128]

        # Top-k selection
        topk_vals, topk_idx = torch.topk(logits, TOP_K, dim=-1)  # [N, 8]

        activations[layer_idx] = topk_idx.numpy()
        all_logits[layer_idx] = logits.detach().numpy()

    return activations, all_logits


def generate_diverse_hidden_states(num_tokens, hidden_size, seed=42):
    """
    Generate diverse hidden states that mimic real transformer activations.
    Mix of: standard normal, heavy-tailed, sparse, clustered.
    """
    rng = np.random.RandomState(seed)

    states = []
    n_per_type = num_tokens // 4

    # 1. Standard normal (general text)
    s1 = rng.randn(n_per_type, hidden_size).astype(np.float32)
    states.append(s1)

    # 2. Heavy-tailed (outlier features common in transformers)
    s2 = rng.randn(n_per_type, hidden_size).astype(np.float32)
    # Add outlier channels (mimics real LLM hidden states)
    outlier_dims = rng.choice(hidden_size, size=20, replace=False)
    s2[:, outlier_dims] *= 10.0
    states.append(s2)

    # 3. Sparse activations (ReLU-like)
    s3 = rng.randn(n_per_type, hidden_size).astype(np.float32)
    mask = rng.rand(n_per_type, hidden_size) < 0.7
    s3[mask] = 0.0
    states.append(s3)

    # 4. Clustered (tokens from similar contexts)
    n_clusters = 50
    centers = rng.randn(n_clusters, hidden_size).astype(np.float32) * 2.0
    remaining = num_tokens - 3 * n_per_type
    cluster_ids = rng.randint(0, n_clusters, size=remaining)
    s4 = centers[cluster_ids] + rng.randn(remaining, hidden_size).astype(np.float32) * 0.3
    states.append(s4)

    all_states = np.concatenate(states, axis=0)
    # Shuffle
    perm = rng.permutation(len(all_states))
    all_states = all_states[perm]

    return torch.from_numpy(all_states)


def compute_activation_stats(activations):
    """Compute per-layer activation frequency histograms."""
    stats = {}
    for layer_idx in range(NUM_LAYERS):
        expert_counts = np.zeros(NUM_EXPERTS, dtype=np.int64)
        indices = activations[layer_idx]  # [N, top_k]
        for expert_id in indices.flatten():
            expert_counts[expert_id] += 1

        total_activations = indices.shape[0] * TOP_K
        freq = expert_counts / total_activations

        stats[layer_idx] = {
            "counts": expert_counts,
            "freq": freq,
            "total_activations": total_activations,
            "active_experts": int(np.sum(expert_counts > 0)),
            "top10_experts": np.argsort(expert_counts)[::-1][:10].tolist(),
            "top10_freq": freq[np.argsort(freq)[::-1][:10]].tolist(),
            "bottom10_experts": np.argsort(expert_counts)[:10].tolist(),
            "bottom10_freq": freq[np.argsort(freq)[:10]].tolist(),
            "gini": compute_gini(freq),
            "entropy": compute_entropy(freq),
        }
    return stats


def compute_gini(freq):
    """Gini coefficient: 0=perfect equality, 1=max inequality."""
    sorted_freq = np.sort(freq)
    n = len(sorted_freq)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_freq) / (n * np.sum(sorted_freq))) - (n + 1) / n)


def compute_entropy(freq):
    """Shannon entropy of activation distribution."""
    freq = freq[freq > 0]
    return float(-np.sum(freq * np.log2(freq)))


def find_pruning_candidates(stats, threshold=0.01):
    """Find experts activated less than threshold fraction."""
    candidates = {}
    for layer_idx in range(NUM_LAYERS):
        freq = stats[layer_idx]["freq"]
        # Uniform would be TOP_K/NUM_EXPERTS = 8/128 = 0.0625
        # threshold=0.01 means < 1% of total activations
        low_experts = np.where(freq < threshold)[0].tolist()
        candidates[layer_idx] = {
            "experts": low_experts,
            "count": len(low_experts),
            "freqs": freq[np.array(low_experts)].tolist() if low_experts else [],
        }
    return candidates


def find_hot_set(stats, top_n=32):
    """Find the hot set of experts per layer."""
    hot_sets = {}
    for layer_idx in range(NUM_LAYERS):
        freq = stats[layer_idx]["freq"]
        top_experts = np.argsort(freq)[::-1][:top_n]
        hot_sets[layer_idx] = {
            "experts": top_experts.tolist(),
            "combined_freq": float(freq[top_experts].sum()),
        }
    return hot_sets


def cross_layer_analysis(stats, hot_sets):
    """Analyze whether same experts are hot across layers."""
    # Count how many layers each expert appears in the top-32
    expert_layer_count = np.zeros(NUM_EXPERTS, dtype=np.int64)
    for layer_idx in range(NUM_LAYERS):
        for e in hot_sets[layer_idx]["experts"]:
            expert_layer_count[e] += 1

    # Experts that are hot in ALL or MOST layers
    universal_hot = np.where(expert_layer_count >= 25)[0]  # hot in 25+ layers
    never_hot = np.where(expert_layer_count == 0)[0]

    # Jaccard similarity between consecutive layers' hot sets
    jaccard_consecutive = []
    for i in range(NUM_LAYERS - 1):
        s1 = set(hot_sets[i]["experts"])
        s2 = set(hot_sets[i + 1]["experts"])
        jaccard = len(s1 & s2) / len(s1 | s2)
        jaccard_consecutive.append(jaccard)

    return {
        "expert_layer_count": expert_layer_count.tolist(),
        "universal_hot_experts": universal_hot.tolist(),
        "never_hot_experts": never_hot.tolist(),
        "jaccard_consecutive": jaccard_consecutive,
        "mean_jaccard": float(np.mean(jaccard_consecutive)),
    }


def estimate_pruning_impact(stats, prune_fractions=[0.10, 0.20, 0.30]):
    """Estimate compute/memory savings from pruning."""
    results = {}

    # Per-expert memory: gate_proj + up_proj + down_proj
    # gate: [704, 2816], up: [704, 2816], down: [2816, 704]
    # NVFP4: ~0.5 bytes per param (4-bit with scales)
    expert_params = 704 * 2816 + 704 * 2816 + 2816 * 704  # = 3 * 704 * 2816
    expert_bytes_nvfp4 = expert_params * 0.5  # rough NVFP4 size
    expert_mb = expert_bytes_nvfp4 / (1024 * 1024)
    total_expert_mb = expert_mb * NUM_EXPERTS * NUM_LAYERS

    for frac in prune_fractions:
        n_prune = int(NUM_EXPERTS * frac)

        # For each layer, find the n_prune least-activated experts
        total_pruned = 0
        activation_loss = 0.0  # fraction of activations that would be affected

        for layer_idx in range(NUM_LAYERS):
            freq = stats[layer_idx]["freq"]
            sorted_idx = np.argsort(freq)
            pruned = sorted_idx[:n_prune]
            total_pruned += n_prune
            activation_loss += freq[pruned].sum()

        activation_loss /= NUM_LAYERS  # average across layers

        memory_saved_mb = total_pruned * expert_mb
        memory_saved_pct = total_pruned / (NUM_EXPERTS * NUM_LAYERS) * 100

        results[frac] = {
            "n_prune_per_layer": n_prune,
            "total_pruned": total_pruned,
            "memory_saved_mb": float(memory_saved_mb),
            "memory_saved_pct": float(memory_saved_pct),
            "avg_activation_loss": float(activation_loss),
            "expert_mb": float(expert_mb),
            "total_expert_mb": float(total_expert_mb),
        }

    return results


def analyze_router_weight_norms(sf):
    """Analyze router weight norms — experts with small router norms get selected less."""
    norms_per_layer = {}
    for layer_idx in range(NUM_LAYERS):
        w = sf.get_tensor(f"model.language_model.layers.{layer_idx}.router.proj.weight").float()
        # w: [128, 2816], each row is one expert's routing vector
        row_norms = torch.norm(w, dim=1).numpy()  # [128]
        norms_per_layer[layer_idx] = row_norms
    return norms_per_layer


def main():
    print("=" * 80)
    print("Expert Activation Profiling: Gemma4 26B MoE")
    print(f"  128 experts, top-8 routing, 30 layers, hidden_size=2816")
    print(f"  Simulating with {NUM_TOKENS} diverse tokens")
    print("=" * 80)

    # Load router weights
    print("\n[1/6] Loading router weights...")
    t0 = time.time()
    sf = safe_open(MODEL_PATH, framework="pt", device="cpu")
    routers = load_router_weights(sf)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Generate hidden states
    print("\n[2/6] Generating diverse hidden states...")
    hidden_states = generate_diverse_hidden_states(NUM_TOKENS, HIDDEN_SIZE)
    print(f"  Shape: {hidden_states.shape}")

    # Simulate routing
    print("\n[3/6] Simulating top-8 routing across 30 layers...")
    t0 = time.time()
    activations, all_logits = simulate_routing(routers, hidden_states)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Compute stats
    print("\n[4/6] Computing activation statistics...")
    stats = compute_activation_stats(activations)

    # Also run with more diverse seeds for robustness
    print("  Running 4 additional seeds for robustness...")
    all_stats_freq = {layer_idx: stats[layer_idx]["freq"].copy() for layer_idx in range(NUM_LAYERS)}
    for seed in [123, 456, 789, 1337]:
        hs = generate_diverse_hidden_states(NUM_TOKENS, HIDDEN_SIZE, seed=seed)
        act2, _ = simulate_routing(routers, hs)
        st2 = compute_activation_stats(act2)
        for layer_idx in range(NUM_LAYERS):
            all_stats_freq[layer_idx] += st2[layer_idx]["freq"]

    # Average frequencies across 5 seeds
    for layer_idx in range(NUM_LAYERS):
        all_stats_freq[layer_idx] /= 5.0
        stats[layer_idx]["freq_avg"] = all_stats_freq[layer_idx]

    # Hot set analysis
    print("\n[5/6] Finding hot sets and pruning candidates...")
    hot_sets = find_hot_set(stats, top_n=32)
    pruning_candidates = find_pruning_candidates(stats, threshold=0.01)
    cross_layer = cross_layer_analysis(stats, hot_sets)
    pruning_impact = estimate_pruning_impact(stats)

    # Router weight norms
    print("\n[6/6] Analyzing router weight norms...")
    router_norms = analyze_router_weight_norms(sf)

    # ==================== REPORT ====================
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Per-layer summary
    print("\n--- Per-Layer Activation Summary ---")
    print(f"{'Layer':>5} {'Active':>7} {'Gini':>6} {'Entropy':>8} {'Top32%':>7} {'<1%cnt':>7} {'Top Expert':>11} {'TopFreq':>8}")
    print("-" * 75)
    for layer_idx in range(NUM_LAYERS):
        s = stats[layer_idx]
        hs = hot_sets[layer_idx]
        pc = pruning_candidates[layer_idx]
        top_expert = s["top10_experts"][0]
        top_freq = s["top10_freq"][0]
        print(f"{layer_idx:>5} {s['active_experts']:>7} {s['gini']:>6.3f} {s['entropy']:>8.3f} "
              f"{hs['combined_freq']:>6.1%} {pc['count']:>7} "
              f"E{top_expert:>3} {top_freq:>7.3%}")

    # Uniform baseline
    uniform_entropy = np.log2(NUM_EXPERTS)
    print(f"\nUniform entropy baseline: {uniform_entropy:.3f} bits")
    print(f"Uniform per-expert freq: {TOP_K/NUM_EXPERTS:.4f} (6.25%)")

    # Cross-layer analysis
    print("\n--- Cross-Layer Analysis ---")
    print(f"Mean Jaccard similarity (consecutive layers): {cross_layer['mean_jaccard']:.3f}")
    print(f"Universal hot experts (in 25+/30 layers): {len(cross_layer['universal_hot_experts'])}")
    if cross_layer['universal_hot_experts']:
        print(f"  Expert IDs: {cross_layer['universal_hot_experts'][:20]}")
    print(f"Never-hot experts (not in top-32 of any layer): {len(cross_layer['never_hot_experts'])}")
    if cross_layer['never_hot_experts']:
        print(f"  Expert IDs: {cross_layer['never_hot_experts'][:20]}")

    # Pruning impact
    print("\n--- Pruning Impact Estimates ---")
    for frac, data in sorted(pruning_impact.items()):
        print(f"\nPrune bottom {frac:.0%} ({data['n_prune_per_layer']} experts/layer, {data['total_pruned']} total):")
        print(f"  Memory saved: {data['memory_saved_mb']:.0f} MB ({data['memory_saved_pct']:.1f}%)")
        print(f"  Avg activation loss: {data['avg_activation_loss']:.4f} ({data['avg_activation_loss']:.2%} of routing selections)")
        print(f"  Per-expert size: {data['expert_mb']:.2f} MB (NVFP4)")
        print(f"  Total expert memory: {data['total_expert_mb']:.0f} MB")

    # Router weight norm analysis
    print("\n--- Router Weight Norm Analysis ---")
    print("Correlation between router norm and activation frequency:")
    correlations = []
    for layer_idx in range(NUM_LAYERS):
        norm = router_norms[layer_idx]
        freq = stats[layer_idx]["freq"]
        corr = np.corrcoef(norm, freq)[0, 1]
        correlations.append(corr)
    print(f"  Mean correlation: {np.mean(correlations):.3f}")
    print(f"  Min/Max: {np.min(correlations):.3f} / {np.max(correlations):.3f}")

    # Per-layer Jaccard
    print("\n--- Consecutive Layer Jaccard Similarities ---")
    for i, j in enumerate(cross_layer['jaccard_consecutive']):
        print(f"  Layer {i} <-> {i+1}: {j:.3f}")

    # Save detailed results
    output = {
        "config": {
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_tokens_simulated": NUM_TOKENS,
            "num_seeds": 5,
        },
        "per_layer": {},
        "cross_layer": cross_layer,
        "pruning_impact": {str(k): v for k, v in pruning_impact.items()},
    }

    for layer_idx in range(NUM_LAYERS):
        s = stats[layer_idx]
        output["per_layer"][layer_idx] = {
            "freq": s["freq"].tolist(),
            "freq_avg": s["freq_avg"].tolist(),
            "gini": s["gini"],
            "entropy": s["entropy"],
            "active_experts": s["active_experts"],
            "top10_experts": s["top10_experts"],
            "top10_freq": s["top10_freq"],
            "bottom10_experts": s["bottom10_experts"],
            "bottom10_freq": s["bottom10_freq"],
            "hot_set_32": hot_sets[layer_idx]["experts"],
            "hot_set_32_combined_freq": hot_sets[layer_idx]["combined_freq"],
            "pruning_candidates_1pct": pruning_candidates[layer_idx]["experts"],
            "router_norm": router_norms[layer_idx].tolist(),
        }

    out_path = Path(__file__).parent / "expert_activation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")

    return output


def main_real_embeddings():
    """Run routing analysis using real token embeddings (more accurate than random)."""
    print("\n" + "=" * 80)
    print("REAL TOKEN EMBEDDING ANALYSIS")
    print("=" * 80)

    sf = safe_open(MODEL_PATH, framework="pt", device="cpu")
    embed_w = sf.get_tensor("model.language_model.embed_tokens.weight").float()

    torch.manual_seed(42)
    sample_idx = torch.randint(0, embed_w.shape[0], (NUM_TOKENS,))
    real_embeds = embed_w[sample_idx]

    routers = load_router_weights(sf)
    activations, all_logits = simulate_routing(routers, real_embeds)
    stats = compute_activation_stats(activations)
    hot_sets = find_hot_set(stats, top_n=32)
    pruning_candidates = find_pruning_candidates(stats, threshold=0.01)

    print("\n--- Per-Layer Summary (Real Embeddings) ---")
    print(f"{'Layer':>5} {'Gini':>6} {'Entropy':>8} {'Top32%':>7} {'<1%cnt':>7}")
    for layer_idx in range(NUM_LAYERS):
        s = stats[layer_idx]
        hs = hot_sets[layer_idx]
        pc = pruning_candidates[layer_idx]
        print(f"{layer_idx:>5} {s['gini']:>6.3f} {s['entropy']:>8.3f} "
              f"{hs['combined_freq']:>6.1%} {pc['count']:>7}")

    return stats


if __name__ == "__main__":
    results = main()
    real_results = main_real_embeddings()
