#!/usr/bin/env python3
"""
Layer importance analysis for Gemma 4 26B-A4B NVFP4 model.

Computes per-layer metrics from safetensors weights:
  1. Weight magnitude (L2 norm of all tensors per layer)
  2. Weight entropy (information content)
  3. Adjacent layer similarity (cosine similarity of flattened weight vectors)
  4. Layer scalar values (residual scaling factors)
  5. Router weight analysis (expert routing diversity)

Outputs a ranked list of layers by pruning priority (least important first).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import time

MODEL_DIR = "/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt"

def load_config():
    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)
    text_cfg = config["text_config"]
    layer_types = text_cfg["layer_types"]
    n_layers = text_cfg["num_hidden_layers"]
    return config, text_cfg, layer_types, n_layers


def load_layer_weights():
    """Load all per-layer weights from safetensors using torch (supports fp8 dtypes)."""
    from safetensors import safe_open

    model_path = f"{MODEL_DIR}/model.safetensors"
    print(f"Loading weights from {model_path}...")
    t0 = time.time()

    sf = safe_open(model_path, framework="pt")
    keys = sf.keys()

    # Group weights by layer
    layer_weights = defaultdict(dict)  # layer_idx -> {key_suffix: tensor}
    other_weights = {}

    import re
    layer_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.(.*)")

    for key in keys:
        m = layer_pattern.match(key)
        if m:
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            layer_weights[layer_idx][suffix] = sf.get_tensor(key)
        else:
            other_weights[key] = key  # just track existence

    print(f"  Loaded in {time.time()-t0:.1f}s, {len(layer_weights)} layers found")
    return sf, layer_weights


def to_float32(tensor):
    """Convert any tensor (including fp8) to float32."""
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return tensor.view(torch.uint8).to(torch.float32)
    return tensor.to(torch.float32)


def compute_weight_magnitude(layer_weights):
    """Compute L2 norm of all weight tensors per layer."""
    print("\n=== Weight Magnitude (L2 Norm) ===")
    magnitudes = {}

    for layer_idx in sorted(layer_weights.keys()):
        weights = layer_weights[layer_idx]
        total_l2 = 0.0
        n_params = 0

        for suffix, tensor in weights.items():
            # Use weight_packed (actual weights) and weight_scale, skip global scales
            if "weight_packed" in suffix or "weight_scale" in suffix or suffix.endswith(".weight"):
                vals = to_float32(tensor).ravel()
                total_l2 += float(torch.sum(vals ** 2))
                n_params += vals.numel()

        magnitudes[layer_idx] = {
            "l2_norm": float(np.sqrt(total_l2)),
            "rms": float(np.sqrt(total_l2 / max(n_params, 1))),
            "n_params": n_params,
        }

    # Print ranked
    ranked = sorted(magnitudes.items(), key=lambda x: x[1]["l2_norm"])
    for layer_idx, m in ranked:
        print(f"  Layer {layer_idx:2d}: L2={m['l2_norm']:10.2f}  RMS={m['rms']:.6f}  params={m['n_params']:,}")

    return magnitudes


def compute_layer_scalar(layer_weights):
    """Extract layer_scalar values (residual stream scaling)."""
    print("\n=== Layer Scalars (Residual Scaling) ===")
    scalars = {}

    for layer_idx in sorted(layer_weights.keys()):
        weights = layer_weights[layer_idx]
        if "layer_scalar" in weights:
            val = float(weights["layer_scalar"].reshape(-1)[0])
            scalars[layer_idx] = val
            print(f"  Layer {layer_idx:2d}: scalar = {val:.6f}")
        else:
            scalars[layer_idx] = None
            print(f"  Layer {layer_idx:2d}: no layer_scalar")

    return scalars


def compute_adjacent_similarity(layer_weights):
    """Compute cosine similarity between adjacent layers' weight vectors."""
    print("\n=== Adjacent Layer Similarity (Cosine) ===")

    # Build per-layer flattened vectors from comparable weight components
    # Use attention q/k/v/o weight_packed and MLP weight_packed
    layer_vectors = {}

    # Use components that exist in ALL layers (both sliding and full attention)
    # MLP + router weights are identical structure across all layers
    # For attention: use weight_scale tensors (small, same size across layer types)
    component_prefixes = [
        "mlp.gate_proj.weight_packed",
        "mlp.up_proj.weight_packed",
        "mlp.down_proj.weight_packed",
        "self_attn.q_proj.weight_scale",
        "self_attn.k_proj.weight_scale",
        "self_attn.v_proj.weight_scale",
        "self_attn.o_proj.weight_scale",
        "mlp.gate_proj.weight_scale",
        "mlp.up_proj.weight_scale",
        "mlp.down_proj.weight_scale",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]

    for layer_idx in sorted(layer_weights.keys()):
        weights = layer_weights[layer_idx]
        parts = []
        for ck in component_prefixes:
            if ck in weights:
                parts.append(to_float32(weights[ck]).reshape(-1))
        if parts:
            layer_vectors[layer_idx] = torch.cat(parts).numpy()

    similarities = {}
    layer_indices = sorted(layer_vectors.keys())

    for i in range(len(layer_indices) - 1):
        idx_a = layer_indices[i]
        idx_b = layer_indices[i + 1]
        va = layer_vectors[idx_a]
        vb = layer_vectors[idx_b]

        # Ensure same length (they should be for sliding-sliding pairs)
        min_len = min(len(va), len(vb))
        va = va[:min_len]
        vb = vb[:min_len]

        dot = np.dot(va, vb)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        cos_sim = dot / (norm_a * norm_b + 1e-10)

        similarities[(idx_a, idx_b)] = float(cos_sim)
        print(f"  Layer {idx_a:2d} <-> {idx_b:2d}: cosine = {cos_sim:.6f}")

    return similarities


def compute_router_entropy(layer_weights):
    """Analyze router weight diversity per layer (expert selection patterns)."""
    print("\n=== Router Weight Analysis ===")
    router_stats = {}

    for layer_idx in sorted(layer_weights.keys()):
        weights = layer_weights[layer_idx]

        router_proj = weights.get("router.proj.weight")
        per_expert_scale = weights.get("router.per_expert_scale")
        router_scale = weights.get("router.scale")

        if router_proj is not None:
            rp = to_float32(router_proj)
            frob = float(torch.norm(rp))

            # Expert scale variance - if all experts have similar scale, routing is uniform
            if per_expert_scale is not None:
                es = to_float32(per_expert_scale).ravel()
                es_std = float(torch.std(es))
                es_mean = float(torch.mean(es))
                es_cv = es_std / (abs(es_mean) + 1e-10)
            else:
                es_std = es_mean = es_cv = 0.0

            router_stats[layer_idx] = {
                "router_frob": frob,
                "expert_scale_mean": es_mean,
                "expert_scale_std": es_std,
                "expert_scale_cv": es_cv,
            }
            print(f"  Layer {layer_idx:2d}: router_frob={frob:.2f}  expert_scale_cv={es_cv:.4f}")

    return router_stats


def compute_weight_entropy(layer_weights):
    """Compute entropy of weight distributions per layer."""
    print("\n=== Weight Distribution Entropy ===")
    entropies = {}

    for layer_idx in sorted(layer_weights.keys()):
        weights = layer_weights[layer_idx]
        all_vals = []

        for suffix, tensor in weights.items():
            if "weight_packed" in suffix:
                all_vals.append(tensor.reshape(-1))

        if not all_vals:
            entropies[layer_idx] = 0.0
            continue

        combined = torch.cat(all_vals)
        # View as uint8 bytes for entropy calculation (works for any dtype)
        raw_bytes = combined.contiguous().view(torch.uint8).numpy()
        unique, counts = np.unique(raw_bytes, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        entropies[layer_idx] = float(entropy)
        print(f"  Layer {layer_idx:2d}: entropy = {entropy:.4f} bits (over {len(unique)} unique values)")

    return entropies


def compute_importance_score(magnitudes, scalars, similarities, router_stats, entropies, layer_types):
    """Compute composite importance score per layer.

    Lower score = more prunable.

    Factors:
    - Weight magnitude (normalized): higher = more important
    - Layer scalar: higher = more important (scales residual contribution)
    - Entropy: higher = more information = more important
    - Adjacent similarity: if similar to neighbor, more prunable
    - Layer type: full_attention layers get importance boost (global context)
    - Position: first/last layers get importance boost (embedding/output interface)
    """
    print("\n=== Composite Importance Scores ===")

    n_layers = len(magnitudes)

    # Normalize each metric to [0, 1]
    def normalize(vals):
        arr = np.array(vals, dtype=np.float64)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.ones_like(arr) * 0.5
        return (arr - mn) / (mx - mn)

    layer_indices = sorted(magnitudes.keys())

    # L2 norms
    l2_vals = [magnitudes[i]["l2_norm"] for i in layer_indices]
    l2_norm = normalize(l2_vals)

    # Scalars
    scalar_vals = [scalars.get(i, 0.0) or 0.0 for i in layer_indices]
    scalar_norm = normalize(scalar_vals)

    # Entropy
    entropy_vals = [entropies.get(i, 0.0) for i in layer_indices]
    entropy_norm = normalize(entropy_vals)

    # Router diversity (CV of expert scales)
    router_cv_vals = [router_stats.get(i, {}).get("expert_scale_cv", 0.0) for i in layer_indices]
    router_norm = normalize(router_cv_vals)

    # Adjacent similarity penalty: avg similarity to neighbors makes layer more prunable
    sim_penalty = np.zeros(n_layers)
    for i, idx in enumerate(layer_indices):
        sims = []
        if i > 0:
            pair = (layer_indices[i-1], idx)
            if pair in similarities:
                sims.append(similarities[pair])
        if i < n_layers - 1:
            pair = (idx, layer_indices[i+1])
            if pair in similarities:
                sims.append(similarities[pair])
        if sims:
            sim_penalty[i] = np.mean(sims)  # higher similarity = more prunable
    sim_penalty_norm = normalize(sim_penalty)

    # Layer type bonus: full_attention = 0.15 boost
    type_bonus = np.array([0.15 if layer_types[i] == "full_attention" else 0.0 for i in layer_indices])

    # Position bonus: first 3 and last 3 layers get boost
    pos_bonus = np.zeros(n_layers)
    for i in range(min(3, n_layers)):
        pos_bonus[i] = 0.10 * (3 - i) / 3
    for i in range(max(0, n_layers - 3), n_layers):
        pos_bonus[i] = 0.10 * (i - (n_layers - 4)) / 3

    # Composite score (higher = more important)
    # Weights chosen to balance different signals
    importance = (
        0.30 * l2_norm +
        0.20 * scalar_norm +
        0.15 * entropy_norm +
        0.10 * router_norm +
        -0.15 * sim_penalty_norm +  # similarity to neighbor reduces importance
        type_bonus +
        pos_bonus
    )

    scores = {}
    for i, idx in enumerate(layer_indices):
        scores[idx] = {
            "importance": float(importance[i]),
            "l2_component": float(l2_norm[i]),
            "scalar_component": float(scalar_norm[i]),
            "entropy_component": float(entropy_norm[i]),
            "router_component": float(router_norm[i]),
            "sim_penalty": float(sim_penalty_norm[i]),
            "type_bonus": float(type_bonus[i]),
            "pos_bonus": float(pos_bonus[i]),
            "layer_type": layer_types[idx],
        }

    # Print ranked by importance (least important first = best pruning candidates)
    ranked = sorted(scores.items(), key=lambda x: x[1]["importance"])
    print(f"\n{'Rank':>4} {'Layer':>5} {'Type':>18} {'Importance':>10} {'L2':>6} {'Scalar':>7} {'Entropy':>7} {'Router':>7} {'SimPen':>7}")
    print("-" * 95)
    for rank, (idx, s) in enumerate(ranked, 1):
        print(f"{rank:4d} {idx:5d} {s['layer_type']:>18} {s['importance']:10.4f} "
              f"{s['l2_component']:6.3f} {s['scalar_component']:7.3f} {s['entropy_component']:7.3f} "
              f"{s['router_component']:7.3f} {s['sim_penalty']:7.3f}")

    return scores, ranked


def generate_pruning_recommendations(ranked, layer_types, similarities):
    """Generate specific pruning recommendations."""
    print("\n" + "=" * 80)
    print("LAYER PRUNING RECOMMENDATIONS")
    print("=" * 80)

    # Global attention layers
    global_attn_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    print(f"\nGlobal attention layers (DO NOT PRUNE): {global_attn_layers}")

    # Pruning candidates (exclude global attention, first 2, last 2)
    protected = set(global_attn_layers) | {0, 1, 28, 29}
    candidates = [(idx, s) for idx, s in ranked if idx not in protected]

    print(f"\nProtected layers (first/last + global attention): {sorted(protected)}")
    print(f"Prunable candidates: {len(candidates)} layers")

    # Top pruning recommendations
    print("\n--- Pruning Priority (most prunable first) ---")
    for i, (idx, s) in enumerate(candidates[:10]):
        print(f"  {i+1}. Layer {idx:2d} ({s['layer_type']}) - importance={s['importance']:.4f}")

    # Pruning tiers
    print("\n--- Pruning Tiers ---")
    tier1 = [idx for idx, s in candidates[:1]]
    tier3 = [idx for idx, s in candidates[:3]]
    tier5 = [idx for idx, s in candidates[:5]]

    print(f"  Tier 1 (remove 1 layer, ~3% speedup):  layers {tier1}")
    print(f"  Tier 2 (remove 3 layers, ~10% speedup): layers {tier3}")
    print(f"  Tier 3 (remove 5 layers, ~17% speedup): layers {tier5}")

    # Estimate speedup
    per_layer_ms = 0.48  # measured decode step time per layer
    total_decode_ms = 30 * per_layer_ms
    for name, layers_to_remove in [("Tier 1", tier1), ("Tier 2", tier3), ("Tier 3", tier5)]:
        n = len(layers_to_remove)
        saved_ms = n * per_layer_ms
        new_ms = total_decode_ms - saved_ms
        speedup = total_decode_ms / new_ms
        new_toks = 1000.0 / new_ms
        print(f"  {name}: remove {n} layers, save {saved_ms:.1f}ms/step, "
              f"speedup {speedup:.2f}x, ~{new_toks:.1f} tok/s")

    return {
        "tier1": tier1,
        "tier3": tier3,
        "tier5": tier5,
        "protected": sorted(protected),
    }


def main():
    config, text_cfg, layer_types, n_layers = load_config()
    print(f"Model: Gemma 4 26B-A4B-it NVFP4")
    print(f"Layers: {n_layers}")
    print(f"Layer types: {sum(1 for t in layer_types if t == 'sliding_attention')} sliding + "
          f"{sum(1 for t in layer_types if t == 'full_attention')} full_attention")
    print(f"Full attention at positions: {[i for i, t in enumerate(layer_types) if t == 'full_attention']}")
    print(f"MoE: {text_cfg['num_experts']} experts, top-{text_cfg['top_k_experts']}")

    sf, layer_weights = load_layer_weights()

    magnitudes = compute_weight_magnitude(layer_weights)
    scalars = compute_layer_scalar(layer_weights)
    similarities = compute_adjacent_similarity(layer_weights)
    router_stats = compute_router_entropy(layer_weights)
    entropies = compute_weight_entropy(layer_weights)

    scores, ranked = compute_importance_score(
        magnitudes, scalars, similarities, router_stats, entropies, layer_types
    )

    recommendations = generate_pruning_recommendations(ranked, layer_types, similarities)

    # Save results as JSON
    output = {
        "model": "gemma-4-26B-A4B-it-NVFP4",
        "n_layers": n_layers,
        "layer_types": layer_types,
        "magnitudes": magnitudes,
        "scalars": scalars,
        "similarities": {f"{a}-{b}": v for (a, b), v in similarities.items()},
        "router_stats": router_stats,
        "entropies": entropies,
        "importance_scores": scores,
        "ranking": [(idx, s["importance"]) for idx, s in ranked],
        "recommendations": recommendations,
    }

    out_path = "/root/projects/autokernel/profiling/layer_importance.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return scores, ranked, recommendations


if __name__ == "__main__":
    main()
