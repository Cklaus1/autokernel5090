#!/usr/bin/env python3
"""
Analyze what pruned model components (experts and layers) actually DO.

Four approaches implemented using checkpoint-level analysis (no model loading):
  1. Expert Activation Probing via Router Weights
  2. Layer Ablation via Weight Statistics
  3. Expert Output Fingerprinting
  4. Differential Checkpoint Analysis (pruned vs full)

Usage:
    # Full analysis with all approaches
    python tools/analyze_pruned_components.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --all

    # Individual approaches
    python tools/analyze_pruned_components.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --expert-probing

    python tools/analyze_pruned_components.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --layer-ablation

    python tools/analyze_pruned_components.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --expert-fingerprint

    # Compare pruned vs full checkpoint
    python tools/analyze_pruned_components.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --pruned-dir /root/models/gemma4-pruned-30pct/ \
        --differential

    # Limit to specific layers
    python tools/analyze_pruned_components.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --all --layers 0,5,11,17,23,29

    # Save JSON report
    python tools/analyze_pruned_components.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --all --output report.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_EXPERTS = 128
TOP_K = 8
EXPERT_PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]

# Topic categories with representative text samples for probing
TOPIC_PROMPTS = {
    "python_code": [
        "def function(): return x + y",
        "import numpy as np",
        "class MyModel(nn.Module):",
        "for i in range(10):",
        "if __name__ == '__main__':",
        "try: result = await fetch(url) except Exception as e:",
        "lambda x: x ** 2 + 1",
        "self.weight = torch.zeros(hidden_size)",
    ],
    "math_formulas": [
        "The integral of x squared dx equals x cubed over three",
        "f(x) = sin(x) + cos(x)",
        "The derivative of ln(x) is 1/x",
        "sum from n=1 to infinity of 1/n^2 = pi^2/6",
        "eigenvalue decomposition A = PDP^-1",
        "gradient descent: theta = theta - alpha * nabla J",
        "probability P(A|B) = P(B|A) * P(A) / P(B)",
        "matrix multiplication C_ij = sum_k A_ik B_kj",
    ],
    "natural_language": [
        "Hello, how are you doing today?",
        "The quick brown fox jumps over the lazy dog.",
        "I think we should consider the following options.",
        "Once upon a time, in a land far away,",
        "The weather forecast predicts rain tomorrow.",
        "She walked along the beach at sunset.",
        "Can you help me understand this concept?",
        "In conclusion, the results demonstrate that",
    ],
    "scientific_text": [
        "The mitochondria is the powerhouse of the cell.",
        "Quantum entanglement occurs when particles become correlated.",
        "The double helix structure of DNA was discovered by Watson and Crick.",
        "Photosynthesis converts carbon dioxide and water into glucose.",
        "The standard model of particle physics describes fundamental forces.",
        "CRISPR-Cas9 enables precise genome editing.",
        "Black holes form when massive stars collapse.",
        "Neurotransmitters cross the synaptic cleft to transmit signals.",
    ],
    "legal_text": [
        "Pursuant to Section 42 of the aforementioned statute,",
        "The defendant hereby pleads not guilty to all charges.",
        "This agreement shall be governed by the laws of the State of",
        "Notwithstanding any provision to the contrary herein,",
        "The plaintiff seeks compensatory and punitive damages.",
        "All parties agree to binding arbitration in the event of dispute.",
        "Whereas the first party (hereinafter referred to as 'Lessor')",
        "Subject to the terms and conditions set forth in this Agreement,",
    ],
    "creative_writing": [
        "The moonlight danced upon the still waters of the lake.",
        "He whispered secrets to the wind, hoping she would hear.",
        "Colors exploded across the canvas like a symphony of light.",
        "The ancient oak tree had witnessed centuries of change.",
        "Her laughter was like music, filling the empty rooms.",
        "In the depths of winter, a single flower bloomed.",
        "The city never sleeps, its heartbeat a constant rhythm.",
        "Stars scattered across the velvet sky like diamonds.",
    ],
    "numbers_and_data": [
        "The population increased from 7.8 billion to 8.1 billion.",
        "Revenue was $42.3 million, up 15.7% year over year.",
        "Coordinates: 37.7749 N, 122.4194 W",
        "The temperature dropped to -40 degrees Celsius.",
        "Batch size: 32, learning rate: 0.001, epochs: 100",
        "HTTP status code 404: Not Found",
        "Version 3.14.159, released 2024-03-14",
        "SHA-256: a3f2b8c9d4e5f6a7b8c9d0e1f2a3b4c5",
    ],
    "multilingual": [
        "Bonjour, comment allez-vous aujourd'hui?",
        "Guten Tag, wie geht es Ihnen?",
        "Hola, como estas?",
        "Konnichiwa, genki desu ka?",
        "Privyet, kak dela?",
        "Merhaba, nasilsiniz?",
        "Annyeonghaseyo, eotteoseyo?",
        "Ni hao, ni hao ma?",
    ],
    "medical_text": [
        "The patient presents with acute myocardial infarction.",
        "Administer 500mg amoxicillin three times daily for 7 days.",
        "MRI reveals a 2.3cm lesion in the left temporal lobe.",
        "Hemoglobin A1c levels indicate poorly controlled diabetes.",
        "Post-operative recovery following laparoscopic cholecystectomy.",
        "The ECG shows ST-segment elevation in leads II, III, and aVF.",
        "Differential diagnosis includes pneumonia and pulmonary embolism.",
        "Blood pressure 140/90 mmHg, pulse 88 bpm, SpO2 97%.",
    ],
    "formatting_and_punctuation": [
        "...",
        "---",
        "***",
        "# Title\n## Subtitle\n### Section",
        "| Column A | Column B | Column C |",
        "<html><body><div>",
        "{ 'key': 'value', 'nested': { 'a': 1 } }",
        "[1, 2, 3, 4, 5]",
    ],
}


def get_num_layers(model_path: str) -> int:
    """Detect number of layers from the safetensors index."""
    index_path = os.path.join(os.path.dirname(model_path), "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        layer_nums = set()
        for k in idx["weight_map"]:
            if "language_model.layers." in k:
                parts = k.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_nums.add(int(parts[i + 1]))
                        except ValueError:
                            pass
        return max(layer_nums) + 1 if layer_nums else 30
    return 30


def get_num_experts_in_checkpoint(sf_handle, layer: int) -> int:
    """Count how many experts exist in a given layer."""
    all_keys = sf_handle.keys()
    prefix = f"model.language_model.layers.{layer}.experts."
    expert_ids = set()
    for k in all_keys:
        if k.startswith(prefix):
            rest = k[len(prefix):]
            eid = rest.split(".")[0]
            try:
                expert_ids.add(int(eid))
            except ValueError:
                pass
    return len(expert_ids)


# ---------------------------------------------------------------------------
# Approach 1: Expert Activation Probing via Router Weights
# ---------------------------------------------------------------------------

def probe_experts_via_router(model_path: str, layers: list[int],
                             tokenizer_path: Optional[str] = None) -> dict:
    """
    For each expert, determine WHAT activates it by projecting known-topic
    embeddings through the router weights.

    Instead of running the model, we:
    1. Load the embedding table
    2. Tokenize representative prompts for each topic
    3. Get mean embeddings per topic
    4. Project through router weights to get expert activation scores
    5. Map each expert to its most-activating topics
    """
    from transformers import AutoTokenizer

    model_dir = os.path.dirname(model_path)
    tok_path = tokenizer_path or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    sf = safe_open(model_path, framework="pt", device="cpu")

    print("=" * 80)
    print("APPROACH 1: EXPERT ACTIVATION PROBING VIA ROUTER WEIGHTS")
    print("=" * 80)

    # Load embedding table
    print("\nLoading embedding table...")
    embeddings = sf.get_tensor("model.language_model.embed_tokens.weight").float()
    print(f"  Embedding shape: {embeddings.shape}")

    # Compute mean embedding per topic
    print("Computing topic embeddings...")
    topic_embeddings = {}
    for topic, prompts in TOPIC_PROMPTS.items():
        all_embeds = []
        for prompt in prompts:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            # Mean-pool token embeddings
            embeds = embeddings[token_ids].mean(dim=0)
            all_embeds.append(embeds)
        topic_embeddings[topic] = torch.stack(all_embeds).mean(dim=0)  # [hidden_size]

    topic_names = list(topic_embeddings.keys())
    topic_matrix = torch.stack([topic_embeddings[t] for t in topic_names])  # [num_topics, hidden]

    results = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        router_key = f"model.language_model.layers.{layer}.router.proj.weight"
        scale_key = f"model.language_model.layers.{layer}.router.scale"
        per_expert_scale_key = f"model.language_model.layers.{layer}.router.per_expert_scale"

        router_w = sf.get_tensor(router_key).float()        # [128, 2816]
        router_scale = sf.get_tensor(scale_key).float()      # [2816]
        per_expert_scale = sf.get_tensor(per_expert_scale_key).float()  # [128]

        # Gemma4 router: scores = softmax(input @ router_w^T * scale * per_expert_scale)
        # We approximate: topic_scores = topic_embeds * router_scale @ router_w^T * per_expert_scale
        scaled_topics = topic_matrix * router_scale.unsqueeze(0)  # [topics, hidden]
        raw_scores = scaled_topics @ router_w.t()                 # [topics, 128]
        raw_scores = raw_scores * per_expert_scale.unsqueeze(0)   # apply per-expert scale

        # For each expert, which topics activate it most?
        layer_result = {}
        for expert_idx in range(router_w.shape[0]):
            expert_scores = raw_scores[:, expert_idx]  # [num_topics]
            sorted_idx = expert_scores.argsort(descending=True)
            top_topic = topic_names[sorted_idx[0].item()]
            top_score = expert_scores[sorted_idx[0]].item()
            mean_score = expert_scores.mean().item()
            std_score = expert_scores.std().item()

            # Specialization ratio: how much more does the top topic activate vs average
            specialization = (top_score - mean_score) / (std_score + 1e-8)

            layer_result[expert_idx] = {
                "top_topics": [
                    {"topic": topic_names[i.item()], "score": expert_scores[i].item()}
                    for i in sorted_idx[:3]
                ],
                "bottom_topics": [
                    {"topic": topic_names[i.item()], "score": expert_scores[i].item()}
                    for i in sorted_idx[-2:]
                ],
                "specialization_z": round(specialization, 3),
                "mean_activation": round(mean_score, 4),
                "std_activation": round(std_score, 4),
                "per_expert_scale": round(per_expert_scale[expert_idx].item(), 6),
            }

        # Print top specialists
        specialists = sorted(
            layer_result.items(),
            key=lambda x: x[1]["specialization_z"],
            reverse=True,
        )

        print(f"  Top 10 most specialized experts:")
        for expert_idx, info in specialists[:10]:
            topics = ", ".join(t["topic"] for t in info["top_topics"][:2])
            print(
                f"    Expert {expert_idx:>3}: z={info['specialization_z']:>6.2f}  "
                f"topics=[{topics}]"
            )

        print(f"  Bottom 5 least specialized (most generic):")
        for expert_idx, info in specialists[-5:]:
            topics = ", ".join(t["topic"] for t in info["top_topics"][:2])
            print(
                f"    Expert {expert_idx:>3}: z={info['specialization_z']:>6.2f}  "
                f"topics=[{topics}]"
            )

        results[layer] = layer_result

    return results


# ---------------------------------------------------------------------------
# Approach 2: Layer Contribution Analysis via Weight Statistics
# ---------------------------------------------------------------------------

def analyze_layer_contributions(model_path: str, layers: list[int]) -> dict:
    """
    Analyze what each layer contributes by examining:
    - Layer scalar (Gemma4 uses per-layer scaling)
    - Attention weight norms (q/k/v/o projections)
    - MLP weight norms
    - Expert diversity within the layer
    - Router weight structure (eigenspectrum)

    Layers with small scalars, low-rank attention, or uniform experts are
    safer to remove.
    """
    sf = safe_open(model_path, framework="pt", device="cpu")

    print("\n" + "=" * 80)
    print("APPROACH 2: LAYER CONTRIBUTION ANALYSIS")
    print("=" * 80)

    results = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        info = {}

        # Layer scalar
        scalar_key = f"model.language_model.layers.{layer}.layer_scalar"
        try:
            layer_scalar = sf.get_tensor(scalar_key).float().item()
        except Exception:
            layer_scalar = 1.0
        info["layer_scalar"] = round(layer_scalar, 6)
        print(f"  Layer scalar: {layer_scalar:.6f}")

        # Attention projection norms
        attn_norms = {}
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            # Try packed weight first, fall back to regular weight
            for suffix in ["weight_packed", "weight"]:
                key = f"model.language_model.layers.{layer}.self_attn.{proj}.{suffix}"
                try:
                    w = sf.get_tensor(key).float()
                    attn_norms[proj] = round(w.norm().item(), 2)
                    break
                except Exception:
                    continue

        info["attention_norms"] = attn_norms
        total_attn = sum(attn_norms.values())
        print(f"  Attention norms: {attn_norms}  (total: {total_attn:.1f})")

        # MLP weight norms (dense MLP, not experts)
        mlp_norms = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for suffix in ["weight_packed", "weight"]:
                key = f"model.language_model.layers.{layer}.mlp.{proj}.{suffix}"
                try:
                    w = sf.get_tensor(key).float()
                    mlp_norms[proj] = round(w.norm().item(), 2)
                    break
                except Exception:
                    continue
        info["mlp_norms"] = mlp_norms
        if mlp_norms:
            print(f"  MLP norms: {mlp_norms}  (total: {sum(mlp_norms.values()):.1f})")

        # Router weight analysis: eigenspectrum reveals latent specialization
        router_key = f"model.language_model.layers.{layer}.router.proj.weight"
        try:
            router_w = sf.get_tensor(router_key).float()  # [128, 2816]
            # SVD of router weights
            U, S, Vh = torch.linalg.svd(router_w, full_matrices=False)
            # Effective rank (how many singular values are significant)
            S_norm = S / S.sum()
            cumsum = S_norm.cumsum(0)
            eff_rank_90 = (cumsum < 0.90).sum().item() + 1
            eff_rank_99 = (cumsum < 0.99).sum().item() + 1
            top5_energy = (S[:5] ** 2).sum().item() / (S ** 2).sum().item()

            info["router_svd"] = {
                "effective_rank_90pct": eff_rank_90,
                "effective_rank_99pct": eff_rank_99,
                "top5_energy_fraction": round(top5_energy, 4),
                "top_singular_values": [round(s, 2) for s in S[:10].tolist()],
                "condition_number": round((S[0] / S[-1]).item(), 2) if S[-1] > 0 else float("inf"),
            }
            print(
                f"  Router SVD: eff_rank_90%={eff_rank_90}, eff_rank_99%={eff_rank_99}, "
                f"top5_energy={top5_energy:.3f}"
            )

            # Per-expert scale distribution
            pes_key = f"model.language_model.layers.{layer}.router.per_expert_scale"
            pes = sf.get_tensor(pes_key).float()
            info["per_expert_scale_stats"] = {
                "mean": round(pes.mean().item(), 6),
                "std": round(pes.std().item(), 6),
                "min": round(pes.min().item(), 6),
                "max": round(pes.max().item(), 6),
                "cv": round((pes.std() / pes.mean()).item(), 4),  # coefficient of variation
            }
            print(
                f"  Per-expert scale: mean={pes.mean():.4f}, std={pes.std():.4f}, "
                f"CV={pes.std()/pes.mean():.4f}"
            )
        except Exception as e:
            print(f"  Router analysis failed: {e}")
            info["router_svd"] = None

        # Expert weight diversity: how different are experts from each other?
        expert_norms = []
        n_experts_found = 0
        for expert in range(NUM_EXPERTS):
            total = 0.0
            found = False
            for proj in EXPERT_PROJ_NAMES:
                for suffix in ["weight_packed", "weight"]:
                    key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.{suffix}"
                    try:
                        w = sf.get_tensor(key).float()
                        total += w.norm().item() ** 2
                        found = True
                        break
                    except Exception:
                        continue
            if found:
                expert_norms.append(total ** 0.5)
                n_experts_found += 1

        if expert_norms:
            norms_arr = np.array(expert_norms)
            info["expert_weight_diversity"] = {
                "num_experts": n_experts_found,
                "norm_mean": round(norms_arr.mean(), 2),
                "norm_std": round(norms_arr.std(), 2),
                "norm_cv": round(norms_arr.std() / norms_arr.mean(), 4),
                "norm_min": round(norms_arr.min(), 2),
                "norm_max": round(norms_arr.max(), 2),
                "norm_range_pct": round((norms_arr.max() - norms_arr.min()) / norms_arr.mean() * 100, 2),
            }
            print(
                f"  Expert diversity: CV={norms_arr.std()/norms_arr.mean():.4f}, "
                f"range={norms_arr.max()-norms_arr.min():.1f} "
                f"({(norms_arr.max()-norms_arr.min())/norms_arr.mean()*100:.1f}% of mean)"
            )

        # Layernorm weights — how much does this layer rescale?
        for ln_name in ["input_layernorm", "post_attention_layernorm",
                        "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
            key = f"model.language_model.layers.{layer}.{ln_name}.weight"
            try:
                ln_w = sf.get_tensor(key).float()
                info[f"{ln_name}_stats"] = {
                    "mean": round(ln_w.mean().item(), 6),
                    "std": round(ln_w.std().item(), 6),
                    "min": round(ln_w.min().item(), 6),
                    "max": round(ln_w.max().item(), 6),
                }
            except Exception:
                pass

        # Risk assessment
        risk_factors = []
        if layer_scalar < 0.1:
            risk_factors.append("very_small_scalar")
        if info.get("router_svd") and info["router_svd"]["effective_rank_90pct"] < 10:
            risk_factors.append("low_rank_router")
        if info.get("expert_weight_diversity") and info["expert_weight_diversity"]["norm_cv"] < 0.01:
            risk_factors.append("uniform_experts")

        removal_risk = "LOW" if len(risk_factors) >= 2 else "MEDIUM" if risk_factors else "HIGH"
        info["removal_risk"] = removal_risk
        info["removal_risk_factors"] = risk_factors
        print(f"  Removal risk: {removal_risk}  factors={risk_factors}")

        results[layer] = info

    return results


# ---------------------------------------------------------------------------
# Approach 3: Expert Output Fingerprinting
# ---------------------------------------------------------------------------

def fingerprint_experts(model_path: str, layers: list[int]) -> dict:
    """
    Characterize what each expert produces by analyzing its weight structure:
    - Gate/up/down projection weight statistics
    - Which output dimensions are "hot" (large magnitude)
    - Sparsity patterns in the weights
    - Cross-expert similarity (redundancy detection)
    """
    sf = safe_open(model_path, framework="pt", device="cpu")

    print("\n" + "=" * 80)
    print("APPROACH 3: EXPERT OUTPUT FINGERPRINTING")
    print("=" * 80)

    results = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        layer_result = {}

        # Collect per-expert fingerprints
        fingerprints = {}  # expert_idx -> feature vector for similarity
        expert_details = {}

        n_experts = get_num_experts_in_checkpoint(sf, layer)
        if n_experts == 0:
            print("  No experts found in this layer")
            continue

        expert_ids = []
        for expert in range(NUM_EXPERTS):
            key = f"model.language_model.layers.{layer}.experts.{expert}.gate_proj.weight_packed"
            try:
                sf.get_tensor(key)
                expert_ids.append(expert)
            except Exception:
                # Try non-packed
                key2 = f"model.language_model.layers.{layer}.experts.{expert}.gate_proj.weight"
                try:
                    sf.get_tensor(key2)
                    expert_ids.append(expert)
                except Exception:
                    pass

        print(f"  Found {len(expert_ids)} experts")

        for expert in expert_ids:
            proj_stats = {}
            concat_features = []

            for proj in EXPERT_PROJ_NAMES:
                for suffix in ["weight_packed", "weight"]:
                    key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.{suffix}"
                    try:
                        w = sf.get_tensor(key).float()
                        break
                    except Exception:
                        w = None

                if w is None:
                    continue

                flat = w.flatten()
                norm = flat.norm().item()
                mean_abs = flat.abs().mean().item()
                sparsity = (flat.abs() < 0.01 * mean_abs).float().mean().item()

                # Per-output-dim magnitude (mean across input dim)
                if w.dim() == 2:
                    dim_magnitudes = w.abs().mean(dim=1)  # [out_dim]
                else:
                    dim_magnitudes = w.reshape(-1, w.shape[-1]).abs().mean(dim=1)

                top_dims = dim_magnitudes.topk(min(10, len(dim_magnitudes)))

                proj_stats[proj] = {
                    "norm": round(norm, 4),
                    "mean_abs": round(mean_abs, 6),
                    "sparsity": round(sparsity, 4),
                    "shape": list(w.shape),
                    "top_output_dims": top_dims.indices.tolist(),
                    "top_output_magnitudes": [round(v, 4) for v in top_dims.values.tolist()],
                }

                # Feature vector for similarity: use weight scale if available
                ws_key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.weight_scale"
                try:
                    ws = sf.get_tensor(ws_key).float().flatten()
                    concat_features.append(ws)
                except Exception:
                    concat_features.append(dim_magnitudes)

            if concat_features:
                fingerprints[expert] = torch.cat(concat_features)

            expert_details[expert] = {
                "projections": proj_stats,
                "total_norm": round(
                    sum(s["norm"] ** 2 for s in proj_stats.values()) ** 0.5, 4
                ),
                "avg_sparsity": round(
                    np.mean([s["sparsity"] for s in proj_stats.values()]), 4
                ),
            }

        # Compute pairwise similarity between experts
        if len(fingerprints) > 1:
            fp_ids = sorted(fingerprints.keys())
            # Normalize fingerprints
            fp_matrix = torch.stack([fingerprints[i] for i in fp_ids])
            fp_normed = fp_matrix / (fp_matrix.norm(dim=1, keepdim=True) + 1e-8)
            sim_matrix = fp_normed @ fp_normed.t()

            # Find most redundant pairs
            pairs = []
            for i in range(len(fp_ids)):
                for j in range(i + 1, len(fp_ids)):
                    pairs.append((fp_ids[i], fp_ids[j], sim_matrix[i, j].item()))
            pairs.sort(key=lambda x: x[2], reverse=True)

            # For each expert, find its most similar neighbor
            for idx, eid in enumerate(fp_ids):
                row = sim_matrix[idx].clone()
                row[idx] = -1  # exclude self
                best_match_local = row.argmax().item()
                best_match_id = fp_ids[best_match_local]
                best_sim = row[best_match_local].item()

                expert_details[eid]["most_similar_expert"] = best_match_id
                expert_details[eid]["similarity_to_nearest"] = round(best_sim, 4)

            # Report
            print(f"  Top 10 most redundant expert pairs:")
            for e1, e2, sim in pairs[:10]:
                print(f"    Expert {e1:>3} <-> Expert {e2:>3}: similarity={sim:.4f}")

            print(f"\n  Top 10 most unique experts:")
            # Expert with lowest max-similarity to any other
            uniqueness = []
            for idx, eid in enumerate(fp_ids):
                row = sim_matrix[idx].clone()
                row[idx] = -1
                max_sim = row.max().item()
                uniqueness.append((eid, max_sim))
            uniqueness.sort(key=lambda x: x[1])
            for eid, max_sim in uniqueness[:10]:
                print(f"    Expert {eid:>3}: max_similarity={max_sim:.4f} (most unique)")

            layer_result["redundant_pairs"] = [
                {"expert_a": a, "expert_b": b, "similarity": round(s, 4)}
                for a, b, s in pairs[:20]
            ]
            layer_result["unique_experts"] = [
                {"expert": eid, "max_similarity": round(ms, 4)}
                for eid, ms in uniqueness[:20]
            ]
        else:
            layer_result["redundant_pairs"] = []
            layer_result["unique_experts"] = []

        layer_result["expert_details"] = {
            str(k): v for k, v in expert_details.items()
        }
        results[layer] = layer_result

    return results


# ---------------------------------------------------------------------------
# Approach 4: Differential Checkpoint Analysis
# ---------------------------------------------------------------------------

def differential_analysis(full_model_path: str, pruned_model_path: str,
                          layers: list[int]) -> dict:
    """
    Compare pruned vs full checkpoint:
    - Which experts were removed?
    - Summary stats of removed vs kept experts
    - Router weight changes
    - Config differences
    """
    print("\n" + "=" * 80)
    print("APPROACH 4: DIFFERENTIAL CHECKPOINT ANALYSIS")
    print("=" * 80)

    full_dir = os.path.dirname(full_model_path)
    pruned_dir = os.path.dirname(pruned_model_path)

    # Load configs
    with open(os.path.join(full_dir, "config.json")) as f:
        full_config = json.load(f)
    with open(os.path.join(pruned_dir, "config.json")) as f:
        pruned_config = json.load(f)

    full_n_experts = full_config["text_config"]["num_experts"]
    pruned_n_experts = pruned_config["text_config"]["num_experts"]

    print(f"\nFull model: {full_n_experts} experts")
    print(f"Pruned model: {pruned_n_experts} experts")
    print(f"Removed: {full_n_experts - pruned_n_experts} experts ({(full_n_experts - pruned_n_experts)/full_n_experts*100:.1f}%)")

    sf_full = safe_open(full_model_path, framework="pt", device="cpu")
    sf_pruned = safe_open(pruned_model_path, framework="pt", device="cpu")

    results = {
        "full_num_experts": full_n_experts,
        "pruned_num_experts": pruned_n_experts,
        "removed_count": full_n_experts - pruned_n_experts,
        "layers": {},
    }

    # Try to load expert importance to identify which were removed
    importance_path = os.path.join(full_dir, "expert_importance.json")
    importance = None
    if os.path.exists(importance_path):
        with open(importance_path) as f:
            importance = json.load(f)

    for layer in layers:
        print(f"\n--- Layer {layer} ---")

        # Count experts in each checkpoint
        full_experts = get_num_experts_in_checkpoint(sf_full, layer)
        pruned_experts = get_num_experts_in_checkpoint(sf_pruned, layer)
        removed = full_experts - pruned_experts

        print(f"  Full: {full_experts} experts, Pruned: {pruned_experts} experts, Removed: {removed}")

        layer_info = {
            "full_experts": full_experts,
            "pruned_experts": pruned_experts,
            "removed_count": removed,
        }

        # Identify removed experts by comparing composite scores
        if importance and "composite" in importance:
            layer_key = str(layer)
            if layer_key in importance["composite"]:
                scores = importance["composite"][layer_key]
                sorted_experts = sorted(scores.items(), key=lambda x: x[1])
                removed_experts = [int(e) for e, _ in sorted_experts[:removed]]
                kept_experts = [int(e) for e, _ in sorted_experts[removed:]]

                layer_info["removed_expert_ids"] = removed_experts
                layer_info["removed_scores"] = {
                    e: round(scores[str(e)], 4) for e in removed_experts
                }

                # Stats comparison
                removed_scores = [scores[str(e)] for e in removed_experts]
                kept_scores = [scores[str(e)] for e in kept_experts]

                if removed_scores and kept_scores:
                    layer_info["score_comparison"] = {
                        "removed_mean": round(np.mean(removed_scores), 4),
                        "removed_max": round(np.max(removed_scores), 4),
                        "kept_mean": round(np.mean(kept_scores), 4),
                        "kept_min": round(np.min(kept_scores), 4),
                        "gap": round(np.min(kept_scores) - np.max(removed_scores), 4),
                    }
                    print(
                        f"  Removed scores: mean={np.mean(removed_scores):.4f}, "
                        f"max={np.max(removed_scores):.4f}"
                    )
                    print(
                        f"  Kept scores:    mean={np.mean(kept_scores):.4f}, "
                        f"min={np.min(kept_scores):.4f}"
                    )
                    print(f"  Gap (kept_min - removed_max): {np.min(kept_scores) - np.max(removed_scores):.4f}")

                print(f"  Removed expert IDs: {removed_experts[:20]}{'...' if len(removed_experts) > 20 else ''}")

        # Router weight comparison
        full_router_key = f"model.language_model.layers.{layer}.router.proj.weight"
        pruned_router_key = f"model.language_model.layers.{layer}.router.proj.weight"
        try:
            full_router = sf_full.get_tensor(full_router_key).float()
            pruned_router = sf_pruned.get_tensor(pruned_router_key).float()

            layer_info["router_shape_full"] = list(full_router.shape)
            layer_info["router_shape_pruned"] = list(pruned_router.shape)

            # Compare router norms
            full_router_norm = full_router.norm().item()
            pruned_router_norm = pruned_router.norm().item()
            layer_info["router_norm_change"] = round(
                (pruned_router_norm - full_router_norm) / full_router_norm * 100, 2
            )
            print(
                f"  Router norm: {full_router_norm:.2f} -> {pruned_router_norm:.2f} "
                f"({layer_info['router_norm_change']:+.2f}%)"
            )
        except Exception as e:
            print(f"  Router comparison failed: {e}")

        results["layers"][layer] = layer_info

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze what pruned model components actually DO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-dir", required=True, help="Full model checkpoint directory")
    parser.add_argument("--pruned-dir", default=None, help="Pruned model directory (for differential analysis)")
    parser.add_argument("--tokenizer-dir", default=None, help="Tokenizer directory (defaults to model-dir)")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices (default: all)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")

    parser.add_argument("--all", action="store_true", help="Run all applicable approaches")
    parser.add_argument("--expert-probing", action="store_true", help="Approach 1: Expert activation probing")
    parser.add_argument("--layer-ablation", action="store_true", help="Approach 2: Layer contribution analysis")
    parser.add_argument("--expert-fingerprint", action="store_true", help="Approach 3: Expert output fingerprinting")
    parser.add_argument("--differential", action="store_true", help="Approach 4: Differential checkpoint analysis")

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        sys.exit(1)

    num_layers = get_num_layers(model_path)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(num_layers))

    print(f"Model: {args.model_dir}")
    print(f"Layers to analyze: {layers}")
    print(f"Detected {num_layers} total layers")

    report = {"model_dir": args.model_dir, "num_layers": num_layers, "analyzed_layers": layers}

    run_probing = args.expert_probing or args.all
    run_ablation = args.layer_ablation or args.all
    run_fingerprint = args.expert_fingerprint or args.all
    run_differential = args.differential or args.all

    if not any([run_probing, run_ablation, run_fingerprint, run_differential]):
        print("\nNo analysis selected. Use --all or specify individual approaches.")
        parser.print_help()
        sys.exit(1)

    t0 = time.time()

    if run_probing:
        report["expert_probing"] = probe_experts_via_router(
            model_path, layers, args.tokenizer_dir
        )

    if run_ablation:
        report["layer_contributions"] = analyze_layer_contributions(model_path, layers)

    if run_fingerprint:
        report["expert_fingerprints"] = fingerprint_experts(model_path, layers)

    if run_differential:
        if not args.pruned_dir:
            print("\n[WARN] --differential requires --pruned-dir. Skipping.")
        else:
            pruned_path = os.path.join(args.pruned_dir, "model.safetensors")
            if os.path.exists(pruned_path):
                report["differential"] = differential_analysis(
                    model_path, pruned_path, layers
                )
            else:
                print(f"\n[WARN] Pruned model not found at {pruned_path}. Skipping.")

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Analysis complete in {elapsed:.1f}s")
    print(f"{'=' * 80}")

    # Summary
    if "expert_probing" in report:
        print("\nEXPERT PROBING SUMMARY:")
        for layer in layers:
            if layer in report["expert_probing"]:
                data = report["expert_probing"][layer]
                high_spec = sum(
                    1 for e in data.values() if e["specialization_z"] > 2.0
                )
                low_spec = sum(
                    1 for e in data.values() if e["specialization_z"] < 0.5
                )
                print(
                    f"  Layer {layer:>2}: {high_spec} highly specialized, "
                    f"{low_spec} generic experts"
                )

    if "layer_contributions" in report:
        print("\nLAYER CONTRIBUTION SUMMARY:")
        for layer in layers:
            if layer in report["layer_contributions"]:
                info = report["layer_contributions"][layer]
                scalar = info.get("layer_scalar", "N/A")
                risk = info.get("removal_risk", "N/A")
                print(f"  Layer {layer:>2}: scalar={scalar}, removal_risk={risk}")

    if "expert_fingerprints" in report:
        print("\nEXPERT FINGERPRINT SUMMARY:")
        for layer in layers:
            if layer in report["expert_fingerprints"]:
                data = report["expert_fingerprints"][layer]
                n_redundant = sum(
                    1 for p in data.get("redundant_pairs", []) if p["similarity"] > 0.99
                )
                n_unique = sum(
                    1 for u in data.get("unique_experts", []) if u["max_similarity"] < 0.95
                )
                print(
                    f"  Layer {layer:>2}: {n_redundant} highly redundant pairs, "
                    f"{n_unique} unique experts"
                )

    if args.output:
        # Convert non-serializable types
        def make_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            return obj

        with open(args.output, "w") as f:
            json.dump(make_serializable(report), f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
