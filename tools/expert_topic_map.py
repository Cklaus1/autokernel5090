#!/usr/bin/env python3
"""
Expert Topic Map: Maps each expert to its likely specialization and produces
a pruning risk assessment.

Combines router weight analysis, embedding projections, and expert weight
fingerprinting to produce a human-readable report of what each expert does.

Usage:
    # Full report for all layers
    python tools/expert_topic_map.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/

    # Specific layers, save JSON
    python tools/expert_topic_map.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --layers 0,5,11,17,23,29 \
        --output expert_map.json

    # With pruning risk assessment for a specific prune percentage
    python tools/expert_topic_map.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --prune-pct 30 \
        --output expert_map.json

    # Quick mode: only router analysis (fast, no weight loading)
    python tools/expert_topic_map.py \
        --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --quick
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
# Topic definitions with token-level categories
# ---------------------------------------------------------------------------

# These are semantic categories. We create "topic probes" by tokenizing
# representative text and projecting the mean embedding through the router.
TOPIC_DEFINITIONS = {
    "code_python": {
        "description": "Python code, function definitions, imports",
        "probes": [
            "def function(): return x + y",
            "import numpy as np",
            "class MyModel(nn.Module):",
            "for i in range(10):",
            "if __name__ == '__main__':",
            "self.weight = torch.zeros(hidden_size)",
            "async def fetch_data(url: str) -> dict:",
            "with open('file.txt', 'r') as f:",
        ],
    },
    "code_other": {
        "description": "Non-Python code (JS, C++, Rust, etc.)",
        "probes": [
            "function getData() { return fetch(url); }",
            "int main(int argc, char** argv) {",
            "fn main() -> Result<(), Box<dyn Error>> {",
            "SELECT * FROM users WHERE id = ?",
            "<div class='container'>",
            "console.log(JSON.stringify(data));",
            "#include <iostream>",
            "pub struct Config { field: String }",
        ],
    },
    "math": {
        "description": "Mathematics, equations, formulas",
        "probes": [
            "The integral of x squared dx",
            "f(x) = sin(x) + cos(x)",
            "sum from n=1 to infinity",
            "eigenvalue decomposition",
            "gradient descent theta",
            "probability P(A|B) = P(B|A) * P(A) / P(B)",
            "the Fourier transform of",
            "convergence of the series",
        ],
    },
    "science": {
        "description": "Scientific and technical text",
        "probes": [
            "The mitochondria is the powerhouse of the cell",
            "quantum entanglement occurs when particles",
            "the double helix structure of DNA",
            "photosynthesis converts carbon dioxide",
            "black holes form when massive stars collapse",
            "neurotransmitters cross the synaptic cleft",
            "the standard model of particle physics",
            "CRISPR-Cas9 enables precise genome editing",
        ],
    },
    "medical": {
        "description": "Medical terminology, diagnoses, prescriptions",
        "probes": [
            "acute myocardial infarction",
            "administer 500mg amoxicillin",
            "MRI reveals a lesion in the temporal lobe",
            "hemoglobin A1c levels indicate diabetes",
            "post-operative recovery following cholecystectomy",
            "ECG shows ST-segment elevation",
            "differential diagnosis includes pneumonia",
            "blood pressure 140/90 mmHg",
        ],
    },
    "legal": {
        "description": "Legal language, contracts, statutes",
        "probes": [
            "pursuant to Section 42 of the statute",
            "the defendant hereby pleads not guilty",
            "governed by the laws of the State",
            "notwithstanding any provision to the contrary",
            "binding arbitration in the event of dispute",
            "hereinafter referred to as the Lessor",
            "subject to the terms and conditions",
            "plaintiff seeks compensatory damages",
        ],
    },
    "conversation": {
        "description": "Casual conversation, chat, dialogue",
        "probes": [
            "Hello, how are you doing today?",
            "That sounds great, let me know when",
            "I think we should go to the park",
            "What do you want for dinner tonight?",
            "Sorry, I didn't mean to upset you",
            "Can you help me with something?",
            "Thanks for letting me know!",
            "Yeah, I agree with that completely",
        ],
    },
    "creative_writing": {
        "description": "Poetry, fiction, narrative prose",
        "probes": [
            "The moonlight danced upon the still waters",
            "He whispered secrets to the wind",
            "Colors exploded across the canvas",
            "In the depths of winter, a flower bloomed",
            "Her laughter was like music",
            "The ancient oak tree had witnessed centuries",
            "Stars scattered across the velvet sky",
            "Once upon a time in a kingdom far away",
        ],
    },
    "numbers_data": {
        "description": "Numbers, statistics, data, coordinates",
        "probes": [
            "population increased from 7.8 billion to 8.1 billion",
            "revenue was $42.3 million, up 15.7%",
            "coordinates 37.7749 N, 122.4194 W",
            "temperature dropped to -40 degrees",
            "batch size 32, learning rate 0.001",
            "HTTP status code 404",
            "version 3.14.159",
            "0x4A3F 0b11010010",
        ],
    },
    "multilingual": {
        "description": "Non-English languages",
        "probes": [
            "Bonjour, comment allez-vous?",
            "Guten Tag, wie geht es Ihnen?",
            "Hola, como estas hoy?",
            "Konnichiwa, genki desu ka?",
            "Privyet, kak dela segodnya?",
            "Merhaba, nasilsiniz bugun?",
            "Nimen hao, ni hao ma?",
            "Annyeonghaseyo, eotteoseyo?",
        ],
    },
    "formatting": {
        "description": "Punctuation, markup, formatting tokens",
        "probes": [
            "...",
            "---",
            "***",
            "# Title ## Subtitle ### Section",
            "| Column A | Column B |",
            "<html><body><div>",
            "{ key: value }",
            "[1, 2, 3, 4, 5]",
        ],
    },
    "reasoning": {
        "description": "Logical reasoning, step-by-step thinking",
        "probes": [
            "First, let me think about this step by step",
            "Therefore, we can conclude that",
            "If A implies B and B implies C, then A implies C",
            "This contradicts our assumption, so",
            "By induction, assume the statement holds for n=k",
            "The necessary and sufficient condition is",
            "Consider the following counterexample:",
            "We prove this by contradiction. Suppose",
        ],
    },
}


def get_num_layers(model_dir: str) -> int:
    """Detect number of layers from safetensors index."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
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


def compute_topic_embeddings(model_path: str, tokenizer_path: str) -> dict:
    """Compute mean embedding for each topic category."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    sf = safe_open(model_path, framework="pt", device="cpu")
    embeddings = sf.get_tensor("model.language_model.embed_tokens.weight").float()

    topic_embeds = {}
    for topic_name, topic_info in TOPIC_DEFINITIONS.items():
        all_embeds = []
        for probe in topic_info["probes"]:
            ids = tokenizer.encode(probe, add_special_tokens=False)
            if ids:
                all_embeds.append(embeddings[ids].mean(dim=0))
        if all_embeds:
            topic_embeds[topic_name] = torch.stack(all_embeds).mean(dim=0)

    return topic_embeds


def compute_special_token_embeddings(model_path: str, tokenizer_path: str) -> dict:
    """Compute embeddings for special token categories (digits, punctuation, etc.)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    sf = safe_open(model_path, framework="pt", device="cpu")
    embeddings = sf.get_tensor("model.language_model.embed_tokens.weight").float()

    categories = {}

    # Digit tokens
    digit_ids = []
    for d in "0123456789":
        ids = tokenizer.encode(d, add_special_tokens=False)
        digit_ids.extend(ids)
    if digit_ids:
        categories["digits"] = embeddings[digit_ids].mean(dim=0)

    # Punctuation
    punct_ids = []
    for p in ".,;:!?()[]{}\"'-_/\\@#$%^&*+=<>~`|":
        ids = tokenizer.encode(p, add_special_tokens=False)
        punct_ids.extend(ids)
    if punct_ids:
        categories["punctuation"] = embeddings[punct_ids].mean(dim=0)

    # Common English words
    common_ids = []
    for w in ["the", "is", "and", "of", "to", "in", "a", "that", "it", "for"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        common_ids.extend(ids)
    if common_ids:
        categories["common_english"] = embeddings[common_ids].mean(dim=0)

    # Uppercase / proper nouns
    upper_ids = []
    for w in ["The", "America", "London", "Microsoft", "President", "January"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        upper_ids.extend(ids)
    if upper_ids:
        categories["proper_nouns"] = embeddings[upper_ids].mean(dim=0)

    # Newlines and whitespace
    ws_ids = []
    for w in ["\n", "\t", "  ", "\n\n"]:
        ids = tokenizer.encode(w, add_special_tokens=False)
        ws_ids.extend(ids)
    if ws_ids:
        categories["whitespace"] = embeddings[ws_ids].mean(dim=0)

    return categories


def map_experts_to_topics(
    model_path: str,
    layers: list[int],
    topic_embeds: dict,
    special_embeds: dict,
) -> dict:
    """
    Project topic embeddings through each layer's router to determine
    which experts each topic would activate.

    Returns per-layer, per-expert topic assignments.
    """
    sf = safe_open(model_path, framework="pt", device="cpu")

    # Combine all probe embeddings
    all_probes = {}
    all_probes.update(topic_embeds)
    all_probes.update({f"special_{k}": v for k, v in special_embeds.items()})

    probe_names = list(all_probes.keys())
    probe_matrix = torch.stack([all_probes[n] for n in probe_names])  # [N_probes, hidden]

    results = {}

    for layer in layers:
        router_w = sf.get_tensor(
            f"model.language_model.layers.{layer}.router.proj.weight"
        ).float()  # [num_experts, hidden]
        router_scale = sf.get_tensor(
            f"model.language_model.layers.{layer}.router.scale"
        ).float()  # [hidden]
        per_expert_scale = sf.get_tensor(
            f"model.language_model.layers.{layer}.router.per_expert_scale"
        ).float()  # [num_experts]

        num_experts = router_w.shape[0]

        # Compute routing scores: input * scale @ router^T * per_expert_scale
        scaled_probes = probe_matrix * router_scale.unsqueeze(0)
        scores = scaled_probes @ router_w.t()  # [N_probes, num_experts]
        scores = scores * per_expert_scale.unsqueeze(0)

        # Softmax to get routing probabilities
        probs = torch.softmax(scores, dim=-1)  # [N_probes, num_experts]

        # For each expert, determine its topic profile
        layer_map = {}
        for expert_idx in range(num_experts):
            expert_probs = probs[:, expert_idx]  # [N_probes]
            expert_scores = scores[:, expert_idx]

            # Which topics route most strongly to this expert?
            sorted_idx = expert_scores.argsort(descending=True)

            # Compute specialization metrics
            mean_score = expert_scores.mean().item()
            std_score = expert_scores.std().item()
            max_score = expert_scores.max().item()

            # Z-score of top topic
            top_z = (max_score - mean_score) / (std_score + 1e-8)

            # Entropy of the probability distribution over topics
            # (low entropy = specialized, high entropy = generic)
            entropy = -(expert_probs * (expert_probs + 1e-10).log()).sum().item()
            max_entropy = np.log(len(probe_names))

            # Classify the expert
            top_topic_idx = sorted_idx[0].item()
            top_topic = probe_names[top_topic_idx]
            second_topic = probe_names[sorted_idx[1].item()]

            if top_z > 3.0:
                classification = f"SPECIALIST: {top_topic}"
                confidence = "high"
            elif top_z > 2.0:
                classification = f"LEANS: {top_topic}"
                confidence = "medium"
            elif top_z > 1.5:
                classification = f"SLIGHT: {top_topic}, {second_topic}"
                confidence = "low"
            else:
                classification = "GENERALIST"
                confidence = "none"

            layer_map[expert_idx] = {
                "classification": classification,
                "confidence": confidence,
                "top_topic": top_topic,
                "top_topic_description": TOPIC_DEFINITIONS.get(
                    top_topic, {"description": top_topic}
                ).get("description", top_topic),
                "specialization_z": round(top_z, 3),
                "entropy": round(entropy, 4),
                "normalized_entropy": round(entropy / max_entropy, 4),
                "per_expert_scale": round(per_expert_scale[expert_idx].item(), 6),
                "topic_scores": {
                    probe_names[i]: round(expert_scores[i].item(), 4)
                    for i in sorted_idx[:5].tolist()
                },
            }

        results[layer] = layer_map

    return results


def compute_expert_redundancy(model_path: str, layers: list[int]) -> dict:
    """
    For each expert, find its most similar neighbors in the same layer.
    Uses weight scale tensors as compact fingerprints.
    """
    sf = safe_open(model_path, framework="pt", device="cpu")
    results = {}

    for layer in layers:
        fingerprints = {}
        for expert in range(128):
            parts = []
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.weight_scale"
                try:
                    ws = sf.get_tensor(key).float().flatten()
                    parts.append(ws)
                except Exception:
                    break
            if len(parts) == 3:
                fingerprints[expert] = torch.cat(parts)

        if len(fingerprints) < 2:
            continue

        fp_ids = sorted(fingerprints.keys())
        fp_matrix = torch.stack([fingerprints[i] for i in fp_ids])
        fp_normed = fp_matrix / (fp_matrix.norm(dim=1, keepdim=True) + 1e-8)
        sim = fp_normed @ fp_normed.t()

        # Compute adaptive threshold: top quartile of pairwise similarities
        all_max_sims = []
        for idx in range(len(fp_ids)):
            row = sim[idx].clone()
            row[idx] = -1
            all_max_sims.append(row.max().item())
        all_max_sims_arr = np.array(all_max_sims)
        adaptive_thresh = float(np.percentile(all_max_sims_arr, 75))

        redundancy = {}
        for idx, eid in enumerate(fp_ids):
            row = sim[idx].clone()
            row[idx] = -1
            best_idx = row.argmax().item()
            best_id = fp_ids[best_idx]
            best_sim = row[best_idx].item()

            # Count neighbors above adaptive threshold
            n_near_dupes = (row > adaptive_thresh).sum().item()

            redundancy[eid] = {
                "most_similar": best_id,
                "similarity": round(best_sim, 4),
                "near_duplicates": n_near_dupes,
                "similarity_threshold": round(adaptive_thresh, 4),
            }

        results[layer] = redundancy

    return results


def pruning_risk_assessment(
    expert_map: dict,
    redundancy: dict,
    importance_path: Optional[str],
    prune_pct: int,
) -> dict:
    """
    Combine topic mapping, redundancy, and importance scores to assess
    pruning risk for each expert.
    """
    importance = None
    if importance_path and os.path.exists(importance_path):
        with open(importance_path) as f:
            importance = json.load(f)

    assessment = {}

    for layer, experts in expert_map.items():
        layer_key = str(layer)
        layer_assessment = {}

        # Get importance scores
        imp_scores = {}
        if importance and "composite" in importance and layer_key in importance["composite"]:
            imp_scores = importance["composite"][layer_key]

        # Sort by importance (lowest = most likely to be pruned)
        sorted_by_imp = sorted(
            imp_scores.items(), key=lambda x: x[1]
        )
        n_to_prune = int(len(sorted_by_imp) * prune_pct / 100)
        pruned_set = set(int(e) for e, _ in sorted_by_imp[:n_to_prune])

        for expert_idx, info in experts.items():
            is_pruned = expert_idx in pruned_set
            red_info = redundancy.get(layer, {}).get(expert_idx, {})

            # Risk factors
            risk_factors = []
            risk_level = "LOW"

            if info["specialization_z"] > 3.0:
                risk_factors.append("highly_specialized")
                risk_level = "HIGH"
            elif info["specialization_z"] > 2.0:
                risk_factors.append("moderately_specialized")
                if risk_level == "LOW":
                    risk_level = "MEDIUM"

            n_dupes = red_info.get("near_duplicates", 0)
            if n_dupes == 0:
                risk_factors.append("no_near_duplicates")
                # Only escalate to HIGH if also specialized
                if info["specialization_z"] > 1.5:
                    risk_level = "HIGH"
                elif risk_level == "LOW":
                    risk_level = "MEDIUM"
            elif n_dupes >= 5:
                risk_factors.append("highly_redundant")
                if risk_level != "HIGH":
                    risk_level = "LOW"

            if info.get("per_expert_scale", 0) > 1.5:
                risk_factors.append("high_scale")
                risk_level = "HIGH" if risk_level != "HIGH" else risk_level

            if info["normalized_entropy"] > 0.95:
                risk_factors.append("very_generic")
                if risk_level == "HIGH":
                    risk_level = "MEDIUM"

            layer_assessment[expert_idx] = {
                "would_be_pruned": is_pruned,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "classification": info["classification"],
                "specialization_z": info["specialization_z"],
                "near_duplicates": red_info.get("near_duplicates", -1),
                "importance_score": round(float(imp_scores.get(str(expert_idx), 0)), 4),
            }

        assessment[layer] = layer_assessment

    return assessment


def print_expert_map(expert_map: dict, redundancy: dict, assessment: Optional[dict] = None):
    """Print a human-readable expert topic map."""
    for layer in sorted(expert_map.keys()):
        experts = expert_map[layer]
        print(f"\n{'=' * 80}")
        print(f"LAYER {layer}")
        print(f"{'=' * 80}")

        # Group by classification
        specialists = []
        leans = []
        generalists = []
        for eid, info in sorted(experts.items()):
            if info["confidence"] == "high":
                specialists.append((eid, info))
            elif info["confidence"] in ("medium", "low"):
                leans.append((eid, info))
            else:
                generalists.append((eid, info))

        if specialists:
            print(f"\n  SPECIALISTS ({len(specialists)}):")
            for eid, info in specialists:
                red = redundancy.get(layer, {}).get(eid, {})
                dupes = red.get("near_duplicates", "?")
                most_sim = red.get("most_similar", "?")
                sim_val = red.get("similarity", 0)
                print(
                    f"    Expert {eid:>3}: {info['classification']:<45} "
                    f"z={info['specialization_z']:>5.2f}  "
                    f"dupes={dupes}  closest={most_sim}({sim_val:.3f})"
                )

        if leans:
            print(f"\n  LEANING ({len(leans)}):")
            for eid, info in leans[:20]:
                print(
                    f"    Expert {eid:>3}: {info['classification']:<45} "
                    f"z={info['specialization_z']:>5.2f}"
                )
            if len(leans) > 20:
                print(f"    ... and {len(leans) - 20} more")

        print(f"\n  GENERALISTS: {len(generalists)} experts")

        # Print pruning risk if assessment available
        if assessment and layer in assessment:
            la = assessment[layer]
            high_risk_pruned = [
                (eid, a) for eid, a in la.items()
                if a["would_be_pruned"] and a["risk_level"] == "HIGH"
            ]
            if high_risk_pruned:
                print(f"\n  WARNING: {len(high_risk_pruned)} HIGH-RISK experts would be pruned:")
                for eid, a in high_risk_pruned:
                    print(
                        f"    Expert {eid:>3}: {a['classification']:<40} "
                        f"risk={a['risk_factors']}"
                    )

            low_risk_kept = [
                (eid, a) for eid, a in la.items()
                if not a["would_be_pruned"] and a["risk_level"] == "LOW"
                and a.get("near_duplicates", 0) >= 5
            ]
            if low_risk_kept:
                print(f"\n  OPPORTUNITY: {len(low_risk_kept)} LOW-RISK generic experts could be pruned instead")


def main():
    parser = argparse.ArgumentParser(
        description="Map each expert to its likely specialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", required=True, help="Model checkpoint directory")
    parser.add_argument("--tokenizer-dir", default=None, help="Tokenizer directory")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices")
    parser.add_argument("--prune-pct", type=int, default=30, help="Prune percentage for risk assessment")
    parser.add_argument("--quick", action="store_true", help="Quick mode: router analysis only")
    parser.add_argument("--output", "-o", default=None, help="Output JSON file")

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        sys.exit(1)

    tok_path = args.tokenizer_dir or args.model_dir
    num_layers = get_num_layers(args.model_dir)

    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(num_layers))

    print(f"Model: {args.model_dir}")
    print(f"Layers: {layers}")
    print(f"Prune assessment: {args.prune_pct}%")
    print()

    t0 = time.time()

    # Step 1: Compute topic embeddings
    print("Computing topic embeddings...")
    topic_embeds = compute_topic_embeddings(model_path, tok_path)
    special_embeds = compute_special_token_embeddings(model_path, tok_path)
    print(f"  {len(topic_embeds)} topic categories, {len(special_embeds)} special categories")

    # Step 2: Map experts to topics
    print("Mapping experts to topics via router projection...")
    expert_map = map_experts_to_topics(model_path, layers, topic_embeds, special_embeds)

    # Step 3: Compute redundancy (skip in quick mode)
    redundancy = {}
    if not args.quick:
        print("Computing expert redundancy via weight fingerprints...")
        redundancy = compute_expert_redundancy(model_path, layers)

    # Step 4: Pruning risk assessment
    importance_path = os.path.join(args.model_dir, "expert_importance.json")
    assessment = pruning_risk_assessment(
        expert_map, redundancy, importance_path, args.prune_pct
    )

    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed:.1f}s\n")

    # Print the map
    print_expert_map(expert_map, redundancy, assessment)

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("GLOBAL SUMMARY")
    print(f"{'=' * 80}")

    total_specialists = 0
    total_leans = 0
    total_generalists = 0
    total_high_risk_pruned = 0
    total_pruned = 0

    topic_expert_counts = defaultdict(int)

    for layer in layers:
        for eid, info in expert_map.get(layer, {}).items():
            if info["confidence"] == "high":
                total_specialists += 1
                topic_expert_counts[info["top_topic"]] += 1
            elif info["confidence"] in ("medium", "low"):
                total_leans += 1
                topic_expert_counts[info["top_topic"]] += 1
            else:
                total_generalists += 1

        if layer in assessment:
            for eid, a in assessment[layer].items():
                if a["would_be_pruned"]:
                    total_pruned += 1
                    if a["risk_level"] == "HIGH":
                        total_high_risk_pruned += 1

    print(f"\nAcross {len(layers)} layers:")
    print(f"  Specialists:  {total_specialists}")
    print(f"  Leaning:      {total_leans}")
    print(f"  Generalists:  {total_generalists}")
    print(f"\nAt {args.prune_pct}% pruning:")
    print(f"  Total pruned:     {total_pruned}")
    print(f"  HIGH risk pruned: {total_high_risk_pruned}")

    print(f"\nTopic coverage (specialists + leaning):")
    for topic, count in sorted(topic_expert_counts.items(), key=lambda x: -x[1]):
        desc = TOPIC_DEFINITIONS.get(topic, {}).get("description", topic)
        print(f"  {topic:<25} {count:>4} experts  ({desc})")

    # Save output
    if args.output:
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

        report = {
            "model_dir": args.model_dir,
            "num_layers": num_layers,
            "analyzed_layers": layers,
            "prune_pct": args.prune_pct,
            "expert_map": make_serializable(expert_map),
            "redundancy": make_serializable(redundancy),
            "assessment": make_serializable(assessment),
            "summary": {
                "total_specialists": total_specialists,
                "total_leaning": total_leans,
                "total_generalists": total_generalists,
                "total_pruned": total_pruned,
                "high_risk_pruned": total_high_risk_pruned,
                "topic_coverage": dict(topic_expert_counts),
            },
        }
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
