#!/usr/bin/env python3
"""
Expert Pruning Pipeline for Gemma4 26B MoE (NVFP4 checkpoint).

Analyzes expert importance across all 30 layers, prunes bottom-N% experts
per layer, re-indexes remaining experts contiguously, updates router weights,
and saves a new checkpoint.

Usage:
    # Analyze expert importance (no pruning)
    python tools/prune_experts.py --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ --analyze

    # Prune 30% of experts and save
    python tools/prune_experts.py --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ \
        --prune-pct 30 --output-dir /root/models/gemma4-pruned-30pct/

    # Batch: prune at 10%, 20%, 30%, 50%
    python tools/prune_experts.py --model-dir /root/models/gemma-4-26B-A4B-it-NVFP4-modelopt/ --batch
"""

import argparse
import copy
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LAYERS = 30
NUM_EXPERTS = 128
TOP_K = 8
EXPERT_PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]
EXPERT_TENSOR_SUFFIXES = ["weight", "weight_scale", "weight_scale_2", "input_scale"]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_expert_importance(model_path: str) -> dict:
    """Compute per-layer, per-expert importance metrics.

    Returns dict with keys:
        weight_l2[layer][expert]       - L2 norm of all packed weight tensors
        weight_var[layer][expert]      - variance of expert from layer mean
        router_bias[layer][expert]     - router proj weight L2 norm for that expert row
        router_scale[layer][expert]    - per_expert_scale value
        composite[layer][expert]       - weighted combination
    """
    f = safe_open(model_path, framework="pt", device="cpu")

    metrics = {
        "weight_l2": defaultdict(dict),
        "weight_var": defaultdict(dict),
        "router_bias": defaultdict(dict),
        "router_scale": defaultdict(dict),
        "composite": defaultdict(dict),
    }

    for layer in range(NUM_LAYERS):
        # --- Router analysis ---
        router_proj_key = f"model.language_model.layers.{layer}.router.proj.weight"
        router_proj = f.get_tensor(router_proj_key).float()  # [128, 2816]

        per_expert_scale_key = f"model.language_model.layers.{layer}.router.per_expert_scale"
        per_expert_scale = f.get_tensor(per_expert_scale_key).float()  # [128]

        for expert in range(NUM_EXPERTS):
            metrics["router_bias"][layer][expert] = router_proj[expert].norm().item()
            metrics["router_scale"][layer][expert] = per_expert_scale[expert].item()

        # --- Expert weight analysis ---
        # For FP4 packed weights (uint8), we compute L2 norm of the raw bytes
        # as a proxy for weight magnitude. The actual FP4 values are packed
        # 2 per byte, but the byte-level norm is monotonically related to
        # the true norm for same-shaped tensors.
        expert_norms = {}
        for expert in range(NUM_EXPERTS):
            total_norm_sq = 0.0
            for proj in EXPERT_PROJ_NAMES:
                w_key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.weight"
                w = f.get_tensor(w_key).float()
                total_norm_sq += w.norm().item() ** 2

                # Also factor in the weight scales (these are in fp8)
                ws_key = f"model.language_model.layers.{layer}.experts.{expert}.{proj}.weight_scale"
                ws = f.get_tensor(ws_key).float()
                total_norm_sq += ws.norm().item() ** 2

            expert_norms[expert] = total_norm_sq ** 0.5

        # Compute variance from mean
        norms_arr = np.array([expert_norms[e] for e in range(NUM_EXPERTS)])
        mean_norm = norms_arr.mean()

        for expert in range(NUM_EXPERTS):
            metrics["weight_l2"][layer][expert] = expert_norms[expert]
            # How different is this expert from the average? Lower = more redundant
            metrics["weight_var"][layer][expert] = abs(expert_norms[expert] - mean_norm)

        # --- Composite score ---
        # Normalize each metric to [0, 1] within the layer, then combine
        def normalize(d):
            vals = np.array([d[e] for e in range(NUM_EXPERTS)])
            mn, mx = vals.min(), vals.max()
            if mx - mn < 1e-12:
                return {e: 0.5 for e in range(NUM_EXPERTS)}
            return {e: (d[e] - mn) / (mx - mn) for e in range(NUM_EXPERTS)}

        n_l2 = normalize(metrics["weight_l2"][layer])
        n_var = normalize(metrics["weight_var"][layer])
        n_rb = normalize(metrics["router_bias"][layer])
        n_rs = normalize(metrics["router_scale"][layer])

        for expert in range(NUM_EXPERTS):
            # Composite: weight magnitude (40%) + router preference (30%) +
            #            weight uniqueness (20%) + expert scale (10%)
            metrics["composite"][layer][expert] = (
                0.40 * n_l2[expert]
                + 0.30 * n_rb[expert]
                + 0.20 * n_var[expert]
                + 0.10 * n_rs[expert]
            )

    return metrics


def print_analysis(metrics: dict):
    """Print summary of expert importance analysis."""
    print("=" * 80)
    print("EXPERT IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Global ranking: average composite score across layers per expert
    global_scores = defaultdict(list)
    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            global_scores[expert].append(metrics["composite"][layer][expert])

    avg_scores = {e: np.mean(s) for e, s in global_scores.items()}
    sorted_experts = sorted(avg_scores.items(), key=lambda x: x[1])

    print("\nGlobal Expert Ranking (average composite score across all layers):")
    print(f"{'Rank':>4} {'Expert':>7} {'Score':>8}  {'Assessment'}")
    print("-" * 50)
    for rank, (expert, score) in enumerate(sorted_experts[:20]):
        print(f"{rank+1:>4} {expert:>7} {score:>8.4f}  LEAST important")
    print("  ...")
    for rank, (expert, score) in enumerate(sorted_experts[-10:], len(sorted_experts) - 10):
        print(f"{rank+1:>4} {expert:>7} {score:>8.4f}  MOST important")

    # Per-layer bottom-5
    print("\n\nPer-Layer Bottom 5 Experts (candidates for pruning):")
    print("-" * 60)
    for layer in range(NUM_LAYERS):
        scores = [(e, metrics["composite"][layer][e]) for e in range(NUM_EXPERTS)]
        scores.sort(key=lambda x: x[1])
        bottom5 = scores[:5]
        experts_str = ", ".join(f"E{e}({s:.3f})" for e, s in bottom5)
        print(f"  Layer {layer:>2}: {experts_str}")

    # Correlation between metrics
    print("\n\nMetric Correlations (Pearson r across all layer-expert pairs):")
    all_l2 = []
    all_rb = []
    all_var = []
    all_rs = []
    for layer in range(NUM_LAYERS):
        for expert in range(NUM_EXPERTS):
            all_l2.append(metrics["weight_l2"][layer][expert])
            all_rb.append(metrics["router_bias"][layer][expert])
            all_var.append(metrics["weight_var"][layer][expert])
            all_rs.append(metrics["router_scale"][layer][expert])

    from numpy import corrcoef
    labels = ["weight_l2", "router_bias", "weight_var", "router_scale"]
    data = [all_l2, all_rb, all_var, all_rs]
    cc = corrcoef(data)
    for i in range(4):
        for j in range(i + 1, 4):
            print(f"  {labels[i]:>15} vs {labels[j]:<15}: r={cc[i][j]:.4f}")

    return avg_scores


def get_experts_to_prune(metrics: dict, prune_pct: float) -> dict[int, list[int]]:
    """For each layer, return list of expert indices to remove.

    Uses per-layer ranking so we remove the least important experts
    in each individual layer (different experts may be pruned in different layers).
    """
    num_to_prune = int(NUM_EXPERTS * prune_pct / 100)
    prune_map = {}

    for layer in range(NUM_LAYERS):
        scores = [(e, metrics["composite"][layer][e]) for e in range(NUM_EXPERTS)]
        scores.sort(key=lambda x: x[1])
        prune_map[layer] = [e for e, _ in scores[:num_to_prune]]

    return prune_map


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_checkpoint(
    model_dir: str,
    output_dir: str,
    prune_pct: float,
    metrics: dict,
):
    """Create a pruned checkpoint.

    1. Determines which experts to remove per layer
    2. Copies non-expert tensors as-is
    3. Re-indexes surviving experts 0..N-1
    4. Updates router proj.weight (remove rows), per_expert_scale (remove entries)
    5. Updates config.json
    """
    model_path = os.path.join(model_dir, "model.safetensors")
    prune_map = get_experts_to_prune(metrics, prune_pct)

    num_to_prune = int(NUM_EXPERTS * prune_pct / 100)
    num_remaining = NUM_EXPERTS - num_to_prune
    print(f"\nPruning {prune_pct}%: removing {num_to_prune} experts per layer, keeping {num_remaining}")

    os.makedirs(output_dir, exist_ok=True)

    f = safe_open(model_path, framework="pt", device="cpu")
    all_keys = list(f.keys())

    new_tensors = {}
    t0 = time.time()

    for key in all_keys:
        # --- Expert tensors: re-index surviving experts ---
        if ".experts." in key:
            # Parse layer and expert index
            # key format: model.language_model.layers.{L}.experts.{E}.{proj}.{suffix}
            parts = key.split(".")
            layer_idx = int(parts[3])
            expert_idx = int(parts[5])
            rest = ".".join(parts[6:])  # e.g. "gate_proj.weight"

            if expert_idx in prune_map[layer_idx]:
                continue  # Skip pruned expert

            # Compute new expert index
            pruned_set = set(prune_map[layer_idx])
            new_idx = expert_idx - sum(1 for p in pruned_set if p < expert_idx)

            new_key = f"model.language_model.layers.{layer_idx}.experts.{new_idx}.{rest}"
            new_tensors[new_key] = f.get_tensor(key)

        # --- Router tensors: slice out pruned expert rows/entries ---
        elif ".router.proj.weight" in key:
            layer_idx = int(key.split(".")[3])
            router_w = f.get_tensor(key)  # [128, 2816]
            keep_mask = torch.ones(NUM_EXPERTS, dtype=torch.bool)
            for e in prune_map[layer_idx]:
                keep_mask[e] = False
            new_tensors[key] = router_w[keep_mask]  # [num_remaining, 2816]

        elif ".router.per_expert_scale" in key:
            layer_idx = int(key.split(".")[3])
            scale = f.get_tensor(key)  # [128]
            keep_mask = torch.ones(NUM_EXPERTS, dtype=torch.bool)
            for e in prune_map[layer_idx]:
                keep_mask[e] = False
            new_tensors[key] = scale[keep_mask]  # [num_remaining]

        elif ".router.scale" in key:
            # This is input normalization scale, not per-expert. Keep as-is.
            new_tensors[key] = f.get_tensor(key)

        else:
            # Non-expert, non-router tensor: copy as-is
            new_tensors[key] = f.get_tensor(key)

    elapsed = time.time() - t0
    print(f"  Tensor filtering took {elapsed:.1f}s")
    print(f"  Original tensors: {len(all_keys)}")
    print(f"  Pruned tensors:   {len(new_tensors)}")

    # Save safetensors
    print(f"  Saving to {output_dir}/model.safetensors ...")
    t0 = time.time()
    save_file(new_tensors, os.path.join(output_dir, "model.safetensors"))
    elapsed = time.time() - t0
    size_gb = os.path.getsize(os.path.join(output_dir, "model.safetensors")) / 1e9
    print(f"  Saved {size_gb:.2f} GB in {elapsed:.1f}s")

    # Update config.json
    with open(os.path.join(model_dir, "config.json")) as cf:
        config = json.load(cf)

    config["text_config"]["num_experts"] = num_remaining
    # Adjust top_k if it exceeds remaining experts
    if config["text_config"]["top_k_experts"] > num_remaining:
        config["text_config"]["top_k_experts"] = num_remaining

    with open(os.path.join(output_dir, "config.json"), "w") as cf:
        json.dump(config, cf, indent=2)

    # Copy other necessary files
    for fname in [
        "tokenizer.json", "tokenizer_config.json", "generation_config.json",
        "processor_config.json", "chat_template.jinja", "hf_quant_config.json",
        "recipe.yaml",
    ]:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, output_dir)

    # Create safetensors index (single file, so minimal index)
    index = {
        "metadata": {"total_size": os.path.getsize(os.path.join(output_dir, "model.safetensors"))},
        "weight_map": {k: "model.safetensors" for k in new_tensors.keys()},
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f_idx:
        json.dump(index, f_idx)

    return num_remaining, size_gb


# ---------------------------------------------------------------------------
# Quality testing
# ---------------------------------------------------------------------------

TEST_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "What is the capital of France?",
    "Translate 'hello world' into Japanese.",
    "What are the main causes of climate change?",
    "Write Python code to reverse a linked list.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is 15 * 23?",
    "Explain the difference between TCP and UDP.",
    "Write a limerick about a programmer.",
    "What are three benefits of regular exercise?",
    "Describe the water cycle in one paragraph.",
    "What is machine learning?",
    "Name five planets in our solar system.",
    "Write a short story opening about a detective.",
    "Explain photosynthesis to a 10-year-old.",
    "What happened in 1969 related to space?",
    "Give me a recipe for scrambled eggs.",
    "What is the Pythagorean theorem?",
    "Compare democracy and monarchy in three sentences.",
]


def test_quality_vllm(model_dir: str, label: str) -> dict:
    """Test model quality using vLLM offline inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("  vLLM not available, skipping quality test")
        return {"error": "vllm not installed"}

    print(f"\n  Loading {label} with vLLM ...")
    t0 = time.time()
    try:
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            max_model_len=1024,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            trust_remote_code=True,
            quantization="modelopt",
        )
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return {"error": str(e)}

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for reproducibility
        max_tokens=128,
    )

    # Quality test
    print(f"  Running {len(TEST_PROMPTS)} quality prompts ...")
    t0 = time.time()
    outputs = llm.generate(TEST_PROMPTS, sampling_params)
    qual_time = time.time() - t0

    results = []
    coherent = 0
    for prompt, output in zip(TEST_PROMPTS, outputs):
        text = output.outputs[0].text.strip()
        # Basic coherence: non-empty, no excessive repetition, has real words
        is_coherent = (
            len(text) > 10
            and len(set(text.split())) > 3  # at least 4 unique words
            and text.count(text[:20]) < 5  # not just repeating
        )
        if is_coherent:
            coherent += 1
        results.append({
            "prompt": prompt[:60],
            "response": text[:200],
            "coherent": is_coherent,
            "tokens": len(output.outputs[0].token_ids),
        })

    total_tokens = sum(r["tokens"] for r in results)
    tok_per_sec = total_tokens / qual_time if qual_time > 0 else 0

    # Throughput benchmark: single long prompt
    print("  Running throughput benchmark ...")
    bench_prompt = "Write a detailed essay about the history of artificial intelligence, covering all major milestones from the 1950s to 2025."
    bench_params = SamplingParams(temperature=0.0, max_tokens=512)
    t0 = time.time()
    bench_out = llm.generate([bench_prompt], bench_params)
    bench_time = time.time() - t0
    bench_tokens = len(bench_out[0].outputs[0].token_ids)
    bench_tps = bench_tokens / bench_time if bench_time > 0 else 0

    # Clean up GPU memory
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "label": label,
        "load_time_s": load_time,
        "coherent": coherent,
        "total_prompts": len(TEST_PROMPTS),
        "coherence_pct": 100.0 * coherent / len(TEST_PROMPTS),
        "quality_tok_per_sec": tok_per_sec,
        "bench_tok_per_sec": bench_tps,
        "bench_tokens": bench_tokens,
        "sample_responses": results[:5],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Expert Pruning Pipeline for Gemma4 MoE")
    parser.add_argument("--model-dir", required=True, help="Path to original checkpoint")
    parser.add_argument("--analyze", action="store_true", help="Only analyze, don't prune")
    parser.add_argument("--prune-pct", type=float, help="Percentage of experts to prune (e.g. 30)")
    parser.add_argument("--output-dir", type=str, help="Where to save pruned checkpoint")
    parser.add_argument("--batch", action="store_true", help="Run 10/20/30/50% pruning batch")
    parser.add_argument("--test", action="store_true", help="Test quality with vLLM after pruning")
    parser.add_argument("--test-only", type=str, help="Only test an existing checkpoint dir")
    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "model.safetensors")

    if args.test_only:
        result = test_quality_vllm(args.test_only, os.path.basename(args.test_only))
        print(json.dumps(result, indent=2, default=str))
        return

    # Step 1: Analyze
    print("Computing expert importance metrics ...")
    t0 = time.time()
    metrics = compute_expert_importance(model_path)
    print(f"Analysis completed in {time.time()-t0:.1f}s")

    avg_scores = print_analysis(metrics)

    if args.analyze:
        # Save metrics to JSON
        out = {}
        for metric_name in metrics:
            out[metric_name] = {}
            for layer in metrics[metric_name]:
                out[metric_name][str(layer)] = metrics[metric_name][layer]
        save_path = os.path.join(args.model_dir, "expert_importance.json")
        with open(save_path, "w") as f:
            json.dump(out, f)
        print(f"\nMetrics saved to {save_path}")
        return

    # Step 2: Prune
    if args.batch:
        results_all = {}
        original_size_gb = os.path.getsize(model_path) / 1e9

        for pct in [10, 20, 30, 50]:
            out_dir = os.path.join(os.path.dirname(args.model_dir.rstrip("/")),
                                   f"gemma4-pruned-{pct}pct")
            print(f"\n{'='*60}")
            print(f"PRUNING {pct}%")
            print(f"{'='*60}")
            num_remaining, size_gb = prune_checkpoint(
                args.model_dir, out_dir, pct, metrics
            )
            results_all[pct] = {
                "num_remaining": num_remaining,
                "size_gb": size_gb,
                "size_reduction_pct": 100 * (1 - size_gb / original_size_gb),
                "output_dir": out_dir,
            }

            if args.test:
                qual = test_quality_vllm(out_dir, f"pruned-{pct}pct")
                results_all[pct]["quality"] = qual

        # Print summary
        print(f"\n{'='*80}")
        print("BATCH PRUNING SUMMARY")
        print(f"{'='*80}")
        print(f"{'Prune%':>7} {'Experts':>8} {'Size(GB)':>9} {'Reduction':>10}", end="")
        if args.test:
            print(f" {'Coherence':>10} {'tok/s':>8}", end="")
        print()
        print("-" * 80)

        print(f"{'0%':>7} {128:>8} {original_size_gb:>9.2f} {'baseline':>10}", end="")
        print()

        for pct, r in sorted(results_all.items()):
            print(f"{pct:>6}% {r['num_remaining']:>8} {r['size_gb']:>9.2f} {r['size_reduction_pct']:>9.1f}%", end="")
            if args.test and "quality" in r and "coherence_pct" in r["quality"]:
                q = r["quality"]
                print(f" {q['coherence_pct']:>9.0f}% {q['bench_tok_per_sec']:>7.1f}", end="")
            print()

        # Save results
        results_path = os.path.join(args.model_dir, "pruning_results.json")
        with open(results_path, "w") as f:
            json.dump(results_all, f, indent=2, default=str)
        print(f"\nResults saved to {results_path}")

    elif args.prune_pct:
        if not args.output_dir:
            args.output_dir = os.path.join(
                os.path.dirname(args.model_dir.rstrip("/")),
                f"gemma4-pruned-{int(args.prune_pct)}pct"
            )
        num_remaining, size_gb = prune_checkpoint(
            args.model_dir, args.output_dir, args.prune_pct, metrics
        )
        if args.test:
            test_quality_vllm(args.output_dir, f"pruned-{int(args.prune_pct)}pct")


if __name__ == "__main__":
    main()
