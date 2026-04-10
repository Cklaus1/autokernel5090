#!/usr/bin/env python3
"""
Profile routing frequencies using REAL token embeddings from BF16 checkpoint.
"""
import torch
import numpy as np
from collections import Counter
import json

def main():
    from safetensors import safe_open
    from transformers import AutoTokenizer

    NVFP4_PATH = "/models/gemma-4-26B-A4B-it-NVFP4-redhat/model.safetensors"
    BF16_PATH = "/models/gemma-4-26B-A4B-it-original/model-00001-of-00002.safetensors"
    ORIG_PATH = "/models/gemma-4-26B-A4B-it-original"
    NUM_LAYERS = 30
    NUM_EXPERTS = 128
    TOP_K = 8

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ORIG_PATH, trust_remote_code=True)

    print("Loading BF16 embedding weights...")
    f_bf16 = safe_open(BF16_PATH, framework="pt", device="cuda")
    embed_weight = f_bf16.get_tensor("model.language_model.embed_tokens.weight").float()
    print(f"Embedding: {embed_weight.shape} (vocab={embed_weight.shape[0]}, dim={embed_weight.shape[1]})")

    print("Loading NVFP4 router weights...")
    f_nvfp4 = safe_open(NVFP4_PATH, framework="pt", device="cuda")

    # Diverse prompts
    PROMPTS = [
        "Explain quantum entanglement in simple terms.",
        "Write a Python function to implement quicksort.",
        "What caused the fall of the Roman Empire?",
        "Describe photosynthesis at the molecular level.",
        "Create a business plan for an AI startup.",
        "The weather is beautiful today.",
        "SELECT * FROM users WHERE age > 25 ORDER BY name;",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "In the beginning, the universe was created.",
        "The mitochondria is the powerhouse of the cell.",
        "How does CRISPR gene editing work?",
        "Explain backpropagation in neural networks.",
        "What are the principles of OOP?",
        "Describe the water cycle.",
        "What is general relativity?",
        "How do vaccines protect against diseases?",
        "Explain the difference between TCP and UDP.",
        "What is blockchain technology?",
        "Describe the lifecycle of a star.",
        "Write a haiku about artificial intelligence.",
        "1 + 1 = 2. 2 + 2 = 4. 3 + 3 = ",
        "The quick brown fox jumps over the lazy dog.",
        "import torch; model = torch.nn.Linear(10, 5)",
        "Once upon a time there lived a dragon who loved math.",
        "According to research published in Nature,",
        "The experimental results demonstrate that",
        "The patient presented with acute symptoms including",
        "The function f(x) = e^x has derivative",
        "Scientists discover new species in the deep ocean.",
        "Dear hiring manager, I am writing to express interest",
        "The GDP growth rate in Q4 2025 was approximately",
        "Mix flour, sugar, butter. Add eggs one at a time.",
    ]

    # Get real token embeddings
    all_tokens = []
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt)
        all_tokens.extend(tokens)

    token_ids = torch.tensor(all_tokens, device="cuda")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Token ID range: {token_ids.min().item()} to {token_ids.max().item()}")
    print(f"Embedding vocab size: {embed_weight.shape[0]}")

    # Safety check
    assert token_ids.max().item() < embed_weight.shape[0], "Token ID out of embedding range"

    all_embeds = embed_weight[token_ids]  # [total_tokens, 2816]
    print(f"Embeddings shape: {all_embeds.shape}")

    # Route through each layer
    layer_freqs = {}
    global_freq = Counter()
    per_layer_stats = []

    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{layer_idx}.router"
        proj_weight = f_nvfp4.get_tensor(f"{prefix}.proj.weight").float()
        scale = f_nvfp4.get_tensor(f"{prefix}.scale").float()
        per_expert_scale = f_nvfp4.get_tensor(f"{prefix}.per_expert_scale").float()

        hidden_scaled = all_embeds * scale.unsqueeze(0)
        logits = hidden_scaled @ proj_weight.T

        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_ids = torch.topk(probs, TOP_K, dim=-1)

        freq = Counter()
        expert_ids = topk_ids.cpu().numpy().flatten()
        for eid in expert_ids:
            freq[int(eid)] += 1
            global_freq[int(eid)] += 1

        layer_freqs[layer_idx] = freq

        total = sum(freq.values())
        probs_arr = np.array([freq.get(i, 0) / total for i in range(NUM_EXPERTS)])
        entropy = -np.sum(probs_arr[probs_arr > 0] * np.log2(probs_arr[probs_arr > 0]))
        top32_pct = sum(c for _, c in freq.most_common(32)) / total * 100
        top16_pct = sum(c for _, c in freq.most_common(16)) / total * 100

        sorted_probs = np.sort(probs_arr)
        n = len(sorted_probs)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_probs) / (n * np.sum(sorted_probs))) - (n + 1) / n

        per_layer_stats.append({
            "layer": layer_idx,
            "entropy": float(entropy),
            "top16_pct": float(top16_pct),
            "top32_pct": float(top32_pct),
            "gini": float(gini),
            "max_pct": float(max(probs_arr) * 100),
            "min_pct": float(min(probs_arr) * 100),
            "active": int(np.sum(probs_arr > 0)),
        })

    # Print all layer stats
    print(f"\n{'Layer':>5} {'Entropy':>8} {'Top16%':>7} {'Top32%':>7} {'Gini':>7} {'Max%':>6} {'Min%':>6} {'Active':>7}")
    for s in per_layer_stats:
        print(f"{s['layer']:5d} {s['entropy']:8.3f} {s['top16_pct']:7.1f} {s['top32_pct']:7.1f} {s['gini']:7.4f} {s['max_pct']:6.2f} {s['min_pct']:6.2f} {s['active']:7d}")

    # Global
    total_global = sum(global_freq.values())
    global_probs = np.array([global_freq.get(i, 0) / total_global for i in range(NUM_EXPERTS)])
    global_entropy = -np.sum(global_probs[global_probs > 0] * np.log2(global_probs[global_probs > 0]))

    print(f"\n{'='*80}")
    print(f"GLOBAL: entropy={global_entropy:.3f}/7.00, "
          f"top32={sum(c for _, c in global_freq.most_common(32))/total_global*100:.1f}%")

    print(f"\nTop-10 globally hottest experts:")
    for rank, (eid, count) in enumerate(global_freq.most_common(10)):
        print(f"  #{rank+1}: Expert {eid} = {count/total_global*100:.2f}%")

    print(f"\nBottom-10 globally coldest experts:")
    for rank, (eid, count) in enumerate(global_freq.most_common()[-10:]):
        print(f"  Expert {eid} = {count/total_global*100:.2f}%")

    # Cross-layer consistency
    all_top32 = [set(e for e, _ in layer_freqs[l].most_common(32)) for l in range(NUM_LAYERS)]
    adj_overlaps = [len(all_top32[i] & all_top32[i+1]) for i in range(NUM_LAYERS-1)]

    jaccards = []
    for i in range(NUM_LAYERS):
        for j in range(i+1, NUM_LAYERS):
            jaccards.append(len(all_top32[i] & all_top32[j]) / len(all_top32[i] | all_top32[j]))

    print(f"\nCross-layer top-32 Jaccard: mean={np.mean(jaccards):.3f}, std={np.std(jaccards):.3f}")
    print(f"Adjacent layer top-32 overlap: mean={np.mean(adj_overlaps):.1f}/32")

    # Key metric: If we reorder so experts 0-31 are the global top-32,
    # what fraction of each layer's routing hits those experts?
    global_top32_set = set(e for e, _ in global_freq.most_common(32))
    print(f"\nIf global top-32 were experts 0-31:")
    for l in range(NUM_LAYERS):
        freq = layer_freqs[l]
        total = sum(freq.values())
        hits = sum(freq.get(e, 0) for e in global_top32_set)
        print(f"  Layer {l}: {hits/total*100:.1f}% of routing would hit reordered 0-31")

    results = {
        "global_entropy": float(global_entropy),
        "global_ranking": [(eid, count) for eid, count in global_freq.most_common()],
        "per_layer_stats": per_layer_stats,
        "cross_layer_jaccard_mean": float(np.mean(jaccards)),
        "adj_overlap_mean": float(np.mean(adj_overlaps)),
    }
    with open("/tmp/routing_real2_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nSaved to /tmp/routing_real2_results.json")


if __name__ == "__main__":
    main()
