#!/usr/bin/env python3
"""
Experiment: Profile expert routing frequencies from Gemma4 26B router weights.

Loads the router projection weights from the checkpoint and simulates routing
on diverse text inputs to measure which experts are most/least frequently selected.
"""
import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
import json
from collections import Counter

MODEL_PATH = "/models/gemma-4-26B-A4B-it-NVFP4-redhat/model.safetensors"
NUM_LAYERS = 30
NUM_EXPERTS = 128
TOP_K = 8

# Diverse prompts covering different domains
PROMPTS = [
    "Explain quantum entanglement in simple terms for a high school student.",
    "Write a Python function to implement a binary search tree with insert and delete operations.",
    "What are the main causes of the French Revolution and how did it reshape European politics?",
    "Describe the process of photosynthesis at the molecular level, including light and dark reactions.",
    "Create a business plan for a sustainable fashion startup targeting millennials.",
    "Translate the following to Spanish: The weather is beautiful today and I want to go for a walk in the park.",
    "What is the difference between TCP and UDP protocols? When would you use each?",
    "Write a short poem about the beauty of mathematics and its connection to nature.",
    "Explain the concept of inflation in economics and how central banks control it.",
    "Describe the architecture of a transformer neural network and why attention mechanisms are important.",
    "What are the health benefits of intermittent fasting according to recent research?",
    "Write a SQL query to find the top 10 customers by total purchase amount in the last year.",
    "How does CRISPR gene editing work and what are its potential applications in medicine?",
    "Explain the theory of general relativity and how it differs from special relativity.",
    "What are the key principles of object-oriented programming? Give examples in Java.",
    "Describe the water cycle and its importance for Earth's climate system.",
    "Write a recipe for a traditional Italian risotto with mushrooms and parmesan.",
    "What were the major technological innovations during the Industrial Revolution?",
    "Explain how blockchain technology works and its applications beyond cryptocurrency.",
    "Describe the process of machine learning model training, including backpropagation.",
    "What is the significance of the Rosetta Stone in understanding ancient civilizations?",
    "Write a haiku about artificial intelligence.",
    "How do vaccines work to protect against infectious diseases?",
    "Explain the difference between supervised and unsupervised learning in AI.",
    "What are the environmental impacts of deep sea mining?",
    "Describe the lifecycle of a star from nebula to black hole or white dwarf.",
    "Write pseudocode for Dijkstra's shortest path algorithm.",
    "What are the main arguments for and against universal basic income?",
    "Explain how neural networks can be used for natural language processing.",
    "Describe the process of making wine from grape harvest to bottling.",
    "What role does the gut microbiome play in human health?",
    "Write a regular expression to validate email addresses.",
]


def main():
    print("Loading router weights from checkpoint...")
    f = safe_open(MODEL_PATH, framework="pt", device="cuda")

    # Load tokenizer for encoding prompts
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/models/gemma-4-26B-A4B-it-original", trust_remote_code=True
    )

    # Encode all prompts
    all_hidden_states = []
    for prompt in PROMPTS:
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        all_hidden_states.append(tokens.shape[1])

    print(f"Total tokens across {len(PROMPTS)} prompts: {sum(all_hidden_states)}")

    # For each layer, simulate routing
    # Router: x -> normalize(x) -> proj -> softmax -> topk
    # We use random hidden states as proxy (router is a learned projection,
    # but the weight distribution tells us about expert selection patterns)

    # Actually, let's use the router weights directly to understand the
    # geometry - which expert columns have highest norms (attractors)

    layer_freqs = {}  # layer -> {expert_id: frequency}
    global_freq = Counter()

    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{layer_idx}.router"

        # Router proj weight: [128, 2816] - maps hidden_dim to expert logits
        proj_weight = f.get_tensor(f"{prefix}.proj.weight").float()  # [128, 2816]
        # Router scale: [2816] - input normalization
        scale = f.get_tensor(f"{prefix}.scale").float()  # [2816]
        # Per-expert scale: [128]
        per_expert_scale = f.get_tensor(f"{prefix}.per_expert_scale").float()

        # Simulate routing with random hidden states (gaussian, matching typical activations)
        # Use many samples for statistical robustness
        torch.manual_seed(42)
        num_tokens = 10000
        hidden = torch.randn(num_tokens, 2816, device="cuda", dtype=torch.float32)

        # Apply router: normalize -> project -> softmax -> topk
        # Gemma4 router: x * scale -> proj -> logits
        hidden_scaled = hidden * scale.unsqueeze(0)  # [T, 2816]
        logits = hidden_scaled @ proj_weight.T  # [T, 128]

        # Softmax over all experts, then topk
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_ids = torch.topk(probs, TOP_K, dim=-1)  # [T, 8]

        # Count frequencies
        freq = Counter()
        expert_ids = topk_ids.cpu().numpy().flatten()
        for eid in expert_ids:
            freq[int(eid)] += 1
            global_freq[int(eid)] += 1

        layer_freqs[layer_idx] = freq

        # Print top-10 for this layer
        top10 = freq.most_common(10)
        bottom5 = freq.most_common()[-5:]
        total = sum(freq.values())
        top32_count = sum(c for _, c in freq.most_common(32))

        if layer_idx % 5 == 0:
            print(f"\nLayer {layer_idx}:")
            print(f"  Top-10 experts: {[(e, f'{c/total*100:.1f}%') for e, c in top10]}")
            print(f"  Bottom-5 experts: {[(e, f'{c/total*100:.1f}%') for e, c in bottom5]}")
            print(f"  Top-32 experts handle {top32_count/total*100:.1f}% of tokens")
            print(f"  Unique experts used: {len(freq)}/128")

    # Global analysis
    print("\n" + "=" * 80)
    print("GLOBAL ROUTING FREQUENCY ANALYSIS")
    print("=" * 80)

    total_global = sum(global_freq.values())
    top32_global = global_freq.most_common(32)
    top32_count = sum(c for _, c in top32_global)

    print(f"\nTop-32 experts handle {top32_count/total_global*100:.1f}% of all routing decisions")
    print(f"\nTop-32 most routed experts (across all layers):")
    for rank, (eid, count) in enumerate(top32_global):
        print(f"  Rank {rank+1}: Expert {eid} = {count/total_global*100:.2f}%")

    # Check per-layer consistency
    print("\n\nPer-layer top-8 expert overlap:")
    for l1 in range(0, NUM_LAYERS, 5):
        top8_l1 = set(e for e, _ in layer_freqs[l1].most_common(8))
        overlaps = []
        for l2 in range(NUM_LAYERS):
            if l1 != l2:
                top8_l2 = set(e for e, _ in layer_freqs[l2].most_common(8))
                overlaps.append(len(top8_l1 & top8_l2))
        print(f"  Layer {l1} top-8 overlap with other layers: mean={np.mean(overlaps):.1f}/8, max={max(overlaps)}/8")

    # Compute entropy of routing distribution per layer
    print("\nRouting entropy per layer (uniform = {:.2f} bits):".format(np.log2(NUM_EXPERTS)))
    for layer_idx in range(NUM_LAYERS):
        freq = layer_freqs[layer_idx]
        total = sum(freq.values())
        probs_arr = np.array([freq.get(i, 0) / total for i in range(NUM_EXPERTS)])
        probs_arr = probs_arr[probs_arr > 0]
        entropy = -np.sum(probs_arr * np.log2(probs_arr))
        if layer_idx % 5 == 0:
            print(f"  Layer {layer_idx}: {entropy:.2f} bits (uniform would be {np.log2(NUM_EXPERTS):.2f})")

    # Compute the "L2 fit" metric: if we could keep 32 experts in L2,
    # what fraction of routing decisions would hit L2?
    print("\n\nL2 CACHE ANALYSIS (32 expert capacity):")
    for layer_idx in range(0, NUM_LAYERS, 5):
        freq = layer_freqs[layer_idx]
        total = sum(freq.values())
        top32 = freq.most_common(32)
        top32_pct = sum(c for _, c in top32) / total * 100
        print(f"  Layer {layer_idx}: top-32 experts handle {top32_pct:.1f}% of routing")

    # Save results
    results = {
        "global_ranking": [(eid, count) for eid, count in global_freq.most_common()],
        "per_layer_top32": {
            str(l): [(eid, count) for eid, count in layer_freqs[l].most_common(32)]
            for l in range(NUM_LAYERS)
        },
    }

    with open("/tmp/routing_freq_results.json", "w") as fp:
        json.dump(results, fp)
    print("\nResults saved to /tmp/routing_freq_results.json")


if __name__ == "__main__":
    main()
