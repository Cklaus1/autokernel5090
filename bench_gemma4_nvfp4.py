#!/usr/bin/env python3
"""Batch throughput sweep for Gemma4 26B NVFP4 on vLLM.

Measures generation throughput at various batch sizes and concurrency levels.
Hits a running vLLM server via OpenAI-compatible API.

Usage:
    python3 bench_gemma4_nvfp4.py [--api-base http://localhost:8000/v1]
"""

import argparse
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_model_name(api_base):
    resp = requests.get(f"{api_base}/models", timeout=10)
    return resp.json()["data"][0]["id"]


def bench_single(api_base, model, prompt, max_tokens, temperature=0.7):
    """Single request, return (prompt_tokens, completion_tokens, elapsed_s)."""
    t0 = time.time()
    resp = requests.post(f"{api_base}/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, timeout=300)
    elapsed = time.time() - t0
    r = resp.json()
    if "error" in r:
        return 0, 0, elapsed, r["error"]
    usage = r["usage"]
    return usage["prompt_tokens"], usage["completion_tokens"], elapsed, None


def bench_concurrent(api_base, model, prompts, max_tokens, concurrency):
    """Run prompts concurrently, return aggregate stats."""
    total_prompt_tok = 0
    total_gen_tok = 0
    errors = 0
    latencies = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for p in prompts:
            futures.append(pool.submit(bench_single, api_base, model, p, max_tokens))
        for f in as_completed(futures):
            pt, ct, elapsed, err = f.result()
            if err:
                errors += 1
            else:
                total_prompt_tok += pt
                total_gen_tok += ct
                latencies.append(elapsed)
    wall = time.time() - t0

    return {
        "concurrency": concurrency,
        "num_requests": len(prompts),
        "total_prompt_tokens": total_prompt_tok,
        "total_gen_tokens": total_gen_tok,
        "wall_time_s": round(wall, 2),
        "gen_tok_per_s": round(total_gen_tok / wall, 1) if wall > 0 else 0,
        "total_tok_per_s": round((total_prompt_tok + total_gen_tok) / wall, 1) if wall > 0 else 0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p50_latency_s": round(sorted(latencies)[len(latencies)//2], 2) if latencies else 0,
        "errors": errors,
    }


# Diverse prompts to avoid prefix cache hits skewing results
PROMPTS = [
    "Explain quantum entanglement to a 10 year old.",
    "Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.",
    "What are the main differences between TCP and UDP? Give examples of when to use each.",
    "Describe the process of photosynthesis in detail, including the light and dark reactions.",
    "Write a short story about a robot discovering emotions for the first time.",
    "Explain how a transformer neural network works, focusing on self-attention.",
    "What are the key principles of clean code architecture? Give concrete examples.",
    "Describe the history of the Internet from ARPANET to modern day.",
    "Write a SQL query to find the top 10 customers by revenue in the last 90 days, with their order counts.",
    "Explain the differences between REST, GraphQL, and gRPC. When would you choose each?",
    "What causes inflation and how do central banks try to control it?",
    "Write a Rust function that implements a thread-safe LRU cache.",
    "Explain the CAP theorem and its implications for distributed systems.",
    "Describe how mRNA vaccines work, from design to immune response.",
    "Write a bash script that monitors disk usage and sends an alert when it exceeds 90%.",
    "What are the main challenges in training large language models? How are they addressed?",
    "Explain the difference between symmetric and asymmetric encryption with examples.",
    "Write a React component that implements infinite scrolling with virtualization.",
    "Describe the water cycle and how climate change affects it.",
    "What are design patterns? Explain the Observer, Strategy, and Factory patterns with code examples.",
    "Explain how GPS works, from satellite signals to position calculation.",
    "Write a Python class that implements a binary search tree with insert, delete, and search.",
    "What is the significance of the Turing test? Has any AI truly passed it?",
    "Describe the process of fermentation in beer brewing.",
    "Explain containerization and how Docker works under the hood.",
    "What are the pros and cons of microservices vs monolithic architecture?",
    "Write a mathematical proof that the square root of 2 is irrational.",
    "Explain how CRISPR gene editing works and its potential applications.",
    "Describe the key battles of World War II in the Pacific theater.",
    "Write a comprehensive comparison of Python, Go, and Rust for backend services.",
    "Explain how blockchain consensus mechanisms work: PoW vs PoS vs BFT.",
    "What are the main types of machine learning? Give real-world examples of each.",
]


def main():
    parser = argparse.ArgumentParser(description="Batch throughput sweep")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-json", default="bench_nvfp4_results.json")
    args = parser.parse_args()

    model = get_model_name(args.api_base)
    print(f"Model: {model}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Warmup
    print("Warmup...")
    bench_single(args.api_base, model, "Hello", 10)

    results = []

    # Test matrix: (num_requests, concurrency)
    test_configs = [
        (1, 1),      # Single request baseline
        (4, 4),      # Small batch
        (8, 8),      # Medium batch
        (16, 16),    # Larger batch
        (32, 32),    # Full batch
        (32, 8),     # Queued: 32 requests at concurrency 8
        (64, 16),    # Heavy load
        (128, 32),   # Stress test
    ]

    print(f"{'Config':<20} {'GenTok/s':>10} {'TotalTok/s':>12} {'AvgLat':>8} {'P50Lat':>8} {'Errors':>7}")
    print("-" * 75)

    for num_req, conc in test_configs:
        # Cycle through prompts
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_req)]

        r = bench_concurrent(args.api_base, model, prompts, args.max_tokens, conc)
        results.append(r)

        label = f"B={num_req},C={conc}"
        print(f"{label:<20} {r['gen_tok_per_s']:>10.1f} {r['total_tok_per_s']:>12.1f} "
              f"{r['avg_latency_s']:>7.2f}s {r['p50_latency_s']:>7.2f}s {r['errors']:>7}")

    # Save results
    with open(args.output_json, "w") as f:
        json.dump({"model": model, "max_tokens": args.max_tokens, "results": results}, f, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
