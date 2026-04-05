#!/usr/bin/env python3
"""FusenCache serving throughput benchmark.

Measures aggregate tok/s, latency percentiles, max concurrency
against a running vLLM server.

Usage:
    python bench_serving.py [--concurrency 1,4,8,16,32] [--port 8000]
"""

import argparse
import asyncio
import json
import time
import statistics
from typing import List

import aiohttp


PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main causes of climate change?",
    "Describe how a neural network learns.",
    "What is the history of the internet?",
    "How does photosynthesis work?",
    "Explain the theory of relativity simply.",
    "What are the benefits of exercise?",
    "How do computers store data?",
    "What is the water cycle?",
    "Describe the solar system.",
    "How does encryption work?",
    "What causes earthquakes?",
    "Explain how vaccines work.",
    "What is machine learning?",
    "How does GPS work?",
    "What is blockchain technology?",
    "How do airplanes fly?",
    "What is the greenhouse effect?",
    "How does the stock market work?",
    "Explain DNA replication.",
    "What is dark matter?",
    "How do batteries work?",
    "What causes thunderstorms?",
    "How does Wi-Fi work?",
    "What is the Big Bang theory?",
    "How do telescopes work?",
    "What is artificial intelligence?",
    "How do solar panels work?",
    "What causes tides?",
    "How does memory work in the brain?",
    "What is quantum entanglement?",
]


async def send_request(session, url, prompt, max_tokens=100):
    """Send one completion request and measure latency."""
    payload = {
        "model": "/root/models/gemma-4-31B-it-AWQ-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }

    t0 = time.monotonic()
    try:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            elapsed = time.monotonic() - t0

            if "choices" in data:
                output = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                completion_tokens = usage.get("completion_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                return {
                    "success": True,
                    "latency": elapsed,
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "ttft": elapsed,  # approximate (non-streaming)
                    "output_preview": output[:60],
                }
            else:
                return {"success": False, "latency": elapsed,
                        "error": str(data)[:100]}
    except Exception as e:
        return {"success": False, "latency": time.monotonic() - t0,
                "error": str(e)[:100]}


async def run_concurrent(url, concurrency, num_requests, max_tokens=100):
    """Run num_requests with given concurrency level."""
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request(prompt):
            async with semaphore:
                return await send_request(session, url, prompt, max_tokens)

        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

        t0 = time.monotonic()
        results = await asyncio.gather(*[limited_request(p) for p in prompts])
        total_elapsed = time.monotonic() - t0

    return results, total_elapsed


def analyze_results(results, total_elapsed, concurrency):
    """Compute throughput and latency stats."""
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    if not successes:
        return {"error": "All requests failed",
                "failures": [r["error"] for r in failures[:3]]}

    latencies = [r["latency"] for r in successes]
    total_completion_tokens = sum(r["completion_tokens"] for r in successes)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successes)

    return {
        "concurrency": concurrency,
        "num_requests": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "total_elapsed_s": round(total_elapsed, 2),
        "requests_per_sec": round(len(successes) / total_elapsed, 2),
        "total_completion_tokens": total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "aggregate_tok_s": round(total_completion_tokens / total_elapsed, 1),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "latency_p99_s": round(sorted(latencies)[int(len(latencies) * 0.99)], 3),
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_min_s": round(min(latencies), 3),
        "latency_max_s": round(max(latencies), 3),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--concurrency", type=str, default="1,4,8,16,32")
    parser.add_argument("--requests-per-level", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    concurrency_levels = [int(c) for c in args.concurrency.split(",")]

    print(f"{'='*80}")
    print(f"FusenCache Serving Throughput Benchmark")
    print(f"Server: http://localhost:{args.port}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Requests per level: {args.requests_per_level}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'='*80}")

    # Warmup
    print("\nWarming up...")
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        await send_request(session, url, "Hi", max_tokens=5)
    print("Warmup done.\n")

    all_results = []

    print(f"{'Concurrency':>12} {'Requests':>10} {'Elapsed':>10} "
          f"{'Req/s':>8} {'Tok/s':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"{'─'*12} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for c in concurrency_levels:
        n = max(args.requests_per_level, c)  # at least concurrency requests
        results, elapsed = await run_concurrent(url, c, n, args.max_tokens)
        stats = analyze_results(results, elapsed, c)

        if "error" not in stats:
            print(f"{c:>12} {stats['successes']:>10} "
                  f"{stats['total_elapsed_s']:>9.1f}s "
                  f"{stats['requests_per_sec']:>7.1f} "
                  f"{stats['aggregate_tok_s']:>7.0f} "
                  f"{stats['latency_p50_s']:>7.2f}s "
                  f"{stats['latency_p95_s']:>7.2f}s "
                  f"{stats['latency_p99_s']:>7.2f}s")
        else:
            print(f"{c:>12} FAILED: {stats.get('error', 'unknown')}")
            if "failures" in stats:
                for f in stats["failures"]:
                    print(f"  {f}")

        all_results.append(stats)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    valid = [r for r in all_results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["aggregate_tok_s"])
        print(f"Peak throughput: {best['aggregate_tok_s']} tok/s "
              f"at concurrency={best['concurrency']}")
        print(f"Best latency: {min(r['latency_p50_s'] for r in valid):.3f}s P50 "
              f"at concurrency={min(valid, key=lambda r: r['latency_p50_s'])['concurrency']}")

    # Save
    with open("fusencache_serving_bench.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to fusencache_serving_bench.json")


if __name__ == "__main__":
    asyncio.run(main())
