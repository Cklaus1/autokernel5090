"""Benchmark 93K serving: sweep concurrency with realistic prompt mix."""

import asyncio
import aiohttp
import time
import random
import json
import sys

BASE_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "/root/models/gemma-4-31B-it-AWQ-4bit"

# Realistic prompt mix: short Q&A, medium instructions, longer analysis
PROMPTS = {
    "tiny": [  # ~10-30 tokens
        "What is 2+2?",
        "Capital of France?",
        "Who painted the Mona Lisa?",
        "Define entropy in one sentence.",
        "What year did WW2 end?",
    ],
    "short": [  # ~50-100 tokens
        "Explain the difference between TCP and UDP in 2-3 sentences.",
        "Write a Python function that checks if a number is prime.",
        "What are the main differences between SQL and NoSQL databases?",
        "Summarize the theory of relativity in simple terms.",
        "List 5 best practices for writing clean code.",
    ],
    "medium": [  # ~200-500 tokens
        "Write a detailed comparison of React, Vue, and Angular for building web applications. Cover performance, learning curve, ecosystem, and when to use each.",
        "Explain how a transformer neural network works, including self-attention, positional encoding, and the encoder-decoder architecture. Use analogies where helpful.",
        "Design a REST API for a todo list application. Include endpoints, HTTP methods, request/response formats, and error handling. Show example JSON.",
        "Write a short story (3 paragraphs) about a robot discovering music for the first time.",
        "Explain the CAP theorem in distributed systems. Give real-world examples of systems that prioritize different combinations of C, A, and P.",
    ],
    "long": [  # ~500-1000 tokens
        "Write a comprehensive guide to setting up a production Kubernetes cluster. Cover networking, storage, monitoring, security, and CI/CD integration. Include specific tool recommendations.",
        "Explain the complete lifecycle of an HTTP request from typing a URL in a browser to seeing the rendered page. Cover DNS, TCP, TLS, HTTP, server processing, and browser rendering.",
        "Design a scalable real-time chat system like Slack. Cover architecture, message storage, presence detection, notifications, search, file sharing, and how to handle millions of concurrent users.",
    ],
}

# Weight distribution: most users send short/medium prompts
PROMPT_WEIGHTS = {"tiny": 30, "short": 35, "medium": 25, "long": 10}


def pick_prompt():
    """Pick a random prompt based on realistic distribution."""
    categories = []
    weights = []
    for cat, w in PROMPT_WEIGHTS.items():
        categories.append(cat)
        weights.append(w)
    cat = random.choices(categories, weights=weights, k=1)[0]
    return cat, random.choice(PROMPTS[cat])


async def send_request(session, prompt, max_tokens=256):
    """Send one chat completion request, return timing + token info."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(BASE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text[:200]}", "latency": time.perf_counter() - t0}
            data = await resp.json()
            latency = time.perf_counter() - t0
            usage = data.get("usage", {})
            output_text = data["choices"][0]["message"]["content"]
            return {
                "latency": latency,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "output_preview": output_text[:80],
            }
    except Exception as e:
        return {"error": str(e)[:200], "latency": time.perf_counter() - t0}


async def run_concurrency_level(concurrency, num_requests=None):
    """Run num_requests concurrent requests at given concurrency level."""
    if num_requests is None:
        num_requests = max(concurrency * 3, 12)  # At least 3 rounds per worker

    # Pre-pick all prompts
    tasks_info = [pick_prompt() for _ in range(num_requests)]

    semaphore = asyncio.Semaphore(concurrency)
    results = []
    categories_used = {"tiny": 0, "short": 0, "medium": 0, "long": 0}

    async def worker(cat, prompt):
        async with semaphore:
            categories_used[cat] += 1
            max_tok = {"tiny": 64, "short": 128, "medium": 256, "long": 384}[cat]
            result = await send_request(session, prompt, max_tokens=max_tok)
            result["category"] = cat
            results.append(result)

    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        t_start = time.perf_counter()
        tasks = [asyncio.create_task(worker(cat, prompt)) for cat, prompt in tasks_info]
        await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_start

    # Compute stats
    errors = [r for r in results if "error" in r]
    successes = [r for r in results if "error" not in r]

    if not successes:
        return {
            "concurrency": concurrency,
            "error": f"All {len(errors)} requests failed: {errors[0]['error'] if errors else 'unknown'}",
        }

    latencies = [r["latency"] for r in successes]
    total_completion_tokens = sum(r["completion_tokens"] for r in successes)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in successes)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "successes": len(successes),
        "errors": len(errors),
        "wall_time": wall_time,
        "throughput_tok_s": total_completion_tokens / wall_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_latency": sum(latencies) / len(latencies),
        "p50_latency": p50,
        "p95_latency": p95,
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "categories": dict(categories_used),
    }


async def main():
    concurrency_levels = [1, 4, 8, 12, 20, 30]
    print(f"{'C':>3} | {'Reqs':>4} | {'OK':>3} | {'Err':>3} | {'Wall':>6} | {'tok/s':>7} | {'P50':>6} | {'P95':>6} | {'Max':>6} | Prompt Mix")
    print("-" * 100)

    for c in concurrency_levels:
        result = await run_concurrency_level(c)
        if "error" in result:
            print(f"{c:>3} | FAILED: {result['error'][:70]}")
            continue

        cats = result["categories"]
        mix = f"T:{cats['tiny']} S:{cats['short']} M:{cats['medium']} L:{cats['long']}"
        print(
            f"{result['concurrency']:>3} | "
            f"{result['num_requests']:>4} | "
            f"{result['successes']:>3} | "
            f"{result['errors']:>3} | "
            f"{result['wall_time']:>5.1f}s | "
            f"{result['throughput_tok_s']:>6.0f} | "
            f"{result['p50_latency']:>5.2f}s | "
            f"{result['p95_latency']:>5.2f}s | "
            f"{result['max_latency']:>5.2f}s | "
            f"{mix}"
        )

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
