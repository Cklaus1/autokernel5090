#!/usr/bin/env python3
"""Mixture-of-Agents router for dual-model PRO 6000 serving.

Routes requests to the best model based on task complexity:
  - STRONG (Gemma4 26B, GPU 0:8000): reasoning, architecture, complex code
  - FAST (Qwen3.6 35B, GPU 1:8001): simple Q&A, formatting, test scaffolding

Usage:
    python moa_router.py "Write a merge sort with tests"
    python moa_router.py --benchmark  # throughput comparison

Requires: both servers running (./serve_dual_model.sh)
"""

import argparse
import asyncio
import re
import time
import aiohttp


STRONG = {"url": "http://localhost:8000/v1", "model": "gemma-4-26B-A4B-it-NVFP4", "label": "Gemma4-26B"}
FAST = {"url": "http://localhost:8001/v1", "model": "Qwen3.6-35B-A3B-FP8", "label": "Qwen3.6-35B"}

# Keywords that signal hard tasks → route to STRONG model
HARD_SIGNALS = [
    r"\barchitect",
    r"\bdesign\b.*\bsystem",
    r"\bsecurity\b.*\breview",
    r"\boptimiz",
    r"\bdebug\b.*\brace",
    r"\bconcurren",
    r"\bprove\b",
    r"\bformal",
    r"\btrade.?off",
    r"\bscal(e|ing|able)",
    r"\bdistribut",
    r"\brefactor.*large",
    r"\bexplain.*why",
    r"\broot cause",
]

# Keywords that signal easy tasks → route to FAST model
EASY_SIGNALS = [
    r"\bhello\b",
    r"\bhi\b",
    r"\bformat",
    r"\btranslat",
    r"\bsummariz",
    r"\blist\b",
    r"\bcount\b",
    r"\bsimple\b",
    r"\bquick\b",
    r"\bone.?liner",
    r"\bboilerplate",
    r"\btemplate",
    r"\bscaffold",
]


def classify(prompt: str) -> dict:
    """Classify prompt complexity. Returns the target backend config."""
    lower = prompt.lower()
    hard_score = sum(1 for p in HARD_SIGNALS if re.search(p, lower))
    easy_score = sum(1 for p in EASY_SIGNALS if re.search(p, lower))

    # Long prompts (>200 chars) lean toward strong model
    if len(prompt) > 200:
        hard_score += 1

    if hard_score > easy_score:
        return STRONG
    return FAST


async def generate(session, backend, prompt, max_tokens=256, temperature=0.7):
    """Send a request to a backend and return (response, latency, tokens)."""
    t0 = time.time()
    async with session.post(
        f"{backend['url']}/chat/completions",
        json={
            "model": backend["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    ) as resp:
        data = await resp.json()
    elapsed = time.time() - t0

    if "error" in data:
        return None, elapsed, 0

    content = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["completion_tokens"]
    return content, elapsed, tokens


async def route_and_generate(prompt, max_tokens=256):
    """Classify, route, and generate."""
    backend = classify(prompt)
    async with aiohttp.ClientSession() as session:
        content, elapsed, tokens = await generate(session, backend, prompt, max_tokens)
    return backend, content, elapsed, tokens


async def benchmark():
    """Compare routing vs always-strong vs always-fast."""
    prompts = [
        ("Hello, what's your name?", "easy"),
        ("Write a Python hello world", "easy"),
        ("List the first 10 prime numbers", "easy"),
        ("Explain the tradeoffs between B-trees and LSM trees for database indexing", "hard"),
        ("Design a distributed rate limiter that handles 1M req/s with consistency", "hard"),
        ("Debug this race condition in async Python code that causes data loss", "hard"),
        ("Format this JSON: {a:1,b:2}", "easy"),
        ("Architect a real-time fraud detection system with sub-100ms latency", "hard"),
    ]

    print(f"{'Prompt':<70} {'Route':<12} {'Tokens':>6} {'Latency':>8}")
    print("-" * 100)

    async with aiohttp.ClientSession() as session:
        for prompt, expected in prompts:
            backend = classify(prompt)
            _, elapsed, tokens = await generate(session, backend, prompt, max_tokens=64)
            route_label = backend["label"]
            match = "✓" if (expected == "hard" and backend is STRONG) or (expected == "easy" and backend is FAST) else "✗"
            print(f"{prompt[:68]:<70} {route_label:<12} {tokens:>6} {elapsed:>7.2f}s {match}")

    print()
    print("✓ = routed correctly, ✗ = unexpected route")


async def main():
    parser = argparse.ArgumentParser(description="MoA Router")
    parser.add_argument("prompt", nargs="?", help="Prompt to route and generate")
    parser.add_argument("--benchmark", action="store_true", help="Run routing benchmark")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.benchmark:
        await benchmark()
    elif args.prompt:
        backend, content, elapsed, tokens = await route_and_generate(args.prompt, args.max_tokens)
        print(f"[Routed to {backend['label']}] ({tokens} tokens, {elapsed:.2f}s)")
        print()
        print(content)
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
