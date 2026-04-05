#!/usr/bin/env python3
"""Benchmark prefix caching: all requests share a system prompt."""

import asyncio
import time
import statistics
import aiohttp
import json

SYSTEM_PROMPT = """You are a helpful AI assistant specialized in technology and science.
You provide clear, accurate, and concise answers. You always cite your reasoning.
When asked about code, you provide working examples. When asked about math,
you show your work step by step. You are friendly and professional.
You never make up facts — if you don't know something, you say so.
Always respond in English. Keep responses under 100 words unless asked for more detail."""

QUESTIONS = [
    "What is quantum computing?",
    "How does machine learning work?",
    "Explain the theory of relativity.",
    "What causes earthquakes?",
    "How do vaccines work?",
    "What is blockchain?",
    "How does GPS work?",
    "What is dark matter?",
    "How do batteries work?",
    "What is AI?",
    "How does encryption work?",
    "What is the greenhouse effect?",
    "How do airplanes fly?",
    "What is DNA?",
    "How does the internet work?",
    "What are black holes?",
    "How does memory work?",
    "What is photosynthesis?",
    "How do computers work?",
    "What is evolution?",
    "How does Wi-Fi work?",
    "What are stars made of?",
    "How does radar work?",
    "What is nuclear fusion?",
    "How do magnets work?",
    "What is the Big Bang?",
    "How does 3D printing work?",
    "What are vitamins?",
    "How does sonar work?",
    "What is an algorithm?",
    "How do telescopes work?",
    "What is antimatter?",
]


async def send_request(session, url, question, max_tokens=100):
    payload = {
        "model": "/root/models/gemma-4-31B-it-AWQ-4bit",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
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
                usage = data.get("usage", {})
                return {
                    "success": True,
                    "latency": elapsed,
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                }
            return {"success": False, "latency": elapsed, "error": str(data)[:80]}
    except Exception as e:
        return {"success": False, "latency": time.monotonic() - t0, "error": str(e)[:80]}


async def run_test(url, concurrency, num_requests, max_tokens=100):
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        async def limited(q):
            async with sem:
                return await send_request(session, url, q, max_tokens)
        questions = [QUESTIONS[i % len(QUESTIONS)] for i in range(num_requests)]
        t0 = time.monotonic()
        results = await asyncio.gather(*[limited(q) for q in questions])
        elapsed = time.monotonic() - t0
    return results, elapsed


async def main():
    url = "http://localhost:8000/v1/chat/completions"

    print("="*70)
    print("Prefix Caching Benchmark (shared system prompt)")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print("="*70)

    # Warmup (primes the prefix cache)
    print("\nWarming up (priming prefix cache)...")
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        await send_request(session, url, "Hello", max_tokens=5)
        await send_request(session, url, "Test", max_tokens=5)
    print("Warmup done.\n")

    print(f"{'C':>4} {'Reqs':>6} {'Time':>8} {'Req/s':>7} {'Tok/s':>7} "
          f"{'P50':>7} {'P95':>7} {'PromptTok':>10}")
    print(f"{'─'*4} {'─'*6} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")

    for c in [1, 4, 8, 16, 32, 64]:
        n = max(32, c)
        results, elapsed = await run_test(url, c, n, max_tokens=100)
        ok = [r for r in results if r["success"]]
        if not ok:
            print(f"{c:>4} FAILED")
            continue
        lats = sorted(r["latency"] for r in ok)
        toks = sum(r["completion_tokens"] for r in ok)
        ptoks = sum(r["prompt_tokens"] for r in ok)
        p50 = lats[len(lats)//2]
        p95 = lats[int(len(lats)*0.95)]
        print(f"{c:>4} {len(ok):>6} {elapsed:>7.1f}s {len(ok)/elapsed:>6.1f} "
              f"{toks/elapsed:>6.0f} {p50:>6.2f}s {p95:>6.2f}s {ptoks:>10}")

    # Save
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
