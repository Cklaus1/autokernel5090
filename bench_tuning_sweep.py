#!/usr/bin/env python3
"""Sweep vLLM serving parameters to find optimal config.

Tests different server configurations by restarting the server
with each config and running the benchmark.

Usage: python bench_tuning_sweep.py
"""

import asyncio
import json
import os
import signal
import subprocess
import time

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
]

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."


async def run_benchmark(url, concurrency=32, num_requests=32, max_tokens=100):
    """Run benchmark at given concurrency."""
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    sem = asyncio.Semaphore(concurrency)

    async def send(session, prompt):
        async with sem:
            payload = {
                "model": "/root/models/gemma-4-31B-it-AWQ-4bit",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
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
                        return {
                            "ok": True,
                            "latency": elapsed,
                            "toks": data.get("usage", {}).get("completion_tokens", 0),
                        }
                    return {"ok": False, "latency": elapsed}
            except Exception:
                return {"ok": False, "latency": time.monotonic() - t0}

    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        await send(session, "Hello")

        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]
        t0 = time.monotonic()
        results = await asyncio.gather(*[send(session, p) for p in prompts])
        elapsed = time.monotonic() - t0

    ok = [r for r in results if r["ok"]]
    if not ok:
        return None

    total_toks = sum(r["toks"] for r in ok)
    lats = sorted(r["latency"] for r in ok)
    return {
        "tok_s": round(total_toks / elapsed, 1),
        "req_s": round(len(ok) / elapsed, 2),
        "p50": round(lats[len(lats)//2], 3),
        "p95": round(lats[int(len(lats)*0.95)], 3),
        "ok": len(ok),
        "elapsed": round(elapsed, 1),
    }


def wait_for_server(port=8000, timeout=600):
    """Wait for server to be ready."""
    import urllib.request
    for _ in range(timeout // 5):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(5)
    return False


def run_config(name, extra_args, concurrency=32):
    """Start server with config, benchmark, kill."""
    print(f"\n{'─'*60}")
    print(f"  Config: {name}")
    print(f"  Args: {extra_args}")
    print(f"{'─'*60}")

    # Kill any existing server
    os.system("pkill -f 'vllm serve' 2>/dev/null")
    time.sleep(3)

    # Start server
    cmd = (
        f"vllm serve /root/models/gemma-4-31B-it-AWQ-4bit "
        f"--kv-cache-dtype fp8_e4m3 "
        f"--gpu-memory-utilization 0.92 "
        f"--max-model-len 8192 "
        f"--trust-remote-code "
        f"--enable-prefix-caching "
        f"--port 8000 "
        f"{extra_args} "
        f"> /tmp/vllm_tuning.log 2>&1"
    )
    proc = subprocess.Popen(cmd, shell=True)

    if not wait_for_server():
        print(f"  FAILED: Server didn't start")
        proc.terminate()
        return None

    # Get KV cache info
    with open("/tmp/vllm_tuning.log") as f:
        log = f.read()
    kv_line = [l for l in log.split("\n") if "KV cache size" in l]
    kv_info = kv_line[-1].split("KV cache size: ")[-1] if kv_line else "unknown"

    # Run benchmark
    url = "http://localhost:8000/v1/chat/completions"
    result = asyncio.run(run_benchmark(url, concurrency=concurrency))

    if result:
        print(f"  KV cache: {kv_info}")
        print(f"  tok/s: {result['tok_s']}")
        print(f"  P50: {result['p50']}s  P95: {result['p95']}s")
        result["name"] = name
        result["kv_cache"] = kv_info
    else:
        print(f"  FAILED: No successful requests")

    # Kill server
    proc.terminate()
    time.sleep(3)
    os.system("pkill -f 'vllm serve' 2>/dev/null")
    time.sleep(2)

    return result


def main():
    print("="*60)
    print("vLLM Serving Parameter Sweep")
    print("="*60)

    configs = [
        ("baseline (defaults)", ""),
        ("max-num-seqs=64", "--max-num-seqs 64"),
        ("max-num-seqs=128", "--max-num-seqs 128"),
        ("max-num-seqs=256", "--max-num-seqs 256"),
        ("block-size=32", "--block-size 32"),
        ("block-size=64", "--block-size 64"),
    ]

    results = []
    for name, args in configs:
        r = run_config(name, args, concurrency=32)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'tok/s':>8} {'P50':>8} {'KV Cache':>15}")
    print(f"{'─'*30} {'─'*8} {'─'*8} {'─'*15}")
    for r in results:
        print(f"{r['name']:<30} {r['tok_s']:>7.0f} {r['p50']:>7.2f}s {r['kv_cache']:>15}")

    if results:
        best = max(results, key=lambda r: r["tok_s"])
        print(f"\nBest: {best['name']} at {best['tok_s']} tok/s")

    with open("fusencache_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
