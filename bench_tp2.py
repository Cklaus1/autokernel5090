#!/usr/bin/env python3
"""TP=2 benchmark for RTX PRO 6000 workstation.

Measures throughput and latency across:
  - Concurrency: 1 to 1024 (192GB allows ~120x more KV than 5090)
  - Context lengths: 4K, 8K, 16K, 32K
  - Comparison: TP=1 (port 8000) vs TP=2 (port 8001)

Usage:
    # Single server (TP=2 on port 8000):
    python bench_tp2.py

    # TP=2 on 8000, TP=1 on 8001 (for scaling comparison):
    python bench_tp2.py --tp2-port 8000 --tp1-port 8001

    # Quick sweep (concurrency only, default context):
    python bench_tp2.py --quick

    # Full sweep (all context lengths):
    python bench_tp2.py --full

    # Custom concurrency levels:
    python bench_tp2.py --concurrency 1,32,64,128,256,512,1024

    # Context-length scaling test:
    python bench_tp2.py --context-sweep

Single-GPU reference (RTX 5090, 32GB):
    C=1:   89 tok/s     C=32:  1,738 tok/s
    C=256: 6,615 tok/s  (peak)

TP=2 projection (PRO 6000, 192GB):
    C=1:   ~100-130 tok/s    (NCCL overhead for small batch)
    C=64:  ~4,000-6,000 tok/s
    C=256: ~8,000-10,000 tok/s
    C=512: ~10,000-12,000 tok/s  (peak, ~1.7x single GPU)
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import aiohttp


# ============================================================
# Prompts by approximate token length
# ============================================================

SHORT_PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Name three primary colors.",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
]

MEDIUM_PROMPTS = [
    "Explain how a transformer neural network works in 3 paragraphs.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the water cycle and its importance to Earth's climate.",
    "Explain the concept of entropy in thermodynamics.",
    "What is quantum entanglement and why does it matter?",
    "How does CRISPR gene editing work? Explain step by step.",
    "Describe the major phases of the software development lifecycle.",
    "What causes the northern lights and where are they visible?",
    "Explain the difference between supervised and unsupervised learning.",
    "How does public key cryptography enable secure communication?",
]

LONG_PROMPTS = [
    ("Provide a comprehensive overview of the history of artificial intelligence "
     "from the 1950s to today, covering key milestones, influential researchers, "
     "major breakthroughs, and the current state of the field. Include discussion "
     "of symbolic AI, neural networks, expert systems, the AI winters, deep learning "
     "revolution, and large language models."),
    ("Write a detailed technical explanation of how modern GPU architectures enable "
     "deep learning. Cover topics including CUDA cores, tensor cores, memory hierarchy "
     "(VRAM, L2, L1/shared memory), bandwidth considerations, kernel fusion, "
     "and how frameworks like PyTorch map operations to GPU hardware."),
    ("Explain the full lifecycle of an HTTP request from the moment a user types a "
     "URL in their browser to when the page renders. Include DNS resolution, TCP "
     "handshake, TLS negotiation, HTTP request formatting, server processing, "
     "CDN involvement, response handling, and browser rendering pipeline."),
]

CONTEXT_PROMPTS = {
    4096:  MEDIUM_PROMPTS,
    8192:  LONG_PROMPTS,
    16384: LONG_PROMPTS,
    32768: LONG_PROMPTS,
}

MODEL_NAME = "gemma-4-26B-A4B-it-NVFP4"

# ============================================================
# Single-GPU reference data (RTX 5090, 32GB, BF16 KV)
# ============================================================
TP1_REFERENCE = {
    1:   89,
    4:   201,
    16:  305,
    32:  1738,
    64:  3193,
    128: 4982,
    192: 4915,
    256: 6615,
    384: 5863,
    512: 6173,
}


@dataclass
class BenchResult:
    concurrency: int
    context_len: int
    max_tokens: int
    num_requests: int
    successes: int
    failures: int
    elapsed_s: float
    aggregate_tok_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    latency_mean_s: float
    tp1_tok_s: Optional[float]      # reference from RTX 5090 (same concurrency)
    scaling_efficiency: Optional[float]  # aggregate_tok_s / (2 * tp1_tok_s)
    server_label: str               # "TP=2" or "TP=1"


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    model: str,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
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
            else:
                return {"success": False, "latency": elapsed,
                        "error": str(data)[:200]}
    except Exception as e:
        return {"success": False, "latency": time.monotonic() - t0,
                "error": str(e)[:200]}


async def run_concurrency_level(
    url: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    context_len: int,
    model: str,
) -> tuple[list, float]:
    prompts_for_ctx = CONTEXT_PROMPTS.get(context_len, MEDIUM_PROMPTS)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def one_request(i: int):
            prompt = prompts_for_ctx[i % len(prompts_for_ctx)]
            async with semaphore:
                return await send_request(session, url, prompt, max_tokens, model)

        t0 = time.monotonic()
        results = await asyncio.gather(*[one_request(i) for i in range(num_requests)])
        elapsed = time.monotonic() - t0

    return results, elapsed


def summarize(
    results: list,
    elapsed: float,
    concurrency: int,
    context_len: int,
    max_tokens: int,
    server_label: str,
) -> BenchResult:
    successes = [r for r in results if r.get("success")]
    failures  = [r for r in results if not r.get("success")]

    if not successes:
        errors = [r.get("error", "?") for r in failures[:3]]
        print(f"  ALL REQUESTS FAILED: {errors}")
        return BenchResult(
            concurrency=concurrency, context_len=context_len, max_tokens=max_tokens,
            num_requests=len(results), successes=0, failures=len(failures),
            elapsed_s=elapsed, aggregate_tok_s=0.0,
            latency_p50_s=0, latency_p95_s=0, latency_p99_s=0, latency_mean_s=0,
            tp1_tok_s=None, scaling_efficiency=None, server_label=server_label,
        )

    latencies = sorted(r["latency"] for r in successes)
    total_completion = sum(r["completion_tokens"] for r in successes)

    n = len(latencies)
    tp1_ref = TP1_REFERENCE.get(concurrency)
    agg_tok_s = total_completion / elapsed if elapsed > 0 else 0.0
    scaling = (agg_tok_s / (2 * tp1_ref)) if tp1_ref else None

    return BenchResult(
        concurrency=concurrency,
        context_len=context_len,
        max_tokens=max_tokens,
        num_requests=len(results),
        successes=len(successes),
        failures=len(failures),
        elapsed_s=round(elapsed, 2),
        aggregate_tok_s=round(agg_tok_s, 1),
        latency_p50_s=round(latencies[n // 2], 3),
        latency_p95_s=round(latencies[min(int(n * 0.95), n - 1)], 3),
        latency_p99_s=round(latencies[min(int(n * 0.99), n - 1)], 3),
        latency_mean_s=round(statistics.mean(latencies), 3),
        tp1_tok_s=float(tp1_ref) if tp1_ref else None,
        scaling_efficiency=round(scaling, 3) if scaling else None,
        server_label=server_label,
    )


async def warmup(url: str, model: str):
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        await send_request(session, url, "Say hello.", 5, model)


async def run_sweep(
    url: str,
    concurrency_levels: List[int],
    context_len: int,
    max_tokens: int,
    requests_per_level: int,
    server_label: str,
    model: str,
) -> List[BenchResult]:
    results = []

    print(f"\n  Context: {context_len} tokens | Max output: {max_tokens} tokens | Server: {server_label}")
    print(f"  {'C':>6}  {'Req':>5}  {'Elapsed':>9}  {'Tok/s':>8}  "
          f"{'P50':>7}  {'P95':>7}  {'P99':>7}  {'TP1-ref':>8}  {'Scale':>7}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*9}  {'─'*8}  "
          f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}")

    for c in concurrency_levels:
        n = max(requests_per_level, c * 2)
        raw, elapsed = await run_concurrency_level(
            url, c, n, max_tokens, context_len, model
        )
        r = summarize(raw, elapsed, c, context_len, max_tokens, server_label)
        results.append(r)

        tp1_str = f"{r.tp1_tok_s:>7.0f}" if r.tp1_tok_s else "     N/A"
        scale_str = f"{r.scaling_efficiency:>6.1%}" if r.scaling_efficiency else "    N/A"
        print(f"  {c:>6}  {r.successes:>5}  {r.elapsed_s:>8.1f}s  "
              f"{r.aggregate_tok_s:>8.0f}  "
              f"{r.latency_p50_s:>6.2f}s  {r.latency_p95_s:>6.2f}s  {r.latency_p99_s:>6.2f}s  "
              f"{tp1_str}  {scale_str}")

    return results


async def compare_tp1_vs_tp2(
    tp1_url: str,
    tp2_url: str,
    concurrency_levels: List[int],
    context_len: int,
    max_tokens: int,
    requests_per_level: int,
    model: str,
) -> tuple[List[BenchResult], List[BenchResult]]:
    print(f"\n{'='*90}")
    print(f"  TP=1 vs TP=2 Comparison — ctx={context_len}, max_tokens={max_tokens}")
    print(f"{'='*90}")

    print("\n  Warming up TP=1...")
    await warmup(tp1_url, model)
    print("  Warming up TP=2...")
    await warmup(tp2_url, model)

    print("\n  --- TP=1 (single GPU) ---")
    tp1_results = await run_sweep(
        tp1_url, concurrency_levels, context_len, max_tokens,
        requests_per_level, "TP=1", model
    )

    print("\n  --- TP=2 (2x GPU, 192GB) ---")
    tp2_results = await run_sweep(
        tp2_url, concurrency_levels, context_len, max_tokens,
        requests_per_level, "TP=2", model
    )

    # Print side-by-side
    print(f"\n  {'C':>6}  {'TP1 tok/s':>10}  {'TP2 tok/s':>10}  {'TP2/TP1':>8}  {'vs 2x ideal':>12}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}")
    for r1, r2 in zip(tp1_results, tp2_results):
        if r1.aggregate_tok_s > 0 and r2.aggregate_tok_s > 0:
            ratio = r2.aggregate_tok_s / r1.aggregate_tok_s
            vs_ideal = ratio / 2.0
            print(f"  {r1.concurrency:>6}  {r1.aggregate_tok_s:>10.0f}  "
                  f"{r2.aggregate_tok_s:>10.0f}  {ratio:>8.2f}x  {vs_ideal:>11.1%}")

    return tp1_results, tp2_results


async def main():
    parser = argparse.ArgumentParser(description="TP=2 benchmark for RTX PRO 6000")
    parser.add_argument("--tp2-port", type=int, default=8000,
                        help="Port for TP=2 server (default: 8000)")
    parser.add_argument("--tp1-port", type=int, default=None,
                        help="Port for TP=1 server (enables comparison mode)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="Model name served by vLLM")
    parser.add_argument("--concurrency", type=str,
                        default="1,4,16,32,64,128,256,512,1024",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--context-lengths", type=str, default="4096,8192,16384,32768",
                        help="Comma-separated context lengths to test")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Max output tokens per request (default: 200)")
    parser.add_argument("--requests-per-level", type=int, default=64,
                        help="Minimum requests per concurrency level (default: 64)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sweep: C=[1,32,64,128,256,512], ctx=4096 only")
    parser.add_argument("--full", action="store_true",
                        help="Full sweep: all concurrency levels x all context lengths")
    parser.add_argument("--context-sweep", action="store_true",
                        help="Context-length scaling: fix C=128, sweep 4K→32K")
    parser.add_argument("--output", type=str, default="bench_tp2_results.json",
                        help="Output JSON file (default: bench_tp2_results.json)")
    args = parser.parse_args()

    tp2_url = f"http://localhost:{args.tp2_port}/v1/chat/completions"
    tp1_url = f"http://localhost:{args.tp1_port}/v1/chat/completions" if args.tp1_port else None

    # Parse concurrency levels
    if args.quick:
        concurrency_levels = [1, 32, 64, 128, 256, 512]
        context_lengths = [4096]
        args.requests_per_level = 32
    elif args.context_sweep:
        concurrency_levels = [128]
        context_lengths = [4096, 8192, 16384, 32768]
        args.requests_per_level = 32
    elif args.full:
        concurrency_levels = [int(c) for c in args.concurrency.split(",")]
        context_lengths = [int(c) for c in args.context_lengths.split(",")]
    else:
        # Default: concurrency sweep at 4K context
        concurrency_levels = [int(c) for c in args.concurrency.split(",")]
        context_lengths = [4096]

    print("=" * 90)
    print("  RTX PRO 6000 TP=2 Benchmark")
    print("  Hardware:   2x RTX PRO 6000 (96GB each, 192GB total, Blackwell SM120)")
    print(f"  TP=2 server: http://localhost:{args.tp2_port}")
    if tp1_url:
        print(f"  TP=1 server: http://localhost:{args.tp1_port}")
    print(f"  Concurrency: {concurrency_levels}")
    print(f"  Contexts:    {context_lengths} tokens")
    print(f"  Max output:  {args.max_tokens} tokens")
    print(f"  Req/level:   {args.requests_per_level}")
    print("=" * 90)
    print()
    print("  Single-GPU reference (RTX 5090, 32GB, BF16 KV):")
    print(f"    C=1:   89 tok/s  |  C=32: 1,738 tok/s")
    print(f"    C=256: 6,615 tok/s (peak)  |  ctx=4K, max_tokens=200")
    print()
    print("  TP=2 projection (this machine, 192GB):")
    print(f"    C=1:   ~100-130 tok/s     (NCCL overhead for small batch)")
    print(f"    C=64:  ~4,000-6,000 tok/s")
    print(f"    C=256: ~8,000-10,000 tok/s")
    print(f"    C=512: ~10,000-12,000 tok/s  (peak, ~1.7x single GPU)")
    print(f"    With FusenCache: ~1.28M token capacity → can sustain C=1024+")

    all_results = []

    if tp1_url:
        # Comparison mode
        for ctx in context_lengths:
            tp1_res, tp2_res = await compare_tp1_vs_tp2(
                tp1_url, tp2_url, concurrency_levels, ctx,
                args.max_tokens, args.requests_per_level, args.model
            )
            all_results.extend([asdict(r) for r in tp1_res + tp2_res])
    else:
        # TP=2 only sweep
        print(f"\nWarming up TP=2 server...")
        await warmup(tp2_url, args.model)
        print("Warmup done.\n")

        for ctx in context_lengths:
            results = await run_sweep(
                tp2_url, concurrency_levels, ctx,
                args.max_tokens, args.requests_per_level, "TP=2", args.model
            )
            all_results.extend([asdict(r) for r in results])

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*90}")
    print("  SUMMARY")
    print(f"{'='*90}")

    tp2_results = [r for r in all_results if r["server_label"] == "TP=2" and r["aggregate_tok_s"] > 0]
    if tp2_results:
        best = max(tp2_results, key=lambda r: r["aggregate_tok_s"])
        print(f"  Peak TP=2 throughput: {best['aggregate_tok_s']:,.0f} tok/s "
              f"at C={best['concurrency']}, ctx={best['context_len']}")
        print(f"  Best TP=2 latency:    "
              f"{min(r['latency_p50_s'] for r in tp2_results):.3f}s P50 "
              f"at C={min(tp2_results, key=lambda r: r['latency_p50_s'])['concurrency']}")

        if any(r["scaling_efficiency"] for r in tp2_results):
            best_scale = max(
                (r for r in tp2_results if r["scaling_efficiency"]),
                key=lambda r: r["aggregate_tok_s"]
            )
            print(f"  TP=2 scaling efficiency: "
                  f"{best_scale['scaling_efficiency']:.1%} of ideal 2x "
                  f"at C={best_scale['concurrency']}")

        if tp1_url:
            tp1_results = [r for r in all_results if r["server_label"] == "TP=1" and r["aggregate_tok_s"] > 0]
            if tp1_results:
                best_tp1 = max(tp1_results, key=lambda r: r["aggregate_tok_s"])
                print(f"\n  Peak TP=1 throughput: {best_tp1['aggregate_tok_s']:,.0f} tok/s "
                      f"at C={best_tp1['concurrency']}")
                ratio = best["aggregate_tok_s"] / best_tp1["aggregate_tok_s"]
                print(f"  TP=2 / TP=1 peak:     {ratio:.2f}x "
                      f"({ratio/2:.1%} of ideal 2x)")

    # Save results
    output = {
        "metadata": {
            "date": "2026-04-09",
            "hardware": "2x RTX PRO 6000 (96GB each, Blackwell SM120)",
            "model": args.model,
            "tp2_port": args.tp2_port,
            "tp1_port": args.tp1_port,
            "concurrency_levels": concurrency_levels,
            "context_lengths": context_lengths,
            "max_tokens": args.max_tokens,
        },
        "tp1_reference_rtx5090": TP1_REFERENCE,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
