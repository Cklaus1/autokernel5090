#!/usr/bin/env python3
"""DP=2 benchmark for RTX PRO 6000 workstation.

Tests two independent vLLM servers (one per GPU, PCIe — no NVLink) and the
fusen_solver routing both together.

Comparison modes:
  - Single GPU (port 8000 alone)
  - Single GPU (port 8001 alone)
  - DP=2 aggregate (fusen_solver routing across ports 8000 + 8001)
  - DP=2 per-GPU throughput (same aggregate / 2)

Context lengths tested: 4K, 8K, 16K, 32K (PRO 6000 96GB allows all of these
comfortably even without TP, unlike 32GB RTX 5090 which was limited at 32K).

Usage:
    # Both GPUs already up (serve_gemma4_dp2.sh):
    python bench_dp2.py                         # quick: C=[1,32,64,128,256] ctx=4K
    python bench_dp2.py --quick                 # same as above
    python bench_dp2.py --full                  # all C × all ctx lengths
    python bench_dp2.py --context-sweep         # C=128 fixed, 4K→32K
    python bench_dp2.py --compare-single        # each GPU alone, then both combined

    # Custom ports:
    python bench_dp2.py --gpu0-port 8000 --gpu1-port 8001

    # DP=2 aggregate via fusen_solver (if running):
    python bench_dp2.py --solver-port 9000 --full

    # Single-GPU only (GPU 0):
    python bench_dp2.py --single-gpu --port 8000

    # Context lengths:
    python bench_dp2.py --context-lengths 4096,16384,32768

Single-GPU reference (RTX 5090, 32GB, BF16 KV, 4K context):
    C=1:   89 tok/s     C=32:  1,738 tok/s
    C=256: 6,615 tok/s  (peak)

Single-GPU PRO 6000 projection (96GB, BF16 KV, 4K context):
    C=1:   ~90-95 tok/s  (same decode speed as 5090 at small batch)
    C=256: ~6,000-6,800 tok/s  (similar peak, KV pressure relief at higher C)
    C=512: ~6,500-7,000 tok/s  (PRO 6000: 300K tokens at 4K → no KV pressure at C=512)

DP=2 aggregate projection (both GPUs, zero overhead):
    C=1 per GPU / C=2 total:  ~180-190 tok/s
    C=256 per GPU / C=512 total:  ~12,000-14,000 tok/s  (perfect 2x scaling)
    FusenCache DP=2:  sustain C=1000+ (500K × 2 = 1M tokens)
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
    ("You are designing a large-scale distributed system for a real-time analytics "
     "platform that must process 10 million events per second with sub-100ms P99 "
     "latency. Describe the architecture including ingestion layer, stream processing, "
     "storage layer, and query engine. Discuss tradeoffs between consistency and "
     "availability, and how you would handle hot partitions."),
]

CONTEXT_PROMPTS = {
    4096:  MEDIUM_PROMPTS,
    8192:  LONG_PROMPTS,
    16384: LONG_PROMPTS,
    32768: LONG_PROMPTS,
}

MODEL_NAME = "gemma4-nvfp4"

# ============================================================
# Reference data
# ============================================================

# RTX 5090 single-GPU (32GB, BF16 KV, 4K context) — measured
RTX5090_REFERENCE = {
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

# PRO 6000 single-GPU projection (96GB, BF16 KV, 4K context) — not yet measured
# Filled in after first run; used for DP=2 vs per-GPU comparison.
PRO6000_SINGLE_REFERENCE: dict = {}   # populated at runtime from GPU 0 results


@dataclass
class BenchResult:
    server_label: str           # "GPU0", "GPU1", "DP2-aggregate", "DP2-per-gpu"
    concurrency: int            # inflight requests sent to THIS server (or both for DP2)
    context_len: int
    max_tokens: int
    num_requests: int
    successes: int
    failures: int
    elapsed_s: float
    aggregate_tok_s: float      # total tokens / elapsed (across all servers for DP2)
    per_gpu_tok_s: float        # aggregate_tok_s / num_gpus
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    latency_mean_s: float
    vs_rtx5090: Optional[float]       # ratio to 5090 single-GPU at same concurrency
    vs_pro6000_single: Optional[float]  # ratio to PRO 6000 single-GPU at same C
    dp2_scaling_efficiency: Optional[float]  # vs 2x single (only for DP2 rows)


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
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
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
            return {"success": False, "latency": elapsed, "error": str(data)[:200]}
    except Exception as e:
        return {"success": False, "latency": time.monotonic() - t0, "error": str(e)[:200]}


async def run_concurrency_level(
    urls: List[str],   # one URL per GPU for DP=2; single-element list for single-GPU
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    context_len: int,
    model: str,
) -> tuple[list, float]:
    """Send num_requests spread round-robin across urls with concurrency limit."""
    prompts_for_ctx = CONTEXT_PROMPTS.get(context_len, MEDIUM_PROMPTS)
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 10)

    async with aiohttp.ClientSession(connector=connector) as session:
        async def one_request(i: int):
            prompt = prompts_for_ctx[i % len(prompts_for_ctx)]
            url = urls[i % len(urls)]
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
    num_gpus: int = 1,
) -> BenchResult:
    successes = [r for r in results if r.get("success")]
    failures  = [r for r in results if not r.get("success")]

    if not successes:
        errors = [r.get("error", "?") for r in failures[:3]]
        print(f"  ALL REQUESTS FAILED: {errors}")
        return BenchResult(
            server_label=server_label, concurrency=concurrency,
            context_len=context_len, max_tokens=max_tokens,
            num_requests=len(results), successes=0, failures=len(failures),
            elapsed_s=elapsed, aggregate_tok_s=0.0, per_gpu_tok_s=0.0,
            latency_p50_s=0, latency_p95_s=0, latency_p99_s=0, latency_mean_s=0,
            vs_rtx5090=None, vs_pro6000_single=None, dp2_scaling_efficiency=None,
        )

    latencies = sorted(r["latency"] for r in successes)
    total_completion = sum(r["completion_tokens"] for r in successes)
    n = len(latencies)

    agg_tok_s = total_completion / elapsed if elapsed > 0 else 0.0
    per_gpu = agg_tok_s / num_gpus

    rtx5090_ref = RTX5090_REFERENCE.get(concurrency // num_gpus) if num_gpus > 1 else RTX5090_REFERENCE.get(concurrency)
    vs_5090 = (agg_tok_s / rtx5090_ref) if rtx5090_ref else None

    pro6k_ref = PRO6000_SINGLE_REFERENCE.get(concurrency // num_gpus) if num_gpus > 1 else PRO6000_SINGLE_REFERENCE.get(concurrency)
    vs_pro = (agg_tok_s / (pro6k_ref * num_gpus)) if (pro6k_ref and num_gpus > 1) else None
    dp2_eff = vs_pro  # same thing: actual vs num_gpus × single

    return BenchResult(
        server_label=server_label,
        concurrency=concurrency,
        context_len=context_len,
        max_tokens=max_tokens,
        num_requests=len(results),
        successes=len(successes),
        failures=len(failures),
        elapsed_s=round(elapsed, 2),
        aggregate_tok_s=round(agg_tok_s, 1),
        per_gpu_tok_s=round(per_gpu, 1),
        latency_p50_s=round(latencies[n // 2], 3),
        latency_p95_s=round(latencies[min(int(n * 0.95), n - 1)], 3),
        latency_p99_s=round(latencies[min(int(n * 0.99), n - 1)], 3),
        latency_mean_s=round(statistics.mean(latencies), 3),
        vs_rtx5090=round(vs_5090, 3) if vs_5090 else None,
        vs_pro6000_single=round(vs_pro, 3) if vs_pro else None,
        dp2_scaling_efficiency=round(dp2_eff, 3) if dp2_eff else None,
    )


async def warmup(urls: List[str], model: str):
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_request(session, url, "Say hello.", 5, model) for url in urls]
        await asyncio.gather(*tasks)


async def run_sweep(
    urls: List[str],
    concurrency_levels: List[int],
    context_len: int,
    max_tokens: int,
    requests_per_level: int,
    server_label: str,
    model: str,
    num_gpus: int = 1,
) -> List[BenchResult]:
    num_gpus_str = f"({num_gpus} GPU{'s' if num_gpus > 1 else ''})"
    results = []
    print(f"\n  [{server_label}] {num_gpus_str}  ctx={context_len}  max_tokens={max_tokens}")
    print(f"  {'C':>6}  {'Req':>5}  {'Elapsed':>9}  {'Agg tok/s':>10}  "
          f"{'Per-GPU':>8}  {'P50':>7}  {'P95':>7}  {'vs 5090':>8}  {'DP2 eff':>8}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*9}  {'─'*10}  "
          f"{'─'*8}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}")

    for c in concurrency_levels:
        n = max(requests_per_level, c * 2)
        raw, elapsed = await run_concurrency_level(
            urls, c, n, max_tokens, context_len, model
        )
        r = summarize(raw, elapsed, c, context_len, max_tokens, server_label, num_gpus)
        results.append(r)

        vs5090_str  = f"{r.vs_rtx5090:>7.2f}x" if r.vs_rtx5090 else "     N/A"
        dp2eff_str  = f"{r.dp2_scaling_efficiency:>7.1%}" if r.dp2_scaling_efficiency else "     N/A"
        pergpu_str  = f"{r.per_gpu_tok_s:>8.0f}" if num_gpus > 1 else "       -"

        print(f"  {c:>6}  {r.successes:>5}  {r.elapsed_s:>8.1f}s  "
              f"{r.aggregate_tok_s:>10.0f}  "
              f"{pergpu_str}  "
              f"{r.latency_p50_s:>6.2f}s  {r.latency_p95_s:>6.2f}s  "
              f"{vs5090_str}  {dp2eff_str}")

    return results


async def compare_single_vs_dp2(
    gpu0_url: str,
    gpu1_url: str,
    concurrency_levels: List[int],
    context_len: int,
    max_tokens: int,
    requests_per_level: int,
    model: str,
) -> tuple[List[BenchResult], List[BenchResult], List[BenchResult]]:
    """Sweep GPU 0 alone, GPU 1 alone, then both together."""
    print(f"\n{'='*100}")
    print(f"  Single GPU vs DP=2 Comparison — ctx={context_len}, max_tokens={max_tokens}")
    print(f"{'='*100}")

    print("\n  Warming up GPU 0...")
    await warmup([gpu0_url], model)
    print("  Warming up GPU 1...")
    await warmup([gpu1_url], model)

    gpu0_results = await run_sweep(
        [gpu0_url], concurrency_levels, context_len, max_tokens,
        requests_per_level, "GPU0-only", model, num_gpus=1
    )
    # Populate single-GPU reference for DP=2 efficiency calculation
    for r in gpu0_results:
        PRO6000_SINGLE_REFERENCE[r.concurrency] = r.aggregate_tok_s

    gpu1_results = await run_sweep(
        [gpu1_url], concurrency_levels, context_len, max_tokens,
        requests_per_level, "GPU1-only", model, num_gpus=1
    )

    # DP=2: double the concurrency (C per GPU stays the same)
    dp2_concurrency = [c * 2 for c in concurrency_levels]
    dp2_results = await run_sweep(
        [gpu0_url, gpu1_url], dp2_concurrency, context_len, max_tokens,
        requests_per_level, "DP2-aggregate", model, num_gpus=2
    )

    # Side-by-side summary
    print(f"\n  {'C/GPU':>6}  {'GPU0 tok/s':>11}  {'GPU1 tok/s':>11}  "
          f"{'DP2 agg':>10}  {'DP2/GPU0':>9}  {'vs 2x ideal':>12}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*11}  {'─'*10}  {'─'*9}  {'─'*12}")
    for r0, r1, r2 in zip(gpu0_results, gpu1_results, dp2_results):
        if r0.aggregate_tok_s > 0 and r2.aggregate_tok_s > 0:
            ratio = r2.aggregate_tok_s / r0.aggregate_tok_s
            vs_ideal = ratio / 2.0
            print(f"  {r0.concurrency:>6}  {r0.aggregate_tok_s:>11.0f}  "
                  f"{r1.aggregate_tok_s:>11.0f}  "
                  f"{r2.aggregate_tok_s:>10.0f}  "
                  f"{ratio:>9.2f}x  {vs_ideal:>11.1%}")

    return gpu0_results, gpu1_results, dp2_results


async def main():
    parser = argparse.ArgumentParser(description="DP=2 benchmark for 2x RTX PRO 6000 Max-Q")
    parser.add_argument("--gpu0-port", type=int, default=8000,
                        help="Port for GPU 0 server (default: 8000)")
    parser.add_argument("--gpu1-port", type=int, default=8001,
                        help="Port for GPU 1 server (default: 8001)")
    parser.add_argument("--solver-port", type=int, default=None,
                        help="Port for fusen_solver (enables solver mode)")
    parser.add_argument("--single-gpu", action="store_true",
                        help="Benchmark GPU 0 only (use --port to select)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for single-GPU mode (default: 8000)")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--concurrency", type=str,
                        default="1,4,16,32,64,128,256,512",
                        help="Comma-separated concurrency levels (per-GPU for DP=2)")
    parser.add_argument("--context-lengths", type=str, default="4096,8192,16384,32768",
                        help="Comma-separated context lengths")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--requests-per-level", type=int, default=64)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: C=[1,32,64,128,256,512], ctx=4096 only")
    parser.add_argument("--full", action="store_true",
                        help="Full: all concurrency × all context lengths")
    parser.add_argument("--context-sweep", action="store_true",
                        help="Context sweep: C=128 fixed, 4K→32K")
    parser.add_argument("--compare-single", action="store_true",
                        help="Compare GPU0 alone vs GPU1 alone vs DP=2 combined")
    parser.add_argument("--output", type=str, default="bench_dp2_results.json")
    args = parser.parse_args()

    gpu0_url = f"http://localhost:{args.gpu0_port}/v1/chat/completions"
    gpu1_url = f"http://localhost:{args.gpu1_port}/v1/chat/completions"
    solver_url = f"http://localhost:{args.solver_port}/v1/chat/completions" if args.solver_port else None

    # Build sweep parameters
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
        # Default: concurrency sweep at 4K only
        concurrency_levels = [int(c) for c in args.concurrency.split(",")]
        context_lengths = [4096]

    # ============================================================
    # Banner
    # ============================================================
    print("=" * 100)
    print("  RTX PRO 6000 Max-Q DP=2 Benchmark")
    print("  Hardware:  2x RTX PRO 6000 Max-Q (96GB each, PCIe, Blackwell SM120)")
    print("  Strategy:  DP=2 (two independent servers, zero inter-GPU communication)")
    print(f"  GPU 0:     http://localhost:{args.gpu0_port}")
    if not args.single_gpu:
        print(f"  GPU 1:     http://localhost:{args.gpu1_port}")
    if solver_url:
        print(f"  Solver:    http://localhost:{args.solver_port}")
    print(f"  Concurrency (per-GPU for DP=2): {concurrency_levels}")
    print(f"  Contexts:  {context_lengths} tokens")
    print(f"  Max output: {args.max_tokens} tokens")
    print("=" * 100)
    print()
    print("  RTX 5090 single-GPU reference (32GB, BF16 KV, 4K ctx):")
    print(f"    C=1: 89 tok/s  |  C=32: 1,738 tok/s  |  C=256: 6,615 tok/s (peak)")
    print()
    print("  PRO 6000 single-GPU projection (96GB, BF16 KV, 4K ctx):")
    print(f"    C=1: ~90-95 tok/s  |  C=256: ~6,000-6,800 tok/s")
    print(f"    C=512: ~6,500-7,000 tok/s  (no KV pressure — 300K tokens at 4K ctx)")
    print()
    print("  DP=2 projection (both GPUs, zero overhead):")
    print(f"    C=512/GPU (C=1024 total): ~12,000-14,000 tok/s  (near-perfect 2x)")
    print(f"    FusenCache DP=2: 1M token KV pool — sustain C=1000+ at 4K context")
    print()

    all_results = []

    # ============================================================
    # Benchmark modes
    # ============================================================
    if args.single_gpu:
        single_url = f"http://localhost:{args.port}/v1/chat/completions"
        label = f"GPU-port{args.port}"
        print(f"Warming up {single_url}...")
        await warmup([single_url], args.model)
        for ctx in context_lengths:
            res = await run_sweep(
                [single_url], concurrency_levels, ctx, args.max_tokens,
                args.requests_per_level, label, args.model, num_gpus=1
            )
            all_results.extend([asdict(r) for r in res])

    elif args.compare_single:
        for ctx in context_lengths:
            g0, g1, dp2 = await compare_single_vs_dp2(
                gpu0_url, gpu1_url, concurrency_levels, ctx,
                args.max_tokens, args.requests_per_level, args.model
            )
            all_results.extend([asdict(r) for r in g0 + g1 + dp2])

    elif solver_url:
        # Benchmark via fusen_solver (treats it as a single endpoint)
        print(f"Warming up fusen_solver ({solver_url})...")
        await warmup([solver_url], args.model)
        for ctx in context_lengths:
            # Concurrency values here are total, solver distributes internally
            dp2_concurrency = [c * 2 for c in concurrency_levels]
            res = await run_sweep(
                [solver_url], dp2_concurrency, ctx, args.max_tokens,
                args.requests_per_level, "solver-DP2", args.model, num_gpus=2
            )
            all_results.extend([asdict(r) for r in res])

    else:
        # Default: DP=2 aggregate sweep (round-robin across both GPUs)
        print(f"Warming up both GPUs...")
        await warmup([gpu0_url, gpu1_url], args.model)
        print("Warmup done.\n")
        for ctx in context_lengths:
            # DP=2: double the concurrency so each GPU sees the right load
            dp2_concurrency = [c * 2 for c in concurrency_levels]
            res = await run_sweep(
                [gpu0_url, gpu1_url], dp2_concurrency, ctx, args.max_tokens,
                args.requests_per_level, "DP2-aggregate", args.model, num_gpus=2
            )
            all_results.extend([asdict(r) for r in res])

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*100}")
    print("  SUMMARY")
    print(f"{'='*100}")

    dp2_res = [r for r in all_results if "DP2" in r["server_label"] and r["aggregate_tok_s"] > 0]
    single_res = [r for r in all_results
                  if r["server_label"] in ("GPU0-only", "GPU1-only", f"GPU-port{args.port}")
                  and r["aggregate_tok_s"] > 0]

    if dp2_res:
        best_dp2 = max(dp2_res, key=lambda r: r["aggregate_tok_s"])
        print(f"  Peak DP=2 aggregate: {best_dp2['aggregate_tok_s']:,.0f} tok/s "
              f"at C={best_dp2['concurrency']}, ctx={best_dp2['context_len']}")
        print(f"  Peak per-GPU:        {best_dp2['per_gpu_tok_s']:,.0f} tok/s")
        if best_dp2.get("dp2_scaling_efficiency"):
            print(f"  DP=2 efficiency:     {best_dp2['dp2_scaling_efficiency']:.1%} of 2x ideal")

    if single_res:
        best_sg = max(single_res, key=lambda r: r["aggregate_tok_s"])
        print(f"  Peak single-GPU:     {best_sg['aggregate_tok_s']:,.0f} tok/s "
              f"at C={best_sg['concurrency']}, ctx={best_sg['context_len']}")

    if dp2_res and single_res:
        ratio = best_dp2["aggregate_tok_s"] / best_sg["aggregate_tok_s"]
        print(f"  DP=2 / single-GPU:   {ratio:.2f}x  ({ratio/2:.1%} of ideal 2x)")

    # Save
    output = {
        "metadata": {
            "date": "2026-04-09",
            "hardware": "2x RTX PRO 6000 Max-Q (96GB each, PCIe, Blackwell SM120)",
            "strategy": "DP=2 (two independent servers)",
            "model": args.model,
            "gpu0_port": args.gpu0_port,
            "gpu1_port": args.gpu1_port,
            "solver_port": args.solver_port,
            "concurrency_levels": concurrency_levels,
            "context_lengths": context_lengths,
            "max_tokens": args.max_tokens,
        },
        "rtx5090_reference": RTX5090_REFERENCE,
        "pro6000_single_reference": PRO6000_SINGLE_REFERENCE,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
