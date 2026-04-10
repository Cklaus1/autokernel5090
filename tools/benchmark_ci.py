#!/usr/bin/env python3
"""CI-style benchmark suite for vLLM serving.

Runs a standard concurrency sweep (C=1, C=32, C=128, C=256), compares
against a stored baseline, reports regressions/improvements, and persists
results in TSV with timestamps.

Usage:
    python tools/benchmark_ci.py                           # run full suite
    python tools/benchmark_ci.py --concurrencies 1 32      # custom sweep
    python tools/benchmark_ci.py --save-baseline           # save as new baseline
    python tools/benchmark_ci.py --compare-only            # compare last run vs baseline
    python tools/benchmark_ci.py --url http://host:8000    # custom server

Designed to run as a cron job or git hook.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# We use stdlib only for HTTP to avoid external deps.
# For async benchmarking we use asyncio + the approach from bench_93k_serving.py.
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONCURRENCIES = [1, 32, 128, 256]
DEFAULT_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "tools" / "ci_results"
BASELINE_FILE = RESULTS_DIR / "baseline.json"
RESULTS_TSV = RESULTS_DIR / "benchmark_history.tsv"

PROMPTS = [
    "What is 2+2?",
    "Explain the difference between TCP and UDP in 2-3 sentences.",
    "Write a Python function that checks if a number is prime.",
    "List 5 best practices for writing clean code.",
    "What are the main differences between SQL and NoSQL databases?",
    "Write a short story (2 paragraphs) about a robot discovering music.",
    "Explain the CAP theorem in distributed systems with examples.",
    "Design a REST API for a todo list application with example JSON.",
]

MAX_TOKENS = 128  # Keep short for CI speed
WARMUP_REQUESTS = 3
BENCH_DURATION_SEC = 30  # Per concurrency level


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    duration_sec: float
    throughput_tok_per_s: float
    avg_latency_s: float
    p50_latency_s: float
    p99_latency_s: float
    timestamp: str


# ---------------------------------------------------------------------------
# HTTP-based benchmark (aiohttp)
# ---------------------------------------------------------------------------

async def _send_request_aiohttp(
    session: "aiohttp.ClientSession",
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Tuple[bool, int, float]:
    """Send one chat completion request. Returns (success, tokens, latency)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    t0 = time.monotonic()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                return False, 0, time.monotonic() - t0
            data = await resp.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return True, tokens, time.monotonic() - t0
    except Exception:
        return False, 0, time.monotonic() - t0


async def run_bench_aiohttp(
    base_url: str,
    model: str,
    concurrency: int,
    duration_sec: float,
) -> BenchResult:
    """Run benchmark at a given concurrency for duration_sec seconds."""
    import random

    url = base_url.rstrip("/") + "/v1/chat/completions"
    results: List[Tuple[bool, int, float]] = []
    t_start = time.monotonic()

    async def worker():
        async with aiohttp.ClientSession() as session:
            while time.monotonic() - t_start < duration_sec:
                prompt = random.choice(PROMPTS)
                result = await _send_request_aiohttp(session, url, model, prompt, MAX_TOKENS)
                results.append(result)

    # Warmup
    async with aiohttp.ClientSession() as session:
        for _ in range(WARMUP_REQUESTS):
            await _send_request_aiohttp(session, url, model, PROMPTS[0], 16)

    # Run
    t_start = time.monotonic()
    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*tasks)
    total_dur = time.monotonic() - t_start

    # Compute stats
    successes = [(ok, tok, lat) for ok, tok, lat in results if ok]
    failures = len(results) - len(successes)
    total_tokens = sum(tok for _, tok, _ in successes)
    latencies = sorted(lat for _, _, lat in successes)

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        p50 = latencies[int(len(latencies) * 0.50)]
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
    else:
        avg_lat = p50 = p99 = 0.0

    return BenchResult(
        concurrency=concurrency,
        total_requests=len(results),
        successful_requests=len(successes),
        failed_requests=failures,
        total_tokens=total_tokens,
        duration_sec=round(total_dur, 2),
        throughput_tok_per_s=round(total_tokens / total_dur, 1) if total_dur > 0 else 0,
        avg_latency_s=round(avg_lat, 3),
        p50_latency_s=round(p50, 3),
        p99_latency_s=round(p99, 3),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Synchronous fallback (urllib, sequential — for envs without aiohttp)
# ---------------------------------------------------------------------------

def _send_request_sync(url: str, model: str, prompt: str, max_tokens: int) -> Tuple[bool, int, float]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    t0 = time.monotonic()
    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return True, tokens, time.monotonic() - t0
    except Exception:
        return False, 0, time.monotonic() - t0


def run_bench_sync(
    base_url: str,
    model: str,
    concurrency: int,
    duration_sec: float,
) -> BenchResult:
    """Synchronous fallback benchmark (concurrency=1 only, sequential)."""
    import random

    url = base_url.rstrip("/") + "/v1/chat/completions"

    # Warmup
    for _ in range(WARMUP_REQUESTS):
        _send_request_sync(url, model, PROMPTS[0], 16)

    results: List[Tuple[bool, int, float]] = []
    t_start = time.monotonic()
    while time.monotonic() - t_start < duration_sec:
        prompt = random.choice(PROMPTS)
        result = _send_request_sync(url, model, prompt, MAX_TOKENS)
        results.append(result)

    total_dur = time.monotonic() - t_start
    successes = [(ok, tok, lat) for ok, tok, lat in results if ok]
    failures = len(results) - len(successes)
    total_tokens = sum(tok for _, tok, _ in successes)
    latencies = sorted(lat for _, _, lat in successes)

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        p50 = latencies[int(len(latencies) * 0.50)]
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
    else:
        avg_lat = p50 = p99 = 0.0

    return BenchResult(
        concurrency=concurrency,
        total_requests=len(results),
        successful_requests=len(successes),
        failed_requests=failures,
        total_tokens=total_tokens,
        duration_sec=round(total_dur, 2),
        throughput_tok_per_s=round(total_tokens / total_dur, 1) if total_dur > 0 else 0,
        avg_latency_s=round(avg_lat, 3),
        p50_latency_s=round(p50, 3),
        p99_latency_s=round(p99, 3),
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def detect_model(base_url: str) -> str:
    try:
        url = base_url.rstrip("/") + "/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                return models[0].get("id", "unknown")
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

def save_baseline(results: List[BenchResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for r in results:
        data[str(r.concurrency)] = {
            "throughput_tok_per_s": r.throughput_tok_per_s,
            "avg_latency_s": r.avg_latency_s,
            "p50_latency_s": r.p50_latency_s,
            "p99_latency_s": r.p99_latency_s,
            "timestamp": r.timestamp,
        }
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Baseline saved to {BASELINE_FILE}")


def load_baseline() -> Optional[Dict]:
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# TSV logging
# ---------------------------------------------------------------------------

TSV_HEADER = (
    "timestamp\tconcurrency\ttotal_requests\tsuccessful\tfailed\t"
    "total_tokens\tduration_sec\tthroughput_tok_s\t"
    "avg_latency_s\tp50_latency_s\tp99_latency_s"
)


def append_tsv(results: List[BenchResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a") as f:
        if write_header:
            f.write(TSV_HEADER + "\n")
        for r in results:
            f.write(
                f"{r.timestamp}\t{r.concurrency}\t{r.total_requests}\t"
                f"{r.successful_requests}\t{r.failed_requests}\t"
                f"{r.total_tokens}\t{r.duration_sec}\t{r.throughput_tok_per_s}\t"
                f"{r.avg_latency_s}\t{r.p50_latency_s}\t{r.p99_latency_s}\n"
            )
    print(f"Results appended to {RESULTS_TSV}")


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def compare_report(results: List[BenchResult], baseline: Optional[Dict]) -> int:
    """Print comparison table. Returns exit code: 0=pass, 1=regression."""
    W = 72
    print("=" * W)
    print("  FusenAI CI Benchmark Report".center(W))
    print("=" * W)

    has_regression = False
    regression_threshold = 0.10  # 10% throughput drop = regression
    improvement_threshold = 0.05  # 5% improvement = notable

    for r in results:
        print(f"\n  Concurrency: {r.concurrency}")
        print(f"    Requests:   {r.successful_requests}/{r.total_requests} "
              f"({r.failed_requests} failed)")
        print(f"    Throughput: {r.throughput_tok_per_s:,.1f} tok/s")
        print(f"    Latency:    avg={r.avg_latency_s:.3f}s  "
              f"p50={r.p50_latency_s:.3f}s  p99={r.p99_latency_s:.3f}s")

        if baseline and str(r.concurrency) in baseline:
            bl = baseline[str(r.concurrency)]
            bl_tps = bl["throughput_tok_per_s"]
            if bl_tps > 0:
                pct = (r.throughput_tok_per_s - bl_tps) / bl_tps * 100
                indicator = "PASS"
                if pct < -regression_threshold * 100:
                    indicator = "REGRESSION"
                    has_regression = True
                elif pct > improvement_threshold * 100:
                    indicator = "IMPROVEMENT"

                print(f"    vs baseline: {pct:+.1f}%  [{indicator}]")
                print(f"      baseline:  {bl_tps:,.1f} tok/s @ {bl.get('timestamp', 'unknown')}")
            else:
                print(f"    vs baseline: N/A (baseline throughput was 0)")
        else:
            print(f"    vs baseline: N/A (no baseline for C={r.concurrency})")

    print("\n" + "=" * W)
    if has_regression:
        print("  RESULT: REGRESSION DETECTED")
        print("=" * W)
        return 1
    else:
        print("  RESULT: PASS")
        print("=" * W)
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_suite_async(
    base_url: str,
    model: str,
    concurrencies: List[int],
    duration: float,
) -> List[BenchResult]:
    results = []
    for c in concurrencies:
        print(f"\nRunning C={c} for {duration}s...")
        r = await run_bench_aiohttp(base_url, model, c, duration)
        print(f"  -> {r.throughput_tok_per_s:,.1f} tok/s, "
              f"{r.successful_requests} reqs, p50={r.p50_latency_s:.3f}s")
        results.append(r)
    return results


def run_suite_sync(
    base_url: str,
    model: str,
    concurrencies: List[int],
    duration: float,
) -> List[BenchResult]:
    results = []
    for c in concurrencies:
        print(f"\nRunning C={c} for {duration}s (sync mode, sequential)...")
        r = run_bench_sync(base_url, model, c, duration)
        print(f"  -> {r.throughput_tok_per_s:,.1f} tok/s, "
              f"{r.successful_requests} reqs, p50={r.p50_latency_s:.3f}s")
        results.append(r)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="CI benchmark suite for vLLM serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                                # full sweep: C=1,32,128,256
  %(prog)s --concurrencies 1 32           # quick test
  %(prog)s --save-baseline                # run + save as baseline
  %(prog)s --compare-only                 # compare last stored results vs baseline
  %(prog)s --duration 60                  # 60s per concurrency level
""",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="vLLM server URL")
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=DEFAULT_CONCURRENCIES,
        help="Concurrency levels to test (default: 1 32 128 256)",
    )
    parser.add_argument("--duration", type=float, default=BENCH_DURATION_SEC,
                        help="Seconds per concurrency level (default: 30)")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save results as the new baseline")
    parser.add_argument("--compare-only", action="store_true",
                        help="Skip benchmark, just compare last TSV entry vs baseline")
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected if omitted)")
    args = parser.parse_args()

    if args.compare_only:
        baseline = load_baseline()
        if baseline is None:
            print("No baseline found. Run with --save-baseline first.")
            sys.exit(1)
        # Read last results from TSV
        if not RESULTS_TSV.exists():
            print("No benchmark history found. Run a benchmark first.")
            sys.exit(1)
        # Parse last batch from TSV
        lines = RESULTS_TSV.read_text().strip().split("\n")
        if len(lines) < 2:
            print("No results in history TSV.")
            sys.exit(1)
        # Find last timestamp batch
        last_ts = lines[-1].split("\t")[0]
        results = []
        for line in lines[1:]:
            parts = line.split("\t")
            if parts[0] == last_ts or True:  # include all from last run
                results.append(BenchResult(
                    concurrency=int(parts[1]),
                    total_requests=int(parts[2]),
                    successful_requests=int(parts[3]),
                    failed_requests=int(parts[4]),
                    total_tokens=int(parts[5]),
                    duration_sec=float(parts[6]),
                    throughput_tok_per_s=float(parts[7]),
                    avg_latency_s=float(parts[8]),
                    p50_latency_s=float(parts[9]),
                    p99_latency_s=float(parts[10]),
                    timestamp=parts[0],
                ))
        # Deduplicate: keep last entry per concurrency
        by_c: Dict[int, BenchResult] = {}
        for r in results:
            by_c[r.concurrency] = r
        exit_code = compare_report(list(by_c.values()), baseline)
        sys.exit(exit_code)

    # Detect model
    model = args.model or detect_model(args.url)
    if model == "unknown":
        print("WARNING: Could not detect model. Is vLLM running at", args.url, "?")
        print("Use --model to specify manually, or ensure server is up.\n")

    print(f"FusenAI CI Benchmark Suite")
    print(f"Server: {args.url}")
    print(f"Model: {model}")
    print(f"Concurrencies: {args.concurrencies}")
    print(f"Duration: {args.duration}s per level")

    # Run benchmarks
    if HAS_AIOHTTP:
        results = asyncio.run(run_suite_async(args.url, model, args.concurrencies, args.duration))
    else:
        print("WARNING: aiohttp not installed. Using synchronous mode (no real concurrency).")
        results = run_suite_sync(args.url, model, args.concurrencies, args.duration)

    # Save results
    append_tsv(results)

    if args.save_baseline:
        save_baseline(results)

    # Compare
    baseline = load_baseline()
    exit_code = compare_report(results, baseline)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
