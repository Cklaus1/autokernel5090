#!/usr/bin/env python3
"""Benchmark: block-size and chunked prefill tuning for vLLM + FusenCache.

Tests the impact of:
  - --block-size: 16 (default) vs 32 vs 64
  - --max-num-batched-tokens: 2048 vs 4096 vs 8192
  - --max-num-partial-prefills: 1 vs 2 vs 4

Each configuration is tested against a running server.  Since the GPU is
already occupied by the production server on port 8000, this script targets
a TEST container on port 8001.  Launch the test container separately with:

    docker run --rm --gpus all --memory=36g \\
      -v /root/models:/models:ro -p 8001:8000 --name vllm-test \\
      vllm-built python3 -m vllm.entrypoints.openai.api_server \\
        --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \\
        --quantization modelopt --max-model-len 4096 \\
        -cc.mode none -cc.cudagraph_mode full \\
        --block-size 32 \\
        --max-num-batched-tokens 8192

Then re-launch the container for each configuration in the sweep matrix and
run: python bench_block_size.py --mode single --port 8001

Or use --mode sweep to drive the full matrix automatically via docker commands
(requires docker socket access and that no other container owns the GPU).

Usage:
    # Run against an already-running test server:
    python bench_block_size.py --port 8001 [--mode single]

    # Run the full sweep (stops/starts containers automatically):
    python bench_block_size.py --mode sweep

    # Run with FusenCache k4v4b64 KV type (uses fusen_kv plugin):
    python bench_block_size.py --port 8001 --kv-type k4v4b64
"""

import argparse
import asyncio
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import aiohttp

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
DEFAULT_PORT = 8001
HEALTH_TIMEOUT_S = 300  # wait up to 5 min for server to be ready

# Test matrix dimensions
BLOCK_SIZES = [16, 32, 64]
MAX_BATCHED_TOKENS = [2048, 4096, 8192]
MAX_PARTIAL_PREFILLS = [1, 2, 4]

# Benchmark workloads: (concurrency, label)
CONCURRENCY_LEVELS = [
    (32, "C=32"),
    (128, "C=128"),
]

# Prompt lengths (tokens) to exercise during each config
CONTEXT_LENGTHS = [256, 1024]  # short + medium to stress both paths

# Tokens generated per request
MAX_TOKENS_OUT = 128

RESULTS_PATH = Path(__file__).parent / "block_size_results.json"

# Docker image and base args (used only in --mode sweep)
DOCKER_IMAGE = "vllm-built"
DOCKER_NAME = "vllm-test"
FUSEN_MOUNT = "/root/projects/autokernel/fusen_kv:/fusen/fusen_kv:ro"


# ── Prompt builder ─────────────────────────────────────────────────────────────

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold, but some things shine bright.",
    "In the beginning, the universe was created, surprising many.",
    "The rain in Spain falls mainly on the plain near the mountain.",
    "Every great achievement was once considered impossible by experts.",
    "Knowledge is power, but wisdom is knowing how to use it well.",
    "Time flies like an arrow, but fruit flies like a banana split.",
    "The best way to predict the future is to invent it yourself.",
]


def make_prompt(target_tokens: int) -> str:
    """Generate a filler prompt of approximately target_tokens length."""
    base = ""
    i = 0
    while len(base) < target_tokens * 5:
        base += _SENTENCES[i % len(_SENTENCES)] + " "
        i += 1
    return base[: int(target_tokens * 4)]  # ~4 chars/token for Gemma


# ── Request helpers ────────────────────────────────────────────────────────────

@dataclass
class ReqResult:
    ttft_ms: float = 0.0
    decode_tps: float = 0.0
    total_ms: float = 0.0
    completion_tokens: int = 0
    error: Optional[str] = None


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = MAX_TOKENS_OUT,
) -> ReqResult:
    result = ReqResult()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    tokens = 0

    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                return result
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                chunk = line[6:]
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    content = data["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        if t_first is None:
                            t_first = time.perf_counter()
                        tokens += 1
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    except Exception as exc:
        result.error = str(exc)
        return result

    t_end = time.perf_counter()
    result.total_ms = (t_end - t_start) * 1000
    result.completion_tokens = tokens
    if t_first is not None:
        result.ttft_ms = (t_first - t_start) * 1000
        decode_sec = t_end - t_first
        if decode_sec > 0 and tokens > 1:
            result.decode_tps = (tokens - 1) / decode_sec
    return result


async def health_check(base_url: str) -> bool:
    """Return True if /health returns 200."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def wait_for_server(base_url: str, timeout_s: int = HEALTH_TIMEOUT_S) -> bool:
    """Poll /health until ready or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await health_check(base_url):
            return True
        await asyncio.sleep(3)
    return False


async def get_kv_usage(base_url: str) -> float:
    """Read KV cache utilisation from Prometheus /metrics endpoint."""
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{base_url}/metrics",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                text = await resp.text()
                for line in text.splitlines():
                    if line.startswith("vllm:kv_cache_usage_perc{"):
                        return float(line.split()[-1])
    except Exception:
        pass
    return -1.0


# ── Benchmark runner ───────────────────────────────────────────────────────────

async def run_concurrency(
    base_url: str,
    model: str,
    concurrency: int,
    ctx_len: int,
    n_warmup: int = 2,
    n_measure: int = 20,
) -> dict:
    """Fire `concurrency` simultaneous requests and collect stats."""
    prompt = make_prompt(ctx_len)

    # Warmup
    async with aiohttp.ClientSession() as session:
        warmup_tasks = [
            send_request(session, base_url, model, prompt)
            for _ in range(min(n_warmup, concurrency))
        ]
        await asyncio.gather(*warmup_tasks)
    await asyncio.sleep(1.0)

    # Measure
    all_results: List[ReqResult] = []
    batches_needed = math.ceil(n_measure / concurrency)

    for _ in range(batches_needed):
        async with aiohttp.ClientSession() as session:
            tasks = [
                send_request(session, base_url, model, prompt)
                for _ in range(concurrency)
            ]
            batch = await asyncio.gather(*tasks)
            all_results.extend(batch)

    kv_pct = await get_kv_usage(base_url)
    ok = [r for r in all_results if not r.error]
    errors = len(all_results) - len(ok)

    if not ok:
        return {
            "concurrency": concurrency,
            "ctx_len": ctx_len,
            "error": "all requests failed",
            "errors": errors,
        }

    ttfts = sorted(r.ttft_ms for r in ok if r.ttft_ms > 0)
    decode_rates = [r.decode_tps for r in ok if r.decode_tps > 0]
    total_toks = sum(r.completion_tokens for r in ok)

    def pct(arr, p):
        if not arr:
            return 0.0
        idx = min(int(len(arr) * p), len(arr) - 1)
        return arr[idx]

    return {
        "concurrency": concurrency,
        "ctx_len": ctx_len,
        "n_requests": len(ok),
        "errors": errors,
        "aggregate_tps": round(total_toks / (sum(r.total_ms for r in ok) / 1000 / len(ok)), 1),
        "avg_decode_tps": round(sum(decode_rates) / len(decode_rates), 1) if decode_rates else 0,
        "ttft_p50_ms": round(pct(ttfts, 0.50), 1),
        "ttft_p95_ms": round(pct(ttfts, 0.95), 1),
        "ttft_p99_ms": round(pct(ttfts, 0.99), 1),
        "kv_usage_pct": round(kv_pct * 100, 2),
    }


async def benchmark_server(base_url: str, model: str, label: str) -> dict:
    """Run the full benchmark matrix against an already-running server."""
    print(f"\n{'='*70}")
    print(f"  Config: {label}")
    print(f"  Server: {base_url}")
    print(f"{'='*70}")

    all_rows = []
    for ctx_len in CONTEXT_LENGTHS:
        for concurrency, conc_label in CONCURRENCY_LEVELS:
            print(
                f"  ctx={ctx_len:>5}  {conc_label:<8}  ... ",
                end="",
                flush=True,
            )
            row = await run_concurrency(base_url, model, concurrency, ctx_len)
            row["config"] = label
            all_rows.append(row)
            if "error" in row:
                print(f"ERROR: {row['error']}")
            else:
                print(
                    f"agg={row['aggregate_tps']:>7.1f} t/s  "
                    f"decode={row['avg_decode_tps']:>6.1f} t/s  "
                    f"p99_ttft={row['ttft_p99_ms']:>7.1f}ms  "
                    f"kv={row['kv_usage_pct']:>5.1f}%"
                )
            await asyncio.sleep(2.0)

    return {"label": label, "rows": all_rows}


# ── Docker helpers (--mode sweep) ─────────────────────────────────────────────

def docker_stop(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def docker_start(
    block_size: int,
    max_batched_tokens: int,
    max_partial_prefills: int,
    kv_type: str,
    port: int,
) -> subprocess.Popen:
    """Launch a vLLM test container with the given config. Returns the process."""
    extra_env = []
    if kv_type == "k4v4b64":
        extra_env = [
            "-e", "PYTHONPATH=/fusen",
            "-v", FUSEN_MOUNT,
        ]
        entrypoint = "python3 /fusen/fusen_kv/launch_vllm.py"
        kv_args = ["--kv-cache-dtype", kv_type]
    else:
        entrypoint = "python3 -m vllm.entrypoints.openai.api_server"
        kv_args = ["--kv-cache-dtype", "fp8_e4m3"]

    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--memory=36g",
        "-v", "/root/models:/models:ro",
        "-p", f"{port}:8000",
        "--name", DOCKER_NAME,
    ] + extra_env + [
        DOCKER_IMAGE,
        *entrypoint.split(),
        "--model", MODEL_PATH,
        "--quantization", "modelopt",
        "--max-model-len", "4096",
        "-cc.mode", "none",
        "-cc.cudagraph_mode", "full",
        "--block-size", str(block_size),
        "--max-num-batched-tokens", str(max_batched_tokens),
        "--max-num-partial-prefills", str(max_partial_prefills),
    ] + kv_args

    print(f"\n  Launching: block_size={block_size}  batched_tokens={max_batched_tokens}"
          f"  partial_prefills={max_partial_prefills}  kv={kv_type}")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def sweep_mode(port: int, kv_type: str) -> None:
    """Drive the full test matrix by starting/stopping docker containers."""
    base_url = f"http://localhost:{port}"
    all_results = []

    for block_size in BLOCK_SIZES:
        for max_bt in MAX_BATCHED_TOKENS:
            for max_pp in MAX_PARTIAL_PREFILLS:
                label = f"bs{block_size}_bt{max_bt}_pp{max_pp}"

                # Stop any existing test container
                docker_stop(DOCKER_NAME)
                await asyncio.sleep(2)

                # Start new container
                proc = docker_start(block_size, max_bt, max_pp, kv_type, port)

                # Wait for it to become healthy
                print(f"  Waiting for server ({label}) ...", end="", flush=True)
                ready = await wait_for_server(base_url)
                if not ready:
                    print(" TIMEOUT — skipping")
                    proc.terminate()
                    continue
                print(" ready")

                try:
                    result = await benchmark_server(base_url, MODEL_PATH, label)
                    all_results.append({
                        "block_size": block_size,
                        "max_batched_tokens": max_bt,
                        "max_partial_prefills": max_pp,
                        "kv_type": kv_type,
                        **result,
                    })
                finally:
                    proc.terminate()
                    docker_stop(DOCKER_NAME)
                    await asyncio.sleep(5)

    _save_and_print(all_results)


async def single_mode(port: int, kv_type: str) -> None:
    """Benchmark a single already-running server on the given port."""
    base_url = f"http://localhost:{port}"
    print(f"Checking server at {base_url} ...")
    ready = await health_check(base_url)
    if not ready:
        print("ERROR: server not responding at /health. Start it first.")
        sys.exit(1)
    print("Server is ready.")

    label = f"manual_port{port}"
    result = await benchmark_server(base_url, MODEL_PATH, label)
    _save_and_print([result])


def _save_and_print(results: list) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    # Pretty summary table
    print("\n" + "=" * 90)
    print(f"{'Config':<30} {'ctx':>5} {'C':>5} {'Agg t/s':>9} {'Decode t/s':>11} {'P99 TTFT':>10} {'KV%':>7}")
    print("-" * 90)
    for entry in results:
        for row in entry.get("rows", []):
            if "error" in row:
                print(f"{entry['label']:<30} {'ERROR':>5}")
                continue
            print(
                f"{entry['label']:<30}"
                f"{row['ctx_len']:>5}"
                f"{row['concurrency']:>5}"
                f"  {row['aggregate_tps']:>8.1f}"
                f"  {row['avg_decode_tps']:>9.1f}"
                f"  {row['ttft_p99_ms']:>9.1f}"
                f"  {row['kv_usage_pct']:>5.1f}%"
            )


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Block-size and chunked prefill benchmark")
    parser.add_argument(
        "--mode",
        choices=["single", "sweep"],
        default="single",
        help=(
            "single: benchmark a running server on --port.  "
            "sweep: iterate over the full matrix by starting docker containers."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port of the test server (default: 8001; production is 8000)",
    )
    parser.add_argument(
        "--kv-type",
        default="fp8_e4m3",
        choices=["fp8_e4m3", "auto", "k4v4b64"],
        help=(
            "KV cache dtype.  Use k4v4b64 to test FusenCache "
            "(requires fusen_kv plugin inside the container)."
        ),
    )
    args = parser.parse_args()

    if args.mode == "sweep":
        print("WARNING: sweep mode will stop/start docker containers on the GPU.")
        print("  Ensure the production server on port 8000 is NOT using the GPU,")
        print("  or run this on a separate machine / time window.")
        print()
        asyncio.run(sweep_mode(args.port, args.kv_type))
    else:
        asyncio.run(single_mode(args.port, args.kv_type))


if __name__ == "__main__":
    main()
