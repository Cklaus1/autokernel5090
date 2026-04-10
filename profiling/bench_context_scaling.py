#!/usr/bin/env python3
"""Benchmark: throughput vs context length on vLLM server.

Measures TTFT, decode throughput, and KV cache utilization at various
prompt lengths and concurrency levels to find the KV pressure point.
"""

import asyncio
import time
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

BASE_URL = "http://localhost:8000"
MODEL = "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MAX_TOKENS = 128  # fixed generation length for all tests

# Context lengths to sweep
CONTEXT_LENGTHS = [128, 256, 512, 1024, 2048, 3072, 3840]

# Concurrency levels
CONCURRENCY_SINGLE = 1
CONCURRENCY_BATCH = 32


def make_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens length."""
    # Gemma tokenizer: ~3.5-4 chars per token for English prose
    # Use varied text to avoid pathological tokenization
    sentences = [
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
    # Build a long base text by cycling through sentences
    base = ""
    i = 0
    while len(base) < target_tokens * 5:  # overshoot, then truncate
        base += sentences[i % len(sentences)] + " "
        i += 1
    # ~4 chars per token is a safe estimate for Gemma
    char_target = int(target_tokens * 4)
    return base[:char_target]


@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_time_ms: float = 0.0
    decode_tps: float = 0.0
    error: Optional[str] = None


async def get_kv_cache_usage() -> float:
    """Query Prometheus metrics for KV cache utilization."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/metrics") as resp:
            text = await resp.text()
            for line in text.split("\n"):
                if line.startswith("vllm:kv_cache_usage_perc{"):
                    return float(line.split()[-1])
    return -1.0


async def send_request(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = MAX_TOKENS,
    stream: bool = True,
) -> RequestResult:
    """Send a single streaming completion request and measure timing."""
    result = RequestResult()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }

    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:200]}"
                return result

            if stream:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content and t_first_token is None:
                            t_first_token = time.perf_counter()
                        if content:
                            token_count += 1
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
            else:
                body = await resp.json()
                usage = body.get("usage", {})
                result.prompt_tokens = usage.get("prompt_tokens", 0)
                result.completion_tokens = usage.get("completion_tokens", 0)
                t_first_token = time.perf_counter()
                token_count = result.completion_tokens

    except Exception as e:
        result.error = str(e)
        return result

    t_end = time.perf_counter()
    result.total_time_ms = (t_end - t_start) * 1000
    if t_first_token is not None:
        result.ttft_ms = (t_first_token - t_start) * 1000
        decode_time = t_end - t_first_token
        if decode_time > 0 and token_count > 1:
            result.decode_tps = (token_count - 1) / decode_time
        elif token_count == 1:
            result.decode_tps = 0.0
    result.completion_tokens = token_count
    return result


async def bench_single_request(ctx_len: int) -> dict:
    """Single-request latency at a given context length."""
    prompt = make_prompt(ctx_len)
    async with aiohttp.ClientSession() as session:
        result = await send_request(session, prompt)
    kv_usage = await get_kv_cache_usage()
    return {
        "context_len": ctx_len,
        "concurrency": 1,
        "ttft_ms": round(result.ttft_ms, 1),
        "decode_tps": round(result.decode_tps, 1),
        "total_time_ms": round(result.total_time_ms, 1),
        "completion_tokens": result.completion_tokens,
        "kv_cache_pct": round(kv_usage * 100, 2),
        "error": result.error,
    }


async def bench_concurrent(ctx_len: int, concurrency: int) -> dict:
    """Concurrent throughput at a given context length."""
    prompt = make_prompt(ctx_len)

    # Measure KV before
    kv_before = await get_kv_cache_usage()

    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        tasks = [send_request(session, prompt) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        t1 = time.perf_counter()

    # Measure KV during (right after all requests started but some may still be running)
    # We'll get it after all complete
    kv_after = await get_kv_cache_usage()

    errors = [r for r in results if r.error]
    ok = [r for r in results if not r.error]

    total_tokens = sum(r.completion_tokens for r in ok)
    wall_time = t1 - t0
    agg_tps = total_tokens / wall_time if wall_time > 0 else 0

    ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
    p50_ttft = sorted(ttfts)[len(ttfts) // 2] if ttfts else 0
    p99_ttft = sorted(ttfts)[int(len(ttfts) * 0.99)] if ttfts else 0

    decode_rates = [r.decode_tps for r in ok if r.decode_tps > 0]
    avg_per_req_tps = sum(decode_rates) / len(decode_rates) if decode_rates else 0

    return {
        "context_len": ctx_len,
        "concurrency": concurrency,
        "aggregate_tps": round(agg_tps, 1),
        "avg_per_req_tps": round(avg_per_req_tps, 1),
        "avg_ttft_ms": round(avg_ttft, 1),
        "p50_ttft_ms": round(p50_ttft, 1),
        "p99_ttft_ms": round(p99_ttft, 1),
        "wall_time_s": round(wall_time, 2),
        "total_tokens": total_tokens,
        "ok_requests": len(ok),
        "errors": len(errors),
        "kv_cache_pct": round(kv_after * 100, 2),
        "error_msgs": [e.error for e in errors][:3] if errors else [],
    }


async def main():
    print("=" * 80)
    print("CONTEXT SCALING BENCHMARK")
    print(f"Model: {MODEL}")
    print(f"Max generation tokens: {MAX_TOKENS}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print("=" * 80)

    # ── Phase 1: Single-request latency sweep ──
    print("\n" + "─" * 60)
    print("PHASE 1: Single-request latency vs context length")
    print("─" * 60)
    print(f"{'Ctx Len':>8} {'TTFT(ms)':>10} {'Decode(t/s)':>12} {'Total(ms)':>10} {'Tokens':>7} {'KV%':>6}")
    print("-" * 60)

    single_results = []
    for ctx_len in CONTEXT_LENGTHS:
        r = await bench_single_request(ctx_len)
        single_results.append(r)
        if r["error"]:
            print(f"{ctx_len:>8} ERROR: {r['error'][:50]}")
        else:
            print(
                f"{ctx_len:>8} {r['ttft_ms']:>10.1f} {r['decode_tps']:>12.1f} "
                f"{r['total_time_ms']:>10.1f} {r['completion_tokens']:>7} {r['kv_cache_pct']:>5.1f}%"
            )
        # Brief pause between tests to let KV cache clear
        await asyncio.sleep(1.0)

    # ── Phase 2: Fixed concurrency sweep ──
    print("\n" + "─" * 60)
    print(f"PHASE 2: Throughput at C={CONCURRENCY_BATCH} vs context length")
    print("─" * 60)
    print(
        f"{'Ctx Len':>8} {'Agg t/s':>9} {'Per-req t/s':>12} {'TTFT p50':>10} "
        f"{'TTFT p99':>10} {'Wall(s)':>8} {'OK/Err':>8} {'KV%':>6}"
    )
    print("-" * 80)

    batch_results = []
    for ctx_len in CONTEXT_LENGTHS:
        r = await bench_concurrent(ctx_len, CONCURRENCY_BATCH)
        batch_results.append(r)
        if r["errors"] > 0 and r["ok_requests"] == 0:
            print(f"{ctx_len:>8} ALL FAILED: {r['error_msgs'][0][:50] if r['error_msgs'] else 'unknown'}")
        else:
            ok_err = f"{r['ok_requests']}/{r['errors']}"
            print(
                f"{ctx_len:>8} {r['aggregate_tps']:>9.1f} {r['avg_per_req_tps']:>12.1f} "
                f"{r['p50_ttft_ms']:>10.1f} {r['p99_ttft_ms']:>10.1f} "
                f"{r['wall_time_s']:>8.2f} {ok_err:>8} {r['kv_cache_pct']:>5.1f}%"
            )
        # Wait for KV to clear before next batch
        await asyncio.sleep(3.0)

    # ── Phase 3: Progressive concurrency at longest context ──
    print("\n" + "─" * 60)
    print("PHASE 3: Scaling concurrency at ctx=2048 tokens")
    print("─" * 60)
    conc_levels = [1, 2, 4, 8, 16, 32]
    print(f"{'Conc':>6} {'Agg t/s':>9} {'Per-req t/s':>12} {'TTFT p50':>10} {'OK/Err':>8} {'KV%':>6}")
    print("-" * 60)

    conc_results = []
    for c in conc_levels:
        r = await bench_concurrent(2048, c)
        conc_results.append(r)
        ok_err = f"{r['ok_requests']}/{r['errors']}"
        print(
            f"{c:>6} {r['aggregate_tps']:>9.1f} {r['avg_per_req_tps']:>12.1f} "
            f"{r['p50_ttft_ms']:>10.1f} {ok_err:>8} {r['kv_cache_pct']:>5.1f}%"
        )
        await asyncio.sleep(3.0)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find KV pressure point from batch results
    pressure_point = None
    for r in batch_results:
        if r["kv_cache_pct"] > 50:
            pressure_point = r["context_len"]
            break

    # Find throughput degradation
    if len(batch_results) >= 2:
        peak_tps = max(r["aggregate_tps"] for r in batch_results if r["ok_requests"] > 0)
        for r in batch_results:
            if r["ok_requests"] > 0 and r["aggregate_tps"] < peak_tps * 0.8:
                print(f"  Throughput drops >20% at ctx={r['context_len']} ({r['aggregate_tps']:.0f} vs peak {peak_tps:.0f} tok/s)")
                break

    if pressure_point:
        print(f"  KV pressure point (>50% util at C=32): ctx={pressure_point}")
    else:
        print(f"  KV never exceeded 50% at C=32 (max: {max(r['kv_cache_pct'] for r in batch_results):.1f}%)")

    # Prefill scaling
    if len(single_results) >= 2:
        s0 = single_results[0]
        sl = single_results[-1]
        if s0["ttft_ms"] > 0 and not s0["error"] and not sl["error"]:
            ratio = sl["ttft_ms"] / s0["ttft_ms"]
            len_ratio = sl["context_len"] / s0["context_len"]
            print(f"  Prefill scaling: {s0['context_len']}->{ sl['context_len']} tokens = {ratio:.1f}x TTFT ({len_ratio:.0f}x context)")

    # Save raw JSON for further analysis
    output = {
        "single_request": single_results,
        "batch_c32": batch_results,
        "concurrency_sweep": conc_results,
    }
    with open("/root/projects/autokernel/profiling/context_scaling_raw.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Raw data saved to profiling/context_scaling_raw.json")


if __name__ == "__main__":
    asyncio.run(main())
