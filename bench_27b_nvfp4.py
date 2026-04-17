#!/usr/bin/env python3
"""Benchmark decode and batch performance for the 27B NVFP4 model on vLLM."""

import time
import json
import openai
import concurrent.futures

BASE_URL = "http://localhost:8000/v1"
MODEL = "mconcat/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4"

client = openai.OpenAI(base_url=BASE_URL, api_key="dummy")

def benchmark_decode(max_tokens=128, prompt="Write a detailed explanation of how GPU tensor cores work.", warmup=2, runs=3):
    """Single-request decode throughput (tokens/sec)."""
    print(f"\n{'='*60}")
    print(f"DECODE BENCHMARK (single request, {max_tokens} tokens)")
    print(f"{'='*60}")

    # Warmup
    for i in range(warmup):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=32,
        )

    results = []
    for i in range(runs):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - t0

        output_tokens = resp.usage.completion_tokens
        tps = output_tokens / elapsed
        results.append(tps)
        print(f"  Run {i+1}: {output_tokens} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")

    avg = sum(results) / len(results)
    print(f"  Average: {avg:.1f} tok/s")
    return avg


def benchmark_batch(batch_size, max_tokens=64, prompt="Explain quicksort in one paragraph."):
    """Concurrent batch throughput."""
    def single_request(_):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - t0
        return resp.usage.completion_tokens, elapsed

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(single_request, i) for i in range(batch_size)]
        results = [f.result() for f in futures]
    t_total = time.perf_counter() - t_start

    total_tokens = sum(r[0] for r in results)
    throughput = total_tokens / t_total
    per_user = throughput / batch_size

    return throughput, per_user, t_total, total_tokens


def benchmark_batch_sweep():
    """Sweep batch sizes to find peak throughput."""
    print(f"\n{'='*60}")
    print(f"BATCH THROUGHPUT SWEEP")
    print(f"{'='*60}")

    # Warmup
    benchmark_batch(1, max_tokens=16)

    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []

    for bs in batch_sizes:
        try:
            throughput, per_user, elapsed, total_tok = benchmark_batch(bs, max_tokens=64)
            print(f"  batch={bs:3d}: {throughput:7.1f} tok/s total, {per_user:6.1f} tok/s/user, {elapsed:.1f}s, {total_tok} tokens")
            results.append((bs, throughput, per_user))
        except Exception as e:
            print(f"  batch={bs:3d}: FAILED - {e}")
            break

    if results:
        best = max(results, key=lambda x: x[1])
        print(f"\n  Peak: batch={best[0]} → {best[1]:.0f} tok/s total")

    return results


def benchmark_prefill(prompt_lengths=[100, 500, 1000, 2000]):
    """Prefill (time-to-first-token) benchmark."""
    print(f"\n{'='*60}")
    print(f"PREFILL BENCHMARK (time to first token)")
    print(f"{'='*60}")

    base_text = "The quick brown fox jumps over the lazy dog. " * 50  # ~500 tokens worth

    for target_len in prompt_lengths:
        # Approximate token count (rough: 1 token ≈ 4 chars for English)
        prompt = base_text * max(1, target_len // 50)
        prompt = prompt[:target_len * 4]  # Rough trim

        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": f"Summarize this text in one sentence: {prompt}"}],
            max_tokens=1,  # Just measure TTFT
            stream=True,
        )
        # Get first chunk
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                ttft = time.perf_counter() - t0
                break
        else:
            ttft = time.perf_counter() - t0

        prefill_tps = target_len / ttft if ttft > 0 else 0
        print(f"  ~{target_len:4d} tokens: TTFT={ttft*1000:.0f}ms, prefill≈{prefill_tps:.0f} tok/s")


def benchmark_tool_calling_overhead():
    """Measure overhead of tool calling vs plain generation."""
    print(f"\n{'='*60}")
    print(f"TOOL CALLING OVERHEAD")
    print(f"{'='*60}")

    tools = [{
        "type": "function",
        "function": {
            "name": "get_data",
            "description": "Get data from database",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }]

    prompt = "What is the total revenue for Q4 2024?"

    # Without tools
    times_plain = []
    for _ in range(3):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
        )
        times_plain.append(time.perf_counter() - t0)

    # With tools
    times_tools = []
    for _ in range(3):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
            max_tokens=64,
        )
        times_tools.append(time.perf_counter() - t0)

    avg_plain = sum(times_plain) / len(times_plain)
    avg_tools = sum(times_tools) / len(times_tools)
    overhead = ((avg_tools - avg_plain) / avg_plain) * 100

    print(f"  Plain generation: {avg_plain*1000:.0f}ms avg")
    print(f"  With tools:       {avg_tools*1000:.0f}ms avg")
    print(f"  Overhead:         {overhead:+.1f}%")


if __name__ == "__main__":
    print(f"Model: {MODEL}")
    print(f"Endpoint: {BASE_URL}")

    # 1. Decode benchmark
    decode_tps = benchmark_decode()

    # 2. Batch sweep
    batch_results = benchmark_batch_sweep()

    # 3. Prefill benchmark
    benchmark_prefill()

    # 4. Tool calling overhead
    benchmark_tool_calling_overhead()

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Decode:     {decode_tps:.1f} tok/s")
    if batch_results:
        best_batch = max(batch_results, key=lambda x: x[1])
        print(f"  Peak batch: {best_batch[1]:.0f} tok/s @ batch={best_batch[0]}")
    print(f"{'='*60}")
