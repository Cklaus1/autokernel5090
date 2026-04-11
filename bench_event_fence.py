#!/usr/bin/env python3
"""Benchmark FusenCache event-fence fix: async scheduling without crashes.

Tests that the CUDA event-based stream fence in backend.py correctly
prevents the async scheduling race condition that caused crashes at C=16+.

Three test modes:
  1. Latency: Measure per-layer forward() overhead of event fence vs sync
  2. Throughput: vLLM serving benchmark at various concurrency levels
  3. Stability: Long-running stress test at high concurrency

Usage:
    # Inside the vLLM docker container:
    docker run --rm --gpus all -v /root/projects/autokernel:/ak vllm-built \
        python3 /ak/bench_event_fence.py --mode latency

    # Or launch server + benchmark:
    docker run --rm --gpus all -v /root/projects/autokernel:/ak vllm-built \
        bash /ak/fusen_kv/launch_k4v4b64.sh &
    python3 bench_event_fence.py --mode throughput --port 8001
"""

import argparse
import json
import subprocess
import sys
import time


def bench_latency():
    """Measure overhead of event fence vs full synchronize.

    Creates a mock FusenKV forward() scenario and times:
    - Event record + wait: ~5us expected
    - Full stream synchronize: ~100us expected
    """
    import torch

    device = torch.device("cuda")
    N_ITERS = 1000
    WARMUP = 100

    # Simulate decode kernel workload (small matmul)
    q = torch.randn(32, 16, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(32, 16, 256, device=device, dtype=torch.bfloat16)

    # --- Baseline: no fence ---
    for _ in range(WARMUP):
        _ = torch.bmm(q, k.transpose(-1, -2))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = torch.bmm(q, k.transpose(-1, -2))
    torch.cuda.synchronize()
    t_nofence = (time.perf_counter() - t0) / N_ITERS * 1e6
    print(f"No fence:          {t_nofence:.1f} us/iter")

    # --- Full synchronize ---
    for _ in range(WARMUP):
        _ = torch.bmm(q, k.transpose(-1, -2))
        torch.cuda.current_stream().synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = torch.bmm(q, k.transpose(-1, -2))
        torch.cuda.current_stream().synchronize()
    torch.cuda.synchronize()
    t_sync = (time.perf_counter() - t0) / N_ITERS * 1e6
    print(f"Full synchronize:  {t_sync:.1f} us/iter  (+{t_sync - t_nofence:.1f} us overhead)")

    # --- Event fence (our approach) ---
    event = torch.cuda.Event()
    event.record()

    for _ in range(WARMUP):
        torch.cuda.current_stream().wait_event(event)
        _ = torch.bmm(q, k.transpose(-1, -2))
        event.record()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        torch.cuda.current_stream().wait_event(event)
        _ = torch.bmm(q, k.transpose(-1, -2))
        event.record()
    torch.cuda.synchronize()
    t_event = (time.perf_counter() - t0) / N_ITERS * 1e6
    print(f"Event fence:       {t_event:.1f} us/iter  (+{t_event - t_nofence:.1f} us overhead)")

    print()
    print(f"Event fence overhead:      {t_event - t_nofence:.1f} us")
    print(f"Synchronize overhead:      {t_sync - t_nofence:.1f} us")
    print(f"Savings per layer:         {(t_sync - t_event):.1f} us")
    print(f"Savings per step (30 layers): {(t_sync - t_event) * 30:.0f} us")

    # --- Simulate 30 layers per step ---
    print("\n--- 30-layer step simulation ---")

    # Sync approach
    for _ in range(WARMUP):
        for _ in range(30):
            _ = torch.bmm(q, k.transpose(-1, -2))
            torch.cuda.current_stream().synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS // 10):
        for _ in range(30):
            _ = torch.bmm(q, k.transpose(-1, -2))
            torch.cuda.current_stream().synchronize()
    torch.cuda.synchronize()
    t_sync_step = (time.perf_counter() - t0) / (N_ITERS // 10) * 1e3
    print(f"Sync approach:  {t_sync_step:.2f} ms/step")

    # Event approach
    events = [torch.cuda.Event() for _ in range(30)]
    for e in events:
        e.record()

    for _ in range(WARMUP):
        for i in range(30):
            torch.cuda.current_stream().wait_event(events[i])
            _ = torch.bmm(q, k.transpose(-1, -2))
            events[i].record()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS // 10):
        for i in range(30):
            torch.cuda.current_stream().wait_event(events[i])
            _ = torch.bmm(q, k.transpose(-1, -2))
            events[i].record()
    torch.cuda.synchronize()
    t_event_step = (time.perf_counter() - t0) / (N_ITERS // 10) * 1e3
    print(f"Event approach: {t_event_step:.2f} ms/step")
    print(f"Step savings:   {t_sync_step - t_event_step:.2f} ms/step")
    print(f"Throughput gain: {(t_sync_step / t_event_step - 1) * 100:.1f}%")


def bench_throughput(port=8001, concurrency_levels=None):
    """Run vLLM serving benchmark at various concurrency levels.

    Requires vLLM server running with FusenKV + async scheduling enabled.
    """
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16, 32, 64, 128, 256]

    results = []
    for c in concurrency_levels:
        print(f"\n--- Concurrency C={c} ---")
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            # Actually, we use the benchmark client
        ]
        # Use vLLM's benchmark_serving.py
        bench_cmd = [
            "python3", "-m", "vllm.benchmarks.benchmark_serving",
            "--backend", "openai-chat",
            "--base-url", f"http://localhost:{port}",
            "--model", "gemma-4-26B-A4B-it-NVFP4-modelopt",
            "--dataset-name", "sharegpt",
            "--dataset-path", "/data/ShareGPT_V3_unfiltered_cleaned_split.json",
            "--num-prompts", str(min(c * 4, 256)),
            "--request-rate", str(c),
            "--max-concurrency", str(c),
            "--seed", "42",
        ]
        print(f"Running: {' '.join(bench_cmd)}")
        try:
            result = subprocess.run(
                bench_cmd, capture_output=True, text=True, timeout=300
            )
            print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            if result.returncode != 0:
                print(f"FAILED (exit code {result.returncode})")
                print(result.stderr[-500:])
                results.append({"concurrency": c, "status": "CRASH"})
            else:
                # Parse output throughput
                for line in result.stdout.split("\n"):
                    if "output throughput" in line.lower() or "tok/s" in line.lower():
                        print(f"  >> {line.strip()}")
                results.append({"concurrency": c, "status": "OK", "output": result.stdout})
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT at C={c}")
            results.append({"concurrency": c, "status": "TIMEOUT"})

    print("\n\n=== SUMMARY ===")
    for r in results:
        print(f"  C={r['concurrency']:>3}: {r['status']}")

    return results


def bench_stability(port=8001, duration_min=5, concurrency=64):
    """Long-running stress test at high concurrency."""
    print(f"Stress test: C={concurrency} for {duration_min} minutes")
    bench_cmd = [
        "python3", "-m", "vllm.benchmarks.benchmark_serving",
        "--backend", "openai-chat",
        "--base-url", f"http://localhost:{port}",
        "--model", "gemma-4-26B-A4B-it-NVFP4-modelopt",
        "--dataset-name", "sharegpt",
        "--dataset-path", "/data/ShareGPT_V3_unfiltered_cleaned_split.json",
        "--num-prompts", str(concurrency * duration_min * 2),
        "--request-rate", str(concurrency // 2),
        "--max-concurrency", str(concurrency),
        "--seed", "42",
    ]
    print(f"Running: {' '.join(bench_cmd)}")
    result = subprocess.run(bench_cmd, capture_output=True, text=True,
                           timeout=duration_min * 60 + 120)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"CRASH after some time. Exit code: {result.returncode}")
        print(result.stderr[-500:])
    else:
        print("STABLE - no crashes!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["latency", "throughput", "stability"],
                       default="latency")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--duration", type=int, default=5, help="Minutes for stability test")
    args = parser.parse_args()

    if args.mode == "latency":
        bench_latency()
    elif args.mode == "throughput":
        bench_throughput(port=args.port)
    elif args.mode == "stability":
        bench_stability(port=args.port, duration_min=args.duration,
                       concurrency=args.concurrency)
