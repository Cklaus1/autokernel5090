#!/usr/bin/env python3
"""
Experiment 101: DFlash speculative decoding on SGLang vs baseline
Target: Qwen3.5-9B, Draft: z-lab/Qwen3.5-9B-DFlash (5 layers, block_size=16)
Compare against MTP3 best: 167 tok/s decode, batch>1 crashes

Test matrix:
  1) Baseline (no spec decode) - single request decode
  2) DFlash - single request decode
  3) DFlash - batch sweep (1, 4, 8, 16, 32)
"""

import subprocess
import sys
import os
import time
import json
import requests
import signal

SGLANG_PYTHON = "/root/sglang_env/bin/python"
TARGET_MODEL = "Qwen/Qwen3.5-9B"
DRAFT_MODEL = "z-lab/Qwen3.5-9B-DFlash"
PORT_BASELINE = 30000
PORT_DFLASH = 30001
MEM_FRACTION = 0.85  # conservative for 32GB

# Nvidia lib paths needed for venv
NVIDIA_LIBS = subprocess.check_output(
    [sys.executable, "-c",
     "import nvidia, os; base=os.path.dirname(nvidia.__file__); "
     "libs=[os.path.join(base,d,'lib') for d in os.listdir(base) "
     "if os.path.isdir(os.path.join(base,d,'lib'))]; print(':'.join(libs))"],
    text=True
).strip()

ENV = os.environ.copy()
ENV["LD_LIBRARY_PATH"] = NVIDIA_LIBS + ":" + ENV.get("LD_LIBRARY_PATH", "")
ENV["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"


def wait_for_server(port, timeout=300):
    """Wait for SGLang server to be ready."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print(f"  Server ready on port {port} ({time.time()-start:.0f}s)")
                return True
        except:
            pass
        time.sleep(2)
    print(f"  TIMEOUT waiting for server on port {port}")
    return False


def generate(port, prompt, max_tokens=512, temperature=0.0):
    """Send a single generate request."""
    url = f"http://127.0.0.1:{port}/generate"
    resp = requests.post(url, json={
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
        }
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()


def generate_batch(port, prompts, max_tokens=512, temperature=0.0):
    """Send batched generate request."""
    url = f"http://127.0.0.1:{port}/generate"
    resp = requests.post(url, json={
        "text": prompts,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
        }
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()


def flush_cache(port):
    requests.get(f"http://127.0.0.1:{port}/flush_cache", timeout=30)


def benchmark_decode(port, prompt, max_tokens=256, warmup=2, runs=5):
    """Benchmark single-request decode throughput."""
    # Warmup
    for _ in range(warmup):
        generate(port, prompt, max_tokens=max_tokens)

    flush_cache(port)

    import torch
    results = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = generate(port, prompt, max_tokens=max_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        meta = out.get("meta_info", {})
        completion_tokens = meta.get("completion_tokens", max_tokens)
        tok_s = completion_tokens / elapsed

        spec_accept = meta.get("spec_accept_length", None)
        spec_verify = meta.get("spec_verify_ct", 0)

        results.append({
            "tokens": completion_tokens,
            "elapsed": elapsed,
            "tok_s": tok_s,
            "spec_accept_length": spec_accept,
            "spec_verify_ct": spec_verify,
        })
        print(f"    Run {i+1}: {tok_s:.1f} tok/s ({completion_tokens} tokens in {elapsed:.2f}s)"
              + (f" accept_len={spec_accept:.2f}" if spec_accept else ""))

    avg_tok_s = sum(r["tok_s"] for r in results) / len(results)
    return avg_tok_s, results


def benchmark_batch(port, prompts_pool, batch_sizes, max_tokens=256, warmup=1):
    """Benchmark batch throughput at various batch sizes."""
    results = {}
    for bs in batch_sizes:
        prompts = prompts_pool[:bs]

        # Warmup
        for _ in range(warmup):
            flush_cache(port)
            generate_batch(port, prompts, max_tokens=max_tokens)

        flush_cache(port)
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs = generate_batch(port, prompts, max_tokens=max_tokens)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        total_tokens = sum(o.get("meta_info", {}).get("completion_tokens", max_tokens) for o in outs)
        tok_s = total_tokens / elapsed
        per_user = tok_s / bs

        # Get acceptance length if available
        accept_lens = []
        for o in outs:
            al = o.get("meta_info", {}).get("spec_accept_length", None)
            if al is not None:
                accept_lens.append(al)
        avg_accept = sum(accept_lens) / len(accept_lens) if accept_lens else None

        results[bs] = {"total_tok_s": tok_s, "per_user_tok_s": per_user,
                        "elapsed": elapsed, "accept_length": avg_accept}
        print(f"    batch={bs}: {tok_s:.0f} total tok/s ({per_user:.1f}/user, {elapsed:.1f}s)"
              + (f" accept_len={avg_accept:.2f}" if avg_accept else ""))

    return results


def launch_server(port, spec_decode=False):
    """Launch SGLang server."""
    cmd = [
        SGLANG_PYTHON, "-m", "sglang.launch_server",
        "--model-path", TARGET_MODEL,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--mem-fraction-static", str(MEM_FRACTION),
        "--trust-remote-code",
    ]

    if spec_decode:
        cmd += [
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", DRAFT_MODEL,
        ]

    # Try Blackwell attention backends
    # SM120: fa4 and trtllm_mha available
    cmd += ["--attention-backend", "trtllm_mha"]
    if spec_decode:
        cmd += ["--speculative-draft-attention-backend", "fa4"]

    print(f"  Launching: {' '.join(cmd[-10:])}")
    proc = subprocess.Popen(cmd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def kill_server(proc):
    """Kill server process tree."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except:
        proc.kill()


# ── Test prompts ──
CODING_PROMPT = """<|im_start|>user
Write a Python function that implements a binary search tree with insert, delete, and search operations. Include proper balancing.<|im_end|>
<|im_start|>assistant
"""

MATH_PROMPT = """<|im_start|>user
Solve step by step: Find all positive integers n such that n^2 + 2n + 2 is divisible by n + 1.<|im_end|>
<|im_start|>assistant
"""

REASONING_PROMPT = """<|im_start|>user
A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a boat that can only carry him and one item at a time. If left alone, the fox will eat the chicken, and the chicken will eat the grain. How can he get everything across safely? Explain your reasoning step by step.<|im_end|>
<|im_start|>assistant
"""

PROMPTS_POOL = [CODING_PROMPT, MATH_PROMPT, REASONING_PROMPT] * 12  # 36 prompts for batching


if __name__ == "__main__":
    print("=" * 70)
    print("EXP 101: DFlash Speculative Decoding on SGLang")
    print(f"Target: {TARGET_MODEL}, Draft: {DRAFT_MODEL}")
    print(f"GPU: RTX 5090 32GB, SM120")
    print(f"Compare vs: MTP3 best=167 tok/s, vLLM baseline=122 tok/s")
    print("=" * 70)

    # ── Phase 1: Baseline (no spec decode) ──
    print("\n[Phase 1] Baseline SGLang (no speculation)")
    proc = launch_server(PORT_BASELINE, spec_decode=False)
    try:
        if wait_for_server(PORT_BASELINE):
            print("  Single-request decode:")
            baseline_decode, _ = benchmark_decode(PORT_BASELINE, CODING_PROMPT, max_tokens=512)

            print("  Batch throughput:")
            baseline_batch = benchmark_batch(PORT_BASELINE, PROMPTS_POOL,
                                             batch_sizes=[1, 4, 8, 16, 32], max_tokens=256)
        else:
            baseline_decode = 0
            baseline_batch = {}
    finally:
        kill_server(proc)
        time.sleep(5)

    # ── Phase 2: DFlash ──
    print("\n[Phase 2] DFlash Speculative Decoding")
    proc = launch_server(PORT_DFLASH, spec_decode=True)
    try:
        if wait_for_server(PORT_DFLASH, timeout=400):
            print("  Single-request decode:")
            dflash_decode, dflash_results = benchmark_decode(PORT_DFLASH, CODING_PROMPT, max_tokens=512)

            print("  Batch throughput:")
            dflash_batch = benchmark_batch(PORT_DFLASH, PROMPTS_POOL,
                                           batch_sizes=[1, 4, 8, 16, 32], max_tokens=256)
        else:
            dflash_decode = 0
            dflash_batch = {}
    finally:
        kill_server(proc)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nSingle-request decode:")
    print(f"  SGLang baseline:   {baseline_decode:.1f} tok/s")
    print(f"  SGLang + DFlash:   {dflash_decode:.1f} tok/s")
    if baseline_decode > 0:
        print(f"  DFlash speedup:    {dflash_decode/baseline_decode:.2f}x")
    print(f"  vLLM baseline:     122.2 tok/s (exp 76)")
    print(f"  vLLM MTP3:         167.3 tok/s (exp 97)")
    if dflash_decode > 0:
        print(f"  vs vLLM baseline:  {dflash_decode/122.2:.2f}x")
        print(f"  vs vLLM MTP3:      {dflash_decode/167.3:.2f}x")

    if dflash_results:
        accept_lens = [r["spec_accept_length"] for r in dflash_results if r.get("spec_accept_length")]
        if accept_lens:
            print(f"  Avg acceptance:    {sum(accept_lens)/len(accept_lens):.2f} tokens (block_size=16)")

    print(f"\nBatch throughput (total tok/s):")
    print(f"  {'BS':>4} | {'Baseline':>10} | {'DFlash':>10} | {'Speedup':>8} | {'Accept':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    for bs in [1, 4, 8, 16, 32]:
        bl = baseline_batch.get(bs, {}).get("total_tok_s", 0)
        df = dflash_batch.get(bs, {}).get("total_tok_s", 0)
        al = dflash_batch.get(bs, {}).get("accept_length", None)
        speedup = df / bl if bl > 0 else 0
        al_str = f"{al:.2f}" if al else "N/A"
        print(f"  {bs:>4} | {bl:>10.0f} | {df:>10.0f} | {speedup:>7.2f}x | {al_str:>8}")

    print(f"\nKey comparison (MTP crashed at batch>1):")
    for bs in [4, 8, 16, 32]:
        df = dflash_batch.get(bs, {}).get("total_tok_s", 0)
        if df > 0:
            print(f"  DFlash batch={bs}: {df:.0f} tok/s (MTP: CRASH)")
