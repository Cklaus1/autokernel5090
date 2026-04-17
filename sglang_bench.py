#!/usr/bin/env python3
"""
SGLang Benchmark Harness — autokernel style
One focused change per experiment. Measures decode tok/s and batch throughput.
"""

import requests
import time
import sys
import json
import os
import subprocess
import signal

def wait_for_server(port, timeout=600):
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False

def generate(port, prompt, max_tokens=256, temperature=0.0):
    resp = requests.post(f"http://127.0.0.1:{port}/generate", json={
        "text": prompt,
        "sampling_params": {"temperature": temperature, "max_new_tokens": max_tokens}
    }, timeout=180)
    resp.raise_for_status()
    return resp.json()

def generate_batch(port, prompts, max_tokens=256, temperature=0.0):
    resp = requests.post(f"http://127.0.0.1:{port}/generate", json={
        "text": prompts,
        "sampling_params": {"temperature": temperature, "max_new_tokens": max_tokens}
    }, timeout=600)
    resp.raise_for_status()
    return resp.json()

def flush_cache(port):
    try:
        requests.get(f"http://127.0.0.1:{port}/flush_cache", timeout=30)
    except:
        pass

CODING_PROMPT = "<|im_start|>user\nWrite a Python binary search tree with insert, delete, search, and balancing.<|im_end|>\n<|im_start|>assistant\n"
MATH_PROMPT = "<|im_start|>user\nSolve: Find all positive integers n where n^2+2n+2 is divisible by n+1.<|im_end|>\n<|im_start|>assistant\n"
REASON_PROMPT = "<|im_start|>user\nA farmer has a fox, chicken, and grain. He crosses a river in a boat carrying one item. Fox eats chicken, chicken eats grain if left alone. How does he get everything across?<|im_end|>\n<|im_start|>assistant\n"
PROMPTS = [CODING_PROMPT, MATH_PROMPT, REASON_PROMPT] * 12  # 36 for batching

def bench_decode(port, warmup=3, runs=5, max_tokens=256):
    """Single-request decode throughput."""
    for _ in range(warmup):
        generate(port, CODING_PROMPT, max_tokens=128)
    flush_cache(port)
    time.sleep(1)

    results = []
    for i in range(runs):
        t0 = time.perf_counter()
        out = generate(port, CODING_PROMPT, max_tokens=max_tokens)
        elapsed = time.perf_counter() - t0
        meta = out.get("meta_info", {})
        tokens = meta.get("completion_tokens", max_tokens)
        tok_s = tokens / elapsed
        accept = meta.get("spec_accept_length", None)
        results.append({"tok_s": tok_s, "tokens": tokens, "elapsed": elapsed, "accept": accept})

    avg = sum(r["tok_s"] for r in results) / len(results)
    accepts = [r["accept"] for r in results if r["accept"] is not None]
    avg_accept = sum(accepts) / len(accepts) if accepts else None
    return avg, avg_accept, results

def bench_batch(port, batch_sizes=[1, 4, 8, 16, 32], max_tokens=256, warmup=1):
    """Batch throughput at various sizes."""
    results = {}
    for bs in batch_sizes:
        batch = PROMPTS[:bs]
        # Warmup
        for _ in range(warmup):
            flush_cache(port)
            time.sleep(0.3)
            try:
                generate_batch(port, batch, max_tokens=max_tokens)
            except:
                results[bs] = {"total_tok_s": 0, "per_user": 0, "accept": None}
                continue

        flush_cache(port)
        time.sleep(0.5)
        t0 = time.perf_counter()
        try:
            outs = generate_batch(port, batch, max_tokens=max_tokens)
        except Exception as e:
            print(f"  batch={bs}: FAILED ({e})")
            results[bs] = {"total_tok_s": 0, "per_user": 0, "accept": None}
            continue
        elapsed = time.perf_counter() - t0
        total_tokens = sum(o.get("meta_info", {}).get("completion_tokens", max_tokens) for o in outs)
        tok_s = total_tokens / elapsed

        accepts = []
        for o in outs:
            a = o.get("meta_info", {}).get("spec_accept_length")
            if a is not None:
                accepts.append(a)
        avg_accept = sum(accepts) / len(accepts) if accepts else None

        results[bs] = {"total_tok_s": tok_s, "per_user": tok_s / bs, "accept": avg_accept}

    return results

def launch_server(port, args_list, timeout=600):
    """Launch SGLang server, return (proc, success)."""
    sglang_python = os.environ.get("SGLANG_PYTHON", "/root/sglang_env/bin/python")
    cmd = [sglang_python, "-m", "sglang.launch_server", "--port", str(port)] + args_list

    log_path = f"/tmp/sglang_exp_{port}.log"
    env = os.environ.copy()

    proc = subprocess.Popen(cmd, env=env, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                           preexec_fn=os.setsid)

    ready = wait_for_server(port, timeout=timeout)
    if not ready:
        # Print last 30 lines of log for debugging
        try:
            with open(log_path) as f:
                lines = f.readlines()
                for line in lines[-30:]:
                    print(f"  LOG: {line.rstrip()}")
        except:
            pass

    return proc, ready, log_path

def kill_server(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except:
        try:
            proc.terminate()
        except:
            pass
    try:
        proc.wait(timeout=15)
    except:
        try:
            proc.kill()
        except:
            pass

def log_result(exp_num, tag, decode_tok_s, batch32_tok_s, accept_len, status, desc,
               log_file="sglang_results.tsv"):
    """Log experiment result in autokernel format."""
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("experiment\ttag\tdecode_tok_s\tbatch32_tok_s\taccept_length\tstatus\tdescription\n")

    with open(log_file, "a") as f:
        accept_str = f"{accept_len:.2f}" if accept_len else "N/A"
        f.write(f"{exp_num}\t{tag}\t{decode_tok_s:.1f}\t{batch32_tok_s:.0f}\t{accept_str}\t{status}\t{desc}\n")

    print(f"\n{'='*60}")
    print(f"EXP {exp_num} [{tag}]: {status}")
    print(f"  Decode: {decode_tok_s:.1f} tok/s | Batch32: {batch32_tok_s:.0f} tok/s | Accept: {accept_str}")
    print(f"  {desc}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Quick self-test
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
    print(f"Benchmarking server on port {port}...")

    print("\nDecode benchmark:")
    decode, accept, _ = bench_decode(port)
    print(f"  Average: {decode:.1f} tok/s" + (f", accept: {accept:.2f}" if accept else ""))

    print("\nBatch benchmark:")
    batch = bench_batch(port)
    for bs, r in sorted(batch.items()):
        print(f"  batch={bs}: {r['total_tok_s']:.0f} tok/s ({r['per_user']:.1f}/user)")
