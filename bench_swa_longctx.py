#!/usr/bin/env python3
"""Long-context e2e throughput bench for the SWA sparse-decode plugin.

Each request's prompt is ~12,000 tokens so that very early in decode the
sequence length crosses ``window * 1.25`` (Gemma4 26B sliding window = 1024
on this model variant, so we exceed it by 10x by the time decode starts).
Sweeps concurrency C in {4, 8, 16, 32} with max_tokens=128.

Runs against an OpenAI-compatible /v1 endpoint. Does NOT edit bench.py (by
design: that file is in the hard-rules no-touch list).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


FILLER_PARAGRAPH = (
    "The sliding-window attention kernel optimizes transformer decoding by only "
    "attending to the most recent W tokens of the context rather than the full "
    "sequence length. At sequence lengths much greater than the window size, "
    "this is a substantial reduction in both HBM bandwidth and FLOPs. The "
    "Triton kernel at kernels/triton/swa_decode.py implements a page-scheduled "
    "sparse variant that skips entire pages that fall outside the window "
    "rather than applying a dense mask after the fact. At seq_len=8192 with "
    "window=4096 the measured speedup is 4.64x on BF16 decode for the Gemma4 "
    "sliding-layer shape. At seq_len=16384 the speedup climbs to 20x. This "
    "file is a filler paragraph used solely to pad the prompt beyond the "
    "sliding window so that the SWA sparse path is exercised end-to-end. "
)


def build_long_prompt(target_tokens: int, suffix: str) -> str:
    """Build a ~target_tokens long prompt by repeating filler. Assumes
    roughly 4 chars = 1 token (conservative for English), so pad to
    ``target_tokens * 4`` chars plus a short task at the end."""
    est_chars = int(target_tokens * 4.0)
    paragraphs = []
    while sum(len(p) for p in paragraphs) < est_chars:
        paragraphs.append(FILLER_PARAGRAPH)
    body = "".join(paragraphs)[:est_chars]
    return f"{body}\n\nQUESTION: {suffix}\n\nANSWER:"


SUFFIXES = [
    "In one short paragraph, summarize the key idea of sliding-window attention.",
    "What is the main bandwidth win of a sparse SWA kernel at long context?",
    "List two reasons page-level truncation beats masked softmax at seq>>window.",
    "Explain briefly why the kernel is gated off when seq_len <= window.",
    "Name the file path of the Triton kernel mentioned in the passage.",
    "State the measured speedup at seq_len=8192 vs FlashInfer dense+mask.",
    "Give one sentence on what 'split-K paged decode' means.",
    "Why does the guard use seq_len > window * 1.25 specifically?",
    "How many sliding layers does Gemma4 26B have out of 30?",
    "What precision do the K/V caches use in this kernel?",
    "Describe in one sentence the role of FlashInfer on global layers.",
    "Give a concise definition of HBM bandwidth.",
    "What does 'cos=0.999999 vs PyTorch FP32' tell us about correctness?",
    "List one reason Triton is chosen over CUDA for this kernel.",
    "Briefly: what is a page in paged KV cache?",
    "Explain the trade-off between global and sliding attention layers.",
]


def get_model(api_base: str) -> str:
    r = requests.get(f"{api_base}/models", timeout=10)
    return r.json()["data"][0]["id"]


def one_request(api_base: str, model: str, prompt: str, max_tokens: int):
    t0 = time.time()
    try:
        r = requests.post(
            f"{api_base}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            timeout=600,
        )
        elapsed = time.time() - t0
        j = r.json()
        if "error" in j:
            return 0, 0, elapsed, j["error"], ""
        pt = j["usage"]["prompt_tokens"]
        ct = j["usage"]["completion_tokens"]
        text = (j["choices"][0].get("text") or "")[:120]
        return pt, ct, elapsed, None, text
    except Exception as e:
        return 0, 0, time.time() - t0, str(e), ""


def sweep_one(api_base: str, model: str, concurrency: int, max_tokens: int,
              prompt_tokens: int):
    # Total requests = 2x concurrency so the run is steady-state, not just a burst.
    num_req = max(concurrency * 2, concurrency)
    prompts = [
        build_long_prompt(prompt_tokens, SUFFIXES[i % len(SUFFIXES)])
        for i in range(num_req)
    ]
    total_pt = 0
    total_ct = 0
    errors = 0
    latencies = []
    sample_texts: list[str] = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(one_request, api_base, model, p, max_tokens) for p in prompts
        ]
        for f in as_completed(futures):
            pt, ct, el, err, txt = f.result()
            if err:
                errors += 1
                if len(sample_texts) < 2:
                    sample_texts.append(f"[ERR]{err}")
            else:
                total_pt += pt
                total_ct += ct
                latencies.append(el)
                if len(sample_texts) < 2:
                    sample_texts.append(txt)
    wall = time.time() - t0

    return {
        "concurrency": concurrency,
        "num_requests": num_req,
        "prompt_tokens_cfg": prompt_tokens,
        "total_prompt_tokens": total_pt,
        "total_gen_tokens": total_ct,
        "wall_s": round(wall, 2),
        "gen_tok_s": round(total_ct / wall, 1) if wall > 0 else 0.0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "errors": errors,
        "samples": sample_texts,
    }


def wait_ready(api_base: str, timeout: int = 300) -> bool:
    url = api_base.replace("/v1", "/health")
    for i in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
        if i % 15 == 0:
            print(f"  waiting for server... ({i}s)", flush=True)
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default="http://localhost:8005/v1")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompt-tokens", type=int, default=12000)
    p.add_argument("--concurrencies", default="4,8,16,32")
    p.add_argument("--output-json", default="bench_swa_longctx.json")
    p.add_argument("--label", default="unknown")
    p.add_argument("--wait-ready", type=int, default=0,
                   help="if >0, wait that many seconds for /health before benching")
    args = p.parse_args()

    if args.wait_ready > 0:
        ok = wait_ready(args.api_base, timeout=args.wait_ready)
        if not ok:
            print(f"Server at {args.api_base} not ready in {args.wait_ready}s; abort.")
            sys.exit(2)

    model = get_model(args.api_base)
    print(f"Label: {args.label}")
    print(f"Model: {model}")
    print(f"Prompt tokens (target): {args.prompt_tokens}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Quick warmup (also helps torch.compile / cuda graphs settle).
    print("Warmup (C=2)...")
    warm = sweep_one(
        args.api_base, model, concurrency=2, max_tokens=32,
        prompt_tokens=args.prompt_tokens,
    )
    print(f"  warm gen_tok_s={warm['gen_tok_s']} err={warm['errors']}")

    concs = [int(c) for c in args.concurrencies.split(",") if c]
    print(f"\nSweep concurrencies: {concs}")
    print(f"{'C':>4} {'req':>5} {'pt':>10} {'gt':>8} {'wall':>7} {'gen tok/s':>10} "
          f"{'avg lat':>8} {'errs':>5}")
    print("-" * 70)

    results = []
    for c in concs:
        r = sweep_one(
            args.api_base, model, concurrency=c, max_tokens=args.max_tokens,
            prompt_tokens=args.prompt_tokens,
        )
        results.append(r)
        print(f"{r['concurrency']:>4} {r['num_requests']:>5} "
              f"{r['total_prompt_tokens']:>10} {r['total_gen_tokens']:>8} "
              f"{r['wall_s']:>7} {r['gen_tok_s']:>10} "
              f"{r['avg_latency_s']:>8} {r['errors']:>5}")

    out = {
        "label": args.label,
        "model": model,
        "prompt_tokens_cfg": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "results": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results -> {args.output_json}")

    # Print one sample per C so we can eyeball coherence.
    print("\nSample generations (first chars):")
    for r in results:
        for s in r["samples"][:1]:
            print(f"  C={r['concurrency']}: {s!r}")


if __name__ == "__main__":
    main()
