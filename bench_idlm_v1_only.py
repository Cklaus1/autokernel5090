#!/usr/bin/env python3
"""Measure I-DLM server throughput + acceptance at configurable concurrency.

Sends 20 diverse prompts to one SGLang I-DLM server, reports mean tok/s and
acceptance rate (if exposed in the API response).

Usage:
    python bench_idlm_v1_only.py --url http://localhost:30001 --concurrency 1
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

PROMPTS = [
    ("simple", "Hello! How are you?"),
    ("simple", "What is 2 + 2?"),
    ("simple", "List 5 colors of the rainbow."),
    ("simple", "What is the capital of France?"),
    ("simple", "Name three fruits."),
    ("medium", "Write a Python function that sorts a list using merge sort."),
    ("medium", "Explain recursion with a short example."),
    ("medium", "What is the difference between TCP and UDP?"),
    ("medium", "Describe how a hash table works."),
    ("medium", "Explain the CAP theorem in distributed systems."),
    ("hard", "Design a distributed cache with TTL eviction and consistent hashing."),
    ("hard", "Debug this race condition: two threads increment a shared counter without a lock. Explain the fix."),
    ("hard", "Compare B-trees vs LSM trees for a write-heavy workload and justify a recommendation."),
    ("hard", "Explain how RLHF works and its limitations in aligning LLMs."),
    ("hard", "Walk through the transformer attention mechanism mathematically."),
    ("long", "Write a 500-word essay about the societal impact of artificial intelligence."),
    ("long", "Implement a full REST API in Python with FastAPI: CRUD for a 'Task' resource with SQLite."),
    ("long", "Write a detailed tutorial on setting up a Kubernetes cluster from scratch, including networking."),
    ("long", "Compose a short story (400+ words) about a robot learning empathy."),
    ("long", "Explain quantum computing from first principles: qubits, superposition, entanglement, and gates."),
]

assert len(PROMPTS) == 20


@dataclass
class Result:
    idx: int
    tokens: int
    elapsed: float
    tok_s: float
    acceptance: Optional[float]
    error: Optional[str]


async def one(session, url: str, idx: int, prompt: str, max_tokens: int = 512) -> Result:
    gen_url = url.rstrip("/") + "/generate"
    # SGLang /generate endpoint supports acceptance rate reporting via meta_info
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": 0.0,
        },
    }
    t0 = time.monotonic()
    try:
        async with session.post(gen_url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            data = await resp.json(content_type=None)
            elapsed = time.monotonic() - t0
            if resp.status != 200:
                return Result(idx, 0, elapsed, 0.0, None, f"HTTP {resp.status}: {str(data)[:80]}")
            meta = data.get("meta_info", {})
            tokens = meta.get("completion_tokens") or meta.get("output_tokens") or 0
            tok_s = tokens / elapsed if elapsed > 0 else 0.0
            accept = (
                meta.get("acceptance_rate")
                or meta.get("spec_accept_rate")
                or meta.get("dllm_accept_rate")
            )
            return Result(idx, tokens, elapsed, tok_s, accept, None)
    except Exception as e:
        return Result(idx, 0, time.monotonic() - t0, 0.0, None, str(e)[:120])


async def run(url: str, concurrency: int) -> list[Result]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(session, idx, prompt):
        async with sem:
            return await one(session, url, idx, prompt)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded(session, i, p) for i, (_, p) in enumerate(PROMPTS)]
        return await asyncio.gather(*tasks)


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:30001")
    ap.add_argument("--concurrency", "-c", type=int, default=1)
    ap.add_argument("--json-out", default=None, help="Write results to this JSON path")
    args = ap.parse_args()

    print(f"URL:         {args.url}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Prompts:     {len(PROMPTS)}")
    print()

    t0 = time.monotonic()
    results = await run(args.url, args.concurrency)
    wall = time.monotonic() - t0

    ok = [r for r in results if r.error is None]
    err = [r for r in results if r.error is not None]

    total_tokens = sum(r.tokens for r in ok)
    mean_toks = (sum(r.tok_s for r in ok) / len(ok)) if ok else 0.0
    server_tok_s = total_tokens / wall if wall > 0 else 0.0
    accepts = [r.acceptance for r in ok if r.acceptance is not None]
    mean_accept = sum(accepts) / len(accepts) if accepts else None

    print(f"{'#':>3}  {'tokens':>7}  {'elapsed':>8}  {'tok/s':>8}  {'accept':>8}  err")
    print("-" * 55)
    for r in results:
        acc = f"{r.acceptance:.2%}" if r.acceptance is not None else "  n/a  "
        err_str = r.error if r.error else ""
        print(f"{r.idx:>3}  {r.tokens:>7}  {r.elapsed:>8.2f}  {r.tok_s:>8.1f}  {acc:>8}  {err_str}")

    print()
    print(f"Wall: {wall:.2f}s")
    print(f"Successful requests: {len(ok)} / {len(results)}")
    print(f"Total completion tokens: {total_tokens}")
    print(f"Mean per-request tok/s: {mean_toks:.1f}")
    print(f"Server aggregate tok/s: {server_tok_s:.1f}")
    if mean_accept is not None:
        print(f"Mean acceptance rate:   {mean_accept:.2%}")
    else:
        print(f"Mean acceptance rate:   n/a (not exposed)")
    if err:
        print()
        print(f"Errors ({len(err)}):")
        for r in err:
            print(f"  [{r.idx}] {r.error}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "url": args.url,
                "concurrency": args.concurrency,
                "wall": wall,
                "successful": len(ok),
                "total_tokens": total_tokens,
                "mean_per_req_tok_s": mean_toks,
                "server_agg_tok_s": server_tok_s,
                "mean_accept": mean_accept,
                "per_request": [
                    {"idx": r.idx, "tokens": r.tokens, "elapsed": r.elapsed,
                     "tok_s": r.tok_s, "accept": r.acceptance, "error": r.error}
                    for r in results
                ],
            }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
