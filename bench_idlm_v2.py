#!/usr/bin/env python3
"""Compare I-DLM acceptance rates: causal (v1) vs mask_mod (v2) attention.

Sends 20 diverse prompts to both servers concurrently, measures tok/s and
extracts acceptance rate when available in the API response.

Usage:
    python bench_idlm_v2.py [--idlm http://localhost:8200/v1] \
                             [--ar   http://localhost:8201/v1] \
                             [--concurrency 1] [--model MODEL]
"""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

PROMPTS = [
    # Simple
    ("simple",   "Hello! How are you?"),
    ("simple",   "What is 2 + 2?"),
    ("simple",   "List 5 colors of the rainbow."),
    ("simple",   "What is the capital of France?"),
    ("simple",   "Name three fruits."),
    # Medium
    ("medium",   "Write a Python function that sorts a list using merge sort."),
    ("medium",   "Explain recursion with a short example."),
    ("medium",   "What is the difference between TCP and UDP?"),
    ("medium",   "Describe how a hash table works."),
    ("medium",   "Explain the CAP theorem in distributed systems."),
    # Hard
    ("hard",     "Design a distributed cache with TTL eviction and consistent hashing."),
    ("hard",     "Debug this race condition: two threads increment a shared counter "
                 "without a lock. Explain the fix in detail."),
    ("hard",     "Compare B-trees vs LSM trees for a write-heavy workload and justify "
                 "a recommendation."),
    ("hard",     "Explain how RLHF works and its limitations in aligning LLMs."),
    ("hard",     "Walk through the transformer attention mechanism mathematically."),
    # Long output
    ("long",     "Write a 500-word essay about the societal impact of artificial "
                 "intelligence."),
    ("long",     "Implement a full REST API in Python with FastAPI: endpoints for "
                 "create, read, update, delete of a 'Task' resource with SQLite."),
    ("long",     "Write a detailed tutorial on setting up a Kubernetes cluster from "
                 "scratch, including networking and storage."),
    ("long",     "Compose a short story (400+ words) about a robot learning empathy."),
    ("long",     "Explain quantum computing from first principles, covering qubits, "
                 "superposition, entanglement, and quantum gates in depth."),
]

assert len(PROMPTS) == 20


@dataclass
class Result:
    prompt_idx: int
    difficulty: str
    prompt_short: str
    tokens: int
    elapsed: float
    tok_s: float
    acceptance_rate: Optional[float]
    error: Optional[str]


async def send_one(session: aiohttp.ClientSession, base_url: str, model: str,
                   idx: int, difficulty: str, prompt: str,
                   max_tokens: int = 512) -> Result:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json(content_type=None)
            elapsed = time.monotonic() - t0
            if resp.status != 200 or "choices" not in data:
                return Result(idx, difficulty, prompt[:40], 0, elapsed, 0.0,
                              None, f"HTTP {resp.status}: {str(data)[:80]}")
            usage = data.get("usage", {})
            tokens = usage.get("completion_tokens", 0)
            tok_s = tokens / elapsed if elapsed > 0 else 0.0
            # SGLang may expose acceptance rate in usage or a custom field
            ar = (usage.get("acceptance_rate")
                  or data.get("acceptance_rate")
                  or data.get("spec_decode_stats", {}).get("acceptance_rate"))
            return Result(idx, difficulty, prompt[:40], tokens, elapsed,
                          tok_s, ar, None)
    except Exception as e:
        elapsed = time.monotonic() - t0
        return Result(idx, difficulty, prompt[:40], 0, elapsed, 0.0, None, str(e)[:80])


async def run_server(base_url: str, model: str, concurrency: int) -> list[Result]:
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)

    async def bounded(idx, diff, prompt):
        async with sem:
            return await send_one(session, base_url, model, idx, diff, prompt)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded(i, d, p) for i, (d, p) in enumerate(PROMPTS)]
        return await asyncio.gather(*tasks)


def fmt_ar(val: Optional[float]) -> str:
    return f"{val:.2%}" if val is not None else "  n/a "


def print_table(idlm_results: list[Result], ar_results: list[Result]) -> None:
    header = (f"{'#':>2}  {'Diff':<6}  {'Prompt':<40}  "
              f"{'IDLM tok/s':>10}  {'AR tok/s':>8}  {'Ratio':>6}  "
              f"{'IDLM AR':>8}  {'Err?'}")
    print(header)
    print("-" * len(header))
    idlm_toks = ar_toks = 0
    for ir, ar in zip(idlm_results, ar_results):
        ratio = (ir.tok_s / ar.tok_s) if ar.tok_s > 0 else float("nan")
        err_flag = ("!" if ir.error or ar.error else " ")
        print(f"{ir.prompt_idx:>2}  {ir.difficulty:<6}  {ir.prompt_short:<40}  "
              f"{ir.tok_s:>10.1f}  {ar.tok_s:>8.1f}  {ratio:>6.2f}x  "
              f"{fmt_ar(ir.acceptance_rate):>8}  {err_flag}")
        idlm_toks += ir.tokens
        ar_toks += ar.tokens

    print()
    idlm_ok = [r for r in idlm_results if not r.error]
    ar_ok   = [r for r in ar_results   if not r.error]
    if idlm_ok:
        mean_idlm = sum(r.tok_s for r in idlm_ok) / len(idlm_ok)
        ar_vals = [r.acceptance_rate for r in idlm_ok if r.acceptance_rate is not None]
        ar_mean = sum(ar_vals) / len(ar_vals) if ar_vals else None
        print(f"I-DLM  mean tok/s: {mean_idlm:>8.1f}  "
              f"mean acceptance: {fmt_ar(ar_mean)}  "
              f"total tokens: {idlm_toks}")
    if ar_ok:
        mean_ar = sum(r.tok_s for r in ar_ok) / len(ar_ok)
        print(f"AR     mean tok/s: {mean_ar:>8.1f}  "
              f"total tokens: {ar_toks}")
    if idlm_ok and ar_ok:
        overall = mean_idlm / mean_ar if mean_ar > 0 else float("nan")
        print(f"\nOverall speedup I-DLM vs AR: {overall:.3f}x")

    errors = [(i, r.error) for i, r in enumerate(idlm_results + ar_results)
              if r.error]
    if errors:
        print("\nErrors:")
        for idx, msg in errors:
            tag = "IDLM" if idx < 20 else "AR"
            print(f"  [{tag} #{idx % 20}] {msg}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idlm", default="http://localhost:8200/v1",
                        help="I-DLM SGLang server base URL")
    parser.add_argument("--ar",   default="http://localhost:8201/v1",
                        help="AR baseline server base URL")
    parser.add_argument("--model", default="default",
                        help="Model name to pass in requests (use 'default' to "
                             "query /v1/models automatically)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Max parallel requests per server")
    args = parser.parse_args()

    # Auto-detect model name if requested
    async def get_model(base_url: str) -> str:
        if args.model != "default":
            return args.model
        url = base_url.rstrip("/") + "/models"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    data = await r.json(content_type=None)
                    return data["data"][0]["id"]
        except Exception:
            return "default"

    idlm_model, ar_model = await asyncio.gather(
        get_model(args.idlm), get_model(args.ar))

    print(f"I-DLM server : {args.idlm}  model={idlm_model}")
    print(f"AR server    : {args.ar}  model={ar_model}")
    print(f"Concurrency  : {args.concurrency}")
    print(f"Prompts      : {len(PROMPTS)}")
    print()

    idlm_res, ar_res = await asyncio.gather(
        run_server(args.idlm, idlm_model, args.concurrency),
        run_server(args.ar,   ar_model,   args.concurrency),
    )

    print_table(idlm_res, ar_res)


if __name__ == "__main__":
    asyncio.run(main())
