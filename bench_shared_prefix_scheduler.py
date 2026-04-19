"""Synthetic shared-prefix scheduler benchmark.

Submits 60 interleaved requests across 3 distinct 500-token system prompts
against a vLLM OpenAI-compatible server, measures wall time and prefix
cache hit rate, and prints a tag-line suitable for appending to
``results.tsv``.

Usage:
    python3 bench_shared_prefix_scheduler.py \
        --base-url http://localhost:8010/v1 \
        --model Qwen3-30B-A3B-NVFP4 \
        --label cache_aware_sched_baseline \
        --concurrency 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time

import aiohttp

# --------------------------------------------------------------------------
# Three distinct 500-token system prompts.  Each repeats a unique phrase
# enough times to overflow the prefix-cache block boundary so that the
# first block-hash differs across prompts.  500 tokens >> BLOCK_SIZE (16)
# so we get many full blocks of identical content per prompt.
# --------------------------------------------------------------------------
PROMPT_A_SEED = (
    "You are Agent Alpha, a meticulous code reviewer. You examine diffs "
    "for correctness, performance, and security. Always cite the exact "
    "line numbers you reference. "
)
PROMPT_B_SEED = (
    "You are Agent Bravo, a friendly customer support representative for "
    "an e-commerce platform. You answer questions about orders, returns, "
    "and shipping. Be concise and polite. "
)
PROMPT_C_SEED = (
    "You are Agent Charlie, a research scientist specialising in applied "
    "mathematics. You derive solutions step by step, citing relevant "
    "theorems and checking boundary conditions. "
)

USER_QUERIES = [
    "How should I approach task {i}?",
    "Please analyse scenario number {i} and advise.",
    "What is the best next step for item {i}?",
    "Walk me through problem {i} carefully.",
    "Evaluate option {i} against the alternatives.",
    "Summarise the key tradeoffs for case {i}.",
    "What risks do you see in proposal {i}?",
    "Explain the rationale behind choice {i}.",
    "Outline a plan to address request {i}.",
    "Give me your top recommendation for item {i}.",
    "Compare approaches for scenario {i}.",
    "Identify the main constraint in task {i}.",
    "What data would you need for case {i}?",
    "Describe the critical path for project {i}.",
    "Justify your reasoning on decision {i}.",
    "Propose a mitigation for risk {i}.",
    "Estimate the effort for change {i}.",
    "Sketch the architecture for system {i}.",
    "Enumerate assumptions behind case {i}.",
    "Benchmark the baseline for workload {i}.",
]


def _pad_prompt(seed: str, target_tokens: int) -> str:
    """Repeat ``seed`` until it is roughly ``target_tokens`` tokens long.

    We approximate 1 token ~= 4 chars, which is close enough for the
    purpose of forcing the first prefix-cache block to be identical
    across users of the same agent.
    """
    target_chars = target_tokens * 4
    out = []
    n = 0
    while n < target_chars:
        out.append(seed)
        n += len(seed)
    return "".join(out)


def build_requests() -> list[tuple[str, str, str]]:
    """Return list of (tag, system_prompt, user_prompt) in interleaved order.

    Tags are 'A'|'B'|'C'.  Interleaving: A0 B0 C0 A1 B1 C1 ... A19 B19 C19.
    """
    sys_a = _pad_prompt(PROMPT_A_SEED, 500)
    sys_b = _pad_prompt(PROMPT_B_SEED, 500)
    sys_c = _pad_prompt(PROMPT_C_SEED, 500)

    reqs: list[tuple[str, str, str]] = []
    for i in range(20):
        for tag, sysp in (("A", sys_a), ("B", sys_b), ("C", sys_c)):
            user = USER_QUERIES[i % len(USER_QUERIES)].format(i=i)
            reqs.append((tag, sysp, user))
    return reqs


async def _fire_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    base_url: str,
    model: str,
    idx: int,
    tag: str,
    sysp: str,
    userp: str,
    max_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sysp},
            {"role": "user", "content": userp},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    async with sem:
        t0 = time.perf_counter()
        async with session.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            body = await resp.json()
        t1 = time.perf_counter()
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    return {
        "idx": idx,
        "tag": tag,
        "latency_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def _fetch_metrics(base_url: str) -> str:
    metrics_url = base_url.rsplit("/", 1)[0] + "/metrics"
    async with aiohttp.ClientSession() as s:
        async with s.get(metrics_url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            return await r.text()


_COUNTER_RE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+([0-9eE.+-]+)$", re.M)


def _scalar(metrics_text: str, name: str) -> float:
    total = 0.0
    for m in _COUNTER_RE.finditer(metrics_text):
        if m.group(1) == name:
            try:
                total += float(m.group(2))
            except ValueError:
                pass
    return total


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8010/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    reqs = build_requests()
    print(f"[{args.label}] prepared {len(reqs)} interleaved requests")

    # Snapshot prefix-cache metrics BEFORE the run
    try:
        pre_metrics = await _fetch_metrics(args.base_url)
        pre_hits = _scalar(pre_metrics, "vllm:prefix_cache_hits") + _scalar(
            pre_metrics, "vllm:gpu_prefix_cache_hits"
        )
        pre_queries = _scalar(pre_metrics, "vllm:prefix_cache_queries") + _scalar(
            pre_metrics, "vllm:gpu_prefix_cache_queries"
        )
    except Exception as e:
        print(f"[warn] metrics pre-fetch failed: {e}")
        pre_hits = pre_queries = 0.0

    # Optional warmup — submit each *system* prompt once so the prefix
    # cache is populated for both runs.  (Matches the "fair start" regime:
    # baseline also benefits from warm prefix, which is what we want to
    # isolate the scheduler effect, not first-touch prefill.)
    if args.warmup:
        print(f"[{args.label}] warming prefix cache...")
        async with aiohttp.ClientSession() as s:
            sem = asyncio.Semaphore(3)
            warm = [
                _fire_one(s, sem, args.base_url, args.model, i, t, p, "warm",
                          max_tokens=4)
                for i, (t, p, _) in enumerate(
                    [("A", reqs[0][1], ""), ("B", reqs[1][1], ""),
                     ("C", reqs[2][1], "")]
                )
            ]
            await asyncio.gather(*warm)
        print(f"[{args.label}] warmup done")

    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [
            _fire_one(
                session, sem, args.base_url, args.model,
                i, tag, sysp, userp, args.max_tokens,
            )
            for i, (tag, sysp, userp) in enumerate(reqs)
        ]
        results = await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - t0

    # Snapshot prefix-cache metrics AFTER
    try:
        post_metrics = await _fetch_metrics(args.base_url)
        post_hits = _scalar(post_metrics, "vllm:prefix_cache_hits") + _scalar(
            post_metrics, "vllm:gpu_prefix_cache_hits"
        )
        post_queries = _scalar(post_metrics, "vllm:prefix_cache_queries") + _scalar(
            post_metrics, "vllm:gpu_prefix_cache_queries"
        )
    except Exception as e:
        print(f"[warn] metrics post-fetch failed: {e}")
        post_hits = post_queries = 0.0

    delta_hits = post_hits - pre_hits
    delta_queries = post_queries - pre_queries
    hit_rate = delta_hits / delta_queries if delta_queries > 0 else 0.0

    total_completion = sum(r["completion_tokens"] for r in results)
    total_prompt = sum(r["prompt_tokens"] for r in results)
    tok_s = (total_completion + total_prompt) / wall_s if wall_s > 0 else 0.0

    summary = {
        "label": args.label,
        "n_requests": len(results),
        "concurrency": args.concurrency,
        "wall_s": wall_s,
        "tok_s": tok_s,
        "prompt_tokens_total": total_prompt,
        "completion_tokens_total": total_completion,
        "prefix_cache_hits_delta": delta_hits,
        "prefix_cache_queries_delta": delta_queries,
        "prefix_cache_hit_rate": hit_rate,
    }
    print("[result] " + json.dumps(summary, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{args.label}] wrote {args.out_json}")


if __name__ == "__main__":
    asyncio.run(main())
