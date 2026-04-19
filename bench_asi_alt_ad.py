#!/usr/bin/env python3
"""
ASI alternative A+D: single-process priority scheduling + chunked prefill + prefix caching.

Mixed-load P99 TTFT benchmark against a single-GPU vLLM server, to test whether
priority scheduling + tight prefill chunking recovers most of disaggregated-1P1D's
P99 TTFT benefit without needing NCCL cross-process transfer (blocked on WSL2).

Bimodal prompts: 50% short (~256 tok), 50% long (~4K tok). Long prompts are
sent with low priority; short (decode-dominated) with high priority.

Usage:
    python3 bench_asi_alt_ad.py \
        --api-base http://localhost:8004/v1 \
        --out workspace/asi_ad_bench.json \
        --concurrencies 1,4,8,64
"""
import argparse, asyncio, json, statistics, time
import aiohttp

SHORT_PROMPT = "Summarize the benefits of exercise in 2 sentences. " * 2   # ~30 tok
LONG_TEMPLATE = (
    "You are a helpful technical writer. Context: " + ("A" * 4) * 800 +
    " Given the above context, write a 2-sentence summary."
)  # ~3500 tok target; pad/truncate to 4K


async def one_request(session, api_base, model, prompt, priority, max_tokens=64):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    if priority is not None:
        payload["priority"] = priority
    t0 = time.perf_counter()
    try:
        async with session.post(f"{api_base}/chat/completions", json=payload,
                                 timeout=aiohttp.ClientTimeout(total=600)) as resp:
            body = await resp.json()
            ttft = (time.perf_counter() - t0) * 1000
            if "error" in body:
                return None, f"err:{str(body['error'])[:80]}", "long" if "A" * 100 in prompt else "short"
            gen = body.get("usage", {}).get("completion_tokens", 0)
            kind = "long" if len(prompt) > 500 else "short"
            return ttft, gen, kind
    except Exception as e:
        return None, str(e)[:100], "err"


async def run_sweep(api_base, concurrency, n_pairs):
    """Send n_pairs × 2 requests: alternating short (hi-pri) and long (lo-pri)."""
    async with aiohttp.ClientSession() as session:
        # Get model
        async with session.get(f"{api_base}/models") as r:
            model = (await r.json())["data"][0]["id"]

        async def worker(i):
            if i % 2 == 0:
                return await one_request(session, api_base, model, SHORT_PROMPT, priority=10)
            else:
                return await one_request(session, api_base, model, LONG_TEMPLATE[:16000], priority=1)

        sem = asyncio.Semaphore(concurrency)

        async def bounded(i):
            async with sem:
                return await worker(i)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[bounded(i) for i in range(n_pairs * 2)])
        wall = time.perf_counter() - t0

    short_ttft = [r[0] for r in results if r[0] is not None and r[2] == "short"]
    long_ttft = [r[0] for r in results if r[0] is not None and r[2] == "long"]
    errors = sum(1 for r in results if r[0] is None)

    def pct(xs, p):
        if not xs:
            return 0
        xs = sorted(xs)
        return xs[int(len(xs) * p / 100)] if p < 100 else xs[-1]

    return {
        "concurrency": concurrency,
        "n_requests": n_pairs * 2,
        "wall_s": round(wall, 2),
        "short": {
            "count": len(short_ttft),
            "p50_ms": round(statistics.median(short_ttft), 1) if short_ttft else 0,
            "p99_ms": round(pct(short_ttft, 99), 1),
            "max_ms": round(max(short_ttft), 1) if short_ttft else 0,
        },
        "long": {
            "count": len(long_ttft),
            "p50_ms": round(statistics.median(long_ttft), 1) if long_ttft else 0,
            "p99_ms": round(pct(long_ttft, 99), 1),
            "max_ms": round(max(long_ttft), 1) if long_ttft else 0,
        },
        "errors": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://localhost:8004/v1")
    ap.add_argument("--out", default="/home/cklaus/projects/autokernel/workspace/asi_ad_bench.json")
    ap.add_argument("--concurrencies", default="4,8,16,64")
    ap.add_argument("--pairs-per-c", type=int, default=20,
                    help="Number of short/long pairs per concurrency")
    args = ap.parse_args()

    results = []
    print(f"{'C':>4} {'wall_s':>7} {'short_p50':>10} {'short_p99':>10} "
          f"{'long_p50':>10} {'long_p99':>10} {'err':>4}")
    print("-" * 70)
    for c in [int(x) for x in args.concurrencies.split(",")]:
        r = asyncio.run(run_sweep(args.api_base, c, args.pairs_per_c))
        results.append(r)
        print(f"{r['concurrency']:>4} {r['wall_s']:>7.1f} "
              f"{r['short']['p50_ms']:>10.1f} {r['short']['p99_ms']:>10.1f} "
              f"{r['long']['p50_ms']:>10.1f} {r['long']['p99_ms']:>10.1f} "
              f"{r['errors']:>4}")
    with open(args.out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
