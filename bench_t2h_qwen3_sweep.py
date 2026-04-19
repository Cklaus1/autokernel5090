#!/usr/bin/env python3
"""T2-H: Qwen3-30B-A3B NVFP4 FP8 KV concurrency sweep on PRO 6000 GPU 1."""
import argparse, json, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = (
    "You are a helpful assistant. Write a detailed 200-word summary of "
    "the history of the Roman Empire, focusing on key emperors, "
    "military campaigns, and cultural achievements from 27 BC to 476 AD."
)


def get_model(api):
    return requests.get(f"{api}/models", timeout=10).json()["data"][0]["id"]


def call_one(api, model, prompt, max_tokens):
    t0 = time.time()
    try:
        r = requests.post(f"{api}/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }, timeout=600).json()
        elapsed = time.time() - t0
        if "error" in r:
            return (0, 0, elapsed, str(r["error"])[:120])
        u = r["usage"]
        return (u["prompt_tokens"], u["completion_tokens"], elapsed, None)
    except Exception as e:
        return (0, 0, time.time() - t0, str(e)[:120])


def sweep(api, model, concurrency, n_requests, max_tokens):
    prompts = [PROMPT] * n_requests
    t0 = time.time()
    total_p = total_g = errors = 0
    latencies = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(call_one, api, model, p, max_tokens) for p in prompts]
        for f in as_completed(futs):
            pt, gt, lat, err = f.result()
            total_p += pt
            total_g += gt
            latencies.append(lat)
            if err:
                errors += 1
    wall = time.time() - t0
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "max_tokens": max_tokens,
        "wall_s": round(wall, 2),
        "gen_tok_s": round(total_g / wall, 1) if wall > 0 else 0,
        "total_tok_s": round((total_p + total_g) / wall, 1) if wall > 0 else 0,
        "p50_latency_s": round(p50, 2),
        "errors": errors,
        "prompt_tokens": total_p,
        "gen_tokens": total_g,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://localhost:8002/v1")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="/home/cklaus/projects/autokernel/bench_t2h_qwen3.json")
    ap.add_argument("--concurrencies", default="64,128,256,384,512,768")
    args = ap.parse_args()

    model = get_model(args.api_base)
    print(f"Model: {model}")

    # Warmup
    call_one(args.api_base, model, "Hello", 10)
    print("Warmup OK")

    results = []
    print(f"{'C':>5} {'N':>6} {'wall_s':>8} {'gen_tok/s':>12} {'total_tok/s':>14} {'p50_s':>8} {'err':>4}")
    print("-" * 70)
    for c in [int(x) for x in args.concurrencies.split(",")]:
        # num_requests >= concurrency so we fill the batch window
        n = max(c, 64)
        r = sweep(args.api_base, model, c, n, args.max_tokens)
        results.append(r)
        print(f"{r['concurrency']:>5} {r['n_requests']:>6} {r['wall_s']:>8.1f} "
              f"{r['gen_tok_s']:>12.1f} {r['total_tok_s']:>14.1f} "
              f"{r['p50_latency_s']:>8.2f} {r['errors']:>4}")

    with open(args.out, "w") as f:
        json.dump({"model": model, "max_tokens": args.max_tokens, "results": results}, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
