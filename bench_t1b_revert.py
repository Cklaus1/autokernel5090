#!/usr/bin/env python3
"""T1-B revert verification: FusenCache k4v4b64 + full CG after restoring pre_t1b backend."""
import json, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = (
    "You are a helpful assistant. Write a detailed 200-word summary of "
    "the history of the Roman Empire from 27 BC to 476 AD."
)


def get_model(api):
    return requests.get(f"{api}/models", timeout=10).json()["data"][0]["id"]


def call_one(api, model, prompt, max_tokens=128):
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


def sweep(api, model, concurrency, n_req):
    prompts = [PROMPT] * n_req
    t0 = time.time()
    total_g = errors = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(call_one, api, model, p) for p in prompts]
        for f in as_completed(futs):
            _, gt, _, err = f.result()
            total_g += gt
            if err: errors += 1
    wall = time.time() - t0
    return {"C": concurrency, "N": n_req, "wall": round(wall, 2),
            "gen_tok_s": round(total_g / wall, 1) if wall else 0, "err": errors}


def main():
    API = "http://localhost:8001/v1"
    model = get_model(API)
    print(f"Model: {model}")
    call_one(API, model, "Hello", 10)  # warmup
    print(f"{'C':>5} {'N':>5} {'wall_s':>8} {'gen_tok/s':>12} {'err':>5}")
    print("-" * 50)
    results = []
    # Note: max_num_seqs=64 in launch, so C>64 will queue
    for c, n in [(32, 64), (64, 128), (128, 128)]:
        r = sweep(API, model, c, n)
        print(f"{r['C']:>5} {r['N']:>5} {r['wall']:>8.1f} {r['gen_tok_s']:>12.1f} {r['err']:>5}")
        results.append(r)
    with open("/home/cklaus/projects/autokernel/bench_t1b_revert.json", "w") as f:
        json.dump({"model": model, "results": results}, f, indent=2)
    print("Saved bench_t1b_revert.json")


if __name__ == "__main__":
    main()
