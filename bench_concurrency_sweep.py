#!/usr/bin/env python3
"""Sweep concurrency levels to verify crash-free operation with event fence.

Sends parallel requests at C=1,2,4,8,16,32,64,128,256 and reports:
- Whether any request failed (crash/timeout)
- Output token throughput
- Latency statistics
"""

import argparse
import asyncio
import json
import time
import aiohttp


MODEL = "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
PROMPT = "Write a short poem about GPU programming. Be creative and use exactly 50 words."
MAX_TOKENS = 80


async def send_request(session, url, request_id):
    """Send one chat completion request, return timing info."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"{PROMPT} (Request #{request_id})"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            body = await resp.json()
            elapsed = time.perf_counter() - t0
            if resp.status != 200:
                return {"id": request_id, "status": "ERROR", "code": resp.status,
                        "elapsed": elapsed, "error": str(body)}
            usage = body.get("usage", {})
            content = body["choices"][0]["message"]["content"]
            return {
                "id": request_id,
                "status": "OK",
                "elapsed": elapsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "content_len": len(content),
            }
    except asyncio.TimeoutError:
        return {"id": request_id, "status": "TIMEOUT",
                "elapsed": time.perf_counter() - t0}
    except Exception as e:
        return {"id": request_id, "status": "CRASH",
                "elapsed": time.perf_counter() - t0, "error": str(e)}


async def run_concurrency_test(url, concurrency, num_requests=None):
    """Run num_requests at the given concurrency level."""
    if num_requests is None:
        num_requests = max(concurrency * 2, 8)

    print(f"\n{'='*60}")
    print(f"  Concurrency = {concurrency}, Requests = {num_requests}")
    print(f"{'='*60}")

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Use semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request(rid):
            async with sem:
                return await send_request(session, url, rid)

        t_start = time.perf_counter()
        tasks = [bounded_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        t_total = time.perf_counter() - t_start

    # Analyze results
    ok = [r for r in results if r["status"] == "OK"]
    errors = [r for r in results if r["status"] != "OK"]

    total_output_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in ok)

    throughput = total_output_tokens / t_total if t_total > 0 else 0

    if ok:
        latencies = [r["elapsed"] for r in ok]
        avg_lat = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    else:
        avg_lat = p50 = p99 = 0

    status = "PASS" if len(errors) == 0 else "FAIL"

    print(f"  Status:     {status}")
    print(f"  OK/Total:   {len(ok)}/{num_requests}")
    if errors:
        for e in errors[:3]:
            print(f"  Error:      [{e['status']}] request #{e['id']}: {e.get('error', 'N/A')[:100]}")
    print(f"  Wall time:  {t_total:.1f}s")
    print(f"  Throughput: {throughput:.1f} output tok/s")
    print(f"  Latency:    avg={avg_lat:.2f}s  p50={p50:.2f}s  p99={p99:.2f}s")
    print(f"  Tokens:     {total_prompt_tokens} prompt + {total_output_tokens} output")

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "ok": len(ok),
        "errors": len(errors),
        "status": status,
        "wall_time": t_total,
        "throughput_tok_s": throughput,
        "avg_latency": avg_lat,
        "p50_latency": p50,
        "p99_latency": p99,
        "total_output_tokens": total_output_tokens,
        "error_details": [{"id": e["id"], "status": e["status"],
                          "error": e.get("error", "")[:200]} for e in errors],
    }


async def main(url, levels, requests_per_level):
    results = []
    for c in levels:
        r = await run_concurrency_test(url, c, requests_per_level)
        results.append(r)

        # If we see crashes, still continue to see the pattern
        if r["errors"] > 0:
            print(f"  ** {r['errors']} errors at C={c}, continuing sweep...")

    print(f"\n\n{'='*70}")
    print(f"  CONCURRENCY SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'C':>4}  {'Status':>6}  {'OK/Total':>10}  {'Throughput':>12}  {'Avg Lat':>8}  {'P99 Lat':>8}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}")
    for r in results:
        print(f"  {r['concurrency']:>4}  {r['status']:>6}  "
              f"{r['ok']:>3}/{r['num_requests']:<4}  "
              f"{r['throughput_tok_s']:>8.1f} t/s  "
              f"{r['avg_latency']:>6.2f}s  "
              f"{r['p99_latency']:>6.2f}s")

    # Save results
    with open("/root/projects/autokernel/event_fence_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to event_fence_sweep_results.json")

    # Overall verdict
    all_pass = all(r["status"] == "PASS" for r in results)
    max_c_pass = max((r["concurrency"] for r in results if r["status"] == "PASS"), default=0)
    if all_pass:
        print(f"\nVERDICT: ALL CONCURRENCY LEVELS PASS -- event fence fix is working!")
    else:
        print(f"\nVERDICT: Max stable concurrency = {max_c_pass}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--levels", type=str, default="1,4,8,16,32,64,128,256",
                       help="Comma-separated concurrency levels")
    parser.add_argument("--requests", type=int, default=None,
                       help="Requests per level (default: max(C*2, 8))")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    levels = [int(x) for x in args.levels.split(",")]

    asyncio.run(main(url, levels, args.requests))
