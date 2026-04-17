#!/usr/bin/env python3
"""Benchmark sweep: measure throughput at different concurrency levels."""

import asyncio
import aiohttp
import json
import time
import sys

MODEL = "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
URL = "http://localhost:8000/v1/chat/completions"

async def send_one(session, prompt, max_tokens=30):
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    t0 = time.time()
    try:
        async with session.post(URL, json=data, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            r = await resp.json()
            dt = time.time() - t0
            if "error" in r:
                return {"ok": False, "error": str(r["error"])[:100], "time": dt}
            usage = r.get("usage", {})
            return {
                "ok": True,
                "completion_tokens": usage.get("completion_tokens", 0),
                "time": dt,
            }
    except Exception as e:
        return {"ok": False, "error": str(e)[:100], "time": time.time() - t0}

async def bench_concurrency(C, max_tokens=30):
    """Run C concurrent requests and measure throughput."""
    async with aiohttp.ClientSession() as session:
        prompts = [f"Count from 1 to {i+5}." for i in range(C)]
        t0 = time.time()
        results = await asyncio.gather(*[send_one(session, p, max_tokens) for p in prompts])
        wall_time = time.time() - t0

        ok = sum(1 for r in results if r["ok"])
        fail = C - ok
        total_tokens = sum(r.get("completion_tokens", 0) for r in results if r["ok"])
        tok_per_sec = total_tokens / wall_time if wall_time > 0 else 0

        return {
            "C": C,
            "ok": ok,
            "fail": fail,
            "total_tokens": total_tokens,
            "wall_time": wall_time,
            "tok_per_sec": tok_per_sec,
        }

async def main():
    concurrencies = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    print(f"{'C':>5} {'OK':>5} {'FAIL':>5} {'Tokens':>8} {'Wall(s)':>8} {'Tok/s':>8}")
    print("-" * 50)

    results = []
    for C in concurrencies:
        # Check server health first
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get("http://localhost:8000/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status != 200:
                        print(f"Server down at C={C}!")
                        break
        except:
            print(f"Server unreachable at C={C}!")
            break

        r = await bench_concurrency(C)
        results.append(r)
        status = "OK" if r["fail"] == 0 else "FAIL"
        print(f"{r['C']:>5} {r['ok']:>5} {r['fail']:>5} {r['total_tokens']:>8} {r['wall_time']:>8.1f} {r['tok_per_sec']:>8.1f}")

        if r["fail"] > 0:
            print(f"  ^^^ {r['fail']} failures at C={C}")
            # Check if server is still alive
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get("http://localhost:8000/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        pass
            except:
                print("Server crashed!")
                break

    # Summary
    print("\n=== SUMMARY ===")
    max_ok_c = 0
    for r in results:
        if r["fail"] == 0:
            max_ok_c = r["C"]
    print(f"Max stable concurrency: C={max_ok_c}")
    if results:
        best = max(results, key=lambda r: r["tok_per_sec"])
        print(f"Peak throughput: {best['tok_per_sec']:.1f} tok/s at C={best['C']}")

if __name__ == "__main__":
    asyncio.run(main())
