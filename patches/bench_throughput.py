#!/usr/bin/env python3
"""Benchmark vLLM throughput at various concurrency levels."""
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API = 'http://localhost:8000/v1'
MODEL = requests.get(f'{API}/models').json()['data'][0]['id']
PROMPTS = ['Count to 20.', 'List 5 fruits.', 'Name 3 oceans.', 'Define speed.']


def bench(conc, max_tokens=128):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(conc * 2)]
    total_gen = 0
    t0 = time.time()

    def do_one(p):
        r = requests.post(f'{API}/chat/completions', json={
            'model': MODEL,
            'messages': [{'role': 'user', 'content': p}],
            'max_tokens': max_tokens,
            'temperature': 0.7,
        }, timeout=300)
        return r.json().get('usage', {}).get('completion_tokens', 0)

    with ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(do_one, p) for p in prompts]
        for f in as_completed(futs):
            total_gen += f.result()
    return total_gen / (time.time() - t0)


# Quick correctness check
print("=== Correctness Check ===")
r = requests.post(f'{API}/chat/completions', json={
    'model': MODEL,
    'messages': [{'role': 'user', 'content': 'Say hello in 5 words.'}],
    'max_tokens': 32,
    'temperature': 0,
}, timeout=60)
resp = r.json()
print(f"Response: {resp['choices'][0]['message']['content'][:100]}")
print(f"Tokens: {resp.get('usage', {})}")

# Warmup
print("\n=== Warmup ===")
bench(4, max_tokens=32)
print("Warmup done")

# Benchmark
print("\n=== Throughput Benchmark ===")
for c in [1, 32, 128, 256]:
    tps = bench(c)
    print(f'C={c:4d}: {tps:.1f} tok/s')
