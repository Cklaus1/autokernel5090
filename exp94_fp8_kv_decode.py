#!/usr/bin/env python3
"""Exp 94: FP8 KV cache + reduced max_model_len for better decode & batch.

Test combinations:
  a) FP8 KV + max_model_len=2048
  b) FP8 KV + max_model_len=1024
  c) Baseline max_model_len=1024 (no FP8 KV)
"""

import os, time, gc

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"
MAX_TOKENS = 200
NUM_WARMUP = 2
NUM_RUNS = 3


def bench_decode(llm, prompt, max_tokens, num_warmup, num_runs):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    for _ in range(num_warmup):
        llm.generate([prompt], sp)
    times = []
    for _ in range(num_runs):
        gc.collect()
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        num_tokens = len(outputs[0].outputs[0].token_ids)
    return num_tokens / (sum(times) / len(times)), times


def bench_batch(llm, prompt, max_tokens, batch_size=32):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    prompts = [prompt] * batch_size
    llm.generate(prompts, sp)
    gc.collect()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    t1 = time.perf_counter()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return total_tokens / (t1 - t0), t1 - t0


def bench_batch_sweep(llm, prompt, max_tokens=200):
    """Find peak batch throughput."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    best_tps = 0
    best_bs = 0
    for bs in [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512]:
        try:
            prompts = [prompt] * bs
            gc.collect()
            t0 = time.perf_counter()
            outputs = llm.generate(prompts, sp)
            t1 = time.perf_counter()
            total = sum(len(o.outputs[0].token_ids) for o in outputs)
            tps = total / (t1 - t0)
            print(f"    batch{bs}: {tps:.0f} tok/s")
            if tps > best_tps:
                best_tps = tps
                best_bs = bs
            if tps < best_tps * 0.8:
                break  # past cliff
        except Exception as e:
            print(f"    batch{bs}: FAILED ({e})")
            break
    return best_tps, best_bs


def run_config(name, kv_dtype, max_model_len):
    from vllm import LLM
    import torch
    print(f"\n{'='*60}")
    print(f"[EXP94] {name}: kv_cache_dtype={kv_dtype}, max_model_len={max_model_len}")
    print(f"{'='*60}")

    try:
        llm = LLM(
            model=MODEL,
            gpu_memory_utilization=0.90,
            max_num_seqs=128,
            max_model_len=max_model_len,
            enable_chunked_prefill=False,
            kv_cache_dtype=kv_dtype,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return None, None, None, None

    tok_s, times = bench_decode(llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS)
    print(f"  Decode: {tok_s:.1f} tok/s")
    print(f"  Runs: {['%.3fs' % t for t in times]}")

    batch_tps, _ = bench_batch(llm, PROMPT, MAX_TOKENS, 32)
    print(f"  Batch32: {batch_tps:.0f} tok/s")

    print("  Batch sweep:")
    peak_tps, peak_bs = bench_batch_sweep(llm, PROMPT)
    print(f"  Peak: {peak_tps:.0f} tok/s at batch {peak_bs}")

    del llm
    torch.cuda.empty_cache()
    return tok_s, batch_tps, peak_tps, peak_bs


def main():
    configs = [
        ("fp8kv_ctx2048", "fp8", 2048),
        ("fp8kv_ctx1024", "fp8", 1024),
        ("auto_ctx1024", "auto", 1024),
    ]

    results = {}
    for name, kv, ctx in configs:
        tok_s, b32, peak, peak_bs = run_config(name, kv, ctx)
        results[name] = (tok_s, b32, peak, peak_bs)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Baseline: decode=120.8, batch32=3157, peak=7075@bs232")
    for name, (tok_s, b32, peak, peak_bs) in results.items():
        if tok_s:
            print(f"  {name}: decode={tok_s:.1f}, batch32={b32:.0f}, peak={peak:.0f}@bs{peak_bs}")

    # Log results
    for name, (tok_s, b32, peak, peak_bs) in results.items():
        if tok_s:
            desc = f"{name}: decode={tok_s:.1f}, batch32={b32:.0f}, peak={peak:.0f}@bs{peak_bs}"
            with open("results.tsv", "a") as f:
                f.write(f"94\t{name}\tvllm_overhead\t{tok_s:.1f}\t{b32:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
