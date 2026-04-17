#!/usr/bin/env python3
"""Exp 88: Disable async scheduling to measure its value.

The model already has async_scheduling=True by default.
This experiment DISABLES it to measure the delta.
"""

import os
import time
import gc

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
    avg_time = sum(times) / len(times)
    tok_per_sec = num_tokens / avg_time
    ms_per_tok = avg_time / num_tokens * 1000
    return tok_per_sec, ms_per_tok, avg_time, num_tokens, times


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


def main():
    from vllm import LLM
    print(f"[EXP88] Loading {MODEL} with async_scheduling=False ...")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
        async_scheduling=False,
    )

    engine_core = llm.llm_engine.engine_core.engine_core
    print(f"[EXP88] batch_queue: {engine_core.batch_queue is not None}")
    print(f"[EXP88] step_fn: {engine_core.step_fn.__name__}")

    print("\n=== Decode (1 request) ===")
    tok_s, ms_tok, _, ntok, times = bench_decode(llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS)
    print(f"  {tok_s:.1f} tok/s  ({ms_tok:.2f} ms/tok)")
    print(f"  Runs: {['%.3fs' % t for t in times]}")

    print("\n=== Batch (32 requests) ===")
    batch_tps, batch_time = bench_batch(llm, PROMPT, MAX_TOKENS, batch_size=32)
    print(f"  {batch_tps:.0f} tok/s total  ({batch_time:.2f}s)")

    exp_name = "exp88_no_async"
    desc = f"async_scheduling=False: decode={tok_s:.1f}, batch32={batch_tps:.0f}"
    print(f"\n[RESULT] {exp_name}\t{tok_s:.1f}\t{batch_tps:.0f}\t{desc}")
    with open("results.tsv", "a") as f:
        f.write(f"88\t{exp_name}\tvllm_overhead\t{tok_s:.1f}\t{batch_tps:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
