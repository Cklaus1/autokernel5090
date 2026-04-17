#!/usr/bin/env python3
"""Exp 97: MTP 3 speculative tokens — batch sweep.

MTP 3 gave 164.3 tok/s decode (+36%) but batch was never tested.
MTP 2 crashed batch32 with CUDA illegal memory access.
This experiment tests MTP 3 with progressively larger batches to find
the crash boundary and measure batch throughput where it works.
"""

import os, time, gc, traceback

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"
MAX_TOKENS = 200
NUM_WARMUP = 2
NUM_RUNS = 3
NUM_SPEC_TOKENS = 3


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
        text = outputs[0].outputs[0].text
    avg_time = sum(times) / len(times)
    return num_tokens / avg_time, avg_time / num_tokens * 1000, num_tokens, text, times


def bench_batch(llm, prompt, max_tokens, batch_size):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    prompts = [prompt] * batch_size
    # Warmup
    llm.generate(prompts, sp)
    gc.collect()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    t1 = time.perf_counter()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return total_tokens / (t1 - t0), t1 - t0


def main():
    import torch
    from vllm import LLM

    print(f"[EXP97] MTP {NUM_SPEC_TOKENS} spec tokens — batch sweep")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
        speculative_config={
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "method": "qwen3_5_mtp",
        },
    )

    # First: confirm decode still works
    print("\n=== Decode (1 request) ===")
    tok_s, ms_tok, ntok, text, times = bench_decode(
        llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS
    )
    print(f"  {tok_s:.1f} tok/s ({ms_tok:.2f} ms/tok)")
    print(f"  Runs: {['%.3fs' % t for t in times]}")
    print(f"  Output: {text[:150]}...")

    # Batch sweep: start small, increase until crash
    batch_results = {}
    for bs in [2, 4, 8, 16, 32, 64]:
        print(f"\n=== Batch {bs} ===")
        try:
            torch.cuda.synchronize()
            batch_tps, batch_time = bench_batch(llm, PROMPT, MAX_TOKENS, bs)
            per_user = batch_tps / bs
            print(f"  {batch_tps:.0f} tok/s total ({per_user:.1f}/user, {batch_time:.2f}s)")
            batch_results[bs] = batch_tps
        except Exception as e:
            print(f"  CRASHED: {e}")
            traceback.print_exc()
            batch_results[bs] = None
            # Try to recover GPU state
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            # If crash, no point trying larger batches
            break

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary — MTP {NUM_SPEC_TOKENS} spec tokens:")
    print(f"  Decode: {tok_s:.1f} tok/s")
    for bs, tps in batch_results.items():
        if tps is not None:
            print(f"  Batch {bs}: {tps:.0f} tok/s total ({tps/bs:.1f}/user)")
        else:
            print(f"  Batch {bs}: CRASHED")

    # Compare with baseline (no spec decode)
    baseline_decode = 120.8  # from exp87b
    baseline_batch32 = 3157  # from exp87b
    print(f"\n  vs Baseline (no spec decode):")
    print(f"  Decode: {tok_s:.1f} vs {baseline_decode} ({tok_s/baseline_decode:.2f}x)")
    if 32 in batch_results and batch_results[32] is not None:
        b32 = batch_results[32]
        print(f"  Batch32: {b32:.0f} vs {baseline_batch32} ({b32/baseline_batch32:.2f}x)")

    # Log
    max_working_batch = max((bs for bs, tps in batch_results.items() if tps is not None), default=0)
    max_batch_tps = batch_results.get(max_working_batch, 0) or 0
    crash_at = min((bs for bs, tps in batch_results.items() if tps is None), default=0)

    desc = f"MTP 3 batch: decode={tok_s:.1f}, max_batch={max_working_batch}@{max_batch_tps:.0f}tok/s"
    if crash_at:
        desc += f", crashes@batch{crash_at}"
    with open("results.tsv", "a") as f:
        f.write(f"97\texp97_mtp3_batch\tvllm_perf\t{tok_s:.1f}\t{max_batch_tps:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
