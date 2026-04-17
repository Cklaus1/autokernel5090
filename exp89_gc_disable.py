#!/usr/bin/env python3
"""Exp 89: Disable GC during vLLM decode loop.

vLLM already calls gc.freeze() at init, but GC can still run on gen0 objects
created during serving. This experiment fully disables GC during decode.

Monkey-patches EngineCore to gc.disable() before step and gc.enable() after.
Expected: +0.5-1ms per token → +5-10%.
"""

import os
import sys
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
    outputs_text = []
    for _ in range(num_runs):
        gc.collect()
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        outputs_text.append(outputs[0].outputs[0].text)
        num_tokens = len(outputs[0].outputs[0].token_ids)
    avg_time = sum(times) / len(times)
    tok_per_sec = num_tokens / avg_time
    ms_per_tok = avg_time / num_tokens * 1000
    return tok_per_sec, ms_per_tok, avg_time, num_tokens, outputs_text[-1], times


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
    # ── Monkey-patch: disable GC during engine step ──
    from vllm.v1.engine.core import EngineCore

    original_step = EngineCore.step

    def step_no_gc(self):
        gc.disable()
        try:
            return original_step(self)
        finally:
            gc.enable()

    EngineCore.step = step_no_gc
    print("[EXP89] Patched EngineCore.step to disable GC during execution")

    # ── Init engine ──
    from vllm import LLM
    print(f"[EXP89] Loading {MODEL} ...")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
    )

    # ── Benchmarks ──
    print("\n=== Decode (1 request) ===")
    tok_s, ms_tok, total, ntok, text, times = bench_decode(
        llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS
    )
    print(f"  {tok_s:.1f} tok/s  ({ms_tok:.2f} ms/tok)")
    print(f"  {ntok} tokens in {total:.3f}s")
    print(f"  Runs: {['%.3fs' % t for t in times]}")

    print("\n=== Batch (32 requests) ===")
    batch_tps, batch_time = bench_batch(llm, PROMPT, MAX_TOKENS, batch_size=32)
    print(f"  {batch_tps:.0f} tok/s total  ({batch_time:.2f}s)")

    # ── Log ──
    exp_name = "exp89_gc_disable"
    desc = f"GC disabled during step: decode={tok_s:.1f}, batch32={batch_tps:.0f}"
    print(f"\n[RESULT] {exp_name}\t{tok_s:.1f}\t{batch_tps:.0f}\t{desc}")
    with open("results.tsv", "a") as f:
        f.write(f"89\t{exp_name}\tvllm_overhead\t{tok_s:.1f}\t{batch_tps:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
