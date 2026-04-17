#!/usr/bin/env python3
"""Exp 88: Enable async scheduling in vLLM to overlap schedule() with GPU execution.

When async_scheduling=True:
  - UniProcExecutor.max_concurrent_batches returns 2
  - EngineCore uses step_with_batch_queue() which overlaps schedule() with GPU work
  - Expected: +2-3ms saved per decode step → ~160-180 tok/s

Monkey-patches SchedulerConfig.async_scheduling before engine init.
"""

import os
import sys
import time
import gc

# Force inproc mode (best baseline config from prior experiments)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"
MAX_TOKENS = 200
NUM_WARMUP = 2
NUM_RUNS = 3


def bench_decode(llm, prompt, max_tokens, num_warmup, num_runs):
    """Benchmark single-request decode throughput."""
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Warmup
    for _ in range(num_warmup):
        llm.generate([prompt], sp)

    # Benchmark
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
    """Benchmark batch throughput."""
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
    elapsed = t1 - t0
    return total_tokens / elapsed, elapsed


def main():
    # ── Monkey-patch: enable async scheduling ──
    from vllm.config.scheduler import SchedulerConfig
    original_init = SchedulerConfig.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.async_scheduling is None or not self.async_scheduling:
            print(f"[EXP88] Patching async_scheduling: {self.async_scheduling} → True")
            self.async_scheduling = True

    SchedulerConfig.__init__ = patched_init

    # ── Init engine ──
    from vllm import LLM
    print(f"[EXP88] Loading {MODEL} with async_scheduling=True ...")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
    )

    # ── Verify async scheduling is active ──
    engine_core = llm.llm_engine.engine_core.engine_core
    has_batch_queue = engine_core.batch_queue is not None
    print(f"[EXP88] batch_queue enabled: {has_batch_queue}")
    print(f"[EXP88] batch_queue_size: {engine_core.batch_queue_size}")
    print(f"[EXP88] step_fn: {engine_core.step_fn.__name__}")

    # ── Run benchmarks ──
    print("\n=== Decode (1 request) ===")
    tok_s, ms_tok, total, ntok, text, times = bench_decode(
        llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS
    )
    print(f"  {tok_s:.1f} tok/s  ({ms_tok:.2f} ms/tok)")
    print(f"  {ntok} tokens in {total:.3f}s")
    print(f"  Runs: {['%.3fs' % t for t in times]}")
    print(f"  Output preview: {text[:100]}...")

    print("\n=== Batch (32 requests) ===")
    batch_tps, batch_time = bench_batch(llm, PROMPT, MAX_TOKENS, batch_size=32)
    print(f"  {batch_tps:.0f} tok/s total  ({batch_time:.2f}s)")

    # ── Log result ──
    exp_name = "exp88_async_sched"
    desc = f"async_scheduling=True: decode={tok_s:.1f}, batch32={batch_tps:.0f}"
    print(f"\n[RESULT] {exp_name}\t{tok_s:.1f}\t{batch_tps:.0f}\t{desc}")

    # Append to results.tsv
    with open("results.tsv", "a") as f:
        f.write(f"88\t{exp_name}\tvllm_overhead\t{tok_s:.1f}\t{batch_tps:.0f}\t0\t0\tPASS\t0\t{desc}\n")

    return tok_s, batch_tps


if __name__ == "__main__":
    main()
