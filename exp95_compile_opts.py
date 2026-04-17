#!/usr/bin/env python3
"""Exp 95: torch.compile optimization options.

Test different compilation configurations:
  a) custom_ops=['all'] — use vLLM's hand-tuned ops instead of Inductor
  b) custom_ops=['none', '+rms_norm', '+silu_and_mul'] — selective custom ops
  c) Inductor combo_kernels disabled (to measure their impact)
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


def run_config(name, compilation_config):
    from vllm import LLM
    import torch
    print(f"\n{'='*60}")
    print(f"[EXP95] {name}")
    print(f"{'='*60}")

    try:
        llm = LLM(
            model=MODEL,
            gpu_memory_utilization=0.90,
            max_num_seqs=128,
            max_model_len=4096,
            enable_chunked_prefill=False,
            compilation_config=compilation_config,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return None, None

    tok_s, times = bench_decode(llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS)
    print(f"  Decode: {tok_s:.1f} tok/s")
    print(f"  Runs: {['%.3fs' % t for t in times]}")

    batch_tps, _ = bench_batch(llm, PROMPT, MAX_TOKENS, 32)
    print(f"  Batch32: {batch_tps:.0f} tok/s")

    del llm
    torch.cuda.empty_cache()
    return tok_s, batch_tps


def main():
    configs = [
        ("custom_ops_all", {"custom_ops": ["all"]}),
        ("custom_ops_selective", {"custom_ops": ["none", "+rms_norm", "+silu_and_mul", "+rotary_embedding"]}),
        ("no_combo_kernels", {"inductor_compile_config": {"combo_kernels": False, "benchmark_combo_kernel": False}}),
    ]

    results = {}
    for name, cc in configs:
        tok_s, b32 = run_config(name, cc)
        results[name] = (tok_s, b32)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Baseline (custom_ops=['none']): decode=120.8, batch32=3157")
    for name, (tok_s, b32) in results.items():
        if tok_s:
            print(f"  {name}: decode={tok_s:.1f}, batch32={b32:.0f}")

    for name, (tok_s, b32) in results.items():
        if tok_s:
            desc = f"{name}: decode={tok_s:.1f}, batch32={b32:.0f}"
            with open("results.tsv", "a") as f:
                f.write(f"95\t{name}\tvllm_overhead\t{tok_s:.1f}\t{b32:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
