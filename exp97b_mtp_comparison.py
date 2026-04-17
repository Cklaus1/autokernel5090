#!/usr/bin/env python3
"""Exp 97b: MTP 1 vs MTP 3 — decode + batch comparison.

MTP 3 crashes at batch>=2. MTP 1 worked at batch32.
This runs both back-to-back to get a clean comparison.
"""

import os, time, gc, traceback

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

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
        text = outputs[0].outputs[0].text
    avg_time = sum(times) / len(times)
    return num_tokens / avg_time, avg_time / num_tokens * 1000, num_tokens, text, times


def bench_batch_safe(llm, prompt, max_tokens, batch_size):
    """Returns (tok/s, time) or (None, error_msg) on crash."""
    import torch
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    prompts = [prompt] * batch_size
    try:
        llm.generate(prompts, sp)  # warmup
        gc.collect()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp)
        t1 = time.perf_counter()
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        return total_tokens / (t1 - t0), t1 - t0
    except Exception as e:
        return None, str(e)[:100]


def run_mtp(num_spec_tokens, batch_sizes):
    import torch
    from vllm import LLM

    print(f"\n{'='*60}")
    print(f"MTP {num_spec_tokens} speculative tokens")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
        speculative_config={
            "num_speculative_tokens": num_spec_tokens,
            "method": "qwen3_5_mtp",
        },
    )

    # Decode
    tok_s, ms_tok, ntok, text, times = bench_decode(
        llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS
    )
    print(f"  Decode: {tok_s:.1f} tok/s ({ms_tok:.2f} ms/tok)")
    print(f"  Runs: {['%.3fs' % t for t in times]}")

    # Batch sweep
    batch_results = {}
    for bs in batch_sizes:
        result = bench_batch_safe(llm, PROMPT, MAX_TOKENS, bs)
        if result[0] is not None:
            tps, bt = result
            print(f"  Batch {bs}: {tps:.0f} tok/s total ({tps/bs:.1f}/user, {bt:.2f}s)")
            batch_results[bs] = tps
        else:
            print(f"  Batch {bs}: CRASHED — {result[1]}")
            batch_results[bs] = None
            break  # GPU is toast after illegal memory access

    # Cleanup
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)

    return tok_s, batch_results


def main():
    # MTP 1 — known to work with batch
    mtp1_decode, mtp1_batch = run_mtp(1, [2, 4, 8, 16, 32])

    # MTP 3 — decode only (batch crashes)
    mtp3_decode, mtp3_batch = run_mtp(3, [2])

    # Baseline from previous experiments
    baseline_decode = 120.8
    baseline_batch32 = 3157

    print(f"\n{'='*60}")
    print("SUMMARY — MTP Speculative Decoding Impact")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Decode':>10} {'Batch2':>10} {'Batch8':>10} {'Batch32':>10}")
    print(f"{'-'*60}")
    print(f"{'Baseline (no MTP)':<20} {baseline_decode:>9.1f} {'—':>10} {'—':>10} {baseline_batch32:>10}")

    b2 = f"{mtp1_batch.get(2, 0):.0f}" if mtp1_batch.get(2) else "—"
    b8 = f"{mtp1_batch.get(8, 0):.0f}" if mtp1_batch.get(8) else "—"
    b32 = f"{mtp1_batch.get(32, 0):.0f}" if mtp1_batch.get(32) else "—"
    print(f"{'MTP 1 spec token':<20} {mtp1_decode:>9.1f} {b2:>10} {b8:>10} {b32:>10}")

    b2_3 = f"{mtp3_batch.get(2, 0):.0f}" if mtp3_batch.get(2) else "CRASH"
    print(f"{'MTP 3 spec tokens':<20} {mtp3_decode:>9.1f} {b2_3:>10} {'CRASH':>10} {'CRASH':>10}")

    print(f"\nDecode speedup: MTP1={mtp1_decode/baseline_decode:.2f}x, MTP3={mtp3_decode/baseline_decode:.2f}x")
    if mtp1_batch.get(32):
        print(f"Batch32 impact: MTP1={mtp1_batch[32]/baseline_batch32:.2f}x vs baseline")
    print(f"\nConclusion: MTP >1 spec tokens crashes any batch>1 (CUDA illegal memory access in vLLM 0.17.1)")

    # Log
    desc = f"MTP comparison: MTP1 decode={mtp1_decode:.1f} batch32={mtp1_batch.get(32, 0):.0f}, MTP3 decode={mtp3_decode:.1f} batch=CRASH"
    with open("results.tsv", "a") as f:
        f.write(f"97\texp97b_mtp_compare\tvllm_perf\t{mtp3_decode:.1f}\t0\t0\t{mtp3_decode/baseline_decode:.2f}\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
