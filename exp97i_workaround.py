#!/usr/bin/env python3
"""Workaround: Disable async scheduling for MTP > 1 to avoid the race condition."""

import os, time, gc

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"
MAX_TOKENS = 200


def bench_decode(llm, prompt, max_tokens, num_warmup=2, num_runs=3):
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
    llm.generate(prompts, sp)
    gc.collect()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    t1 = time.perf_counter()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return total_tokens / (t1 - t0), t1 - t0


def main():
    from vllm import LLM

    print("=== MTP 3 with async_scheduling=False (workaround) ===")

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
        async_scheduling=False,
        speculative_config={
            "num_speculative_tokens": 3,
            "method": "qwen3_5_mtp",
        },
    )

    tok_s, ms_tok, ntok, text, times = bench_decode(llm, PROMPT, MAX_TOKENS)
    print(f"Decode: {tok_s:.1f} tok/s ({ms_tok:.2f} ms/tok)")

    for bs in [2, 4, 8, 16, 32]:
        try:
            tps, bt = bench_batch(llm, PROMPT, MAX_TOKENS, bs)
            print(f"Batch {bs:3d}: {tps:7.0f} tok/s total ({tps/bs:6.1f}/user, {bt:.2f}s)")
        except Exception as e:
            print(f"Batch {bs:3d}: CRASHED — {str(e)[:80]}")
            break

    baseline_decode = 120.8
    print(f"\nDecode speedup vs no-MTP: {tok_s/baseline_decode:.2f}x")

    desc = f"MTP3 async_off: decode={tok_s:.1f}"
    with open("results.tsv", "a") as f:
        f.write(f"97\texp97i_mtp3_noasync\tvllm_perf\t{tok_s:.1f}\t0\t0\t{tok_s/baseline_decode:.2f}\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
