#!/usr/bin/env python3
"""Exp 93: Speculative decoding with Qwen3.5's built-in MTP heads.

Qwen3.5-9B has native MTP (multi-token prediction) support.
vLLM supports this via speculative_config with method="qwen3_5_mtp".
Expected: 2-3x decode throughput if acceptance rate is good.
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
        text = outputs[0].outputs[0].text
    avg_time = sum(times) / len(times)
    return num_tokens / avg_time, avg_time / num_tokens * 1000, num_tokens, text, times


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


def try_spec_decode(num_spec_tokens):
    from vllm import LLM
    print(f"\n{'='*60}")
    print(f"[EXP93] MTP spec decode with {num_spec_tokens} speculative tokens")
    print(f"{'='*60}")

    try:
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
    except Exception as e:
        print(f"  FAILED to init: {e}")
        return None, None, str(e)

    tok_s, ms_tok, ntok, text, times = bench_decode(llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS)
    print(f"  Decode: {tok_s:.1f} tok/s ({ms_tok:.2f} ms/tok)")
    print(f"  Runs: {['%.3fs' % t for t in times]}")
    print(f"  Output: {text[:100]}...")

    batch_tps, batch_time = bench_batch(llm, PROMPT, MAX_TOKENS, batch_size=32)
    print(f"  Batch32: {batch_tps:.0f} tok/s ({batch_time:.2f}s)")

    del llm
    import torch
    torch.cuda.empty_cache()
    gc.collect()
    import time as _time
    _time.sleep(2)  # Let GPU memory settle
    return tok_s, batch_tps, text


def main():
    results = {}
    # Run each in isolation — only test one at a time due to GPU memory
    for n in [1]:
        tok_s, batch_tps, text = try_spec_decode(n)
        results[n] = (tok_s, batch_tps)
        if tok_s is None:
            break

    print(f"\n{'='*60}")
    print("Summary:")
    for n, (tok_s, batch_tps) in results.items():
        if tok_s:
            print(f"  {n} spec tokens: decode={tok_s:.1f} tok/s, batch32={batch_tps:.0f}")

    # Log best result
    best_n = max(results, key=lambda n: results[n][0] or 0)
    tok_s, batch_tps = results[best_n]
    if tok_s:
        exp_name = f"exp93_mtp_{best_n}"
        desc = f"MTP spec decode {best_n} tokens: decode={tok_s:.1f}, batch32={batch_tps:.0f}"
        with open("results.tsv", "a") as f:
            f.write(f"93\t{exp_name}\tvllm_overhead\t{tok_s:.1f}\t{batch_tps:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
