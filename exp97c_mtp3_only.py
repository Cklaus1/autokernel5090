#!/usr/bin/env python3
"""Exp 97c: MTP 3 spec tokens — decode + attempt batch 2 in fresh process."""

import os, time, gc, traceback

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


def bench_batch_safe(llm, prompt, max_tokens, batch_size):
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
        return None, str(e)[:200]


def main():
    from vllm import LLM

    print(f"[EXP97c] MTP 3 speculative tokens — fresh process")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
        speculative_config={
            "num_speculative_tokens": 3,
            "method": "qwen3_5_mtp",
        },
    )

    # Decode
    tok_s, ms_tok, ntok, text, times = bench_decode(llm, PROMPT, MAX_TOKENS)
    print(f"  Decode: {tok_s:.1f} tok/s ({ms_tok:.2f} ms/tok)")
    print(f"  Runs: {['%.3fs' % t for t in times]}")
    print(f"  Output preview: {text[:100]}...")

    # Batch sweep — try small sizes first
    for bs in [2, 4, 8, 16, 32]:
        result = bench_batch_safe(llm, PROMPT, MAX_TOKENS, bs)
        if result[0] is not None:
            tps, bt = result
            print(f"  Batch {bs}: {tps:.0f} tok/s total ({tps/bs:.1f}/user, {bt:.2f}s)")
        else:
            print(f"  Batch {bs}: CRASHED — {result[1][:100]}")
            break

    # Summary comparison
    print(f"\n{'='*60}")
    print("MTP 3 Speculative Decoding vs Baselines:")
    print(f"  MTP 3 decode: {tok_s:.1f} tok/s")
    print(f"  MTP 1 decode: ~157 tok/s (from exp97b)")
    print(f"  No MTP:       ~121 tok/s (from exp87b)")
    print(f"  Decode speedup vs no-MTP: {tok_s/121:.2f}x")


if __name__ == "__main__":
    main()
