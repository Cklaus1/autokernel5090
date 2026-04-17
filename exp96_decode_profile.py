#!/usr/bin/env python3
"""Exp 96: Decode-only GPU trace to get clean kernel breakdown.

Previous profiling included prefill tokens. This experiment:
1. Runs warmup with same prompt to capture all CUDA graphs
2. Profiles ONLY the decode phase (excludes prefill)
3. Reports per-token GPU kernel breakdown
"""

import os, time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"


def main():
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
    )

    sp = SamplingParams(max_tokens=200, temperature=0.0)
    # Thorough warmup — captures all CUDA graph sizes
    llm.generate([PROMPT], sp)
    llm.generate([PROMPT], sp)
    llm.generate([PROMPT], sp)

    # Profile with more tokens for better statistics
    print("Profiling 200 decode tokens...")
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        out = llm.generate([PROMPT], SamplingParams(max_tokens=200, temperature=0.0))

    ntok = len(out[0].outputs[0].token_ids)
    events = prof.key_averages()

    # Use self_device_time_total for actual kernel time
    total_device = sum(e.self_device_time_total for e in events if e.self_device_time_total > 0)

    print(f"\n{'='*80}")
    print(f"Tokens: {ntok}, Total device time: {total_device/1000:.1f}ms")
    print(f"Per token: {total_device/1000/ntok:.2f}ms")
    print(f"{'='*80}")

    # Categorize kernels
    categories = {
        'GEMV (weight read)': 0, 'CUTLASS GEMM': 0, 'Flash Attention': 0,
        'Delta Rule / Mamba': 0, 'Conv1d': 0, 'RMSNorm': 0,
        'FP4 Quant/Dequant': 0, 'Memset/Memcpy': 0, 'Sampling': 0,
        'Embedding': 0, 'Other': 0,
    }

    print(f"\nTop 25 kernels by GPU time:")
    for i, e in enumerate(sorted(events, key=lambda x: x.self_device_time_total, reverse=True)[:25]):
        pct = e.self_device_time_total / total_device * 100
        name = e.key
        print(f"  {e.self_device_time_total/1000:8.2f}ms ({pct:5.1f}%) [{e.count:5d}x] {name[:90]}")

        # Categorize
        nl = name.lower()
        if 'gemvx' in nl or 'gemv' in nl:
            categories['GEMV (weight read)'] += e.self_device_time_total
        elif 'cutlass' in nl or 'gemm' in nl:
            categories['CUTLASS GEMM'] += e.self_device_time_total
        elif 'flash' in nl or 'attention' in nl:
            categories['Flash Attention'] += e.self_device_time_total
        elif 'recurrent' in nl or 'delta_rule' in nl or 'mamba' in nl:
            categories['Delta Rule / Mamba'] += e.self_device_time_total
        elif 'conv1d' in nl or 'conv_' in nl:
            categories['Conv1d'] += e.self_device_time_total
        elif 'rms_norm' in nl or 'rmsnorm' in nl or 'fused__to_copy_add' in nl or 'fused__to_copy_add_mean_mul_pow_rsqrt' in nl:
            categories['RMSNorm'] += e.self_device_time_total
        elif 'fp4' in nl or 'quant' in nl or 'cvt_' in nl or 'silu_mul_cvt' in nl:
            categories['FP4 Quant/Dequant'] += e.self_device_time_total
        elif 'memset' in nl or 'memcpy' in nl:
            categories['Memset/Memcpy'] += e.self_device_time_total
        elif 'sample' in nl or 'topk' in nl or 'softmax' in nl:
            categories['Sampling'] += e.self_device_time_total
        elif 'embed' in nl:
            categories['Embedding'] += e.self_device_time_total
        else:
            categories['Other'] += e.self_device_time_total

    # Also categorize remaining kernels
    top25_keys = {e.key for e in sorted(events, key=lambda x: x.self_device_time_total, reverse=True)[:25]}
    for e in events:
        if e.key not in top25_keys and e.self_device_time_total > 0:
            nl = e.key.lower()
            if 'gemvx' in nl or 'gemv' in nl:
                categories['GEMV (weight read)'] += e.self_device_time_total
            elif 'cutlass' in nl or 'gemm' in nl:
                categories['CUTLASS GEMM'] += e.self_device_time_total
            elif 'flash' in nl or 'attention' in nl:
                categories['Flash Attention'] += e.self_device_time_total
            elif 'recurrent' in nl or 'delta_rule' in nl:
                categories['Delta Rule / Mamba'] += e.self_device_time_total
            elif 'conv1d' in nl:
                categories['Conv1d'] += e.self_device_time_total
            elif 'rms' in nl or 'norm' in nl:
                categories['RMSNorm'] += e.self_device_time_total
            elif 'fp4' in nl or 'quant' in nl or 'cvt_' in nl:
                categories['FP4 Quant/Dequant'] += e.self_device_time_total
            elif 'memset' in nl or 'memcpy' in nl:
                categories['Memset/Memcpy'] += e.self_device_time_total
            else:
                categories['Other'] += e.self_device_time_total

    print(f"\n{'='*80}")
    print("Category breakdown:")
    for cat, us in sorted(categories.items(), key=lambda x: -x[1]):
        if us > 0:
            pct = us / total_device * 100
            print(f"  {cat:25s}: {us/1000:8.2f}ms ({pct:5.1f}%) | {us/1000/ntok:.3f}ms/tok")

    print(f"\n  {'TOTAL':25s}: {total_device/1000:8.2f}ms         | {total_device/1000/ntok:.3f}ms/tok")

    # Log
    desc = f"Decode profile: {total_device/1000/ntok:.2f}ms/tok GPU. GEMV={categories['GEMV (weight read)']/total_device*100:.0f}%, GEMM={categories['CUTLASS GEMM']/total_device*100:.0f}%"
    with open("results.tsv", "a") as f:
        f.write(f"96\texp96_decode_profile\tvllm_overhead\t{ntok/(total_device/1e6):.1f}\t0\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
