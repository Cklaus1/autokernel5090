#!/usr/bin/env python3
"""Final test: MTP 3 config with workaround (loop capped at 1 iteration = 2 draft tokens)."""
import os, gc, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"

from vllm import LLM, SamplingParams
llm = LLM(
    model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=200, temperature=0.0)
llm.generate([PROMPT], sp)
llm.generate([PROMPT], sp)

# Decode benchmark
times = []
for _ in range(3):
    gc.collect()
    t0 = time.perf_counter()
    outputs = llm.generate([PROMPT], sp)
    t1 = time.perf_counter()
    times.append(t1 - t0)
    ntok = len(outputs[0].outputs[0].token_ids)
avg = sum(times) / len(times)
tok_s = ntok / avg
print(f"Decode: {tok_s:.1f} tok/s ({avg/ntok*1000:.2f} ms/tok)")

# Batch sweep
for bs in [2, 4, 8, 16, 32, 64]:
    try:
        llm.generate([PROMPT]*bs, sp)  # warmup
        gc.collect()
        t0 = time.perf_counter()
        outputs = llm.generate([PROMPT]*bs, sp)
        t1 = time.perf_counter()
        ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = ntok / (t1 - t0)
        print(f"Batch {bs:3d}: {tps:7.0f} tok/s total ({tps/bs:6.1f}/user)")
    except Exception as e:
        print(f"Batch {bs:3d}: CRASHED — {str(e)[:60]}")
        break

# Comparison
baseline_decode = 120.8
baseline_batch32 = 3157
mtp1_decode = 157.0
mtp1_batch32 = 2703
print(f"\nComparison:")
print(f"  No MTP:   {baseline_decode:.0f} tok/s decode, {baseline_batch32} batch32")
print(f"  MTP 1:    {mtp1_decode:.0f} tok/s decode, {mtp1_batch32} batch32")
print(f"  MTP 3*:   {tok_s:.0f} tok/s decode (loop capped at 1 iter)")
print(f"  Decode speedup vs no-MTP: {tok_s/baseline_decode:.2f}x")
