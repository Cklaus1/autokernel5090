#!/usr/bin/env python3
"""FINAL TEST: MTP 3 with enforce_eager + short_conv fix."""
import os, gc, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOG_LEVEL"] = "WARNING"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"

from vllm import LLM, SamplingParams
llm = LLM(
    model=MODEL, gpu_memory_utilization=0.90, max_num_seqs=128,
    max_model_len=4096, enable_chunked_prefill=False,
    enforce_eager=True,  # Required for MTP >1 on hybrid Mamba models
    speculative_config={"num_speculative_tokens": 3, "method": "qwen3_5_mtp"},
)
sp = SamplingParams(max_tokens=200, temperature=0.0)
llm.generate([PROMPT], sp)
llm.generate([PROMPT], sp)

# Decode
times = []
for _ in range(3):
    gc.collect(); t0 = time.perf_counter()
    outputs = llm.generate([PROMPT], sp)
    t1 = time.perf_counter(); times.append(t1 - t0)
    ntok = len(outputs[0].outputs[0].token_ids)
tok_s = ntok / (sum(times)/len(times))
print(f"Decode: {tok_s:.1f} tok/s")

# Batch sweep
for bs in [2, 4, 8, 16, 32]:
    try:
        llm.generate([PROMPT]*bs, sp)
        gc.collect(); t0 = time.perf_counter()
        outputs = llm.generate([PROMPT]*bs, sp)
        t1 = time.perf_counter()
        ntok = sum(len(o.outputs[0].token_ids) for o in outputs)
        tps = ntok / (t1 - t0)
        print(f"Batch {bs:3d}: {tps:7.0f} tok/s ({tps/bs:.1f}/user)")
    except Exception as e:
        print(f"Batch {bs:3d}: CRASHED")
        break

print(f"\nDecode speedup vs no-MTP: {tok_s/120.8:.2f}x")
