"""CUDA_LAUNCH_BLOCKING crash pinpointing test.

Run inside the vllm-built container with:
  CUDA_LAUNCH_BLOCKING=1 FUSEN_SYNC=1 python3 /fusen/profiling/clb_test.py
"""
import sys
import os

sys.path.insert(0, "/fusen")

# Pre-register the plugin BEFORE vLLM is imported so that the spawned
# EngineCore subprocess (which runs this same file) also picks it up
# via the dist-info entry_point we installed in clb_run.sh.
from fusen_kv.plugin import register
register()


def main():
    from vllm import LLM, SamplingParams
    import torch

    print("=== CUDA_LAUNCH_BLOCKING crash pinpointing test ===", flush=True)
    print(f"CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING', 'NOT SET')}", flush=True)
    print(f"FUSEN_SYNC={os.environ.get('FUSEN_SYNC', 'NOT SET')}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

    print("Starting LLM init...", flush=True)
    llm = LLM(
        model="/models/gemma-4-26B-A4B-it-NVFP4-modelopt",
        quantization="modelopt",
        max_model_len=512,
        max_num_seqs=4,
        enforce_eager=True,
        trust_remote_code=True,
        kv_cache_dtype="k4v4b64",
        gpu_memory_utilization=0.92,
    )

    print("LLM ready. Starting inference loop...", flush=True)
    # Keep max_tokens short to complete many decode steps within time budget.
    # CUDA_LAUNCH_BLOCKING=1 serializes all CUDA ops, making each token ~25x slower.
    # At ~0.5s/token with CLB, 20 tokens * 20 requests = ~200s total.
    sp = SamplingParams(max_tokens=20, temperature=0.0)
    prompts = [
        "Count from 1 to 10",
        "List the first 20 prime numbers",
        "Write a short poem about the ocean",
        "Explain quantum computing in simple terms",
        "What is the capital of France?",
        "Describe the water cycle",
        "What is machine learning?",
        "Name the planets in our solar system",
    ]
    crashed = False
    n_requests = 0
    for i in range(20):
        prompt = prompts[i % len(prompts)]
        try:
            out = llm.generate([prompt], sp)
            text = out[0].outputs[0].text[:40]
            print(f"  Req {n_requests:03d}: OK --- {text!r}", flush=True)
            n_requests += 1
        except Exception as e:
            print(f"  Req {n_requests:03d}: CRASH --- {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            crashed = True
            break

    if not crashed:
        print(f"\nAll {n_requests} requests completed without crash.", flush=True)
    else:
        print(f"\nCRASHED after {n_requests} requests.", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
