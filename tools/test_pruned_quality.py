#!/usr/bin/env python3
"""Test quality and throughput of pruned Gemma4 checkpoints with vLLM.

Run as standalone script - do NOT import torch before launching vLLM.
"""

import json
import os
import sys
import time

TEST_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "What is the capital of France?",
    "Translate 'hello world' into Japanese.",
    "What are the main causes of climate change?",
    "Write Python code to reverse a linked list.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is 15 * 23?",
    "Explain the difference between TCP and UDP.",
    "Write a limerick about a programmer.",
    "What are three benefits of regular exercise?",
    "Describe the water cycle in one paragraph.",
    "What is machine learning?",
    "Name five planets in our solar system.",
    "Write a short story opening about a detective.",
    "Explain photosynthesis to a 10-year-old.",
    "What happened in 1969 related to space?",
    "Give me a recipe for scrambled eggs.",
    "What is the Pythagorean theorem?",
    "Compare democracy and monarchy in three sentences.",
]


def format_prompt(text: str) -> str:
    """Wrap raw text in Gemma chat template."""
    return f"<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"


def test_single_model(model_dir: str, label: str, gpu_mem: float = 0.93) -> dict:
    """Run quality+throughput test for a single model."""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"TESTING: {label}")
    print(f"Model: {model_dir}")
    print(f"GPU mem util: {gpu_mem}")
    print(f"{'='*60}")

    t0 = time.time()
    llm = LLM(
        model=model_dir,
        tensor_parallel_size=1,
        max_model_len=512,
        gpu_memory_utilization=gpu_mem,
        enforce_eager=True,
        trust_remote_code=True,
        quantization="modelopt",
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Quality test (greedy)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128,
                                     stop=["<end_of_turn>"])

    formatted_prompts = [format_prompt(p) for p in TEST_PROMPTS]
    print(f"Running {len(TEST_PROMPTS)} quality prompts ...")
    t0 = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    qual_time = time.time() - t0

    results = []
    coherent = 0
    total_tokens = 0
    for prompt, output in zip(TEST_PROMPTS, outputs):
        text = output.outputs[0].text.strip()
        ntok = len(output.outputs[0].token_ids)
        total_tokens += ntok
        is_coherent = (
            len(text) > 10
            and len(set(text.split())) > 3
            and text.count(text[:20]) < 5
        )
        if is_coherent:
            coherent += 1
        results.append({
            "prompt": prompt[:60],
            "response": text[:200],
            "coherent": is_coherent,
            "tokens": ntok,
        })

    qual_tps = total_tokens / qual_time if qual_time > 0 else 0

    # Throughput benchmark: longer generation
    print("Running throughput benchmark (256 tokens) ...")
    bench_params = SamplingParams(temperature=0.0, max_tokens=256,
                                   stop=["<end_of_turn>"])
    bench_prompt = format_prompt("Write a detailed essay about the history of artificial intelligence, covering all major milestones from the 1950s to 2025.")
    t0 = time.time()
    bench_out = llm.generate([bench_prompt], bench_params)
    bench_time = time.time() - t0
    bench_tokens = len(bench_out[0].outputs[0].token_ids)
    bench_tps = bench_tokens / bench_time

    bench_text = bench_out[0].outputs[0].text[:500]

    # Print sample outputs
    print(f"\nCoherence: {coherent}/{len(TEST_PROMPTS)} ({100*coherent/len(TEST_PROMPTS):.0f}%)")
    print(f"Quality tok/s: {qual_tps:.1f}")
    print(f"Bench tok/s: {bench_tps:.1f} ({bench_tokens} tokens in {bench_time:.1f}s)")
    print(f"\nSample responses:")
    for r in results[:5]:
        print(f"  Q: {r['prompt']}")
        print(f"  A: {r['response'][:150]}")
        print(f"  OK: {r['coherent']}")
        print()
    print(f"\nBench essay excerpt:")
    print(f"  {bench_text[:300]}")

    result = {
        "label": label,
        "model_dir": model_dir,
        "load_time_s": round(load_time, 1),
        "coherent": coherent,
        "total_prompts": len(TEST_PROMPTS),
        "coherence_pct": round(100.0 * coherent / len(TEST_PROMPTS), 1),
        "quality_tok_per_sec": round(qual_tps, 1),
        "bench_tok_per_sec": round(bench_tps, 1),
        "bench_tokens": bench_tokens,
        "bench_text": bench_text,
        "sample_responses": results[:5],
        "all_responses": results,
    }

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_pruned_quality.py <model_dir> <label> [gpu_mem_util]")
        sys.exit(1)

    model_dir = sys.argv[1]
    label = sys.argv[2]
    gpu_mem = float(sys.argv[3]) if len(sys.argv) > 3 else 0.93
    result = test_single_model(model_dir, label, gpu_mem)

    # Save result
    out_path = f"/tmp/pruning_test_{label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResult saved to {out_path}")
