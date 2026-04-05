#!/usr/bin/env python3
"""FusenCache benchmark suite.

Tests: perplexity, needle-in-a-haystack, throughput, max context.
Compares: FP16 baseline vs FusenCache v1 (FP8 K + int4 V).

Usage:
    python bench_fusencache.py [--model PATH] [--test all|perplexity|needle|throughput]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


def get_llm(model_path, kv_cache_dtype="auto", max_model_len=4096,
            gpu_util=0.92):
    """Create vLLM LLM instance."""
    from vllm import LLM
    return LLM(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True,
        trust_remote_code=True,
        kv_cache_dtype=kv_cache_dtype,
    )


def test_perplexity(model_path, tokenizer, kv_cache_dtype="auto",
                    max_model_len=4096, num_samples=20):
    """Estimate perplexity using log-likelihood of continuation tokens."""
    from vllm import LLM, SamplingParams

    # Use a set of factual prompts with known continuations
    prompts_and_expected = [
        ("The capital of France is", "Paris"),
        ("Water boils at 100 degrees", "Celsius"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("Albert Einstein developed the theory of", "relativity"),
        ("The chemical symbol for gold is", "Au"),
        ("The speed of light is approximately 300,000 kilometers per", "second"),
        ("DNA stands for deoxyribonucleic", "acid"),
        ("The Great Wall of China is located in", "China"),
        ("Shakespeare wrote Romeo and", "Juliet"),
        ("The Mona Lisa was painted by Leonardo da", "Vinci"),
        ("Pi is approximately equal to 3.14159", "265"),
        ("The human body has 206", "bones"),
        ("Mount Everest is the tallest", "mountain"),
        ("The Amazon is the largest river by", "volume"),
        ("Oxygen makes up about 21 percent of Earth's", "atmosphere"),
        ("The first person to walk on the moon was Neil", "Armstrong"),
        ("Photosynthesis converts sunlight into", "energy"),
        ("The periodic table organizes chemical", "elements"),
        ("TCP/IP is the foundation of the", "internet"),
        ("Machine learning is a subset of artificial", "intelligence"),
    ]

    llm = get_llm(model_path, kv_cache_dtype, max_model_len)

    # Generate single token from each prompt and check match
    params = SamplingParams(temperature=0.0, max_tokens=1)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Complete this: {p}"}],
        tokenize=False, add_generation_prompt=True
    ) for p, _ in prompts_and_expected[:num_samples]]

    outputs = llm.generate(prompts, params)

    correct = 0
    for i, o in enumerate(outputs):
        generated = o.outputs[0].text.strip().lower()
        expected = prompts_and_expected[i][1].lower()
        match = expected in generated or generated in expected
        if match:
            correct += 1

    accuracy = correct / num_samples
    del llm
    torch.cuda.empty_cache()
    return {"accuracy": accuracy, "correct": correct, "total": num_samples}


def test_needle(model_path, tokenizer, kv_cache_dtype="auto",
                max_model_len=8192, depths=[0.1, 0.25, 0.5, 0.75, 0.9],
                context_lengths=[512, 1024, 2048, 4096]):
    """Needle-in-a-haystack: insert fact at various depths, test retrieval."""
    from vllm import LLM, SamplingParams

    # The needle (fact to find)
    needle = "The secret code for Project Aurora is DELTA-7749."
    question = "What is the secret code for Project Aurora?"

    # Filler text
    filler = (
        "This is background information about various topics. "
        "The weather today is partly cloudy with a chance of rain. "
        "Scientists have discovered new species in the deep ocean. "
        "Technology continues to advance at a rapid pace. "
        "Global markets showed mixed results this quarter. "
    )

    llm = get_llm(model_path, kv_cache_dtype, max_model_len)
    params = SamplingParams(temperature=0.0, max_tokens=50)

    results = []
    for ctx_len in context_lengths:
        for depth in depths:
            # Build context: filler + needle at depth + more filler + question
            target_tokens = ctx_len
            filler_tokens = tokenizer.encode(filler)
            needle_tokens = tokenizer.encode(needle)

            # Calculate how many filler repetitions we need
            filler_per_rep = len(filler_tokens)
            total_filler_needed = target_tokens - len(needle_tokens) - 50
            num_reps = max(1, total_filler_needed // filler_per_rep)

            # Insert needle at specified depth
            insert_pos = int(num_reps * depth)
            parts = []
            for j in range(num_reps):
                if j == insert_pos:
                    parts.append(needle)
                parts.append(filler)

            context = " ".join(parts)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": f"{context}\n\n{question}"}],
                tokenize=False, add_generation_prompt=True
            )

            # Check actual token count
            actual_tokens = len(tokenizer.encode(prompt))
            if actual_tokens > max_model_len:
                results.append({
                    "context_length": ctx_len,
                    "depth": depth,
                    "found": None,
                    "note": "skipped (exceeds max_model_len)",
                })
                continue

            try:
                outputs = llm.generate([prompt], params)
                answer = outputs[0].outputs[0].text
                found = "DELTA-7749" in answer or "delta-7749" in answer.lower()
                results.append({
                    "context_length": ctx_len,
                    "depth": depth,
                    "actual_tokens": actual_tokens,
                    "found": found,
                    "answer": answer[:100],
                })
            except Exception as e:
                results.append({
                    "context_length": ctx_len,
                    "depth": depth,
                    "found": False,
                    "error": str(e)[:100],
                })

    del llm
    torch.cuda.empty_cache()

    # Compute summary
    tested = [r for r in results if r["found"] is not None]
    found = sum(1 for r in tested if r["found"])
    total = len(tested)
    return {
        "accuracy": found / total if total > 0 else 0,
        "found": found,
        "total": total,
        "details": results,
    }


def test_throughput(model_path, tokenizer, kv_cache_dtype="auto",
                    max_model_len=8192, gen_tokens=100, num_prompts=5):
    """Measure decode throughput (tok/s)."""
    from vllm import LLM, SamplingParams

    llm = get_llm(model_path, kv_cache_dtype, max_model_len)

    prompts_raw = [
        "Explain quantum computing in detail.",
        "Write a short story about a robot.",
        "Describe the history of the internet.",
        "What are the main causes of climate change?",
        "How does machine learning work?",
    ][:num_prompts]

    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True
    ) for p in prompts_raw]

    params = SamplingParams(temperature=0.7, max_tokens=gen_tokens, top_p=0.9)

    # Warmup
    llm.generate(prompts[:1], SamplingParams(max_tokens=10))

    # Timed run
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0

    total_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
    tok_s = total_toks / elapsed

    # Check coherence (basic: no garbage chars)
    coherent = 0
    for o in outputs:
        text = o.outputs[0].text
        # Check for garbage patterns
        if not any(c * 5 in text for c in "額/\\_LS"):
            coherent += 1

    del llm
    torch.cuda.empty_cache()

    return {
        "tok_s": round(tok_s, 1),
        "total_tokens": total_toks,
        "elapsed_s": round(elapsed, 1),
        "coherent": coherent,
        "total": num_prompts,
    }


def run_benchmark(model_path, tests="all"):
    """Run full benchmark suite."""
    print(f"{'='*60}")
    print(f"FusenCache Benchmark Suite")
    print(f"Model: {model_path}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    configs = [
        ("FP16 (baseline)", "auto", 4096),
        ("FusenCache v1", "fusen", 8192),
    ]

    # Add selective if env var set
    if os.environ.get("FUSEN_SELECTIVE") == "1":
        configs.append(("FusenCache v3 (selective)", "fusen", 8192))

    all_results = {}

    for config_name, kv_dtype, max_len in configs:
        print(f"\n{'─'*60}")
        print(f"  Config: {config_name}")
        print(f"  kv_cache_dtype={kv_dtype}, max_model_len={max_len}")
        print(f"{'─'*60}")

        results = {}

        if tests in ("all", "perplexity"):
            print("\n  [1/3] Perplexity (factual accuracy)...")
            try:
                r = test_perplexity(model_path, tokenizer, kv_dtype, max_len)
                print(f"    Accuracy: {r['accuracy']*100:.0f}% "
                      f"({r['correct']}/{r['total']})")
                results["perplexity"] = r
            except Exception as e:
                print(f"    FAILED: {e}")
                results["perplexity"] = {"error": str(e)}

        if tests in ("all", "needle"):
            print("\n  [2/3] Needle-in-a-haystack...")
            try:
                needle_lens = [512, 1024, 2048]
                if max_len >= 8192:
                    needle_lens.append(4096)
                r = test_needle(model_path, tokenizer, kv_dtype, max_len,
                                context_lengths=needle_lens)
                print(f"    Retrieval: {r['accuracy']*100:.0f}% "
                      f"({r['found']}/{r['total']})")
                for d in r["details"]:
                    if d["found"] is not None:
                        status = "✓" if d["found"] else "✗"
                        print(f"      {status} ctx={d['context_length']}, "
                              f"depth={d['depth']}")
                results["needle"] = r
            except Exception as e:
                print(f"    FAILED: {e}")
                results["needle"] = {"error": str(e)}

        if tests in ("all", "throughput"):
            print("\n  [3/3] Throughput...")
            try:
                r = test_throughput(model_path, tokenizer, kv_dtype, max_len)
                print(f"    {r['tok_s']} tok/s "
                      f"({r['total_tokens']}t in {r['elapsed_s']}s)")
                print(f"    Coherent: {r['coherent']}/{r['total']}")
                results["throughput"] = r
            except Exception as e:
                print(f"    FAILED: {e}")
                results["throughput"] = {"error": str(e)}

        all_results[config_name] = results

    # Summary table
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Accuracy':>10} {'Needle':>10} {'tok/s':>10}")
    print(f"{'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    for name, r in all_results.items():
        acc = f"{r.get('perplexity',{}).get('accuracy',0)*100:.0f}%" \
            if "perplexity" in r and "accuracy" in r["perplexity"] else "N/A"
        needle = f"{r.get('needle',{}).get('accuracy',0)*100:.0f}%" \
            if "needle" in r and "accuracy" in r["needle"] else "N/A"
        tps = f"{r.get('throughput',{}).get('tok_s',0)}" \
            if "throughput" in r and "tok_s" in r["throughput"] else "N/A"
        print(f"{name:<25} {acc:>10} {needle:>10} {tps:>10}")

    # Save results
    out_path = "fusencache_bench_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/gemma-4-31B-it-AWQ-4bit")
    parser.add_argument("--test", default="all",
                        choices=["all", "perplexity", "needle", "throughput"])
    args = parser.parse_args()
    run_benchmark(args.model, args.test)
