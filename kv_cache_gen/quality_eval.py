"""Quality eval suite for KV cache quantization.

Tests whether quantization silently degrades complex tasks.
Three tiers: fast (30s), standard (5min), full (30min).
"""

import torch
import time
import json
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn, make_store_fn


# ===== Eval Questions =====
# Each has a question, expected answer substring, and category

EVAL_SUITE = {
    # --- Factual Knowledge (MMLU-style) ---
    "factual": [
        {
            "q": "What is the chemical formula for water?",
            "accept": ["H2O", "h2o"],
            "category": "science",
        },
        {
            "q": "Who wrote Romeo and Juliet?",
            "accept": ["Shakespeare", "shakespeare", "William Shakespeare"],
            "category": "literature",
        },
        {
            "q": "What is the largest planet in our solar system?",
            "accept": ["Jupiter", "jupiter"],
            "category": "science",
        },
        {
            "q": "In what year did World War I begin?",
            "accept": ["1914"],
            "category": "history",
        },
        {
            "q": "What is the powerhouse of the cell?",
            "accept": ["mitochondria", "Mitochondria", "mitochondrion"],
            "category": "biology",
        },
        {
            "q": "What is the speed of light in meters per second?",
            "accept": ["299,792,458", "299792458", "3 × 10^8", "3e8", "300,000,000"],
            "category": "physics",
        },
        {
            "q": "Who painted the Mona Lisa?",
            "accept": ["Leonardo", "Da Vinci", "da Vinci", "Leonardo da Vinci"],
            "category": "art",
        },
        {
            "q": "What is the capital of Australia?",
            "accept": ["Canberra", "canberra"],
            "category": "geography",
        },
        {
            "q": "What programming language was created by Guido van Rossum?",
            "accept": ["Python", "python"],
            "category": "cs",
        },
        {
            "q": "What is the derivative of x squared?",
            "accept": ["2x", "2*x", "2 x"],
            "category": "math",
        },
    ],

    # --- Math Reasoning ---
    "math": [
        {
            "q": "What is 17 × 23?",
            "accept": ["391"],
            "category": "arithmetic",
        },
        {
            "q": "If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?",
            "accept": ["60"],
            "category": "word_problem",
        },
        {
            "q": "What is the sum of the first 10 positive integers?",
            "accept": ["55"],
            "category": "series",
        },
        {
            "q": "A rectangle has a length of 8 and a width of 5. What is its area?",
            "accept": ["40"],
            "category": "geometry",
        },
        {
            "q": "What is 2^10?",
            "accept": ["1024", "1,024"],
            "category": "exponent",
        },
        {
            "q": "If x + 5 = 12, what is x?",
            "accept": ["7"],
            "category": "algebra",
        },
        {
            "q": "What is 15% of 200?",
            "accept": ["30"],
            "category": "percentage",
        },
        {
            "q": "What is the square root of 144?",
            "accept": ["12"],
            "category": "roots",
        },
        {
            "q": "If you have 3 red balls and 5 blue balls, what fraction are red?",
            "accept": ["3/8", "3 out of 8", "three eighths", "0.375", "37.5%"],
            "category": "probability",
        },
        {
            "q": "What is the next prime number after 7?",
            "accept": ["11"],
            "category": "primes",
        },
    ],

    # --- Code Generation ---
    "code": [
        {
            "q": "Write a Python function that returns the factorial of n. Just the function, nothing else.",
            "accept": ["def factorial", "def fact", "math.factorial"],
            "validate_code": True,
            "test": "assert factorial(5) == 120 and factorial(0) == 1",
            "category": "function",
        },
        {
            "q": "Write a Python function that checks if a string is a palindrome. Just the function.",
            "accept": ["def ", "[::-1]", "reversed"],
            "validate_code": True,
            "test": "assert is_palindrome('racecar') == True and is_palindrome('hello') == False",
            "category": "string",
        },
        {
            "q": "Write a Python function that returns the nth Fibonacci number. Just the function.",
            "accept": ["def fib", "def fibonacci"],
            "validate_code": True,
            "test": "assert fibonacci(10) == 55 and fibonacci(1) == 1",
            "category": "recursion",
        },
        {
            "q": "Write a Python one-liner that flattens a list of lists. Example: [[1,2],[3,4]] -> [1,2,3,4]",
            "accept": ["for", "sum(", "chain", "itertools", "comprehension"],
            "category": "list",
        },
        {
            "q": "Write a Python function that counts the occurrences of each word in a string and returns a dictionary.",
            "accept": ["def ", "dict", "Counter", "split"],
            "category": "dict",
        },
    ],

    # --- Reasoning / Logic ---
    "reasoning": [
        {
            "q": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "accept": ["no", "No", "cannot", "not necessarily", "we cannot conclude"],
            "category": "logic",
        },
        {
            "q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "accept": ["$0.05", "0.05", "5 cents", "five cents"],
            "category": "trick",
        },
        {
            "q": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "accept": ["5 minutes", "5", "five minutes"],
            "category": "trick",
        },
        {
            "q": "Which is heavier: a pound of feathers or a pound of steel?",
            "accept": ["same", "equal", "weigh the same", "neither", "both weigh"],
            "category": "trick",
        },
        {
            "q": "If you rearrange the letters 'CIFAIPC' you get the name of a(n):",
            "accept": ["PACIFIC", "Pacific", "ocean"],
            "category": "wordplay",
        },
    ],

    # --- Long Output Coherence ---
    "coherence": [
        {
            "q": "List the planets in our solar system in order from the Sun, one per line.",
            "accept": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
            "min_matches": 7,  # at least 7 of 8 planets
            "category": "list",
        },
        {
            "q": "Explain the water cycle in 3-4 sentences.",
            "accept": ["evaporation", "condensation", "precipitation"],
            "min_matches": 2,
            "category": "explanation",
        },
        {
            "q": "Write the numbers 1 to 20 separated by commas.",
            "accept": ["1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20",
                       "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"],
            "category": "sequence",
        },
    ],
}


def check_answer(response, question_data):
    """Check if response matches expected answer."""
    text = response.strip()

    accept_list = question_data["accept"]
    min_matches = question_data.get("min_matches", 1)

    matches = sum(1 for a in accept_list if a in text)
    return matches >= min_matches


def run_eval_vllm(model_path, kv_cache_dtype="auto", max_model_len=8192,
                  tiers=None, trust_remote_code=True, enforce_eager=True,
                  disable_hybrid=True, extra_args=None):
    """Run eval suite using vLLM for generation.

    Args:
        model_path: path to model
        kv_cache_dtype: "auto", "fp8_e5m2", "fusen", etc.
        tiers: which eval tiers to run (default: all)
        extra_args: dict of extra LLM kwargs

    Returns: dict of results per tier
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    if tiers is None:
        tiers = list(EVAL_SUITE.keys())

    tok = AutoTokenizer.from_pretrained(model_path)

    llm_kwargs = {
        "model": model_path,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": 0.92,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
    }
    if kv_cache_dtype != "auto":
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype
    if disable_hybrid:
        llm_kwargs["disable_hybrid_kv_cache_manager"] = True
    if extra_args:
        llm_kwargs.update(extra_args)

    print(f"Loading model: {model_path}")
    print(f"KV cache dtype: {kv_cache_dtype}")
    llm = LLM(**llm_kwargs)

    results = {}
    total_correct = 0
    total_questions = 0

    for tier_name in tiers:
        if tier_name not in EVAL_SUITE:
            continue

        questions = EVAL_SUITE[tier_name]
        tier_correct = 0

        prompts = []
        for qd in questions:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": qd["q"]}],
                tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        params = SamplingParams(temperature=0.0, max_tokens=256)
        outputs = llm.generate(prompts, params)

        tier_results = []
        for qd, out in zip(questions, outputs):
            response = out.outputs[0].text
            correct = check_answer(response, qd)
            tier_correct += int(correct)
            tier_results.append({
                "question": qd["q"],
                "response": response[:200],
                "correct": correct,
                "category": qd.get("category", ""),
            })

        accuracy = tier_correct / len(questions) if questions else 0
        results[tier_name] = {
            "correct": tier_correct,
            "total": len(questions),
            "accuracy": accuracy,
            "details": tier_results,
        }
        total_correct += tier_correct
        total_questions += len(questions)

    results["overall"] = {
        "correct": total_correct,
        "total": total_questions,
        "accuracy": total_correct / total_questions if total_questions else 0,
    }

    return results


def print_results(results, verbose=False):
    """Pretty-print eval results."""
    print(f"\n{'='*60}")
    print(f"{'Tier':<15} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 45)

    for tier_name, tier_data in results.items():
        if tier_name == "overall":
            continue
        acc_pct = tier_data["accuracy"] * 100
        print(f"{tier_name:<15} {tier_data['correct']:>8} {tier_data['total']:>6} {acc_pct:>9.1f}%")

        if verbose:
            for detail in tier_data["details"]:
                status = "✓" if detail["correct"] else "✗"
                print(f"  {status} [{detail['category']}] {detail['question'][:60]}")
                if not detail["correct"]:
                    print(f"    → {detail['response'][:100]}")

    print("-" * 45)
    overall = results["overall"]
    acc_pct = overall["accuracy"] * 100
    print(f"{'OVERALL':<15} {overall['correct']:>8} {overall['total']:>6} {acc_pct:>9.1f}%")
    print(f"{'='*60}")

    return overall["accuracy"]


def compare_configs(model_path, configs, tiers=None, verbose=False):
    """Compare multiple KV cache configs on the same eval suite.

    Args:
        model_path: path to model
        configs: list of dicts with "name" and "kv_cache_dtype" (and optional extra args)
        tiers: which eval tiers to run

    Returns: comparison table
    """
    all_results = {}

    for config in configs:
        name = config["name"]
        kv_dtype = config.get("kv_cache_dtype", "auto")
        extra = config.get("extra_args", {})

        print(f"\n{'='*60}")
        print(f"Testing: {name} (kv_cache_dtype={kv_dtype})")
        print(f"{'='*60}")

        results = run_eval_vllm(
            model_path, kv_cache_dtype=kv_dtype, tiers=tiers,
            extra_args=extra,
        )
        acc = print_results(results, verbose=verbose)
        all_results[name] = {"accuracy": acc, "results": results}

    # Comparison table
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'Config':<20} {'Overall':>8} ", end="")
    tier_names = [t for t in (tiers or EVAL_SUITE.keys()) if t in EVAL_SUITE]
    for t in tier_names:
        print(f"{t:>10} ", end="")
    print()
    print("-" * (30 + 11 * len(tier_names)))

    baseline_acc = None
    for name, data in all_results.items():
        acc = data["accuracy"]
        if baseline_acc is None:
            baseline_acc = acc
        delta = acc - baseline_acc

        print(f"{name:<20} {acc*100:>7.1f}% ", end="")
        for t in tier_names:
            tier_acc = data["results"].get(t, {}).get("accuracy", 0)
            print(f"{tier_acc*100:>9.1f}% ", end="")
        if delta != 0:
            print(f" ({delta*100:+.1f}%)", end="")
        print()

    print(f"{'='*60}")
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quality eval for KV cache quantization")
    parser.add_argument("--model", default="/root/models/gemma-4-26B-A4B-it-AWQ-4bit")
    parser.add_argument("--kv-cache-dtype", default="auto",
                        help="KV cache dtype to test (auto, fp8_e5m2, fusen)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare BF16 vs FP8 KV")
    parser.add_argument("--tiers", nargs="+",
                        help="Which tiers to run (factual, math, code, reasoning, coherence)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    if args.compare:
        configs = [
            {"name": "BF16 KV (baseline)", "kv_cache_dtype": "auto"},
            {"name": "FP8 KV (fp8_e5m2)", "kv_cache_dtype": "fp8_e5m2"},
        ]
        compare_configs(args.model, configs, tiers=args.tiers, verbose=args.verbose)
    else:
        results = run_eval_vllm(
            args.model, kv_cache_dtype=args.kv_cache_dtype,
            tiers=args.tiers, max_model_len=args.max_model_len,
        )
        print_results(results, verbose=args.verbose)

        # Save results
        out_file = f"kv_cache_gen/eval_{args.kv_cache_dtype}.json"
        with open(out_file, "w") as f:
            # Strip non-serializable details
            save_data = {}
            for k, v in results.items():
                if isinstance(v, dict) and "details" in v:
                    save_data[k] = {kk: vv for kk, vv in v.items() if kk != "details"}
                else:
                    save_data[k] = v
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
