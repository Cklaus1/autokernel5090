#!/usr/bin/env python3
"""
Quality benchmarks for NVFP4 quantized Gemma4 26B model.
Tests GSM8K (math), MMLU (knowledge), and HumanEval (code) via OpenAI-compatible API.
"""

import json
import re
import time
import os
import sys
import random
import concurrent.futures
from pathlib import Path
from typing import Optional

import requests

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "gemma-4-26B-A4B-it-NVFP4")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "8"))
OUTPUT_DIR = Path(__file__).parent

# Sample sizes (use env vars to override)
GSM8K_N = int(os.environ.get("GSM8K_N", "200"))
MMLU_N = int(os.environ.get("MMLU_N", "200"))
HUMANEVAL_N = int(os.environ.get("HUMANEVAL_N", "50"))


def chat_completion(messages: list[dict], max_tokens: int = 1024,
                    temperature: float = 0.0) -> str:
    """Send chat completion request."""
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                json={
                    "model": MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API error after 3 attempts: {e}", file=sys.stderr)
                return ""


# ── GSM8K ───────────────────────────────────────────────────────────────────
def load_gsm8k(n: int) -> list[dict]:
    """Load GSM8K test set."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        samples = list(ds)
        random.seed(42)
        random.shuffle(samples)
        return samples[:n]
    except Exception as e:
        print(f"Failed to load GSM8K: {e}")
        return []


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numeric answer from GSM8K response."""
    # Look for #### pattern (standard GSM8K format)
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(",", "")
    # Look for "the answer is X" pattern
    m = re.search(r'(?:the\s+)?answer\s+is\s*[:\s]*\$?\s*(-?[\d,]+(?:\.\d+)?)', text, re.I)
    if m:
        return m.group(1).replace(",", "")
    # Look for boxed answer (LaTeX)
    m = re.search(r'\\boxed\{(-?[\d,]+(?:\.\d+)?)\}', text)
    if m:
        return m.group(1).replace(",", "")
    # Last number in the response
    numbers = re.findall(r'(-?[\d,]+(?:\.\d+)?)', text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_gsm8k_gold(answer_text: str) -> str:
    """Extract gold answer from GSM8K answer field."""
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', answer_text)
    if m:
        return m.group(1).replace(",", "")
    return ""


def eval_gsm8k_single(sample: dict) -> dict:
    """Evaluate a single GSM8K sample."""
    question = sample["question"]
    gold = extract_gsm8k_gold(sample["answer"])

    messages = [
        {"role": "user", "content": (
            f"Solve this math problem step by step. "
            f"End your response with 'The answer is [NUMBER]'.\n\n"
            f"Question: {question}"
        )}
    ]

    response = chat_completion(messages, max_tokens=1024)
    predicted = extract_gsm8k_answer(response)

    correct = False
    if predicted and gold:
        try:
            correct = abs(float(predicted) - float(gold)) < 0.01
        except ValueError:
            correct = predicted.strip() == gold.strip()

    return {
        "question": question[:100],
        "gold": gold,
        "predicted": predicted,
        "correct": correct,
        "response_len": len(response),
    }


def run_gsm8k(n: int) -> dict:
    """Run GSM8K benchmark."""
    print(f"\n{'='*60}")
    print(f"GSM8K Math Benchmark (n={n})")
    print(f"{'='*60}")

    samples = load_gsm8k(n)
    if not samples:
        return {"error": "Failed to load dataset"}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = {ex.submit(eval_gsm8k_single, s): i for i, s in enumerate(samples)}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            r = future.result()
            results.append(r)
            if (i + 1) % 20 == 0:
                correct_so_far = sum(1 for r in results if r["correct"])
                print(f"  Progress: {i+1}/{len(samples)}, "
                      f"accuracy so far: {correct_so_far/(i+1)*100:.1f}%")

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0
    print(f"\n  GSM8K Result: {correct}/{len(results)} = {accuracy*100:.1f}%")

    return {
        "task": "gsm8k",
        "n_samples": len(results),
        "n_correct": correct,
        "accuracy": accuracy,
        "accuracy_pct": round(accuracy * 100, 1),
    }


# ── MMLU ────────────────────────────────────────────────────────────────────
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def load_mmlu(n: int) -> list[dict]:
    """Load MMLU test samples across subjects."""
    try:
        from datasets import load_dataset
        all_samples = []
        # Load from multiple subjects to get diverse coverage
        per_subject = max(1, n // len(MMLU_SUBJECTS) + 1)
        for subject in MMLU_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subject, split="test")
                for item in list(ds)[:per_subject]:
                    item["subject"] = subject
                    all_samples.append(item)
            except Exception:
                continue
        random.seed(42)
        random.shuffle(all_samples)
        return all_samples[:n]
    except Exception as e:
        print(f"Failed to load MMLU: {e}")
        return []


def eval_mmlu_single(sample: dict) -> dict:
    """Evaluate a single MMLU sample."""
    question = sample["question"]
    choices = sample["choices"]
    gold_idx = sample["answer"]
    gold_letter = "ABCD"[gold_idx]
    subject = sample.get("subject", "unknown")

    choice_text = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

    messages = [
        {"role": "user", "content": (
            f"Answer the following multiple choice question. "
            f"Reply with ONLY the letter (A, B, C, or D) of the correct answer.\n\n"
            f"Question: {question}\n{choice_text}\n\nAnswer:"
        )}
    ]

    response = chat_completion(messages, max_tokens=32)

    # Extract letter answer
    predicted = None
    response_clean = response.strip()
    # Direct single letter
    if response_clean and response_clean[0] in "ABCD":
        predicted = response_clean[0]
    else:
        # Look for patterns like "The answer is A" or "(A)" or "A."
        m = re.search(r'\b([ABCD])\b', response_clean)
        if m:
            predicted = m.group(1)

    correct = predicted == gold_letter

    return {
        "subject": subject,
        "gold": gold_letter,
        "predicted": predicted,
        "correct": correct,
    }


def run_mmlu(n: int) -> dict:
    """Run MMLU benchmark."""
    print(f"\n{'='*60}")
    print(f"MMLU Knowledge Benchmark (n={n})")
    print(f"{'='*60}")

    samples = load_mmlu(n)
    if not samples:
        return {"error": "Failed to load dataset"}

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = {ex.submit(eval_mmlu_single, s): i for i, s in enumerate(samples)}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            r = future.result()
            results.append(r)
            if (i + 1) % 50 == 0:
                correct_so_far = sum(1 for r in results if r["correct"])
                print(f"  Progress: {i+1}/{len(samples)}, "
                      f"accuracy so far: {correct_so_far/(i+1)*100:.1f}%")

    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0

    # Per-subject breakdown
    subject_results = {}
    for r in results:
        subj = r["subject"]
        if subj not in subject_results:
            subject_results[subj] = {"correct": 0, "total": 0}
        subject_results[subj]["total"] += 1
        if r["correct"]:
            subject_results[subj]["correct"] += 1

    print(f"\n  MMLU Result: {correct}/{len(results)} = {accuracy*100:.1f}%")

    return {
        "task": "mmlu",
        "n_samples": len(results),
        "n_correct": correct,
        "accuracy": accuracy,
        "accuracy_pct": round(accuracy * 100, 1),
        "per_subject": {k: round(v["correct"]/v["total"]*100, 1)
                       for k, v in subject_results.items() if v["total"] > 0},
    }


# ── HumanEval ──────────────────────────────────────────────────────────────
def load_humaneval(n: int) -> list[dict]:
    """Load HumanEval problems."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        samples = list(ds)[:n]
        return samples
    except Exception as e:
        print(f"Failed to load HumanEval: {e}")
        return []


def extract_code(response: str, entry_point: str) -> str:
    """Extract Python code from model response."""
    # Try to find code in markdown blocks
    blocks = re.findall(r'```(?:python)?\n(.*?)```', response, re.DOTALL)
    if blocks:
        # Find the block containing the function definition
        for block in blocks:
            if entry_point in block:
                return block
        return blocks[0]

    # If no code blocks, try to extract the function directly
    lines = response.split("\n")
    code_lines = []
    in_func = False
    for line in lines:
        if f"def {entry_point}" in line:
            in_func = True
        if in_func:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    return response


def eval_humaneval_single(sample: dict) -> dict:
    """Evaluate a single HumanEval problem."""
    prompt = sample["prompt"]
    test_code = sample["test"]
    entry_point = sample["entry_point"]
    task_id = sample["task_id"]

    messages = [
        {"role": "user", "content": (
            f"Complete the following Python function. Return ONLY the complete function, "
            f"no explanations.\n\n```python\n{prompt}```"
        )}
    ]

    response = chat_completion(messages, max_tokens=1024)
    code = extract_code(response, entry_point)

    # Try to execute
    passed = False
    error_msg = ""
    try:
        # Build full test: function + test cases
        # The prompt already has the function signature, we need the completion
        full_code = prompt + "\n" + code if not code.startswith("def ") else code

        # Add test
        exec_code = full_code + "\n\n" + test_code + f"\n\ncheck({entry_point})"

        exec_globals = {}
        exec(exec_code, exec_globals)
        passed = True
    except Exception as e:
        error_msg = str(e)[:200]

    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "passed": passed,
        "error": error_msg if not passed else "",
    }


def run_humaneval(n: int) -> dict:
    """Run HumanEval benchmark."""
    print(f"\n{'='*60}")
    print(f"HumanEval Code Benchmark (n={n})")
    print(f"{'='*60}")

    samples = load_humaneval(n)
    if not samples:
        return {"error": "Failed to load dataset"}

    results = []
    # Run sequentially for code execution safety, but API calls concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = {ex.submit(eval_humaneval_single, s): i for i, s in enumerate(samples)}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            r = future.result()
            results.append(r)
            if (i + 1) % 10 == 0:
                passed_so_far = sum(1 for r in results if r["passed"])
                print(f"  Progress: {i+1}/{len(samples)}, "
                      f"pass@1 so far: {passed_so_far/(i+1)*100:.1f}%")

    passed = sum(1 for r in results if r["passed"])
    pass_rate = passed / len(results) if results else 0
    print(f"\n  HumanEval Result: {passed}/{len(results)} = {pass_rate*100:.1f}% pass@1")

    return {
        "task": "humaneval",
        "n_samples": len(results),
        "n_passed": passed,
        "pass_at_1": pass_rate,
        "pass_at_1_pct": round(pass_rate * 100, 1),
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("NVFP4 Gemma4 26B Quality Benchmarks")
    print(f"Model: {MODEL}")
    print(f"API: {BASE_URL}")
    print("=" * 60)

    # Verify server is up
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=5)
        r.raise_for_status()
        print("Server is running.\n")
    except Exception as e:
        print(f"Server not reachable: {e}")
        sys.exit(1)

    all_results = {}
    t0 = time.time()

    # Run benchmarks
    all_results["gsm8k"] = run_gsm8k(GSM8K_N)
    all_results["mmlu"] = run_mmlu(MMLU_N)
    all_results["humaneval"] = run_humaneval(HUMANEVAL_N)

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for task, res in all_results.items():
        if "error" in res:
            print(f"  {task}: ERROR - {res['error']}")
        elif task == "humaneval":
            print(f"  {task}: {res['pass_at_1_pct']}% pass@1 "
                  f"({res['n_passed']}/{res['n_samples']})")
        else:
            print(f"  {task}: {res['accuracy_pct']}% "
                  f"({res['n_correct']}/{res['n_samples']})")

    print(f"\n  Total time: {elapsed:.0f}s")

    # Save results
    results_path = OUTPUT_DIR / "quality_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": MODEL,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_time_s": round(elapsed, 1),
            "results": all_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    main()
