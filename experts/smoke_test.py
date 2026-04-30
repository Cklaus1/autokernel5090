#!/usr/bin/env python3
"""Smoke-test 10 diverse prompts against a vLLM OpenAI server.

Args:
    python3 smoke_test.py [port] [model_name]

Checks qualitative output: no repetition loops, coherent structure, length.
"""

import json
import re
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8004
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gemma-4-merged-L29"

PROMPTS = [
    ("code_python", "Write a Python function that returns the n-th Fibonacci number using memoization."),
    ("code_sql", "Write a SQL query to find the top 5 customers by total revenue in 2024."),
    ("math_word", "A train leaves Chicago at 60 mph, another leaves Denver at 75 mph. They are 1200 miles apart and head toward each other. When do they meet? Show your work."),
    ("math_algebra", "Solve for x: 3x^2 - 12x + 9 = 0. Show the steps."),
    ("reasoning", "You have 3 coins. One is a fake and weighs less. Using a balance scale, how can you find the fake with one weighing?"),
    ("science", "Explain why the sky appears blue, in 3-4 sentences."),
    ("creative_poem", "Write a 4-line poem about a rainy afternoon."),
    ("creative_story", "Write the opening paragraph of a mystery story set in a lighthouse."),
    ("factual", "What year did the Apollo 11 landing occur, and who were the three astronauts?"),
    ("instruction", "List 5 tips for staying focused while working from home."),
]


def generate(prompt: str, max_tokens: int = 180, temperature: float = 0.2) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    elapsed = time.time() - t0
    text = out["choices"][0]["message"]["content"]
    usage = out.get("usage", {})
    return {
        "text": text,
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "finish_reason": out["choices"][0].get("finish_reason", "?"),
    }


def has_repetition_loop(text: str) -> bool:
    """Flag obvious repetition loops (same 5+ gram repeated >=3 times).

    Also catches single-token spam (e.g. "aaaaa...") and same line repeated 3x+.
    """
    if not text.strip():
        return True
    # single-char repetition of 10+
    if re.search(r"(.)\1{20,}", text):
        return True
    # single-word repetition (same word 4+ times in a row)
    words = re.findall(r"\S+", text)
    for i in range(len(words) - 3):
        if words[i] == words[i+1] == words[i+2] == words[i+3]:
            return True
    # 5-gram repeated 3+ times
    for n in (5, 7, 10):
        seen = {}
        for i in range(len(words) - n):
            gram = tuple(words[i:i+n])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= 3:
                return True
    # same line repeated
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    counts = {}
    for l in lines:
        counts[l] = counts.get(l, 0) + 1
        if counts[l] >= 4 and len(l) > 5:
            return True
    return False


def main():
    results = []
    total_loops = 0
    print(f"Model: {MODEL} @ http://localhost:{PORT}\n")
    for tag, prompt in PROMPTS:
        try:
            r = generate(prompt)
        except Exception as exc:
            print(f"--- {tag} ---")
            print(f"  ERROR: {exc}")
            results.append({"tag": tag, "prompt": prompt, "error": str(exc)})
            total_loops += 1
            continue
        text = r["text"]
        first50 = " ".join(text.split()[:50])
        loop = has_repetition_loop(text)
        if loop:
            total_loops += 1
        status = "LOOP/BAD" if loop else "OK"
        print(f"--- {tag} [{status}] ({r['completion_tokens']} tok, {r['elapsed_s']:.1f}s, finish={r['finish_reason']}) ---")
        print(f"  PROMPT: {prompt}")
        print(f"  OUT: {first50}")
        if len(text.split()) > 50:
            print(f"  ... (+{len(text.split())-50} more words)")
        print()
        results.append({"tag": tag, "prompt": prompt, "text": text,
                        "first50": first50, "loop": loop,
                        "completion_tokens": r["completion_tokens"],
                        "elapsed_s": r["elapsed_s"]})

    print(f"\n=== Summary: {10-total_loops}/10 coherent, {total_loops} problematic ===")
    with open("/tmp/smoke_results_merged.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
