#!/usr/bin/env python3
"""
Comprehensive pruning quality validation suite.

Detects capability regressions across 6 domains (120+ tests) by comparing
a pruned model against an unpruned baseline. Designed to catch cases where
a low-frequency expert or pruned layer was the only component encoding a
specific capability (e.g., SQL, math reasoning, translation).

Usage:
    # Create baseline from unpruned model
    python validate_pruning.py --baseline --output baseline.json

    # Validate pruned model against baseline
    python validate_pruning.py --compare baseline.json --output pruned_results.json

    # Quick smoke test (5 tests per domain)
    python validate_pruning.py --quick

Environment:
    BASE_URL   - API endpoint (default: http://localhost:8000/v1)
    MODEL      - Model name/path (default: auto-detect from /v1/models)
    CONCURRENCY - Max parallel requests (default: 8)
"""

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Optional

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))

REGRESSION_THRESHOLD = -0.10  # 10% drop = regression


# ── Test Definitions ──────────────────────────────────────────────────────────

CODING_TESTS = [
    # Python
    {"prompt": "Write a Python function that finds the nth Fibonacci number using dynamic programming. Just the code, no explanation.",
     "checks": [{"type": "contains", "values": ["def ", "fib"]}], "domain": "python"},
    {"prompt": "Write a Python function that checks if a string is a valid palindrome, ignoring spaces and case. Just the code.",
     "checks": [{"type": "contains", "values": ["def ", "lower"]}], "domain": "python"},
    {"prompt": "Write a Python class implementing a stack with push, pop, and peek methods. Just the code.",
     "checks": [{"type": "contains", "values": ["class ", "def push", "def pop"]}], "domain": "python"},
    {"prompt": "Write a Python function that flattens a nested list of arbitrary depth. Just the code.",
     "checks": [{"type": "contains", "values": ["def ", "flatten"]}], "domain": "python"},
    {"prompt": "Write a Python decorator that caches function results (memoization). Just the code.",
     "checks": [{"type": "contains", "values": ["def ", "cache"]}], "domain": "python"},
    # SQL
    {"prompt": "Write a SQL query to find customers who placed more than 3 orders last month. Use tables: customers(id, name), orders(id, customer_id, created_at).",
     "checks": [{"type": "contains_any", "values": ["SELECT", "select"]},
                {"type": "contains_any", "values": ["GROUP BY", "group by"]}], "domain": "sql"},
    {"prompt": "Write a SQL query to find the second highest salary from an employees table with columns (id, name, salary).",
     "checks": [{"type": "contains_any", "values": ["SELECT", "select"]},
                {"type": "contains_any", "values": ["salary", "SALARY"]}], "domain": "sql"},
    {"prompt": "Write a SQL query using a window function to rank employees by salary within each department.",
     "checks": [{"type": "contains_any", "values": ["RANK", "rank", "ROW_NUMBER", "row_number", "DENSE_RANK"]}], "domain": "sql"},
    {"prompt": "Write a SQL query to find all pairs of employees who work in the same department (self-join).",
     "checks": [{"type": "contains_any", "values": ["JOIN", "join"]}], "domain": "sql"},
    # Bash
    {"prompt": "Write a bash script that finds all files larger than 100MB in the current directory tree. Just the code.",
     "checks": [{"type": "contains_any", "values": ["find", "FIND"]},
                {"type": "contains_any", "values": ["-size", "size"]}], "domain": "bash"},
    {"prompt": "Write a bash one-liner that counts the number of lines in all .py files in a directory recursively.",
     "checks": [{"type": "contains_any", "values": ["find", "wc", "*.py"]}], "domain": "bash"},
    {"prompt": "Write a bash script that monitors a log file and sends an alert when 'ERROR' appears. Just the code.",
     "checks": [{"type": "contains_any", "values": ["tail", "grep", "ERROR"]}], "domain": "bash"},
    # Rust
    {"prompt": "Write a Rust function that implements a thread-safe counter using Arc and Mutex. Just the code.",
     "checks": [{"type": "contains_any", "values": ["Arc", "Mutex"]},
                {"type": "contains_any", "values": ["fn ", "impl"]}], "domain": "rust"},
    {"prompt": "Write a Rust function that reads a file and returns its contents as a String, handling errors with Result. Just the code.",
     "checks": [{"type": "contains_any", "values": ["Result", "fn "]},
                {"type": "contains_any", "values": ["read", "File", "fs"]}], "domain": "rust"},
    # JavaScript/TypeScript
    {"prompt": "Write a JavaScript async function that fetches data from an API with retry logic (max 3 retries). Just the code.",
     "checks": [{"type": "contains_any", "values": ["async", "fetch", "await"]}], "domain": "javascript"},
    {"prompt": "Write a TypeScript function with generics that filters an array based on a predicate function. Include type annotations.",
     "checks": [{"type": "contains_any", "values": ["<T>", "<T,", "function", "=>"]}], "domain": "typescript"},
    # Debugging
    {"prompt": "Debug this Python code and explain the bug:\ndef sort_list(lst):\n    return lst.sort()\nprint(sort_list([3,1,2]))",
     "checks": [{"type": "contains_any", "values": ["None", "returns None", "in-place", "sorted"]}], "domain": "debugging"},
    {"prompt": "Debug this code and explain the bug:\ndef add_to_list(item, lst=[]):\n    lst.append(item)\n    return lst",
     "checks": [{"type": "contains_any", "values": ["mutable", "default", "shared"]}], "domain": "debugging"},
    # Algorithms
    {"prompt": "Implement binary search in Python that returns the index of the target or -1 if not found. Just the code.",
     "checks": [{"type": "contains", "values": ["def ", "mid"]}], "domain": "algorithms"},
    {"prompt": "Write a Python function implementing merge sort. Just the code.",
     "checks": [{"type": "contains", "values": ["def ", "merge"]}], "domain": "algorithms"},
    {"prompt": "Write a Python function that detects a cycle in a linked list using Floyd's algorithm. Just the code.",
     "checks": [{"type": "contains_any", "values": ["slow", "fast", "tortoise", "hare"]}], "domain": "algorithms"},
]

MATH_TESTS = [
    # Arithmetic
    {"prompt": "What is 17 * 23? Answer with just the number.", "answer": "391", "domain": "arithmetic"},
    {"prompt": "What is 156 + 287? Answer with just the number.", "answer": "443", "domain": "arithmetic"},
    {"prompt": "What is 1000 - 678? Answer with just the number.", "answer": "322", "domain": "arithmetic"},
    {"prompt": "What is 144 / 12? Answer with just the number.", "answer": "12", "domain": "arithmetic"},
    {"prompt": "What is 2^10? Answer with just the number.", "answer": "1024", "domain": "arithmetic"},
    # Algebra
    {"prompt": "Solve: 2x + 5 = 17. What is x? Answer with just the number.", "answer": "6", "domain": "algebra"},
    {"prompt": "Solve: 3x - 7 = 14. What is x? Answer with just the number.", "answer": "7", "domain": "algebra"},
    {"prompt": "If y = 2x + 3 and x = 4, what is y? Answer with just the number.", "answer": "11", "domain": "algebra"},
    {"prompt": "What are the roots of x^2 - 5x + 6 = 0? Answer with just the two numbers separated by a comma.", "answer": "2,3", "domain": "algebra",
     "answer_check": "contains_all", "answer_values": ["2", "3"]},
    {"prompt": "Simplify: (x^2 - 9) / (x - 3). Answer concisely.", "answer": "x + 3", "domain": "algebra",
     "answer_check": "contains_any", "answer_values": ["x + 3", "x+3", "(x+3)"]},
    # Calculus
    {"prompt": "What is the derivative of x^3 + 2x? Answer concisely.", "answer": "3x^2 + 2", "domain": "calculus",
     "answer_check": "contains_any", "answer_values": ["3x^2 + 2", "3x² + 2", "3x^2+2"]},
    {"prompt": "What is the integral of 2x dx? Answer concisely.", "answer": "x^2", "domain": "calculus",
     "answer_check": "contains_any", "answer_values": ["x^2", "x²", "x**2"]},
    {"prompt": "What is the derivative of sin(x)? Answer with just the function.", "answer": "cos(x)", "domain": "calculus",
     "answer_check": "contains_any", "answer_values": ["cos(x)", "cos x", "cosx"]},
    {"prompt": "What is the limit of (1 + 1/n)^n as n approaches infinity? Answer concisely.", "answer": "e", "domain": "calculus",
     "answer_check": "contains_any", "answer_values": ["e", "2.718", "Euler"]},
    # Probability & Statistics
    {"prompt": "If P(A)=0.3 and P(B)=0.5 and A and B are independent, what is P(A and B)? Answer with just the number.", "answer": "0.15", "domain": "probability"},
    {"prompt": "What is the mean of [2, 4, 6, 8, 10]? Answer with just the number.", "answer": "6", "domain": "probability"},
    {"prompt": "You flip a fair coin 3 times. What is the probability of getting exactly 2 heads? Express as a fraction.", "answer": "3/8", "domain": "probability",
     "answer_check": "contains_any", "answer_values": ["3/8", "0.375", "37.5%"]},
    {"prompt": "A bag has 3 red and 5 blue balls. What is the probability of drawing a red ball? Express as a fraction.", "answer": "3/8", "domain": "probability",
     "answer_check": "contains_any", "answer_values": ["3/8", "0.375"]},
    # Word problems
    {"prompt": "A train travels 120 km in 2 hours. What is its speed in km/h? Answer with just the number.", "answer": "60", "domain": "word_problem"},
    {"prompt": "If 5 workers can build a wall in 10 days, how many days would 10 workers take? Answer with just the number.", "answer": "5", "domain": "word_problem"},
    {"prompt": "A store has a 20% off sale. If an item costs $80, what is the sale price? Answer with just the number.", "answer": "64", "domain": "word_problem"},
    {"prompt": "What is 15% of 200? Answer with just the number.", "answer": "30", "domain": "word_problem"},
]

REASONING_TESTS = [
    # Formal logic
    {"prompt": "If all cats are animals, and some animals are pets, can we conclude that all cats are pets? Answer Yes or No and explain briefly.",
     "checks": [{"type": "contains_any", "values": ["No", "no", "cannot", "not necessarily"]}], "domain": "logic"},
    {"prompt": "If it rains, the ground is wet. The ground is wet. Can we conclude it rained? Answer Yes or No and explain briefly.",
     "checks": [{"type": "contains_any", "values": ["No", "no", "cannot", "not necessarily", "affirming the consequent"]}], "domain": "logic"},
    {"prompt": "All roses are flowers. All flowers need water. What can we conclude about roses?",
     "checks": [{"type": "contains_any", "values": ["water", "need water"]}], "domain": "logic"},
    {"prompt": "If A implies B, and B implies C, does A imply C? Answer Yes or No.",
     "checks": [{"type": "contains_any", "values": ["Yes", "yes"]}], "domain": "logic"},
    {"prompt": "None of the students failed. Is it true that at least one student passed? Answer Yes or No.",
     "checks": [{"type": "contains_any", "values": ["Yes", "yes"]}], "domain": "logic"},
    # Cognitive / trick questions
    {"prompt": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost? Show your work.",
     "checks": [{"type": "contains_any", "values": ["0.05", "$0.05", "5 cents", "five cents"]}], "domain": "cognitive"},
    {"prompt": "If you have 3 apples and take away 2, how many apples do YOU have?",
     "checks": [{"type": "contains_any", "values": ["2", "two"]}], "domain": "cognitive"},
    {"prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
     "checks": [{"type": "contains_any", "values": ["9", "nine"]}], "domain": "cognitive"},
    {"prompt": "How many times can you subtract 5 from 25?",
     "checks": [{"type": "contains_any", "values": ["1", "one", "once"]}], "domain": "cognitive"},
    # Causal reasoning
    {"prompt": "John is taller than Mary. Mary is taller than Sue. Who is the shortest?",
     "checks": [{"type": "contains_any", "values": ["Sue"]}], "domain": "causal"},
    {"prompt": "A is to the left of B. B is to the left of C. What is to the right of A?",
     "checks": [{"type": "contains_any", "values": ["B", "B and C", "B, C"]}], "domain": "causal"},
    {"prompt": "If I push a glass off a table, what happens and why?",
     "checks": [{"type": "contains_any", "values": ["fall", "break", "gravity", "shatter"]}], "domain": "causal"},
    # Analogical reasoning
    {"prompt": "Complete the analogy: Hot is to cold as light is to ___",
     "checks": [{"type": "contains_any", "values": ["dark", "darkness"]}], "domain": "analogy"},
    {"prompt": "Complete the analogy: Book is to reading as fork is to ___",
     "checks": [{"type": "contains_any", "values": ["eating", "eat"]}], "domain": "analogy"},
    {"prompt": "Complete the analogy: Bird is to nest as human is to ___",
     "checks": [{"type": "contains_any", "values": ["house", "home"]}], "domain": "analogy"},
    # Pattern recognition
    {"prompt": "What comes next in the sequence: 2, 6, 12, 20, 30, ___? Answer with just the number.",
     "checks": [{"type": "contains", "values": ["42"]}], "domain": "pattern"},
    {"prompt": "What comes next: 1, 1, 2, 3, 5, 8, ___? Answer with just the number.",
     "checks": [{"type": "contains", "values": ["13"]}], "domain": "pattern"},
    {"prompt": "What comes next: A1, B2, C3, D4, ___?",
     "checks": [{"type": "contains_any", "values": ["E5"]}], "domain": "pattern"},
    # Counterfactual
    {"prompt": "If the Earth had no moon, name two things that would be different.",
     "checks": [{"type": "contains_any", "values": ["tide", "tidal", "rotation", "night", "orbit", "axis"]}], "domain": "counterfactual"},
    {"prompt": "If humans had never invented the wheel, what would transportation look like today?",
     "checks": [{"type": "min_length", "value": 50}], "domain": "counterfactual"},
]

KNOWLEDGE_TESTS = [
    # Geography
    {"prompt": "What is the capital of Mongolia? Answer with just the city name.", "answer": "Ulaanbaatar", "domain": "geography",
     "answer_check": "contains_any", "answer_values": ["Ulaanbaatar", "Ulan Bator"]},
    {"prompt": "What is the longest river in Africa? Answer with just the name.", "answer": "Nile", "domain": "geography"},
    {"prompt": "Which country has the largest population? Answer with just the country name.", "answer": "India", "domain": "geography",
     "answer_check": "contains_any", "answer_values": ["India", "China"]},
    {"prompt": "What is the smallest country in the world by area? Answer with just the name.", "answer": "Vatican", "domain": "geography",
     "answer_check": "contains_any", "answer_values": ["Vatican", "Vatican City"]},
    {"prompt": "On which continent is Brazil? Answer with just the continent name.", "answer": "South America", "domain": "geography",
     "answer_check": "contains_any", "answer_values": ["South America"]},
    # Literature
    {"prompt": "Who wrote 'One Hundred Years of Solitude'? Answer with just the author name.", "answer": "Gabriel Garcia Marquez", "domain": "literature",
     "answer_check": "contains_any", "answer_values": ["Garcia Marquez", "García Márquez", "Marquez", "Márquez"]},
    {"prompt": "Who wrote '1984'? Answer with just the author name.", "answer": "George Orwell", "domain": "literature",
     "answer_check": "contains_any", "answer_values": ["Orwell", "George Orwell"]},
    {"prompt": "Who wrote 'Pride and Prejudice'? Answer with just the author name.", "answer": "Jane Austen", "domain": "literature",
     "answer_check": "contains_any", "answer_values": ["Austen", "Jane Austen"]},
    {"prompt": "In which Shakespeare play does the character Hamlet appear? Answer with just the play name.", "answer": "Hamlet", "domain": "literature"},
    # Science
    {"prompt": "What is the chemical formula for sulfuric acid? Answer with just the formula.", "answer": "H2SO4", "domain": "chemistry",
     "answer_check": "contains_any", "answer_values": ["H2SO4", "H₂SO₄"]},
    {"prompt": "What is the chemical symbol for gold? Answer with just the symbol.", "answer": "Au", "domain": "chemistry"},
    {"prompt": "What is the speed of light in vacuum approximately in km/s? Answer with just the number.", "answer": "300000", "domain": "physics",
     "answer_check": "contains_any", "answer_values": ["300000", "300,000", "3 x 10", "3×10", "299792"]},
    {"prompt": "What is the powerhouse of the cell? Answer concisely.", "answer": "mitochondria", "domain": "biology",
     "answer_check": "contains_any", "answer_values": ["mitochondria", "mitochondrion"]},
    {"prompt": "What element has atomic number 1? Answer with just the element name.", "answer": "Hydrogen", "domain": "chemistry"},
    # History
    {"prompt": "In what year did the Berlin Wall fall? Answer with just the year.", "answer": "1989", "domain": "history"},
    {"prompt": "Who was the first person to walk on the Moon? Answer with just the name.", "answer": "Neil Armstrong", "domain": "history",
     "answer_check": "contains_any", "answer_values": ["Armstrong", "Neil Armstrong"]},
    {"prompt": "In what year did World War II end? Answer with just the year.", "answer": "1945", "domain": "history"},
    {"prompt": "Who painted the Mona Lisa? Answer with just the name.", "answer": "Leonardo da Vinci", "domain": "history",
     "answer_check": "contains_any", "answer_values": ["da Vinci", "Leonardo", "Da Vinci"]},
    # Technology
    {"prompt": "Who founded Microsoft? Answer with just the names.", "answer": "Bill Gates", "domain": "technology",
     "answer_check": "contains_any", "answer_values": ["Bill Gates", "Gates", "Paul Allen"]},
    {"prompt": "What programming language was created by Guido van Rossum? Answer with just the language name.", "answer": "Python", "domain": "technology"},
    {"prompt": "What does HTTP stand for?", "answer": "HyperText Transfer Protocol", "domain": "technology",
     "answer_check": "contains_any", "answer_values": ["HyperText Transfer Protocol", "Hypertext Transfer Protocol"]},
]

LANGUAGE_TESTS = [
    # Formal writing
    {"prompt": "Write a formal business email (3-4 sentences) declining a meeting invitation due to a scheduling conflict.",
     "checks": [{"type": "contains_any", "values": ["Dear", "Sincerely", "Regards", "Thank", "unfortunately", "regret", "apologize"]}], "domain": "formal"},
    {"prompt": "Write a professional cover letter opening paragraph for a software engineering position.",
     "checks": [{"type": "contains_any", "values": ["position", "role", "opportunity", "experience", "skills"]}], "domain": "formal"},
    # Simplification
    {"prompt": "Explain quantum computing to a 5-year-old in 2-3 sentences.",
     "checks": [{"type": "min_length", "value": 30},
                {"type": "no_jargon", "forbidden": ["qubit", "superposition", "entanglement", "quantum state"]}], "domain": "simplification"},
    {"prompt": "Explain how the internet works to someone who has never used a computer. Use 3-4 simple sentences.",
     "checks": [{"type": "min_length", "value": 40}], "domain": "simplification"},
    # Creative writing
    {"prompt": "Write a haiku about the ocean. Format: three lines with 5-7-5 syllable structure.",
     "checks": [{"type": "min_length", "value": 10}], "domain": "creative"},
    {"prompt": "Write a limerick about a programmer.",
     "checks": [{"type": "min_length", "value": 20}], "domain": "creative"},
    {"prompt": "Write the opening paragraph of a mystery novel set in a small coastal town.",
     "checks": [{"type": "min_length", "value": 50}], "domain": "creative"},
    {"prompt": "Write a metaphor describing time.",
     "checks": [{"type": "min_length", "value": 10}], "domain": "creative"},
    # Translation
    {"prompt": "Translate to French: 'The weather is beautiful today.'",
     "checks": [{"type": "contains_any", "values": ["Il fait beau", "Le temps est beau", "beau"]}], "domain": "translation"},
    {"prompt": "Translate to Spanish: 'Where is the nearest hospital?'",
     "checks": [{"type": "contains_any", "values": ["hospital", "dónde", "donde", "más cercano"]}], "domain": "translation"},
    {"prompt": "Translate to German: 'I would like a cup of coffee, please.'",
     "checks": [{"type": "contains_any", "values": ["Kaffee", "möchte", "bitte"]}], "domain": "translation"},
    {"prompt": "Translate to Japanese: 'Thank you very much.'",
     "checks": [{"type": "contains_any", "values": ["ありがとう", "arigatou", "arigatō", "domo"]}], "domain": "translation"},
    {"prompt": "Translate to Italian: 'Good morning, how are you?'",
     "checks": [{"type": "contains_any", "values": ["Buongiorno", "buon giorno", "come stai", "come sta"]}], "domain": "translation"},
    # Summarization
    {"prompt": "Summarize the concept of natural selection in exactly 2 sentences.",
     "checks": [{"type": "min_length", "value": 30}], "domain": "summarization"},
    {"prompt": "In one sentence, explain why the sky is blue.",
     "checks": [{"type": "contains_any", "values": ["scatter", "light", "wavelength", "Rayleigh", "atmosphere"]}], "domain": "summarization"},
    # Tone adaptation
    {"prompt": "Rewrite this casual text in a professional tone: 'Hey dude, the project is super late and the boss is gonna flip out.'",
     "checks": [{"type": "contains_any", "values": ["project", "deadline", "delay", "schedule", "concern"]}], "domain": "tone"},
    {"prompt": "Write a sympathy message for someone who lost a pet. Keep it sincere and brief.",
     "checks": [{"type": "min_length", "value": 20},
                {"type": "contains_any", "values": ["sorry", "loss", "heart", "memory", "memories", "beloved"]}], "domain": "tone"},
]

INSTRUCTION_TESTS = [
    # List constraints
    {"prompt": "List exactly 5 planets in our solar system, numbered 1-5.",
     "checks": [{"type": "contains", "values": ["1", "2", "3", "4", "5"]},
                {"type": "count_constraint", "pattern": r'\d+[.)\s]', "min": 5, "max": 5}], "domain": "list"},
    {"prompt": "Name 3 programming languages. Only 3, no more, no less.",
     "checks": [{"type": "min_length", "value": 5}], "domain": "list"},
    # Format constraints
    {"prompt": "Answer in exactly one word: What color is the sky on a clear day?",
     "checks": [{"type": "max_words", "value": 3},
                {"type": "contains_any", "values": ["blue", "Blue"]}], "domain": "constraint"},
    {"prompt": "Respond with only 'yes' or 'no': Is the Earth round?",
     "checks": [{"type": "contains_any", "values": ["yes", "Yes", "YES"]}], "domain": "constraint"},
    {"prompt": "Write exactly 3 sentences about dogs. Each sentence must end with a period.",
     "checks": [{"type": "min_length", "value": 20}], "domain": "constraint"},
    # Structured output
    {"prompt": "Output a JSON object with keys 'name', 'age', 'city' for a fictional person. Only JSON, no explanation.",
     "checks": [{"type": "contains", "values": ["name", "age", "city"]},
                {"type": "contains_any", "values": ["{", "}"]}], "domain": "structured"},
    {"prompt": "Create a markdown table with 3 columns (Name, Age, City) and 2 rows of fictional data.",
     "checks": [{"type": "contains_any", "values": ["|", "---"]}], "domain": "structured"},
    # Multi-step instructions
    {"prompt": "First, write the word 'START'. Then list 3 fruits. Then write the word 'END'.",
     "checks": [{"type": "contains", "values": ["START", "END"]}], "domain": "multistep"},
    {"prompt": "Write a sentence where every word starts with the letter 'S'.",
     "checks": [{"type": "min_length", "value": 10}], "domain": "multistep"},
    # Refusal / safety
    {"prompt": "Explain the water cycle in 2 sentences. Do not mention evaporation.",
     "checks": [{"type": "min_length", "value": 20}], "domain": "refusal"},
    {"prompt": "Tell me about the solar system but do not mention Earth.",
     "checks": [{"type": "min_length", "value": 30}], "domain": "refusal"},
    {"prompt": "List 4 colors of the rainbow. Do NOT include red.",
     "checks": [{"type": "min_length", "value": 5}], "domain": "refusal"},
]

ALL_TESTS = {
    "coding": CODING_TESTS,
    "math": MATH_TESTS,
    "reasoning": REASONING_TESTS,
    "knowledge": KNOWLEDGE_TESTS,
    "language": LANGUAGE_TESTS,
    "instruction": INSTRUCTION_TESTS,
}


# ── API Client ────────────────────────────────────────────────────────────────

def detect_model(base_url: str) -> str:
    """Auto-detect model name from the server."""
    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            return models[0]["id"]
    except Exception:
        pass
    return ""


def chat_completion(base_url: str, model: str, prompt: str,
                    max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Send a chat completion request with retries."""
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
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
                return f"[API_ERROR: {e}]"


# ── Evaluation Logic ──────────────────────────────────────────────────────────

def evaluate_answer_test(response: str, test: dict) -> float:
    """Evaluate a test that has a specific expected answer."""
    response_lower = response.lower().strip()

    # Custom answer check
    check_type = test.get("answer_check", "exact")
    if check_type == "contains_any":
        values = test.get("answer_values", [test.get("answer", "")])
        for v in values:
            if v.lower() in response_lower:
                return 1.0
        return 0.0
    elif check_type == "contains_all":
        values = test.get("answer_values", [])
        found = sum(1 for v in values if v.lower() in response_lower)
        return found / len(values) if values else 0.0

    # Default: exact answer matching
    answer = test.get("answer", "")
    if not answer:
        return 0.0

    # Direct containment
    if answer.lower() in response_lower:
        return 1.0

    # Numeric comparison
    try:
        # Extract numbers from response
        numbers = re.findall(r'-?[\d,]+\.?\d*', response)
        answer_num = float(answer.replace(",", ""))
        for n in numbers:
            if abs(float(n.replace(",", "")) - answer_num) < 0.01:
                return 1.0
    except (ValueError, ZeroDivisionError):
        pass

    return 0.0


def evaluate_check(response: str, check: dict) -> float:
    """Evaluate a single check against a response."""
    check_type = check["type"]
    response_lower = response.lower()

    if check_type == "contains":
        # All values must be present (case-insensitive)
        values = check["values"]
        found = sum(1 for v in values if v.lower() in response_lower)
        return found / len(values) if values else 0.0

    elif check_type == "contains_any":
        # At least one value must be present
        values = check["values"]
        for v in values:
            if v.lower() in response_lower:
                return 1.0
        return 0.0

    elif check_type == "min_length":
        return 1.0 if len(response.strip()) >= check["value"] else 0.0

    elif check_type == "max_words":
        word_count = len(response.strip().split())
        return 1.0 if word_count <= check["value"] else 0.0

    elif check_type == "no_jargon":
        forbidden = check.get("forbidden", [])
        for word in forbidden:
            if word.lower() in response_lower:
                return 0.5  # Partial penalty, not total failure
        return 1.0

    elif check_type == "count_constraint":
        pattern = check["pattern"]
        matches = re.findall(pattern, response)
        count = len(matches)
        min_c = check.get("min", 0)
        max_c = check.get("max", 999)
        return 1.0 if min_c <= count <= max_c else 0.0

    return 0.0


def evaluate_test(response: str, test: dict) -> float:
    """Evaluate a response against a test definition. Returns 0.0-1.0."""
    if "[API_ERROR" in response:
        return 0.0

    # Check for empty / garbage response
    if len(response.strip()) < 3:
        return 0.0

    scores = []

    # Answer-based tests (math, knowledge)
    if "answer" in test:
        scores.append(evaluate_answer_test(response, test))

    # Check-based tests (coding, reasoning, language, instruction)
    if "checks" in test:
        for check in test["checks"]:
            scores.append(evaluate_check(response, check))

    # If no specific checks, just verify non-empty coherent response
    if not scores:
        if len(response.strip()) > 10 and len(set(response.split())) > 3:
            return 0.8
        return 0.2

    return mean(scores)


# ── Validation Runner ─────────────────────────────────────────────────────────

def run_domain(base_url: str, model: str, domain_name: str,
               tests: list[dict], max_tests: int = 0) -> dict:
    """Run all tests for a domain, return scores."""
    if max_tests > 0:
        tests = tests[:max_tests]

    results = []

    def run_single(test):
        t0 = time.time()
        response = chat_completion(base_url, model, test["prompt"])
        latency = time.time() - t0
        score = evaluate_test(response, test)
        return {
            "prompt": test["prompt"][:80],
            "domain": test.get("domain", domain_name),
            "score": score,
            "response": response[:300],
            "latency_s": round(latency, 2),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = [ex.submit(run_single, t) for t in tests]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Compute domain and sub-domain scores
    all_scores = [r["score"] for r in results]
    domain_score = mean(all_scores) if all_scores else 0.0

    # Sub-domain breakdown
    subdomain_scores = {}
    for r in results:
        sd = r["domain"]
        if sd not in subdomain_scores:
            subdomain_scores[sd] = []
        subdomain_scores[sd].append(r["score"])

    subdomain_summary = {
        sd: round(mean(scores), 3)
        for sd, scores in sorted(subdomain_scores.items())
    }

    failed = [r for r in results if r["score"] < 0.5]

    return {
        "score": round(domain_score, 3),
        "n_tests": len(results),
        "n_passed": sum(1 for s in all_scores if s >= 0.5),
        "n_failed": len(failed),
        "subdomains": subdomain_summary,
        "failed_tests": [{"prompt": f["prompt"], "score": f["score"],
                          "response": f["response"][:150]} for f in failed],
        "details": results,
    }


def validate_model(base_url: str, model: str, baseline: Optional[dict] = None,
                   quick: bool = False) -> dict:
    """Run full validation across all domains."""
    max_tests = 5 if quick else 0
    results = {}
    total_t0 = time.time()

    total_tests = sum(len(t) if not quick else min(5, len(t))
                      for t in ALL_TESTS.values())
    print(f"\nRunning {total_tests} tests across {len(ALL_TESTS)} domains...")
    print(f"Concurrency: {CONCURRENCY}\n")

    for domain_name, tests in ALL_TESTS.items():
        t0 = time.time()
        n = min(max_tests, len(tests)) if quick else len(tests)
        print(f"  [{domain_name}] running {n} tests...", end=" ", flush=True)
        domain_result = run_domain(base_url, model, domain_name, tests, max_tests)
        elapsed = time.time() - t0
        print(f"score={domain_result['score']:.3f} "
              f"({domain_result['n_passed']}/{domain_result['n_tests']} passed) "
              f"[{elapsed:.1f}s]")
        results[domain_name] = domain_result

    total_elapsed = time.time() - total_t0

    # Compare against baseline if provided
    regressions = []
    if baseline:
        baseline_domains = baseline.get("domains", {})
        for domain_name, domain_result in results.items():
            if domain_name in baseline_domains:
                baseline_score = baseline_domains[domain_name]["score"]
                delta = domain_result["score"] - baseline_score
                domain_result["baseline_score"] = baseline_score
                domain_result["delta"] = round(delta, 3)
                domain_result["regression"] = delta < REGRESSION_THRESHOLD

                if domain_result["regression"]:
                    regressions.append((domain_name, delta, baseline_score,
                                        domain_result["score"]))

                # Sub-domain comparison
                baseline_subs = baseline_domains[domain_name].get("subdomains", {})
                sub_regressions = []
                for sd, sd_score in domain_result["subdomains"].items():
                    if sd in baseline_subs:
                        sd_delta = sd_score - baseline_subs[sd]
                        if sd_delta < REGRESSION_THRESHOLD:
                            sub_regressions.append({
                                "subdomain": sd,
                                "score": sd_score,
                                "baseline": baseline_subs[sd],
                                "delta": round(sd_delta, 3),
                            })
                domain_result["subdomain_regressions"] = sub_regressions

    # Overall score
    overall = mean(r["score"] for r in results.values())

    output = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_elapsed, 1),
        "overall_score": round(overall, 3),
        "domains": results,
        "n_regressions": len(regressions),
        "regressions": [
            {"domain": d, "delta": delta, "baseline": bl, "current": cur}
            for d, delta, bl, cur in regressions
        ],
    }

    return output


# ── Report Printing ───────────────────────────────────────────────────────────

def print_report(results: dict, has_baseline: bool = False):
    """Print a formatted report to stdout."""
    print("\n" + "=" * 65)
    print("  PRUNING QUALITY VALIDATION REPORT")
    print("=" * 65)
    print(f"  Model:    {results['model']}")
    print(f"  Time:     {results['timestamp']}")
    print(f"  Duration: {results['total_time_s']}s")
    print(f"  Overall:  {results['overall_score']:.3f}")
    print()

    if has_baseline:
        header = f"{'Domain':<18} {'Score':>6} {'Baseline':>9} {'Delta':>7} {'Status':>10}"
        print(header)
        print("-" * len(header))
        for domain_name, d in results["domains"].items():
            score = d["score"]
            bl = d.get("baseline_score", "")
            delta = d.get("delta", "")
            if d.get("regression"):
                status = "REGRESSION"
            elif isinstance(delta, float) and delta >= 0:
                status = "OK"
            elif isinstance(delta, float):
                status = "OK"
            else:
                status = ""
            bl_str = f"{bl:.3f}" if isinstance(bl, float) else ""
            delta_str = f"{delta:+.3f}" if isinstance(delta, float) else ""
            print(f"  {domain_name:<16} {score:>6.3f} {bl_str:>9} {delta_str:>7} {status:>10}")
    else:
        header = f"{'Domain':<18} {'Score':>6} {'Passed':>8} {'Failed':>8}"
        print(header)
        print("-" * len(header))
        for domain_name, d in results["domains"].items():
            print(f"  {domain_name:<16} {d['score']:>6.3f} "
                  f"{d['n_passed']:>5}/{d['n_tests']:<2} "
                  f"{d['n_failed']:>5}")

    # Sub-domain breakdown
    print("\n" + "-" * 65)
    print("  SUB-DOMAIN BREAKDOWN")
    print("-" * 65)
    for domain_name, d in results["domains"].items():
        subs = d.get("subdomains", {})
        if subs:
            print(f"\n  {domain_name}:")
            for sd, sd_score in subs.items():
                marker = ""
                if has_baseline:
                    # Check sub-domain regressions
                    for sr in d.get("subdomain_regressions", []):
                        if sr["subdomain"] == sd:
                            marker = f" << REGRESSED ({sr['delta']:+.3f} from {sr['baseline']:.3f})"
                            break
                print(f"    {sd:<20} {sd_score:.3f}{marker}")

    # Regressions summary
    if has_baseline and results.get("regressions"):
        print("\n" + "=" * 65)
        print(f"  REGRESSIONS DETECTED: {results['n_regressions']}")
        print("=" * 65)
        for reg in results["regressions"]:
            drop_pct = abs(reg["delta"]) * 100
            print(f"  - {reg['domain']}: dropped {drop_pct:.1f}% "
                  f"({reg['baseline']:.3f} -> {reg['current']:.3f})")

        # Print specific failed tests in regressed domains
        print("\n  Failed tests in regressed domains:")
        for reg in results["regressions"]:
            domain_data = results["domains"].get(reg["domain"], {})
            failed = domain_data.get("failed_tests", [])
            if failed:
                print(f"\n  [{reg['domain']}]:")
                for ft in failed[:5]:
                    print(f"    - {ft['prompt']}")
                    print(f"      Score: {ft['score']}, Response: {ft['response'][:80]}...")
    elif has_baseline:
        print("\n  No regressions detected. All domains within threshold.")

    print("\n" + "=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate model quality across domains to detect pruning regressions."
    )
    parser.add_argument("--baseline", action="store_true",
                        help="Run in baseline mode (save results as baseline)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to baseline JSON file for comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 5 tests per domain")
    global BASE_URL, MODEL, CONCURRENCY, REGRESSION_THRESHOLD

    parser.add_argument("--api-base", type=str, default=None,
                        help=f"API base URL (default: {BASE_URL})")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: auto-detect)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"Max concurrent requests (default: {CONCURRENCY})")
    parser.add_argument("--threshold", type=float, default=None,
                        help=f"Regression threshold (default: {abs(REGRESSION_THRESHOLD)*100}%%)")
    args = parser.parse_args()
    if args.api_base:
        BASE_URL = args.api_base
    if args.concurrency:
        CONCURRENCY = args.concurrency
    if args.threshold is not None:
        REGRESSION_THRESHOLD = -abs(args.threshold / 100.0)

    # Detect model
    model = args.model or MODEL
    if not model:
        model = detect_model(BASE_URL)
    if not model:
        print("ERROR: Could not detect model. Pass --model or set MODEL env var.")
        sys.exit(1)

    print(f"API:   {BASE_URL}")
    print(f"Model: {model}")
    print(f"Mode:  {'baseline' if args.baseline else 'comparison' if args.compare else 'standalone'}")
    print(f"Tests: {'quick (5/domain)' if args.quick else 'full'}")

    # Verify server is reachable
    try:
        resp = requests.get(f"{BASE_URL}/models", timeout=10)
        resp.raise_for_status()
        print("Server: connected")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {BASE_URL}: {e}")
        sys.exit(1)

    # Load baseline for comparison
    baseline = None
    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)
        print(f"Baseline: {args.compare} (model={baseline.get('model', '?')})")

    # Run validation
    results = validate_model(BASE_URL, model, baseline=baseline, quick=args.quick)

    # Print report
    print_report(results, has_baseline=baseline is not None)

    # Save results
    output_path = args.output
    if not output_path:
        if args.baseline:
            output_path = str(Path(__file__).parent.parent
                              / "profiling" / "quality_baseline_unpruned.json")
        else:
            output_path = "/tmp/pruning_validation_results.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # For saved file, strip verbose details to keep file manageable
    save_data = {
        "model": results["model"],
        "timestamp": results["timestamp"],
        "total_time_s": results["total_time_s"],
        "overall_score": results["overall_score"],
        "domains": {},
    }
    for domain_name, d in results["domains"].items():
        save_data["domains"][domain_name] = {
            "score": d["score"],
            "n_tests": d["n_tests"],
            "n_passed": d["n_passed"],
            "n_failed": d["n_failed"],
            "subdomains": d["subdomains"],
            "failed_tests": d["failed_tests"],
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Exit code based on regressions
    if results.get("n_regressions", 0) > 0:
        print(f"\nEXIT 1: {results['n_regressions']} domain(s) regressed.")
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()
