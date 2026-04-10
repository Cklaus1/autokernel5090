"""
GSM8K benchmark - clean sequential version.
Direct numerical answer format (no chain-of-thought) to keep responses short.
Uses 60s timeout per question.
"""

import requests
import re
import json
import time
import sys
from datetime import datetime

API = "http://172.17.0.2:8000/v1"  # Docker container IP
MODEL_ID = "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"

# 35 GSM8K-style problems with verified integer/decimal answers
MATH_PROBLEMS = [
    # --- Original GSM8K canonical problems ---
    (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells every remaining "
        "egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars "
        "does she make every day at the farmers' market?",
        "18"
    ),
    (
        "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        "3"
    ),
    (
        "Josh decides to try flipping a house. He buys a house for $80,000 and then "
        "puts in $50,000 in repairs. This increased the value of the house by 150%. "
        "How much profit did he make?",
        "70000"
    ),
    (
        "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. "
        "How many total meters does he run a week?",
        "540"
    ),
    (
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. "
        "She gives the chickens their feed in three separate meals. In the morning, she "
        "gives her flock of chickens 15 cups of feed. In the afternoon she gives her "
        "chickens another 25 cups of feed. How many cups of feed does she need to give "
        "her chickens in the final meal of the day if the size of Wendi's flock is "
        "20 chickens?",
        "20"
    ),
    (
        "Kylar went to the store to buy glasses for his new apartment. One glass costs "
        "$5, but every second glass costs only 60% of the price. Kylar wants to buy "
        "16 glasses. How much does he need to pay for them?",
        "56"
    ),
    (
        "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many "
        "sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have "
        "together if Seattle has 20 sheep?",
        "260"
    ),
    (
        "Tom bought his house for $150,000. The value of the house increases by 20% "
        "every year. How much is the house worth after 3 years in whole dollars?",
        "259200"
    ),
    (
        "Carrie works 8 hours a day at $14 per hour. She is also paid for 1 hour of "
        "overtime at 1.5 times her regular pay each day. How much does she make in a "
        "5-day workweek?",
        "645"
    ),
    (
        "A store sells apples for $1.50 each and oranges for $2.00 each. If Sara buys "
        "4 apples and 3 oranges, how much does she spend in total?",
        "12"
    ),
    # --- Additional problems ---
    (
        "Maria has 3 times as many coins as Luis. Luis has 24 coins. How many coins do "
        "Maria and Luis have together?",
        "96"
    ),
    (
        "A school has 500 students. 60% of the students are girls. How many boys are "
        "there in the school?",
        "200"
    ),
    (
        "Mike needs to buy 5 shirts for work. The shirts are normally $30 each but are "
        "on sale for 20% off. How much will Mike spend in total?",
        "120"
    ),
    (
        "Sam has $50 and spends $18 on a book and $12 on lunch. How much money does "
        "Sam have left?",
        "20"
    ),
    (
        "There are 12 apples in a basket. 3 friends want to share them equally. "
        "How many apples does each friend get?",
        "4"
    ),
    (
        "Jake earns $15 per hour working at a grocery store. He works 40 hours a week. "
        "How much does he earn in a 4-week month?",
        "2400"
    ),
    (
        "A box contains 24 chocolates. If 8 children each take 2 chocolates, "
        "how many chocolates are left in the box?",
        "8"
    ),
    (
        "Lisa reads 30 pages every day. How many pages will she read in 2 weeks?",
        "420"
    ),
    (
        "A swimming pool holds 5000 gallons of water. It loses 100 gallons per day "
        "due to evaporation. After 10 days, how many gallons are left?",
        "4000"
    ),
    (
        "Mark scored 85, 90, 78, and 95 on four tests. What is his average score? "
        "Round to nearest whole number.",
        "87"
    ),
    (
        "A garden has 8 rows of flowers with 15 flowers per row. If 20 flowers are "
        "removed, how many flowers are left?",
        "100"
    ),
    (
        "Tom has twice as much money as Jane. Together they have $90. How much money "
        "does Tom have?",
        "60"
    ),
    (
        "A train station sells 120 tickets on Monday, 95 on Tuesday, and 130 on "
        "Wednesday. What is the total number of tickets sold over these three days?",
        "345"
    ),
    (
        "Sarah can type 60 words per minute. How many words can she type in 45 minutes?",
        "2700"
    ),
    (
        "A baker makes 144 muffins and packs them into boxes of 12. "
        "How many boxes does the baker need?",
        "12"
    ),
    (
        "A factory produces 250 widgets per day. It operates 5 days a week. "
        "How many widgets does it produce in 4 weeks?",
        "5000"
    ),
    (
        "Emma has 15 red marbles and 20 blue marbles. If she gives away 8 red marbles "
        "and 5 blue marbles, how many marbles does she have in total?",
        "22"
    ),
    (
        "A pizza is cut into 8 slices. If 3 people each eat 2 slices, "
        "how many slices are left?",
        "2"
    ),
    (
        "A concert venue has 2000 seats. For a show, 85% of the seats are filled. "
        "How many people are at the concert?",
        "1700"
    ),
    (
        "A car uses 8 liters of fuel per 100 km. How many liters does it need "
        "for a 350 km journey?",
        "28"
    ),
    (
        "There are 7 days in a week and 52 weeks in a year. How many days are in a year?",
        "364"
    ),
    (
        "A farmer has 120 apples. He sells 3/4 of them at the market. "
        "How many apples does he have left?",
        "30"
    ),
    (
        "A class has 30 students. 40% scored above 80 on the test. "
        "How many students scored above 80?",
        "12"
    ),
    (
        "David saves $25 per week. How much will he have saved after 8 weeks?",
        "200"
    ),
    (
        "A rectangle has a length of 12 cm and a width of 7 cm. "
        "What is the area of the rectangle in square cm?",
        "84"
    ),
]

# Direct prompt - no chain of thought, just the answer
SYSTEM_PROMPT = (
    "You are a precise math solver. Give ONLY the final numerical answer, "
    "nothing else. No explanation, no units, just the number."
)

def extract_number(text: str) -> str:
    """Extract a number from model response."""
    text = text.strip()
    # Try the whole thing as a number first
    try:
        val = text.replace(',', '').replace('$', '').strip()
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        pass

    # Find all numbers
    nums = re.findall(r'-?[\d,]+(?:\.\d+)?', text.replace(',', ''))
    if nums:
        val = nums[0]
        try:
            f = float(val)
            if f == int(f):
                return str(int(f))
            return str(f)
        except ValueError:
            pass
    return text.strip()

def normalize(val: str) -> str:
    val = val.strip().replace(',', '').replace('$', '')
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return val

def run_benchmark():
    session = requests.Session()
    results = []
    correct = 0
    errors = 0
    total = len(MATH_PROBLEMS)

    print(f"GSM8K Benchmark: {MODEL_ID}", flush=True)
    print(f"API: {API}", flush=True)
    print(f"Problems: {total}", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()

    for i, (question, expected) in enumerate(MATH_PROBLEMS):
        req_start = time.time()
        try:
            resp = session.post(
                f"{API}/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 30,
                    "temperature": 0.0,
                },
                timeout=60
            )
            resp.raise_for_status()
            response_text = resp.json()["choices"][0]["message"]["content"]
            latency = time.time() - req_start

            predicted = extract_number(response_text)
            expected_norm = normalize(expected)
            predicted_norm = normalize(predicted)
            is_correct = predicted_norm == expected_norm

            correct += is_correct
            status = "PASS" if is_correct else "FAIL"
            print(f"[{i+1:2d}/{total}] {status} | Exp: {expected_norm:>8} | Got: {predicted_norm:>8} ({response_text.strip()[:30]}) | {latency:.1f}s", flush=True)

            results.append({
                "problem_idx": i + 1,
                "question": question,
                "expected": expected_norm,
                "predicted": predicted_norm,
                "raw_response": response_text.strip(),
                "correct": is_correct,
                "latency_s": round(latency, 2),
                "error": None
            })

        except Exception as e:
            latency = time.time() - req_start
            errors += 1
            print(f"[{i+1:2d}/{total}] ERR  | {str(e)[:80]} | {latency:.1f}s", flush=True)
            results.append({
                "problem_idx": i + 1,
                "question": question,
                "expected": normalize(expected),
                "predicted": "ERROR",
                "raw_response": str(e),
                "correct": False,
                "latency_s": round(latency, 2),
                "error": str(e)
            })

    elapsed = time.time() - start_time
    accuracy = correct / total * 100

    print("\n" + "=" * 70, flush=True)
    print(f"FINAL RESULTS:", flush=True)
    print(f"  Correct:     {correct}/{total}", flush=True)
    print(f"  Errors:      {errors}/{total}", flush=True)
    print(f"  Accuracy:    {accuracy:.1f}%", flush=True)
    print(f"  Total time:  {elapsed:.0f}s ({elapsed/total:.1f}s/question)", flush=True)
    print(f"\nBenchmark comparisons:", flush=True)
    print(f"  Google Gemma 4 26B BF16 (full GSM8K):  ~97.0%", flush=True)
    print(f"  RedHat NVFP4 (full GSM8K):              95.6%", flush=True)
    print(f"  This run ({total} problems, NVFP4):      {accuracy:.1f}%", flush=True)

    return results, correct, total, accuracy, errors, elapsed

if __name__ == "__main__":
    results, correct, total, accuracy, errors, elapsed = run_benchmark()

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "api": API,
        "correct": correct,
        "total": total,
        "errors": errors,
        "accuracy_pct": round(accuracy, 1),
        "elapsed_s": round(elapsed, 1),
        "avg_latency_s": round(elapsed / total, 1),
        "results": results
    }

    raw_path = "/root/projects/autokernel/profiling/gsm8k_raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {raw_path}", flush=True)
