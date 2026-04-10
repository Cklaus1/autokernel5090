"""
GSM8K benchmark - final robust version.
Uses chain-of-thought with proper answer extraction.
Concurrent requests with generous timeouts.
"""

import asyncio
import aiohttp
import re
import json
import time
from datetime import datetime

API = "http://172.17.0.2:8000/v1"
MODEL_ID = "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"

# 35 GSM8K-style problems with verified answers
MATH_PROBLEMS = [
    # Original GSM8K canonical problems
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
    # Additional problems
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
        "Round to the nearest whole number.",
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

# Chain-of-thought prompt - let the model think, then extract the answer
SYSTEM_PROMPT = (
    "You are a math solver. Solve the problem step by step. "
    "At the end of your solution, write the final answer on its own line as:\n"
    "#### <number>\n"
    "Use the GSM8K format."
)

def extract_answer(text: str) -> str:
    """Extract answer from GSM8K-format response (#### <number>)."""
    # Primary: GSM8K format ####
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        val = m.group(1).replace(',', '').strip()
        try:
            f = float(val)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return val

    # Fallback: "Answer: <number>"
    m = re.search(r'[Aa]nswer:?\s*\$?([\-\d,\.]+)', text)
    if m:
        val = m.group(1).replace(',', '').strip()
        try:
            f = float(val)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return val

    # Fallback: last number in response
    nums = re.findall(r'\b[\d,]+(?:\.\d+)?\b', text.replace(',', ''))
    if nums:
        val = nums[-1]
        try:
            f = float(val)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            pass

    return ""

def normalize(val: str) -> str:
    val = val.strip().replace(',', '').replace('$', '')
    try:
        f = float(val)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return val

async def query_one(session, idx, question, expected, semaphore):
    """Run one problem, with retry on failure."""
    async with semaphore:
        for attempt in range(3):
            req_start = time.time()
            try:
                async with session.post(
                    f"{API}/chat/completions",
                    json={
                        "model": MODEL_ID,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": question}
                        ],
                        "max_tokens": 500,
                        "temperature": 0.0,
                    },
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    data = await resp.json()
                    response_text = data["choices"][0]["message"]["content"]
                    latency = time.time() - req_start

                    predicted = extract_answer(response_text)
                    expected_norm = normalize(expected)
                    predicted_norm = normalize(predicted)
                    is_correct = predicted_norm == expected_norm

                    return {
                        "problem_idx": idx,
                        "question": question,
                        "expected": expected_norm,
                        "predicted": predicted_norm,
                        "raw_response": response_text,
                        "correct": is_correct,
                        "latency_s": round(latency, 2),
                        "error": None,
                        "attempts": attempt + 1
                    }
            except (asyncio.TimeoutError, aiohttp.ServerDisconnectedError) as e:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                return {
                    "problem_idx": idx,
                    "question": question,
                    "expected": normalize(expected),
                    "predicted": "ERROR",
                    "raw_response": str(e),
                    "correct": False,
                    "latency_s": round(time.time() - req_start, 2),
                    "error": str(e),
                    "attempts": attempt + 1
                }
            except Exception as e:
                return {
                    "problem_idx": idx,
                    "question": question,
                    "expected": normalize(expected),
                    "predicted": "ERROR",
                    "raw_response": str(e),
                    "correct": False,
                    "latency_s": round(time.time() - req_start, 2),
                    "error": str(e),
                    "attempts": attempt + 1
                }

async def run_benchmark():
    print(f"GSM8K Benchmark")
    print(f"Model: {MODEL_ID}")
    print(f"API: {API}")
    print(f"Problems: {len(MATH_PROBLEMS)}")
    print(f"Concurrency: 5 (with 300s timeout, up to 3 retries)")
    print("=" * 70)

    start_time = time.time()
    semaphore = asyncio.Semaphore(5)

    connector = aiohttp.TCPConnector(limit=10, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            query_one(session, i + 1, q, a, semaphore)
            for i, (q, a) in enumerate(MATH_PROBLEMS)
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda x: x["problem_idx"])
    elapsed = time.time() - start_time

    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    accuracy = correct / total * 100

    print("\nDetailed Results:")
    print("-" * 70)
    for r in results:
        if r["error"]:
            status = "ERR "
        elif r["correct"]:
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{r['problem_idx']:2d}] {status} | Exp: {r['expected']:>8} | Got: {r['predicted']:>8} | {r['latency_s']:.0f}s", end="")
        if not r["correct"] and not r["error"]:
            # Show first 200 chars of response for debugging
            resp_preview = r["raw_response"][:150].replace('\n', ' ')
            print(f"\n     > {resp_preview}")
        elif r["error"]:
            print(f"\n     ERR: {r['error'][:80]}")
        else:
            print()

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS:")
    print(f"  Correct:  {correct}/{total}")
    print(f"  Errors:   {errors}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Time:     {elapsed:.0f}s total | {elapsed/total:.1f}s avg per problem")
    print(f"\nComparison:")
    print(f"  Google Gemma 4 26B BF16 (full 1319 problems):  ~97.0%")
    print(f"  RedHat NVFP4 (full 1319 problems):              95.6%")
    print(f"  This run ({total} problems, NVFP4 w/ k4v4b64):  {accuracy:.1f}%")

    return results, correct, total, accuracy, errors, elapsed

if __name__ == "__main__":
    results, correct, total, accuracy, errors, elapsed = asyncio.run(run_benchmark())

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "api": API,
        "quantization": "NVFP4 (modelopt)",
        "kv_cache_dtype": "k4v4b64 (FusenCache)",
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
    print(f"\nSaved to: {raw_path}")
