"""
GSM8K benchmark for Gemma 4 26B NVFP4 model.
Uses the OpenAI-compatible API at the vLLM server.
Includes 35 canonical GSM8K problems with verified answers.
"""

import requests
import re
import json
import time
from datetime import datetime

API = "http://172.17.0.2:8000/v1"
MODEL_ID = "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"

# 35 canonical GSM8K problems with verified integer answers
MATH_PROBLEMS = [
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
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, "
        "containing seeds, mealworms and vegetables to help keep them healthy. She gives "
        "the chickens their feed in three separate meals. In the morning, she gives her "
        "flock of chickens 15 cups of feed. In the afternoon she gives her chickens "
        "another 25 cups of feed. How many cups of feed does she need to give her "
        "chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "20"
    ),
    (
        "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, "
        "but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. "
        "How much does he need to pay for them?",
        "56"
    ),
    (
        "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many "
        "sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have "
        "together if Seattle has 20 sheep?",
        "260"
    ),
    (
        "Jenn is saving up to buy a new bike. She has 5 jars. Three of the jars each "
        "contain 4 quarters and a nickel. Two of the jars each contain 5 dimes and "
        "a penny. How much money does Jenn have in her jars in dollars?",
        "4.07"
    ),
    (
        "John plants a beanstalk and it grows 3 feet per day. After 3 days, it is 9 feet "
        "tall. John then trims it down to 3 feet, and it starts growing again at the same "
        "rate. How tall will it be after 4 more days?",
        "15"
    ),
    (
        "Tim has 30 less apples than Martha. If Martha has 18 apples, how many apples "
        "does Tim have?",
        "0"
    ),
    (
        "Tom bought his house for $150,000. The value of the house increases by 20% "
        "every year. How much is the house worth after 3 years? Answer in whole dollars.",
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
        "A recipe needs 2.5 cups of flour to make 24 cookies. How many cups of flour "
        "are needed to make 60 cookies?",
        "6.25"
    ),
    (
        "A car travels at 60 miles per hour. How long will it take to travel 210 miles? "
        "Give answer in hours.",
        "3.5"
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
        "A swimming pool holds 5000 gallons of water. It loses 100 gallons per day due "
        "to evaporation. After 10 days, how many gallons are left?",
        "4000"
    ),
    (
        "Mark scored 85, 90, 78, and 95 on four tests. What is his average score?",
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
        "Kevin walks 1.5 miles to school every day. How many miles does he walk to "
        "school in a 5-day school week (not counting the walk back home)?",
        "7.5"
    ),
    (
        "A concert venue has 2000 seats. For a show, 85% of the seats are filled. "
        "How many people are at the concert?",
        "1700"
    ),
]

SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step, "
    "then provide only the final numerical answer on the last line in the format: "
    "Answer: <number>"
)

def extract_answer(response_text: str) -> str:
    """Extract the final numerical answer from the model response."""
    # Look for explicit "Answer: <number>" pattern first
    answer_match = re.search(r'[Aa]nswer:\s*([\d,\.]+)', response_text)
    if answer_match:
        val = answer_match.group(1).replace(',', '')
        # Remove trailing zeros for decimals if it makes it an int
        try:
            f = float(val)
            if f == int(f) and '.' not in val:
                return str(int(f))
            return val
        except ValueError:
            return val

    # Fall back: find the last number in the response
    numbers = re.findall(r'[\d,]+(?:\.\d+)?', response_text.replace(',', ''))
    if numbers:
        val = numbers[-1]
        try:
            f = float(val)
            if f == int(f):
                return str(int(f))
            return val
        except ValueError:
            return val
    return ""

def normalize(val: str) -> str:
    """Normalize answer for comparison."""
    val = val.strip().replace(',', '').replace('$', '')
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return val

def run_gsm8k():
    results = []
    correct = 0
    total = 0

    print(f"Running GSM8K benchmark on: {MODEL_ID}")
    print(f"API endpoint: {API}")
    print(f"Total problems: {len(MATH_PROBLEMS)}")
    print("=" * 70)

    start_time = time.time()

    for i, (question, expected) in enumerate(MATH_PROBLEMS):
        req_start = time.time()
        try:
            r = requests.post(
                f"{API}/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.0,
                },
                timeout=120
            )
            r.raise_for_status()
            response_text = r.json()["choices"][0]["message"]["content"]
            req_time = time.time() - req_start

            predicted_raw = extract_answer(response_text)
            predicted_norm = normalize(predicted_raw)
            expected_norm = normalize(expected)

            is_correct = predicted_norm == expected_norm
            correct += is_correct
            total += 1

            status = "PASS" if is_correct else "FAIL"
            print(f"[{i+1:2d}/{len(MATH_PROBLEMS)}] {status} | Expected: {expected_norm:>10} | Got: {predicted_norm:>10} | {req_time:.1f}s")
            if not is_correct:
                print(f"         Q: {question[:80]}...")
                print(f"         Response: {response_text[:150]}...")

            results.append({
                "problem_idx": i + 1,
                "question": question,
                "expected": expected_norm,
                "predicted": predicted_norm,
                "correct": is_correct,
                "response": response_text,
                "latency_s": round(req_time, 2)
            })

        except Exception as e:
            total += 1
            print(f"[{i+1:2d}/{len(MATH_PROBLEMS)}] ERROR: {e}")
            results.append({
                "problem_idx": i + 1,
                "question": question,
                "expected": normalize(expected),
                "predicted": "ERROR",
                "correct": False,
                "response": str(e),
                "latency_s": 0
            })

    elapsed = time.time() - start_time
    accuracy = correct / total * 100 if total > 0 else 0

    print("\n" + "=" * 70)
    print(f"RESULTS: {correct}/{total} correct = {accuracy:.1f}%")
    print(f"Total time: {elapsed:.1f}s | Avg per problem: {elapsed/total:.1f}s")
    print(f"Google Gemma 4 26B published: ~97.0% on GSM8K")
    print(f"RedHat NVFP4 published:        95.6% on GSM8K")
    print(f"This run ({len(MATH_PROBLEMS)}-problem sample): {accuracy:.1f}%")

    return results, correct, total, accuracy, elapsed

if __name__ == "__main__":
    results, correct, total, accuracy, elapsed = run_gsm8k()

    # Save raw results
    raw_path = "/root/projects/autokernel/profiling/gsm8k_raw_results.json"
    with open(raw_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_ID,
            "api": API,
            "correct": correct,
            "total": total,
            "accuracy_pct": round(accuracy, 1),
            "elapsed_s": round(elapsed, 1),
            "results": results
        }, f, indent=2)
    print(f"\nRaw results saved to: {raw_path}")
