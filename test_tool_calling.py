#!/usr/bin/env python3
"""Comprehensive tool-calling evaluation for vLLM-served models.

Tests 8 categories of agentic tool/skill calling:
  1. Simple single tool call
  2. Parallel tool calls
  3. Sequential multi-turn tool use
  4. Irrelevance detection (should NOT call tools)
  5. Parameter extraction accuracy
  6. Nested/complex arguments
  7. Tool choice under ambiguity
  8. Multi-step reasoning chain

Usage:
    python test_tool_calling.py [--base-url http://localhost:8000/v1] [--model MODEL_NAME]
"""

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any

import openai

# ── Tool definitions ─────────────────────────────────────────────────────────

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "country": {"type": "string", "description": "Country code (ISO 3166-1 alpha-2)"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["city"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results (1-10)"},
                "language": {"type": "string", "description": "Language code for results"},
            },
            "required": ["query"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
            },
            "required": ["expression"],
        },
    },
}

SEND_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body text"},
                "cc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "CC recipients",
                },
                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

CREATE_EVENT_TOOL = {
    "type": "function",
    "function": {
        "name": "create_calendar_event",
        "description": "Create a calendar event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start_time": {"type": "string", "description": "ISO 8601 datetime"},
                "end_time": {"type": "string", "description": "ISO 8601 datetime"},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses",
                },
                "location": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["title", "start_time", "end_time"],
        },
    },
}

DB_QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a read-only SQL query against the application database",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT query"},
                "database": {"type": "string", "enum": ["users", "orders", "products"]},
                "limit": {"type": "integer", "description": "Max rows to return"},
            },
            "required": ["sql", "database"],
        },
    },
}

FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file from the filesystem",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute file path"},
                "encoding": {"type": "string", "description": "File encoding (default: utf-8)"},
            },
            "required": ["path"],
        },
    },
}

ALL_TOOLS = [WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL, SEND_EMAIL_TOOL,
             CREATE_EVENT_TOOL, DB_QUERY_TOOL, FILE_TOOL]

# ── Test definitions ─────────────────────────────────────────────────────────

@dataclass
class TestCase:
    name: str
    category: str
    messages: list[dict]
    tools: list[dict]
    check: Any  # callable(response) -> (passed: bool, detail: str)
    tool_choice: str = "auto"


@dataclass
class TestResult:
    name: str
    category: str
    passed: bool
    detail: str
    error: str = ""


def has_tool_call(resp, name=None):
    """Check if response has a tool call, optionally matching name."""
    msg = resp.choices[0].message
    if not msg.tool_calls:
        return False
    if name is None:
        return True
    return any(tc.function.name == name for tc in msg.tool_calls)


def get_tool_args(resp, name=None):
    """Get parsed arguments from first matching tool call."""
    msg = resp.choices[0].message
    if not msg.tool_calls:
        return None
    for tc in msg.tool_calls:
        if name is None or tc.function.name == name:
            try:
                return json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                return None
    return None


def count_tool_calls(resp, name=None):
    msg = resp.choices[0].message
    if not msg.tool_calls:
        return 0
    if name is None:
        return len(msg.tool_calls)
    return sum(1 for tc in msg.tool_calls if tc.function.name == name)


# ── Test cases ───────────────────────────────────────────────────────────────

def build_tests():
    tests = []

    # ── Category 1: Simple single tool call ──────────────────────────────
    def check_simple_weather(resp):
        if not has_tool_call(resp, "get_weather"):
            return False, "Expected get_weather call"
        args = get_tool_args(resp, "get_weather")
        if not args or "city" not in args:
            return False, f"Missing 'city' arg: {args}"
        if "tokyo" not in args["city"].lower():
            return False, f"Expected Tokyo, got: {args['city']}"
        return True, f"get_weather({json.dumps(args)})"

    tests.append(TestCase(
        name="simple_weather",
        category="1_simple",
        messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        check=check_simple_weather,
    ))

    def check_simple_search(resp):
        if not has_tool_call(resp, "web_search"):
            return False, "Expected web_search call"
        args = get_tool_args(resp, "web_search")
        if not args or "query" not in args:
            return False, f"Missing 'query': {args}"
        return True, f"web_search({json.dumps(args)})"

    tests.append(TestCase(
        name="simple_search",
        category="1_simple",
        messages=[{"role": "user", "content": "Search the web for the latest news about SpaceX Starship"}],
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        check=check_simple_search,
    ))

    def check_simple_email(resp):
        if not has_tool_call(resp, "send_email"):
            return False, "Expected send_email call"
        args = get_tool_args(resp, "send_email")
        if not args:
            return False, "Could not parse args"
        if "alice@example.com" not in args.get("to", ""):
            return False, f"Wrong recipient: {args.get('to')}"
        if not args.get("subject"):
            return False, "Missing subject"
        if not args.get("body"):
            return False, "Missing body"
        return True, f"send_email(to={args['to']}, subject={args['subject'][:30]}...)"

    tests.append(TestCase(
        name="simple_email",
        category="1_simple",
        messages=[{"role": "user", "content": "Send an email to alice@example.com with subject 'Meeting Tomorrow' and body 'Hi Alice, can we meet at 3pm tomorrow? Thanks, Bob'"}],
        tools=[SEND_EMAIL_TOOL, SEARCH_TOOL],
        check=check_simple_email,
    ))

    # ── Category 2: Parallel tool calls ──────────────────────────────────
    def check_parallel_weather(resp):
        n = count_tool_calls(resp, "get_weather")
        if n < 2:
            return False, f"Expected >=2 parallel get_weather calls, got {n}"
        calls = []
        for tc in resp.choices[0].message.tool_calls:
            if tc.function.name == "get_weather":
                calls.append(json.loads(tc.function.arguments))
        cities = [c.get("city", "").lower() for c in calls]
        has_london = any("london" in c for c in cities)
        has_paris = any("paris" in c for c in cities)
        has_berlin = any("berlin" in c for c in cities)
        if not (has_london and has_paris and has_berlin):
            return False, f"Expected London, Paris, Berlin; got {cities}"
        return True, f"{n} parallel calls: {cities}"

    tests.append(TestCase(
        name="parallel_3_cities",
        category="2_parallel",
        messages=[{"role": "user", "content": "What's the weather in London, Paris, and Berlin right now?"}],
        tools=[WEATHER_TOOL],
        check=check_parallel_weather,
    ))

    def check_parallel_mixed(resp):
        has_w = has_tool_call(resp, "get_weather")
        has_s = has_tool_call(resp, "web_search")
        n = count_tool_calls(resp)
        if not (has_w and has_s):
            return False, f"Expected both get_weather and web_search, got {n} calls"
        return True, f"{n} mixed parallel calls"

    tests.append(TestCase(
        name="parallel_mixed_tools",
        category="2_parallel",
        messages=[{"role": "user", "content": "I need two things: 1) the current weather in San Francisco, and 2) search the web for 'best restaurants in San Francisco'"}],
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        check=check_parallel_mixed,
    ))

    # ── Category 3: Irrelevance detection ────────────────────────────────
    def check_no_tool(resp):
        if has_tool_call(resp):
            names = [tc.function.name for tc in resp.choices[0].message.tool_calls]
            return False, f"Should NOT have called tools, but called: {names}"
        content = resp.choices[0].message.content or ""
        return True, f"Correctly no tool call. Response: {content[:80]}..."

    tests.append(TestCase(
        name="irrelevance_math",
        category="3_irrelevance",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=[CALCULATOR_TOOL, DB_QUERY_TOOL],
        check=check_no_tool,
    ))

    tests.append(TestCase(
        name="irrelevance_chitchat",
        category="3_irrelevance",
        messages=[{"role": "user", "content": "Tell me a joke about programmers"}],
        tools=[WEATHER_TOOL, SEND_EMAIL_TOOL, DB_QUERY_TOOL],
        check=check_no_tool,
    ))

    tests.append(TestCase(
        name="irrelevance_opinion",
        category="3_irrelevance",
        messages=[{"role": "user", "content": "What do you think about the future of AI?"}],
        tools=[FILE_TOOL, CALCULATOR_TOOL],
        check=check_no_tool,
    ))

    # ── Category 4: Parameter extraction accuracy ────────────────────────
    def check_email_params(resp):
        if not has_tool_call(resp, "send_email"):
            return False, "Expected send_email"
        args = get_tool_args(resp, "send_email")
        if not args:
            return False, "No args"
        checks = []
        if "bob@company.com" in args.get("to", ""):
            checks.append("to=correct")
        else:
            checks.append(f"to=WRONG({args.get('to')})")
        cc = args.get("cc", [])
        if isinstance(cc, list) and len(cc) >= 2:
            checks.append(f"cc={len(cc)} recipients")
        else:
            checks.append(f"cc=WRONG({cc})")
        if args.get("priority") == "high":
            checks.append("priority=correct")
        else:
            checks.append(f"priority=WRONG({args.get('priority')})")
        all_ok = all("WRONG" not in c for c in checks)
        return all_ok, "; ".join(checks)

    tests.append(TestCase(
        name="param_extraction_email",
        category="4_params",
        messages=[{"role": "user", "content": "Send a high priority email to bob@company.com, CC alice@company.com and charlie@company.com, subject 'Urgent: Server Down', body 'The production server is unresponsive. Please investigate immediately.'"}],
        tools=[SEND_EMAIL_TOOL],
        check=check_email_params,
    ))

    def check_event_params(resp):
        if not has_tool_call(resp, "create_calendar_event"):
            return False, "Expected create_calendar_event"
        args = get_tool_args(resp, "create_calendar_event")
        if not args:
            return False, "No args"
        checks = []
        if args.get("title"):
            checks.append(f"title='{args['title'][:30]}'")
        else:
            checks.append("title=MISSING")
        if args.get("start_time"):
            checks.append(f"start={args['start_time']}")
        else:
            checks.append("start=MISSING")
        attendees = args.get("attendees", [])
        if isinstance(attendees, list) and len(attendees) >= 2:
            checks.append(f"attendees={len(attendees)}")
        else:
            checks.append(f"attendees=WRONG({attendees})")
        if args.get("location"):
            checks.append(f"location='{args['location'][:20]}'")
        all_ok = "MISSING" not in " ".join(checks) and "WRONG" not in " ".join(checks)
        return all_ok, "; ".join(checks)

    tests.append(TestCase(
        name="param_extraction_event",
        category="4_params",
        messages=[{"role": "user", "content": "Create a meeting called 'Q4 Planning' on 2025-01-15 from 2pm to 4pm EST with attendees sarah@co.com and mike@co.com at Conference Room B"}],
        tools=[CREATE_EVENT_TOOL],
        check=check_event_params,
    ))

    # ── Category 5: Nested/complex arguments ─────────────────────────────
    def check_sql(resp):
        if not has_tool_call(resp, "query_database"):
            return False, "Expected query_database"
        args = get_tool_args(resp, "query_database")
        if not args:
            return False, "No args"
        sql = args.get("sql", "").lower()
        db = args.get("database", "")
        has_select = "select" in sql
        has_where = "where" in sql
        has_order = "order by" in sql
        correct_db = db == "orders"
        detail = f"sql='{sql[:60]}', db={db}, SELECT={has_select}, WHERE={has_where}, ORDER={has_order}"
        return has_select and correct_db, detail

    tests.append(TestCase(
        name="complex_sql_query",
        category="5_complex",
        messages=[{"role": "user", "content": "Query the orders database to find the top 10 orders by total amount from the last 30 days, ordered by amount descending"}],
        tools=[DB_QUERY_TOOL],
        check=check_sql,
    ))

    # ── Category 6: Tool choice under ambiguity ─────────────────────────
    def check_ambiguity_weather(resp):
        # "How hot is it outside?" with both weather and search — should pick weather
        if has_tool_call(resp, "get_weather"):
            return True, "Correctly chose get_weather for temperature question"
        if has_tool_call(resp, "web_search"):
            return False, "Chose web_search instead of get_weather"
        return False, "No tool call made"

    tests.append(TestCase(
        name="ambiguity_temperature",
        category="6_ambiguity",
        messages=[
            {"role": "system", "content": "The user is located in New York City."},
            {"role": "user", "content": "How hot is it outside right now?"},
        ],
        tools=[WEATHER_TOOL, SEARCH_TOOL, CALCULATOR_TOOL],
        check=check_ambiguity_weather,
    ))

    def check_ambiguity_calc_vs_search(resp):
        # Should use calculator, not search
        if has_tool_call(resp, "calculator"):
            return True, "Correctly chose calculator"
        if has_tool_call(resp, "web_search"):
            return False, "Chose web_search instead of calculator"
        # Might answer directly — also acceptable
        if not has_tool_call(resp):
            content = resp.choices[0].message.content or ""
            if "47" in content:
                return True, f"Answered directly (acceptable): {content[:60]}"
            return False, f"No tool and wrong answer: {content[:60]}"
        return False, "Wrong tool"

    tests.append(TestCase(
        name="ambiguity_math_vs_search",
        category="6_ambiguity",
        messages=[{"role": "user", "content": "What is 23 + 24?"}],
        tools=[SEARCH_TOOL, CALCULATOR_TOOL, WEATHER_TOOL],
        check=check_ambiguity_calc_vs_search,
    ))

    # ── Category 7: Multi-turn with tool results ─────────────────────────
    # We do this one programmatically in run_multi_turn()

    # ── Category 8: Reasoning chain (should call multiple tools sequentially)
    def check_reasoning(resp):
        # For a complex request, model should call at least one tool
        n = count_tool_calls(resp)
        if n == 0:
            return False, "No tool calls for complex request"
        names = [tc.function.name for tc in resp.choices[0].message.tool_calls]
        return True, f"Called {n} tool(s): {names}"

    tests.append(TestCase(
        name="reasoning_multi_tool",
        category="8_reasoning",
        messages=[{"role": "user", "content": "I need to prepare for my trip to Tokyo next week. Can you check the weather there and also search for the best sushi restaurants?"}],
        tools=[WEATHER_TOOL, SEARCH_TOOL, CREATE_EVENT_TOOL],
        check=check_reasoning,
    ))

    return tests


# ── Multi-turn test (category 7) ────────────────────────────────────────────

def run_multi_turn(client, model):
    """Test a 3-step multi-turn tool-use conversation."""
    results = []
    tools = [WEATHER_TOOL, SEND_EMAIL_TOOL]

    # Turn 1: user asks, model should call get_weather
    messages = [
        {"role": "user", "content": "Check the weather in Seattle, and then send an email to team@example.com with the weather report."}
    ]
    resp1 = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", max_tokens=1024)
    msg1 = resp1.choices[0].message

    if not has_tool_call(resp1, "get_weather"):
        results.append(TestResult("multi_turn_step1", "7_multi_turn", False, "Expected get_weather call first"))
        return results
    results.append(TestResult("multi_turn_step1", "7_multi_turn", True, f"Called get_weather"))

    # Turn 2: feed weather result, model should call send_email
    tc = [t for t in msg1.tool_calls if t.function.name == "get_weather"][0]
    messages.append(msg1)
    messages.append({
        "role": "tool",
        "content": json.dumps({"temperature": 12, "condition": "rainy", "humidity": 85, "unit": "celsius"}),
        "tool_call_id": tc.id,
    })
    # If there were other tool calls, provide dummy results
    for t in msg1.tool_calls:
        if t.id != tc.id:
            messages.append({
                "role": "tool",
                "content": json.dumps({"status": "pending"}),
                "tool_call_id": t.id,
            })

    resp2 = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", max_tokens=1024)
    msg2 = resp2.choices[0].message

    if not has_tool_call(resp2, "send_email"):
        # Maybe it already sent email in parallel in step 1
        if has_tool_call(resp1, "send_email"):
            results.append(TestResult("multi_turn_step2", "7_multi_turn", True, "send_email was called in parallel (acceptable)"))
        else:
            detail = f"Expected send_email, got: {[t.function.name for t in msg2.tool_calls] if msg2.tool_calls else 'no calls'}"
            results.append(TestResult("multi_turn_step2", "7_multi_turn", False, detail))
            return results
    else:
        args = get_tool_args(resp2, "send_email")
        has_team = "team@example.com" in (args or {}).get("to", "")
        body = (args or {}).get("body", "")
        mentions_weather = any(w in body.lower() for w in ["rain", "12", "seattle", "weather"])
        results.append(TestResult("multi_turn_step2", "7_multi_turn", has_team and mentions_weather,
                                  f"to={args.get('to')}, body mentions weather: {mentions_weather}"))

    # Turn 3: feed email result, model should give final summary
    if msg2.tool_calls:
        messages.append(msg2)
        for t in msg2.tool_calls:
            messages.append({
                "role": "tool",
                "content": json.dumps({"status": "sent", "message_id": "msg_12345"}),
                "tool_call_id": t.id,
            })
        resp3 = client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", max_tokens=1024)
        msg3 = resp3.choices[0].message
        has_summary = msg3.content and len(msg3.content) > 10
        no_extra_calls = not msg3.tool_calls
        results.append(TestResult("multi_turn_step3", "7_multi_turn", has_summary and no_extra_calls,
                                  f"Summary: {(msg3.content or '')[:80]}..., extra_calls={not no_extra_calls}"))

    return results


# ── Runner ───────────────────────────────────────────────────────────────────

def run_all(base_url, model):
    client = openai.OpenAI(base_url=base_url, api_key="dummy")
    tests = build_tests()
    results: list[TestResult] = []

    for t in tests:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=t.messages,
                tools=t.tools,
                tool_choice=t.tool_choice,
                max_tokens=1024,
            )
            passed, detail = t.check(resp)
            results.append(TestResult(t.name, t.category, passed, detail))
        except Exception as e:
            results.append(TestResult(t.name, t.category, False, "", str(e)))

    # Multi-turn tests
    try:
        mt_results = run_multi_turn(client, model)
        results.extend(mt_results)
    except Exception as e:
        results.append(TestResult("multi_turn", "7_multi_turn", False, "", str(e)))

    return results


def print_results(results):
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    total_pass = 0
    total = 0

    print("\n" + "=" * 80)
    print("TOOL CALLING EVALUATION RESULTS")
    print("=" * 80)

    for cat in sorted(categories.keys()):
        cat_results = categories[cat]
        cat_pass = sum(1 for r in cat_results if r.passed)
        cat_total = len(cat_results)
        total_pass += cat_pass
        total += cat_total

        print(f"\n── {cat} ({cat_pass}/{cat_total}) ──")
        for r in cat_results:
            icon = "PASS" if r.passed else "FAIL"
            print(f"  [{icon}] {r.name}")
            if r.detail:
                print(f"         {r.detail}")
            if r.error:
                print(f"         ERROR: {r.error}")

    print("\n" + "=" * 80)
    pct = (total_pass / total * 100) if total else 0
    print(f"TOTAL: {total_pass}/{total} ({pct:.0f}%)")
    print("=" * 80)

    return total_pass, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool calling evaluation")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="mconcat/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-NVFP4")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Endpoint: {args.base_url}")
    print("Running evaluation...")

    results = run_all(args.base_url, args.model)
    passed, total = print_results(results)
    sys.exit(0 if passed == total else 1)
