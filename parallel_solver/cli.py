"""CLI for the Parallel Problem-Solving system.

Usage:
    python3 -m parallel_solver solve --problem "..." --codebase ./src/ --agents 8
    python3 -m parallel_solver interactive --codebase ./src/ --agents 4
    python3 -m parallel_solver benchmark --problems problems.json --agents 1,2,4,8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from parallel_solver.orchestrator import ProblemOrchestrator, STRATEGY_PRESETS
from parallel_solver.prefix_manager import PrefixManager, STRATEGY_PROMPTS
from parallel_solver.streaming import StreamEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Terminal colors
COLORS = [
    "\033[91m",  # red
    "\033[92m",  # green
    "\033[93m",  # yellow
    "\033[94m",  # blue
    "\033[95m",  # magenta
    "\033[96m",  # cyan
    "\033[97m",  # white
    "\033[33m",  # dark yellow
]
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def color_for_agent(agent_id: int) -> str:
    return COLORS[agent_id % len(COLORS)]


def print_header(text: str) -> None:
    width = min(80, len(text) + 4)
    print(f"\n{BOLD}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{RESET}\n")


def print_result_summary(result) -> None:
    """Print a formatted summary of the solve result."""
    print_header("Results")

    print(f"  Strategies:  {', '.join(result.strategies_used)}")
    print(f"  Total time:  {result.total_time_ms / 1000:.1f}s")
    print(f"  Prefix warm: {result.prefix_warm_ms:.0f}ms")
    print(f"  Generation:  {result.generation_ms / 1000:.1f}s")
    print(f"  Scoring:     {result.scoring_ms / 1000:.1f}s")
    print(f"  Tokens:      {result.total_tokens:,}")
    print(f"  Throughput:  {result.aggregate_tps:.0f} tok/s (aggregate)")
    print()

    if result.scored_solutions:
        print(f"{BOLD}  Ranked Solutions:{RESET}")
        for i, s in enumerate(result.scored_solutions):
            marker = " <-- BEST" if i == 0 else ""
            c = color_for_agent(s.agent_id)
            print(
                f"  {c}#{s.agent_id} ({s.strategy}){RESET}: "
                f"overall={s.overall:.2f} "
                f"(correct={s.correctness:.2f}, "
                f"complete={s.completeness:.2f}, "
                f"quality={s.code_quality:.2f})"
                f"{BOLD}{marker}{RESET}"
            )
            if s.explanation:
                print(f"      {DIM}{s.explanation[:120]}{RESET}")
        print()

    if result.best_solution:
        print_header(f"Best Solution (Agent #{result.best_solution.agent_id}, {result.best_solution.strategy})")
        print(result.best_solution.content)
        print()

    if result.merged_solution:
        print_header("Merged Solution (combined from top solutions)")
        print(result.merged_solution)
        print()


async def cmd_solve(args: argparse.Namespace) -> None:
    """Solve a problem using parallel agents."""
    # Load codebase
    codebase_path = Path(args.codebase)
    codebase = PrefixManager.load_codebase(codebase_path, max_tokens_approx=args.max_prefix_tokens)
    logger.info("Loaded codebase: ~%d chars from %s", len(codebase), codebase_path)

    # Load problem
    if args.problem_file:
        problem = Path(args.problem_file).read_text()
    else:
        problem = args.problem

    if not problem:
        print("Error: provide --problem or --problem-file", file=sys.stderr)
        sys.exit(1)

    # Parse strategies
    strategies = None
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",")]

    # Load tests
    tests = None
    if args.tests:
        tests_path = Path(args.tests)
        if tests_path.is_file():
            tests = [tests_path.read_text()]
        elif tests_path.is_dir():
            tests = [f.read_text() for f in sorted(tests_path.glob("test_*.py"))]

    # Set up event handler for streaming output
    agent_lines: dict[int, int] = {}

    def on_event(event: StreamEvent) -> None:
        c = color_for_agent(event.agent_id)
        # Print new tokens inline with agent label
        if event.delta:
            label = f"{c}[{event.strategy}]{RESET} "
            # Only print label at start of new lines
            text = event.delta.replace("\n", f"\n{label}")
            if event.agent_id not in agent_lines:
                agent_lines[event.agent_id] = 0
                sys.stdout.write(f"\n{label}")
            sys.stdout.write(text)
            sys.stdout.flush()

    orch = ProblemOrchestrator(
        args.api,
        args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
    )

    print_header(f"Parallel Solver: {args.agents} agents")
    print(f"  Problem: {problem[:200]}{'...' if len(problem) > 200 else ''}")
    print(f"  Codebase: {codebase_path} (~{len(codebase) // 4:,} tokens)")
    print()

    result = await orch.solve(
        problem,
        codebase,
        strategies=strategies,
        preset=args.preset,
        max_agents=args.agents,
        tests=tests,
        merge=args.merge,
        on_event=on_event if args.stream else None,
    )

    print()  # newline after streaming
    print_result_summary(result)

    # Save results
    if args.output:
        output = {
            "problem": problem,
            "strategies": result.strategies_used,
            "total_time_ms": result.total_time_ms,
            "total_tokens": result.total_tokens,
            "aggregate_tps": result.aggregate_tps,
            "solutions": [
                {
                    "agent_id": s.agent_id,
                    "strategy": s.strategy,
                    "overall_score": s.overall,
                    "content": s.content,
                }
                for s in result.scored_solutions
            ],
            "merged_solution": result.merged_solution,
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        logger.info("Results saved to %s", args.output)


async def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive mode: enter problems one at a time."""
    codebase_path = Path(args.codebase)
    codebase = PrefixManager.load_codebase(codebase_path, max_tokens_approx=args.max_prefix_tokens)

    orch = ProblemOrchestrator(
        args.api,
        args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout,
    )

    # Warm prefix once
    print_header("Parallel Solver Interactive Mode")
    print(f"  Codebase: {codebase_path} (~{len(codebase) // 4:,} tokens)")
    print(f"  Agents: {args.agents}")
    print(f"  Model: {args.model}")
    print(f"\n  Type a problem and press Enter. Type 'quit' to exit.\n")

    await orch.prefix_mgr.warm_prefix(codebase)
    print(f"  {DIM}Prefix cache warmed.{RESET}\n")

    while True:
        try:
            problem = input(f"{BOLD}Problem> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        problem = problem.strip()
        if not problem:
            continue
        if problem.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        # Check for inline strategy override: "!refactor Fix the caching"
        preset = args.preset
        if problem.startswith("!"):
            parts = problem.split(None, 1)
            if len(parts) == 2 and parts[0][1:] in STRATEGY_PRESETS:
                preset = parts[0][1:]
                problem = parts[1]

        result = await orch.solve(
            problem,
            codebase,
            preset=preset,
            max_agents=args.agents,
            merge=args.merge,
        )
        print_result_summary(result)


async def cmd_benchmark(args: argparse.Namespace) -> None:
    """Benchmark parallel solving at different agent counts."""
    # Load problems
    problems_path = Path(args.problems)
    with open(problems_path) as f:
        problems_data = json.load(f)

    # Expect format: [{"problem": "...", "codebase": "...", "tests": [...]}]
    if isinstance(problems_data, list):
        problems = problems_data
    else:
        problems = problems_data.get("problems", [])

    agent_counts = [int(x) for x in args.agents.split(",")]

    print_header("Parallel Solver Benchmark")
    print(f"  Problems: {len(problems)}")
    print(f"  Agent counts: {agent_counts}")
    print()

    results = []

    for n_agents in agent_counts:
        orch = ProblemOrchestrator(
            args.api,
            args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        for i, prob in enumerate(problems):
            problem = prob["problem"]
            codebase = prob.get("codebase", "")
            if "codebase_path" in prob:
                codebase = PrefixManager.load_codebase(prob["codebase_path"])
            tests = prob.get("tests")

            print(f"  [{n_agents} agents] Problem {i + 1}/{len(problems)}: {problem[:80]}...")

            result = await orch.solve(
                problem,
                codebase,
                max_agents=n_agents,
                tests=tests,
            )

            best_score = result.best_solution.overall if result.best_solution else 0.0
            entry = {
                "agents": n_agents,
                "problem_index": i,
                "problem": problem[:100],
                "total_time_ms": result.total_time_ms,
                "total_tokens": result.total_tokens,
                "aggregate_tps": result.aggregate_tps,
                "best_score": best_score,
                "num_solutions": len(result.scored_solutions),
            }
            results.append(entry)
            print(
                f"    -> {result.total_time_ms / 1000:.1f}s, "
                f"{result.total_tokens} tok, "
                f"{result.aggregate_tps:.0f} tok/s, "
                f"best={best_score:.2f}"
            )

    # Summary
    print_header("Benchmark Summary")
    for n_agents in agent_counts:
        entries = [r for r in results if r["agents"] == n_agents]
        avg_time = sum(r["total_time_ms"] for r in entries) / len(entries) if entries else 0
        avg_score = sum(r["best_score"] for r in entries) / len(entries) if entries else 0
        avg_tps = sum(r["aggregate_tps"] for r in entries) / len(entries) if entries else 0
        print(
            f"  {n_agents} agents: "
            f"avg_time={avg_time / 1000:.1f}s, "
            f"avg_score={avg_score:.2f}, "
            f"avg_tps={avg_tps:.0f}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Benchmark results saved to %s", args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel Problem-Solving: use batch GPU capacity to solve ONE problem faster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api", default="http://localhost:8000", help="vLLM API base URL")
    parser.add_argument("--model", default="default", help="Model name on vLLM")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per agent")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per agent (seconds)")
    parser.add_argument("--max-prefix-tokens", type=int, default=50000, help="Max codebase prefix tokens")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # solve
    p_solve = subparsers.add_parser("solve", help="Solve a problem with parallel agents")
    p_solve.add_argument("--problem", help="Problem description")
    p_solve.add_argument("--problem-file", help="File containing problem description")
    p_solve.add_argument("--codebase", required=True, help="Path to codebase directory or file")
    p_solve.add_argument("--agents", type=int, default=4, help="Number of parallel agents")
    p_solve.add_argument("--strategies", help="Comma-separated strategies (direct,review,...)")
    p_solve.add_argument("--preset", choices=list(STRATEGY_PRESETS.keys()), help="Strategy preset")
    p_solve.add_argument("--tests", help="Path to test file or directory")
    p_solve.add_argument("--merge", action="store_true", help="Merge top solutions")
    p_solve.add_argument("--stream", action="store_true", help="Stream agent output in real-time")
    p_solve.add_argument("--output", help="Save results to JSON file")

    # interactive
    p_int = subparsers.add_parser("interactive", help="Interactive problem-solving mode")
    p_int.add_argument("--codebase", required=True, help="Path to codebase directory or file")
    p_int.add_argument("--agents", type=int, default=4, help="Number of parallel agents")
    p_int.add_argument("--preset", choices=list(STRATEGY_PRESETS.keys()), help="Default strategy preset")
    p_int.add_argument("--merge", action="store_true", help="Merge top solutions")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark at different agent counts")
    p_bench.add_argument("--problems", required=True, help="Path to problems JSON file")
    p_bench.add_argument("--agents", default="1,2,4,8", help="Comma-separated agent counts")
    p_bench.add_argument("--output", help="Save benchmark results to JSON file")

    args = parser.parse_args()

    if args.command == "solve":
        asyncio.run(cmd_solve(args))
    elif args.command == "interactive":
        asyncio.run(cmd_interactive(args))
    elif args.command == "benchmark":
        asyncio.run(cmd_benchmark(args))


if __name__ == "__main__":
    main()
