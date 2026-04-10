"""Standalone CLI for the Fusen parallel solver.

Usage:
    fusen-solver solve --problem "Fix the bug" --codebase ./src/ --agents 4
    fusen-solver interactive --codebase ./src/
    fusen-solver stats
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from fusen_solver.config import load_config, Config
from fusen_solver.core.interfaces import Problem, Solution
from fusen_solver.core.solver import FusenSolver, SolveResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
COLORS = ["\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]


def _color(idx: int) -> str:
    return COLORS[idx % len(COLORS)]


def _make_backend(config: Config):
    """Create the LLM backend from config."""
    primary = config.backends.get("primary", {})
    backend_type = primary.get("type", "vllm")

    if backend_type == "vllm":
        from fusen_solver.backends.vllm_backend import VLLMBackend

        return VLLMBackend(
            base_url=primary.get("url", "http://localhost:8000/v1"),
            model=primary.get("model", "default"),
            max_context_tokens=primary.get("max_context", 131072),
        )
    elif backend_type == "openai":
        from fusen_solver.backends.openai_backend import OpenAIBackend

        return OpenAIBackend(
            api_key=primary.get("api_key"),
            model=primary.get("model", "gpt-4o"),
            max_context_tokens=primary.get("max_context", 128000),
        )
    elif backend_type == "anthropic":
        from fusen_solver.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(
            api_key=primary.get("api_key"),
            model=primary.get("model", "claude-sonnet-4-20250514"),
            max_context_tokens=primary.get("max_context", 200000),
        )
    elif backend_type == "ollama":
        from fusen_solver.backends.ollama_backend import OllamaBackend

        return OllamaBackend(
            model=primary.get("model", "llama3:70b"),
            base_url=primary.get("url", "http://localhost:11434"),
            max_context_tokens=primary.get("max_context", 8192),
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def _load_codebase(path: str | Path, max_chars: int = 200000) -> dict[str, str]:
    """Load a codebase directory into a filename -> content dict."""
    root = Path(path)

    if root.is_file():
        return {root.name: root.read_text(errors="replace")[:max_chars]}

    context: dict[str, str] = {}
    total_chars = 0
    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".mypy_cache"}

    for f in sorted(root.rglob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True):
        rel = f.relative_to(root)
        if any(part in skip_dirs for part in rel.parts):
            continue
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue

        if total_chars + len(text) > max_chars:
            break

        context[str(rel)] = text
        total_chars += len(text)

    return context


def _print_result(result: SolveResult) -> None:
    """Pretty-print solve results."""
    mode_str = f" ({result.mode})" if result.mode != "isolated" else ""
    rounds_str = f", {len(result.rounds)} rounds" if result.rounds else ""
    print(f"\n{BOLD}{'=' * 70}")
    print(f"  Results: {result.num_agents} agents{mode_str}{rounds_str}, {result.total_time_s:.1f}s total")
    print(f"{'=' * 70}{RESET}\n")

    if result.solutions:
        print(f"{BOLD}  Ranked Solutions:{RESET}")
        for i, sol in enumerate(result.solutions):
            marker = " <-- BEST" if i == 0 else ""
            c = _color(i)
            subs = ", ".join(f"{k}={v:.2f}" for k, v in sol.subscores.items())
            print(
                f"  {c}#{i} ({sol.strategy_used}){RESET}: "
                f"score={sol.score:.2f} ({subs}){BOLD}{marker}{RESET}"
            )
        print()

    if result.best:
        print(f"{BOLD}  Best Solution ({result.best.strategy_used}):{RESET}")
        print(f"  {DIM}{result.best.explanation[:500]}{RESET}")
        print()

    if result.merged:
        print(f"{BOLD}  Merged Solution:{RESET}")
        print(f"  {DIM}{result.merged.explanation[:500]}{RESET}")
        print()

    # Racing stats
    racing_stats = result.metadata.get("racing_stats") if result.metadata else None
    if racing_stats:
        print(f"{BOLD}  Racing Stats:{RESET}")
        print(f"    Winner: agent #{racing_stats['winner_idx']}")
        print(f"    Winner time: {racing_stats['winner_time_s']:.2f}s")
        print(f"    Cancelled agents: {racing_stats['cancelled_agents']}")
        print(f"    KV cache saved: {racing_stats['kv_savings_pct']:.0f}%")
        if racing_stats['timed_out']:
            print(f"    {BOLD}(timed out -- took best available){RESET}")
        if racing_stats['rejections_before_accept'] > 0:
            print(f"    Rejections before accept: {racing_stats['rejections_before_accept']}")
        print()


async def cmd_solve(args: argparse.Namespace) -> None:
    """Solve a problem using parallel agents."""
    config = load_config(args.config)
    backend = _make_backend(config)
    solver = FusenSolver(
        backend=backend,
        default_n=args.agents,
        auto_n=not args.no_auto_n,
    )

    # Load inputs
    context = _load_codebase(args.codebase)
    problem_text = args.problem
    if args.problem_file:
        problem_text = Path(args.problem_file).read_text()

    if not problem_text:
        print("Error: provide --problem or --problem-file", file=sys.stderr)
        sys.exit(1)

    problem = Problem(
        description=problem_text,
        context=context,
        problem_type=args.type or "auto",
        solve_mode=args.mode or "auto",
        max_rounds=args.rounds,
        racing_accept_threshold=args.racing_threshold,
        racing_timeout=args.racing_timeout,
    )

    mode_label = f", mode={problem.solve_mode}" if problem.solve_mode != "auto" else ""
    print(f"\n{BOLD}Fusen Solver: {args.agents} agents{mode_label}{RESET}")
    print(f"  Problem: {problem_text[:200]}")
    print(f"  Codebase: {args.codebase} ({len(context)} files)")
    if problem.solve_mode == "collaborative":
        print(f"  Rounds: {problem.max_rounds}")
    if problem.solve_mode == "racing":
        print(f"  Racing: threshold={problem.racing_accept_threshold}, timeout={problem.racing_timeout}s")
    print()

    result = await solver.solve(problem, merge=args.merge)
    _print_result(result)

    if args.output:
        output_data = {
            "problem": problem_text,
            "strategies": result.strategies_used,
            "total_time_s": result.total_time_s,
            "solutions": [
                {
                    "strategy": s.strategy_used,
                    "score": s.score,
                    "subscores": s.subscores,
                    "explanation": s.explanation[:1000],
                }
                for s in result.solutions
            ],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        logger.info("Results saved to %s", args.output)


async def cmd_interactive(args: argparse.Namespace) -> None:
    """Interactive mode: enter problems one at a time."""
    config = load_config(args.config)
    backend = _make_backend(config)
    solver = FusenSolver(
        backend=backend,
        default_n=args.agents,
    )

    context = _load_codebase(args.codebase)

    print(f"\n{BOLD}Fusen Solver Interactive Mode{RESET}")
    print(f"  Codebase: {args.codebase} ({len(context)} files)")
    print(f"  Agents: {args.agents}")
    print(f"\n  Type a problem and press Enter. 'quit' to exit.\n")

    while True:
        try:
            problem_text = input(f"{BOLD}Problem> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        problem_text = problem_text.strip()
        if not problem_text or problem_text.lower() in ("quit", "exit", "q"):
            if not problem_text:
                continue
            print("Bye.")
            break

        problem = Problem(description=problem_text, context=context)
        result = await solver.solve(problem)
        _print_result(result)


async def cmd_shadow(args: argparse.Namespace) -> None:
    """Run fusen_solver as a shadow on a task, optionally comparing to BCode output."""
    from fusen_solver.integrations.bcode_shadow import BCodeShadow

    config = load_config(args.config)
    shadow = BCodeShadow(
        config=config,
        timeout_s=args.timeout,
    )

    # Load task/PRD
    prd = args.prd
    if args.prd_file:
        prd = Path(args.prd_file).read_text()
    if not prd:
        print("Error: provide --prd or --prd-file", file=sys.stderr)
        sys.exit(1)

    # Load optional BCode output for comparison
    bcode_output = None
    if args.bcode_output:
        bcode_output = json.loads(Path(args.bcode_output).read_text())

    print(f"\n{BOLD}Fusen Solver Shadow Mode{RESET}")
    print(f"  Task: {prd[:200]}")
    print(f"  Workspace: {args.workspace}")
    if bcode_output:
        print(f"  BCode output: {args.bcode_output}")
    print()

    result = await shadow.shadow_run(
        task=prd,
        codebase_path=args.workspace,
        bcode_output=bcode_output,
    )

    print(f"{BOLD}Shadow Result:{RESET}")
    print(f"  Fusen score: {result.fusen_score:.3f}")
    print(f"  Fusen time:  {result.fusen_time_s:.1f}s")
    print(f"  Fusen files: {result.fusen_files}")
    print(f"  Fusen mode:  {result.fusen_mode_used}")
    print(f"  Strategy:    {result.fusen_strategy_used}")
    if bcode_output:
        print(f"  BCode score: {result.bcode_score:.3f}")
        print(f"  Winner:      {BOLD}{result.winner}{RESET}")
    if result.notes:
        print(f"  Notes:       {result.notes}")
    print()


async def cmd_shadow_stats(args: argparse.Namespace) -> None:
    """Show aggregated shadow comparison statistics."""
    from fusen_solver.integrations.bcode_shadow import BCodeShadow

    shadow = BCodeShadow()
    stats = shadow.get_stats()

    if stats["total_runs"] == 0:
        print("No shadow runs recorded yet. Run 'fusen-solver shadow' first.")
        return

    print(f"\n{BOLD}Shadow Mode Statistics{RESET}\n")
    print(f"  Total runs:     {stats['total_runs']}")
    print(f"  Fusen win rate: {stats['fusen_win_rate']:.1%}")
    print(f"  BCode win rate: {stats['bcode_win_rate']:.1%}")
    print(f"  Tie rate:       {stats['tie_rate']:.1%}")
    print(f"  Avg fusen score: {stats['avg_fusen_score']:.3f}")
    print(f"  Avg bcode score: {stats['avg_bcode_score']:.3f}")
    print(f"  Avg fusen time:  {stats['avg_fusen_time_s']:.1f}s")
    print(f"  Avg bcode time:  {stats['avg_bcode_time_s']:.1f}s")
    if stats.get("trend"):
        print(f"  Trend:          {stats['trend']}")
    print()


async def cmd_shadow_promote(args: argparse.Namespace) -> None:
    """Check if fusen_solver should be promoted from shadow to primary."""
    from fusen_solver.integrations.bcode_shadow import BCodeShadow

    shadow = BCodeShadow()
    stats = shadow.get_stats()
    should = shadow.should_promote(
        min_runs=args.min_runs,
        min_win_rate=args.min_win_rate,
    )

    print(f"\n{BOLD}Promotion Check{RESET}\n")
    print(f"  Total runs:     {stats['total_runs']} (min: {args.min_runs})")
    print(f"  Fusen win rate: {stats['fusen_win_rate']:.1%} (min: {args.min_win_rate:.1%})")

    if should:
        print(f"\n  {BOLD}PROMOTE: fusen_solver meets promotion criteria.{RESET}")
    else:
        reasons = []
        if stats["total_runs"] < args.min_runs:
            reasons.append(f"need {args.min_runs - stats['total_runs']} more runs")
        if stats["fusen_win_rate"] < args.min_win_rate:
            reasons.append(f"win rate {stats['fusen_win_rate']:.1%} < {args.min_win_rate:.1%}")
        print(f"\n  NOT YET: {'; '.join(reasons)}")
    print()


async def cmd_stats(args: argparse.Namespace) -> None:
    """Show learning engine statistics."""
    from fusen_solver.learning.tracker import AgentMemory, LearningEngine

    engine = LearningEngine()
    stats = engine.get_stats()

    if not stats:
        print("No learning data yet. Solve some problems first.")
    else:
        print(f"\n{BOLD}Learning Engine Stats{RESET}\n")
        for ptype, strategies in stats.items():
            if ptype.startswith("_"):
                continue
            print(f"  {BOLD}{ptype}{RESET}:")
            for name, data in strategies.items():
                print(
                    f"    {name}: {data['wins']}/{data['attempts']} wins "
                    f"({data['win_rate']:.1%}), confidence={data['confidence']:.1%}"
                )
            print()

    # Show agent memory stats
    memory = AgentMemory()
    all_memories = memory.get_all()
    if all_memories:
        print(f"\n{BOLD}Agent Memory ({len(all_memories)} insights){RESET}\n")
        by_type: dict[str, list] = {}
        for m in all_memories:
            by_type.setdefault(m["type"], []).append(m)
        for ptype, mems in sorted(by_type.items()):
            print(f"  {BOLD}{ptype}{RESET}: {len(mems)} insights")
            for m in mems[-3:]:  # show last 3 per type
                used = m.get("used_count", 0)
                print(f"    - {m['insight'][:100]} (used {used}x)")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fusen Solver: parallel AI problem solving for any coding platform",
    )
    parser.add_argument("--config", help="Path to config YAML file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # solve
    p_solve = subparsers.add_parser("solve", help="Solve a problem with parallel agents")
    p_solve.add_argument("--problem", help="Problem description")
    p_solve.add_argument("--problem-file", help="File containing problem description")
    p_solve.add_argument("--codebase", required=True, help="Path to codebase")
    p_solve.add_argument("--agents", type=int, default=4, help="Number of parallel agents")
    p_solve.add_argument("--type", help="Problem type (bug_fix, feature, refactor, ...)")
    p_solve.add_argument("--merge", action="store_true", help="Merge top solutions")
    p_solve.add_argument("--no-auto-n", action="store_true", help="Disable auto agent count")
    p_solve.add_argument(
        "--mode",
        choices=["isolated", "collaborative", "decomposed", "racing", "auto"],
        default="auto",
        help="Solve mode: isolated (single round), collaborative (multi-round), decomposed (per-file parallel), racing (first-wins), auto (data-driven)",
    )
    p_solve.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Max rounds for collaborative mode (default: 3)",
    )
    p_solve.add_argument(
        "--racing-threshold",
        type=float,
        default=0.7,
        help="Score threshold for accepting a racing solution (default: 0.7)",
    )
    p_solve.add_argument(
        "--racing-timeout",
        type=float,
        default=30.0,
        help="Max seconds for racing mode before taking best available (default: 30)",
    )
    p_solve.add_argument("--output", help="Save results to JSON file")
    p_solve.add_argument(
        "--session-id",
        help="Session ID for backend affinity (e.g. hash of codebase). "
             "Ensures all requests for the same codebase hit the same GPU.",
    )

    # interactive
    p_int = subparsers.add_parser("interactive", help="Interactive mode")
    p_int.add_argument("--codebase", required=True, help="Path to codebase")
    p_int.add_argument("--agents", type=int, default=4, help="Number of parallel agents")

    # shadow
    p_shadow = subparsers.add_parser("shadow", help="Run shadow comparison against BCode")
    p_shadow.add_argument("--prd", help="Task / PRD description")
    p_shadow.add_argument("--prd-file", help="File containing PRD description")
    p_shadow.add_argument("--workspace", required=True, help="Path to workspace/codebase")
    p_shadow.add_argument("--bcode-output", help="Path to BCode result JSON (for comparison)")
    p_shadow.add_argument(
        "--timeout", type=float, default=120.0,
        help="Max seconds for shadow run (default: 120)",
    )

    # shadow-stats
    subparsers.add_parser("shadow-stats", help="Show shadow comparison statistics")

    # shadow-promote
    p_promote = subparsers.add_parser("shadow-promote", help="Check if fusen_solver should be promoted")
    p_promote.add_argument(
        "--min-runs", type=int, default=50,
        help="Minimum shadow runs required (default: 50)",
    )
    p_promote.add_argument(
        "--min-win-rate", type=float, default=0.6,
        help="Minimum fusen win rate for promotion (default: 0.6)",
    )

    # stats
    subparsers.add_parser("stats", help="Show learning engine statistics")

    args = parser.parse_args()

    if args.command == "solve":
        asyncio.run(cmd_solve(args))
    elif args.command == "interactive":
        asyncio.run(cmd_interactive(args))
    elif args.command == "shadow":
        asyncio.run(cmd_shadow(args))
    elif args.command == "shadow-stats":
        asyncio.run(cmd_shadow_stats(args))
    elif args.command == "shadow-promote":
        asyncio.run(cmd_shadow_promote(args))
    elif args.command == "stats":
        asyncio.run(cmd_stats(args))


if __name__ == "__main__":
    main()
