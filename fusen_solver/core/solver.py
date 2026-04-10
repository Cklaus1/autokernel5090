"""Main FusenSolver class -- the central orchestrator.

Takes a Problem, selects strategies, dispatches N parallel LLM calls,
scores the results, and returns the best Solution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fusen_solver.core.interfaces import (
    LLMBackend,
    Problem,
    Solution,
    Strategy,
)
from fusen_solver.core.priority import compute_priority
from fusen_solver.learning.tracker import AgentMemory, LearningEngine
from fusen_solver.scoring.engine import ScoringEngine
from fusen_solver.strategies.engine import StrategyEngine

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    """Full result of a parallel solve operation."""

    problem: Problem
    solutions: list[Solution] = field(default_factory=list)
    best: Solution | None = None
    merged: Solution | None = None
    strategies_used: list[str] = field(default_factory=list)
    # Timing
    total_time_s: float = 0.0
    generation_time_s: float = 0.0
    scoring_time_s: float = 0.0
    # Stats
    num_agents: int = 0
    # Collaborative mode fields
    mode: str = "isolated"  # "isolated", "collaborative", "decomposed", or "racing"
    rounds: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FusenSolver:
    """Universal parallel problem solver.

    Usage:
        from fusen_solver import FusenSolver
        from fusen_solver.backends import OpenAIBackend

        solver = FusenSolver(backend=OpenAIBackend(api_key="..."))
        result = await solver.solve(Problem(
            description="Fix the race condition in server.py",
            context={"server.py": open("server.py").read()},
            problem_type="bug_fix",
        ))
        print(result.best.explanation)
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        strategy_engine: StrategyEngine | None = None,
        scoring_engine: ScoringEngine | None = None,
        learning_engine: LearningEngine | None = None,
        memory: AgentMemory | None = None,
        max_tokens: int = 4096,
        default_n: int = 4,
        auto_n: bool = True,
    ):
        self.backend = backend
        self.strategy_engine = strategy_engine or StrategyEngine()
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.learning_engine = learning_engine or LearningEngine()
        self.memory = memory or AgentMemory()
        self.max_tokens = max_tokens
        self.default_n = default_n
        self.auto_n = auto_n

    async def solve(
        self,
        problem: Problem,
        *,
        n: int | None = None,
        strategies: list[Strategy] | None = None,
        merge: bool = False,
        on_solution: Callable[[int, Solution], None] | None = None,
    ) -> SolveResult:
        """Solve a problem using N parallel agents with diverse strategies.

        Supports four solve modes via ``problem.solve_mode``:
        - ``"isolated"`` (default legacy behavior): single round of parallel agents.
        - ``"collaborative"``: multi-round solving with specialized roles.
        - ``"decomposed"``: split into per-file sub-problems, solve in parallel,
          then merge and verify integration.
        - ``"auto"``: data-driven selection between isolated, collaborative,
          and decomposed.

        Args:
            problem: The problem to solve.
            n: Number of parallel agents. None = auto-select.
            strategies: Override strategy selection. None = auto-select.
            merge: Whether to merge top solutions into a final one.
            on_solution: Callback when each agent finishes (agent_index, solution).

        Returns:
            SolveResult with ranked solutions and optional merge.
        """
        # Determine mode
        mode = problem.solve_mode
        if mode == "auto":
            mode = self.learning_engine.suggest_mode(problem)
            logger.info("Auto mode selected: %s", mode)

        if mode == "collaborative":
            return await self.solve_collaborative(problem)

        if mode == "decomposed":
            return await self.solve_decomposed(problem)

        if mode == "racing":
            return await self.solve_racing(problem, n=n, strategies=strategies)

        return await self.solve_isolated(
            problem, n=n, strategies=strategies, merge=merge, on_solution=on_solution
        )

    async def solve_isolated(
        self,
        problem: Problem,
        *,
        n: int | None = None,
        strategies: list[Strategy] | None = None,
        merge: bool = False,
        on_solution: Callable[[int, Solution], None] | None = None,
    ) -> SolveResult:
        """Solve a problem using N parallel agents in a single round (original behavior).

        Args:
            problem: The problem to solve.
            n: Number of parallel agents. None = auto-select.
            strategies: Override strategy selection. None = auto-select.
            merge: Whether to merge top solutions into a final one.
            on_solution: Callback when each agent finishes (agent_index, solution).

        Returns:
            SolveResult with ranked solutions and optional merge.
        """
        t_start = time.perf_counter()

        # 1. Determine number of agents
        if n is None:
            if self.auto_n:
                n = self.learning_engine.suggest_n(problem)
            else:
                n = self.default_n

        # 2. Select strategies
        if strategies is None:
            weights = self.learning_engine.get_weights(problem.problem_type)
            strategies = self.strategy_engine.select_strategies(
                problem, n=n, weights=weights
            )

        strategies = strategies[:n]

        result = SolveResult(
            problem=problem,
            strategies_used=[s.name for s in strategies],
            num_agents=len(strategies),
            mode="isolated",
        )
        logger.info(
            "Solving (isolated) with %d agents: %s",
            len(strategies),
            [s.name for s in strategies],
        )

        # 3. Build prompts and run agents in parallel
        t_gen = time.perf_counter()
        codebase_text = self._format_codebase(problem.context)

        tasks = [
            self._run_agent(problem, strategy, codebase_text, i)
            for i, strategy in enumerate(strategies)
        ]
        raw_solutions = await asyncio.gather(*tasks, return_exceptions=True)
        result.generation_time_s = time.perf_counter() - t_gen

        # 4. Collect valid solutions
        solutions: list[Solution] = []
        for i, sol in enumerate(raw_solutions):
            if isinstance(sol, Exception):
                logger.warning("Agent %d failed: %s", i, sol)
                continue
            if on_solution is not None:
                on_solution(i, sol)
            solutions.append(sol)

        # 5. Score solutions
        t_score = time.perf_counter()
        if solutions:
            scored = await self.scoring_engine.score_all(
                problem, solutions, backend=self.backend
            )
            scored.sort(key=lambda s: s.score, reverse=True)
            result.solutions = scored
            result.best = scored[0] if scored else None
        result.scoring_time_s = time.perf_counter() - t_score

        # 6. Optionally merge top solutions
        if merge and len(result.solutions) >= 2:
            result.merged = await self._merge_solutions(
                problem, result.solutions[:4], codebase_text
            )

        result.total_time_s = time.perf_counter() - t_start
        logger.info(
            "Solve complete: %d solutions, best=%.2f, %.1fs total",
            len(result.solutions),
            result.best.score if result.best else 0.0,
            result.total_time_s,
        )

        return result

    async def solve_collaborative(self, problem: Problem) -> SolveResult:
        """Solve a problem using multi-round collaborative agents.

        Each round runs specialized roles in parallel. Outputs from earlier
        rounds are synthesized and fed as context to subsequent rounds.
        Supports early exit when a round produces a verified solution.

        Args:
            problem: The problem to solve (uses problem.max_rounds).

        Returns:
            SolveResult with collaborative round history and best solution.
        """
        from fusen_solver.strategies.presets import COLLABORATIVE_ROLES

        t_start = time.perf_counter()

        context: dict[str, Any] = {
            "problem": problem.description,
            "codebase": self._format_codebase(problem.context),
        }
        all_rounds: list[dict[str, Any]] = []
        all_solutions: list[Solution] = []
        total_agents = 0

        for round_num in range(1, problem.max_rounds + 1):
            round_key = f"round_{round_num}"
            roles = COLLABORATIVE_ROLES.get(round_key, [])
            if not roles:
                logger.info("No roles defined for %s, stopping.", round_key)
                break

            logger.info(
                "Collaborative round %d: %d roles (%s)",
                round_num,
                len(roles),
                [r.name for r in roles],
            )

            # Build prompts for each role and run in parallel
            tasks = []
            for role in roles:
                prompt = self._build_collaborative_prompt(role, context, round_num)
                tasks.append(
                    self.backend.generate(
                        prompt, max_tokens=self.max_tokens, temperature=0.7
                    )
                )

            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            round_outputs: list[dict[str, Any]] = []
            for i, (role, raw) in enumerate(zip(roles, raw_results)):
                if isinstance(raw, Exception):
                    logger.warning("Round %d, role %s failed: %s", round_num, role.name, raw)
                    round_outputs.append({
                        "role": role.name,
                        "output": f"(failed: {raw})",
                        "success": False,
                    })
                else:
                    round_outputs.append({
                        "role": role.name,
                        "output": raw,
                        "success": True,
                    })
                    # Build a Solution from each successful agent output
                    sol = Solution(
                        code=self._extract_code_blocks(raw),
                        explanation=raw,
                        strategy_used=f"collaborative_{round_key}_{role.name}",
                        metadata={"round": round_num, "role": role.name},
                    )
                    all_solutions.append(sol)

            total_agents += len(roles)

            # Synthesize round outputs into context for next round
            synthesis = await self._synthesize_round(round_outputs, round_num, context)
            context[round_key] = synthesis
            all_rounds.append(synthesis)

            # Early exit: if synthesis indicates a verified solution
            if synthesis.get("has_solution") and synthesis.get("tests_pass"):
                logger.info("Early exit at round %d: verified solution found.", round_num)
                break

        # Score all collected solutions
        t_score = time.perf_counter()
        best: Solution | None = None
        if all_solutions:
            scored = await self.scoring_engine.score_all(
                problem, all_solutions, backend=self.backend
            )
            scored.sort(key=lambda s: s.score, reverse=True)
            all_solutions = scored
            best = scored[0] if scored else None
        scoring_time = time.perf_counter() - t_score

        result = SolveResult(
            problem=problem,
            solutions=all_solutions,
            best=best,
            strategies_used=[s.strategy_used for s in all_solutions],
            total_time_s=time.perf_counter() - t_start,
            scoring_time_s=scoring_time,
            num_agents=total_agents,
            mode="collaborative",
            rounds=all_rounds,
        )

        logger.info(
            "Collaborative solve complete: %d rounds, %d solutions, best=%.2f, %.1fs total",
            len(all_rounds),
            len(all_solutions),
            best.score if best else 0.0,
            result.total_time_s,
        )

        return result

    async def solve_racing(
        self,
        problem: Problem,
        *,
        n: int | None = None,
        strategies: list[Strategy] | None = None,
    ) -> SolveResult:
        """Race N agents. First accepted solution wins. Cancel losers.

        Launches all agents as async tasks, uses ``asyncio.wait(FIRST_COMPLETED)``
        to get results as they arrive. When a solution scores above the accept
        threshold, all remaining agents are cancelled (HTTP connections closed,
        freeing vLLM KV cache). If no solution passes threshold before timeout,
        takes the best available.

        Args:
            problem: The problem to solve (uses racing_accept_threshold, racing_timeout).
            n: Number of parallel agents. None = auto-select.
            strategies: Override strategy selection. None = auto-select.

        Returns:
            SolveResult with racing stats in metadata.
        """
        from fusen_solver.streaming import RacingCoordinator

        t_start = time.perf_counter()

        # 1. Determine number of agents
        if n is None:
            if self.auto_n:
                n = self.learning_engine.suggest_n(problem)
            else:
                n = self.default_n

        # 2. Select strategies
        if strategies is None:
            weights = self.learning_engine.get_weights(problem.problem_type)
            strategies = self.strategy_engine.select_strategies(
                problem, n=n, weights=weights
            )
        strategies = strategies[:n]

        coordinator = RacingCoordinator(
            accept_threshold=problem.racing_accept_threshold,
            timeout=problem.racing_timeout,
        )

        result = SolveResult(
            problem=problem,
            strategies_used=[s.name for s in strategies],
            num_agents=len(strategies),
            mode="racing",
        )

        logger.info(
            "Racing solve with %d agents (threshold=%.2f, timeout=%.1fs): %s",
            len(strategies),
            problem.racing_accept_threshold,
            problem.racing_timeout,
            [s.name for s in strategies],
        )

        # 3. Build prompts and launch agents as tasks
        codebase_text = self._format_codebase(problem.context)
        pending: set[asyncio.Task[Solution]] = set()
        task_to_idx: dict[asyncio.Task[Solution], int] = {}

        for i, strategy in enumerate(strategies):
            req = coordinator.register(i)
            task = asyncio.create_task(
                self._run_agent(problem, strategy, codebase_text, i),
                name=f"racing_agent_{i}",
            )
            req.task = task
            pending.add(task)
            task_to_idx[task] = i

        # 4. Wait for completions, score each, accept or continue
        t_gen = time.perf_counter()
        accepted_solution: Solution | None = None
        accepted_idx: int = -1
        all_solutions: list[Solution] = []
        rejections = 0
        deadline = t_start + problem.racing_timeout

        while pending:
            remaining_time = deadline - time.perf_counter()
            if remaining_time <= 0:
                logger.info("Racing timeout reached (%.1fs)", problem.racing_timeout)
                coordinator.stats.timed_out = True
                await coordinator.cancel_all()
                break

            done, pending = await asyncio.wait(
                pending,
                timeout=remaining_time,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                logger.info("Racing timeout reached (%.1fs)", problem.racing_timeout)
                coordinator.stats.timed_out = True
                await coordinator.cancel_all()
                break

            for task in done:
                agent_idx = task_to_idx[task]
                # Mark the request as completed
                for req in coordinator._requests:
                    if req.agent_idx == agent_idx:
                        req.end_time = time.perf_counter()
                        break

                if task.cancelled():
                    continue

                exc = task.exception()
                if exc is not None:
                    logger.warning("Racing agent %d failed: %s", agent_idx, exc)
                    continue

                sol = task.result()

                # Score this solution
                try:
                    scored = await self.scoring_engine.score_all(
                        problem, [sol], backend=self.backend
                    )
                    if scored:
                        sol = scored[0]
                except Exception as e:
                    logger.warning("Scoring failed for agent %d: %s", agent_idx, e)

                all_solutions.append(sol)

                logger.info(
                    "Racing agent %d finished: score=%.2f (threshold=%.2f)",
                    agent_idx,
                    sol.score,
                    problem.racing_accept_threshold,
                )

                # Check acceptance
                if sol.score >= problem.racing_accept_threshold:
                    accepted_solution = sol
                    accepted_idx = agent_idx
                    logger.info(
                        "Racing: agent %d accepted (score=%.2f). Cancelling %d remaining.",
                        agent_idx,
                        sol.score,
                        len(pending),
                    )
                    await coordinator.cancel_all_except(agent_idx)
                    pending.clear()
                    break
                else:
                    rejections += 1
                    logger.info(
                        "Racing: agent %d rejected (score=%.2f < %.2f), %d agents remaining.",
                        agent_idx,
                        sol.score,
                        problem.racing_accept_threshold,
                        len(pending),
                    )

        result.generation_time_s = time.perf_counter() - t_gen

        # 5. Pick best solution
        if all_solutions:
            all_solutions.sort(key=lambda s: s.score, reverse=True)
            result.solutions = all_solutions
            result.best = all_solutions[0]

        # 6. Finalize stats
        coordinator.stats.rejections_before_accept = rejections
        winner_time = 0.0
        if accepted_idx >= 0:
            for req in coordinator._requests:
                if req.agent_idx == accepted_idx:
                    winner_time = req.elapsed
                    break
        racing_stats = coordinator.finalize_stats(
            winner_idx=accepted_idx,
            winner_time=winner_time,
        )

        result.total_time_s = time.perf_counter() - t_start
        result.metadata = {
            "racing_stats": {
                "winner_idx": racing_stats.winner_idx,
                "winner_time_s": racing_stats.winner_time_s,
                "cancelled_agents": racing_stats.cancelled_agents,
                "estimated_tokens_saved": racing_stats.estimated_tokens_saved,
                "kv_savings_pct": racing_stats.kv_savings_pct,
                "timed_out": racing_stats.timed_out,
                "rejections_before_accept": racing_stats.rejections_before_accept,
                "agent_times": racing_stats.agent_times,
            },
        }

        # 7. Record racing win position in learning engine
        if accepted_idx >= 0:
            self.learning_engine.record_racing_win(
                problem_type=problem.problem_type if problem.problem_type != "auto" else "unknown",
                num_agents=len(strategies),
                winner_position=accepted_idx,
                winner_time=winner_time,
            )

        logger.info(
            "Racing solve complete: %d solutions, best=%.2f, winner=agent_%d, "
            "cancelled=%d, kv_saved=%.0f%%, %.1fs total",
            len(result.solutions),
            result.best.score if result.best else 0.0,
            accepted_idx,
            racing_stats.cancelled_agents,
            racing_stats.kv_savings_pct,
            result.total_time_s,
        )

        return result

    # ------------------------------------------------------------------
    # Decomposed mode: per-file parallel generation
    # ------------------------------------------------------------------

    # Common decomposition patterns for well-known project types.
    DECOMPOSITION_PATTERNS: dict[str, list[str]] = {
        "rest_api": ["models.py", "routes.py", "middleware.py", "tests/test_api.py"],
        "cli_tool": ["cli.py", "core.py", "utils.py", "tests/test_core.py"],
        "library": ["__init__.py", "core.py", "helpers.py", "types.py", "tests/test_core.py"],
    }

    async def solve_decomposed(self, problem: Problem) -> SolveResult:
        """Decompose problem into per-file/per-module sub-problems, solve in parallel.

        Steps:
        1. Ask the LLM to decompose the problem into a list of files with
           descriptions and dependency ordering.
        2. Generate independent files in parallel; dependent files wait for
           their dependencies so earlier outputs serve as context.
        3. Merge all file solutions into one combined solution.
        4. Run an integration verification pass to fix cross-file conflicts.

        Args:
            problem: The problem to solve.

        Returns:
            SolveResult with mode="decomposed".
        """
        t_start = time.perf_counter()

        # Step 1: Decompose into files
        decomposition = await self._decompose_into_files(problem)
        logger.info(
            "Decomposed into %d files: %s",
            len(decomposition),
            [f["file"] for f in decomposition],
        )

        # Step 2: Generate each file, respecting dependency order
        levels = self._build_dependency_levels(decomposition)
        all_file_results: list[tuple[dict[str, Any], SolveResult]] = []
        generated_so_far: dict[str, str] = {}

        for level in levels:
            tasks = []
            level_specs = []
            for file_spec in level:
                # Include already-generated dependency code as extra context
                dep_context = dict(problem.context)
                for dep_file in file_spec.get("depends_on", []):
                    if dep_file in generated_so_far:
                        dep_context[dep_file] = generated_so_far[dep_file]

                sub_problem = Problem(
                    description=(
                        f"Write the file `{file_spec['file']}`: "
                        f"{file_spec['description']}\n\n"
                        f"This is part of a larger project: {problem.description}\n\n"
                        "Write ONLY the contents of this single file. "
                        "Include all imports, classes, and functions needed."
                    ),
                    context=dep_context,
                    problem_type=problem.problem_type,
                    solve_mode="isolated",
                    constraints=problem.constraints,
                    language=problem.language,
                )
                tasks.append(self.solve_isolated(sub_problem, n=1))
                level_specs.append(file_spec)

            level_results = await asyncio.gather(*tasks, return_exceptions=True)

            for file_spec, result in zip(level_specs, level_results):
                if isinstance(result, Exception):
                    logger.warning(
                        "Failed to generate %s: %s", file_spec["file"], result
                    )
                    continue
                all_file_results.append((file_spec, result))
                if result.best and result.best.code:
                    code_values = list(result.best.code.values())
                    if code_values:
                        generated_so_far[file_spec["file"]] = code_values[0]

        t_gen = time.perf_counter() - t_start

        # Step 3: Merge all file solutions
        merged = self._merge_file_solutions(decomposition, all_file_results)

        # Step 4: Integration verification
        integration_result = await self._verify_integration(merged, problem)

        t_total = time.perf_counter() - t_start
        final_solution = integration_result if integration_result else merged

        result = SolveResult(
            problem=problem,
            solutions=[final_solution],
            best=final_solution,
            strategies_used=["decomposed"],
            total_time_s=t_total,
            generation_time_s=t_gen,
            num_agents=len(all_file_results),
            mode="decomposed",
            rounds=[{
                "decomposition": decomposition,
                "files_generated": list(generated_so_far.keys()),
                "integration_verified": integration_result is not None,
            }],
        )

        logger.info(
            "Decomposed solve complete: %d files, %.1fs total",
            len(all_file_results),
            t_total,
        )

        return result

    async def _decompose_into_files(
        self, problem: Problem
    ) -> list[dict[str, Any]]:
        """Ask the LLM to decompose a problem into per-file specifications.

        Returns a list of dicts with keys:
        - ``file``: filename (e.g. ``"routes.py"``)
        - ``description``: what this file should contain
        - ``depends_on``: list of filenames this file depends on

        Falls back to a known ``DECOMPOSITION_PATTERNS`` entry when the LLM
        output cannot be parsed.
        """
        import json as _json
        import re

        # Check if the description matches a known pattern
        desc_lower = problem.description.lower()
        matched_pattern: str | None = None
        for pattern_name, keywords in {
            "rest_api": ["rest api", "web api", "flask", "fastapi", "django", "endpoints"],
            "cli_tool": ["cli", "command line", "command-line", "argparse", "click"],
            "library": ["library", "package", "module", "pip install"],
        }.items():
            if any(kw in desc_lower for kw in keywords):
                matched_pattern = pattern_name
                break

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a software architect. Given a project description, "
                    "decompose it into individual files that should be created.\n\n"
                    "Return a JSON array where each element has:\n"
                    '- "file": filename (e.g. "routes.py")\n'
                    '- "description": what this file should contain\n'
                    '- "depends_on": list of other filenames this file imports from\n\n'
                    "Order the array so that dependencies come before dependents.\n"
                    "Return ONLY the JSON array, no other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Project: {problem.description}\n\n"
                    f"Existing codebase files: "
                    f"{list(problem.context.keys()) if problem.context else '(none)'}\n\n"
                    "Decompose this project into files."
                ),
            },
        ]

        try:
            raw = await self.backend.generate(
                messages, max_tokens=self.max_tokens, temperature=0.3
            )
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                decomposition = _json.loads(json_match.group(0))
                if isinstance(decomposition, list) and all(
                    isinstance(d, dict) and "file" in d for d in decomposition
                ):
                    for entry in decomposition:
                        entry.setdefault("description", "")
                        entry.setdefault("depends_on", [])
                    return decomposition
        except Exception as e:
            logger.warning("Decomposition LLM call failed: %s", e)

        # Fallback: use matched pattern or generic layout
        if matched_pattern and matched_pattern in self.DECOMPOSITION_PATTERNS:
            files = self.DECOMPOSITION_PATTERNS[matched_pattern]
        else:
            files = ["main.py", "core.py", "utils.py", "tests/test_main.py"]

        return [
            {"file": f, "description": f"Part of: {problem.description}", "depends_on": []}
            for f in files
        ]

    @staticmethod
    def _build_dependency_levels(
        decomposition: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Sort file specs into dependency levels for parallel execution.

        Files with no unresolved dependencies go into the first level.
        Files whose dependencies are all in earlier levels go into the next.
        Circular dependencies are broken by placing all remaining files together.
        """
        remaining = list(decomposition)
        resolved: set[str] = set()
        levels: list[list[dict[str, Any]]] = []

        max_iterations = len(remaining) + 1
        for _ in range(max_iterations):
            if not remaining:
                break

            current_level: list[dict[str, Any]] = []
            still_remaining: list[dict[str, Any]] = []

            for spec in remaining:
                deps = set(spec.get("depends_on", []))
                if deps.issubset(resolved):
                    current_level.append(spec)
                else:
                    still_remaining.append(spec)

            if not current_level:
                # Circular dependency -- break the cycle
                current_level = still_remaining
                still_remaining = []

            levels.append(current_level)
            resolved.update(spec["file"] for spec in current_level)
            remaining = still_remaining

        return levels

    @staticmethod
    def _merge_file_solutions(
        decomposition: list[dict[str, Any]],
        file_results: list[tuple[dict[str, Any], SolveResult]],
    ) -> Solution:
        """Merge per-file generation results into a single Solution."""
        merged_code: dict[str, str] = {}
        explanations: list[str] = []

        for file_spec, result in file_results:
            target_file = file_spec["file"]
            if result.best and result.best.code:
                code_values = list(result.best.code.values())
                if code_values:
                    merged_code[target_file] = code_values[0]
            if result.best and result.best.explanation:
                explanations.append(
                    f"## {target_file}\n{result.best.explanation[:500]}"
                )

        return Solution(
            code=merged_code,
            explanation="\n\n".join(explanations),
            strategy_used="decomposed",
            metadata={
                "files": [f["file"] for f in decomposition],
                "generated": list(merged_code.keys()),
            },
        )

    async def _verify_integration(
        self, merged: Solution, problem: Problem
    ) -> Solution | None:
        """Ask the LLM to verify and fix cross-file integration issues.

        Checks import consistency, interface compatibility, and duplicate
        definitions.  Returns a corrected Solution or ``None`` on failure.
        """
        if not merged.code:
            return None

        files_text = ""
        for filename, content in sorted(merged.code.items()):
            files_text += f"# === {filename} ===\n{content}\n\n"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an integration engineer. You have been given a set of "
                    "files that were generated independently for a project. Your job:\n"
                    "1. Check that all imports between files are correct\n"
                    "2. Verify that interfaces (function signatures, class APIs) match\n"
                    "3. Fix any conflicts or missing pieces\n"
                    "4. Ensure the files work together as a cohesive project\n\n"
                    "Return ALL files with any necessary fixes applied. "
                    "Use ```filename.py\\n...\\n``` code blocks for each file."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Project Goal\n\n{problem.description}\n\n"
                    f"## Generated Files\n\n{files_text}\n\n"
                    "Review these files for integration issues and return corrected versions."
                ),
            },
        ]

        try:
            raw = await self.backend.generate(
                messages, max_tokens=self.max_tokens * 2, temperature=0.3
            )
            corrected_code = self._extract_code_blocks(raw)

            final_code = dict(merged.code)
            for block_name, block_content in corrected_code.items():
                if block_name in final_code:
                    final_code[block_name] = block_content
                else:
                    for orig_name in list(final_code.keys()):
                        if orig_name.endswith(block_name) or block_name.endswith(orig_name):
                            final_code[orig_name] = block_content
                            break
                    else:
                        final_code[block_name] = block_content

            return Solution(
                code=final_code,
                explanation=f"Integration-verified solution:\n\n{raw[:1000]}",
                strategy_used="decomposed_integrated",
                metadata={
                    "files": list(final_code.keys()),
                    "integration_pass": True,
                },
            )
        except Exception as e:
            logger.warning("Integration verification failed: %s", e)
            return None

    async def solve_best_of_n(
        self,
        problem: Problem,
        n: int = 8,
    ) -> SolveResult:
        """Generate N solutions with the same strategy, pick the best.

        Uses sampling diversity (temperature > 0) rather than strategy diversity.
        Good for well-defined problems.
        """
        from fusen_solver.strategies.presets import get_strategy

        direct = get_strategy("direct")
        strategies = [direct] * n
        return await self.solve(problem, n=n, strategies=strategies)

    async def record_feedback(
        self,
        problem: Problem,
        solutions: list[Solution],
        accepted_idx: int,
    ) -> None:
        """Record which solution the user accepted, feeding the learning engine.

        Also extracts a key insight from the winning solution and stores it
        in agent memory for future problems of the same type.
        """
        self.learning_engine.record(problem, solutions, accepted_idx)

        # Extract and remember an insight from the winning solution
        if 0 <= accepted_idx < len(solutions):
            winning = solutions[accepted_idx]
            try:
                insight = await self._extract_insight(problem, winning)
                if insight:
                    ptype = problem.problem_type if problem.problem_type != "auto" else "unknown"
                    self.memory.remember(ptype, insight, source=winning.strategy_used)
            except Exception as e:
                logger.warning("Failed to extract insight: %s", e)

    async def _extract_insight(self, problem: Problem, solution: Solution) -> str:
        """Ask the LLM to distill the key insight from a winning solution."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract concise, reusable insights from solved coding problems. "
                    "Respond with a single sentence describing the key insight."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {problem.description}\n\n"
                    f"Winning solution (strategy: {solution.strategy_used}):\n"
                    f"{solution.explanation[:2000]}\n\n"
                    "What was the key insight that made this solution work? "
                    "Answer in one sentence."
                ),
            },
        ]
        return await self.backend.generate(messages, max_tokens=200, temperature=0.3)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_collaborative_prompt(
        self,
        role: Any,  # AgentRole from presets
        context: dict[str, Any],
        round_num: int,
    ) -> list[dict[str, str]]:
        """Build chat messages for a collaborative agent role.

        The prompt includes the problem description, codebase, the role's
        specific instructions, and accumulated context from prior rounds
        (if the role receives context).
        """
        system = (
            f"You are a specialized '{role.name}' agent in round {round_num} "
            f"of a collaborative problem-solving session.\n\n"
            f"## Codebase\n\n```\n{context.get('codebase', '')}\n```"
        )

        # Accumulate prior round context
        prior_context = ""
        if role.receives_context:
            for r in range(1, round_num):
                rkey = f"round_{r}"
                if rkey in context:
                    round_data = context[rkey]
                    prior_context += f"\n\n## Round {r} Results\n\n"
                    for output in round_data.get("outputs", []):
                        prior_context += (
                            f"### {output['role']}\n{output['output']}\n\n"
                        )

        user = (
            f"## Problem\n\n{context.get('problem', '')}\n\n"
            f"## Your Role: {role.name}\n\n{role.prompt_template}\n\n"
            f"{prior_context}"
            "## Instructions\n\n"
            "Write the complete output for your role. Include ALL code -- "
            "do not use placeholders or ellipsis."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def _synthesize_round(
        self,
        round_outputs: list[dict[str, Any]],
        round_num: int,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Synthesize outputs from a collaborative round into a summary.

        Uses the LLM to combine multiple agent outputs into a coherent
        synthesis. Detects whether a solution and passing tests were produced.

        Returns:
            Dict with keys: outputs, summary, has_solution, tests_pass, solutions.
        """
        # Build a combined text of all round outputs
        outputs_text = ""
        for output in round_outputs:
            outputs_text += (
                f"### {output['role']}\n\n{output['output']}\n\n---\n\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a synthesis agent. Combine the outputs from multiple "
                    "specialized agents into a coherent summary. Identify:\n"
                    "1. Key insights and findings\n"
                    "2. Code solutions (if any)\n"
                    "3. Whether tests were written and if they would pass\n\n"
                    "End your response with a JSON block:\n"
                    "```json\n"
                    '{"has_solution": true/false, "tests_pass": true/false}\n'
                    "```"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Problem\n\n{context.get('problem', '')}\n\n"
                    f"## Round {round_num} Outputs\n\n{outputs_text}\n\n"
                    "Synthesize these outputs."
                ),
            },
        ]

        try:
            synthesis_text = await self.backend.generate(
                messages, max_tokens=self.max_tokens
            )
        except Exception as e:
            logger.warning("Synthesis failed for round %d: %s", round_num, e)
            synthesis_text = ""

        # Parse status from the synthesis
        has_solution = False
        tests_pass = False
        try:
            import re

            json_match = re.search(r"```json\s*({.*?})\s*```", synthesis_text, re.DOTALL)
            if json_match:
                import json

                status = json.loads(json_match.group(1))
                has_solution = bool(status.get("has_solution", False))
                tests_pass = bool(status.get("tests_pass", False))
        except (ValueError, KeyError):
            pass

        # Extract code solutions from round outputs
        solutions: list[dict[str, str]] = []
        for output in round_outputs:
            if output.get("success"):
                code = self._extract_code_blocks(output["output"])
                if code:
                    solutions.append(code)

        return {
            "outputs": round_outputs,
            "summary": synthesis_text,
            "has_solution": has_solution,
            "tests_pass": tests_pass,
            "solutions": solutions,
        }

    async def _run_agent(
        self,
        problem: Problem,
        strategy: Strategy,
        codebase_text: str,
        agent_idx: int,
    ) -> Solution:
        """Run a single agent with a specific strategy."""
        messages = self._build_messages(problem, strategy, codebase_text)

        # Inject recalled memories into the system prompt
        memories = self.memory.recall(
            problem.problem_type if problem.problem_type != "auto" else "unknown"
        )
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            # Append memory context to the system message
            messages[0]["content"] += (
                f"\n\nInsights from previous similar problems:\n{memory_text}"
            )

        priority = compute_priority(problem, strategy.name)

        content = await self.backend.generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=strategy.temperature,
            priority=priority,
        )

        return Solution(
            code=self._extract_code_blocks(content),
            explanation=content,
            strategy_used=strategy.name,
            metadata={"agent_idx": agent_idx, "raw_response": content, "priority": priority},
        )

    async def _merge_solutions(
        self,
        problem: Problem,
        solutions: list[Solution],
        codebase_text: str,
    ) -> Solution:
        """Merge top solutions using the LLM."""
        solutions_text = ""
        for i, sol in enumerate(solutions):
            solutions_text += (
                f"### Solution {i + 1} (strategy: {sol.strategy_used}, "
                f"score: {sol.score:.2f})\n\n{sol.explanation}\n\n---\n\n"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert code synthesizer. Analyze multiple solutions "
                    "to the same problem and produce one final solution that combines "
                    "the best ideas from each.\n\n"
                    f"## Codebase\n\n```\n{codebase_text}\n```"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Problem\n\n{problem.description}\n\n"
                    f"## Solutions\n\n{solutions_text}\n\n"
                    "Produce a single merged solution. Include ALL code -- no placeholders."
                ),
            },
        ]

        content = await self.backend.generate(messages, max_tokens=self.max_tokens * 2)

        return Solution(
            code=self._extract_code_blocks(content),
            explanation=content,
            strategy_used="merged",
            metadata={"source_strategies": [s.strategy_used for s in solutions]},
        )

    @staticmethod
    def _build_messages(
        problem: Problem,
        strategy: Strategy,
        codebase_text: str,
    ) -> list[dict[str, str]]:
        """Build chat messages with shared codebase prefix + unique strategy suffix."""
        system = (
            "You are an expert software engineer working on a coding problem. "
            "Below is the codebase you are working with. Study it carefully.\n\n"
            f"## Codebase\n\n```\n{codebase_text}\n```"
        )

        constraints_text = ""
        if problem.constraints:
            constraints_text = "\n\n## Constraints\n\n" + "\n".join(
                f"- {c}" for c in problem.constraints
            )

        user = (
            f"## Problem\n\n{problem.description}\n\n"
            f"## Your Approach\n\n{strategy.prompt}"
            f"{constraints_text}\n\n"
            "## Instructions\n\n"
            "Write the complete solution. Include ALL modified code -- do not use "
            "placeholders or ellipsis. Explain your reasoning briefly before the code."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _format_codebase(context: dict[str, str]) -> str:
        """Format the context dict into a single string for the prompt."""
        if not context:
            return "(no codebase provided)"
        parts = []
        for filename, content in sorted(context.items()):
            parts.append(f"# === {filename} ===\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_code_blocks(text: str) -> dict[str, str]:
        """Extract code blocks from LLM response into filename -> content map."""
        import re

        blocks: dict[str, str] = {}
        # Match ```language\n...``` or ```filename\n...```
        pattern = r"```(?:(\S+)\n)?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        for i, (label, code) in enumerate(matches):
            # Try to use label as filename if it looks like one
            if label and ("." in label or "/" in label):
                key = label.strip()
            else:
                key = f"block_{i}.txt"
            blocks[key] = code.strip()

        # If no code blocks found, treat entire response as the solution
        if not blocks:
            blocks["solution.txt"] = text

        return blocks
