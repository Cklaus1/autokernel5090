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
from fusen_solver.learning.tracker import LearningEngine
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
        max_tokens: int = 4096,
        default_n: int = 4,
        auto_n: bool = True,
    ):
        self.backend = backend
        self.strategy_engine = strategy_engine or StrategyEngine()
        self.scoring_engine = scoring_engine or ScoringEngine()
        self.learning_engine = learning_engine or LearningEngine()
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
        )
        logger.info(
            "Solving with %d agents: %s",
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
        """Record which solution the user accepted, feeding the learning engine."""
        self.learning_engine.record(problem, solutions, accepted_idx)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        problem: Problem,
        strategy: Strategy,
        codebase_text: str,
        agent_idx: int,
    ) -> Solution:
        """Run a single agent with a specific strategy."""
        messages = self._build_messages(problem, strategy, codebase_text)

        content = await self.backend.generate(
            messages,
            max_tokens=self.max_tokens,
            temperature=strategy.temperature,
        )

        return Solution(
            code=self._extract_code_blocks(content),
            explanation=content,
            strategy_used=strategy.name,
            metadata={"agent_idx": agent_idx, "raw_response": content},
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
