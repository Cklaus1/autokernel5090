"""Problem Orchestrator: decomposes a problem and coordinates parallel agents.

This is the brain of the Parallel Problem-Solving system. Given a coding problem
and codebase, it:
1. Analyzes the problem to choose the best strategies.
2. Launches N agents in parallel (each with a different strategy).
3. Streams results as they arrive.
4. Scores and ranks solutions.
5. Optionally merges the best solutions into one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import aiohttp

from parallel_solver.prefix_manager import PrefixManager, STRATEGY_PROMPTS
from parallel_solver.solution_scorer import SolutionScorer, ScoredSolution
from parallel_solver.streaming import ParallelStreamer, AgentResult, StreamEvent

logger = logging.getLogger(__name__)

# Default strategy sets for different problem types
STRATEGY_PRESETS: dict[str, list[str]] = {
    "bug_fix": ["direct", "review", "test_first", "adversarial"],
    "feature": ["direct", "decompose", "alternative", "test_first"],
    "refactor": ["rewrite", "review", "test_first", "decompose"],
    "architecture": ["decompose", "alternative", "research", "review"],
    "optimization": ["direct", "alternative", "research", "review"],
    "explore": list(STRATEGY_PROMPTS.keys()),  # all strategies
}


@dataclass
class SolveResult:
    """Result of a parallel solve operation."""

    problem: str
    strategies_used: list[str]
    agent_results: list[AgentResult] = field(default_factory=list)
    scored_solutions: list[ScoredSolution] = field(default_factory=list)
    merged_solution: str | None = None
    best_solution: ScoredSolution | None = None
    # Timing
    total_time_ms: float = 0.0
    prefix_warm_ms: float = 0.0
    generation_ms: float = 0.0
    scoring_ms: float = 0.0
    # Aggregate stats
    total_tokens: int = 0
    aggregate_tps: float = 0.0


class ProblemOrchestrator:
    """Decomposes a problem and coordinates parallel solver agents.

    Usage:
        orch = ProblemOrchestrator("http://localhost:8000", "model-name")
        result = await orch.solve(
            problem="Fix the race condition in server.py",
            codebase=open("server.py").read(),
            strategies=["direct", "review", "test_first", "adversarial"],
        )
        print(result.best_solution.content)
    """

    def __init__(
        self,
        vllm_api: str,
        model: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout_seconds: float = 300.0,
    ):
        self.api = vllm_api
        self.model = model
        self.prefix_mgr = PrefixManager(vllm_api, model)
        self.scorer = SolutionScorer(vllm_api, model)
        self.streamer = ParallelStreamer(
            vllm_api,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    async def solve(
        self,
        problem: str,
        codebase: str,
        *,
        strategies: list[str] | None = None,
        preset: str | None = None,
        max_agents: int = 8,
        tests: list[str] | None = None,
        merge: bool = False,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> SolveResult:
        """Solve a problem using parallel agents with diverse strategies.

        Args:
            problem: Description of the problem to solve.
            codebase: Source code context (shared prefix for all agents).
            strategies: List of strategy names (from STRATEGY_PROMPTS).
            preset: Use a predefined strategy set (e.g., "bug_fix", "refactor").
            max_agents: Maximum number of parallel agents.
            tests: Optional test cases for scoring.
            merge: Whether to merge top solutions into a final one.
            on_event: Optional callback for streaming events.

        Returns:
            SolveResult with ranked solutions and optional merge.
        """
        t_start = time.perf_counter()
        result = SolveResult(problem=problem, strategies_used=[])

        # 1. Select strategies
        if strategies is None:
            if preset and preset in STRATEGY_PRESETS:
                strategies = STRATEGY_PRESETS[preset]
            else:
                strategies = await self._select_strategies(problem, max_agents)

        strategies = strategies[:max_agents]
        result.strategies_used = strategies
        logger.info("Using %d strategies: %s", len(strategies), strategies)

        # 2. Warm the prefix cache
        t_warm = time.perf_counter()
        await self.prefix_mgr.warm_prefix(codebase)
        result.prefix_warm_ms = (time.perf_counter() - t_warm) * 1000

        # 3. Build requests for each agent
        requests = []
        for i, strategy in enumerate(strategies):
            messages = self.prefix_mgr.build_context(codebase, problem, strategy)
            requests.append({
                "agent_id": i,
                "strategy": strategy,
                "messages": messages,
            })

        # 4. Run all agents in parallel
        t_gen = time.perf_counter()
        result.agent_results = await self.streamer.run_all(requests, on_event=on_event)
        result.generation_ms = (time.perf_counter() - t_gen) * 1000

        # Aggregate stats
        result.total_tokens = sum(r.tokens_generated for r in result.agent_results)
        gen_seconds = result.generation_ms / 1000
        result.aggregate_tps = result.total_tokens / gen_seconds if gen_seconds > 0 else 0

        # 5. Score solutions
        t_score = time.perf_counter()
        solutions_for_scoring = [
            {
                "agent_id": r.agent_id,
                "strategy": r.strategy,
                "content": r.content,
            }
            for r in result.agent_results
            if r.content and not r.error
        ]

        if solutions_for_scoring:
            result.scored_solutions = await self.scorer.score_all(
                problem, solutions_for_scoring, tests
            )
            result.best_solution = result.scored_solutions[0] if result.scored_solutions else None
        result.scoring_ms = (time.perf_counter() - t_score) * 1000

        # 6. Optionally merge top solutions
        if merge and len(result.scored_solutions) >= 2:
            result.merged_solution = await self.scorer.merge_insights(
                problem, result.scored_solutions[:4], codebase
            )

        result.total_time_ms = (time.perf_counter() - t_start) * 1000

        logger.info(
            "Solve complete: %d agents, %d tokens, %.0f tok/s aggregate, %.1fs total",
            len(strategies),
            result.total_tokens,
            result.aggregate_tps,
            result.total_time_ms / 1000,
        )

        return result

    async def solve_streaming(
        self,
        problem: str,
        codebase: str,
        *,
        strategies: list[str] | None = None,
        preset: str | None = None,
        max_agents: int = 8,
    ) -> AsyncIterator[StreamEvent]:
        """Solve with real-time streaming of all agents' progress.

        Yields StreamEvent objects as tokens arrive from each agent.
        """
        if strategies is None:
            if preset and preset in STRATEGY_PRESETS:
                strategies = STRATEGY_PRESETS[preset]
            else:
                strategies = await self._select_strategies(problem, max_agents)

        strategies = strategies[:max_agents]

        await self.prefix_mgr.warm_prefix(codebase)

        requests = []
        for i, strategy in enumerate(strategies):
            messages = self.prefix_mgr.build_context(codebase, problem, strategy)
            requests.append({
                "agent_id": i,
                "strategy": strategy,
                "messages": messages,
            })

        async for event in self.streamer.stream_all(requests):
            yield event

    async def best_of_n(
        self,
        problem: str,
        codebase: str,
        n: int = 8,
        *,
        tests: list[str] | None = None,
    ) -> SolveResult:
        """Generate N independent solutions with the same strategy and pick the best.

        Unlike solve() which uses diverse strategies, this uses the same "direct"
        approach N times with temperature > 0, relying on sampling diversity.
        Useful for well-defined problems where you want reliability.
        """
        strategies = ["direct"] * n
        return await self.solve(
            problem,
            codebase,
            strategies=strategies,
            max_agents=n,
            tests=tests,
        )

    async def decompose_and_solve(
        self,
        problem: str,
        codebase: str,
        *,
        max_sub_agents: int = 4,
    ) -> SolveResult:
        """Decompose the problem into sub-problems and solve each in parallel.

        Uses one agent to decompose, then launches one agent per sub-problem.
        Finally, merges the sub-solutions.
        """
        # Step 1: Decompose
        sub_problems = await self._decompose_problem(problem, codebase)
        if not sub_problems:
            # Fall back to normal solve
            return await self.solve(problem, codebase, preset="feature")

        logger.info("Decomposed into %d sub-problems", len(sub_problems))

        # Step 2: Solve each sub-problem in parallel
        strategies = ["direct"] * len(sub_problems)
        requests = []
        for i, sub in enumerate(sub_problems):
            messages = self.prefix_mgr.build_context(
                codebase,
                sub,
                "direct",
                extra_context=f"This is sub-problem {i + 1} of {len(sub_problems)} "
                f"for the larger problem: {problem}",
            )
            requests.append({
                "agent_id": i,
                "strategy": f"sub_{i}",
                "messages": messages,
            })

        await self.prefix_mgr.warm_prefix(codebase)
        agent_results = await self.streamer.run_all(requests)

        # Step 3: Merge sub-solutions
        solutions_for_merge = [
            ScoredSolution(
                agent_id=r.agent_id,
                strategy=r.strategy,
                content=r.content,
                overall=0.5,
                explanation=f"Sub-problem: {sub_problems[r.agent_id]}",
            )
            for r in agent_results
            if r.content and not r.error
        ]

        merged = await self.scorer.merge_insights(problem, solutions_for_merge, codebase)

        result = SolveResult(
            problem=problem,
            strategies_used=[f"decompose:{len(sub_problems)}"],
            agent_results=agent_results,
            merged_solution=merged,
            total_tokens=sum(r.tokens_generated for r in agent_results),
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _select_strategies(self, problem: str, max_agents: int) -> list[str]:
        """Use the LLM to pick the best strategies for this problem."""
        available = list(STRATEGY_PROMPTS.keys())

        messages = [
            {
                "role": "system",
                "content": (
                    "You select problem-solving strategies. Available strategies:\n"
                    + "\n".join(f"- {name}: {desc[:80]}" for name, desc in STRATEGY_PROMPTS.items())
                    + f"\n\nPick up to {max_agents} strategies that are most useful "
                    "for this problem. Respond with ONLY a JSON list of strategy names."
                ),
            },
            {"role": "user", "content": problem},
        ]

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 128,
                    "temperature": 0.1,
                }
                async with session.post(
                    f"{self.api}/v1/chat/completions", json=payload
                ) as resp:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]

                    # Parse JSON list from response
                    import re

                    match = re.search(r"\[.*?\]", text, re.DOTALL)
                    if match:
                        strategies = json.loads(match.group())
                        # Validate
                        valid = [s for s in strategies if s in available]
                        if valid:
                            return valid[:max_agents]
        except Exception as e:
            logger.warning("Strategy selection failed, using defaults: %s", e)

        # Fallback: diverse default set
        return ["direct", "review", "test_first", "alternative"][:max_agents]

    async def _decompose_problem(self, problem: str, codebase: str) -> list[str]:
        """Use the LLM to decompose a problem into sub-problems."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You decompose coding problems into independent sub-problems. "
                    "Each sub-problem should be solvable independently. "
                    "Respond with ONLY a JSON list of sub-problem descriptions."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Problem: {problem}\n\n"
                    f"Codebase summary (first 2000 chars):\n{codebase[:2000]}\n\n"
                    "Decompose into 2-4 independent sub-problems."
                ),
            },
        ]

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 512,
                    "temperature": 0.3,
                }
                async with session.post(
                    f"{self.api}/v1/chat/completions", json=payload
                ) as resp:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]

                    import re

                    match = re.search(r"\[.*?\]", text, re.DOTALL)
                    if match:
                        return json.loads(match.group())
        except Exception as e:
            logger.warning("Decomposition failed: %s", e)

        return []
