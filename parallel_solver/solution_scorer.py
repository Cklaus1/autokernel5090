"""Scores, ranks, and merges solutions from parallel agents.

Two scoring modes:
1. Test-based: run provided test cases against the solution (ground truth).
2. LLM-based: use the model itself to review and score solutions (no tests needed).

The merge step takes the best ideas from multiple solutions and combines them
into a single, stronger solution.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ScoredSolution:
    """A solution with its scores."""

    agent_id: int
    strategy: str
    content: str
    # Scores (0-1 scale)
    completeness: float = 0.0
    correctness: float = 0.0
    code_quality: float = 0.0
    test_pass_rate: float = 0.0
    overall: float = 0.0
    # Metadata
    explanation: str = ""
    tests_passed: int = 0
    tests_total: int = 0


class SolutionScorer:
    """Scores and ranks solutions from parallel agents."""

    def __init__(self, vllm_api: str, model: str):
        self.api = vllm_api.rstrip("/")
        self.model = model

    async def score_all(
        self,
        problem: str,
        solutions: list[dict],
        tests: list[str] | None = None,
    ) -> list[ScoredSolution]:
        """Score all solutions and return them ranked best-to-worst.

        Args:
            problem: The original problem statement.
            solutions: List of dicts with keys: agent_id, strategy, content.
            tests: Optional list of test code strings to run.

        Returns:
            Scored solutions sorted by overall score (descending).
        """
        scored: list[ScoredSolution] = []

        # Run test-based and LLM-based scoring in parallel
        tasks = []
        for sol in solutions:
            tasks.append(self._score_one(problem, sol, tests))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sol, result in zip(solutions, results):
            if isinstance(result, Exception):
                logger.warning("Scoring failed for agent %d: %s", sol["agent_id"], result)
                scored.append(
                    ScoredSolution(
                        agent_id=sol["agent_id"],
                        strategy=sol["strategy"],
                        content=sol["content"],
                        explanation=f"Scoring error: {result}",
                    )
                )
            else:
                scored.append(result)

        scored.sort(key=lambda s: s.overall, reverse=True)
        return scored

    async def merge_insights(
        self,
        problem: str,
        scored_solutions: list[ScoredSolution],
        codebase: str = "",
    ) -> str:
        """Take the best parts from multiple solutions and combine them.

        Uses the LLM to analyze all solutions and produce a merged result
        that combines the strongest aspects of each.
        """
        solutions_text = ""
        for s in scored_solutions:
            solutions_text += (
                f"### Solution from Agent {s.agent_id} "
                f"(strategy: {s.strategy}, score: {s.overall:.2f})\n"
                f"Strengths: {s.explanation}\n\n"
                f"```\n{s.content}\n```\n\n---\n\n"
            )

        system = (
            "You are an expert code reviewer and synthesizer. "
            "Your job is to analyze multiple solutions to the same problem "
            "and produce one final solution that combines the best ideas."
        )
        if codebase:
            system += f"\n\nCodebase context:\n```\n{codebase}\n```"

        user = (
            f"## Problem\n\n{problem}\n\n"
            f"## Solutions\n\n{solutions_text}\n\n"
            "## Your Task\n\n"
            "1. Identify the strongest aspects of each solution.\n"
            "2. Identify any bugs or weaknesses in each.\n"
            "3. Produce a SINGLE merged solution that takes the best from each.\n"
            "4. The merged solution must be complete -- no placeholders.\n"
            "5. Briefly explain which parts came from which solution."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 8192,
                "temperature": 0.3,
            }
            async with session.post(
                f"{self.api}/v1/chat/completions", json=payload
            ) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _score_one(
        self,
        problem: str,
        solution: dict,
        tests: list[str] | None,
    ) -> ScoredSolution:
        """Score a single solution."""
        scored = ScoredSolution(
            agent_id=solution["agent_id"],
            strategy=solution["strategy"],
            content=solution["content"],
        )

        # Test-based scoring (if tests provided)
        if tests:
            passed, total = self._run_tests(solution["content"], tests)
            scored.tests_passed = passed
            scored.tests_total = total
            scored.test_pass_rate = passed / total if total > 0 else 0.0
            scored.correctness = scored.test_pass_rate

        # LLM-based scoring (always)
        llm_scores = await self._llm_score(problem, solution["content"])
        scored.completeness = llm_scores.get("completeness", 0.0)
        scored.code_quality = llm_scores.get("code_quality", 0.0)
        scored.explanation = llm_scores.get("explanation", "")

        # If no tests, use LLM correctness estimate
        if not tests:
            scored.correctness = llm_scores.get("correctness", 0.0)

        # Overall score: weighted combination
        scored.overall = (
            0.40 * scored.correctness
            + 0.30 * scored.completeness
            + 0.30 * scored.code_quality
        )

        return scored

    async def _llm_score(self, problem: str, solution: str) -> dict:
        """Use the LLM to score a solution on multiple dimensions."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code reviewer. Score the solution on three dimensions, "
                    "each from 0.0 to 1.0. Respond ONLY with valid JSON:\n"
                    '{"completeness": 0.X, "correctness": 0.X, "code_quality": 0.X, '
                    '"explanation": "brief strengths/weaknesses"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Problem\n{problem}\n\n"
                    f"## Solution\n```\n{solution}\n```\n\n"
                    "Score this solution."
                ),
            },
        ]

        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.1,
                }
                async with session.post(
                    f"{self.api}/v1/chat/completions", json=payload
                ) as resp:
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]

                    # Extract JSON from response (handle markdown code blocks)
                    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
                    if json_match:
                        import json as json_mod

                        scores = json_mod.loads(json_match.group())
                        return {
                            "completeness": float(scores.get("completeness", 0.5)),
                            "correctness": float(scores.get("correctness", 0.5)),
                            "code_quality": float(scores.get("code_quality", 0.5)),
                            "explanation": scores.get("explanation", ""),
                        }
        except Exception as e:
            logger.warning("LLM scoring failed: %s", e)

        return {"completeness": 0.5, "correctness": 0.5, "code_quality": 0.5, "explanation": ""}

    @staticmethod
    def _run_tests(solution_code: str, tests: list[str]) -> tuple[int, int]:
        """Run test cases against the solution code.

        Writes the solution to a temp file, appends each test, and runs it.
        Returns (passed, total).
        """
        passed = 0
        total = len(tests)

        for test in tests:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(solution_code)
                f.write("\n\n")
                f.write(test)
                f.flush()
                tmp_path = f.name

            try:
                result = subprocess.run(
                    ["python3", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    passed += 1
                else:
                    logger.debug(
                        "Test failed:\n%s\nstderr: %s",
                        test[:200],
                        result.stderr[:500],
                    )
            except subprocess.TimeoutExpired:
                logger.debug("Test timed out: %s", test[:200])
            except Exception as e:
                logger.debug("Test error: %s", e)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        return passed, total
