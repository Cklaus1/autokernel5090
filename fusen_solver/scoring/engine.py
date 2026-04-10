"""Multi-signal solution scoring engine.

Scores solutions using multiple signals:
1. Syntax check (instant, binary)
2. Test execution (if tests provided) — runs in Docker sandbox when available
3. LLM review (quality assessment)
4. Diff quality (minimal changes, clean diff)
5. Confidence (model's self-reported confidence)
"""

from __future__ import annotations

import ast
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fusen_solver.core.interfaces import Problem, Solution
from fusen_solver.scoring.sandbox import TestSandbox, _docker_available

if TYPE_CHECKING:
    from fusen_solver.core.interfaces import LLMBackend

logger = logging.getLogger(__name__)

# Module-level sandbox instance (shared across ScoringEngine instances so that
# Docker availability is only probed once per process).
_sandbox: TestSandbox | None = None


def _get_sandbox() -> TestSandbox:
    """Return the module-level sandbox, creating it lazily on first call."""
    global _sandbox
    if _sandbox is None:
        _sandbox = TestSandbox()
    return _sandbox


class ScoringEngine:
    """Scores solutions using multiple signals with configurable weights."""

    def __init__(
        self,
        *,
        test_weight: float = 0.4,
        review_weight: float = 0.3,
        diff_weight: float = 0.15,
        syntax_weight: float = 0.1,
        confidence_weight: float = 0.05,
    ):
        self.weights = {
            "tests": test_weight,
            "review": review_weight,
            "diff": diff_weight,
            "syntax": syntax_weight,
            "confidence": confidence_weight,
        }

    async def score_all(
        self,
        problem: Problem,
        solutions: list[Solution],
        *,
        backend: LLMBackend | None = None,
    ) -> list[Solution]:
        """Score all solutions and return them with updated scores.

        Args:
            problem: The original problem.
            solutions: Solutions to score.
            backend: LLM backend for review scoring (optional).

        Returns:
            The same solutions with score and subscores populated.
        """
        for sol in solutions:
            subscores: dict[str, float] = {}

            # 1. Syntax check (instant)
            subscores["syntax"] = self._check_syntax(sol)

            # 2. Test execution
            if problem.tests:
                subscores["tests"] = self._run_tests(problem, sol)
            else:
                # No tests: redistribute weight to review
                subscores["tests"] = -1.0  # sentinel: will be excluded

            # 3. LLM review
            if backend is not None:
                subscores["review"] = await self._llm_review(problem, sol, backend)
            else:
                subscores["review"] = 0.5  # neutral default

            # 4. Diff quality
            subscores["diff"] = self._diff_quality(problem, sol)

            # 5. Confidence
            subscores["confidence"] = sol.metadata.get("confidence", 0.5)

            # Compute weighted score
            sol.subscores = subscores
            sol.score = self._weighted_score(subscores, has_tests=bool(problem.tests))

        return solutions

    def _weighted_score(self, subscores: dict[str, float], has_tests: bool) -> float:
        """Compute weighted average, redistributing weights if tests are missing.

        When a signal is absent (subscore == -1 sentinel or weight == 0), its
        weight is excluded from both the numerator *and* the denominator so that
        the remaining weights are effectively renormalized to sum to 1.0.
        """
        weights = dict(self.weights)

        if not has_tests:
            # Redistribute test weight to review
            weights["review"] += weights["tests"]
            weights["tests"] = 0.0

        # Only count signals that are actually present (weight > 0 and subscore >= 0).
        active_weight = sum(
            w for signal, w in weights.items()
            if w > 0 and signal in subscores and subscores[signal] >= 0
        )
        if active_weight <= 0:
            return 0.5

        score = 0.0
        for signal, weight in weights.items():
            if weight > 0 and signal in subscores and subscores[signal] >= 0:
                score += weight * subscores[signal]

        return min(1.0, max(0.0, score / active_weight))

    @staticmethod
    def _check_syntax(solution: Solution) -> float:
        """Check if all code blocks in the solution parse correctly."""
        if not solution.code:
            return 0.0

        parseable = 0
        total = 0

        for filename, content in solution.code.items():
            # Only check files that look like Python
            if not (filename.endswith(".py") or filename.endswith(".txt")):
                continue
            total += 1
            try:
                ast.parse(content)
                parseable += 1
            except SyntaxError:
                pass

        if total == 0:
            return 0.5  # no Python files to check

        return parseable / total

    @staticmethod
    def _run_tests(problem: Problem, solution: Solution) -> float:
        """Run test commands against the solution. Returns pass rate 0-1.

        Delegates to ``TestSandbox``, which runs commands inside a Docker
        container (no network, 256 MB memory cap) when Docker is available,
        and falls back to an unsandboxed subprocess call otherwise.
        """
        if not problem.tests:
            return 0.5

        sandbox = _get_sandbox()
        result = sandbox.run_tests(solution.code, problem.tests)

        if not result["sandboxed"]:
            logger.debug("Tests ran unsandboxed (Docker unavailable)")

        total = result["total"]
        passed = result["passed"]

        # Log individual failures at DEBUG level for easier diagnosis.
        for r in result["results"]:
            if not r.get("passed", False):
                if "error" in r:
                    logger.debug("Test error (%s): %s", r["command"], r["error"])
                else:
                    logger.debug(
                        "Test failed: %s\nstderr: %s",
                        r["command"],
                        r.get("stderr", "")[:300],
                    )

        return passed / total if total > 0 else 0.5

    @staticmethod
    async def _llm_review(problem: Problem, solution: Solution, backend: LLMBackend) -> float:
        """Use the LLM to score a solution's quality."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code reviewer. Score the solution on correctness, "
                    "completeness, and code quality. Respond ONLY with valid JSON:\n"
                    '{"correctness": 0.X, "completeness": 0.X, "quality": 0.X}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## Problem\n{problem.description}\n\n"
                    f"## Solution\n{solution.explanation[:4000]}\n\n"
                    "Score this solution (each dimension 0.0 to 1.0)."
                ),
            },
        ]

        try:
            response = await backend.generate(messages, max_tokens=128, temperature=0.1)
            match = re.search(r"\{[^}]+\}", response, re.DOTALL)
            if match:
                scores = json.loads(match.group())
                return (
                    float(scores.get("correctness", 0.5))
                    + float(scores.get("completeness", 0.5))
                    + float(scores.get("quality", 0.5))
                ) / 3.0
        except Exception as e:
            logger.warning("LLM review failed: %s", e)

        return 0.5

    @staticmethod
    def _diff_quality(problem: Problem, solution: Solution) -> float:
        """Score the diff quality: prefer minimal, focused changes.

        Heuristics:
        - Penalize very large diffs (probably rewrote too much)
        - Penalize empty diffs (probably didn't solve anything)
        - Reward moderate-sized, focused changes
        """
        if not solution.code:
            return 0.0

        total_lines = sum(len(content.splitlines()) for content in solution.code.values())

        if total_lines == 0:
            return 0.0

        # Compare to original if available
        if not problem.context:
            # No original to compare against; score based on solution size
            # Prefer moderate-length solutions (10-200 lines)
            if total_lines < 5:
                return 0.3
            elif total_lines < 200:
                return 0.8
            elif total_lines < 500:
                return 0.6
            else:
                return 0.4

        # Count changed lines
        original_lines = sum(len(c.splitlines()) for c in problem.context.values())
        ratio = total_lines / max(original_lines, 1)

        # Best: similar size to original (small, focused changes)
        if 0.8 <= ratio <= 1.5:
            return 0.9
        elif 0.5 <= ratio <= 2.0:
            return 0.7
        elif 0.3 <= ratio <= 3.0:
            return 0.5
        else:
            return 0.3
