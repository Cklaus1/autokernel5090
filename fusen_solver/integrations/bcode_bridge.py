"""Bridge between BCode's task format and fusen_solver's Problem/Solution.

Provides clean translation without tight coupling -- BCode and fusen_solver
remain independent, connected only through this adapter layer.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from fusen_solver.core.interfaces import Problem, Solution

logger = logging.getLogger(__name__)

# Problem type inference patterns (PRD text -> fusen_solver problem_type)
_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"\bfix\b|\bbug\b|\bcrash\b|\berror\b|\bbroken\b", "bug_fix"),
    (r"\brefactor\b|\bclean\s*up\b|\brestructure\b", "refactor"),
    (r"\btest\b|\bspec\b|\bcoverage\b", "test"),
    (r"\boptimiz\b|\bperformance\b|\bspeed\b|\bfast\b", "optimize"),
    (r"\breview\b|\baudit\b", "review"),
    (r"\barchitect\b|\bdesign\b|\bsystem\b", "architecture"),
]


class BCodeBridge:
    """Translate between BCode and fusen_solver formats."""

    def prd_to_problem(
        self,
        prd: str,
        workspace: str,
        language: str = "auto",
        *,
        max_context_chars: int = 200_000,
    ) -> Problem:
        """Convert a BCode PRD to a fusen_solver Problem.

        Args:
            prd: The PRD / task description from BCode.
            workspace: Path to the project workspace directory.
            language: Programming language hint (default: auto-detect).
            max_context_chars: Cap on total context size to avoid blowing
                the LLM's context window.

        Returns:
            A Problem instance ready for FusenSolver.solve().
        """
        context = self._load_workspace(workspace, max_chars=max_context_chars)
        problem_type = self._infer_type(prd)

        return Problem(
            description=prd,
            context=context,
            problem_type=problem_type,
            language=language,
            solve_mode="auto",
        )

    def solution_to_bcode_output(self, solution: Solution) -> dict[str, Any]:
        """Convert fusen_solver Solution to BCode's expected output format.

        BCode expects a dict with files list, score (0-100), and metadata.
        """
        return {
            "files": [
                {"path": path, "content": content}
                for path, content in solution.code.items()
            ],
            "score": round(solution.score * 100, 1),  # BCode uses 0-100
            "explanation": solution.explanation,
            "strategy": solution.strategy_used,
            "subscores": solution.subscores,
        }

    def bcode_output_to_solution(self, output: dict[str, Any]) -> Solution:
        """Convert BCode output to fusen_solver Solution for comparison.

        Args:
            output: BCode result dict with 'files', 'score', etc.

        Returns:
            A Solution instance suitable for scoring/comparison.
        """
        files = output.get("files", [])
        code: dict[str, str] = {}
        for f in files:
            path = f.get("path", "")
            content = f.get("content", "")
            if path and content:
                code[path] = content

        raw_score = output.get("score", 0)
        # BCode scores are 0-100; normalize to 0-1
        score = raw_score / 100.0 if raw_score > 1.0 else raw_score

        return Solution(
            code=code,
            explanation=output.get("explanation", ""),
            strategy_used="bcode_pipeline",
            score=score,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_type(prd: str) -> str:
        """Infer fusen_solver problem_type from PRD text."""
        prd_lower = prd.lower()
        for pattern, ptype in _TYPE_PATTERNS:
            if re.search(pattern, prd_lower):
                return ptype
        return "feature"  # default for new work

    @staticmethod
    def _load_workspace(
        workspace: str,
        max_chars: int = 200_000,
    ) -> dict[str, str]:
        """Load workspace files into a context dict.

        Walks the workspace directory, collecting source files ordered by
        recency, stopping when max_chars is reached.
        """
        root = Path(workspace)
        if not root.exists():
            logger.warning("Workspace path does not exist: %s", workspace)
            return {}

        if root.is_file():
            try:
                return {root.name: root.read_text(errors="replace")[:max_chars]}
            except OSError:
                return {}

        skip_dirs = {
            "__pycache__", ".git", "node_modules", ".venv", "venv",
            ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
            "coverage", ".tox",
        }
        source_exts = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
            ".c", ".cpp", ".h", ".hpp", ".rb", ".swift", ".kt", ".scala",
            ".sh", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini",
            ".html", ".css", ".sql", ".md",
        }

        context: dict[str, str] = {}
        total_chars = 0

        try:
            candidates = sorted(
                (f for f in root.rglob("*") if f.is_file() and f.suffix in source_exts),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError as exc:
            logger.warning("Error walking workspace %s: %s", workspace, exc)
            return {}

        for f in candidates:
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
