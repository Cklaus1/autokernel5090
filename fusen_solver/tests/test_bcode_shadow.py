"""Tests for BCode shadow mode integration.

Covers:
- BCodeBridge: PRD -> Problem conversion
- BCodeBridge: Solution -> BCode output format
- BCodeBridge: BCode output -> Solution conversion
- BCodeShadow: comparison logic (fusen wins, bcode wins, tie)
- BCodeShadow: stats aggregation
- BCodeShadow: promotion check (below/above threshold)
- BCodeShadow: shadow run with mock backend
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusen_solver.core.interfaces import Problem, Solution
from fusen_solver.core.solver import SolveResult
from fusen_solver.integrations.bcode_bridge import BCodeBridge
from fusen_solver.integrations.bcode_shadow import BCodeShadow, ShadowResult


# ======================================================================
# BCodeBridge tests
# ======================================================================


class TestBCodeBridge:
    def setup_method(self):
        self.bridge = BCodeBridge()

    def test_prd_to_problem_basic(self, tmp_path):
        """PRD text is passed through as Problem.description."""
        (tmp_path / "main.py").write_text("print('hello')")
        problem = self.bridge.prd_to_problem("Build a REST API", str(tmp_path))

        assert isinstance(problem, Problem)
        assert problem.description == "Build a REST API"
        assert problem.solve_mode == "auto"
        assert "main.py" in problem.context

    def test_prd_to_problem_infers_feature(self, tmp_path):
        """Default problem type is 'feature' for new work."""
        (tmp_path / "app.py").write_text("")
        problem = self.bridge.prd_to_problem("Build a new dashboard", str(tmp_path))
        assert problem.problem_type == "feature"

    def test_prd_to_problem_infers_bug_fix(self, tmp_path):
        """PRDs mentioning 'fix' or 'bug' map to bug_fix type."""
        (tmp_path / "app.py").write_text("")
        problem = self.bridge.prd_to_problem("Fix the crash on login", str(tmp_path))
        assert problem.problem_type == "bug_fix"

    def test_prd_to_problem_infers_refactor(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        problem = self.bridge.prd_to_problem("Refactor the auth module", str(tmp_path))
        assert problem.problem_type == "refactor"

    def test_prd_to_problem_infers_test(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        problem = self.bridge.prd_to_problem("Add test coverage for utils", str(tmp_path))
        assert problem.problem_type == "test"

    def test_prd_to_problem_infers_optimize(self, tmp_path):
        (tmp_path / "app.py").write_text("")
        problem = self.bridge.prd_to_problem("Optimize the query performance", str(tmp_path))
        assert problem.problem_type == "optimize"

    def test_prd_to_problem_missing_workspace(self):
        """Non-existent workspace returns empty context."""
        problem = self.bridge.prd_to_problem("Do something", "/nonexistent/path")
        assert problem.context == {}

    def test_solution_to_bcode_output(self):
        """Solution is converted to BCode dict format with 0-100 score."""
        sol = Solution(
            code={"api.py": "from flask import Flask", "test.py": "import pytest"},
            explanation="Built a REST API",
            strategy_used="thorough",
            score=0.85,
            subscores={"syntax": 1.0, "review": 0.8},
        )
        output = self.bridge.solution_to_bcode_output(sol)

        assert len(output["files"]) == 2
        assert output["score"] == 85.0
        assert output["explanation"] == "Built a REST API"
        assert output["strategy"] == "thorough"
        assert output["subscores"]["syntax"] == 1.0

    def test_bcode_output_to_solution(self):
        """BCode dict is converted to Solution with normalized score."""
        bcode_out = {
            "files": [
                {"path": "server.py", "content": "import http"},
                {"path": "client.py", "content": "import requests"},
            ],
            "score": 72,
            "explanation": "Implemented the feature",
        }
        sol = self.bridge.bcode_output_to_solution(bcode_out)

        assert isinstance(sol, Solution)
        assert len(sol.code) == 2
        assert sol.code["server.py"] == "import http"
        assert sol.score == pytest.approx(0.72)
        assert sol.strategy_used == "bcode_pipeline"

    def test_bcode_output_to_solution_empty(self):
        """Empty BCode output produces a zero-score Solution."""
        sol = self.bridge.bcode_output_to_solution({})
        assert sol.score == 0.0
        assert sol.code == {}

    def test_roundtrip_solution(self):
        """Solution -> BCode output -> Solution preserves key data."""
        original = Solution(
            code={"main.py": "print(1)"},
            explanation="Quick fix",
            strategy_used="fast",
            score=0.65,
        )
        bcode = self.bridge.solution_to_bcode_output(original)
        restored = self.bridge.bcode_output_to_solution(bcode)

        assert restored.code == original.code
        assert restored.score == pytest.approx(original.score, abs=0.01)


# ======================================================================
# BCodeShadow tests
# ======================================================================


class TestBCodeShadowComparison:
    """Test comparison logic without requiring a real backend."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = str(Path(self.tmpdir) / "shadow.jsonl")
        self.shadow = BCodeShadow(log_path=self.log_path)

    def test_fusen_wins(self):
        """Higher fusen score -> winner is fusen."""
        bcode_out = {"files": [{"path": "a.py", "content": "x"}], "score": 60}
        # Inject a mock solve result
        self.shadow._latest_solve_result = SolveResult(
            problem=Problem(description="test"),
            best=Solution(code={"a.py": "y"}, score=0.85, strategy_used="thorough"),
            mode="isolated",
        )
        result = self.shadow.compare(bcode_out, task="test task")

        assert result.winner == "fusen"
        assert result.fusen_score > result.bcode_score

    def test_bcode_wins(self):
        """Higher bcode score -> winner is bcode."""
        bcode_out = {"files": [{"path": "a.py", "content": "x"}], "score": 90}
        self.shadow._latest_solve_result = SolveResult(
            problem=Problem(description="test"),
            best=Solution(code={"a.py": "y"}, score=0.50, strategy_used="fast"),
            mode="isolated",
        )
        result = self.shadow.compare(bcode_out, task="test task")

        assert result.winner == "bcode"

    def test_tie(self):
        """Scores within 0.02 tolerance -> tie."""
        bcode_out = {"files": [{"path": "a.py", "content": "x"}], "score": 70}
        self.shadow._latest_solve_result = SolveResult(
            problem=Problem(description="test"),
            best=Solution(code={"a.py": "y"}, score=0.71, strategy_used="balanced"),
            mode="isolated",
        )
        result = self.shadow.compare(bcode_out, task="test task")

        assert result.winner == "tie"

    def test_no_fusen_result(self):
        """No fusen result -> bcode wins by default."""
        bcode_out = {"files": [{"path": "a.py", "content": "x"}], "score": 50}
        self.shadow._latest_solve_result = None
        result = self.shadow.compare(bcode_out, task="test task")

        assert result.winner == "bcode"
        assert result.fusen_score == 0.0


class TestBCodeShadowStats:
    """Test stats aggregation and promotion logic."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = str(Path(self.tmpdir) / "shadow.jsonl")
        self.shadow = BCodeShadow(log_path=self.log_path)

    def _write_entries(self, entries: list[dict]):
        with open(self.log_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    def test_empty_stats(self):
        """No log entries -> zero stats."""
        stats = self.shadow.get_stats()
        assert stats["total_runs"] == 0
        assert stats["fusen_win_rate"] == 0.0

    def test_stats_with_data(self):
        """Stats correctly count wins."""
        entries = [
            {"winner": "fusen", "fusen_score": 0.8, "bcode_score": 0.6,
             "fusen_time_s": 10, "bcode_time_s": 5},
            {"winner": "fusen", "fusen_score": 0.9, "bcode_score": 0.7,
             "fusen_time_s": 12, "bcode_time_s": 6},
            {"winner": "bcode", "fusen_score": 0.5, "bcode_score": 0.8,
             "fusen_time_s": 15, "bcode_time_s": 4},
            {"winner": "tie", "fusen_score": 0.7, "bcode_score": 0.7,
             "fusen_time_s": 8, "bcode_time_s": 8},
        ]
        self._write_entries(entries)

        stats = self.shadow.get_stats()
        assert stats["total_runs"] == 4
        assert stats["fusen_win_rate"] == pytest.approx(0.5)
        assert stats["bcode_win_rate"] == pytest.approx(0.25)
        assert stats["tie_rate"] == pytest.approx(0.25)
        assert stats["avg_fusen_score"] == pytest.approx(0.725)

    def test_promotion_not_enough_runs(self):
        """Below min_runs -> should not promote."""
        entries = [{"winner": "fusen"} for _ in range(10)]
        self._write_entries(entries)

        assert not self.shadow.should_promote(min_runs=50, min_win_rate=0.6)

    def test_promotion_low_win_rate(self):
        """Enough runs but low win rate -> should not promote."""
        entries = (
            [{"winner": "fusen"} for _ in range(20)]
            + [{"winner": "bcode"} for _ in range(40)]
        )
        self._write_entries(entries)

        assert not self.shadow.should_promote(min_runs=50, min_win_rate=0.6)

    def test_promotion_meets_criteria(self):
        """Enough runs and high win rate -> should promote."""
        entries = (
            [{"winner": "fusen"} for _ in range(40)]
            + [{"winner": "bcode"} for _ in range(10)]
            + [{"winner": "tie"} for _ in range(5)]
        )
        self._write_entries(entries)

        # 40/55 = 72.7% win rate, 55 runs
        assert self.shadow.should_promote(min_runs=50, min_win_rate=0.6)

    def test_log_append(self):
        """Logging appends to JSONL without overwriting."""
        r1 = ShadowResult(
            task="task1", bcode_score=0.5, fusen_score=0.7,
            bcode_time_s=5, fusen_time_s=10, bcode_files=1, fusen_files=2,
            fusen_mode_used="isolated", fusen_strategy_used="thorough",
            winner="fusen",
        )
        r2 = ShadowResult(
            task="task2", bcode_score=0.8, fusen_score=0.6,
            bcode_time_s=4, fusen_time_s=12, bcode_files=3, fusen_files=1,
            fusen_mode_used="racing", fusen_strategy_used="fast",
            winner="bcode",
        )
        self.shadow._log(r1)
        self.shadow._log(r2)

        entries = self.shadow._read_log()
        assert len(entries) == 2
        assert entries[0]["task"] == "task1"
        assert entries[1]["task"] == "task2"


class TestBCodeShadowRun:
    """Test shadow_run with mocked solver."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = str(Path(self.tmpdir) / "shadow.jsonl")
        self.workspace = tempfile.mkdtemp()
        Path(self.workspace, "main.py").write_text("print('hello')")

    @pytest.mark.asyncio
    async def test_shadow_run_success(self):
        """Shadow run with mocked solver produces a logged result."""
        mock_solve_result = SolveResult(
            problem=Problem(description="test"),
            best=Solution(
                code={"main.py": "print('world')"},
                score=0.75,
                strategy_used="balanced",
            ),
            mode="isolated",
            total_time_s=2.5,
        )

        shadow = BCodeShadow(log_path=self.log_path)

        with patch.object(shadow, "_make_solver") as mock_make:
            mock_solver = MagicMock()
            mock_solver.solve = AsyncMock(return_value=mock_solve_result)
            mock_make.return_value = mock_solver

            result = await shadow.shadow_run(
                task="Improve the greeting",
                codebase_path=self.workspace,
            )

        assert isinstance(result, ShadowResult)
        assert result.fusen_score == 0.75
        assert result.fusen_files == 1
        assert result.fusen_mode_used == "isolated"
        assert result.fusen_strategy_used == "balanced"
        assert result.timestamp != ""

        # Verify it was logged
        entries = shadow._read_log()
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_shadow_run_with_bcode_comparison(self):
        """Shadow run compares against provided BCode output."""
        mock_solve_result = SolveResult(
            problem=Problem(description="test"),
            best=Solution(code={"main.py": "v2"}, score=0.80, strategy_used="thorough"),
            mode="collaborative",
            total_time_s=5.0,
        )

        bcode_output = {
            "files": [{"path": "main.py", "content": "v1"}],
            "score": 65,
        }

        shadow = BCodeShadow(log_path=self.log_path)

        with patch.object(shadow, "_make_solver") as mock_make:
            mock_solver = MagicMock()
            mock_solver.solve = AsyncMock(return_value=mock_solve_result)
            mock_make.return_value = mock_solver

            result = await shadow.shadow_run(
                task="Update greeting",
                codebase_path=self.workspace,
                bcode_output=bcode_output,
                bcode_time_s=3.0,
            )

        assert result.winner == "fusen"
        assert result.bcode_score == pytest.approx(0.65)
        assert result.fusen_score == 0.80
        assert result.bcode_time_s == 3.0

    @pytest.mark.asyncio
    async def test_shadow_run_timeout(self):
        """Shadow run that times out still produces a result."""
        shadow = BCodeShadow(log_path=self.log_path, timeout_s=0.01)

        async def slow_solve(*args, **kwargs):
            await asyncio.sleep(10)
            return SolveResult(problem=Problem(description="test"))

        with patch.object(shadow, "_make_solver") as mock_make:
            mock_solver = MagicMock()
            mock_solver.solve = slow_solve
            mock_make.return_value = mock_solver

            result = await shadow.shadow_run(
                task="Slow task",
                codebase_path=self.workspace,
            )

        assert result.fusen_score == 0.0
        assert "did not produce" in result.notes

    @pytest.mark.asyncio
    async def test_shadow_run_exception(self):
        """Shadow run that raises still produces a result (no crash)."""
        shadow = BCodeShadow(log_path=self.log_path)

        with patch.object(shadow, "_make_solver") as mock_make:
            mock_solver = MagicMock()
            mock_solver.solve = AsyncMock(side_effect=RuntimeError("Backend down"))
            mock_make.return_value = mock_solver

            result = await shadow.shadow_run(
                task="Failing task",
                codebase_path=self.workspace,
            )

        assert result.fusen_score == 0.0
        assert isinstance(result, ShadowResult)
