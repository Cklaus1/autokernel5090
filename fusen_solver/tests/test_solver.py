"""Tests for the Fusen parallel solver.

All tests run without live LLM backends -- HTTP calls are mocked.
Run with: python -m pytest fusen_solver/tests/ -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fusen_solver.core.interfaces import LLMBackend, Problem, Solution, Strategy
from fusen_solver.core.solver import FusenSolver, SolveResult
from fusen_solver.strategies.engine import StrategyEngine
from fusen_solver.strategies.presets import STRATEGY_CATALOG, STRATEGY_PRESETS, get_strategy
from fusen_solver.learning.tracker import LearningEngine
from fusen_solver.scoring.engine import ScoringEngine


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockBackend(LLMBackend):
    """Mock LLM backend for testing."""

    def __init__(self, response: str = "def fix(): return 42"):
        self._response = response

    async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None):
        return self._response

    async def stream(self, messages, *, max_tokens=4096, temperature=0.7, stop=None):
        for word in self._response.split():
            yield word + " "

    @property
    def supports_batch(self):
        return True

    @property
    def max_context(self):
        return 128000


SAMPLE_PROBLEM = Problem(
    description="Fix the off-by-one error in pagination",
    context={"app.py": "def paginate(items, page, size):\n    start = page * size\n    return items[start:start+size]\n"},
    problem_type="bug_fix",
)


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------


class TestInterfaces:
    def test_problem_defaults(self):
        p = Problem(description="test")
        assert p.problem_type == "auto"
        assert p.language == "auto"
        assert p.priority == "quality"
        assert p.context == {}

    def test_solution_defaults(self):
        s = Solution()
        assert s.score == 0.0
        assert s.code == {}
        assert s.strategy_used == ""

    def test_strategy_defaults(self):
        s = Strategy(name="test", prompt="do something")
        assert s.weight == 1.0
        assert s.temperature == 0.7
        assert s.tags == []


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------


class TestStrategies:
    def test_all_strategies_have_prompts(self):
        for name, strategy in STRATEGY_CATALOG.items():
            assert len(strategy.prompt) > 20, f"Strategy '{name}' has empty prompt"

    def test_presets_reference_valid_strategies(self):
        for preset_name, strategy_names in STRATEGY_PRESETS.items():
            assert len(strategy_names) >= 2, f"Preset '{preset_name}' needs >= 2 strategies"
            for sname in strategy_names:
                assert sname in STRATEGY_CATALOG, f"Preset '{preset_name}' references unknown '{sname}'"

    def test_get_strategy_known(self):
        s = get_strategy("direct")
        assert s.name == "direct"
        assert "root cause" in s.prompt.lower() or "direct" in s.prompt.lower()

    def test_get_strategy_unknown_passthrough(self):
        s = get_strategy("my_custom_approach")
        assert s.name == "my_custom_approach"
        assert s.prompt == "my_custom_approach"

    def test_strategy_engine_select(self):
        engine = StrategyEngine()
        strategies = engine.select_strategies(SAMPLE_PROBLEM, n=3)
        assert len(strategies) <= 3
        assert all(isinstance(s, Strategy) for s in strategies)

    def test_strategy_engine_infer_type(self):
        engine = StrategyEngine()
        bug_problem = Problem(description="Fix the crash in the login handler")
        strategies = engine.select_strategies(bug_problem, n=2)
        # Should select from bug_fix preset
        assert len(strategies) > 0

    def test_strategy_engine_with_weights(self):
        engine = StrategyEngine()
        weights = {"direct": 10.0, "test_first": 0.01}
        strategies = engine.select_strategies(SAMPLE_PROBLEM, n=2, weights=weights)
        # "direct" should be very likely to be selected
        assert len(strategies) == 2


# ---------------------------------------------------------------------------
# Learning engine tests
# ---------------------------------------------------------------------------


class TestLearningEngine:
    def test_empty_weights(self):
        engine = LearningEngine(db_path="/tmp/fusen_test_nonexistent.json")
        weights = engine.get_weights("bug_fix")
        assert weights == {}

    def test_suggest_n_default(self):
        engine = LearningEngine(db_path="/tmp/fusen_test_nonexistent.json")
        n = engine.suggest_n(SAMPLE_PROBLEM)
        assert n == 4  # default with no data

    def test_record_and_weights(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine = LearningEngine(db_path=db_path, min_data=2)

            # Record several outcomes
            for _ in range(5):
                sols = [
                    Solution(strategy_used="direct", score=0.8),
                    Solution(strategy_used="review", score=0.6),
                ]
                engine.record(
                    Problem(description="test", problem_type="bug_fix"),
                    sols,
                    accepted_idx=0,  # direct wins
                )

            weights = engine.get_weights("bug_fix")
            assert "direct" in weights
            assert "review" in weights
            # Direct should have higher weight (more wins)
            assert weights["direct"] > weights["review"]
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            # Write some data
            engine1 = LearningEngine(db_path=db_path, min_data=1)
            sols = [Solution(strategy_used="direct", score=0.9)]
            engine1.record(
                Problem(description="test", problem_type="feature"),
                sols,
                accepted_idx=0,
            )

            # Load in a new instance
            engine2 = LearningEngine(db_path=db_path, min_data=1)
            stats = engine2.get_stats()
            assert "feature" in stats
            assert "direct" in stats["feature"]
        finally:
            Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Scoring engine tests
# ---------------------------------------------------------------------------


class TestScoringEngine:
    def test_syntax_check_valid(self):
        engine = ScoringEngine()
        sol = Solution(code={"test.py": "def add(a, b):\n    return a + b\n"})
        score = engine._check_syntax(sol)
        assert score == 1.0

    def test_syntax_check_invalid(self):
        engine = ScoringEngine()
        sol = Solution(code={"test.py": "def add(a, b:\n    return a + b\n"})
        score = engine._check_syntax(sol)
        assert score == 0.0

    def test_syntax_check_mixed(self):
        engine = ScoringEngine()
        sol = Solution(code={
            "good.py": "x = 1",
            "bad.py": "x = :",
        })
        score = engine._check_syntax(sol)
        assert score == 0.5

    def test_diff_quality_no_context(self):
        engine = ScoringEngine()
        problem = Problem(description="test")
        sol = Solution(code={"main.py": "x = 1\n" * 50})
        score = engine._diff_quality(problem, sol)
        assert 0.0 < score <= 1.0

    def test_diff_quality_with_context(self):
        engine = ScoringEngine()
        problem = Problem(
            description="test",
            context={"main.py": "x = 1\n" * 50},
        )
        # Solution is similar size to original
        sol = Solution(code={"main.py": "x = 2\n" * 50})
        score = engine._diff_quality(problem, sol)
        assert score >= 0.7

    def test_run_tests_pass(self):
        engine = ScoringEngine()
        problem = Problem(
            description="test",
            tests=["python3 -c 'assert 1+1==2'"],
        )
        sol = Solution(code={})
        score = engine._run_tests(problem, sol)
        assert score == 1.0

    def test_run_tests_fail(self):
        engine = ScoringEngine()
        problem = Problem(
            description="test",
            tests=["python3 -c 'assert 1+1==3'"],
        )
        sol = Solution(code={})
        score = engine._run_tests(problem, sol)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Solver integration tests
# ---------------------------------------------------------------------------


class TestFusenSolver:
    @pytest.mark.asyncio
    async def test_basic_solve(self):
        backend = MockBackend(response="```python\ndef fix():\n    return 42\n```\n\nFixed the bug.")
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)
        result = await solver.solve(SAMPLE_PROBLEM)

        assert isinstance(result, SolveResult)
        assert len(result.solutions) > 0
        assert result.best is not None
        assert result.best.score > 0
        assert result.total_time_s > 0

    @pytest.mark.asyncio
    async def test_solve_with_merge(self):
        backend = MockBackend(response="```python\ndef fix():\n    return 42\n```")
        solver = FusenSolver(backend=backend, default_n=3, auto_n=False)
        result = await solver.solve(SAMPLE_PROBLEM, merge=True)

        assert result.merged is not None

    @pytest.mark.asyncio
    async def test_solve_best_of_n(self):
        backend = MockBackend()
        solver = FusenSolver(backend=backend)
        result = await solver.solve_best_of_n(SAMPLE_PROBLEM, n=3)

        assert result.num_agents == 3
        # All should use "direct" strategy
        assert all(s == "direct" for s in result.strategies_used)

    @pytest.mark.asyncio
    async def test_solve_with_callback(self):
        backend = MockBackend()
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)
        received = []

        def on_solution(idx, sol):
            received.append((idx, sol))

        await solver.solve(SAMPLE_PROBLEM, on_solution=on_solution)
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_code_extraction(self):
        response = (
            "Here is the fix:\n\n"
            "```app.py\n"
            "def paginate(items, page, size):\n"
            "    start = (page - 1) * size\n"
            "    return items[start:start+size]\n"
            "```\n"
        )
        backend = MockBackend(response=response)
        solver = FusenSolver(backend=backend, default_n=1, auto_n=False)
        result = await solver.solve(SAMPLE_PROBLEM)

        assert result.best is not None
        assert "app.py" in result.best.code

    @pytest.mark.asyncio
    async def test_record_feedback(self):
        backend = MockBackend()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            learning = LearningEngine(db_path=db_path)
            solver = FusenSolver(backend=backend, learning_engine=learning)

            solutions = [
                Solution(strategy_used="direct", score=0.9),
                Solution(strategy_used="review", score=0.7),
            ]
            await solver.record_feedback(SAMPLE_PROBLEM, solutions, accepted_idx=0)

            stats = learning.get_stats()
            assert "bug_fix" in stats
        finally:
            Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Backend tests
# ---------------------------------------------------------------------------


class TestBackends:
    def test_vllm_backend_properties(self):
        from fusen_solver.backends.vllm_backend import VLLMBackend

        b = VLLMBackend(model="test-model")
        assert b.supports_batch is True
        assert b.max_context == 131072
        assert "test-model" in b.name

    def test_openai_backend_properties(self):
        from fusen_solver.backends.openai_backend import OpenAIBackend

        b = OpenAIBackend(api_key="test", model="gpt-4o")
        assert b.supports_batch is True
        assert "gpt-4o" in b.name

    def test_anthropic_backend_properties(self):
        from fusen_solver.backends.anthropic_backend import AnthropicBackend

        b = AnthropicBackend(api_key="test", model="claude-sonnet-4-20250514")
        assert b.supports_batch is True
        assert "claude" in b.name

    def test_anthropic_message_conversion(self):
        from fusen_solver.backends.anthropic_backend import AnthropicBackend

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        system, user_msgs = AnthropicBackend._convert_messages(messages)
        assert system == "You are helpful."
        assert len(user_msgs) == 1
        assert user_msgs[0]["role"] == "user"

    def test_ollama_backend_properties(self):
        from fusen_solver.backends.ollama_backend import OllamaBackend

        b = OllamaBackend(model="llama3:70b")
        assert b.supports_batch is False
        assert "llama3" in b.name

    def test_multi_backend_routing(self):
        from fusen_solver.backends.multi_backend import MultiBackend

        default = MockBackend("default response")
        review = MockBackend("review response")
        multi = MultiBackend(default=default, routes={"review": review})

        assert multi.route("review") is review
        assert multi.route("direct") is default
        assert multi.route(None) is default


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config(self):
        from fusen_solver.config import default_config

        config = default_config()
        assert "primary" in config.backends
        assert config.backends["primary"]["type"] == "vllm"

    def test_load_config_nonexistent(self):
        from fusen_solver.config import load_config

        config = load_config("/nonexistent/path.yaml")
        assert "primary" in config.backends


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
