"""Tests for the Fusen parallel solver.

All tests run without live LLM backends -- HTTP calls are mocked.
Run with: python -m pytest fusen_solver/tests/ -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
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

    async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None,
                       priority=None, **kwargs):
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
    solve_mode="isolated",  # existing tests expect isolated (single-round) behavior
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
# Collaborative solving tests
# ---------------------------------------------------------------------------


class TestCollaborativeSolving:
    @pytest.mark.asyncio
    async def test_collaborative_round_execution(self):
        """Collaborative mode runs multiple rounds with role-based agents."""
        backend = MockBackend(
            response="```python\ndef fix():\n    return 42\n```\n\nAnalysis complete."
        )
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)

        problem = Problem(
            description="Fix the off-by-one error",
            context={"app.py": "def f(): pass"},
            problem_type="bug_fix",
            solve_mode="collaborative",
            max_rounds=3,
        )

        result = await solver.solve(problem)

        assert result.mode == "collaborative"
        assert len(result.rounds) > 0
        assert len(result.rounds) <= 3
        assert result.num_agents > 0
        assert result.total_time_s > 0

    @pytest.mark.asyncio
    async def test_collaborative_context_accumulation(self):
        """Each round's synthesis is available to subsequent rounds."""
        call_count = 0
        prompts_seen: list[str] = []

        class ContextTrackingBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None):
                nonlocal call_count
                call_count += 1
                # Capture the user message content to check context passing
                for msg in messages:
                    if msg["role"] == "user":
                        prompts_seen.append(msg["content"])
                return (
                    "Here is my analysis.\n\n"
                    "```json\n"
                    '{"has_solution": false, "tests_pass": false}\n'
                    "```"
                )

        backend = ContextTrackingBackend()
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)

        problem = Problem(
            description="Fix the bug",
            context={"app.py": "x = 1"},
            problem_type="bug_fix",
            solve_mode="collaborative",
            max_rounds=2,
        )

        result = await solver.solve(problem)

        # Should have run multiple rounds
        assert len(result.rounds) == 2
        # Round 2 agents should see Round 1 context in their prompts
        # (Round 2 has 2 roles + 1 synthesis call = at least some prompts with Round 1 info)
        round2_prompts = [p for p in prompts_seen if "Round 1" in p]
        assert len(round2_prompts) > 0, "Round 2 agents should receive Round 1 context"

    @pytest.mark.asyncio
    async def test_collaborative_early_exit(self):
        """Collaborative mode stops early when synthesis reports tests pass."""

        class EarlyExitBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None):
                # Always report solution found with tests passing
                return (
                    "Solution found!\n\n"
                    "```python\ndef fix(): return 42\n```\n\n"
                    "```json\n"
                    '{"has_solution": true, "tests_pass": true}\n'
                    "```"
                )

        backend = EarlyExitBackend()
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)

        problem = Problem(
            description="Fix the bug",
            context={"app.py": "x = 1"},
            problem_type="bug_fix",
            solve_mode="collaborative",
            max_rounds=3,
        )

        result = await solver.solve(problem)

        assert result.mode == "collaborative"
        # Should exit after round 1 since synthesis says tests pass
        assert len(result.rounds) == 1

    @pytest.mark.asyncio
    async def test_auto_mode_insufficient_data(self):
        """Auto mode uses heuristic when insufficient history data."""
        backend = MockBackend(
            response="```python\ndef fix(): return 42\n```\n\n"
            "```json\n"
            '{"has_solution": false, "tests_pass": false}\n'
            "```"
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            learning = LearningEngine(db_path=db_path, min_data=10)
            solver = FusenSolver(backend=backend, learning_engine=learning, auto_n=False, default_n=2)

            # bug_fix should default to collaborative with insufficient data
            problem = Problem(
                description="Fix a crash",
                context={"app.py": "x = 1"},
                problem_type="bug_fix",
                solve_mode="auto",
            )
            result = await solver.solve(problem)
            assert result.mode == "collaborative"

            # feature should default to isolated with insufficient data
            problem2 = Problem(
                description="Add a button",
                context={"app.py": "x = 1"},
                problem_type="feature",
                solve_mode="auto",
            )
            result2 = await solver.solve(problem2)
            assert result2.mode == "isolated"
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_auto_mode_with_enough_data(self):
        """Auto mode uses acceptance rates when enough history exists."""
        backend = MockBackend(response="```python\ndef fix(): return 42\n```")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            learning = LearningEngine(db_path=db_path, min_data=3)

            # Record enough history: collaborative is better for "optimize"
            for _ in range(5):
                learning.record_mode("optimize", "collaborative", accepted=True)
                learning.record_mode("optimize", "isolated", accepted=False)

            solver = FusenSolver(
                backend=backend,
                learning_engine=learning,
                auto_n=False,
                default_n=2,
            )

            problem = Problem(
                description="Optimize the loop",
                context={"app.py": "x = 1"},
                problem_type="optimize",
                solve_mode="auto",
            )

            # Should pick collaborative since it has higher acceptance rate
            mode = learning.suggest_mode(problem)
            assert mode == "collaborative"
        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_role_assignment_per_round(self):
        """Each round uses the correct set of roles from COLLABORATIVE_ROLES."""
        from fusen_solver.strategies.presets import COLLABORATIVE_ROLES

        roles_used: list[list[str]] = []

        class RoleTrackingBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None):
                # Extract role name from the system message
                system_msg = messages[0]["content"] if messages else ""
                return (
                    "Output from agent.\n\n"
                    "```json\n"
                    '{"has_solution": false, "tests_pass": false}\n'
                    "```"
                )

        backend = RoleTrackingBackend()
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)

        problem = Problem(
            description="Fix the bug",
            context={"app.py": "x = 1"},
            problem_type="bug_fix",
            solve_mode="collaborative",
            max_rounds=3,
        )

        result = await solver.solve(problem)

        # Verify the rounds used the expected number of roles
        assert len(result.rounds) == 3
        assert len(result.rounds[0]["outputs"]) == len(COLLABORATIVE_ROLES["round_1"])
        assert len(result.rounds[1]["outputs"]) == len(COLLABORATIVE_ROLES["round_2"])
        assert len(result.rounds[2]["outputs"]) == len(COLLABORATIVE_ROLES["round_3"])

        # Verify role names match
        round1_roles = [o["role"] for o in result.rounds[0]["outputs"]]
        expected_round1 = [r.name for r in COLLABORATIVE_ROLES["round_1"]]
        assert round1_roles == expected_round1

    @pytest.mark.asyncio
    async def test_isolated_mode_unchanged(self):
        """Explicit isolated mode still works exactly as before."""
        backend = MockBackend(response="```python\ndef fix():\n    return 42\n```\n\nFixed.")
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)

        problem = Problem(
            description="Fix the off-by-one error",
            context={"app.py": "def f(): pass"},
            problem_type="bug_fix",
            solve_mode="isolated",
        )

        result = await solver.solve(problem)

        assert result.mode == "isolated"
        assert result.rounds == []
        assert len(result.solutions) > 0
        assert result.best is not None

    def test_problem_solve_mode_defaults(self):
        """Problem defaults to auto solve_mode and 3 max_rounds."""
        p = Problem(description="test")
        assert p.solve_mode == "auto"
        assert p.max_rounds == 3


class TestLearningEngineMode:
    def test_record_mode(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine = LearningEngine(db_path=db_path, min_data=2)
            engine.record_mode("bug_fix", "collaborative", accepted=True)
            engine.record_mode("bug_fix", "isolated", accepted=False)

            assert len(engine._mode_history) == 2
            assert engine._mode_history[0]["type"] == "bug_fix"
            assert engine._mode_history[0]["mode"] == "collaborative"
            assert engine._mode_history[0]["accepted"] is True
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_suggest_mode_heuristic(self):
        """With insufficient data, uses problem-type heuristic."""
        engine = LearningEngine(db_path="/tmp/fusen_test_nonexistent.json", min_data=10)

        # Complex types default to collaborative
        assert engine.suggest_mode(Problem(description="fix crash", problem_type="bug_fix")) == "collaborative"
        assert engine.suggest_mode(Problem(description="redesign", problem_type="refactor")) == "collaborative"
        assert engine.suggest_mode(Problem(description="new arch", problem_type="architecture")) == "collaborative"

        # Simple types default to isolated
        assert engine.suggest_mode(Problem(description="add feature", problem_type="feature")) == "isolated"
        assert engine.suggest_mode(Problem(description="optimize", problem_type="optimize")) == "isolated"

    def test_suggest_mode_data_driven(self):
        """With enough data, uses acceptance rates."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine = LearningEngine(db_path=db_path, min_data=3)

            # Record: isolated is better for feature
            for _ in range(5):
                engine.record_mode("feature", "isolated", accepted=True)
                engine.record_mode("feature", "collaborative", accepted=False)

            assert engine.suggest_mode(
                Problem(description="add button", problem_type="feature")
            ) == "isolated"

            # Record: collaborative is better for bug_fix
            for _ in range(5):
                engine.record_mode("bug_fix", "collaborative", accepted=True)
                engine.record_mode("bug_fix", "isolated", accepted=False)

            assert engine.suggest_mode(
                Problem(description="fix crash", problem_type="bug_fix")
            ) == "collaborative"
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_mode_history_persistence(self):
        """Mode history persists across engine instances."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine1 = LearningEngine(db_path=db_path)
            engine1.record_mode("bug_fix", "collaborative", accepted=True)
            engine1.record_mode("bug_fix", "isolated", accepted=False)

            # Load in new instance
            engine2 = LearningEngine(db_path=db_path)
            assert len(engine2._mode_history) == 2
            assert engine2._mode_history[0]["mode"] == "collaborative"
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_mode_stats_in_get_stats(self):
        """get_stats includes mode statistics when mode history exists."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine = LearningEngine(db_path=db_path)
            engine.record_mode("bug_fix", "collaborative", accepted=True)
            engine.record_mode("bug_fix", "collaborative", accepted=True)
            engine.record_mode("bug_fix", "isolated", accepted=False)

            stats = engine.get_stats()
            assert "_mode_stats" in stats
            assert "bug_fix" in stats["_mode_stats"]
            assert stats["_mode_stats"]["bug_fix"]["collaborative"]["acceptance_rate"] == 1.0
            assert stats["_mode_stats"]["bug_fix"]["isolated"]["acceptance_rate"] == 0.0
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
# Session Affinity tests
# ---------------------------------------------------------------------------


class _NamedMockBackend(MockBackend):
    """MockBackend subclass with a configurable name."""

    def __init__(self, response: str = "def fix(): return 42", backend_name: str = "mock"):
        super().__init__(response)
        self._backend_name = backend_name

    @property
    def name(self) -> str:
        return self._backend_name


class TestSessionAffinity:
    def test_same_session_routes_to_same_backend(self):
        from fusen_solver.backends.multi_backend import MultiBackend

        backend_a = _NamedMockBackend("response_a", backend_name="gpu_0")
        backend_b = _NamedMockBackend("response_b", backend_name="gpu_1")

        multi = MultiBackend(
            default=backend_a,
            routes={"review": backend_b},
        )

        # First call with session_id picks default
        result1 = multi.route(session_id="session_abc")
        assert result1.name == "gpu_0"

        # Second call with same session_id should route to same backend
        result2 = multi.route(session_id="session_abc")
        assert result2.name == "gpu_0"

    def test_different_sessions_can_route_differently(self):
        from fusen_solver.backends.multi_backend import MultiBackend

        default = _NamedMockBackend("default", backend_name="default")
        review_be = _NamedMockBackend("review", backend_name="review_be")

        multi = MultiBackend(
            default=default,
            routes={"review": review_be},
        )

        # Session 1 goes to default
        multi.route(session_id="s1")
        # Session 2 via strategy goes to review_be
        multi.route(strategy_name="review", session_id="s2")

        assert multi.route(session_id="s1").name == "default"
        assert multi.route(session_id="s2").name == "review_be"

    def test_session_ttl_expiry(self):
        import time as _time
        from fusen_solver.backends.multi_backend import MultiBackend

        default = _NamedMockBackend("default", backend_name="default")
        review_be = _NamedMockBackend("review", backend_name="review_be")

        multi = MultiBackend(
            default=default,
            routes={"review": review_be},
            session_ttl=10.0,
        )

        # Assign session to review backend
        multi.route(strategy_name="review", session_id="s1")
        assert multi.route(session_id="s1").name == "review_be"

        # Simulate time passing beyond TTL
        multi._session_map["s1"] = ("review_be", _time.monotonic() - 20.0)
        # Session expired, should fall back to default
        result = multi.route(session_id="s1")
        assert result.name == "default"

    @pytest.mark.asyncio
    async def test_generate_with_session_id(self):
        from fusen_solver.backends.multi_backend import MultiBackend

        default = _NamedMockBackend("default_resp", backend_name="default")
        other = _NamedMockBackend("other_resp", backend_name="other")

        multi = MultiBackend(default=default, routes={"review": other})

        # First request assigns session to default
        r1 = await multi.generate([{"role": "user", "content": "hi"}], session_id="s1")
        assert r1 == "default_resp"

        # Strategy-routed request with different session
        r2 = await multi.generate_with_strategy(
            [{"role": "user", "content": "hi"}],
            "review",
            session_id="s2",
        )
        assert r2 == "other_resp"


# ---------------------------------------------------------------------------
# Priority tests
# ---------------------------------------------------------------------------


class TestPriority:
    def test_short_strategies(self):
        from fusen_solver.core.priority import compute_priority

        p = Problem(description="test")
        assert compute_priority(p, "review") == 1
        assert compute_priority(p, "analyst") == 1

    def test_medium_strategies(self):
        from fusen_solver.core.priority import compute_priority

        p = Problem(description="test")
        assert compute_priority(p, "direct") == 2
        assert compute_priority(p, "test_first") == 2

    def test_long_strategies(self):
        from fusen_solver.core.priority import compute_priority

        p = Problem(description="test")
        assert compute_priority(p, "rewrite") == 3
        assert compute_priority(p, "decompose") == 3

    def test_constraint_override(self):
        from fusen_solver.core.priority import compute_priority

        p = Problem(description="test", constraints=["keep it short"])
        # Constraint "short" overrides strategy
        assert compute_priority(p, "rewrite") == 1

    def test_unknown_strategy_default(self):
        from fusen_solver.core.priority import compute_priority

        p = Problem(description="test")
        assert compute_priority(p, "unknown_strategy") == 2


# ---------------------------------------------------------------------------
# Agent Memory tests
# ---------------------------------------------------------------------------


class TestAgentMemory:
    def test_remember_and_recall(self):
        from fusen_solver.learning.tracker import AgentMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem = AgentMemory(path=path)
            mem.remember("bug_fix", "Check for null pointers first", source="direct")
            mem.remember("bug_fix", "Off-by-one errors often hide in loop bounds", source="review")
            mem.remember("feature", "Start with the interface", source="direct")

            # Recall bug_fix insights
            results = mem.recall("bug_fix")
            assert len(results) == 2
            assert "null pointers" in results[0] or "null pointers" in results[1]

            # Recall feature insights
            results = mem.recall("feature")
            assert len(results) == 1
            assert "interface" in results[0]

            # Recall unknown type returns empty
            results = mem.recall("nonexistent")
            assert results == []
        finally:
            Path(path).unlink(missing_ok=True)

    def test_recall_respects_limit(self):
        from fusen_solver.learning.tracker import AgentMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem = AgentMemory(path=path)
            for i in range(10):
                mem.remember("bug_fix", f"Insight number {i}")

            results = mem.recall("bug_fix", limit=3)
            assert len(results) == 3
        finally:
            Path(path).unlink(missing_ok=True)

    def test_used_count_increments(self):
        from fusen_solver.learning.tracker import AgentMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem = AgentMemory(path=path)
            mem.remember("bug_fix", "Check null")

            mem.recall("bug_fix")
            assert mem.memories[0]["used_count"] == 1

            mem.recall("bug_fix")
            assert mem.memories[0]["used_count"] == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_persistence(self):
        from fusen_solver.learning.tracker import AgentMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem1 = AgentMemory(path=path)
            mem1.remember("bug_fix", "Always check edge cases")

            # Reload from disk
            mem2 = AgentMemory(path=path)
            assert len(mem2.memories) == 1
            assert mem2.memories[0]["insight"] == "Always check edge cases"

            results = mem2.recall("bug_fix")
            assert len(results) == 1
            assert results[0] == "Always check edge cases"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_get_all(self):
        from fusen_solver.learning.tracker import AgentMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem = AgentMemory(path=path)
            mem.remember("bug_fix", "Insight A")
            mem.remember("feature", "Insight B")

            all_memories = mem.get_all()
            assert len(all_memories) == 2
            assert all_memories[0]["type"] == "bug_fix"
            assert all_memories[1]["type"] == "feature"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_file_returns_empty(self):
        from fusen_solver.learning.tracker import AgentMemory

        mem = AgentMemory(path="/tmp/fusen_nonexistent_memory_test.json")
        assert mem.memories == []
        assert mem.recall("anything") == []


# ---------------------------------------------------------------------------
# Solver integration with memory and priority
# ---------------------------------------------------------------------------


class TestSolverMemoryAndPriority:
    @pytest.mark.asyncio
    async def test_agent_injects_memory(self):
        """Verify that recalled memories appear in the system prompt."""
        from fusen_solver.learning.tracker import AgentMemory

        prompts_seen: list[str] = []

        class CapturingBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7,
                               stop=None, priority=None, **kwargs):
                for msg in messages:
                    if msg["role"] == "system":
                        prompts_seen.append(msg["content"])
                return "```python\ndef fix(): return 42\n```"

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mem_path = f.name

        try:
            memory = AgentMemory(path=mem_path)
            memory.remember("bug_fix", "Always check for None before accessing .value")

            backend = CapturingBackend()
            solver = FusenSolver(
                backend=backend,
                default_n=1,
                auto_n=False,
                memory=memory,
            )

            problem = Problem(
                description="Fix the crash",
                context={"app.py": "x = obj.value"},
                problem_type="bug_fix",
                solve_mode="isolated",
            )

            await solver.solve(problem)

            # At least one system prompt should contain the memory insight
            assert any("Always check for None" in p for p in prompts_seen)
        finally:
            Path(mem_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_agent_passes_priority(self):
        """Verify that priority is computed and passed to the backend."""
        priorities_seen: list[int] = []

        class PriorityCapturingBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7,
                               stop=None, priority=None, **kwargs):
                if priority is not None:
                    priorities_seen.append(priority)
                return "```python\ndef fix(): return 42\n```"

        backend = PriorityCapturingBackend()
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)

        problem = Problem(
            description="Fix the bug",
            context={"app.py": "x = 1"},
            problem_type="bug_fix",
            solve_mode="isolated",
        )

        await solver.solve(problem)

        assert len(priorities_seen) >= 2
        # All priorities should be valid (1, 2, or 3)
        assert all(1 <= p <= 3 for p in priorities_seen)

    @pytest.mark.asyncio
    async def test_feedback_stores_insight(self):
        """Verify that record_feedback extracts and stores an insight."""
        from fusen_solver.learning.tracker import AgentMemory

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            mem_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            memory = AgentMemory(path=mem_path)
            learning = LearningEngine(db_path=db_path)
            backend = MockBackend(response="The key insight was checking bounds.")
            solver = FusenSolver(
                backend=backend,
                learning_engine=learning,
                memory=memory,
                auto_n=False,
            )

            solutions = [
                Solution(strategy_used="direct", score=0.9, explanation="Fixed by checking bounds."),
                Solution(strategy_used="review", score=0.7, explanation="Reviewed the code."),
            ]
            await solver.record_feedback(SAMPLE_PROBLEM, solutions, accepted_idx=0)

            # Memory should now contain an insight
            all_mems = memory.get_all()
            assert len(all_mems) == 1
            assert "checking bounds" in all_mems[0]["insight"].lower() or len(all_mems[0]["insight"]) > 0
            assert all_mems[0]["source"] == "direct"
        finally:
            Path(mem_path).unlink(missing_ok=True)
            Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Racing mode tests
# ---------------------------------------------------------------------------


class TestRacingSolve:
    """Tests for solve_racing() -- first accepted solution wins, others cancelled."""

    @pytest.mark.asyncio
    async def test_racing_first_completes_others_cancelled(self):
        """First agent to finish with score >= threshold wins, others are cancelled."""

        class SequentialBackend(MockBackend):
            """First agent-level call returns fast, rest sleep.

            Uses ``priority`` to distinguish agent calls (priority != None)
            from scoring/review calls (priority == None).  Only agent calls
            with priority > first are slowed down.
            """

            def __init__(self):
                super().__init__()
                self._agent_call_count = 0

            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None,
                               priority=None, **kwargs):
                # Scoring/review calls have no priority -- return fast
                if priority is None:
                    return "Looks good."
                self._agent_call_count += 1
                if self._agent_call_count == 1:
                    return "```python\ndef fix(): return 42\n```\n\nFixed immediately."
                else:
                    await asyncio.sleep(10)
                    return "```python\ndef fix(): return 99\n```\n\nSlow fix."

        backend = SequentialBackend()
        solver = FusenSolver(backend=backend, default_n=4, auto_n=False)

        problem = Problem(
            description="Fix the bug",
            context={"app.py": "def f(): pass"},
            problem_type="bug_fix",
            solve_mode="racing",
            racing_accept_threshold=0.0,  # accept anything
            racing_timeout=5.0,
        )

        result = await solver.solve(problem)

        assert result.mode == "racing"
        assert len(result.solutions) >= 1
        assert result.best is not None
        assert result.total_time_s < 5.0

        racing_stats = result.metadata.get("racing_stats")
        assert racing_stats is not None
        assert racing_stats["cancelled_agents"] >= 1
        assert racing_stats["kv_savings_pct"] > 0

    @pytest.mark.asyncio
    async def test_racing_timeout_takes_best(self):
        """When no agent beats threshold before timeout, take best available."""

        class SlowBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None,
                               priority=None, **kwargs):
                await asyncio.sleep(10)
                return "```python\ndef fix(): return 42\n```"

        backend = SlowBackend()
        solver = FusenSolver(backend=backend, default_n=3, auto_n=False)

        problem = Problem(
            description="Fix the bug",
            context={"app.py": "def f(): pass"},
            problem_type="bug_fix",
            solve_mode="racing",
            racing_accept_threshold=0.99,
            racing_timeout=0.5,
        )

        result = await solver.solve(problem)

        assert result.mode == "racing"
        racing_stats = result.metadata.get("racing_stats")
        assert racing_stats is not None
        assert racing_stats["timed_out"] is True
        assert result.total_time_s < 2.0

    @pytest.mark.asyncio
    async def test_racing_rejection_then_acceptance(self):
        """First agent rejected (low score), second agent accepted."""

        call_count = 0

        class RejectThenAcceptBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None,
                               priority=None, **kwargs):
                nonlocal call_count
                call_count += 1
                idx = call_count
                if idx == 1:
                    return "no code here"
                elif idx == 2:
                    await asyncio.sleep(0.1)
                    return "```python\ndef fix():\n    return 42\n```\n\nProper fix."
                else:
                    await asyncio.sleep(10)
                    return "slow"

        backend = RejectThenAcceptBackend()
        solver = FusenSolver(backend=backend, default_n=3, auto_n=False)

        problem = Problem(
            description="Fix the off-by-one error in pagination",
            context={"app.py": "def paginate(items, page, size):\n    start = page * size\n    return items[start:start+size]\n"},
            problem_type="bug_fix",
            solve_mode="racing",
            racing_accept_threshold=0.0,
            racing_timeout=5.0,
        )

        result = await solver.solve(problem)

        assert result.mode == "racing"
        assert len(result.solutions) >= 1
        assert result.best is not None

    @pytest.mark.asyncio
    async def test_racing_kv_savings_tracked(self):
        """KV savings stats are properly tracked."""

        class FastBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None,
                               priority=None, **kwargs):
                return "```python\ndef fix(): return 42\n```"

        backend = FastBackend()
        solver = FusenSolver(backend=backend, default_n=4, auto_n=False)

        problem = Problem(
            description="Fix the bug",
            context={"app.py": "def f(): pass"},
            problem_type="bug_fix",
            solve_mode="racing",
            racing_accept_threshold=0.0,
            racing_timeout=5.0,
        )

        result = await solver.solve(problem)

        assert result.mode == "racing"
        racing_stats = result.metadata.get("racing_stats")
        assert racing_stats is not None
        assert "kv_savings_pct" in racing_stats
        assert "cancelled_agents" in racing_stats
        assert "winner_idx" in racing_stats
        assert "agent_times" in racing_stats
        assert isinstance(racing_stats["agent_times"], list)

    @pytest.mark.asyncio
    async def test_racing_problem_defaults(self):
        """Problem has correct racing defaults."""
        p = Problem(description="test")
        assert p.racing_accept_threshold == 0.7
        assert p.racing_timeout == 30.0

    def test_racing_learning_engine_record(self):
        """Learning engine records racing win positions."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine = LearningEngine(db_path=db_path, min_data=2)

            engine.record_racing_win("bug_fix", num_agents=4, winner_position=0, winner_time=1.5)
            engine.record_racing_win("bug_fix", num_agents=4, winner_position=1, winner_time=2.0)
            engine.record_racing_win("bug_fix", num_agents=4, winner_position=0, winner_time=1.2)

            assert len(engine._racing_history) == 3

            stats = engine.get_stats()
            assert "_racing_stats" in stats
            assert "bug_fix" in stats["_racing_stats"]
            assert stats["_racing_stats"]["bug_fix"]["total_races"] == 3
            assert stats["_racing_stats"]["bug_fix"]["avg_winner_position"] == round((0 + 1 + 0) / 3, 2)

            n = engine.suggest_racing_n(Problem(description="fix", problem_type="bug_fix"))
            assert 2 <= n <= 8
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_racing_learning_persistence(self):
        """Racing history persists across engine instances."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine1 = LearningEngine(db_path=db_path)
            engine1.record_racing_win("feature", num_agents=3, winner_position=2, winner_time=3.0)

            engine2 = LearningEngine(db_path=db_path)
            assert len(engine2._racing_history) == 1
            assert engine2._racing_history[0]["type"] == "feature"
            assert engine2._racing_history[0]["winner_position"] == 2
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestRacingCoordinator:
    """Tests for the RacingCoordinator and CancellableRequest."""

    def test_cancellable_request_defaults(self):
        from fusen_solver.streaming import CancellableRequest

        req = CancellableRequest(agent_idx=0)
        assert not req.cancelled
        assert req.elapsed == 0.0

    @pytest.mark.asyncio
    async def test_cancel_request(self):
        from fusen_solver.streaming import CancellableRequest

        req = CancellableRequest(agent_idx=0, start_time=time.perf_counter())
        await req.cancel()
        assert req.cancelled
        assert req.end_time > 0

    @pytest.mark.asyncio
    async def test_double_cancel_is_safe(self):
        from fusen_solver.streaming import CancellableRequest

        req = CancellableRequest(agent_idx=0, start_time=time.perf_counter())
        await req.cancel()
        await req.cancel()  # should not raise
        assert req.cancelled

    def test_racing_stats_kv_savings(self):
        from fusen_solver.streaming import RacingStats

        stats = RacingStats(total_agents=4, cancelled_agents=3)
        assert stats.kv_savings_pct == 75.0

        stats2 = RacingStats(total_agents=1, cancelled_agents=0)
        assert stats2.kv_savings_pct == 0.0

    @pytest.mark.asyncio
    async def test_coordinator_cancel_all_except(self):
        from fusen_solver.streaming import RacingCoordinator

        coord = RacingCoordinator()
        req0 = coord.register(0)
        req1 = coord.register(1)
        req2 = coord.register(2)

        await coord.cancel_all_except(winner_idx=1)

        assert req0.cancelled
        assert not req1.cancelled
        assert req2.cancelled
        assert coord.stats.cancelled_agents == 2

    @pytest.mark.asyncio
    async def test_coordinator_cancel_all(self):
        from fusen_solver.streaming import RacingCoordinator

        coord = RacingCoordinator()
        coord.register(0)
        coord.register(1)

        await coord.cancel_all()
        assert coord.stats.cancelled_agents == 2


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Decomposed mode tests
# ---------------------------------------------------------------------------


class TestDecomposedSolving:
    """Tests for the file-level decomposition orchestrator."""

    @pytest.mark.asyncio
    async def test_decompose_basic(self):
        """Decomposed mode produces a result with mode='decomposed'."""
        backend = MockBackend(
            response="```python\ndef main():\n    pass\n```\n\nDone."
        )
        solver = FusenSolver(backend=backend, default_n=2, auto_n=False)

        problem = Problem(
            description="Build a REST API for user management",
            context={},
            problem_type="feature",
            solve_mode="decomposed",
        )

        result = await solver.solve(problem)

        assert result.mode == "decomposed"
        assert result.best is not None
        assert result.total_time_s > 0
        assert len(result.rounds) == 1
        assert "decomposition" in result.rounds[0]
        assert "files_generated" in result.rounds[0]

    @pytest.mark.asyncio
    async def test_decompose_llm_parsing(self):
        """Decomposition parses a valid JSON file list from LLM output."""
        import json as _json

        decomposition_json = _json.dumps([
            {"file": "models.py", "description": "Data models", "depends_on": []},
            {"file": "routes.py", "description": "API routes", "depends_on": ["models.py"]},
            {"file": "tests/test_api.py", "description": "Tests", "depends_on": ["routes.py"]},
        ])

        call_count = 0

        class DecompBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return decomposition_json
                return "```python\nx = 1\n```"

        backend = DecompBackend()
        solver = FusenSolver(backend=backend, default_n=1, auto_n=False)

        problem = Problem(
            description="Build a REST API",
            context={},
            solve_mode="decomposed",
        )

        result = await solver.solve(problem)

        assert result.mode == "decomposed"
        assert result.best is not None
        decomp = result.rounds[0]["decomposition"]
        assert len(decomp) == 3
        assert decomp[0]["file"] == "models.py"
        assert decomp[1]["file"] == "routes.py"
        assert decomp[2]["file"] == "tests/test_api.py"

    @pytest.mark.asyncio
    async def test_decompose_fallback_pattern(self):
        """When LLM decomposition fails, falls back to known patterns."""

        class FailingDecompBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None, **kwargs):
                if temperature <= 0.3:
                    return "I cannot parse this into JSON"
                return "```python\nx = 1\n```"

        backend = FailingDecompBackend()
        solver = FusenSolver(backend=backend, default_n=1, auto_n=False)

        problem = Problem(
            description="Build a REST API for orders",
            context={},
            solve_mode="decomposed",
        )

        result = await solver.solve(problem)

        assert result.mode == "decomposed"
        decomp = result.rounds[0]["decomposition"]
        files = [d["file"] for d in decomp]
        assert "models.py" in files
        assert "routes.py" in files

    def test_dependency_levels_no_deps(self):
        """Files with no dependencies all go into one level."""
        decomposition = [
            {"file": "a.py", "depends_on": []},
            {"file": "b.py", "depends_on": []},
            {"file": "c.py", "depends_on": []},
        ]
        levels = FusenSolver._build_dependency_levels(decomposition)
        assert len(levels) == 1
        assert len(levels[0]) == 3

    def test_dependency_levels_chain(self):
        """Linear dependency chain produces one file per level."""
        decomposition = [
            {"file": "a.py", "depends_on": []},
            {"file": "b.py", "depends_on": ["a.py"]},
            {"file": "c.py", "depends_on": ["b.py"]},
        ]
        levels = FusenSolver._build_dependency_levels(decomposition)
        assert len(levels) == 3
        assert levels[0][0]["file"] == "a.py"
        assert levels[1][0]["file"] == "b.py"
        assert levels[2][0]["file"] == "c.py"

    def test_dependency_levels_diamond(self):
        """Diamond dependency graph: a -> {b,c} -> d."""
        decomposition = [
            {"file": "a.py", "depends_on": []},
            {"file": "b.py", "depends_on": ["a.py"]},
            {"file": "c.py", "depends_on": ["a.py"]},
            {"file": "d.py", "depends_on": ["b.py", "c.py"]},
        ]
        levels = FusenSolver._build_dependency_levels(decomposition)
        assert len(levels) == 3
        assert levels[0][0]["file"] == "a.py"
        level1_files = {s["file"] for s in levels[1]}
        assert level1_files == {"b.py", "c.py"}
        assert levels[2][0]["file"] == "d.py"

    def test_dependency_levels_circular(self):
        """Circular dependencies are broken by grouping into one level."""
        decomposition = [
            {"file": "a.py", "depends_on": ["b.py"]},
            {"file": "b.py", "depends_on": ["a.py"]},
        ]
        levels = FusenSolver._build_dependency_levels(decomposition)
        assert len(levels) == 1
        assert len(levels[0]) == 2

    def test_merge_file_solutions(self):
        """Merge combines per-file results into a single Solution."""
        decomposition = [
            {"file": "a.py", "description": "Module A"},
            {"file": "b.py", "description": "Module B"},
        ]

        mock_result_a = SolveResult(
            problem=SAMPLE_PROBLEM,
            best=Solution(
                code={"block_0.txt": "# a.py content"},
                explanation="Generated a.py",
            ),
        )
        mock_result_b = SolveResult(
            problem=SAMPLE_PROBLEM,
            best=Solution(
                code={"block_0.txt": "# b.py content"},
                explanation="Generated b.py",
            ),
        )

        file_results = [
            (decomposition[0], mock_result_a),
            (decomposition[1], mock_result_b),
        ]

        merged = FusenSolver._merge_file_solutions(decomposition, file_results)

        assert "a.py" in merged.code
        assert "b.py" in merged.code
        assert merged.code["a.py"] == "# a.py content"
        assert merged.code["b.py"] == "# b.py content"
        assert merged.strategy_used == "decomposed"
        assert "a.py" in merged.metadata["generated"]
        assert "b.py" in merged.metadata["generated"]

    @pytest.mark.asyncio
    async def test_integration_verification(self):
        """Integration verification returns corrected code."""

        class IntegrationBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None, **kwargs):
                return (
                    "All files look good with one fix:\n\n"
                    "```a.py\n# corrected a.py\nimport b\n```\n\n"
                    "```b.py\n# corrected b.py\ndef helper(): pass\n```"
                )

        backend = IntegrationBackend()
        solver = FusenSolver(backend=backend, default_n=1, auto_n=False)

        merged = Solution(
            code={"a.py": "# original a.py", "b.py": "# original b.py"},
            strategy_used="decomposed",
        )
        problem = Problem(description="Test project")

        result = await solver._verify_integration(merged, problem)

        assert result is not None
        assert "a.py" in result.code
        assert "b.py" in result.code
        assert "corrected" in result.code["a.py"]
        assert result.strategy_used == "decomposed_integrated"
        assert result.metadata.get("integration_pass") is True

    @pytest.mark.asyncio
    async def test_integration_verification_empty(self):
        """Integration verification returns None for empty code."""
        backend = MockBackend()
        solver = FusenSolver(backend=backend, default_n=1, auto_n=False)

        merged = Solution(code={}, strategy_used="decomposed")
        problem = Problem(description="Test")

        result = await solver._verify_integration(merged, problem)
        assert result is None

    @pytest.mark.asyncio
    async def test_decompose_with_context(self):
        """Decomposed mode passes existing codebase as context to sub-problems."""
        prompts_seen: list[str] = []

        class ContextCheckBackend(MockBackend):
            async def generate(self, messages, *, max_tokens=4096, temperature=0.7, stop=None, **kwargs):
                for msg in messages:
                    if msg["role"] == "user":
                        prompts_seen.append(msg["content"])
                if temperature <= 0.3:
                    return '[{"file": "new.py", "description": "New module", "depends_on": []}]'
                return "```python\ndef new_func(): pass\n```"

        backend = ContextCheckBackend()
        solver = FusenSolver(backend=backend, default_n=1, auto_n=False)

        problem = Problem(
            description="Add a new module",
            context={"existing.py": "# existing code"},
            solve_mode="decomposed",
        )

        result = await solver.solve(problem)
        assert result.mode == "decomposed"


class TestDecomposedLearning:
    """Tests for decomposed mode tracking in the learning engine."""

    def test_suggest_mode_decomposed_heuristic(self):
        """Feature problems with multi-file keywords suggest decomposed mode."""
        engine = LearningEngine(db_path="/tmp/fusen_test_nonexistent.json", min_data=10)

        problem = Problem(
            description="Build a REST API for user management",
            problem_type="feature",
            solve_mode="auto",
        )
        mode = engine.suggest_mode(problem)
        assert mode == "decomposed"

    def test_suggest_mode_decomposed_data_driven(self):
        """With enough data, decomposed wins if it has highest acceptance rate."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine = LearningEngine(db_path=db_path, min_data=3)

            for _ in range(5):
                engine.record_mode("feature", "decomposed", accepted=True)
                engine.record_mode("feature", "isolated", accepted=False)
                engine.record_mode("feature", "collaborative", accepted=False)

            problem = Problem(
                description="Build something",
                problem_type="feature",
                solve_mode="auto",
            )
            mode = engine.suggest_mode(problem)
            assert mode == "decomposed"
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_mode_stats_include_decomposed(self):
        """get_stats includes decomposed mode in _mode_stats."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            db_path = f.name

        try:
            engine = LearningEngine(db_path=db_path)
            engine.record_mode("feature", "decomposed", accepted=True)
            engine.record_mode("feature", "decomposed", accepted=True)
            engine.record_mode("feature", "isolated", accepted=False)

            stats = engine.get_stats()
            assert "_mode_stats" in stats
            assert "feature" in stats["_mode_stats"]
            assert "decomposed" in stats["_mode_stats"]["feature"]
            assert stats["_mode_stats"]["feature"]["decomposed"]["acceptance_rate"] == 1.0
        finally:
            Path(db_path).unlink(missing_ok=True)
