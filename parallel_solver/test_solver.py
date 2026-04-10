"""Tests for the Parallel Problem-Solving system.

Tests run without a live vLLM instance by mocking HTTP calls.
Run with: python3 -m pytest parallel_solver/test_solver.py -v
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parallel_solver.prefix_manager import PrefixManager, STRATEGY_PROMPTS
from parallel_solver.solution_scorer import SolutionScorer, ScoredSolution
from parallel_solver.streaming import ParallelStreamer, AgentResult, StreamEvent
from parallel_solver.orchestrator import ProblemOrchestrator, STRATEGY_PRESETS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_API = "http://localhost:8000"
MOCK_MODEL = "test-model"
SAMPLE_CODEBASE = "def add(a, b):\n    return a + b\n"
SAMPLE_PROBLEM = "The add function should handle None inputs gracefully."


# ---------------------------------------------------------------------------
# PrefixManager tests
# ---------------------------------------------------------------------------


class TestPrefixManager:
    def test_build_context_returns_two_messages(self):
        pm = PrefixManager(MOCK_API, MOCK_MODEL)
        messages = pm.build_context(SAMPLE_CODEBASE, SAMPLE_PROBLEM, "direct")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_contains_codebase(self):
        pm = PrefixManager(MOCK_API, MOCK_MODEL)
        messages = pm.build_context(SAMPLE_CODEBASE, SAMPLE_PROBLEM, "direct")
        assert SAMPLE_CODEBASE in messages[0]["content"]

    def test_user_message_contains_problem(self):
        pm = PrefixManager(MOCK_API, MOCK_MODEL)
        messages = pm.build_context(SAMPLE_CODEBASE, SAMPLE_PROBLEM, "direct")
        assert SAMPLE_PROBLEM in messages[1]["content"]

    def test_different_strategies_share_system_message(self):
        pm = PrefixManager(MOCK_API, MOCK_MODEL)
        ctx_a = pm.build_context(SAMPLE_CODEBASE, SAMPLE_PROBLEM, "direct")
        ctx_b = pm.build_context(SAMPLE_CODEBASE, SAMPLE_PROBLEM, "review")
        # System messages (prefix) must be identical for cache hits
        assert ctx_a[0]["content"] == ctx_b[0]["content"]
        # User messages should differ
        assert ctx_a[1]["content"] != ctx_b[1]["content"]

    def test_all_strategies_have_prompts(self):
        for name in STRATEGY_PROMPTS:
            assert len(STRATEGY_PROMPTS[name]) > 20, f"Strategy '{name}' has empty prompt"

    def test_build_context_with_extra_context(self):
        pm = PrefixManager(MOCK_API, MOCK_MODEL)
        messages = pm.build_context(
            SAMPLE_CODEBASE, SAMPLE_PROBLEM, "direct", extra_context="See also: utils.py"
        )
        assert "See also: utils.py" in messages[1]["content"]

    def test_build_merge_context(self):
        pm = PrefixManager(MOCK_API, MOCK_MODEL)
        solutions = [
            {"strategy": "direct", "content": "fix 1", "score": 0.9},
            {"strategy": "review", "content": "fix 2", "score": 0.7},
        ]
        messages = pm.build_merge_context(SAMPLE_CODEBASE, SAMPLE_PROBLEM, solutions)
        assert len(messages) == 2
        assert "fix 1" in messages[1]["content"]
        assert "fix 2" in messages[1]["content"]

    def test_load_codebase_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("hello = 'world'\n")
            f.flush()
            result = PrefixManager.load_codebase(f.name)
            assert "hello" in result
            Path(f.name).unlink()

    def test_load_codebase_from_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "main.py").write_text("x = 1\n")
            (Path(d) / "util.py").write_text("y = 2\n")
            result = PrefixManager.load_codebase(d)
            assert "x = 1" in result
            assert "y = 2" in result

    def test_load_codebase_respects_token_limit(self):
        with tempfile.TemporaryDirectory() as d:
            # Write a large file
            (Path(d) / "big.py").write_text("x = 1\n" * 100000)
            result = PrefixManager.load_codebase(d, max_tokens_approx=100)
            # Should be truncated (100 tokens ~ 400 chars)
            assert len(result) < 1000


# ---------------------------------------------------------------------------
# SolutionScorer tests
# ---------------------------------------------------------------------------


class TestSolutionScorer:
    def test_run_tests_passing(self):
        code = "def add(a, b): return a + b\n"
        tests = ["assert add(1, 2) == 3\nassert add(0, 0) == 0\n"]
        passed, total = SolutionScorer._run_tests(code, tests)
        assert passed == 1
        assert total == 1

    def test_run_tests_failing(self):
        code = "def add(a, b): return a - b\n"  # bug
        tests = ["assert add(1, 2) == 3\n"]
        passed, total = SolutionScorer._run_tests(code, tests)
        assert passed == 0
        assert total == 1

    def test_run_tests_multiple(self):
        code = "def add(a, b): return a + b\n"
        tests = [
            "assert add(1, 2) == 3\n",
            "assert add(-1, 1) == 0\n",
            "assert add(0, 0) == 99\n",  # should fail
        ]
        passed, total = SolutionScorer._run_tests(code, tests)
        assert passed == 2
        assert total == 3

    def test_run_tests_timeout(self):
        code = "import time\ndef slow(): time.sleep(100)\n"
        tests = ["slow()\n"]
        passed, total = SolutionScorer._run_tests(code, tests)
        assert passed == 0
        assert total == 1

    def test_run_tests_syntax_error(self):
        code = "def add(a, b: return a + b\n"  # syntax error
        tests = ["add(1, 2)\n"]
        passed, total = SolutionScorer._run_tests(code, tests)
        assert passed == 0
        assert total == 1


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestParallelStreamer:
    def test_agent_result_defaults(self):
        r = AgentResult(agent_id=0, strategy="direct", content="hello")
        assert r.agent_id == 0
        assert r.finished is False
        assert r.error is None

    def test_stream_event_defaults(self):
        e = StreamEvent(agent_id=1, strategy="review", delta="x", cumulative="x")
        assert e.finished is False
        assert e.error is None


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestrator:
    def test_strategy_presets_exist(self):
        for name, strategies in STRATEGY_PRESETS.items():
            assert len(strategies) >= 2, f"Preset '{name}' needs at least 2 strategies"
            for s in strategies:
                assert s in STRATEGY_PROMPTS, f"Preset '{name}' has unknown strategy '{s}'"

    def test_strategy_presets_cover_all_use_cases(self):
        expected = {"bug_fix", "feature", "refactor", "architecture", "optimization", "explore"}
        assert set(STRATEGY_PRESETS.keys()) == expected

    def test_explore_preset_includes_all_strategies(self):
        assert set(STRATEGY_PRESETS["explore"]) == set(STRATEGY_PROMPTS.keys())


# ---------------------------------------------------------------------------
# Integration-style tests (with mocked HTTP)
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    """Tests that verify the orchestrator wiring without a live API."""

    @pytest.fixture
    def mock_chat_response(self):
        """Returns a mock aiohttp response for chat completions."""

        def make_response(content: str = "fixed code here", usage_tokens: int = 100):
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": usage_tokens, "completion_tokens": 50},
            }

        return make_response

    @pytest.mark.asyncio
    async def test_prefix_warm(self, mock_chat_response):
        pm = PrefixManager(MOCK_API, MOCK_MODEL)

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_chat_response("OK", 500))
        mock_resp.status = 200

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp)))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)
            stats = await pm.warm_prefix(SAMPLE_CODEBASE)

        assert stats.warm is True
        assert stats.prefix_tokens == 500
        assert len(stats.prefix_hash) == 16

    @pytest.mark.asyncio
    async def test_llm_scoring(self, mock_chat_response):
        scorer = SolutionScorer(MOCK_API, MOCK_MODEL)

        score_json = json.dumps({
            "completeness": 0.9,
            "correctness": 0.8,
            "code_quality": 0.7,
            "explanation": "Good fix but missing edge case",
        })

        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value=mock_chat_response(score_json))
        mock_resp.status = 200

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp)))

        with patch("aiohttp.ClientSession") as mock_cs:
            mock_cs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_cs.return_value.__aexit__ = AsyncMock(return_value=False)
            scores = await scorer._llm_score(SAMPLE_PROBLEM, "def add(a, b): return a + b")

        assert scores["completeness"] == 0.9
        assert scores["correctness"] == 0.8
        assert "edge case" in scores["explanation"]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
