"""Tests for IncrementalContext and TestSandbox.

All tests run without Docker — the sandbox is exercised through its host
fallback path and via direct mocking of the Docker subprocess call.

Run with:
    cd fusen_solver && python -m pytest tests/test_incremental_and_sandbox.py -v
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from fusen_solver.core.incremental_context import IncrementalContext
from fusen_solver.scoring.sandbox import TestSandbox, _docker_available


# ===========================================================================
# IncrementalContext tests
# ===========================================================================


class TestIncrementalContext:
    """Tests for IncrementalContext."""

    # ------------------------------------------------------------------
    # compute_diff
    # ------------------------------------------------------------------

    def test_all_new_on_first_call(self):
        ctx = IncrementalContext()
        codebase = {"a.py": "x = 1", "b.py": "y = 2"}
        diff = ctx.compute_diff(codebase)
        assert set(diff["added"]) == {"a.py", "b.py"}
        assert diff["modified"] == {}
        assert diff["removed"] == []

    def test_no_changes_returns_empty_diff(self):
        ctx = IncrementalContext()
        codebase = {"a.py": "x = 1"}
        ctx.compute_diff(codebase)  # first call — populates hashes
        diff = ctx.compute_diff(codebase)  # second call — same content
        assert diff["added"] == {}
        assert diff["modified"] == {}
        assert diff["removed"] == []

    def test_detects_modification(self):
        ctx = IncrementalContext()
        codebase = {"a.py": "x = 1"}
        ctx.compute_diff(codebase)
        codebase["a.py"] = "x = 2"  # mutate
        diff = ctx.compute_diff(codebase)
        assert "a.py" in diff["modified"]
        assert diff["added"] == {}
        assert diff["removed"] == []

    def test_detects_added_file(self):
        ctx = IncrementalContext()
        ctx.compute_diff({"a.py": "x = 1"})
        diff = ctx.compute_diff({"a.py": "x = 1", "b.py": "y = 2"})
        assert "b.py" in diff["added"]

    def test_detects_removed_file(self):
        ctx = IncrementalContext()
        ctx.compute_diff({"a.py": "x = 1", "b.py": "y = 2"})
        diff = ctx.compute_diff({"a.py": "x = 1"})
        assert "b.py" in diff["removed"]

    def test_hash_table_updated_after_diff(self):
        """After compute_diff, the internal state reflects the new snapshot."""
        ctx = IncrementalContext()
        ctx.compute_diff({"a.py": "v1"})
        ctx.compute_diff({"a.py": "v2"})
        # Now v2 is the baseline — another call with same content => no changes.
        diff = ctx.compute_diff({"a.py": "v2"})
        assert diff["modified"] == {}

    # ------------------------------------------------------------------
    # has_changes
    # ------------------------------------------------------------------

    def test_has_changes_false_when_same(self):
        ctx = IncrementalContext()
        codebase = {"a.py": "x = 1"}
        ctx.compute_diff(codebase)  # seed hashes
        assert ctx.has_changes(codebase) is False

    def test_has_changes_true_on_modification(self):
        ctx = IncrementalContext()
        ctx.compute_diff({"a.py": "x = 1"})
        assert ctx.has_changes({"a.py": "x = 99"}) is True

    def test_has_changes_true_on_new_file(self):
        ctx = IncrementalContext()
        ctx.compute_diff({"a.py": "x = 1"})
        assert ctx.has_changes({"a.py": "x = 1", "b.py": "y = 2"}) is True

    def test_has_changes_true_on_removal(self):
        ctx = IncrementalContext()
        ctx.compute_diff({"a.py": "x = 1", "b.py": "y = 2"})
        assert ctx.has_changes({"a.py": "x = 1"}) is True

    def test_has_changes_does_not_mutate_hashes(self):
        """has_changes must be side-effect-free."""
        ctx = IncrementalContext()
        ctx.compute_diff({"a.py": "x = 1"})
        ctx.has_changes({"a.py": "x = 99"})  # check — must not update hashes
        # After the check, hashes still reflect the ORIGINAL snapshot.
        diff = ctx.compute_diff({"a.py": "x = 1"})
        assert diff["modified"] == {}  # still no change from the original

    # ------------------------------------------------------------------
    # build_prefix
    # ------------------------------------------------------------------

    def test_first_build_returns_full_context(self):
        ctx = IncrementalContext()
        codebase = {"a.py": "x = 1", "b.py": "y = 2"}
        prefix = ctx.build_prefix(codebase)
        assert "a.py" in prefix
        assert "b.py" in prefix
        assert "x = 1" in prefix
        assert "y = 2" in prefix

    def test_unchanged_codebase_returns_same_prefix(self):
        ctx = IncrementalContext()
        codebase = {"a.py": "x = 1"}
        prefix1 = ctx.build_prefix(codebase)
        prefix2 = ctx.build_prefix(codebase)
        assert prefix1 == prefix2

    def test_changed_file_updates_prefix(self):
        ctx = IncrementalContext()
        prefix1 = ctx.build_prefix({"a.py": "x = 1"})
        prefix2 = ctx.build_prefix({"a.py": "x = 99"})
        assert "x = 99" in prefix2
        assert prefix1 != prefix2

    def test_force_full_rebuilds_even_when_unchanged(self):
        ctx = IncrementalContext()
        codebase = {"a.py": "x = 1"}
        prefix1 = ctx.build_prefix(codebase)
        prefix2 = ctx.build_prefix(codebase, force_full=True)
        # Content should be identical even though we forced a rebuild.
        assert prefix1 == prefix2

    def test_files_sorted_deterministically(self):
        """Prefix file order must be alphabetical for reproducible cache hits."""
        ctx = IncrementalContext()
        codebase = {"z.py": "z", "a.py": "a", "m.py": "m"}
        prefix = ctx.build_prefix(codebase)
        idx_a = prefix.index("a.py")
        idx_m = prefix.index("m.py")
        idx_z = prefix.index("z.py")
        assert idx_a < idx_m < idx_z

    def test_empty_codebase(self):
        ctx = IncrementalContext()
        assert ctx.build_prefix({}) == ""

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def test_reset_clears_state(self):
        ctx = IncrementalContext()
        ctx.build_prefix({"a.py": "x = 1"})
        ctx.reset()
        # After reset, everything looks new again.
        diff = ctx.compute_diff({"a.py": "x = 1"})
        assert "a.py" in diff["added"]

    def test_reset_clears_cached_prefix(self):
        ctx = IncrementalContext()
        ctx.build_prefix({"a.py": "x = 1"})
        ctx.reset()
        # _cached_prefix should be empty string after reset.
        assert ctx._cached_prefix == ""


# ===========================================================================
# TestSandbox tests
# ===========================================================================


class TestSandboxFallback:
    """Test TestSandbox's host-fallback path (Docker mocked as unavailable)."""

    @pytest.fixture()
    def sandbox_no_docker(self):
        """Return a TestSandbox that believes Docker is unavailable."""
        sb = TestSandbox(timeout=10)
        sb._docker_ok = False  # skip Docker availability probe
        return sb

    def test_passing_command(self, sandbox_no_docker):
        result = sandbox_no_docker.run_tests(
            code={"hello.py": 'print("hello")'},
            test_commands=["python hello.py"],
        )
        assert result["total"] == 1
        assert result["passed"] == 1
        assert result["sandboxed"] is False
        assert result["results"][0]["passed"] is True

    def test_failing_command(self, sandbox_no_docker):
        result = sandbox_no_docker.run_tests(
            code={"bad.py": "raise RuntimeError('boom')"},
            test_commands=["python bad.py"],
        )
        assert result["passed"] == 0
        assert result["results"][0]["passed"] is False

    def test_timeout_reported(self):
        sb = TestSandbox(timeout=1)
        sb._docker_ok = False
        result = sb.run_tests(
            code={},
            test_commands=["sleep 5"],
        )
        assert result["passed"] == 0
        assert result["results"][0].get("error") == "timeout"

    def test_multiple_commands_independent(self, sandbox_no_docker):
        """A failing command must not prevent subsequent commands from running."""
        result = sandbox_no_docker.run_tests(
            code={"ok.py": "x = 1"},
            test_commands=[
                "python -c 'raise SystemExit(1)'",
                "python ok.py",
            ],
        )
        assert result["total"] == 2
        assert result["passed"] == 1
        statuses = [r["passed"] for r in result["results"]]
        assert statuses == [False, True]

    def test_empty_commands_list(self, sandbox_no_docker):
        result = sandbox_no_docker.run_tests(code={}, test_commands=[])
        assert result["total"] == 0
        assert result["passed"] == 0
        assert result["results"] == []

    def test_code_files_written_to_tmpdir(self, sandbox_no_docker):
        """The code files must be visible to the test commands."""
        result = sandbox_no_docker.run_tests(
            code={"data.txt": "hello"},
            test_commands=["cat data.txt"],
        )
        assert result["passed"] == 1

    def test_stdout_captured_and_truncated(self, sandbox_no_docker):
        long_output = "x" * 2000
        result = sandbox_no_docker.run_tests(
            code={},
            test_commands=[f"python -c \"print('{'x' * 2000}')\""],
        )
        stdout = result["results"][0].get("stdout", "")
        assert len(stdout) <= 1000

    def test_pass_rate_returned_correctly(self, sandbox_no_docker):
        result = sandbox_no_docker.run_tests(
            code={},
            test_commands=[
                "true",
                "true",
                "false",
            ],
        )
        assert result["total"] == 3
        assert result["passed"] == 2


class TestSandboxDocker:
    """Test TestSandbox when Docker subprocess is mocked to succeed."""

    @pytest.fixture()
    def sandbox_with_docker(self):
        sb = TestSandbox(image="python:3.12-slim", timeout=10)
        sb._docker_ok = True
        return sb

    def _make_docker_result(self, returncode: int, stdout: str = "", stderr: str = ""):
        mock = MagicMock()
        mock.returncode = returncode
        mock.stdout = stdout
        mock.stderr = stderr
        return mock

    def test_docker_command_structure(self, sandbox_with_docker):
        """Docker run is called with the expected flags."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_docker_result(0, stdout="ok")
            result = sandbox_with_docker.run_tests(
                code={"f.py": "x=1"},
                test_commands=["python f.py"],
            )

        assert mock_run.called
        call_args = mock_run.call_args[0][0]  # positional list arg
        assert call_args[0] == "docker"
        assert "run" in call_args
        assert "--network=none" in call_args
        assert "--memory=256m" in call_args
        assert result["passed"] == 1
        assert result["sandboxed"] is True

    def test_docker_failure_counted(self, sandbox_with_docker):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_docker_result(1, stderr="error!")
            result = sandbox_with_docker.run_tests(
                code={},
                test_commands=["bad_cmd"],
            )
        assert result["passed"] == 0
        assert result["results"][0]["passed"] is False

    def test_docker_timeout_handled(self, sandbox_with_docker):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("docker", 10)):
            result = sandbox_with_docker.run_tests(
                code={},
                test_commands=["sleep 999"],
            )
        assert result["passed"] == 0
        assert result["results"][0]["error"] == "timeout"

    def test_docker_passes_image(self, sandbox_with_docker):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = self._make_docker_result(0)
            sandbox_with_docker.run_tests(code={}, test_commands=["true"])
        call_args = mock_run.call_args[0][0]
        assert "python:3.12-slim" in call_args


# ===========================================================================
# _docker_available helper
# ===========================================================================


class TestDockerAvailable:
    def test_returns_false_when_no_docker_binary(self):
        with patch("shutil.which", return_value=None):
            assert _docker_available() is False

    def test_returns_false_when_daemon_unreachable(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                assert _docker_available() is False

    def test_returns_true_when_docker_ok(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                assert _docker_available() is True

    def test_returns_false_on_exception(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run", side_effect=Exception("boom")):
                assert _docker_available() is False


# ===========================================================================
# ScoringEngine integration: _run_tests delegates to sandbox
# ===========================================================================


class TestScoringEngineUsagesSandbox:
    """Verify that ScoringEngine._run_tests uses the sandbox module."""

    def test_run_tests_uses_sandbox(self):
        from fusen_solver.core.interfaces import Problem, Solution
        from fusen_solver.scoring.engine import ScoringEngine

        problem = Problem(
            description="test",
            context={"f.py": "x=1"},
            tests=["python f.py"],
        )
        solution = Solution(code={"f.py": "x = 1\n"})

        # Patch _get_sandbox to return a sandbox that always reports 1/1 pass.
        mock_sandbox = MagicMock()
        mock_sandbox.run_tests.return_value = {
            "total": 1,
            "passed": 1,
            "sandboxed": False,
            "results": [{"command": "python f.py", "passed": True, "stdout": "", "stderr": ""}],
        }

        import fusen_solver.scoring.engine as engine_mod
        original = engine_mod._get_sandbox
        engine_mod._get_sandbox = lambda: mock_sandbox

        try:
            score = ScoringEngine._run_tests(problem, solution)
        finally:
            engine_mod._get_sandbox = original

        assert score == 1.0
        mock_sandbox.run_tests.assert_called_once_with(solution.code, problem.tests)
