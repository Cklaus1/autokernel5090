"""Docker-based sandboxed test execution.

Provides ``TestSandbox``, which runs test commands inside a throw-away Docker
container with no network access and a capped memory budget.  Falls back
gracefully when Docker is unavailable.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _docker_available() -> bool:
    """Return True if the ``docker`` CLI is present and the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


class TestSandbox:
    """Run tests in a Docker container for safety.

    Each ``run_tests()`` call:

    1. Writes all ``code`` files into a temporary directory on the host.
    2. Mounts that directory read-only into a fresh container.
    3. Executes each ``test_commands`` entry via ``sh -c`` inside the container.
    4. Returns a structured result dict with per-command pass/fail info.

    If Docker is unavailable the sandbox falls back to running commands on the
    host in a temporary directory (same behaviour as the legacy
    ``ScoringEngine._run_tests``).  Callers can detect this via the
    ``"sandboxed"`` key in the returned dict.

    Args:
        image:   Docker image to use.  Must have Python available if the tests
                 are Python-based.  Defaults to ``"python:3.12-slim"``.
        timeout: Per-command wall-clock timeout in seconds.
    """

    def __init__(self, image: str = "python:3.12-slim", timeout: int = 30) -> None:
        self.image = image
        self.timeout = timeout
        self._docker_ok: bool | None = None  # lazily checked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_tests(self, code: dict[str, str], test_commands: list[str]) -> dict:
        """Run *test_commands* against *code* and return structured results.

        Args:
            code:          Mapping of relative file path to file content.
                           All files are written into a shared working directory.
            test_commands: Shell commands to execute.  Each is run independently;
                           failures do not stop subsequent commands.

        Returns:
            A dict with::

                {
                    "total":     int,          # number of commands
                    "passed":    int,          # commands with returncode == 0
                    "sandboxed": bool,         # True if Docker was used
                    "results": [
                        {
                            "command": str,
                            "passed":  bool,
                            "stdout":  str,    # first 1 000 chars
                            "stderr":  str,    # first 1 000 chars
                        },
                        # or on timeout/error:
                        {
                            "command": str,
                            "passed":  False,
                            "error":   str,
                        },
                    ],
                }
        """
        if self._docker_ok is None:
            self._docker_ok = _docker_available()

        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_code(code, tmpdir)

            if self._docker_ok:
                results = self._run_in_docker(tmpdir, test_commands)
                sandboxed = True
            else:
                logger.warning(
                    "Docker not available — running tests on host (unsandboxed)"
                )
                results = self._run_on_host(tmpdir, test_commands)
                sandboxed = False

        passed = sum(1 for r in results if r.get("passed", False))
        return {
            "total": len(results),
            "passed": passed,
            "sandboxed": sandboxed,
            "results": results,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_code(code: dict[str, str], tmpdir: str) -> None:
        """Write all code files into *tmpdir*, creating subdirectories as needed."""
        for path, content in code.items():
            full_path = os.path.join(tmpdir, path)
            os.makedirs(os.path.dirname(full_path) or tmpdir, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(content)

    def _run_in_docker(self, tmpdir: str, test_commands: list[str]) -> list[dict]:
        """Execute each command inside a Docker container."""
        results: list[dict] = []
        for cmd in test_commands:
            try:
                proc = subprocess.run(
                    [
                        "docker", "run", "--rm",
                        "--network=none",   # no outbound network access
                        "--memory=256m",    # cap memory
                        "--cpus=1",         # avoid monopolising the host
                        f"-v{tmpdir}:/code:ro",
                        "-w", "/code",
                        self.image,
                        "sh", "-c", cmd,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                results.append({
                    "command": cmd,
                    "passed": proc.returncode == 0,
                    "stdout": proc.stdout[:1000],
                    "stderr": proc.stderr[:1000],
                })
            except subprocess.TimeoutExpired:
                results.append({"command": cmd, "passed": False, "error": "timeout"})
            except Exception as exc:
                results.append({"command": cmd, "passed": False, "error": str(exc)})
        return results

    def _run_on_host(self, tmpdir: str, test_commands: list[str]) -> list[dict]:
        """Execute each command on the host in *tmpdir* (fallback path)."""
        results: list[dict] = []
        for cmd in test_commands:
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                )
                results.append({
                    "command": cmd,
                    "passed": proc.returncode == 0,
                    "stdout": proc.stdout[:1000],
                    "stderr": proc.stderr[:1000],
                })
            except subprocess.TimeoutExpired:
                results.append({"command": cmd, "passed": False, "error": "timeout"})
            except Exception as exc:
                results.append({"command": cmd, "passed": False, "error": str(exc)})
        return results
