"""Shadow mode: run fusen_solver in parallel with BCode, compare, log.

Zero impact on BCode -- the shadow runs async, never blocks, never
modifies BCode's output. Results are logged to a JSONL file for
data-driven promotion decisions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fusen_solver.config import Config, load_config
from fusen_solver.core.interfaces import Problem, Solution
from fusen_solver.core.solver import FusenSolver, SolveResult
from fusen_solver.integrations.bcode_bridge import BCodeBridge

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = "~/.fusen_solver/shadow_log.jsonl"
_DEFAULT_TIMEOUT_S = 120


@dataclass
class ShadowResult:
    """Outcome of a single shadow comparison."""

    task: str
    bcode_score: float  # from BCode's own scoring (0-1)
    fusen_score: float  # from fusen_solver's scoring (0-1)
    bcode_time_s: float
    fusen_time_s: float
    bcode_files: int  # number of files generated
    fusen_files: int
    fusen_mode_used: str  # isolated/collaborative/decomposed/racing
    fusen_strategy_used: str  # winning strategy name
    winner: str  # "bcode", "fusen", "tie"
    notes: str = ""
    timestamp: str = ""


class BCodeShadow:
    """Run fusen_solver as a shadow alongside BCode.

    Usage:
        shadow = BCodeShadow()
        result = await shadow.shadow_run(task_prd, workspace)
        # result is logged automatically; BCode is never blocked
    """

    def __init__(
        self,
        config: Config | None = None,
        log_path: str = _DEFAULT_LOG_PATH,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ):
        self._config = config or load_config()
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self._bridge = BCodeBridge()
        self._latest_result: ShadowResult | None = None
        self._latest_solve_result: SolveResult | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def shadow_run(
        self,
        task: str,
        codebase_path: str,
        bcode_output: dict[str, Any] | None = None,
        bcode_time_s: float = 0.0,
    ) -> ShadowResult:
        """Run fusen_solver on the same task BCode is processing.

        Args:
            task: The PRD / task description (same input BCode receives).
            codebase_path: Path to the workspace/codebase directory.
            bcode_output: Optional BCode result dict for comparison.
                If not provided, only fusen_solver metrics are recorded.
            bcode_time_s: Wall-clock time BCode took (if known).

        Returns:
            A ShadowResult with comparison metrics, also logged to JSONL.
        """
        problem = self._bridge.prd_to_problem(task, codebase_path)

        # Run fusen_solver with timeout
        fusen_start = time.monotonic()
        solve_result: SolveResult | None = None

        try:
            solver = self._make_solver()
            solve_result = await asyncio.wait_for(
                solver.solve(problem),
                timeout=self.timeout_s,
            )
            self._latest_solve_result = solve_result
        except asyncio.TimeoutError:
            logger.warning("Shadow run timed out after %.0fs", self.timeout_s)
        except Exception:
            logger.exception("Shadow run failed")

        fusen_time = time.monotonic() - fusen_start

        # Build comparison
        result = self._build_result(
            task=task,
            solve_result=solve_result,
            fusen_time_s=fusen_time,
            bcode_output=bcode_output,
            bcode_time_s=bcode_time_s,
        )
        self._latest_result = result
        self._log(result)

        return result

    def compare(
        self,
        bcode_output: dict[str, Any],
        fusen_output: dict[str, Any] | None = None,
        task: str = "",
    ) -> ShadowResult:
        """Compare BCode and fusen_solver outputs after both have finished.

        Can use either a raw fusen_output dict or the last shadow_run's
        SolveResult stored internally.
        """
        bcode_sol = self._bridge.bcode_output_to_solution(bcode_output)

        fusen_sol: Solution | None = None
        fusen_mode = "unknown"
        fusen_strategy = "unknown"

        if fusen_output is not None:
            fusen_sol = self._bridge.bcode_output_to_solution(fusen_output)
            fusen_mode = fusen_output.get("mode", "unknown")
            fusen_strategy = fusen_output.get("strategy", "unknown")
        elif self._latest_solve_result and self._latest_solve_result.best:
            fusen_sol = self._latest_solve_result.best
            fusen_mode = self._latest_solve_result.mode
            fusen_strategy = fusen_sol.strategy_used

        fusen_score = fusen_sol.score if fusen_sol else 0.0
        fusen_files = len(fusen_sol.code) if fusen_sol else 0

        winner = self._determine_winner(bcode_sol.score, fusen_score)

        return ShadowResult(
            task=task,
            bcode_score=bcode_sol.score,
            fusen_score=fusen_score,
            bcode_time_s=0.0,
            fusen_time_s=0.0,
            bcode_files=len(bcode_sol.code),
            fusen_files=fusen_files,
            fusen_mode_used=fusen_mode,
            fusen_strategy_used=fusen_strategy,
            winner=winner,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def get_stats(self) -> dict[str, Any]:
        """Aggregate shadow results: win rates, average scores, trends."""
        entries = self._read_log()
        if not entries:
            return {
                "total_runs": 0,
                "fusen_win_rate": 0.0,
                "bcode_win_rate": 0.0,
                "tie_rate": 0.0,
                "avg_fusen_score": 0.0,
                "avg_bcode_score": 0.0,
                "avg_fusen_time_s": 0.0,
                "avg_bcode_time_s": 0.0,
            }

        total = len(entries)
        fusen_wins = sum(1 for e in entries if e.get("winner") == "fusen")
        bcode_wins = sum(1 for e in entries if e.get("winner") == "bcode")
        ties = sum(1 for e in entries if e.get("winner") == "tie")

        def _avg(key: str) -> float:
            vals = [e.get(key, 0.0) for e in entries]
            return sum(vals) / len(vals) if vals else 0.0

        # Trend: last 10 vs first 10
        trend_note = ""
        if total >= 20:
            first_10 = entries[:10]
            last_10 = entries[-10:]
            early_rate = sum(1 for e in first_10 if e.get("winner") == "fusen") / 10
            recent_rate = sum(1 for e in last_10 if e.get("winner") == "fusen") / 10
            if recent_rate > early_rate + 0.1:
                trend_note = "improving"
            elif recent_rate < early_rate - 0.1:
                trend_note = "declining"
            else:
                trend_note = "stable"

        return {
            "total_runs": total,
            "fusen_win_rate": fusen_wins / total,
            "bcode_win_rate": bcode_wins / total,
            "tie_rate": ties / total,
            "avg_fusen_score": _avg("fusen_score"),
            "avg_bcode_score": _avg("bcode_score"),
            "avg_fusen_time_s": _avg("fusen_time_s"),
            "avg_bcode_time_s": _avg("bcode_time_s"),
            "trend": trend_note,
        }

    def should_promote(
        self,
        min_runs: int = 50,
        min_win_rate: float = 0.6,
    ) -> bool:
        """Data-driven check: should fusen_solver be promoted from shadow?

        Returns True only when there is enough data AND fusen_solver
        consistently outperforms BCode.
        """
        stats = self.get_stats()
        return (
            stats["total_runs"] >= min_runs
            and stats["fusen_win_rate"] >= min_win_rate
        )

    @property
    def latest_result(self) -> ShadowResult | None:
        """Most recent shadow run result (for use by hooks)."""
        return self._latest_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_solver(self) -> FusenSolver:
        """Create a FusenSolver from the stored config."""
        from fusen_solver.integrations.cli import _make_backend

        backend = _make_backend(self._config)
        return FusenSolver(backend=backend)

    def _build_result(
        self,
        task: str,
        solve_result: SolveResult | None,
        fusen_time_s: float,
        bcode_output: dict[str, Any] | None,
        bcode_time_s: float,
    ) -> ShadowResult:
        """Assemble a ShadowResult from solve output and optional BCode output."""
        fusen_score = 0.0
        fusen_files = 0
        fusen_mode = "unknown"
        fusen_strategy = "unknown"

        if solve_result and solve_result.best:
            fusen_score = solve_result.best.score
            fusen_files = len(solve_result.best.code)
            fusen_mode = solve_result.mode
            fusen_strategy = solve_result.best.strategy_used

        bcode_score = 0.0
        bcode_files = 0
        if bcode_output:
            bcode_sol = self._bridge.bcode_output_to_solution(bcode_output)
            bcode_score = bcode_sol.score
            bcode_files = len(bcode_sol.code)

        winner = self._determine_winner(bcode_score, fusen_score)

        notes = ""
        if solve_result is None:
            notes = "fusen_solver did not produce a result"

        return ShadowResult(
            task=task[:500],  # truncate very long PRDs
            bcode_score=bcode_score,
            fusen_score=fusen_score,
            bcode_time_s=bcode_time_s,
            fusen_time_s=fusen_time_s,
            bcode_files=bcode_files,
            fusen_files=fusen_files,
            fusen_mode_used=fusen_mode,
            fusen_strategy_used=fusen_strategy,
            winner=winner,
            notes=notes,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _determine_winner(bcode_score: float, fusen_score: float) -> str:
        """Determine winner with a 0.02 tolerance band for ties."""
        diff = fusen_score - bcode_score
        if abs(diff) < 0.02:
            return "tie"
        return "fusen" if diff > 0 else "bcode"

    def _log(self, result: ShadowResult) -> None:
        """Append a ShadowResult to the JSONL log file."""
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(result)) + "\n")
        except OSError as exc:
            logger.warning("Failed to write shadow log: %s", exc)

    def _read_log(self) -> list[dict[str, Any]]:
        """Read all entries from the JSONL log file."""
        if not self.log_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        try:
            with open(self.log_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as exc:
            logger.warning("Failed to read shadow log: %s", exc)

        return entries
