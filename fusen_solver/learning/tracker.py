"""Learning engine: tracks historical success rates and adapts.

Records which strategies win for which problem types, then uses Bayesian
updating to adjust strategy weights over time. More data = more confident
= stronger weighting toward what works.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fusen_solver.core.interfaces import Problem, Solution

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "~/.fusen_solver/history.json"
MIN_DATA_FOR_ADAPTATION = 10


@dataclass
class StrategyRecord:
    """Aggregated stats for a single strategy."""

    wins: int = 0
    attempts: int = 0

    @property
    def win_rate(self) -> float:
        if self.attempts == 0:
            return 0.5  # prior
        return self.wins / self.attempts

    @property
    def confidence(self) -> float:
        """How confident we are in the win rate (0 to 1)."""
        if self.attempts == 0:
            return 0.0
        # Simple sigmoid-like confidence based on number of attempts
        return 1.0 - (1.0 / (1.0 + self.attempts / MIN_DATA_FOR_ADAPTATION))


class LearningEngine:
    """Tracks strategy success rates and adapts selection weights.

    Persists history to a JSON file. Uses Bayesian-style updating:
    - Prior: uniform weights (all strategies equally likely)
    - Posterior: weighted by observed win rates
    - Confidence: grows with more data points
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, min_data: int = MIN_DATA_FOR_ADAPTATION):
        self.db_path = Path(db_path).expanduser()
        self.min_data = min_data
        # Nested: problem_type -> strategy_name -> StrategyRecord
        self._stats: dict[str, dict[str, StrategyRecord]] = defaultdict(
            lambda: defaultdict(StrategyRecord)
        )
        # Raw history for analysis
        self._history: list[dict[str, Any]] = []
        self._load()

    def record(
        self,
        problem: Problem,
        solutions: list[Solution],
        accepted_idx: int,
    ) -> None:
        """Record which strategy won for this problem.

        Args:
            problem: The problem that was solved.
            solutions: All solutions that were generated.
            accepted_idx: Index of the accepted solution in the list.
        """
        ptype = problem.problem_type if problem.problem_type != "auto" else "unknown"

        for i, sol in enumerate(solutions):
            record = self._stats[ptype][sol.strategy_used]
            record.attempts += 1
            if i == accepted_idx:
                record.wins += 1

        self._history.append({
            "problem_type": ptype,
            "language": problem.language,
            "strategies": [s.strategy_used for s in solutions],
            "scores": [s.score for s in solutions],
            "accepted": accepted_idx,
            "accepted_strategy": solutions[accepted_idx].strategy_used if accepted_idx < len(solutions) else None,
        })

        self._save()
        logger.info(
            "Recorded: type=%s, winner=%s (idx=%d of %d)",
            ptype,
            solutions[accepted_idx].strategy_used if accepted_idx < len(solutions) else "none",
            accepted_idx,
            len(solutions),
        )

    def get_weights(self, problem_type: str) -> dict[str, float]:
        """Return strategy weights based on historical success.

        Uses Bayesian updating: prior is uniform (1.0), posterior is
        a blend of prior and observed win rate, weighted by confidence.

        Returns:
            Mapping of strategy_name -> weight (higher = more likely to be selected).
        """
        if problem_type not in self._stats:
            return {}  # no data, use defaults

        type_stats = self._stats[problem_type]
        total_attempts = sum(r.attempts for r in type_stats.values())

        if total_attempts < self.min_data:
            return {}  # not enough data to adapt

        weights: dict[str, float] = {}
        for strategy_name, record in type_stats.items():
            prior = 1.0
            observed = record.win_rate
            confidence = record.confidence
            # Blend: (1 - confidence) * prior + confidence * observed
            weights[strategy_name] = (1.0 - confidence) * prior + confidence * observed

        return weights

    def suggest_n(self, problem: Problem) -> int:
        """Suggest number of parallel agents based on problem difficulty.

        Simple heuristic:
        - If we have historical data and acceptance rates are high, use fewer agents.
        - If acceptance rates are low (hard problems), use more agents.
        - Default: 4 agents.
        """
        ptype = problem.problem_type if problem.problem_type != "auto" else "unknown"

        if ptype not in self._stats:
            return 4  # default

        type_stats = self._stats[ptype]
        total_attempts = sum(r.attempts for r in type_stats.values())

        if total_attempts < self.min_data:
            return 4  # not enough data

        # Average win rate across strategies for this problem type
        avg_win_rate = sum(r.win_rate for r in type_stats.values()) / max(len(type_stats), 1)

        # High win rate = easy problems = fewer agents needed
        # Low win rate = hard problems = more agents
        if avg_win_rate > 0.7:
            return 2
        elif avg_win_rate > 0.4:
            return 4
        elif avg_win_rate > 0.2:
            return 6
        else:
            return 8

    def get_stats(self) -> dict[str, Any]:
        """Return a summary of learning stats for display."""
        summary: dict[str, Any] = {}
        for ptype, strategies in self._stats.items():
            summary[ptype] = {
                name: {
                    "wins": r.wins,
                    "attempts": r.attempts,
                    "win_rate": round(r.win_rate, 3),
                    "confidence": round(r.confidence, 3),
                }
                for name, r in sorted(strategies.items(), key=lambda x: x[1].win_rate, reverse=True)
            }
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load history from disk."""
        if not self.db_path.exists():
            return

        try:
            data = json.loads(self.db_path.read_text())
            self._history = data.get("history", [])

            # Rebuild stats from history
            for entry in self._history:
                ptype = entry.get("problem_type", "unknown")
                strategies = entry.get("strategies", [])
                accepted_idx = entry.get("accepted", -1)

                for i, strategy_name in enumerate(strategies):
                    record = self._stats[ptype][strategy_name]
                    record.attempts += 1
                    if i == accepted_idx:
                        record.wins += 1

            logger.info("Loaded %d history entries from %s", len(self._history), self.db_path)
        except Exception as e:
            logger.warning("Failed to load history: %s", e)

    def _save(self) -> None:
        """Save history to disk."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"history": self._history}
            self.db_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save history: %s", e)
