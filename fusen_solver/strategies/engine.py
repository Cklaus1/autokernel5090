"""Data-driven strategy selection engine.

Selects N strategies based on problem type and historical success rates
from the learning engine. Adapts over time as more feedback is collected.
"""

from __future__ import annotations

import random
from fusen_solver.core.interfaces import Problem, Strategy
from fusen_solver.strategies.presets import STRATEGY_PRESETS, STRATEGY_CATALOG, get_strategy


class StrategyEngine:
    """Selects and configures strategies for a given problem."""

    def select_strategies(
        self,
        problem: Problem,
        *,
        n: int = 4,
        weights: dict[str, float] | None = None,
    ) -> list[Strategy]:
        """Select N strategies based on problem type and historical success rates.

        Args:
            problem: The problem to select strategies for.
            n: Number of strategies to select.
            weights: Optional strategy -> weight mapping from the learning engine.

        Returns:
            List of Strategy objects, length <= n.
        """
        # 1. Get candidate strategies for this problem type
        candidates = self._get_candidates(problem)

        # 2. Apply learned weights if available
        if weights:
            for s in candidates:
                if s.name in weights:
                    s.weight = weights[s.name]

        # 3. Select top-N by weighted sampling (no replacement)
        selected = self._weighted_select(candidates, n)

        return selected

    def generate_prompts(
        self,
        problem: Problem,
        strategies: list[Strategy],
    ) -> list[str]:
        """Generate the strategy-specific prompt for each strategy.

        All prompts share the same codebase context (for prefix caching).
        Only the strategy instructions differ.
        """
        return [s.prompt for s in strategies]

    def _get_candidates(self, problem: Problem) -> list[Strategy]:
        """Get candidate strategies for a problem type."""
        problem_type = problem.problem_type

        if problem_type == "auto":
            # Use a diverse default set
            problem_type = self._infer_type(problem)

        preset_names = STRATEGY_PRESETS.get(problem_type, STRATEGY_PRESETS.get("bug_fix", []))

        candidates = []
        for name in preset_names:
            strategy = get_strategy(name)
            candidates.append(Strategy(
                name=strategy.name,
                prompt=strategy.prompt,
                weight=strategy.weight,
                temperature=strategy.temperature,
                tags=list(strategy.tags),
            ))

        return candidates

    @staticmethod
    def _infer_type(problem: Problem) -> str:
        """Infer problem type from description keywords."""
        desc = problem.description.lower()

        type_keywords = {
            "bug_fix": ["fix", "bug", "error", "crash", "broken", "wrong", "fail", "issue"],
            "feature": ["add", "implement", "create", "new", "feature", "build"],
            "refactor": ["refactor", "clean", "simplify", "restructure", "rename"],
            "optimize": ["optimize", "speed", "fast", "slow", "performance", "latency"],
            "test": ["test", "coverage", "spec", "assert"],
            "review": ["review", "audit", "security", "check"],
            "architecture": ["architecture", "design", "system", "module", "interface"],
        }

        best_type = "bug_fix"
        best_score = 0

        for ptype, keywords in type_keywords.items():
            score = sum(1 for kw in keywords if kw in desc)
            if score > best_score:
                best_score = score
                best_type = ptype

        return best_type

    @staticmethod
    def _weighted_select(candidates: list[Strategy], n: int) -> list[Strategy]:
        """Select N strategies using weighted sampling without replacement."""
        if len(candidates) <= n:
            return candidates

        selected: list[Strategy] = []
        remaining = list(candidates)

        for _ in range(n):
            if not remaining:
                break

            total_weight = sum(s.weight for s in remaining)
            if total_weight <= 0:
                # Equal weights fallback
                choice = random.choice(remaining)
            else:
                r = random.uniform(0, total_weight)
                cumulative = 0.0
                choice = remaining[-1]
                for s in remaining:
                    cumulative += s.weight
                    if cumulative >= r:
                        choice = s
                        break

            selected.append(choice)
            remaining.remove(choice)

        return selected
