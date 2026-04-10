"""Priority computation for API requests.

Lower priority number = higher priority. Based on expected response length
so that short, latency-sensitive requests are served first.
"""

from __future__ import annotations

from fusen_solver.core.interfaces import Problem


# Strategy names that typically produce short responses
_SHORT_STRATEGIES = frozenset({"review", "analyst"})
# Strategy names that typically produce medium responses
_MEDIUM_STRATEGIES = frozenset({"direct", "test_first"})
# Strategy names that typically produce long responses
_LONG_STRATEGIES = frozenset({"rewrite", "decompose"})


def compute_priority(problem: Problem, strategy: str) -> int:
    """Compute request priority based on expected response length.

    Args:
        problem: The problem being solved.
        strategy: The strategy name being used.

    Returns:
        Priority level (1 = highest, 3 = lowest).
    """
    # Check constraints for explicit short-response hints
    if problem.constraints and any("short" in c for c in problem.constraints):
        return 1

    if strategy in _SHORT_STRATEGIES:
        return 1
    if strategy in _MEDIUM_STRATEGIES:
        return 2
    if strategy in _LONG_STRATEGIES:
        return 3

    return 2  # default: medium priority
