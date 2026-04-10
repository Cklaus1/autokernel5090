"""Strategy selection engine and built-in presets."""

from fusen_solver.strategies.engine import StrategyEngine
from fusen_solver.strategies.presets import STRATEGY_PRESETS, STRATEGY_CATALOG, get_strategy

__all__ = ["StrategyEngine", "STRATEGY_PRESETS", "STRATEGY_CATALOG", "get_strategy"]
