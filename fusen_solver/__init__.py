"""Fusen Solver: universal parallel AI problem solving for any coding platform.

Usage:
    from fusen_solver import FusenSolver, Problem
    from fusen_solver.backends import VLLMBackend

    solver = FusenSolver(backend=VLLMBackend())
    result = await solver.solve(Problem(
        description="Fix the race condition",
        context={"server.py": open("server.py").read()},
    ))
    print(result.best.explanation)
"""

from fusen_solver.core.interfaces import LLMBackend, PlatformPlugin, Problem, Solution, Strategy
from fusen_solver.core.solver import FusenSolver, SolveResult

__version__ = "0.1.0"

__all__ = [
    "FusenSolver",
    "SolveResult",
    "LLMBackend",
    "PlatformPlugin",
    "Problem",
    "Solution",
    "Strategy",
]
