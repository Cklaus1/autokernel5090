"""REST API server for the Fusen parallel solver.

Exposes the solver as an HTTP service that any client can call.
Uses Python's built-in asyncio for minimal dependencies.

Usage:
    python -m fusen_solver.integrations.api --port 8080
    curl -X POST http://localhost:8080/solve -d '{"problem": "...", "context": {...}}'
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fusen_solver.config import load_config
from fusen_solver.core.interfaces import Problem
from fusen_solver.core.solver import FusenSolver

logger = logging.getLogger(__name__)

# We use aiohttp.web since aiohttp is already a dependency
try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore


class FusenAPI:
    """REST API wrapper around FusenSolver."""

    def __init__(self, solver: FusenSolver):
        self.solver = solver

    async def handle_solve(self, request: web.Request) -> web.Response:
        """POST /solve -- solve a coding problem."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        problem = Problem(
            description=body.get("problem", body.get("description", "")),
            context=body.get("context", {}),
            problem_type=body.get("problem_type", "auto"),
            constraints=body.get("constraints", []),
            tests=body.get("tests", []),
            language=body.get("language", "auto"),
            priority=body.get("priority", "quality"),
        )

        if not problem.description:
            return web.json_response({"error": "Missing 'problem' field"}, status=400)

        n = body.get("num_agents", body.get("n"))
        merge = body.get("merge", False)

        result = await self.solver.solve(problem, n=n, merge=merge)

        response: dict[str, Any] = {
            "strategies_used": result.strategies_used,
            "num_agents": result.num_agents,
            "total_time_s": round(result.total_time_s, 2),
            "solutions": [
                {
                    "strategy": s.strategy_used,
                    "score": round(s.score, 3),
                    "subscores": {k: round(v, 3) for k, v in s.subscores.items()},
                    "code": s.code,
                    "explanation": s.explanation,
                }
                for s in result.solutions
            ],
        }

        if result.best:
            response["best"] = {
                "strategy": result.best.strategy_used,
                "score": round(result.best.score, 3),
                "code": result.best.code,
                "explanation": result.best.explanation,
            }

        if result.merged:
            response["merged"] = {
                "code": result.merged.code,
                "explanation": result.merged.explanation,
            }

        return web.json_response(response)

    async def handle_feedback(self, request: web.Request) -> web.Response:
        """POST /feedback -- record solution acceptance/rejection."""
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Simplified feedback: just record problem type and winning strategy
        from fusen_solver.core.interfaces import Solution

        problem = Problem(
            description=body.get("problem", ""),
            problem_type=body.get("problem_type", "unknown"),
        )
        solutions = [
            Solution(strategy_used=s.get("strategy", "unknown"), score=s.get("score", 0))
            for s in body.get("solutions", [])
        ]
        accepted_idx = body.get("accepted_idx", 0)

        await self.solver.record_feedback(problem, solutions, accepted_idx)

        return web.json_response({"status": "recorded"})

    async def handle_health(self, request: web.Request) -> web.Response:
        """GET /health -- health check."""
        return web.json_response({
            "status": "ok",
            "backend": self.solver.backend.name,
        })

    async def handle_stats(self, request: web.Request) -> web.Response:
        """GET /stats -- learning engine statistics."""
        stats = self.solver.learning_engine.get_stats()
        return web.json_response(stats)

    def create_app(self) -> web.Application:
        """Create the aiohttp web application."""
        if web is None:
            raise ImportError("aiohttp is required for the API server")

        app = web.Application()
        app.router.add_post("/solve", self.handle_solve)
        app.router.add_post("/feedback", self.handle_feedback)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/stats", self.handle_stats)
        return app


def run_server(host: str = "0.0.0.0", port: int = 8080, config_path: str | None = None) -> None:
    """Run the API server."""
    if web is None:
        raise ImportError("aiohttp is required: pip install fusen-solver[api]")

    from fusen_solver.integrations.cli import _make_backend

    config = load_config(config_path)
    backend = _make_backend(config)
    solver = FusenSolver(backend=backend)

    api = FusenAPI(solver)
    app = api.create_app()

    logger.info("Starting Fusen Solver API on %s:%d", host, port)
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fusen Solver REST API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--config", help="Config YAML path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_server(host=args.host, port=args.port, config_path=args.config)
