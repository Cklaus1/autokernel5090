"""Multi-backend router: dispatches strategies to different LLM backends.

Example: route "direct fix" to a fast local model, "architecture review" to
Claude Opus, and "test generation" to GPT-4o.
"""

from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from fusen_solver.core.interfaces import LLMBackend

logger = logging.getLogger(__name__)


class MultiBackend(LLMBackend):
    """Routes requests to different backends based on metadata.

    Supports session affinity: when a session_id is provided, subsequent
    requests with the same session_id are routed to the same backend.
    This benefits prefix caching when using DP=2 (multiple GPUs), since
    agents working on the same codebase stay on the same GPU.

    Usage:
        multi = MultiBackend(
            default=vllm_backend,
            routes={
                "review": anthropic_backend,  # use Claude for review
                "security": anthropic_backend,
            },
        )
    """

    def __init__(
        self,
        default: LLMBackend,
        routes: dict[str, LLMBackend] | None = None,
        fallback: LLMBackend | None = None,
        session_ttl: float = 300.0,
    ):
        self._default = default
        self._routes = routes or {}
        self._fallback = fallback
        # Session affinity: session_id -> (backend_name, timestamp)
        self._session_map: dict[str, tuple[str, float]] = {}
        self._session_ttl: float = session_ttl

    def _evict_expired_sessions(self) -> None:
        """Remove session mappings that have exceeded the TTL."""
        now = time.monotonic()
        expired = [
            sid for sid, (_, ts) in self._session_map.items()
            if now - ts > self._session_ttl
        ]
        for sid in expired:
            del self._session_map[sid]

    def _resolve_backend_by_name(self, backend_name: str) -> LLMBackend | None:
        """Look up a backend by its name across default, routes, and fallback."""
        if self._default.name == backend_name:
            return self._default
        for _key, b in self._routes.items():
            if b.name == backend_name:
                return b
        if self._fallback and self._fallback.name == backend_name:
            return self._fallback
        return None

    def route(
        self,
        strategy_name: str | None = None,
        session_id: str | None = None,
    ) -> LLMBackend:
        """Select the backend for a given strategy and/or session.

        Session affinity takes precedence: if a session_id maps to a known
        backend (and has not expired), that backend is returned regardless
        of strategy_name.
        """
        self._evict_expired_sessions()

        # Session affinity lookup
        if session_id and session_id in self._session_map:
            backend_name, _ = self._session_map[session_id]
            backend = self._resolve_backend_by_name(backend_name)
            if backend is not None:
                # Refresh timestamp
                self._session_map[session_id] = (backend_name, time.monotonic())
                return backend
            # Backend disappeared (e.g. reconfigured); fall through

        # Strategy-based routing
        if strategy_name and strategy_name in self._routes:
            backend = self._routes[strategy_name]
        else:
            backend = self._default

        # Record session affinity
        if session_id:
            self._session_map[session_id] = (backend.name, time.monotonic())

        return backend

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> str:
        backend = self.route(session_id=session_id)
        try:
            return await backend.generate(
                messages, max_tokens=max_tokens, temperature=temperature, stop=stop
            )
        except Exception as e:
            if self._fallback:
                logger.warning("Primary backend failed (%s), using fallback", e)
                return await self._fallback.generate(
                    messages, max_tokens=max_tokens, temperature=temperature, stop=stop
                )
            raise

    async def generate_with_strategy(
        self,
        messages: list[dict[str, str]],
        strategy_name: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> str:
        """Generate using the backend routed for this strategy (and session)."""
        backend = self.route(strategy_name, session_id=session_id)
        try:
            return await backend.generate(
                messages, max_tokens=max_tokens, temperature=temperature, stop=stop
            )
        except Exception as e:
            if self._fallback and backend is not self._fallback:
                logger.warning(
                    "Backend %s failed for strategy '%s' (%s), using fallback",
                    backend.name, strategy_name, e,
                )
                return await self._fallback.generate(
                    messages, max_tokens=max_tokens, temperature=temperature, stop=stop
                )
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        backend = self._default
        async for token in backend.stream(
            messages, max_tokens=max_tokens, temperature=temperature, stop=stop
        ):
            yield token

    @property
    def supports_batch(self) -> bool:
        return self._default.supports_batch

    @property
    def max_context(self) -> int:
        return self._default.max_context

    @property
    def name(self) -> str:
        routes_str = ", ".join(f"{k}={v.name}" for k, v in self._routes.items())
        return f"Multi(default={self._default.name}, {routes_str})"
