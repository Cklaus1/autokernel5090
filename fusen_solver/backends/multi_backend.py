"""Multi-backend router: dispatches strategies to different LLM backends.

Example: route "direct fix" to a fast local model, "architecture review" to
Claude Opus, and "test generation" to GPT-4o.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from fusen_solver.core.interfaces import LLMBackend

logger = logging.getLogger(__name__)


class MultiBackend(LLMBackend):
    """Routes requests to different backends based on metadata.

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
    ):
        self._default = default
        self._routes = routes or {}
        self._fallback = fallback

    def route(self, strategy_name: str | None = None) -> LLMBackend:
        """Select the backend for a given strategy."""
        if strategy_name and strategy_name in self._routes:
            return self._routes[strategy_name]
        return self._default

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        # Default route (no strategy context)
        backend = self._default
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
    ) -> str:
        """Generate using the backend routed for this strategy."""
        backend = self.route(strategy_name)
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
