"""vLLM backend adapter -- local GPU inference with batch support.

This is the fastest backend: vLLM batches all N parallel requests on the GPU,
and prefix caching means the shared codebase context is computed only once.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import aiohttp

from fusen_solver.core.interfaces import LLMBackend
from fusen_solver.backends.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class VLLMBackend(LLMBackend):
    """Adapter for a local vLLM instance with OpenAI-compatible API.

    Usage:
        backend = VLLMBackend(
            base_url="http://localhost:8000/v1",
            model="gemma-4-26B-A4B-it-NVFP4",
        )

    The backend reuses a single aiohttp.ClientSession across all requests
    (one TCP connection pool) instead of creating a new session per call.
    Use as an async context manager or call ``close()`` when done.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        *,
        timeout: float = 300.0,
        max_context_tokens: int = 131072,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_context = max_context_tokens
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return (or lazily create) the shared aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "VLLMBackend":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        priority: int | None = None,
        **kwargs,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop
        if priority is not None:
            payload["priority"] = priority

        async def _call() -> str:
            session = await self._get_session()
            async with session.post(
                f"{self._base_url}/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"vLLM error {resp.status}: {error}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        return await retry_with_backoff(_call)

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop

        session = await self._get_session()
        async with session.post(
            f"{self._base_url}/chat/completions", json=payload
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"vLLM error {resp.status}: {error}")

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    @property
    def supports_batch(self) -> bool:
        return True  # vLLM batches on GPU

    @property
    def max_context(self) -> int:
        return self._max_context

    @property
    def name(self) -> str:
        return f"vLLM({self._model})"
