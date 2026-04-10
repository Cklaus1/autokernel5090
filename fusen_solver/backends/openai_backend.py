"""OpenAI API backend adapter.

Supports GPT-4, GPT-4o, o1, etc. Uses the official openai Python package
if available, falls back to raw aiohttp.
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

from fusen_solver.core.interfaces import LLMBackend
from fusen_solver.backends.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """Adapter for the OpenAI API.

    Usage:
        backend = OpenAIBackend(api_key="sk-...", model="gpt-4o")

    The backend reuses a single aiohttp.ClientSession across all requests.
    Use as an async context manager or call ``close()`` when done.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        *,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_context_tokens: int = 128000,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_context = max_context_tokens
        self._session = None  # aiohttp.ClientSession, created lazily

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _get_session(self):
        import aiohttp
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "OpenAIBackend":
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
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"OpenAI error {resp.status}: {error}")
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
            f"{self._base_url}/chat/completions",
            json=payload,
            headers=self._headers(),
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"OpenAI error {resp.status}: {error}")

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
        return True  # Concurrent API calls

    @property
    def max_context(self) -> int:
        return self._max_context

    @property
    def name(self) -> str:
        return f"OpenAI({self._model})"
