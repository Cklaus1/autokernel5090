"""Ollama backend adapter -- local models via Ollama's API.

Ollama runs models locally and exposes an OpenAI-compatible API.
Note: batch support depends on the Ollama configuration (usually single-threaded).
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

import aiohttp

from fusen_solver.core.interfaces import LLMBackend

logger = logging.getLogger(__name__)


class OllamaBackend(LLMBackend):
    """Adapter for Ollama's local API.

    Usage:
        backend = OllamaBackend(model="llama3:70b")
    """

    def __init__(
        self,
        model: str = "llama3:70b",
        *,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        max_context_tokens: int = 8192,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_context = max_context_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        ) as session:
            async with session.post(
                f"{self._base_url}/api/chat", json=payload
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {error}")
                data = await resp.json()
                return data.get("message", {}).get("content", "")

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        ) as session:
            async with session.post(
                f"{self._base_url}/api/chat", json=payload
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {error}")

                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded:
                        continue
                    try:
                        data = json.loads(decoded)
                        text = data.get("message", {}).get("content", "")
                        if text:
                            yield text
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

    @property
    def supports_batch(self) -> bool:
        return False  # Ollama is typically single-threaded

    @property
    def max_context(self) -> int:
        return self._max_context

    @property
    def name(self) -> str:
        return f"Ollama({self._model})"
