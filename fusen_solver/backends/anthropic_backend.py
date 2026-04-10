"""Anthropic Claude API backend adapter.

Supports Claude Opus, Sonnet, Haiku via the Anthropic Messages API.
Uses the official anthropic Python package if available, falls back to raw aiohttp.
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

from fusen_solver.core.interfaces import LLMBackend

logger = logging.getLogger(__name__)


class AnthropicBackend(LLMBackend):
    """Adapter for the Anthropic Claude API.

    Usage:
        backend = AnthropicBackend(api_key="sk-ant-...", model="claude-sonnet-4-20250514")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        *,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
        max_context_tokens: int = 200000,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_context = max_context_tokens

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        import aiohttp

        # Convert OpenAI message format to Anthropic format
        system_msg, user_messages = self._convert_messages(messages)

        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
        }
        if system_msg:
            payload["system"] = system_msg
        if stop:
            payload["stop_sequences"] = stop

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        ) as session:
            async with session.post(
                f"{self._base_url}/v1/messages",
                json=payload,
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Anthropic error {resp.status}: {error}")
                data = await resp.json()
                # Anthropic returns content as a list of blocks
                content_blocks = data.get("content", [])
                return "".join(
                    block.get("text", "")
                    for block in content_blocks
                    if block.get("type") == "text"
                )

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        import aiohttp

        system_msg, user_messages = self._convert_messages(messages)

        payload: dict = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages,
            "stream": True,
        }
        if system_msg:
            payload["system"] = system_msg
        if stop:
            payload["stop_sequences"] = stop

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._timeout)
        ) as session:
            async with session.post(
                f"{self._base_url}/v1/messages",
                json=payload,
                headers=self._headers(),
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"Anthropic error {resp.status}: {error}")

                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data: "):
                        continue
                    data_str = decoded[6:]
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text
                    except (json.JSONDecodeError, KeyError):
                        continue

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Convert OpenAI-format messages to Anthropic format.

        Anthropic uses a separate 'system' parameter rather than a system message
        in the messages list.

        Returns:
            (system_content, user_messages)
        """
        system = ""
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system += msg["content"] + "\n"
            else:
                user_messages.append(msg)

        return system.strip(), user_messages

    @property
    def supports_batch(self) -> bool:
        return True  # Concurrent API calls

    @property
    def max_context(self) -> int:
        return self._max_context

    @property
    def name(self) -> str:
        return f"Anthropic({self._model})"
