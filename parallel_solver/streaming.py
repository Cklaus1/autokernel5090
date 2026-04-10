"""Concurrent streaming from multiple vLLM agents.

Launches N chat completion requests in parallel using asyncio + aiohttp,
streams token-by-token from all agents simultaneously, and supports
early cancellation when one agent finishes or a timeout is reached.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from a single agent."""

    agent_id: int
    strategy: str
    content: str
    tokens_generated: int = 0
    latency_ms: float = 0.0
    tokens_per_sec: float = 0.0
    finished: bool = False
    cancelled: bool = False
    error: str | None = None


@dataclass
class StreamEvent:
    """A single streaming event from any agent."""

    agent_id: int
    strategy: str
    delta: str  # incremental text
    cumulative: str  # full text so far
    finished: bool = False
    error: str | None = None


class ParallelStreamer:
    """Streams completions from multiple vLLM agents in parallel.

    All requests hit the same vLLM instance, which batches them on the GPU.
    Because agents share a prefix (codebase), the KV cache for the prefix
    is computed once and reused across all agents.
    """

    def __init__(
        self,
        vllm_api: str,
        model: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout_seconds: float = 300.0,
        cancel_on_first_finish: bool = False,
    ):
        self.api = vllm_api.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.cancel_on_first_finish = cancel_on_first_finish

    async def run_all(
        self,
        requests: list[dict],
        *,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> list[AgentResult]:
        """Run all agent requests in parallel and collect results.

        Args:
            requests: List of dicts, each with keys:
                - agent_id: int
                - strategy: str
                - messages: list[dict] (chat messages)
            on_event: Optional callback for each streaming token.

        Returns:
            List of AgentResult, one per request.
        """
        cancel_event = asyncio.Event()
        results: list[AgentResult] = []

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
        ) as session:
            tasks = [
                self._stream_one(session, req, cancel_event, on_event)
                for req in requests
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final: list[AgentResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                req = requests[i]
                final.append(
                    AgentResult(
                        agent_id=req["agent_id"],
                        strategy=req["strategy"],
                        content="",
                        error=str(r),
                    )
                )
            else:
                final.append(r)

        return final

    async def stream_all(
        self,
        requests: list[dict],
    ) -> AsyncIterator[StreamEvent]:
        """Yield streaming events from all agents as they arrive.

        Events are interleaved in arrival order -- the caller sees real-time
        progress from all agents simultaneously.
        """
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        cancel_event = asyncio.Event()
        active_count = len(requests)

        async def _producer(session: aiohttp.ClientSession, req: dict) -> None:
            nonlocal active_count

            async def _enqueue(event: StreamEvent) -> None:
                await queue.put(event)

            try:
                result = await self._stream_one(session, req, cancel_event, _enqueue)
                # Send final event
                await queue.put(
                    StreamEvent(
                        agent_id=req["agent_id"],
                        strategy=req["strategy"],
                        delta="",
                        cumulative=result.content,
                        finished=True,
                    )
                )
            except Exception as e:
                await queue.put(
                    StreamEvent(
                        agent_id=req["agent_id"],
                        strategy=req["strategy"],
                        delta="",
                        cumulative="",
                        error=str(e),
                    )
                )
            finally:
                active_count -= 1
                if active_count == 0:
                    await queue.put(None)  # sentinel

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
        ) as session:
            tasks = [
                asyncio.create_task(_producer(session, req)) for req in requests
            ]

            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event

            # Ensure all tasks complete
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _stream_one(
        self,
        session: aiohttp.ClientSession,
        req: dict,
        cancel_event: asyncio.Event,
        on_event: Callable | None,
    ) -> AgentResult:
        """Stream a single agent's completion."""
        agent_id = req["agent_id"]
        strategy = req["strategy"]
        messages = req["messages"]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        content_parts: list[str] = []
        tokens = 0
        t0 = time.perf_counter()

        try:
            async with session.post(
                f"{self.api}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    return AgentResult(
                        agent_id=agent_id,
                        strategy=strategy,
                        content="",
                        error=f"HTTP {resp.status}: {error_text}",
                    )

                async for line in resp.content:
                    if cancel_event.is_set():
                        return AgentResult(
                            agent_id=agent_id,
                            strategy=strategy,
                            content="".join(content_parts),
                            tokens_generated=tokens,
                            cancelled=True,
                        )

                    decoded = line.decode("utf-8").strip()
                    if not decoded.startswith("data: "):
                        continue
                    data_str = decoded[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        content_parts.append(text)
                        tokens += 1

                        if on_event is not None:
                            event = StreamEvent(
                                agent_id=agent_id,
                                strategy=strategy,
                                delta=text,
                                cumulative="".join(content_parts),
                            )
                            if asyncio.iscoroutinefunction(on_event):
                                await on_event(event)
                            else:
                                on_event(event)

        except asyncio.TimeoutError:
            return AgentResult(
                agent_id=agent_id,
                strategy=strategy,
                content="".join(content_parts),
                tokens_generated=tokens,
                error="timeout",
            )
        except aiohttp.ClientError as e:
            return AgentResult(
                agent_id=agent_id,
                strategy=strategy,
                content="".join(content_parts),
                tokens_generated=tokens,
                error=str(e),
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        content = "".join(content_parts)
        tps = (tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0.0

        result = AgentResult(
            agent_id=agent_id,
            strategy=strategy,
            content=content,
            tokens_generated=tokens,
            latency_ms=elapsed_ms,
            tokens_per_sec=tps,
            finished=True,
        )

        if self.cancel_on_first_finish:
            cancel_event.set()
            logger.info("Agent %d (%s) finished first, cancelling others", agent_id, strategy)

        return result
