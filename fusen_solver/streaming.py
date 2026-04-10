"""Cancellable streaming helpers for racing agent mode.

When multiple agents race, the first accepted solution should trigger
cancellation of all other in-flight requests. For vLLM, closing the HTTP
connection is sufficient -- vLLM's ``with_cancellation`` decorator detects
the disconnect and frees the KV cache immediately.

For non-streaming (batch) requests via aiohttp, cancelling the asyncio
task causes aiohttp to close the connection, which vLLM also detects.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class CancellableRequest:
    """Tracks a single in-flight LLM request that can be cancelled.

    Closing the aiohttp response (or the session) triggers a disconnect
    that vLLM detects, aborting the generation and freeing GPU KV cache.
    """

    agent_idx: int
    task: asyncio.Task[Any] | None = None
    session: aiohttp.ClientSession | None = None
    response: aiohttp.ClientResponse | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    cancelled: bool = False
    # Estimated tokens generated before cancellation (for KV savings tracking)
    estimated_tokens_generated: int = 0

    @property
    def elapsed(self) -> float:
        end = self.end_time if self.end_time else time.perf_counter()
        return end - self.start_time if self.start_time else 0.0

    async def cancel(self) -> None:
        """Cancel this request, closing the HTTP connection.

        For vLLM: closing the connection triggers ``with_cancellation``
        which aborts generation and frees the KV cache slots.
        """
        if self.cancelled:
            return
        self.cancelled = True
        self.end_time = time.perf_counter()

        # Close the HTTP response first (signals disconnect to vLLM)
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass

        # Close the session
        if self.session is not None:
            try:
                await self.session.close()
            except Exception:
                pass

        # Cancel the asyncio task
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):
                pass

        logger.debug(
            "Cancelled agent %d after %.2fs (~%d tokens generated)",
            self.agent_idx,
            self.elapsed,
            self.estimated_tokens_generated,
        )


@dataclass
class RacingStats:
    """Statistics from a racing solve, including KV cache savings."""

    total_agents: int = 0
    winner_idx: int = -1
    winner_time_s: float = 0.0
    cancelled_agents: int = 0
    # KV savings: estimated tokens that were NOT generated due to cancellation
    estimated_tokens_saved: int = 0
    # Per-agent timing
    agent_times: list[float] = field(default_factory=list)
    # Whether we hit the timeout
    timed_out: bool = False
    # Number of solutions rejected before accepting one
    rejections_before_accept: int = 0

    @property
    def kv_savings_pct(self) -> float:
        """Estimate of KV cache memory saved by early cancellation.

        This is a rough estimate based on tokens not generated. In practice
        the savings are higher because vLLM can immediately reuse the freed
        KV blocks for new requests.
        """
        if self.total_agents <= 1:
            return 0.0
        # If we cancelled N-1 agents, we saved roughly (N-1)/N of total compute
        if self.cancelled_agents == 0:
            return 0.0
        return self.cancelled_agents / self.total_agents * 100.0


class RacingCoordinator:
    """Coordinates racing agents with first-accepted-wins semantics.

    Launches N agents as asyncio tasks, monitors completions via
    ``asyncio.wait(FIRST_COMPLETED)``, scores each result, and cancels
    all remaining agents once an accepted solution is found.
    """

    def __init__(self, accept_threshold: float = 0.7, timeout: float = 30.0):
        self.accept_threshold = accept_threshold
        self.timeout = timeout
        self._requests: list[CancellableRequest] = []
        self.stats = RacingStats()

    def register(self, agent_idx: int) -> CancellableRequest:
        """Register a new agent for racing."""
        req = CancellableRequest(agent_idx=agent_idx, start_time=time.perf_counter())
        self._requests.append(req)
        return req

    async def cancel_all_except(self, winner_idx: int) -> None:
        """Cancel all agents except the winner."""
        cancel_tasks = []
        for req in self._requests:
            if req.agent_idx != winner_idx and not req.cancelled:
                cancel_tasks.append(req.cancel())
                self.stats.cancelled_agents += 1
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

    async def cancel_all(self) -> None:
        """Cancel all remaining agents (used on timeout)."""
        cancel_tasks = []
        for req in self._requests:
            if not req.cancelled:
                cancel_tasks.append(req.cancel())
                self.stats.cancelled_agents += 1
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

    def finalize_stats(self, winner_idx: int, winner_time: float) -> RacingStats:
        """Compute final racing statistics."""
        self.stats.total_agents = len(self._requests)
        self.stats.winner_idx = winner_idx
        self.stats.winner_time_s = winner_time

        # Collect per-agent timing
        self.stats.agent_times = [req.elapsed for req in self._requests]

        # Estimate tokens saved from cancelled agents
        total_saved = 0
        for req in self._requests:
            if req.cancelled:
                total_saved += req.estimated_tokens_generated
        self.stats.estimated_tokens_saved = total_saved

        return self.stats
