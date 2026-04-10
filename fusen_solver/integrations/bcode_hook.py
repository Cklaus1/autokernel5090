"""Minimal, non-invasive hooks for BCode integration.

These are fire-and-forget callbacks that BCode can optionally call.
They never block, never raise, and never modify BCode's output.

Usage from BCode:
    from fusen_solver.integrations.bcode_hook import on_bcode_task_start, on_bcode_task_complete

    # When a task starts (fire and forget):
    asyncio.create_task(on_bcode_task_start(prd, workspace))

    # When a task finishes (records comparison):
    asyncio.create_task(on_bcode_task_complete(prd, workspace, bcode_output, elapsed))
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Module-level shadow instance and pending task tracking.
# Using module state keeps the hooks stateless from BCode's perspective.
_pending_shadows: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]
_shadow_start_times: dict[str, float] = {}


async def on_bcode_task_start(prd: str, workspace: str) -> None:
    """Called when BCode starts a task. Launches shadow in background.

    This is fire-and-forget -- it swallows all exceptions so it can
    never break BCode, even if fusen_solver is misconfigured.
    """
    try:
        from fusen_solver.integrations.bcode_shadow import BCodeShadow

        shadow = BCodeShadow()
        task_key = _task_key(prd)

        # Cancel any stale shadow for the same task
        old = _pending_shadows.pop(task_key, None)
        if old and not old.done():
            old.cancel()

        _shadow_start_times[task_key] = time.monotonic()

        async def _run() -> None:
            await shadow.shadow_run(prd, workspace)

        _pending_shadows[task_key] = asyncio.create_task(_run())
        logger.debug("Shadow started for task: %s...", prd[:80])

    except Exception:
        # Never let shadow failures propagate to BCode
        logger.debug("Shadow start failed (non-fatal)", exc_info=True)


async def on_bcode_task_complete(
    prd: str,
    workspace: str,
    bcode_output: dict[str, Any],
    bcode_time_s: float = 0.0,
) -> None:
    """Called when BCode finishes. Records comparison if shadow also finished.

    If the shadow hasn't finished yet, waits up to 10s for it, then
    records whatever is available.
    """
    try:
        from fusen_solver.integrations.bcode_shadow import BCodeShadow

        task_key = _task_key(prd)
        shadow_task = _pending_shadows.pop(task_key, None)

        if shadow_task is not None and not shadow_task.done():
            # Give the shadow a brief grace period
            try:
                await asyncio.wait_for(asyncio.shield(shadow_task), timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.debug("Shadow did not finish in grace period")

        # Record comparison regardless -- BCodeShadow.compare works
        # even if the shadow run didn't complete (scores will be 0).
        shadow = BCodeShadow()
        result = shadow.compare(bcode_output, task=prd)
        result.bcode_time_s = bcode_time_s
        shadow._log(result)

        logger.debug(
            "Shadow comparison logged: winner=%s (bcode=%.2f, fusen=%.2f)",
            result.winner,
            result.bcode_score,
            result.fusen_score,
        )

        # Clean up start time
        _shadow_start_times.pop(task_key, None)

    except Exception:
        # Never let shadow failures propagate to BCode
        logger.debug("Shadow complete hook failed (non-fatal)", exc_info=True)


def _task_key(prd: str) -> str:
    """Derive a stable key from a PRD for dedup purposes."""
    return prd[:200]
