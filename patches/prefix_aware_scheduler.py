"""Prefix-cache-aware vLLM V1 scheduler plugin.

Registered via entry_points:

    [vllm.general_plugins]
    prefix_aware_sched = prefix_aware_scheduler:register

Enabled with ``VLLM_PLUGINS=prefix_aware_sched``.

What it does
------------
vLLM's V1 scheduler drains ``self.waiting`` in FIFO order. When many
requests share a common system prompt that is already in the prefix
cache, processing them interleaved with requests from a *different*
system prompt causes the scheduler to thrash: the KV blocks for
prompt-A get evicted, reloaded, evicted again as batches alternate
prefix groups.

This plugin reorders the waiting queue once per scheduling tick so
that requests sharing the same first prefix-cache block-hash land in
the same batch. Tie-break is arrival time (preserves FIFO within a
prefix group, so fairness is not violated beyond one window).

The reorder is a cheap Python sort (~microseconds at queue depths we
care about) executed before the original ``schedule()`` body runs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# [FIXED-SILENT-NONE] Fix 5: block_hashes=None silent FIFO degeneration now emits WARNING (once).
# Ref: plans/silent_none_sweep_20260419.md §C5, plans/KILL_PATTERNS.md §P1
logger.info("[FIXED-SILENT-NONE] prefix_aware_scheduler: block_hashes=None "
            "fallthrough now emits one-time WARNING")

_INSTALLED = False
_bh_none_warned = False  # one-time WARNING flag (per process) for missing block_hashes


def _prefix_key(request):
    """Sort key: (first full prefix-cache block-hash, arrival_time).

    ``Request.block_hashes`` is populated at construction time by the
    scheduler's ``get_request_block_hasher`` when prefix caching is on.
    Each element is a ``BlockHash`` covering BLOCK_SIZE tokens of the
    prompt. Using the *first* block groups requests that share the
    initial system prompt (the common case for multi-agent workloads).

    Requests that are too short to have produced any full block fall
    back to hash 0 and will naturally bucket together at the front.
    """
    global _bh_none_warned
    bh = getattr(request, "block_hashes", None)
    # [FIXED-SILENT-NONE] Fix 5 — Ref: plans/silent_none_sweep_20260419.md §C5, plans/KILL_PATTERNS.md §P1
    if not _bh_none_warned and bh is None:
        logger.warning(
            "[PREFIX-SCHED] request has no block_hashes — sort degenerates to FIFO "
            "(plugin active but no reordering; prefix caching may be off or "
            "Request.block_hashes renamed upstream)"
        )
        _bh_none_warned = True
    first_hash = 0
    if bh:
        h = bh[0]
        # BlockHash objects implement __hash__; reuse int hash for cheap
        # comparison without depending on bytes ordering.
        first_hash = hash(h)
    return (first_hash, request.arrival_time)


def register() -> None:
    """Install the monkey-patch on the V1 Scheduler class."""
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.request_queue import (
        FCFSRequestQueue,
        PriorityRequestQueue,
    )
    from vllm.v1.core.sched.scheduler import Scheduler

    original_schedule = Scheduler.schedule

    def _reorder_fcfs(queue: FCFSRequestQueue) -> None:
        """Sort the deque in-place by prefix-hash then arrival time."""
        if len(queue) < 2:
            return
        ordered = sorted(queue, key=_prefix_key)
        queue.clear()
        queue.extend(ordered)

    def patched_schedule(self):
        # Only reorder FCFS queues — PriorityRequestQueue already has
        # its own (priority, arrival) ordering and we must not break it.
        if isinstance(self.waiting, FCFSRequestQueue):
            _reorder_fcfs(self.waiting)
        if isinstance(self.skipped_waiting, FCFSRequestQueue):
            _reorder_fcfs(self.skipped_waiting)
        return original_schedule(self)

    Scheduler.schedule = patched_schedule
    # AsyncScheduler inherits Scheduler.schedule, but be explicit in case
    # of future subclass overrides.
    if "schedule" in AsyncScheduler.__dict__:
        AsyncScheduler.schedule = patched_schedule
    _INSTALLED = True
    logger.info(
        "[prefix_aware_sched] installed: Scheduler.schedule patched to "
        "sort waiting queue by (first_block_hash, arrival_time)"
    )
