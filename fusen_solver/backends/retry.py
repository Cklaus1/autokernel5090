"""Retry utility for transient LLM API failures.

Provides exponential backoff with jitter for HTTP 429/5xx errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Status codes that warrant a retry (rate limit, server errors).
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Call *fn* up to *max_retries* times with exponential backoff.

    Only retries on exceptions that carry an HTTP status code in
    ``_RETRYABLE_STATUSES`` (aiohttp ``ClientResponseError`` exposes
    ``.status``).  All other exceptions are re-raised immediately on
    the first occurrence.

    Args:
        fn: Zero-argument async callable to retry.
        max_retries: Total number of attempts (including the first).
        base_delay: Delay in seconds before the second attempt; doubles
            each subsequent attempt.

    Returns:
        Whatever *fn* returns on success.

    Raises:
        The last exception raised by *fn* when all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            status = getattr(exc, "status", None)
            if attempt == max_retries - 1:
                # Final attempt — propagate.
                raise
            if status in _RETRYABLE_STATUSES:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Retryable error (status=%s, attempt=%d/%d): %s — retrying in %.1fs",
                    status,
                    attempt + 1,
                    max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                # Non-retryable error — propagate immediately.
                raise
    # Should be unreachable, but satisfies type checkers.
    raise last_exc  # type: ignore[misc]
