"""
EarlyAbortStreamWrapper — thin middleware for vLLM SSE streams.

Monitors streaming output for semantic completion signals and calls
engine.abort(request_id) to free KV cache blocks immediately, rather
than waiting for the model to generate to EOS or max_tokens.

Design reference: plans/early_kv_termination.md, Layer 2
RTX PRO 6000 motivation: plans/rtx_pro6000_experiments.md ASI-2

Usage (fusen_solver / OpenAI-compatible proxy):

    wrapper = EarlyAbortStreamWrapper(
        engine_client=engine,
        request_id=req_id,
        stream=engine.generate(prompt, request_id=req_id),
    )
    async for chunk in wrapper:
        yield chunk  # forward SSE chunks to HTTP client
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Sequence

# ---------------------------------------------------------------------------
# Signal patterns that indicate semantic completion
# ---------------------------------------------------------------------------

# Default patterns: tool call close tag, answer close tag, chain-of-thought
# close tag, and repeated blank lines (common paragraph break used as EOS
# by some instruction-tuned models).
DEFAULT_ABORT_PATTERNS: tuple[str, ...] = (
    "</tool_call>",
    "</answer>",
    "</think>",
    "\n\n\n",  # three consecutive newlines — strong signal of trailing padding
)

# Compiled regex for repeated-newline detection (≥3 newlines anywhere in buffer)
_REPEATED_NEWLINES_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(chunk: object) -> str:
    """Pull delta text out of a vLLM output chunk.

    Works with both:
    - vLLM internal RequestOutput objects (chunk.outputs[0].text or .delta)
    - OpenAI-style SSE dict/str chunks from an HTTP proxy
    """
    # vLLM RequestOutput / CompletionOutput
    if hasattr(chunk, "outputs"):
        parts = []
        for out in chunk.outputs:
            if hasattr(out, "text") and out.text:
                parts.append(out.text)
        return "".join(parts)

    # OpenAI SSE dict (already parsed)
    if isinstance(chunk, dict):
        try:
            return chunk["choices"][0]["delta"].get("content", "") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    # Raw string (unusual but safe)
    if isinstance(chunk, str):
        return chunk

    return ""


def _make_early_abort_chunk(request_id: str) -> dict:
    """Synthesize a final SSE chunk with finish_reason='early_abort'.

    Clients that inspect finish_reason will see this is not a natural stop
    and can handle accordingly (e.g. suppress, log, retry without stream).
    """
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "unknown",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "early_abort",
                "logprobs": None,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

@dataclass
class EarlyAbortStreamWrapper:
    """Async iterator that wraps a vLLM generation stream.

    Forwards every chunk to the caller unchanged, then inspects the
    accumulated text buffer for abort_patterns.  On first match it:
      1. Calls await engine_client.abort(request_id) — returns KV blocks NOW.
      2. Yields a synthetic final chunk with finish_reason="early_abort".
      3. Stops iteration (breaks the upstream generator).

    Args:
        engine_client: Any object with an async `abort(request_id)` method.
                       Compatible with vLLM's AsyncLLMEngine and AsyncLLM (v1).
        request_id:    The request_id passed to engine.generate().
        stream:        The async iterator returned by engine.generate().
        abort_patterns: Strings/regexes to watch for in the output buffer.
                        Defaults to DEFAULT_ABORT_PATTERNS.
        max_buffer:    Maximum characters to retain in the rolling text buffer.
                       Keeps memory bounded for very long streams.
    """

    engine_client: object
    request_id: str
    stream: AsyncIterator
    abort_patterns: Sequence[str] = field(
        default_factory=lambda: list(DEFAULT_ABORT_PATTERNS)
    )
    max_buffer: int = 8192  # ~2K tokens at 4 chars/token — enough for any tag

    def _matches(self, buffer: str) -> str | None:
        """Return the first matching pattern, or None."""
        for pattern in self.abort_patterns:
            if pattern in buffer:
                return pattern
        if _REPEATED_NEWLINES_RE.search(buffer):
            return "\\n{3,}"
        return None

    async def __aiter__(self) -> AsyncIterator:
        buffer = ""
        aborted = False

        async for chunk in self.stream:
            yield chunk  # always forward first — client gets complete data

            if aborted:
                # Drain any remaining buffered chunks the engine already queued
                # (engine_core IPC lag means a few more chunks may arrive after
                # abort() returns).  Forward them but don't update buffer.
                continue

            text = _extract_text(chunk)
            if text:
                buffer += text
                # Keep buffer bounded: retain only the tail
                if len(buffer) > self.max_buffer:
                    buffer = buffer[-self.max_buffer :]

            matched = self._matches(buffer)
            if matched:
                aborted = True
                # Free KV blocks on the engine side immediately.
                # abort() is idempotent — safe to call even if request is
                # already finishing naturally.
                try:
                    await self.engine_client.abort(self.request_id)
                except Exception:
                    pass  # engine may have already finished; not fatal

                # Synthesize a terminal chunk so downstream knows why the
                # stream ended without a natural finish_reason from the model.
                yield _make_early_abort_chunk(self.request_id)
                return  # stop iterating — do not read further from upstream


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def wrap_stream(
    engine_client,
    request_id: str,
    stream: AsyncIterator,
    extra_patterns: Sequence[str] = (),
) -> EarlyAbortStreamWrapper:
    """Convenience constructor merging default + caller-supplied patterns."""
    patterns = list(DEFAULT_ABORT_PATTERNS) + list(extra_patterns)
    return EarlyAbortStreamWrapper(
        engine_client=engine_client,
        request_id=request_id,
        stream=stream,
        abort_patterns=patterns,
    )
