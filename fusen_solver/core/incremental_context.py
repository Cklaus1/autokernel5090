"""Incremental context tracking for efficient prefix caching.

Instead of re-sending the full codebase on every solve() call, IncrementalContext
tracks file hashes across calls and exposes what changed. When the LLM backend
supports prefix caching (e.g. vLLM), the unchanged prefix portion is served
from cache, saving tokens and reducing latency.
"""

from __future__ import annotations

import hashlib


class IncrementalContext:
    """Track codebase changes and maintain prefix efficiently.

    Usage::

        ctx = IncrementalContext()

        # First call: full context is built and cached.
        prefix = ctx.build_prefix(codebase)

        # Later calls: only changed files trigger a rebuild.
        # If nothing changed, the cached prefix string is returned instantly.
        prefix = ctx.build_prefix(updated_codebase)

        # Inspect what changed between two snapshots without building a prefix.
        diff = ctx.compute_diff(codebase)
        print(diff["added"], diff["modified"], diff["removed"])
    """

    def __init__(self) -> None:
        self._file_hashes: dict[str, str] = {}  # path -> MD5 of last-sent content
        self._cached_prefix: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_diff(self, codebase: dict[str, str]) -> dict:
        """Find what changed since the last call.

        Side effect: updates the internal hash table so the next call to
        ``compute_diff`` or ``build_prefix`` starts from the new snapshot.

        Args:
            codebase: Mapping of file path to current file content.

        Returns:
            A dict with keys:
            - ``"added"``:    ``{path: content}`` for new files.
            - ``"modified"``: ``{path: content}`` for files whose content changed.
            - ``"removed"``:  ``[path, ...]`` for files that disappeared.
        """
        added: dict[str, str] = {}
        modified: dict[str, str] = {}
        removed: list[str] = []

        current_hashes: dict[str, str] = {}
        for path, content in codebase.items():
            h = hashlib.md5(content.encode()).hexdigest()
            current_hashes[path] = h
            if path not in self._file_hashes:
                added[path] = content
            elif self._file_hashes[path] != h:
                modified[path] = content

        for path in self._file_hashes:
            if path not in current_hashes:
                removed.append(path)

        self._file_hashes = current_hashes
        return {"added": added, "modified": modified, "removed": removed}

    def build_prefix(self, codebase: dict[str, str], *, force_full: bool = False) -> str:
        """Build a context string suitable for use as an LLM system prompt prefix.

        The internal hash table is updated on each call (same as ``compute_diff``).
        The returned string is identical in content whether this is the first call
        or an incremental update — callers can always use it verbatim as the
        system prompt.  The savings come from vLLM / Anthropic prefix-cache hits
        on the unchanged leading portion of the prompt.

        Args:
            codebase:   Mapping of file path to current file content.
            force_full: If ``True``, rebuild even if nothing changed.  Useful
                        when rotating to a new backend session.

        Returns:
            Full context string (all files concatenated with headers).
        """
        diff = self.compute_diff(codebase)

        if force_full or not self._cached_prefix:
            # First call or forced rebuild: send everything.
            self._cached_prefix = self._full_context(codebase)
        elif diff["added"] or diff["modified"] or diff["removed"]:
            # Something changed: rebuild so the prefix reflects reality.
            # vLLM prefix caching works on an exact prefix-string match, so the
            # unchanged *leading* files still hit cache even though we rebuild.
            self._cached_prefix = self._full_context(codebase)
        # else: nothing changed — return cached prefix as-is (zero extra work).

        return self._cached_prefix

    def has_changes(self, codebase: dict[str, str]) -> bool:
        """Quick check whether *anything* has changed since the last snapshot.

        Unlike ``compute_diff``, this method does **not** update the internal
        hash table, so it is safe to call multiple times without side effects.

        Args:
            codebase: Mapping of file path to current file content.

        Returns:
            ``True`` if at least one file was added, modified, or removed.
        """
        if len(codebase) != len(self._file_hashes):
            return True
        for path, content in codebase.items():
            h = hashlib.md5(content.encode()).hexdigest()
            if self._file_hashes.get(path) != h:
                return True
        return False

    def reset(self) -> None:
        """Clear all cached state, forcing a full rebuild on the next call."""
        self._file_hashes = {}
        self._cached_prefix = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _full_context(codebase: dict[str, str]) -> str:
        """Render all files into a single context string.

        Files are sorted by path so the string is deterministic across runs,
        which maximises prefix-cache hit rates.
        """
        parts: list[str] = []
        for path in sorted(codebase):
            content = codebase[path]
            parts.append(f"### {path}\n```\n{content}\n```")
        return "\n\n".join(parts)
