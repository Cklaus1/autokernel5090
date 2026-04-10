"""Prefix manager -- builds the system/context prefix for LLM prompts.

Responsibilities:
    - Load a full codebase into the prompt (small repos).
    - Intelligently select the most relevant files from a large codebase
      (large repos where stuffing everything would exceed the context window).
    - Compose system instructions with the selected context.

The smart selection path delegates to ``CodebaseIndex`` in
``fusen_solver.core.codebase_index``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fusen_solver.core.codebase_index import CodebaseIndex

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Conservative default token budget so there is room left for the problem
#: description, strategy prompt, and the LLM's own reply.
DEFAULT_TOKEN_BUDGET = 50_000

#: Character limit used when deciding whether a whole-codebase load is "safe".
#: Approx. 128 K tokens × 4 chars/token.
_WHOLE_CODEBASE_CHAR_LIMIT = 512_000


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class PrefixManager:
    """Build context prefixes for LLM prompts.

    Parameters
    ----------
    system_prompt:
        Base system instructions prepended to every prompt.
    token_budget:
        Maximum tokens to use for file context in smart-load mode.
    """

    def __init__(
        self,
        system_prompt: str = "",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self.system_prompt = system_prompt
        self.token_budget = token_budget
        # Optional cached index -- reused if the same root is loaded repeatedly.
        self._index: CodebaseIndex | None = None
        self._index_root: str | None = None

    # ------------------------------------------------------------------
    # Simple whole-codebase load (small repos)
    # ------------------------------------------------------------------

    def load_codebase(
        self,
        root_path: str,
        *,
        extensions: list[str] | None = None,
    ) -> str:
        """Load **all** source files under *root_path* into a context string.

        This is suitable for small repos where the total content fits within
        the LLM context window.  For large repos use ``load_codebase_smart``.

        Parameters
        ----------
        root_path:
            Root directory of the codebase.
        extensions:
            Whitelist of file extensions to include (e.g. ``[".py", ".ts"]``).
            When ``None`` every non-binary text file is included.

        Returns
        -------
        str
            Fenced-block context string ready for inclusion in a prompt.
        """
        root = Path(root_path).resolve()
        exts = {e.lower() for e in extensions} if extensions else None
        parts: list[str] = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Skip noise directories.
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".")
                and d not in {"node_modules", "__pycache__", "venv", ".venv", "dist", "build"}
            ]
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                if exts is not None and Path(fname).suffix.lower() not in exts:
                    continue
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, root)
                try:
                    content = Path(abs_path).read_text(encoding="utf-8", errors="replace")
                except (OSError, PermissionError):
                    continue
                ext = Path(fname).suffix.lstrip(".")
                parts.append(f"### {rel_path}\n```{ext}\n{content}\n```")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Smart context selection (large repos)
    # ------------------------------------------------------------------

    def load_codebase_smart(
        self,
        root_path: str,
        problem: str,
        token_budget: int | None = None,
    ) -> str:
        """Load **relevant** parts of a codebase for a given problem.

        For large codebases that exceed the LLM context window this method
        scores every file with a multi-signal relevance function and greedily
        packs the highest-scoring files within *token_budget*.

        Signals used (see ``CodebaseIndex`` for details):
            1. Keyword overlap between the problem description and file content.
            2. Import/export graph affinity.
            3. File recency (recently modified files score higher).
            4. File-type hints inferred from the problem description.

        Parameters
        ----------
        root_path:
            Root directory of the codebase.
        problem:
            Natural-language description of the task / problem.
        token_budget:
            Maximum tokens to include.  Falls back to ``self.token_budget``.

        Returns
        -------
        str
            Fenced-block context string ready for inclusion in a prompt.
        """
        budget = token_budget if token_budget is not None else self.token_budget
        index = self._get_index(root_path)
        relevant_files = index.select_relevant(problem, token_budget=budget)
        return index.build_context(relevant_files)

    # ------------------------------------------------------------------
    # Auto-selecting loader
    # ------------------------------------------------------------------

    def load_codebase_auto(
        self,
        root_path: str,
        problem: str = "",
        token_budget: int | None = None,
    ) -> str:
        """Load codebase context, automatically choosing the loading strategy.

        If the estimated total codebase size fits within the token budget the
        whole codebase is loaded (``load_codebase``).  Otherwise the smart
        selection path is used (``load_codebase_smart``).

        Parameters
        ----------
        root_path:
            Root directory of the codebase.
        problem:
            Problem description (used only in smart-selection mode).
        token_budget:
            Maximum tokens.  Falls back to ``self.token_budget``.

        Returns
        -------
        str
            Context string.
        """
        budget = token_budget if token_budget is not None else self.token_budget
        index = self._get_index(root_path)
        estimated_total = sum(i.estimated_tokens for i in index.files.values())
        if estimated_total <= budget:
            # Small enough -- load everything via the index's build_context
            # so we get consistent formatting.
            all_files = list(index.files.keys())
            return index.build_context(all_files)
        # Large codebase -- use smart selection.
        relevant_files = index.select_relevant(problem, token_budget=budget)
        return index.build_context(relevant_files)

    # ------------------------------------------------------------------
    # System-prompt composition
    # ------------------------------------------------------------------

    def build_system_message(
        self,
        *,
        extra_context: str = "",
        codebase_context: str = "",
    ) -> str:
        """Compose the full system message for an LLM prompt.

        Parameters
        ----------
        extra_context:
            Any additional context to append after the system prompt.
        codebase_context:
            Pre-built codebase context string (from one of the load methods).

        Returns
        -------
        str
            Full system message text.
        """
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt.strip())
        if codebase_context:
            parts.append("## Codebase Context\n\n" + codebase_context)
        if extra_context:
            parts.append(extra_context.strip())
        return "\n\n".join(parts)

    def build_messages(
        self,
        user_message: str,
        *,
        codebase_context: str = "",
        extra_context: str = "",
    ) -> list[dict[str, str]]:
        """Build a full chat message list for an LLM call.

        Parameters
        ----------
        user_message:
            The user's question or problem description.
        codebase_context:
            Pre-built codebase context string.
        extra_context:
            Additional context appended to the system message.

        Returns
        -------
        list[dict[str, str]]
            OpenAI-compatible message list.
        """
        system = self.build_system_message(
            extra_context=extra_context,
            codebase_context=codebase_context,
        )
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})
        return messages

    # ------------------------------------------------------------------
    # Index cache management
    # ------------------------------------------------------------------

    def _get_index(self, root_path: str) -> CodebaseIndex:
        """Return (possibly cached) ``CodebaseIndex`` for *root_path*."""
        root = os.path.abspath(root_path)
        if self._index is None or self._index_root != root:
            self._index = CodebaseIndex(root)
            self._index_root = root
        return self._index

    def invalidate_index(self) -> None:
        """Discard the cached index so the next call re-indexes the codebase."""
        self._index = None
        self._index_root = None

    def index_summary(self, root_path: str) -> str:
        """Return a human-readable summary of the indexed codebase."""
        return self._get_index(root_path).summary()
