"""Smart codebase indexing for large codebases.

When a codebase exceeds the LLM context window (e.g. >128K tokens), it is
impossible to include all files in a single prompt.  ``CodebaseIndex`` walks
the repository, extracts lightweight metadata for every source file, and then
uses a multi-signal relevance scoring strategy to select the *most useful*
files for a given problem description within a configurable token budget.

Scoring signals (weighted sum):
    1. Keyword overlap  -- terms from the problem description found in the file.
    2. Import affinity  -- if the problem mentions a module/symbol, files that
                           import or export it score higher.
    3. Recency          -- recently-modified files are more likely to be the
                           ones that need attention.
    4. Type hint        -- the problem description may contain phrases like
                           "fix test" or "update config"; matching file-type
                           heuristics boost those files.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Directories that are almost never relevant to LLM context.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "_build",
        "site-packages",
        ".tox",
        "htmlcov",
        ".cache",
    }
)

#: Binary / non-text file extensions that are never useful in LLM context.
_SKIP_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc", ".pyo", ".pyd",
        ".so", ".dylib", ".dll", ".a", ".lib",
        ".exe", ".bin", ".elf",
        ".o", ".obj",
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg", ".webp",
        ".mp3", ".mp4", ".wav", ".ogg", ".flac",
        ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".pkl", ".pickle", ".npy", ".npz", ".pt", ".pth", ".ckpt",
        ".parquet", ".feather", ".arrow",
        ".db", ".sqlite", ".sqlite3",
        ".lock",  # lockfiles are huge and irrelevant
    }
)

#: Language detection by extension.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".cfg": "ini",
    ".ini": "ini",
    ".env": "dotenv",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".lua": "lua",
    ".r": "r",
    ".R": "r",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".triton": "triton",
}

#: Words in problem descriptions that hint at file types to prioritise.
_TYPE_HINTS: list[tuple[list[str], list[str]]] = [
    # (problem keywords, preferred language/filename patterns)
    (["test", "tests", "spec", "unittest", "pytest"], ["test_", "_test.", "spec.", "_spec."]),
    (["config", "configuration", "settings", "setup"], ["config", "settings", "setup", ".toml", ".yaml", ".yml", ".ini", ".cfg"]),
    (["readme", "docs", "documentation"], ["readme", ".md", ".rst", "docs/"]),
    (["ci", "workflow", "pipeline", "github actions", "action"], [".github/", "workflow", "pipeline", "ci"]),
    (["docker", "container", "dockerfile"], ["dockerfile", "docker-compose", ".dockerfile"]),
    (["schema", "migration", "model", "orm", "database", "sql"], [".sql", "schema", "migration", "model"]),
]

# Approximate tokens per character for plain text (conservative estimate).
_CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FileInfo:
    """Lightweight metadata extracted from a single source file."""

    path: str                          # relative to root
    size_bytes: int = 0
    language: str = "text"
    last_modified: float = 0.0         # UNIX timestamp
    estimated_tokens: int = 0
    imports: list[str] = field(default_factory=list)    # module names this file imports
    exports: list[str] = field(default_factory=list)    # top-level names defined here
    content_preview: str = ""          # first 500 chars, for quick keyword scanning

    # Set after indexing completes
    _content_cache: str | None = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CodebaseIndex:
    """Index a codebase for intelligent context selection.

    Parameters
    ----------
    root_path:
        Absolute or relative path to the root of the repository.
    max_file_tokens:
        Files larger than this are indexed (metadata) but never selected --
        they would consume the whole budget by themselves.
    """

    def __init__(self, root_path: str, max_file_tokens: int = 20_000) -> None:
        self.root = os.path.abspath(root_path)
        self.max_file_tokens = max_file_tokens
        self.files: dict[str, FileInfo] = {}   # relative path → info
        self._index_time: float = 0.0
        self.index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self) -> None:
        """Walk the codebase and extract file metadata.

        Populates ``self.files`` with one ``FileInfo`` per source file,
        skipping binary files, hidden directories, and known noise paths.
        """
        t0 = time.monotonic()
        self.files.clear()

        for dirpath, dirnames, filenames in os.walk(self.root):
            # Prune skip-dirs in-place so os.walk doesn't descend into them.
            dirnames[:] = [
                d for d in dirnames
                if d not in _SKIP_DIRS and not d.startswith(".")
            ]

            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, self.root)

                # Skip by extension
                ext = Path(fname).suffix.lower()
                if ext in _SKIP_EXTENSIONS:
                    continue

                # Skip hidden files
                if fname.startswith("."):
                    continue

                info = self._extract_metadata(abs_path, rel_path)
                if info is not None:
                    self.files[rel_path] = info

        self._index_time = time.monotonic() - t0

    def _extract_metadata(self, abs_path: str, rel_path: str) -> FileInfo | None:
        """Return a ``FileInfo`` for a single file, or ``None`` if unreadable."""
        try:
            stat = os.stat(abs_path)
        except OSError:
            return None

        size_bytes = stat.st_size
        last_modified = stat.st_mtime
        ext = Path(rel_path).suffix.lower()
        language = _EXT_TO_LANG.get(ext, "text")

        # Estimate tokens before reading (cheap).
        estimated_tokens = max(1, size_bytes // _CHARS_PER_TOKEN)

        # Read content -- but only if the file is reasonably small.
        # Very large files get metadata only.
        content_preview = ""
        imports: list[str] = []
        exports: list[str] = []

        try:
            # Read up to 512 KB to avoid huge files.
            with open(abs_path, encoding="utf-8", errors="replace") as fh:
                content = fh.read(524_288)

            content_preview = content[:500]
            imports, exports = _extract_symbols(content, language)
            # Refine token estimate from actual byte count.
            estimated_tokens = max(1, len(content) // _CHARS_PER_TOKEN)

        except (OSError, PermissionError):
            pass

        return FileInfo(
            path=rel_path,
            size_bytes=size_bytes,
            language=language,
            last_modified=last_modified,
            estimated_tokens=estimated_tokens,
            imports=imports,
            exports=exports,
            content_preview=content_preview,
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_relevant(
        self,
        problem: str,
        token_budget: int = 50_000,
    ) -> list[str]:
        """Select the most relevant files for *problem* within *token_budget*.

        Uses a weighted multi-signal relevance score:
          - Keyword overlap with the problem description.
          - Import/export graph affinity.
          - File recency (recently modified = more likely relevant).
          - File-type hints derived from the problem description.

        Files are greedily packed into the budget in descending score order.
        A file is skipped (not excluded) if it alone exceeds the remaining
        budget; smaller files are still considered.

        Returns
        -------
        list[str]
            Relative file paths, ordered from most to least relevant.
        """
        if not self.files:
            return []

        keywords = _tokenise(problem)
        type_boosts = _type_boost_patterns(problem)

        scored: list[tuple[float, str]] = []
        for rel_path, info in self.files.items():
            if info.estimated_tokens > self.max_file_tokens:
                continue  # never include files that would eat the whole budget
            score = self._relevance_score(rel_path, info, keywords, type_boosts)
            scored.append((score, rel_path))

        scored.sort(reverse=True)

        selected: list[str] = []
        tokens_used = 0
        for score, rel_path in scored:
            info = self.files[rel_path]
            file_tokens = info.estimated_tokens
            if tokens_used + file_tokens > token_budget:
                # Skip this file but keep trying smaller ones.
                continue
            selected.append(rel_path)
            tokens_used += file_tokens

        return selected

    def _relevance_score(
        self,
        rel_path: str,
        info: FileInfo,
        keywords: set[str],
        type_boosts: list[str],
    ) -> float:
        """Compute a [0, 1] relevance score for a single file.

        The four signals are normalised independently and combined with
        fixed weights that sum to 1.
        """
        # --- Signal 1: keyword overlap --------------------------------
        # Check path + content_preview + exports/imports for problem keywords.
        text = " ".join(
            [rel_path.lower(), info.content_preview.lower()]
            + [i.lower() for i in info.imports]
            + [e.lower() for e in info.exports]
        )
        text_tokens = _tokenise(text)
        if keywords:
            overlap = len(keywords & text_tokens)
            keyword_score = min(1.0, overlap / len(keywords))
        else:
            keyword_score = 0.0

        # --- Signal 2: import graph affinity --------------------------
        # If a keyword matches an import or export symbol exactly, boost.
        symbol_hits = sum(
            1
            for sym in info.imports + info.exports
            if sym.lower() in keywords
        )
        import_score = min(1.0, symbol_hits / max(1, len(keywords)))

        # --- Signal 3: recency ----------------------------------------
        # Normalise to [0, 1]: newest file gets 1.0, oldest gets ~0.
        # We use an exponential decay with a 30-day half-life.
        age_days = (time.time() - info.last_modified) / 86_400
        recency_score = math.exp(-age_days * math.log(2) / 30)

        # --- Signal 4: file-type hint ---------------------------------
        path_lower = rel_path.lower()
        type_score = 1.0 if any(pat in path_lower for pat in type_boosts) else 0.0

        # Weighted combination.  Keyword overlap dominates; recency is a
        # gentle tiebreaker.
        score = (
            0.55 * keyword_score
            + 0.20 * import_score
            + 0.10 * recency_score
            + 0.15 * type_score
        )
        return score

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def build_context(self, files: list[str]) -> str:
        """Build a prompt-ready context string from selected file paths.

        Each file is rendered as a fenced markdown block with its relative
        path as the header.  Files that cannot be read are silently skipped.

        Parameters
        ----------
        files:
            Relative file paths (as returned by ``select_relevant``).

        Returns
        -------
        str
            Concatenated context ready to be inserted into an LLM prompt.
        """
        parts: list[str] = []
        for rel_path in files:
            abs_path = os.path.join(self.root, rel_path)
            try:
                with open(abs_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except (OSError, PermissionError):
                continue

            info = self.files.get(rel_path)
            lang = info.language if info else "text"
            parts.append(f"### {rel_path}\n```{lang}\n{content}\n```")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a short human-readable summary of the index."""
        total_tokens = sum(i.estimated_tokens for i in self.files.values())
        langs: dict[str, int] = {}
        for info in self.files.values():
            langs[info.language] = langs.get(info.language, 0) + 1
        lang_str = ", ".join(f"{l}:{n}" for l, n in sorted(langs.items(), key=lambda x: -x[1])[:5])
        return (
            f"CodebaseIndex: {len(self.files)} files, "
            f"~{total_tokens:,} tokens estimated, "
            f"languages=[{lang_str}], "
            f"indexed in {self._index_time:.2f}s"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _tokenise(text: str) -> set[str]:
    """Split text into a set of lowercase alphanumeric tokens (length >= 2)."""
    return {w for w in re.split(r"[^a-z0-9_]+", text.lower()) if len(w) >= 2}


def _extract_symbols(content: str, language: str) -> tuple[list[str], list[str]]:
    """Extract import and export (top-level definition) names from source.

    Returns ``(imports, exports)`` as lists of string identifiers.
    Only lightweight regex-based extraction -- no AST parsing.
    """
    imports: list[str] = []
    exports: list[str] = []

    if language == "python":
        # import foo, import foo.bar, from foo import bar
        for m in re.finditer(
            r"^(?:import\s+([\w.]+)|from\s+([\w.]+)\s+import\s+([\w, *]+))",
            content,
            re.MULTILINE,
        ):
            if m.group(1):
                imports.append(m.group(1).split(".")[0])
            elif m.group(2):
                imports.append(m.group(2).split(".")[0])
                for sym in re.split(r"\s*,\s*", m.group(3)):
                    sym = sym.strip()
                    if sym and sym != "*":
                        imports.append(sym)

        # def foo / class Foo / FOO = ... (module-level)
        for m in re.finditer(
            r"^(?:def|class|async def)\s+(\w+)|^(\w+)\s*=",
            content,
            re.MULTILINE,
        ):
            name = m.group(1) or m.group(2)
            if name and not name.startswith("_"):
                exports.append(name)

    elif language in ("javascript", "typescript"):
        # import ... from 'module'
        for m in re.finditer(r"from\s+['\"]([^'\"]+)['\"]", content):
            imports.append(m.group(1).split("/")[-1])
        # export default / export function / export const
        for m in re.finditer(
            r"export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)",
            content,
        ):
            exports.append(m.group(1))

    elif language == "go":
        for m in re.finditer(r"\"([^\"]+)\"", content[:2000]):
            # heuristic: short strings in the header are likely imports
            pkg = m.group(1).split("/")[-1]
            imports.append(pkg)
        for m in re.finditer(r"^func\s+(\w+)", content, re.MULTILINE):
            exports.append(m.group(1))

    elif language == "rust":
        for m in re.finditer(r"use\s+([\w:]+)", content):
            imports.append(m.group(1).split("::")[-1])
        for m in re.finditer(r"^pub\s+fn\s+(\w+)", content, re.MULTILINE):
            exports.append(m.group(1))

    # Deduplicate while preserving order.
    imports = list(dict.fromkeys(imports))
    exports = list(dict.fromkeys(exports))
    return imports, exports


def _type_boost_patterns(problem: str) -> list[str]:
    """Return path substrings to boost based on type hints in the problem."""
    problem_lower = problem.lower()
    patterns: list[str] = []
    for keywords, path_pats in _TYPE_HINTS:
        if any(kw in problem_lower for kw in keywords):
            patterns.extend(path_pats)
    return patterns
