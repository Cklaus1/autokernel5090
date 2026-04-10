"""Tests for CodebaseIndex and PrefixManager.

All tests are self-contained: they create temporary directories with
synthetic files so no real repository is required.

Run with:
    python -m pytest fusen_solver/tests/test_codebase_index.py -v
"""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest

from fusen_solver.core.codebase_index import (
    CodebaseIndex,
    FileInfo,
    _extract_symbols,
    _tokenise,
    _type_boost_patterns,
)
from fusen_solver.prefix_manager import PrefixManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Write *files* (rel_path → content) under *tmp_path*, return the root."""
    for rel, content in files.items():
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(textwrap.dedent(content), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_basic(self):
        assert _tokenise("fix the bug in pagination") == {
            "fix", "the", "bug", "in", "pagination"
        }

    def test_filters_short(self):
        tokens = _tokenise("a b ok foo")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "ok" in tokens
        assert "foo" in tokens

    def test_case_insensitive(self):
        assert _tokenise("Fix THE Bug") == _tokenise("fix the bug")

    def test_punctuation_stripped(self):
        tokens = _tokenise("fix-the-bug_in pagination.py!")
        assert "fix" in tokens
        assert "bug" in tokens
        assert "pagination" in tokens
        assert "py" in tokens


class TestExtractSymbols:
    def test_python_imports(self):
        src = "import os\nimport sys\nfrom pathlib import Path, PurePath\n"
        imports, exports = _extract_symbols(src, "python")
        assert "os" in imports
        assert "sys" in imports
        assert "pathlib" in imports
        assert "Path" in imports
        assert "PurePath" in imports

    def test_python_exports(self):
        src = "def foo():\n    pass\nclass Bar:\n    pass\nBAZ = 1\n"
        imports, exports = _extract_symbols(src, "python")
        assert "foo" in exports
        assert "Bar" in exports
        assert "BAZ" in exports

    def test_python_private_not_exported(self):
        src = "def _internal(): pass\n"
        _, exports = _extract_symbols(src, "python")
        assert "_internal" not in exports

    def test_js_imports_and_exports(self):
        src = (
            "import { useState } from 'react';\n"
            "import utils from './utils';\n"
            "export function App() {}\n"
            "export const VERSION = '1.0';\n"
        )
        imports, exports = _extract_symbols(src, "javascript")
        assert "react" in imports
        assert "utils" in imports
        assert "App" in exports
        assert "VERSION" in exports

    def test_unknown_language_returns_empty(self):
        imports, exports = _extract_symbols("some text", "text")
        assert imports == []
        assert exports == []


class TestTypeBoostPatterns:
    def test_test_keyword(self):
        patterns = _type_boost_patterns("fix the failing tests")
        assert any("test" in p for p in patterns)

    def test_config_keyword(self):
        patterns = _type_boost_patterns("update configuration settings")
        assert any("config" in p for p in patterns)

    def test_no_match(self):
        patterns = _type_boost_patterns("sort the list of numbers")
        assert patterns == []


# ---------------------------------------------------------------------------
# Integration tests: CodebaseIndex
# ---------------------------------------------------------------------------


class TestCodebaseIndex:
    def test_index_walks_files(self, tmp_path):
        make_tree(tmp_path, {
            "main.py": "import os\ndef main(): pass\n",
            "utils/helper.py": "def helper(): pass\n",
            "README.md": "# Project\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        assert "main.py" in idx.files
        assert "utils/helper.py" in idx.files
        assert "README.md" in idx.files

    def test_skips_git_directory(self, tmp_path):
        make_tree(tmp_path, {
            ".git/HEAD": "ref: refs/heads/main\n",
            "app.py": "x = 1\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        keys = list(idx.files.keys())
        assert not any(".git" in k for k in keys)
        assert "app.py" in idx.files

    def test_skips_pycache(self, tmp_path):
        make_tree(tmp_path, {
            "__pycache__/foo.cpython-311.pyc": "\x00binary\x00",
            "foo.py": "x = 1\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        keys = list(idx.files.keys())
        assert not any("__pycache__" in k for k in keys)

    def test_language_detection(self, tmp_path):
        make_tree(tmp_path, {
            "a.py": "x=1\n",
            "b.ts": "const x: number = 1;\n",
            "c.go": "package main\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        assert idx.files["a.py"].language == "python"
        assert idx.files["b.ts"].language == "typescript"
        assert idx.files["c.go"].language == "go"

    def test_estimated_tokens_positive(self, tmp_path):
        make_tree(tmp_path, {"big.py": "x = 1\n" * 1000})
        idx = CodebaseIndex(str(tmp_path))
        assert idx.files["big.py"].estimated_tokens > 0

    def test_imports_extracted(self, tmp_path):
        make_tree(tmp_path, {
            "app.py": "import flask\nfrom pathlib import Path\n"
        })
        idx = CodebaseIndex(str(tmp_path))
        info = idx.files["app.py"]
        assert "flask" in info.imports
        assert "pathlib" in info.imports

    def test_summary_string(self, tmp_path):
        make_tree(tmp_path, {"a.py": "x=1\n"})
        idx = CodebaseIndex(str(tmp_path))
        s = idx.summary()
        assert "CodebaseIndex" in s
        assert "file" in s


class TestSelectRelevant:
    def test_returns_list_of_paths(self, tmp_path):
        make_tree(tmp_path, {
            "pagination.py": "def paginate(items, page, size): pass\n",
            "auth.py": "def login(): pass\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        selected = idx.select_relevant("fix the pagination bug", token_budget=10_000)
        assert isinstance(selected, list)
        assert all(isinstance(p, str) for p in selected)

    def test_keyword_match_ranks_higher(self, tmp_path):
        make_tree(tmp_path, {
            "pagination.py": "def paginate(items, page, size): pass\n",
            "auth.py": "def login(): pass\nclass Session: pass\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        selected = idx.select_relevant("fix pagination off-by-one", token_budget=10_000)
        # pagination.py should appear before auth.py
        assert "pagination.py" in selected
        assert selected.index("pagination.py") < selected.index("auth.py")

    def test_respects_token_budget(self, tmp_path):
        # Create a large file and a small file; budget only fits the small one.
        big_content = "x = 1\n" * 5000  # ~30K chars → ~7500 tokens
        small_content = "y = 2\n"        # tiny
        make_tree(tmp_path, {
            "big.py": big_content,
            "small.py": small_content,
        })
        idx = CodebaseIndex(str(tmp_path))
        # Budget = 100 tokens -- only small.py should fit.
        selected = idx.select_relevant("problem", token_budget=100)
        assert "small.py" in selected
        assert "big.py" not in selected

    def test_empty_codebase(self, tmp_path):
        idx = CodebaseIndex(str(tmp_path))
        assert idx.select_relevant("anything") == []

    def test_type_boost_test_files(self, tmp_path):
        make_tree(tmp_path, {
            "test_pagination.py": "def test_paginate(): pass\n",
            "pagination.py": "def paginate(): pass\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        selected = idx.select_relevant("fix the failing tests", token_budget=10_000)
        # test file should be selected (both should appear given the budget)
        assert "test_pagination.py" in selected

    def test_max_file_tokens_excludes_huge_file(self, tmp_path):
        huge = "x = 1\n" * 100_000  # ~600 K chars → ~150 K tokens >> default 20K max
        make_tree(tmp_path, {
            "huge.py": huge,
            "small.py": "y = 2\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        selected = idx.select_relevant("problem", token_budget=1_000_000)
        # huge.py exceeds max_file_tokens so it is never selected
        assert "huge.py" not in selected
        assert "small.py" in selected


class TestBuildContext:
    def test_produces_fenced_blocks(self, tmp_path):
        make_tree(tmp_path, {"app.py": "x = 1\n"})
        idx = CodebaseIndex(str(tmp_path))
        ctx = idx.build_context(["app.py"])
        assert "### app.py" in ctx
        assert "```python" in ctx
        assert "x = 1" in ctx

    def test_missing_file_silently_skipped(self, tmp_path):
        make_tree(tmp_path, {"app.py": "x = 1\n"})
        idx = CodebaseIndex(str(tmp_path))
        ctx = idx.build_context(["app.py", "nonexistent.py"])
        assert "### app.py" in ctx
        assert "nonexistent" not in ctx

    def test_multiple_files_separated(self, tmp_path):
        make_tree(tmp_path, {
            "a.py": "a = 1\n",
            "b.py": "b = 2\n",
        })
        idx = CodebaseIndex(str(tmp_path))
        ctx = idx.build_context(["a.py", "b.py"])
        assert "### a.py" in ctx
        assert "### b.py" in ctx


# ---------------------------------------------------------------------------
# Integration tests: PrefixManager
# ---------------------------------------------------------------------------


class TestPrefixManager:
    def test_load_codebase_smart_returns_string(self, tmp_path):
        make_tree(tmp_path, {
            "server.py": "import flask\ndef run(): pass\n",
            "tests/test_server.py": "def test_run(): pass\n",
        })
        pm = PrefixManager(system_prompt="You are a helpful assistant.")
        ctx = pm.load_codebase_smart(
            str(tmp_path),
            problem="fix the server crash",
            token_budget=10_000,
        )
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_load_codebase_smart_selects_relevant_file(self, tmp_path):
        make_tree(tmp_path, {
            "server.py": "import flask\ndef run(): pass\n",
            "unrelated.py": "def completely_different_thing(): pass\n",
        })
        pm = PrefixManager()
        ctx = pm.load_codebase_smart(
            str(tmp_path),
            problem="fix the flask server",
            token_budget=10_000,
        )
        assert "server.py" in ctx

    def test_load_codebase_returns_all_files(self, tmp_path):
        make_tree(tmp_path, {
            "a.py": "a = 1\n",
            "b.py": "b = 2\n",
        })
        pm = PrefixManager()
        ctx = pm.load_codebase(str(tmp_path))
        assert "a.py" in ctx
        assert "b.py" in ctx

    def test_load_codebase_extension_filter(self, tmp_path):
        make_tree(tmp_path, {
            "app.py": "x = 1\n",
            "style.css": "body { color: red; }\n",
        })
        pm = PrefixManager()
        ctx = pm.load_codebase(str(tmp_path), extensions=[".py"])
        assert "app.py" in ctx
        assert "style.css" not in ctx

    def test_load_codebase_auto_small_repo(self, tmp_path):
        make_tree(tmp_path, {"tiny.py": "x = 1\n"})
        pm = PrefixManager(token_budget=10_000)
        ctx = pm.load_codebase_auto(str(tmp_path), problem="fix x", token_budget=10_000)
        assert "tiny.py" in ctx

    def test_build_system_message_includes_prompt_and_context(self, tmp_path):
        pm = PrefixManager(system_prompt="Be helpful.")
        msg = pm.build_system_message(codebase_context="### a.py\n```python\nx=1\n```")
        assert "Be helpful." in msg
        assert "a.py" in msg

    def test_build_messages_structure(self, tmp_path):
        pm = PrefixManager(system_prompt="You are an expert.")
        msgs = pm.build_messages("What is wrong?", codebase_context="ctx")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is wrong?"

    def test_index_caching(self, tmp_path):
        make_tree(tmp_path, {"a.py": "x = 1\n"})
        pm = PrefixManager()
        idx1 = pm._get_index(str(tmp_path))
        idx2 = pm._get_index(str(tmp_path))
        assert idx1 is idx2  # same object returned (cached)

    def test_invalidate_index(self, tmp_path):
        make_tree(tmp_path, {"a.py": "x = 1\n"})
        pm = PrefixManager()
        idx1 = pm._get_index(str(tmp_path))
        pm.invalidate_index()
        idx2 = pm._get_index(str(tmp_path))
        assert idx1 is not idx2  # new index created after invalidation

    def test_index_summary(self, tmp_path):
        make_tree(tmp_path, {"a.py": "x = 1\n"})
        pm = PrefixManager()
        summary = pm.index_summary(str(tmp_path))
        assert "CodebaseIndex" in summary

    def test_no_system_prompt(self):
        pm = PrefixManager()
        msgs = pm.build_messages("hello")
        # When system prompt is empty and no context, only user message.
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
