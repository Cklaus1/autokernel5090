#!/usr/bin/env python3
"""Lint test: W8_silent_none_lint_test

Prevents a recurrence of the Qwen3 fused-norm v1 silent-None bug
(.kernel vs .backend, +19% throughput lost for weeks) and the 4 other
CRITICAL patterns identified by the W6 silent-None sweep.

Rules
-----
Rule 1: getattr(x, "<attr>", None) must be followed within 10 lines by
        either a WARNING log on the None branch OR a "# SAFE: <reason>"
        comment explaining why None is expected/safe.

Rule 2: `except AttributeError: pass` in plugin/dispatch code fails the
        test. The bare pass silently swallows a missing attribute; the
        correct fix is to log at WARNING level (per KILL_PATTERNS §P1).

Rule 3: Every `os.environ.get(ENV, "0") == "1"` env gate must be matched
        by at least one `export ENV=` line in any launch_*.sh.  Dormant
        env gates waste review cycles and hide dead code paths.

Scope
-----
- patches/*.py  (includes subdirs)
- fusen_kv/*.py  (includes subdirs, excludes eval_perplexity.py which
  uses getattr for model-structure introspection, not dispatch)
- autokernel_v2/plugins/*.py
- kernels/*.py  (wrapper .py only, not .cu)

Known exceptions (getattr-None in these contexts are intentionally safe)
------------------------------------------------------------------------
- `patches/fusencache/`  — upstream vLLM copy; not our code
- `patches/flashinfer-upstream-proposal.py`  — doc/proposal file, not live code
- `fusen_kv/eval_perplexity.py`  — eval-only introspection; not a dispatch hot-path
- `kernels/test_*.py`  — test helpers; not plugin code
- `patches/wire_fused_norm_fp4.py`  — _tls cache pattern; SAFE (thread-local cache,
  None triggers allocation below, covered by try/except allocation logic)
- `patches/swa_gemma4_plugin.py:287`  — layer_id string fallback chain; SAFE (cosmetic
  label, not a feature dispatch decision)
- `patches/vllm_async_scheduling_fix.py`  — _forward_done_event optional async
  machinery; has explicit not-None guard immediately after each getattr

Already-fixed files (W7_silent_none_5_fixes) that MUST PASS
-------------------------------------------------------------
The 5 CRITICAL fixes from silent_none_fixes_20260419.md added WARNING logs
to the None-branch of each pattern; those files should produce zero
violations against Rule 1.

Usage
-----
    python3 -m pytest tests/test_no_silent_none_dispatch.py -v
    python3 tests/test_no_silent_none_dispatch.py  # standalone
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
from typing import NamedTuple

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Scan scope configuration
# ---------------------------------------------------------------------------

SCAN_DIRS = [
    "patches",
    "fusen_kv",
    "autokernel_v2/plugins",
    "kernels",
]

# Files/directories excluded from Rule 1 + Rule 2 (not our dispatch code)
EXCLUDED_PATHS = {
    # Upstream vLLM copies inside patches/fusencache/ — not our plugin code
    "patches/fusencache",
    # Proposal/doc file, not live dispatch code
    "patches/flashinfer-upstream-proposal.py",
    # Eval-only model introspection; no dispatch hot-path
    "fusen_kv/eval_perplexity.py",
    # Test helpers in kernels/; not plugin dispatch
    "kernels/test_fp8_decode.py",
    "kernels/test_fused_norm_fp4.py",
}

# Per-file, per-line exclusions for Rule 1 (getattr-None) that are genuinely
# safe but cannot be trivially annotated with # SAFE because they are in
# existing upstream-ish code or have multi-line context we cannot modify here.
#
# Format: set of (relative_path_str, line_number) pairs, where line_number is
# the 1-based line of the getattr( call itself.
SAFE_GETATTR_LOCATIONS: set[tuple[str, int]] = {
    # swa_gemma4_plugin.py:287 — layer_id label, not a feature dispatch
    ("patches/swa_gemma4_plugin.py", 287),
    # vllm_async_scheduling_fix.py — both getattrs have immediate not-None guards
    ("patches/vllm_async_scheduling_fix.py", 281),
    ("patches/vllm_async_scheduling_fix.py", 291),
    # wire_fused_norm_fp4.py — _tls thread-local cache; None triggers alloc
    ("patches/wire_fused_norm_fp4.py", 172),
    ("patches/wire_fused_norm_fp4.py", 173),
    # t1b_piecewise_cudagraph_fix.py — shadow metadata default; benign per sweep
    ("patches/t1b_piecewise_cudagraph_fix.py", 136),
    # fusen_kv/backend.py — num_gpu_blocks safe default 256 per sweep §SUSPECT
    ("fusen_kv/backend.py", 508),
    # fusen_kv/plugin.py — infrastructure config drift; info-level per sweep
    ("fusen_kv/plugin.py", 130),
    ("fusen_kv/plugin.py", 268),
}

# Rule 3: env gates with "0" default that are intentionally never set by
# any launcher (documented dormant code; not a bug).
KNOWN_DORMANT_ENV_GATES: set[str] = {
    # Dormant old code branch per sweep §SUSPECT
    "AUTOKERNEL_FUSED_RESHAPE_SCALES",
    # Selective caching dormant; parallel register() entry-point works
    "FUSEN_SELECTIVE",
    # Debug/diagnostics — dev-only; appropriate to not set in production launchers
    "FUSEN_DEBUG",
    "FUSEN_SYNC",
    # Auto-path for dynamic chunked prefill — dormant; register() entry works fine
    # Ref: silent_none_sweep §env-var table "register() entry still works"
    "AUTOKERNEL_DYNAMIC_CHUNK_AUTO",
    # T2-N v3 silu epilogue — experimental, off by default until bench banks the win
    # Ref: patches/fused_shuffle_quant_wrapper.py:119 "Off by default until bench"
    "AUTOKERNEL_T2N_SILU_EPILOGUE",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_excluded(rel: pathlib.Path) -> bool:
    """Return True if rel is under any excluded path."""
    rel_str = str(rel)
    for exc in EXCLUDED_PATHS:
        if rel_str == exc or rel_str.startswith(exc + "/") or rel_str.startswith(exc + "\\"):
            return True
    return False


def _scan_python_files(dir_name: str) -> list[pathlib.Path]:
    """Return all .py files under ROOT/dir_name, excluding EXCLUDED_PATHS."""
    base = ROOT / dir_name
    if not base.exists():
        return []
    files = []
    for p in sorted(base.rglob("*.py")):
        rel = p.relative_to(ROOT)
        if not _is_excluded(rel):
            files.append(p)
    return files


def _all_scanned_files() -> list[pathlib.Path]:
    result = []
    for d in SCAN_DIRS:
        result.extend(_scan_python_files(d))
    return result


class Violation(NamedTuple):
    rule: str
    file: pathlib.Path
    line: int
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.file.relative_to(ROOT)}:{self.line} — {self.message}"


# ---------------------------------------------------------------------------
# Rule 1: getattr(x, "attr", None) must be followed by WARNING or # SAFE
# ---------------------------------------------------------------------------

_RE_GETATTR_NONE = re.compile(
    r'getattr\s*\([^,]+,\s*"([^"]+)"\s*,\s*None\s*\)'
)
# Matches a WARNING-level log call (logging.warning / logger.warning / warnings.warn)
# Case-sensitive to avoid matching "# WARNING" comments or debug-level calls.
_RE_WARNING_LOG = re.compile(
    r'(?:logger|logging|log|_log)\.warning\s*\('
    r'|logging\.getLogger[^)]*\)\.warning\s*\('
    r'|warnings\.warn\s*\('
    r'|_warn_mod\.warn\s*\('
    r'|RuntimeWarning'
)
_RE_SAFE_COMMENT = re.compile(r'#\s*SAFE\s*:', re.IGNORECASE)

_WINDOW = 10  # lines to look ahead after the getattr line


def _check_rule1(path: pathlib.Path) -> list[Violation]:
    """
    For every getattr(x, "attr", None) in path, check that within the next
    WINDOW lines there is either a WARNING log call or a # SAFE: comment.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = source.splitlines()
    rel = path.relative_to(ROOT)
    rel_str = str(rel)
    violations = []

    for m in _RE_GETATTR_NONE.finditer(source):
        # Compute 1-based line number of the match
        lineno = source.count("\n", 0, m.start()) + 1

        # Check SAFE_GETATTR_LOCATIONS exclusion list
        if (rel_str, lineno) in SAFE_GETATTR_LOCATIONS:
            continue

        # Look at lines [lineno-1 .. lineno-1+WINDOW] (0-based indexing)
        start_idx = lineno - 1  # line of getattr itself
        end_idx = min(start_idx + _WINDOW + 1, len(lines))
        window_text = "\n".join(lines[start_idx:end_idx])

        has_warning = bool(_RE_WARNING_LOG.search(window_text))
        has_safe = bool(_RE_SAFE_COMMENT.search(window_text))

        if not has_warning and not has_safe:
            attr_name = m.group(1)
            violations.append(Violation(
                rule="Rule1",
                file=path,
                line=lineno,
                message=(
                    f'getattr(x, "{attr_name}", None) with no WARNING log or '
                    f'"# SAFE:" comment within {_WINDOW} lines — '
                    f"silent-None dispatch (see KILL_PATTERNS §P1)"
                ),
            ))

    return violations


# ---------------------------------------------------------------------------
# Rule 2: except AttributeError: pass is forbidden in plugin/dispatch code
# ---------------------------------------------------------------------------

def _check_rule2(path: pathlib.Path) -> list[Violation]:
    """
    Walk the AST looking for ExceptHandler nodes where:
      - type is AttributeError (or a bare `except:` with pass body)
      - body consists solely of `pass`
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return []

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Check for AttributeError specifically
        is_attr_error = (
            node.type is not None
            and isinstance(node.type, ast.Name)
            and node.type.id == "AttributeError"
        )
        if not is_attr_error:
            continue

        # Check if body is only `pass`
        body = node.body
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            violations.append(Violation(
                rule="Rule2",
                file=path,
                line=node.lineno,
                message=(
                    "bare `except AttributeError: pass` silently swallows a "
                    "missing attribute — add WARNING log (see KILL_PATTERNS §P1, "
                    "silent_none_sweep_20260419.md §C4)"
                ),
            ))

    return violations


# ---------------------------------------------------------------------------
# Rule 3: os.environ.get(ENV, "0") == "1" gates must appear in a launcher
# ---------------------------------------------------------------------------

# Matches: os.environ.get("NAME", "0") == "1"
# or:      os.environ.get("NAME", "0") != "0"   (equivalent gate)
_RE_ENV_GATE = re.compile(
    r'os\.environ\.get\(\s*["\']([A-Z_][A-Z0-9_]*)["\'][^)]*"0"\s*\)\s*==\s*"1"'
)

# Also catch the != "0" form (same semantics as == "1")
_RE_ENV_GATE_NEQ = re.compile(
    r'os\.environ\.get\(\s*["\']([A-Z_][A-Z0-9_]*)["\'][^)]*"0"\s*\)\s*!=\s*"0"'
)


def _collect_env_gates() -> dict[str, list[tuple[pathlib.Path, int]]]:
    """
    Scan all scoped files for env-gate patterns.
    Returns {VAR_NAME: [(file, lineno), ...]}
    """
    gates: dict[str, list[tuple[pathlib.Path, int]]] = {}
    for path in _all_scanned_files():
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pat in (_RE_ENV_GATE, _RE_ENV_GATE_NEQ):
            for m in pat.finditer(source):
                var = m.group(1)
                lineno = source.count("\n", 0, m.start()) + 1
                gates.setdefault(var, []).append((path, lineno))
    return gates


def _collect_launcher_exports() -> set[str]:
    """
    Scan all launch_*.sh files in ROOT for lines matching `export VAR=`.
    Returns set of exported variable names.
    """
    exported: set[str] = set()
    _re_export = re.compile(r'\bexport\s+([A-Z_][A-Z0-9_]*)(?:=|\s*$)')
    for sh in sorted(ROOT.glob("launch_*.sh")):
        try:
            text = sh.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in _re_export.finditer(text):
            exported.add(m.group(1))
    return exported


def _check_rule3() -> list[Violation]:
    """
    For each env-gate variable, check that at least one launch_*.sh exports it.
    Returns a Violation per variable that has no launcher coverage.
    """
    gates = _collect_env_gates()
    exported = _collect_launcher_exports()
    violations = []

    for var, locations in sorted(gates.items()):
        if var in KNOWN_DORMANT_ENV_GATES:
            continue  # explicitly documented as dormant / intentionally unset
        if var in exported:
            continue
        # Report at the first occurrence
        file0, line0 = locations[0]
        all_sites = ", ".join(
            f"{p.relative_to(ROOT)}:{ln}" for p, ln in locations
        )
        violations.append(Violation(
            rule="Rule3",
            file=file0,
            line=line0,
            message=(
                f'env gate {var!r} (default "0") is never set by any '
                f"launch_*.sh — dormant gate (see silent_none_sweep §env-var table). "
                f"All sites: {all_sites}"
            ),
        ))

    return violations


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

def test_no_silent_none_dispatch():
    """Rule 1 + Rule 2: no unguarded getattr-None and no bare except AttributeError: pass."""
    all_violations = []

    for path in _all_scanned_files():
        all_violations.extend(_check_rule1(path))
        all_violations.extend(_check_rule2(path))

    if all_violations:
        lines = ["\n\nSilent-None dispatch violations found:\n"]
        for v in all_violations:
            lines.append(f"  {v}")
        lines.append(
            "\nFix: add WARNING log in None branch OR add '# SAFE: <reason>' "
            "comment within 10 lines of the getattr() call."
        )
        pytest.fail("\n".join(lines))


def test_env_gates_referenced_in_launchers():
    """Rule 3: every == '1' env gate must be exported by at least one launch_*.sh."""
    violations = _check_rule3()

    if violations:
        lines = ["\n\nDormant env-gate violations found:\n"]
        for v in violations:
            lines.append(f"  {v}")
        lines.append(
            "\nFix: either add 'export VAR=...' to the relevant launch_*.sh, "
            "or add the var name to KNOWN_DORMANT_ENV_GATES if it is intentionally "
            "never set (and document why)."
        )
        pytest.fail("\n".join(lines))


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

def _run_standalone() -> int:
    """Run all rules and print results. Returns exit code."""
    print("W8 silent-None lint test — scanning…\n")

    files = _all_scanned_files()
    print(f"Scanned {len(files)} Python files across {SCAN_DIRS}\n")

    all_violations: list[Violation] = []
    for path in files:
        all_violations.extend(_check_rule1(path))
        all_violations.extend(_check_rule2(path))
    all_violations.extend(_check_rule3())

    if not all_violations:
        print("PASS — no violations found.")
        return 0

    print(f"FAIL — {len(all_violations)} violation(s):\n")
    for v in all_violations:
        print(f"  {v}")
    return 1


if __name__ == "__main__":
    sys.exit(_run_standalone())
