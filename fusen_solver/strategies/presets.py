"""Built-in strategy templates and presets.

Each strategy defines a different approach the LLM takes when solving
a problem. Presets group strategies by problem type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fusen_solver.core.interfaces import Strategy


# ---------------------------------------------------------------------------
# Collaborative roles for multi-round solving
# ---------------------------------------------------------------------------


@dataclass
class AgentRole:
    """Specialized role for collaborative solving."""

    name: str  # "analyst", "coder", "tester", "reviewer"
    prompt_template: str
    receives_context: bool = True  # sees previous round results


COLLABORATIVE_ROLES: dict[str, list[AgentRole]] = {
    "round_1": [
        AgentRole(
            "analyst",
            "Analyze the root cause of this problem. Be specific about WHERE and WHY.",
        ),
        AgentRole(
            "researcher",
            "Research the codebase context. What patterns are used? What constraints exist?",
        ),
        AgentRole(
            "test_writer",
            "Write test cases that would catch this bug/verify this feature.",
        ),
    ],
    "round_2": [
        AgentRole(
            "coder_a",
            "Using the analysis and tests from Round 1, write the solution.",
        ),
        AgentRole(
            "coder_b",
            "Write an alternative solution, different approach from coder_a.",
        ),
    ],
    "round_3": [
        AgentRole(
            "reviewer",
            "Review both solutions against the tests. Which is better and why?",
        ),
        AgentRole(
            "merger",
            "Take the best parts of both solutions. Write the final version.",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Strategy catalog: all built-in strategies
# ---------------------------------------------------------------------------

STRATEGY_CATALOG: dict[str, Strategy] = {
    "direct": Strategy(
        name="direct",
        prompt=(
            "You are a direct problem solver. Identify the root cause and produce "
            "the minimal, targeted fix. Do not refactor unrelated code. "
            "Return ONLY the corrected code with a brief explanation of the fix."
        ),
        tags=["fast", "minimal", "safe"],
    ),
    "alternative": Strategy(
        name="alternative",
        prompt=(
            "You are an algorithm expert. Propose a fundamentally different approach "
            "to solve this problem -- a different data structure, algorithm, or design "
            "pattern. Explain why your alternative is better, then provide the full "
            "implementation."
        ),
        temperature=0.9,
        tags=["creative", "exploratory"],
    ),
    "test_first": Strategy(
        name="test_first",
        prompt=(
            "You are a test-driven developer. First, write comprehensive tests that "
            "capture the expected behavior (including edge cases). Then write or fix "
            "the code to make all tests pass. Return both the tests and the implementation."
        ),
        tags=["thorough", "safe", "testable"],
    ),
    "decompose": Strategy(
        name="decompose",
        prompt=(
            "You are a systems thinker. Break this problem into 2-4 independent "
            "sub-problems. Solve each one separately, then combine the solutions. "
            "Clearly label each sub-problem and its solution."
        ),
        tags=["structured", "thorough"],
    ),
    "review": Strategy(
        name="review",
        prompt=(
            "You are a senior code reviewer. First, identify ALL issues in the code "
            "(bugs, performance, readability, edge cases, security). Rank them by "
            "severity. Then fix the top issues, explaining each change."
        ),
        tags=["thorough", "quality"],
    ),
    "research": Strategy(
        name="research",
        prompt=(
            "You are a technical researcher. Explain what is going wrong and WHY "
            "(root cause analysis). Propose 3 different fixes with trade-offs "
            "(correctness, performance, complexity). Recommend the best one and "
            "implement it."
        ),
        tags=["analytical", "thorough"],
    ),
    "rewrite": Strategy(
        name="rewrite",
        prompt=(
            "You are a clean-code advocate. Rewrite the problematic code from "
            "scratch with a focus on clarity, correctness, and maintainability. "
            "Preserve the external interface but redesign internals."
        ),
        temperature=0.8,
        tags=["creative", "quality"],
    ),
    "adversarial": Strategy(
        name="adversarial",
        prompt=(
            "You are a QA adversary. Think of every way this code can break: "
            "edge cases, concurrency, overflow, empty inputs, malicious inputs. "
            "Write a fix that handles ALL of them, with comments explaining each "
            "defensive measure."
        ),
        tags=["thorough", "safe", "defensive"],
    ),
    "prototype_then_refine": Strategy(
        name="prototype_then_refine",
        prompt=(
            "First write a quick, working prototype that solves the problem. "
            "Then refine it: improve error handling, add types, clean up naming, "
            "optimize hot paths. Show both the prototype and the refined version."
        ),
        tags=["iterative", "practical"],
    ),
    "incremental": Strategy(
        name="incremental",
        prompt=(
            "Make the smallest possible change that improves the code. Each change "
            "should be independently correct and testable. List changes in order "
            "of priority. Implement the top 3-5 changes."
        ),
        tags=["safe", "minimal", "incremental"],
    ),
    "profile_first": Strategy(
        name="profile_first",
        prompt=(
            "Before changing anything, analyze the performance characteristics: "
            "time complexity, space complexity, hot paths, cache behavior. "
            "Then make targeted optimizations backed by this analysis."
        ),
        tags=["analytical", "performance"],
    ),
    "security": Strategy(
        name="security",
        prompt=(
            "Audit this code for security issues: injection, auth bypass, data leaks, "
            "race conditions, path traversal, etc. Fix all vulnerabilities found. "
            "Explain each vulnerability and its CVSS-like severity."
        ),
        tags=["security", "thorough"],
    ),
}


# ---------------------------------------------------------------------------
# Presets: strategy groups by problem type
# ---------------------------------------------------------------------------

STRATEGY_PRESETS: dict[str, list[str]] = {
    "bug_fix": ["direct", "test_first", "adversarial", "research"],
    "feature": ["direct", "test_first", "decompose", "prototype_then_refine"],
    "refactor": ["incremental", "rewrite", "review", "decompose"],
    "architecture": ["decompose", "alternative", "research", "review"],
    "optimize": ["profile_first", "alternative", "direct", "review"],
    "test": ["test_first", "adversarial", "decompose", "review"],
    "review": ["security", "review", "adversarial", "research"],
    "explore": list(STRATEGY_CATALOG.keys()),
}


def get_strategy(name: str) -> Strategy:
    """Get a strategy by name, or create a pass-through strategy for unknown names."""
    if name in STRATEGY_CATALOG:
        return STRATEGY_CATALOG[name]
    # Allow custom strategies as plain prompts
    return Strategy(name=name, prompt=name)
