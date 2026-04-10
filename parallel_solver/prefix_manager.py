"""Manages shared codebase prefix for vLLM prefix caching.

The key insight: when N agents share the same system prompt (containing the codebase),
vLLM's prefix caching stores those tokens ONCE on the GPU. Each agent only needs
unique KV cache for its strategy-specific continuation.

Example: 50K token codebase prefix + 9 agents each with 30K unique tokens
= 50K + 9*30K = 320K tokens (not 9*80K = 720K tokens).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class PrefixStats:
    """Statistics about prefix cache usage."""

    prefix_tokens: int = 0
    prefix_hash: str = ""
    warm: bool = False
    warm_latency_ms: float = 0.0
    agents_sharing: int = 0


STRATEGY_PROMPTS: dict[str, str] = {
    "direct": (
        "You are a direct problem solver. Identify the root cause and produce "
        "the minimal, targeted fix. Do not refactor unrelated code. "
        "Return ONLY the corrected code with a brief explanation of the fix."
    ),
    "alternative": (
        "You are an algorithm expert. Propose a fundamentally different approach "
        "to solve this problem -- a different data structure, algorithm, or design "
        "pattern. Explain why your alternative is better, then provide the full "
        "implementation."
    ),
    "test_first": (
        "You are a test-driven developer. First, write comprehensive tests that "
        "capture the expected behavior (including edge cases). Then write or fix "
        "the code to make all tests pass. Return both the tests and the implementation."
    ),
    "decompose": (
        "You are a systems thinker. Break this problem into 2-4 independent "
        "sub-problems. Solve each one separately, then combine the solutions. "
        "Clearly label each sub-problem and its solution."
    ),
    "review": (
        "You are a senior code reviewer. First, identify ALL issues in the code "
        "(bugs, performance, readability, edge cases, security). Rank them by "
        "severity. Then fix the top issues, explaining each change."
    ),
    "research": (
        "You are a technical researcher. Explain what is going wrong and WHY "
        "(root cause analysis). Propose 3 different fixes with trade-offs "
        "(correctness, performance, complexity). Recommend the best one and "
        "implement it."
    ),
    "rewrite": (
        "You are a clean-code advocate. Rewrite the problematic code from "
        "scratch with a focus on clarity, correctness, and maintainability. "
        "Preserve the external interface but redesign internals."
    ),
    "adversarial": (
        "You are a QA adversary. Think of every way this code can break: "
        "edge cases, concurrency, overflow, empty inputs, malicious inputs. "
        "Write a fix that handles ALL of them, with comments explaining each "
        "defensive measure."
    ),
}


class PrefixManager:
    """Manages shared codebase prefix for efficient vLLM prefix caching.

    All agents share the same system prompt containing the codebase. vLLM caches
    the KV entries for this common prefix, so it is computed only once regardless
    of how many agents are running in parallel.
    """

    def __init__(self, vllm_api: str, model: str):
        self.api = vllm_api.rstrip("/")
        self.model = model
        self.stats = PrefixStats()
        self._codebase_cache: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def warm_prefix(self, codebase: str) -> PrefixStats:
        """Send a short request with the codebase to warm the prefix cache.

        The first request forces vLLM to compute and cache KV entries for the
        codebase tokens. All subsequent requests reuse the cached prefix,
        skipping the expensive prefill phase for those tokens.
        """
        self._codebase_cache = codebase
        self.stats.prefix_hash = hashlib.sha256(codebase.encode()).hexdigest()[:16]

        messages = [
            {"role": "system", "content": self._system_prompt(codebase)},
            {"role": "user", "content": "Acknowledge that you have read the codebase. Reply with OK."},
        ]

        t0 = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 8,
                "temperature": 0.0,
            }
            async with session.post(
                f"{self.api}/v1/chat/completions", json=payload
            ) as resp:
                data = await resp.json()
                if "usage" in data:
                    self.stats.prefix_tokens = data["usage"].get("prompt_tokens", 0)

        self.stats.warm_latency_ms = (time.perf_counter() - t0) * 1000
        self.stats.warm = True
        logger.info(
            "Prefix warmed: %d tokens, %.0f ms, hash=%s",
            self.stats.prefix_tokens,
            self.stats.warm_latency_ms,
            self.stats.prefix_hash,
        )
        return self.stats

    def build_context(
        self,
        codebase: str,
        problem: str,
        strategy: str,
        *,
        extra_context: str = "",
    ) -> list[dict[str, str]]:
        """Build a chat message list with shared prefix + unique strategy.

        The system message (codebase) is identical across all agents, enabling
        prefix cache hits. The user message differs per strategy so each agent
        explores a distinct approach.
        """
        strategy_instruction = STRATEGY_PROMPTS.get(strategy, strategy)

        user_content = f"## Problem\n\n{problem}\n\n## Your Approach\n\n{strategy_instruction}"
        if extra_context:
            user_content += f"\n\n## Additional Context\n\n{extra_context}"
        user_content += (
            "\n\n## Instructions\n\n"
            "Write the complete solution. Include ALL modified code -- do not use "
            "placeholders or ellipsis. Explain your reasoning briefly before the code."
        )

        return [
            {"role": "system", "content": self._system_prompt(codebase)},
            {"role": "user", "content": user_content},
        ]

    def build_merge_context(
        self,
        codebase: str,
        problem: str,
        solutions: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build context for merging insights from multiple solutions."""
        solutions_text = ""
        for i, sol in enumerate(solutions, 1):
            solutions_text += (
                f"### Solution {i} (strategy: {sol['strategy']}, "
                f"score: {sol.get('score', 'N/A')})\n\n"
                f"{sol['content']}\n\n---\n\n"
            )

        return [
            {"role": "system", "content": self._system_prompt(codebase)},
            {
                "role": "user",
                "content": (
                    f"## Problem\n\n{problem}\n\n"
                    f"## Solutions from parallel agents\n\n{solutions_text}\n\n"
                    "## Your Task\n\n"
                    "Analyze all solutions above. Take the best ideas from each. "
                    "Produce a single, final solution that combines the strongest "
                    "aspects. Explain which parts you took from which solution and why."
                ),
            },
        ]

    @staticmethod
    def load_codebase(path: str | Path, max_tokens_approx: int = 50_000) -> str:
        """Load a codebase directory into a single string for the prefix.

        Reads Python files sorted by modification time (newest first).
        Stops when approximate token budget is reached (~4 chars/token).
        """
        root = Path(path)
        if root.is_file():
            return root.read_text(errors="replace")

        files: list[Path] = sorted(
            root.rglob("*.py"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        char_budget = max_tokens_approx * 4
        parts: list[str] = []
        total_chars = 0

        for f in files:
            # Skip typical non-essential paths
            rel = f.relative_to(root)
            skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
            if any(part in skip_dirs for part in rel.parts):
                continue

            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue

            header = f"# === {rel} ===\n"
            chunk = header + text + "\n\n"
            if total_chars + len(chunk) > char_budget:
                # Include partial if we have room for at least the header
                remaining = char_budget - total_chars
                if remaining > len(header) + 200:
                    parts.append(chunk[:remaining] + "\n# ... truncated ...\n")
                break
            parts.append(chunk)
            total_chars += len(chunk)

        return "".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _system_prompt(codebase: str) -> str:
        return (
            "You are an expert software engineer working on a coding problem. "
            "Below is the full codebase you are working with. Study it carefully "
            "before responding.\n\n"
            "## Codebase\n\n"
            f"```\n{codebase}\n```"
        )
