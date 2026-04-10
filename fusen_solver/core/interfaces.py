"""Abstract interfaces that any platform can implement.

These define the contract between the Fusen solver and the outside world:
- Problem/Solution: universal data representations
- LLMBackend: adapter for any LLM provider
- PlatformPlugin: integration point for coding tools
- Strategy: a named approach with prompt template
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class Problem:
    """Universal problem representation.

    Works for any coding task: bug fix, feature, refactor, optimization, etc.
    The context dict maps filenames to their content (the relevant codebase).
    """

    description: str
    context: dict[str, str] = field(default_factory=dict)  # filename -> content
    problem_type: str = "auto"  # bug_fix, feature, refactor, architecture, optimize, test, review
    constraints: list[str] = field(default_factory=list)  # "must pass tests", "no new deps", etc.
    tests: list[str] = field(default_factory=list)  # test commands to validate
    language: str = "auto"
    priority: str = "quality"  # "quality", "speed", "balanced"
    solve_mode: str = "auto"  # "isolated", "collaborative", "auto"
    max_rounds: int = 3  # for collaborative mode
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    """Universal solution representation.

    The code dict maps filenames to modified content. The solver returns one
    Solution per strategy attempted, each scored independently.
    """

    code: dict[str, str] = field(default_factory=dict)  # filename -> content
    explanation: str = ""
    strategy_used: str = ""
    score: float = 0.0  # 0-1 overall score
    subscores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Strategy:
    """A named problem-solving approach with prompt template."""

    name: str
    prompt: str  # instructions for the LLM
    weight: float = 1.0  # selection weight (higher = more likely to be chosen)
    temperature: float = 0.7
    tags: list[str] = field(default_factory=list)  # e.g., ["creative", "safe", "thorough"]


class LLMBackend(ABC):
    """Adapter for any LLM provider.

    Implement this to add support for a new LLM backend (OpenAI, Anthropic,
    vLLM, Ollama, etc.). The solver calls generate() for each parallel agent.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> str:
        """Generate a completion from the given messages.

        Args:
            messages: Chat messages in OpenAI format [{"role": ..., "content": ...}].
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Optional stop sequences.

        Returns:
            The generated text content.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a completion token-by-token.

        Yields:
            Incremental text deltas.
        """
        ...

    @property
    @abstractmethod
    def supports_batch(self) -> bool:
        """Whether this backend can run N requests in parallel efficiently.

        True for vLLM (GPU batching), OpenAI (concurrent API calls), etc.
        False for single-threaded local models.
        """
        ...

    @property
    @abstractmethod
    def max_context(self) -> int:
        """Maximum context window in tokens."""
        ...

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        return self.__class__.__name__


class PlatformPlugin(ABC):
    """Interface for integrating with AI coding platforms.

    Implement this to hook the Fusen solver into Claude Code, Cursor,
    Copilot, Aider, VS Code extensions, or any other coding tool.
    """

    @abstractmethod
    async def on_problem(self, problem: Problem) -> Solution:
        """Receive a problem and return the best solution.

        The plugin runs the full parallel solve pipeline internally
        and returns a single merged/best solution.
        """
        ...

    @abstractmethod
    async def on_feedback(self, problem: Problem, solution: Solution, accepted: bool) -> None:
        """Record whether the user accepted or rejected a solution.

        This feeds the learning engine so future strategy selection improves.
        """
        ...
