"""
AutoKernel v2 Plugin Registry -- Discoverable, data-driven, self-verifying optimization plugins.

Every optimization is a plugin that:
  - Decides if it applies based on profile + GPU data (not guesswork)
  - Auto-configures itself from profiling data
  - Measures its own impact (speedup, quality delta)
  - Can roll itself back if it causes regression
  - Declares which other plugins it compounds/conflicts with

Usage:
    from autokernel_v2.plugin_registry import PluginRegistry
    from autokernel_v2.plugins.disable_inductor import DisableInductorPlugin

    registry = PluginRegistry()
    registry.register(DisableInductorPlugin())

    results = registry.apply_best(profile, gpu, max_rounds=5)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .types import GPUInfo, ProfileResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin result
# ---------------------------------------------------------------------------

@dataclass
class PluginResult:
    """Outcome of applying a single plugin."""

    applied: bool
    speedup: float  # measured, not estimated
    quality_delta: float  # 0.0 = no change, -0.1 = 10% worse
    rollback_fn: Callable[[], None]  # undo if needed
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_improvement(self) -> bool:
        return self.applied and self.speedup > 1.0 and self.quality_delta >= -0.01


# ---------------------------------------------------------------------------
# Abstract plugin interface
# ---------------------------------------------------------------------------

class OptimizationPlugin(ABC):
    """Every optimization is a plugin with these requirements."""

    @abstractmethod
    def name(self) -> str:
        """Unique human-readable identifier for this plugin."""
        ...

    @abstractmethod
    def version(self) -> str:
        """Semver string -- bump when behavior changes."""
        ...

    @abstractmethod
    def applies_to(self, profile: ProfileResult, gpu_info: GPUInfo) -> bool:
        """Data-driven: does this help for this model+GPU combination?"""
        ...

    @abstractmethod
    def configure(self, profile: ProfileResult) -> dict[str, Any]:
        """Data-driven: sweep/auto-select optimal config from profile data."""
        ...

    @abstractmethod
    def expected_impact(self, profile: ProfileResult) -> float:
        """Estimated speedup multiplier before applying (e.g. 2.0 = 2x)."""
        ...

    @abstractmethod
    def apply(self, config: dict[str, Any]) -> None:
        """Apply the optimization with the given configuration."""
        ...

    @abstractmethod
    def verify(self) -> tuple[float, float]:
        """
        Self-verification after apply().

        Returns:
            (speedup, quality_delta) where speedup > 1.0 means faster and
            quality_delta >= 0.0 means no regression.
        """
        ...

    @abstractmethod
    def rollback(self) -> None:
        """Undo the optimization, restoring previous state."""
        ...

    def compounds_with(self) -> list[str]:
        """Names of other plugins that stack positively with this one."""
        return []

    def conflicts_with(self) -> list[str]:
        """Names of other plugins that are mutually exclusive."""
        return []

    def __repr__(self) -> str:
        return f"<Plugin {self.name()} v{self.version()}>"


# ---------------------------------------------------------------------------
# Plugin Registry
# ---------------------------------------------------------------------------

class PluginRegistry:
    """Discovers, manages, and orchestrates optimization plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, OptimizationPlugin] = {}
        self._applied: list[str] = []  # names of currently-applied plugins
        self._results_log: list[PluginResult] = []

    # -- Registration -------------------------------------------------------

    def register(self, plugin: OptimizationPlugin) -> None:
        """Register a plugin. Overwrites any existing plugin with same name."""
        name = plugin.name()
        if name in self._plugins:
            logger.info("Replacing existing plugin %s", name)
        self._plugins[name] = plugin
        logger.info("Registered plugin: %s v%s", name, plugin.version())

    def unregister(self, name: str) -> None:
        """Remove a plugin by name."""
        self._plugins.pop(name, None)

    @property
    def registered(self) -> list[OptimizationPlugin]:
        """All registered plugins, sorted by name."""
        return sorted(self._plugins.values(), key=lambda p: p.name())

    # -- Discovery ----------------------------------------------------------

    def discover_applicable(
        self,
        profile: ProfileResult,
        gpu: GPUInfo,
    ) -> list[OptimizationPlugin]:
        """
        Return plugins that apply to this model+GPU, sorted by expected impact
        (highest first). Excludes plugins that conflict with already-applied ones.
        """
        applicable: list[OptimizationPlugin] = []
        conflict_set = self._build_conflict_set()

        for plugin in self._plugins.values():
            pname = plugin.name()

            # Skip already applied
            if pname in self._applied:
                continue

            # Skip if conflicts with something already applied
            if pname in conflict_set:
                logger.debug("Skipping %s: conflicts with applied plugin", pname)
                continue

            try:
                if plugin.applies_to(profile, gpu):
                    applicable.append(plugin)
            except Exception:
                logger.warning("Plugin %s raised in applies_to, skipping", pname, exc_info=True)

        # Sort by expected impact descending
        applicable.sort(key=lambda p: p.expected_impact(profile), reverse=True)
        return applicable

    # -- Orchestration ------------------------------------------------------

    def apply_best(
        self,
        profile: ProfileResult,
        gpu: GPUInfo,
        max_rounds: int = 5,
    ) -> list[PluginResult]:
        """
        Iteratively apply the best applicable plugin up to max_rounds times.

        Each round:
          1. Discover applicable plugins
          2. Pick the one with highest expected impact
          3. Configure, apply, verify
          4. Keep if improved, rollback otherwise
          5. Update profile for next round

        Returns list of PluginResults (one per round attempted).
        """
        results: list[PluginResult] = []

        for round_num in range(1, max_rounds + 1):
            candidates = self.discover_applicable(profile, gpu)
            if not candidates:
                logger.info("Round %d: no applicable plugins remaining", round_num)
                break

            plugin = candidates[0]
            logger.info(
                "Round %d: trying %s (expected %.2fx)",
                round_num, plugin.name(), plugin.expected_impact(profile),
            )

            result = self._try_plugin(plugin, profile)
            results.append(result)
            self._results_log.append(result)

            if result.is_improvement:
                self._applied.append(plugin.name())
                logger.info(
                    "Round %d: APPLIED %s -- %.2fx speedup, quality delta %.3f",
                    round_num, plugin.name(), result.speedup, result.quality_delta,
                )
            else:
                logger.info(
                    "Round %d: SKIPPED %s -- %.2fx speedup, quality delta %.3f",
                    round_num, plugin.name(), result.speedup, result.quality_delta,
                )

        return results

    def compound_test(
        self,
        plugins: list[OptimizationPlugin],
        profile: Optional[ProfileResult] = None,
    ) -> list[PluginResult]:
        """
        Test a specific combination of plugins applied in order.

        Applies each plugin sequentially, verifying after each. If any plugin
        causes regression, it is rolled back but remaining plugins are still tried.

        Returns one PluginResult per plugin in the input list.
        """
        results: list[PluginResult] = []

        for plugin in plugins:
            # Check for conflicts with previously applied in this batch
            applied_names = [r.metadata.get("plugin_name", "") for r in results if r.applied]
            conflicts = set(plugin.conflicts_with())
            if conflicts & set(applied_names):
                results.append(PluginResult(
                    applied=False,
                    speedup=1.0,
                    quality_delta=0.0,
                    rollback_fn=lambda: None,
                    metadata={
                        "plugin_name": plugin.name(),
                        "skip_reason": "conflicts with already-applied plugin",
                    },
                ))
                continue

            config = plugin.configure(profile) if profile else {}
            result = self._try_plugin_with_config(plugin, config)
            results.append(result)

        return results

    # -- Internal -----------------------------------------------------------

    def _try_plugin(
        self,
        plugin: OptimizationPlugin,
        profile: ProfileResult,
    ) -> PluginResult:
        """Configure, apply, verify a single plugin. Rollback on regression."""
        try:
            config = plugin.configure(profile)
        except Exception as exc:
            logger.warning("Plugin %s configure() failed: %s", plugin.name(), exc)
            return PluginResult(
                applied=False, speedup=1.0, quality_delta=0.0,
                rollback_fn=lambda: None,
                metadata={"plugin_name": plugin.name(), "error": f"configure: {exc}"},
            )
        return self._try_plugin_with_config(plugin, config)

    def _try_plugin_with_config(
        self,
        plugin: OptimizationPlugin,
        config: dict[str, Any],
    ) -> PluginResult:
        """Apply with given config, verify, rollback on regression."""
        pname = plugin.name()

        # Apply
        try:
            plugin.apply(config)
        except Exception as exc:
            logger.warning("Plugin %s apply() failed: %s", pname, exc)
            return PluginResult(
                applied=False, speedup=1.0, quality_delta=0.0,
                rollback_fn=lambda: None,
                metadata={"plugin_name": pname, "error": f"apply: {exc}"},
            )

        # Verify
        try:
            speedup, quality_delta = plugin.verify()
        except Exception as exc:
            logger.warning("Plugin %s verify() failed, rolling back: %s", pname, exc)
            plugin.rollback()
            return PluginResult(
                applied=False, speedup=1.0, quality_delta=0.0,
                rollback_fn=lambda: None,
                metadata={"plugin_name": pname, "error": f"verify: {exc}"},
            )

        # Decide: keep or rollback
        keep = speedup > 1.0 and quality_delta >= -0.01
        if not keep:
            plugin.rollback()

        return PluginResult(
            applied=keep,
            speedup=speedup,
            quality_delta=quality_delta,
            rollback_fn=plugin.rollback,
            metadata={
                "plugin_name": pname,
                "plugin_version": plugin.version(),
                "config": config,
            },
        )

    def _build_conflict_set(self) -> set[str]:
        """Build the set of plugin names that conflict with currently applied ones."""
        conflicts: set[str] = set()
        for applied_name in self._applied:
            plugin = self._plugins.get(applied_name)
            if plugin:
                conflicts.update(plugin.conflicts_with())
        return conflicts

    def reset(self) -> None:
        """Roll back all applied plugins and clear state."""
        for name in reversed(self._applied):
            plugin = self._plugins.get(name)
            if plugin:
                try:
                    plugin.rollback()
                except Exception:
                    logger.warning("Failed to rollback %s during reset", name, exc_info=True)
        self._applied.clear()
        self._results_log.clear()
