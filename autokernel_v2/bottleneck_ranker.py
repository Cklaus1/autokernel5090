"""
AutoKernel v2 Bottleneck Ranker -- Rank optimization targets by Amdahl's law headroom.

For each operation in a profile, computes:
  - headroom = time_fraction * (1 - utilization)
  - ceiling_speedup = 1 / utilization (how much faster this op could be)
  - amdahl_max_speedup = 1 / (1 - time_fraction * (1 - 1/ceiling_speedup))

Operations are ranked by headroom (highest first), which gives the expected
wall-clock time savings if this op were fully optimized.

Usage:
    from autokernel_v2.bottleneck_ranker import BottleneckRanker
    ranker = BottleneckRanker()
    targets = ranker.rank(profile_result)
"""

from __future__ import annotations

from .types import (
    OpCategory,
    OptimizationTarget,
    ProfileResult,
)


class BottleneckRanker:
    """
    Rank profiled operations by optimization potential using Amdahl's law.

    The key insight: the best target to optimize is NOT necessarily the slowest
    operation -- it's the one where (fraction of total time) * (room for
    improvement) is largest. A 50% utilization op that takes 40% of time
    is a better target than a 10% utilization op that takes 5% of time.
    """

    def __init__(
        self,
        min_time_fraction: float = 0.01,
        min_headroom: float = 0.005,
        exclude_categories: list[OpCategory] | None = None,
    ):
        """
        Args:
            min_time_fraction: Ignore ops below this fraction of total time
            min_headroom: Ignore ops with headroom below this threshold
            exclude_categories: Op categories to skip (e.g., COMMUNICATION)
        """
        self.min_time_fraction = min_time_fraction
        self.min_headroom = min_headroom
        self.exclude_categories = set(exclude_categories or [])

    def rank(self, profile: ProfileResult) -> list[OptimizationTarget]:
        """
        Rank profiled ops by optimization headroom.

        Returns a sorted list of OptimizationTarget, highest headroom first.
        """
        targets = []

        for op in profile.ops:
            # Skip excluded categories
            if op.category in self.exclude_categories:
                continue

            # Skip trivial ops
            if op.time_fraction < self.min_time_fraction:
                continue

            # Compute utilization -- clamp to [0.01, 0.99] to avoid division issues
            util = max(0.01, min(op.utilization, 0.99))

            # Headroom: how much of total time could be saved
            headroom = op.time_fraction * (1.0 - util)
            if headroom < self.min_headroom:
                continue

            # Ceiling speedup for this specific op
            ceiling_speedup = 1.0 / util

            # Amdahl's law: max end-to-end speedup if this op hits 100% utilization
            # S_max = 1 / (1 - f + f/ceiling)  where f = time_fraction
            f = op.time_fraction
            amdahl_denom = (1.0 - f) + f * util
            amdahl_max = 1.0 / amdahl_denom if amdahl_denom > 0 else 1.0

            # Build notes about the optimization opportunity
            notes = []
            if op.arithmetic_intensity < 10:
                notes.append("memory-bound (low arithmetic intensity)")
            elif op.arithmetic_intensity > 100:
                notes.append("compute-bound (high arithmetic intensity)")
            else:
                notes.append("mixed compute/memory bound")

            if util < 0.3:
                notes.append(f"very low utilization ({util:.0%}) -- large optimization potential")
            elif util < 0.6:
                notes.append(f"moderate utilization ({util:.0%}) -- room for improvement")
            else:
                notes.append(f"decent utilization ({util:.0%}) -- diminishing returns")

            if op.category == OpCategory.LINEAR and f > 0.3:
                notes.append("dominant linear op -- consider quantization or fusion")
            if op.category == OpCategory.ATTENTION:
                notes.append("attention -- consider flash attention variants or KV cache compression")
            if op.category == OpCategory.NORM:
                notes.append("normalization -- consider fusing with adjacent ops")

            targets.append(OptimizationTarget(
                op_name=op.name,
                category=op.category,
                time_fraction=f,
                utilization=util,
                headroom=headroom,
                ceiling_speedup=ceiling_speedup,
                amdahl_max_speedup=amdahl_max,
                shapes=op.shapes,
                dtype=op.dtype,
                memory_mb=op.memory_mb,
                notes=notes,
            ))

        # Sort by headroom (descending) -- this is the "expected time saved" metric
        targets.sort(key=lambda t: t.headroom, reverse=True)
        return targets

    def summary(self, targets: list[OptimizationTarget], top_n: int = 5) -> str:
        """Format a human-readable summary of the top optimization targets."""
        lines = [
            "=" * 80,
            "BOTTLENECK RANKING (by Amdahl's law headroom)",
            "=" * 80,
            f"{'Rank':<5} {'Operation':<20} {'Time%':<8} {'Util%':<8} {'Headroom':<10} {'Max E2E Speedup':<16} {'Notes'}",
            "-" * 80,
        ]

        for i, t in enumerate(targets[:top_n]):
            notes_str = "; ".join(t.notes[:1])  # just the first note for compact display
            lines.append(
                f"{i+1:<5} {t.op_name:<20} {t.time_fraction:>6.1%}  {t.utilization:>6.1%}  "
                f"{t.headroom:>8.4f}  {t.amdahl_max_speedup:>14.2f}x  {notes_str}"
            )

        lines.append("-" * 80)

        if targets:
            total_headroom = sum(t.headroom for t in targets)
            lines.append(f"Total headroom across all targets: {total_headroom:.4f}")
            lines.append(f"If ALL targets optimized to 100%: ~{1/(1-total_headroom):.2f}x end-to-end speedup")
        else:
            lines.append("No optimization targets found (model is well-optimized or profile is empty)")

        lines.append("=" * 80)
        return "\n".join(lines)
