"""
AutoKernel v2 Optimizer -- Main orchestration loop for automatic optimization.

Implements the core optimization loop:
  Profile -> Rank Bottlenecks -> Generate Candidates -> Benchmark -> Apply Winners -> Iterate

Each round focuses on the highest-headroom bottleneck and tries up to 10
candidate optimizations. Winners (>1% improvement) are applied and compounded.
The loop converges when headroom drops below 1% or max rounds are exhausted.

Usage:
    from autokernel_v2.optimizer import AutoOptimizer
    optimizer = AutoOptimizer(gpu_info=gpu)
    report = optimizer.optimize("path/to/model", max_rounds=5)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .types import (
    BenchmarkResult,
    Candidate,
    GPUInfo,
    OptimizationRound,
    OptimizationTarget,
    ProfileResult,
    detect_gpu_runtime,
)
from .profiler import ModelProfiler
from .bottleneck_ranker import BottleneckRanker
from .candidate_generator import CandidateGenerator
from .knowledge_base import KnowledgeBase


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """Configuration for the optimization loop."""
    max_rounds: int = 5
    min_headroom: float = 0.01         # stop when top headroom < 1%
    min_speedup: float = 1.01          # only apply if >1% improvement
    max_candidates_per_round: int = 10
    target_metric: str = "throughput"   # throughput, latency, memory
    batch_size: int = 1
    seq_len: int = 2048
    verbose: bool = True
    output_dir: str = "autokernel_v2_results"


# ---------------------------------------------------------------------------
# Optimization Report
# ---------------------------------------------------------------------------

@dataclass
class OptimizationReport:
    """Final report from an optimization run."""
    model_name: str
    gpu_name: str
    config: OptimizerConfig
    rounds: list[OptimizationRound] = field(default_factory=list)
    initial_profile: Optional[ProfileResult] = None
    final_profile: Optional[ProfileResult] = None
    total_speedup: float = 1.0
    total_time_seconds: float = 0.0
    applied_optimizations: list[str] = field(default_factory=list)
    convergence_reason: str = ""

    def summary(self) -> str:
        """Generate human-readable optimization report."""
        lines = [
            "",
            "=" * 80,
            "AUTOKERNEL V2 OPTIMIZATION REPORT",
            "=" * 80,
            f"Model:  {self.model_name}",
            f"GPU:    {self.gpu_name}",
            f"Target: {self.config.target_metric}",
            f"Rounds: {len(self.rounds)} / {self.config.max_rounds}",
            f"Time:   {self.total_time_seconds:.1f}s",
            "",
        ]

        if self.initial_profile:
            lines.append(f"Initial decode time: {self.initial_profile.total_time_us:.1f} us")
            lines.append(f"Initial throughput:  {self.initial_profile.throughput_tokens_per_sec:.1f} tok/s")
        if self.final_profile:
            lines.append(f"Final decode time:   {self.final_profile.total_time_us:.1f} us")
            lines.append(f"Final throughput:    {self.final_profile.throughput_tokens_per_sec:.1f} tok/s")

        lines.extend([
            "",
            f"Total speedup: {self.total_speedup:.2f}x",
            f"Convergence:   {self.convergence_reason}",
            "",
        ])

        if self.applied_optimizations:
            lines.append("Applied optimizations:")
            for i, opt in enumerate(self.applied_optimizations, 1):
                lines.append(f"  {i}. {opt}")
            lines.append("")

        # Per-round details
        for rnd in self.rounds:
            lines.append(f"--- Round {rnd.round_number} ---")
            lines.append(f"  Target: {rnd.target.op_name} "
                        f"(headroom={rnd.target.headroom:.4f}, "
                        f"Amdahl max={rnd.target.amdahl_max_speedup:.2f}x)")
            lines.append(f"  Candidates tried: {len(rnd.results)}")
            if rnd.best_result:
                br = rnd.best_result
                lines.append(f"  Best: {br.candidate.name} -> {br.speedup:.2f}x "
                            f"({'APPLIED' if br.applied else 'SKIPPED'})")
            else:
                lines.append("  Best: none met threshold")
            lines.append(f"  Cumulative speedup: {rnd.cumulative_speedup:.2f}x")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Serialize report to JSON-compatible dict."""
        return {
            "model_name": self.model_name,
            "gpu_name": self.gpu_name,
            "total_speedup": self.total_speedup,
            "total_time_seconds": self.total_time_seconds,
            "convergence_reason": self.convergence_reason,
            "num_rounds": len(self.rounds),
            "applied_optimizations": self.applied_optimizations,
            "rounds": [
                {
                    "round": r.round_number,
                    "target": r.target.op_name,
                    "target_headroom": r.target.headroom,
                    "candidates_tried": len(r.results),
                    "best_speedup": r.best_result.speedup if r.best_result else 0.0,
                    "best_name": r.best_result.candidate.name if r.best_result else "",
                    "applied": r.best_result.applied if r.best_result else False,
                    "cumulative_speedup": r.cumulative_speedup,
                }
                for r in self.rounds
            ],
        }


# ---------------------------------------------------------------------------
# AutoOptimizer
# ---------------------------------------------------------------------------

class AutoOptimizer:
    """
    Main optimization orchestrator.

    Runs the full optimization loop:
      1. Profile model to get per-op breakdown
      2. Rank bottlenecks by Amdahl's law headroom
      3. Generate candidate optimizations for top bottleneck
      4. Benchmark candidates (simulated or real)
      5. Apply the best if it beats threshold
      6. Repeat until convergence or max rounds

    In v0, benchmarking is analytical (estimated from the knowledge base
    and GPU specs). Future versions will run actual kernel benchmarks.
    """

    def __init__(
        self,
        gpu_info: Optional[GPUInfo] = None,
        config: Optional[OptimizerConfig] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self.gpu = gpu_info or detect_gpu_runtime()
        self.config = config or OptimizerConfig()
        self.kb = knowledge_base or KnowledgeBase()
        self.profiler = ModelProfiler(verbose=self.config.verbose)
        self.ranker = BottleneckRanker()
        self.generator = CandidateGenerator(self.gpu, self.kb)

    def optimize(
        self,
        model_path: str,
        gpu_info: Optional[GPUInfo] = None,
        max_rounds: Optional[int] = None,
    ) -> OptimizationReport:
        """
        Run the full optimization loop on a model.

        Args:
            model_path: Path to model directory, config.json, or profile JSON
            gpu_info: Override GPU info (uses self.gpu if None)
            max_rounds: Override max rounds (uses self.config.max_rounds if None)

        Returns:
            OptimizationReport with all rounds, applied optimizations, and speedups
        """
        gpu = gpu_info or self.gpu
        rounds_limit = max_rounds or self.config.max_rounds
        start_time = time.time()

        report = OptimizationReport(
            model_name=os.path.basename(model_path),
            gpu_name=gpu.name,
            config=self.config,
        )

        if self.config.verbose:
            print(f"\n{'='*80}")
            print(f"AutoKernel v2 Optimization")
            print(f"Model: {model_path}")
            print(f"GPU:   {gpu.name} ({gpu.peak_tflops_fp16:.0f} TFLOPS FP16, "
                  f"{gpu.memory_gb:.0f} GB, {gpu.peak_bandwidth_gb_s:.0f} GB/s)")
            print(f"{'='*80}\n")

        # Initial profile
        profile = self.profiler.profile(
            model_path, gpu,
            batch_size=self.config.batch_size,
            seq_len=self.config.seq_len,
        )
        report.initial_profile = profile

        if self.config.verbose:
            print(f"Initial profile: {profile.total_time_us:.1f} us total, "
                  f"{len(profile.ops)} op categories")
            for op in profile.top_ops(5):
                print(f"  {op.name:<20} {op.time_fraction:>6.1%} time, "
                      f"{op.utilization:>6.1%} util, "
                      f"AI={op.arithmetic_intensity:.1f}")
            print()

        cumulative_speedup = 1.0
        applied_patterns: set[str] = set()  # track applied pattern names to avoid repeats

        for round_num in range(1, rounds_limit + 1):
            if self.config.verbose:
                print(f"\n--- Round {round_num}/{rounds_limit} ---")

            # Rank bottlenecks
            targets = self.ranker.rank(profile)

            if not targets:
                report.convergence_reason = "no optimization targets found"
                if self.config.verbose:
                    print("No optimization targets found. Model appears well-optimized.")
                break

            top_target = targets[0]

            if top_target.headroom < self.config.min_headroom:
                report.convergence_reason = (
                    f"converged: top headroom {top_target.headroom:.4f} < "
                    f"threshold {self.config.min_headroom}"
                )
                if self.config.verbose:
                    print(f"Converged: headroom {top_target.headroom:.4f} < {self.config.min_headroom}")
                break

            if self.config.verbose:
                print(f"Target: {top_target.op_name}")
                print(f"  Time fraction: {top_target.time_fraction:.1%}")
                print(f"  Utilization:   {top_target.utilization:.1%}")
                print(f"  Headroom:      {top_target.headroom:.4f}")
                print(f"  Amdahl max:    {top_target.amdahl_max_speedup:.2f}x")

            # Generate candidates (excluding already-applied patterns)
            candidates = [
                c for c in self.generator.generate(top_target)
                if c.name not in applied_patterns
            ]

            if not candidates:
                report.convergence_reason = "all candidates exhausted for top bottleneck"
                if self.config.verbose:
                    print("  No new candidates available. Stopping.")
                break

            if self.config.verbose:
                print(f"  Generated {len(candidates)} candidates:")
                for c in candidates[:5]:
                    print(f"    - {c.name}: {c.expected_impact:.1f}x expected "
                          f"({c.confidence:.0%} confidence)")

            # Benchmark candidates (analytical in v0)
            results = []
            for candidate in candidates[:self.config.max_candidates_per_round]:
                result = self._benchmark_candidate(candidate, profile)
                results.append(result)

            # Find best result
            passing = [r for r in results if r.correctness and r.speedup > 1.0]
            best = max(passing, key=lambda r: r.speedup) if passing else None

            # Create round record
            rnd = OptimizationRound(
                round_number=round_num,
                target=top_target,
                candidates=candidates,
                results=results,
            )

            if best and best.speedup >= self.config.min_speedup:
                best.applied = True
                rnd.best_result = best

                # Apply the optimization (update profile analytically)
                profile = self._apply_optimization(profile, top_target, best)
                cumulative_speedup *= best.speedup

                applied_patterns.add(best.candidate.name)
                report.applied_optimizations.append(
                    f"{best.candidate.name}: {best.speedup:.2f}x on {top_target.op_name}"
                )

                # Update knowledge base confidence
                self.kb.update_confidence(
                    best.candidate.name.replace("kb_", ""),
                    success=True,
                    model_gpu=f"{report.model_name}-{gpu.name}",
                )

                if self.config.verbose:
                    print(f"  APPLIED: {best.candidate.name} -> {best.speedup:.2f}x")
                    print(f"  Cumulative speedup: {cumulative_speedup:.2f}x")
            else:
                rnd.best_result = best
                if self.config.verbose:
                    if best:
                        print(f"  Best candidate {best.candidate.name} ({best.speedup:.2f}x) "
                              f"below threshold {self.config.min_speedup:.2f}x")
                    else:
                        print("  No passing candidates found")

                # Update KB for failed candidates
                for r in results:
                    if not r.correctness or r.speedup <= 1.0:
                        self.kb.update_confidence(
                            r.candidate.name.replace("kb_", ""),
                            success=False,
                        )

            rnd.cumulative_speedup = cumulative_speedup
            report.rounds.append(rnd)

        # Finalize report
        report.final_profile = profile
        report.total_speedup = cumulative_speedup
        report.total_time_seconds = time.time() - start_time

        if not report.convergence_reason:
            report.convergence_reason = f"max rounds ({rounds_limit}) reached"

        # Save report
        self._save_report(report, model_path)

        if self.config.verbose:
            print(report.summary())

        return report

    def _benchmark_candidate(
        self,
        candidate: Candidate,
        profile: ProfileResult,
    ) -> BenchmarkResult:
        """
        Benchmark a candidate optimization.

        In v0, this is analytical -- we estimate the speedup from the candidate's
        expected impact, confidence, and the target's properties. Future versions
        will run actual Triton kernel benchmarks via bench.py.
        """
        target = candidate.target

        # Analytical estimation: speedup = 1 + (expected_impact - 1) * confidence * damping
        # The damping factor accounts for real-world effects not captured by estimates
        damping = 0.6  # conservative: we expect to achieve 60% of estimated impact

        raw_speedup = candidate.expected_impact
        confidence = candidate.confidence

        # Adjust for GPU-specific factors
        if "fp4" in candidate.name and not self.gpu.has_fp4:
            raw_speedup = 1.0  # FP4 won't work on non-Blackwell
            confidence = 0.0
        if "fp8" in candidate.name and not self.gpu.has_fp8:
            raw_speedup = 1.0
            confidence = 0.0

        # Estimated speedup on this specific op
        op_speedup = 1.0 + (raw_speedup - 1.0) * confidence * damping

        # Amdahl's law: translate op speedup to end-to-end speedup
        f = target.time_fraction
        e2e_speedup = 1.0 / ((1.0 - f) + f / op_speedup) if op_speedup > 0 else 1.0

        # Estimate correctness (high confidence patterns are likely correct)
        correctness = confidence > 0.3

        return BenchmarkResult(
            candidate=candidate,
            baseline_time_us=profile.total_time_us,
            optimized_time_us=profile.total_time_us / e2e_speedup if e2e_speedup > 0 else profile.total_time_us,
            speedup=e2e_speedup,
            correctness=correctness,
            notes=f"Analytical estimate: op_speedup={op_speedup:.2f}x, e2e={e2e_speedup:.2f}x",
        )

    def _apply_optimization(
        self,
        profile: ProfileResult,
        target: OptimizationTarget,
        result: BenchmarkResult,
    ) -> ProfileResult:
        """
        Apply an optimization to the profile (update times analytically).

        Returns a new ProfileResult with the target op's time reduced.
        """
        op_speedup = result.candidate.expected_impact * result.candidate.confidence * 0.6
        op_speedup = max(op_speedup, 1.01)  # at least 1% improvement

        new_ops = []
        new_total = 0.0

        for op in profile.ops:
            if op.name == target.op_name:
                new_time = op.time_us / op_speedup
                new_util = min(op.utilization * op_speedup, 0.99)
                from dataclasses import replace
                new_op = replace(op, time_us=new_time, utilization=new_util)
                new_ops.append(new_op)
                new_total += new_time
            else:
                new_ops.append(op)
                new_total += op.time_us

        # Recompute time fractions
        for op in new_ops:
            op.time_fraction = op.time_us / new_total if new_total > 0 else 0.0

        from dataclasses import replace
        return replace(
            profile,
            total_time_us=new_total,
            ops=sorted(new_ops, key=lambda o: o.time_us, reverse=True),
        )

    def _save_report(self, report: OptimizationReport, model_path: str) -> None:
        """Save optimization report to disk."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_path).stem

        # Save JSON report
        json_path = output_dir / f"report_{model_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(report.to_json(), f, indent=2)

        # Save text summary
        txt_path = output_dir / f"report_{model_name}_{timestamp}.txt"
        with open(txt_path, "w") as f:
            f.write(report.summary())

        if self.config.verbose:
            print(f"\nReport saved to: {json_path}")
