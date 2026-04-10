"""
AutoKernel v2 CLI -- One-command optimization for any model on any GPU.

Usage:
    python3 -m autokernel_v2 optimize --model /path/to/model --gpu rtx-5090
    python3 -m autokernel_v2 profile --model /path/to/model
    python3 -m autokernel_v2 rank --profile /path/to/profile.json
    python3 -m autokernel_v2 candidates --target attention --gpu rtx-5090
    python3 -m autokernel_v2 knowledge --list
    python3 -m autokernel_v2 knowledge --lookup linear --gpu hopper_blackwell
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .types import (
    GPUInfo,
    KNOWN_GPUS,
    OpCategory,
    OptimizationTarget,
    detect_gpu_runtime,
)
from .profiler import ModelProfiler
from .bottleneck_ranker import BottleneckRanker
from .candidate_generator import CandidateGenerator
from .knowledge_base import KnowledgeBase
from .optimizer import AutoOptimizer, OptimizerConfig


def _resolve_gpu(gpu_arg: str | None) -> GPUInfo:
    """Resolve GPU from argument or runtime detection."""
    if gpu_arg is None:
        gpu = detect_gpu_runtime()
        if gpu.name == "Unknown":
            print("WARNING: No GPU detected. Using default RTX 5090 specs for planning.")
            return KNOWN_GPUS["rtx-5090"]
        return gpu

    gpu_key = gpu_arg.lower().replace(" ", "-")
    if gpu_key in KNOWN_GPUS:
        return KNOWN_GPUS[gpu_key]

    # Fuzzy match
    for key, info in KNOWN_GPUS.items():
        if gpu_key in key or key in gpu_key:
            return info

    print(f"WARNING: Unknown GPU '{gpu_arg}'. Known GPUs: {', '.join(KNOWN_GPUS.keys())}")
    print("Attempting runtime detection...")
    return detect_gpu_runtime()


def cmd_optimize(args: argparse.Namespace) -> int:
    """Run the full optimization loop."""
    gpu = _resolve_gpu(args.gpu)

    config = OptimizerConfig(
        max_rounds=args.max_rounds,
        target_metric=args.target,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        verbose=not args.quiet,
        output_dir=args.output_dir,
    )

    optimizer = AutoOptimizer(gpu_info=gpu, config=config)
    report = optimizer.optimize(args.model, max_rounds=args.max_rounds)

    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    # Summary is already printed by optimizer when verbose=True

    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    """Profile a model."""
    gpu = _resolve_gpu(args.gpu)
    profiler = ModelProfiler(verbose=True)

    result = profiler.profile(
        args.model, gpu,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    if args.json:
        print(json.dumps(profiler.to_json(result), indent=2))
    else:
        print(f"\nModel: {result.model_name}")
        print(f"GPU:   {gpu.name}")
        print(f"Total decode time: {result.total_time_us:.1f} us")
        print(f"Throughput: {result.throughput_tokens_per_sec:.1f} tok/s")
        print(f"\nMemory: {result.memory_total_mb:.0f} MB total "
              f"(weights: {result.memory_weights_mb:.0f}, "
              f"KV cache: {result.memory_kv_cache_mb:.0f})")
        print(f"\nOperation breakdown ({len(result.ops)} categories):")
        print(f"{'Category':<20} {'Time%':<8} {'Util%':<8} {'AI':<8} {'Time (us)':<12}")
        print("-" * 60)
        for op in result.ops:
            print(f"{op.name:<20} {op.time_fraction:>6.1%}  {op.utilization:>6.1%}  "
                  f"{op.arithmetic_intensity:>6.1f}  {op.time_us:>10.1f}")

    return 0


def cmd_rank(args: argparse.Namespace) -> int:
    """Rank bottlenecks from a profile."""
    gpu = _resolve_gpu(args.gpu)
    profiler = ModelProfiler()

    if args.profile:
        result = profiler.profile(args.profile, gpu, mode="profile_json")
    elif args.model:
        result = profiler.profile(args.model, gpu, batch_size=args.batch_size, seq_len=args.seq_len)
    else:
        print("ERROR: Specify --profile or --model")
        return 1

    ranker = BottleneckRanker()
    targets = ranker.rank(result)

    if args.json:
        print(json.dumps([
            {
                "op_name": t.op_name,
                "category": t.category.value,
                "time_fraction": t.time_fraction,
                "utilization": t.utilization,
                "headroom": t.headroom,
                "amdahl_max_speedup": t.amdahl_max_speedup,
                "notes": t.notes,
            }
            for t in targets
        ], indent=2))
    else:
        print(ranker.summary(targets, top_n=args.top))

    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    """Generate optimization candidates for a target."""
    gpu = _resolve_gpu(args.gpu)
    kb = KnowledgeBase()
    generator = CandidateGenerator(gpu, kb)

    # Create a synthetic target from args
    try:
        category = OpCategory(args.target)
    except ValueError:
        print(f"ERROR: Unknown category '{args.target}'. "
              f"Valid: {', '.join(c.value for c in OpCategory)}")
        return 1

    target = OptimizationTarget(
        op_name=args.target,
        category=category,
        time_fraction=0.5,    # assume dominant op
        utilization=0.3,      # assume underutilized
        headroom=0.35,
        ceiling_speedup=3.33,
        amdahl_max_speedup=1.54,
        shapes=json.loads(args.shapes) if args.shapes else {},
        dtype=args.dtype,
    )

    candidates = generator.generate(target)

    if args.json:
        print(json.dumps([
            {
                "name": c.name,
                "description": c.description,
                "strategy": c.strategy,
                "expected_impact": c.expected_impact,
                "confidence": c.confidence,
                "effort": c.effort,
                "implementation_plan": c.implementation_plan,
            }
            for c in candidates
        ], indent=2))
    else:
        print(generator.summary(candidates))

    return 0


def cmd_knowledge(args: argparse.Namespace) -> int:
    """Query the knowledge base."""
    kb = KnowledgeBase()

    if args.list:
        if args.json:
            print(json.dumps(kb.to_dict(), indent=2))
        else:
            print(kb.summary())
        return 0

    if args.lookup:
        try:
            category = OpCategory(args.lookup)
        except ValueError:
            category = None

        matches = kb.lookup(
            category=category,
            gpu_arch=args.gpu_arch,
            strategy=args.strategy,
        )

        if args.json:
            print(json.dumps([
                {
                    "name": p.name,
                    "description": p.description,
                    "expected_impact": p.expected_impact,
                    "confidence": p.confidence,
                    "validated_on": p.validated_on,
                }
                for p in matches
            ], indent=2))
        else:
            if not matches:
                print("No matching patterns found.")
            else:
                print(f"Found {len(matches)} matching patterns:\n")
                for p in matches:
                    print(f"  {p.name}")
                    print(f"    {p.description[:80]}")
                    print(f"    Impact: {p.expected_impact:.1f}x | "
                          f"Confidence: {p.confidence:.0%} | "
                          f"Effort: {p.effort}")
                    print(f"    Validated on: {', '.join(p.validated_on)}")
                    print()

        return 0

    print("Specify --list or --lookup <category>")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="autokernel_v2",
        description="AutoKernel v2 -- Automatic GPU kernel optimization discovery",
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # optimize
    p_opt = subparsers.add_parser("optimize", help="Run full optimization loop")
    p_opt.add_argument("--model", required=True, help="Path to model dir, config.json, or profile")
    p_opt.add_argument("--gpu", help="GPU identifier (e.g., rtx-5090, h100-sxm)")
    p_opt.add_argument("--target", default="throughput", choices=["throughput", "latency", "memory"])
    p_opt.add_argument("--max-rounds", type=int, default=5)
    p_opt.add_argument("--batch-size", type=int, default=1)
    p_opt.add_argument("--seq-len", type=int, default=2048)
    p_opt.add_argument("--output-dir", default="autokernel_v2_results")
    p_opt.add_argument("--quiet", action="store_true")
    p_opt.set_defaults(func=cmd_optimize)

    # profile
    p_prof = subparsers.add_parser("profile", help="Profile a model")
    p_prof.add_argument("--model", required=True, help="Path to model dir or config.json")
    p_prof.add_argument("--gpu", help="GPU identifier")
    p_prof.add_argument("--batch-size", type=int, default=1)
    p_prof.add_argument("--seq-len", type=int, default=2048)
    p_prof.set_defaults(func=cmd_profile)

    # rank
    p_rank = subparsers.add_parser("rank", help="Rank bottlenecks")
    p_rank.add_argument("--profile", help="Path to profile JSON")
    p_rank.add_argument("--model", help="Path to model (will profile first)")
    p_rank.add_argument("--gpu", help="GPU identifier")
    p_rank.add_argument("--batch-size", type=int, default=1)
    p_rank.add_argument("--seq-len", type=int, default=2048)
    p_rank.add_argument("--top", type=int, default=10)
    p_rank.set_defaults(func=cmd_rank)

    # candidates
    p_cand = subparsers.add_parser("candidates", help="Generate optimization candidates")
    p_cand.add_argument("--target", required=True, help="Op category (linear, attention, norm, etc.)")
    p_cand.add_argument("--gpu", help="GPU identifier")
    p_cand.add_argument("--shapes", help="JSON dict of shapes (e.g., '{\"M\":1,\"N\":4096,\"K\":4096}')")
    p_cand.add_argument("--dtype", default="float16")
    p_cand.set_defaults(func=cmd_candidates)

    # knowledge
    p_kb = subparsers.add_parser("knowledge", help="Query knowledge base")
    p_kb.add_argument("--list", action="store_true", help="List all patterns")
    p_kb.add_argument("--lookup", help="Lookup patterns by category")
    p_kb.add_argument("--gpu-arch", help="Filter by GPU architecture")
    p_kb.add_argument("--strategy", help="Filter by strategy (fusion, quantization, etc.)")
    p_kb.set_defaults(func=cmd_knowledge)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
