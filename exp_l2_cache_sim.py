#!/usr/bin/env python3
"""
Experiment: Measure whether expert memory layout affects L2 cache hits.

We simulate the MoE expert access pattern:
1. Allocate a tensor shaped [128, expert_rows, expert_cols] (like w13_weight)
2. Access experts in different orders and measure throughput
3. Compare: sequential, random, hot-set-first, real-routing-order

Key insight: The RTX 5090 L2 cache is 96 MB. Each expert's w13_weight is
~1.88 MB (1408 * 1408 bytes uint8). So ~51 experts fit in L2 for w13 alone.
With all 3 projections per expert (~3.19 MB), ~30 experts fit.
"""
import torch
import time
import numpy as np
import json

def benchmark_expert_access(weights, expert_order, num_iters=100, warmup=20):
    """
    Simulate accessing experts by indexing into [E, ...] tensor.
    This mimics what the grouped GEMM kernel does when it gathers expert weights.
    """
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        for eid in expert_order:
            _ = weights[eid].sum()
    torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(num_iters):
        for eid in expert_order:
            _ = weights[eid].sum()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters  # seconds per iteration


def benchmark_gather_access(weights, expert_indices_list, num_iters=200, warmup=50):
    """
    More realistic: gather multiple experts at once (like batched MoE).
    expert_indices_list is a list of tensors, each containing expert IDs for one "layer call".
    """
    torch.cuda.synchronize()

    for _ in range(warmup):
        for indices in expert_indices_list:
            gathered = weights[indices]  # [K, rows, cols]
            _ = gathered.sum()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        for indices in expert_indices_list:
            gathered = weights[indices]
            _ = gathered.sum()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters


def benchmark_matmul_access(weights, expert_indices, hidden, num_iters=200, warmup=50):
    """
    Most realistic: gather expert weights and do matmul, simulating actual MoE compute.
    """
    torch.cuda.synchronize()

    for _ in range(warmup):
        w = weights[expert_indices]  # [K, out, in]
        # Batch matmul: [K, 1, in] @ [K, in, out] -> [K, 1, out]
        x = hidden.unsqueeze(0).expand(len(expert_indices), -1).unsqueeze(1)
        _ = torch.bmm(x.half(), w.half().transpose(1, 2))
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iters):
        w = weights[expert_indices]
        x = hidden.unsqueeze(0).expand(len(expert_indices), -1).unsqueeze(1)
        _ = torch.bmm(x.half(), w.half().transpose(1, 2))
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / num_iters


def main():
    torch.manual_seed(42)
    NUM_EXPERTS = 128
    TOP_K = 8
    NUM_TOKENS = 32  # batch size

    # Actual expert weight dimensions from Gemma4 NVFP4
    # w13 (gate_up): [128, 1408, 1408] uint8 = 1.88 MB/expert
    # w2 (down): [128, 2816, 352] uint8 = 0.95 MB/expert

    print("=" * 80)
    print("EXPERT WEIGHT L2 CACHE EXPERIMENT")
    print("=" * 80)
    print(f"RTX 5090 L2 cache: 96 MB")
    print(f"Experts: {NUM_EXPERTS}, top_k: {TOP_K}")
    print()

    # ---- Experiment 1: Raw weight gather bandwidth ----
    print("--- Experiment 1: Weight gather throughput ---")

    # Use actual w13 dimensions
    w13 = torch.randint(0, 256, (NUM_EXPERTS, 1408, 1408), dtype=torch.uint8, device="cuda")
    expert_size_mb = 1408 * 1408 / 1024 / 1024
    print(f"w13_weight shape: {w13.shape}, per-expert: {expert_size_mb:.2f} MB")
    print(f"Total tensor: {w13.numel() / 1024 / 1024:.0f} MB")
    print(f"L2 fits ~{96 / expert_size_mb:.0f} experts")

    # Pattern A: Access experts 0-7 (contiguous, likely in L2)
    contiguous_ids = list(range(8))
    # Pattern B: Access experts spread across memory (0, 16, 32, ..., 112)
    spread_ids = list(range(0, 128, 16))
    # Pattern C: Random 8 experts
    random_ids = sorted(np.random.choice(128, 8, replace=False).tolist())
    # Pattern D: Same 8 experts repeatedly (hot set)
    hot_ids = list(range(8))

    patterns = {
        "contiguous_0_7": contiguous_ids,
        "spread_0_16_32": spread_ids,
        "random_8": random_ids,
    }

    for name, ids in patterns.items():
        indices = torch.tensor(ids, device="cuda")
        # Warmup to load into L2
        for _ in range(50):
            _ = w13[indices].sum()
        torch.cuda.synchronize()

        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            gathered = w13[indices]
            _ = gathered.sum()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_us = np.mean(times) * 1e6
        p50_us = np.percentile(times, 50) * 1e6
        p99_us = np.percentile(times, 99) * 1e6
        data_mb = len(ids) * expert_size_mb
        bw_gbs = data_mb / 1024 / (np.mean(times))
        print(f"  {name:25s}: avg={avg_us:8.1f}us  p50={p50_us:8.1f}us  p99={p99_us:8.1f}us  "
              f"BW={bw_gbs:.1f} GB/s  ({len(ids)} experts, {data_mb:.1f} MB)")

    # ---- Experiment 2: Simulate real MoE layer access pattern ----
    print("\n--- Experiment 2: Simulated MoE layer sequence ---")
    print("Simulating 30 consecutive MoE layers, each routing 32 tokens to 8 experts")

    # Generate routing decisions: 32 tokens * 8 experts per layer
    np.random.seed(42)

    # Scenario A: Uniform routing (all experts equally likely)
    uniform_routing = []
    for layer in range(30):
        layer_experts = set()
        for _ in range(32):
            selected = np.random.choice(128, TOP_K, replace=False)
            layer_experts.update(selected)
        uniform_routing.append(sorted(layer_experts))

    # Scenario B: Skewed routing (some experts much more popular)
    # Zipf-like distribution
    expert_probs = np.array([1.0 / (i + 1) for i in range(128)])
    expert_probs /= expert_probs.sum()

    skewed_routing = []
    for layer in range(30):
        layer_experts = set()
        for _ in range(32):
            selected = np.random.choice(128, TOP_K, replace=False, p=expert_probs)
            layer_experts.update(selected)
        skewed_routing.append(sorted(layer_experts))

    # Scenario C: Reordered - hot experts are 0-31, same Zipf distribution
    reordered_routing = []
    reorder_map = np.argsort(-expert_probs)  # most popular first
    for layer in range(30):
        layer_experts = set()
        for _ in range(32):
            selected = np.random.choice(128, TOP_K, replace=False, p=expert_probs)
            # Map to reordered indices
            remapped = [int(np.where(reorder_map == e)[0][0]) for e in selected]
            layer_experts.update(remapped)
        reordered_routing.append(sorted(layer_experts))

    scenarios = {
        "uniform": uniform_routing,
        "skewed_natural": skewed_routing,
        "skewed_reordered": reordered_routing,
    }

    for name, routing in scenarios.items():
        avg_experts_per_layer = np.mean([len(r) for r in routing])
        avg_unique_across = len(set().union(*routing))

        # Time the full 30-layer access pattern
        indices_list = [torch.tensor(r, device="cuda") for r in routing]

        torch.cuda.synchronize()
        # Warmup
        for _ in range(20):
            for idx in indices_list:
                _ = w13[idx].sum()
        torch.cuda.synchronize()

        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            start = time.perf_counter()
            for idx in indices_list:
                gathered = w13[idx]
                _ = gathered.sum()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        p50_ms = np.percentile(times, 50) * 1000
        total_data_mb = sum(len(r) * expert_size_mb for r in routing)

        print(f"\n  {name}:")
        print(f"    Avg experts/layer: {avg_experts_per_layer:.0f}, unique total: {avg_unique_across}")
        print(f"    30-layer pass: avg={avg_ms:.2f}ms  p50={p50_ms:.2f}ms")
        print(f"    Total data: {total_data_mb:.0f} MB")
        print(f"    Effective BW: {total_data_mb / 1024 / np.mean(times):.1f} GB/s")

    # ---- Experiment 3: L2 cache thrashing test ----
    print("\n--- Experiment 3: L2 cache reuse between adjacent layers ---")
    print("Testing: if layer N and N+1 share experts, does access speed up?")

    # Two consecutive accesses to same experts vs different experts
    same_experts = torch.tensor(list(range(32)), device="cuda")
    diff_experts_a = torch.tensor(list(range(32)), device="cuda")
    diff_experts_b = torch.tensor(list(range(64, 96)), device="cuda")

    # Warm up same experts, then access again
    torch.cuda.synchronize()
    for _ in range(100):
        _ = w13[same_experts].sum()
    torch.cuda.synchronize()

    # Time: access same set again (should be in L2)
    times_same = []
    for _ in range(200):
        _ = w13[same_experts].sum()  # prime
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[same_experts].sum()  # reuse
        torch.cuda.synchronize()
        times_same.append(time.perf_counter() - start)

    # Time: access different set (forces L2 eviction)
    times_diff = []
    for _ in range(200):
        _ = w13[diff_experts_a].sum()  # prime with set A
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[diff_experts_b].sum()  # access set B (not in L2)
        torch.cuda.synchronize()
        times_diff.append(time.perf_counter() - start)

    avg_same = np.mean(times_same) * 1e6
    avg_diff = np.mean(times_diff) * 1e6
    print(f"  Same 32 experts (L2 hit):   {avg_same:.1f} us")
    print(f"  Different 32 experts (miss): {avg_diff:.1f} us")
    print(f"  Speedup from L2 reuse: {avg_diff/avg_same:.2f}x")

    # ---- Experiment 4: Does contiguous storage help prefetching? ----
    print("\n--- Experiment 4: Contiguous vs scattered expert access ---")

    # Access experts [0,1,2,...,7] vs [0,16,32,...,112]
    contiguous = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda")
    scattered = torch.tensor([0, 16, 32, 48, 64, 80, 96, 112], device="cuda")

    # Flush L2 by accessing all experts
    _ = w13.sum()
    torch.cuda.synchronize()

    times_contig = []
    for _ in range(500):
        # Flush with full tensor access
        _ = w13[64:96].sum()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[contiguous].sum()
        torch.cuda.synchronize()
        times_contig.append(time.perf_counter() - start)

    _ = w13.sum()
    torch.cuda.synchronize()

    times_scatter = []
    for _ in range(500):
        _ = w13[64:96].sum()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[scattered].sum()
        torch.cuda.synchronize()
        times_scatter.append(time.perf_counter() - start)

    avg_contig = np.mean(times_contig) * 1e6
    avg_scatter = np.mean(times_scatter) * 1e6
    print(f"  Contiguous [0..7]:  {avg_contig:.1f} us")
    print(f"  Scattered [0,16,32,...]: {avg_scatter:.1f} us")
    print(f"  Ratio (scatter/contig): {avg_scatter/avg_contig:.3f}x")

    del w13
    torch.cuda.empty_cache()

    # ---- Experiment 5: Full MoE simulation with GEMM ----
    print("\n--- Experiment 5: Expert gather + GEMM (realistic MoE) ---")

    # Smaller scale to fit in memory alongside the actual model
    # Simulate w13: [E, out_features, in_features]
    w13_half = torch.randn(128, 1408, 704, dtype=torch.float16, device="cuda")
    hidden = torch.randn(704, dtype=torch.float16, device="cuda")

    # Time: gather 8 contiguous experts + batch matmul
    contig_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda")
    scatter_ids = torch.tensor([3, 17, 42, 58, 79, 93, 105, 121], device="cuda")

    def time_gather_gemm(ids, num_iters=500, warmup=100):
        torch.cuda.synchronize()
        for _ in range(warmup):
            w = w13_half[ids]
            x = hidden.unsqueeze(0).expand(len(ids), -1).unsqueeze(1)
            _ = torch.bmm(x, w.transpose(1, 2))
        torch.cuda.synchronize()

        times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            w = w13_half[ids]
            x = hidden.unsqueeze(0).expand(len(ids), -1).unsqueeze(1)
            _ = torch.bmm(x, w.transpose(1, 2))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        return times

    times_c = time_gather_gemm(contig_ids)
    times_s = time_gather_gemm(scatter_ids)

    print(f"  Contiguous experts + GEMM: {np.mean(times_c)*1e6:.1f} us (p50={np.percentile(times_c,50)*1e6:.1f})")
    print(f"  Scattered experts + GEMM:  {np.mean(times_s)*1e6:.1f} us (p50={np.percentile(times_s,50)*1e6:.1f})")
    print(f"  Ratio: {np.mean(times_s)/np.mean(times_c):.3f}x")

    del w13_half
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)


if __name__ == "__main__":
    main()
