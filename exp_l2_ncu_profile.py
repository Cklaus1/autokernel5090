#!/usr/bin/env python3
"""
Experiment: Direct measurement of L2 cache behavior during MoE expert access.
Uses actual NVFP4 weight dimensions from Gemma4.
"""
import torch
import time
import numpy as np

def main():
    NUM_EXPERTS = 128
    device = "cuda"

    print("=" * 80)
    print("EXPERT ORDERING vs L2 CACHE PERFORMANCE (NVFP4 dimensions)")
    print("=" * 80)

    # Actual NVFP4 dimensions from Gemma4:
    # w13_weight: [128, 1408, 1408] uint8 = 1.89 MB/expert
    # w13_weight_scale: [128, 1408, 176] fp8 = 0.24 MB/expert
    # w2_weight: [128, 2816, 352] uint8 = 0.95 MB/expert
    # w2_weight_scale: [128, 2816, 44] fp8 = 0.12 MB/expert
    # Total per expert: 3.19 MB

    w13 = torch.randint(0, 256, (NUM_EXPERTS, 1408, 1408), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (NUM_EXPERTS, 2816, 352), dtype=torch.uint8, device=device)

    w13_per_expert_mb = 1408 * 1408 / 1024 / 1024
    w2_per_expert_mb = 2816 * 352 / 1024 / 1024
    total_per_expert_mb = w13_per_expert_mb + w2_per_expert_mb

    print(f"w13 per expert: {w13_per_expert_mb:.2f} MB, w2 per expert: {w2_per_expert_mb:.2f} MB")
    print(f"Total per expert (weights only): {total_per_expert_mb:.2f} MB")
    print(f"L2 cache: 96 MB, fits ~{96/total_per_expert_mb:.0f} experts (weights only)")
    print(f"w13 total: {w13.numel()/1024/1024:.0f} MB, w2 total: {w2.numel()/1024/1024:.0f} MB")
    print()

    # ---- Test 1: Gather bandwidth by access pattern ----
    print("--- Test 1: w13 gather bandwidth by access pattern ---")
    num_iters = 1000
    warmup = 200

    patterns = {
        "contiguous[0:8]": list(range(8)),
        "contiguous[60:68]": list(range(60, 68)),
        "random_8": sorted(np.random.RandomState(42).choice(128, 8, replace=False).tolist()),
        "max_stride[0,16,..]": [0, 16, 32, 48, 64, 80, 96, 112],
    }

    for name, expert_ids in patterns.items():
        ids = torch.tensor(expert_ids, device=device)

        # Warmup
        for _ in range(warmup):
            w_sub = w13[ids]
            _ = w_sub.to(torch.float16).sum()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            w_sub = w13[ids]
            _ = w_sub.to(torch.float16).sum()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        per_iter_us = elapsed_ms * 1000 / num_iters
        data_mb = len(expert_ids) * w13_per_expert_mb
        bw_gbs = data_mb / 1024 / (per_iter_us / 1e6)
        print(f"  {name:25s}: {per_iter_us:7.1f} us  BW={bw_gbs:5.1f} GB/s  ids={expert_ids}")

    print()

    # ---- Test 2: Full MoE pass (w13 + w2 for 8 experts) ----
    print("--- Test 2: Full expert access (w13 + w2, 8 experts) ---")

    patterns_full = {
        "contiguous[0:8]": list(range(8)),
        "random_8": sorted(np.random.RandomState(42).choice(128, 8, replace=False).tolist()),
        "max_stride": [0, 16, 32, 48, 64, 80, 96, 112],
    }

    for name, expert_ids in patterns_full.items():
        ids = torch.tensor(expert_ids, device=device)

        for _ in range(warmup):
            w13_sub = w13[ids]
            w2_sub = w2[ids]
            _ = w13_sub.to(torch.float16).sum() + w2_sub.to(torch.float16).sum()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            w13_sub = w13[ids]
            w2_sub = w2[ids]
            _ = w13_sub.to(torch.float16).sum() + w2_sub.to(torch.float16).sum()
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        per_iter_us = elapsed_ms * 1000 / num_iters
        data_mb = len(expert_ids) * total_per_expert_mb
        bw_gbs = data_mb / 1024 / (per_iter_us / 1e6)
        print(f"  {name:25s}: {per_iter_us:7.1f} us  BW={bw_gbs:5.1f} GB/s")

    print()

    # ---- Test 3: L2 temporal reuse between "layers" ----
    print("--- Test 3: L2 temporal reuse between consecutive MoE layers ---")

    # Scenario A: Layer N and N+1 access SAME 8 experts
    same_ids = torch.tensor(list(range(8)), device=device)

    for _ in range(200):
        _ = w13[same_ids].to(torch.float16).sum()
    torch.cuda.synchronize()

    times_reuse = []
    for _ in range(500):
        _ = w13[same_ids].to(torch.float16).sum()  # "layer N"
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[same_ids].to(torch.float16).sum()  # "layer N+1" - same experts
        torch.cuda.synchronize()
        times_reuse.append(time.perf_counter() - start)

    # Scenario B: Layer N and N+1 access DIFFERENT 8 experts
    ids_a = torch.tensor(list(range(8)), device=device)
    ids_b = torch.tensor(list(range(64, 72)), device=device)

    for _ in range(200):
        _ = w13[ids_a].to(torch.float16).sum()
        _ = w13[ids_b].to(torch.float16).sum()
    torch.cuda.synchronize()

    times_noreuse = []
    for _ in range(500):
        _ = w13[ids_a].to(torch.float16).sum()  # "layer N"
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[ids_b].to(torch.float16).sum()  # "layer N+1" - different experts
        torch.cuda.synchronize()
        times_noreuse.append(time.perf_counter() - start)

    # Scenario C: 32 experts (fills L2 budget: 32*1.89=60MB, fits in 96MB L2)
    ids_32_same = torch.tensor(list(range(32)), device=device)
    ids_32_diff = torch.tensor(list(range(64, 96)), device=device)

    for _ in range(100):
        _ = w13[ids_32_same].to(torch.float16).sum()
    torch.cuda.synchronize()

    times_32_reuse = []
    for _ in range(200):
        _ = w13[ids_32_same].to(torch.float16).sum()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[ids_32_same].to(torch.float16).sum()
        torch.cuda.synchronize()
        times_32_reuse.append(time.perf_counter() - start)

    times_32_noreuse = []
    for _ in range(200):
        _ = w13[ids_32_same].to(torch.float16).sum()
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = w13[ids_32_diff].to(torch.float16).sum()
        torch.cuda.synchronize()
        times_32_noreuse.append(time.perf_counter() - start)

    print(f"  8 experts, same set (L2 reuse):       {np.mean(times_reuse)*1e6:.1f} us")
    print(f"  8 experts, different set (L2 miss):    {np.mean(times_noreuse)*1e6:.1f} us")
    print(f"  8-expert temporal speedup:             {np.mean(times_noreuse)/np.mean(times_reuse):.3f}x")
    print()
    print(f"  32 experts, same set (L2 reuse):       {np.mean(times_32_reuse)*1e6:.1f} us")
    print(f"  32 experts, different set (L2 miss):   {np.mean(times_32_noreuse)*1e6:.1f} us")
    print(f"  32-expert temporal speedup:            {np.mean(times_32_noreuse)/np.mean(times_32_reuse):.3f}x")
    print()

    # ---- Test 4: Realistic batch scenario ----
    print("--- Test 4: Realistic B=32 scenario ---")
    print("B=32, top_k=8: 256 expert slots across 128 experts")
    print("On average ~2 tokens/expert, but distribution is skewed")

    np.random.seed(42)

    # Simulate skewed routing (matching real Gemma4 routing entropy ~6.4)
    # Zipf-like with alpha ~0.3 gives entropy around 6.4
    probs = np.array([(i+1)**(-0.3) for i in range(128)])
    probs /= probs.sum()

    # Generate 30 layers of routing decisions
    all_active_per_layer = []
    for _ in range(30):
        active = set()
        for _ in range(32):  # B=32 tokens
            selected = np.random.choice(128, 8, replace=False, p=probs)
            active.update(selected)
        all_active_per_layer.append(sorted(active))

    avg_active = np.mean([len(a) for a in all_active_per_layer])
    print(f"  Average active experts per layer: {avg_active:.0f}/128")

    # Pairwise overlap between consecutive layers
    overlaps = []
    for i in range(29):
        overlap = len(set(all_active_per_layer[i]) & set(all_active_per_layer[i+1]))
        overlaps.append(overlap)
    print(f"  Adjacent layer overlap: {np.mean(overlaps):.0f} experts (of ~{avg_active:.0f} active)")

    # Time: natural order vs reordered (hot experts first)
    # With reordering, the most-accessed experts are at indices 0-31
    # which means they're contiguous in memory

    indices_natural = [torch.tensor(a, device=device) for a in all_active_per_layer]

    # Reordered: map expert IDs so most popular are lowest indices
    reorder_map = np.argsort(-probs)  # most popular -> index 0
    reverse_map = np.argsort(reorder_map)  # maps old index -> new index
    indices_reordered = [torch.tensor(sorted(reverse_map[a].tolist()), device=device) for a in all_active_per_layer]

    # Time natural order
    for _ in range(50):
        for idx in indices_natural:
            _ = w13[idx].to(torch.float16).sum()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        for idx in indices_natural:
            _ = w13[idx].to(torch.float16).sum()
    end_event.record()
    torch.cuda.synchronize()
    natural_ms = start_event.elapsed_time(end_event) / 100

    # Time reordered
    for _ in range(50):
        for idx in indices_reordered:
            _ = w13[idx].to(torch.float16).sum()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(100):
        for idx in indices_reordered:
            _ = w13[idx].to(torch.float16).sum()
    end_event.record()
    torch.cuda.synchronize()
    reordered_ms = start_event.elapsed_time(end_event) / 100

    print(f"\n  30-layer pass (w13 only):")
    print(f"    Natural order:   {natural_ms:.2f} ms")
    print(f"    Reordered order: {reordered_ms:.2f} ms")
    print(f"    Speedup: {natural_ms/reordered_ms:.4f}x")

    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)


if __name__ == "__main__":
    main()
