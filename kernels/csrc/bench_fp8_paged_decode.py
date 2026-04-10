#!/usr/bin/env python3
"""
Benchmark FP8 Paged Decode Attention: CUDA C++ vs Triton FP8 vs FA2 BF16.

Target: 165us at B=32, seq=2048, d=256
FA2 BF16 baseline: ~323us
Triton FP8 baseline: ~430us
"""

import sys
import os
import math
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.fp8_decode_attention import (
    fp8_decode_attention as triton_fp8_decode,
    create_test_data,
)
from kernels.csrc.build_fp8_decode import build_kernel, load_library, SO_PATH

if not load_library():
    print("Building CUDA kernel...")
    build_kernel()
else:
    print(f"Loaded pre-built kernel from {SO_PATH}")


def cuda_fp8_decode(
    q, k_cache_fp8, v_cache_fp8, block_table, seq_lens,
    k_scale, v_scale, sm_scale=None, logits_soft_cap=0.0,
    num_kv_splits=0,
):
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache_fp8.shape[2]
    block_size = k_cache_fp8.shape[1]
    kv_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    max_seq = seq_lens.max().item()
    num_pages = (max_seq + block_size - 1) // block_size

    if num_kv_splits == 0:
        GQA_H = min(2, kv_group_size) if kv_group_size >= 2 else 1
        num_head_groups = (num_q_heads + GQA_H - 1) // GQA_H
        base_blocks = batch * num_head_groups
        target_blocks = 170 * 4
        if base_blocks >= target_blocks:
            num_kv_splits = 1
        else:
            num_kv_splits = max(1, min(num_pages, target_blocks // max(base_blocks, 1)))
            pages_per_split = num_pages // num_kv_splits
            while pages_per_split < 2 and num_kv_splits > 1:
                num_kv_splits = num_kv_splits // 2
                pages_per_split = num_pages // num_kv_splits

    per_head_scale = 1 if k_scale.numel() > 1 else 0

    mid_out = torch.zeros(
        (batch, num_q_heads, num_kv_splits, head_dim + 1),
        dtype=torch.float32, device=q.device)
    output = torch.empty_like(q)

    torch.ops.fp8_decode.paged_attention(
        output, q,
        k_cache_fp8, v_cache_fp8,
        k_scale, v_scale,
        block_table, seq_lens,
        mid_out,
        sm_scale, logits_soft_cap,
        num_kv_splits, head_dim, num_kv_heads,
        kv_group_size, block_size, per_head_scale,
    )
    return output


def benchmark_fn(fn, warmup=20, iters=100):
    """Benchmark a function using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Use median
    median_ms = times[len(times) // 2]
    mean_ms = sum(times) / len(times)
    return median_ms * 1000, mean_ms * 1000  # convert to microseconds


def compute_bandwidth(batch, num_q_heads, num_kv_heads, head_dim, seq_len,
                      bytes_per_kv_elem, latency_us):
    """Compute achieved bandwidth in GB/s."""
    # Data read: Q (once) + K + V from cache
    q_bytes = batch * num_q_heads * head_dim * 2  # BF16
    kv_bytes = batch * seq_len * num_kv_heads * head_dim * bytes_per_kv_elem * 2  # K + V
    total_bytes = q_bytes + kv_bytes
    bandwidth_gbs = total_bytes / (latency_us * 1e-6) / 1e9
    return bandwidth_gbs, total_bytes


def main():
    torch.manual_seed(42)

    # RTX 5090 peak bandwidth: ~1792 GB/s (GDDR7)
    PEAK_BW_GBS = 1792.0

    configs = [
        # (batch, num_q_heads, num_kv_heads, head_dim, seq_len, name)
        (32, 16, 8, 256, 2048, "B=32 seq=2048 d=256 (target config)"),
        (32, 16, 8, 256, 4096, "B=32 seq=4096 d=256"),
        (64, 16, 8, 256, 2048, "B=64 seq=2048 d=256"),
        (32, 32, 8, 128, 2048, "B=32 seq=2048 d=128 GQA=4"),
        (8, 16, 8, 256, 8192, "B=8 seq=8192 d=256"),
    ]

    print("\n" + "=" * 80)
    print("FP8 Paged Decode Attention Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Peak memory bandwidth: {PEAK_BW_GBS:.0f} GB/s")
    print("=" * 80)

    for batch, nqh, nkh, hd, sl, name in configs:
        print(f"\n--- {name} ---")

        data = create_test_data(
            batch=batch, num_q_heads=nqh, num_kv_heads=nkh,
            head_dim=hd, seq_len=sl, block_size=16)

        # CUDA FP8
        try:
            cuda_lat, cuda_mean = benchmark_fn(lambda: cuda_fp8_decode(
                data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
                data["block_table"], data["seq_lens"],
                data["k_scale"], data["v_scale"]))
            cuda_bw, cuda_bytes = compute_bandwidth(
                batch, nqh, nkh, hd, sl, 1, cuda_lat)  # FP8 = 1 byte
            cuda_pct = cuda_bw / PEAK_BW_GBS * 100
            print(f"  CUDA  FP8: {cuda_lat:8.1f} us (mean={cuda_mean:.1f}), "
                  f"BW={cuda_bw:.0f} GB/s ({cuda_pct:.1f}% peak)")
        except Exception as e:
            print(f"  CUDA  FP8: ERROR - {e}")
            cuda_lat = float('inf')

        # Triton FP8
        try:
            triton_lat, triton_mean = benchmark_fn(lambda: triton_fp8_decode(
                data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
                data["block_table"], data["seq_lens"],
                data["k_scale"], data["v_scale"]))
            triton_bw, _ = compute_bandwidth(
                batch, nqh, nkh, hd, sl, 1, triton_lat)
            triton_pct = triton_bw / PEAK_BW_GBS * 100
            print(f"  Triton FP8: {triton_lat:8.1f} us (mean={triton_mean:.1f}), "
                  f"BW={triton_bw:.0f} GB/s ({triton_pct:.1f}% peak)")
        except Exception as e:
            print(f"  Triton FP8: ERROR - {e}")
            triton_lat = float('inf')

        # Speedup
        if cuda_lat < float('inf') and triton_lat < float('inf'):
            speedup = triton_lat / cuda_lat
            print(f"  Speedup (CUDA over Triton): {speedup:.2f}x")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
