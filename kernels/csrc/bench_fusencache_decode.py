#!/usr/bin/env python3
"""
Benchmark: CUDA C++ decode attention vs Triton decode attention.

Measures latency at various batch sizes and sequence lengths.
The key metric is whether the CUDA kernel is fast enough to benefit
from CUDA graph capture (eliminating Triton JIT overhead).
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fusencache import build_kernel, load_library

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kv_cache_gen.spec import PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn


def setup_data(B, Hq, Hk, D, seq_len, page_size=16, device='cuda'):
    spec = PREDEFINED_SPECS['k4v4b64']
    slot_bytes = spec.slot_bytes(D)
    max_blocks = (seq_len + page_size - 1) // page_size
    total_blocks = B * max_blocks + 10
    kv_cache = torch.randint(0, 256, (total_blocks, page_size, Hk, slot_bytes),
                             dtype=torch.uint8, device=device)
    block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        for blk in range(max_blocks):
            block_table[b, blk] = b * max_blocks + blk
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    query = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=device) * 0.1
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // min_block
    max_slots = total_blocks * page_size
    scales = torch.randn(max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device) * 0.5 + 1.0
    return query, kv_cache, scales, block_table, seq_lens, spec


def bench_triton(query, kv_cache, scales, block_table, seq_lens, spec,
                 num_splits, Hk, soft_cap, warmup=10, iters=100):
    B, Hq, D = query.shape
    sm_scale = 1.0 / (D ** 0.5)
    decode_fn = make_decode_fn(spec, block_kv=16, num_kv_splits=num_splits,
                               logits_soft_cap=soft_cap, persistent_buffers=True)
    # Warmup
    for _ in range(warmup):
        decode_fn(query, kv_cache, scales, block_table, seq_lens, sm_scale, Hk)
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        decode_fn(query, kv_cache, scales, block_table, seq_lens, sm_scale, Hk)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def bench_cuda(query, kv_cache, scales, block_table, seq_lens, spec,
               num_splits, Hk, Hq, D, soft_cap, warmup=10, iters=100):
    B = query.shape[0]
    sm_scale = 1.0 / (D ** 0.5)
    kv_group_size = Hq // Hk
    page_size = kv_cache.shape[1]

    mid_out = torch.empty(B, Hq, num_splits, D + 1,
                          dtype=torch.float32, device=query.device)
    output = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=query.device)

    def run():
        torch.ops.fusencache.decode_attention(
            output, query, kv_cache, scales, block_table, seq_lens, mid_out,
            sm_scale, soft_cap, num_splits, D, Hk, kv_group_size, page_size,
            spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
            spec.k_sym_offset, spec.v_sym_offset,
        )

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def bench_cuda_graph(query, kv_cache, scales, block_table, seq_lens, spec,
                     num_splits, Hk, Hq, D, soft_cap, warmup=10, iters=100):
    """Benchmark with CUDA graph capture — the whole point of the C++ kernel."""
    B = query.shape[0]
    sm_scale = 1.0 / (D ** 0.5)
    kv_group_size = Hq // Hk
    page_size = kv_cache.shape[1]

    mid_out = torch.empty(B, Hq, num_splits, D + 1,
                          dtype=torch.float32, device=query.device)
    output = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=query.device)

    # Warmup (required before capture)
    for _ in range(3):
        torch.ops.fusencache.decode_attention(
            output, query, kv_cache, scales, block_table, seq_lens, mid_out,
            sm_scale, soft_cap, num_splits, D, Hk, kv_group_size, page_size,
            spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
            spec.k_sym_offset, spec.v_sym_offset,
        )
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.fusencache.decode_attention(
            output, query, kv_cache, scales, block_table, seq_lens, mid_out,
            sm_scale, soft_cap, num_splits, D, Hk, kv_group_size, page_size,
            spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
            spec.k_sym_offset, spec.v_sym_offset,
        )

    # Warmup graph replay
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    # Measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    print("=" * 70)
    print("FusenCache Decode Attention Benchmark: CUDA C++ vs Triton")
    print("=" * 70)

    if not load_library():
        print("\n[BUILD] Building CUDA kernel...")
        build_kernel()

    Hq, Hk, D = 16, 8, 256
    soft_cap = 50.0
    device = 'cuda'

    configs = [
        # (B, seq_len, num_splits)
        (1,   64,   16),
        (1,   256,  16),
        (1,   512,  32),
        (1,   1024, 32),
        (8,   64,   16),
        (8,   256,  16),
        (8,   512,  32),
        (32,  64,   16),
        (32,  256,  16),
        (64,  64,   16),
        (128, 64,   16),
        (240, 64,   16),
    ]

    print(f"\nConfig: Hq={Hq}, Hk={Hk}, D={D}, soft_cap={soft_cap}")
    print(f"{'B':>5} {'SeqLen':>7} {'Splits':>7} {'Triton(us)':>11} "
          f"{'CUDA(us)':>9} {'CUDAGraph(us)':>14} {'Speedup':>8}")
    print("-" * 70)

    for B, seq_len, num_splits in configs:
        query, kv_cache, scales, block_table, seq_lens, spec = \
            setup_data(B, Hq, Hk, D, seq_len, device=device)

        try:
            t_triton = bench_triton(query, kv_cache, scales, block_table,
                                    seq_lens, spec, num_splits, Hk, soft_cap)
        except Exception as e:
            t_triton = float('nan')

        try:
            t_cuda = bench_cuda(query, kv_cache, scales, block_table,
                                seq_lens, spec, num_splits, Hk, Hq, D, soft_cap)
        except Exception as e:
            t_cuda = float('nan')

        try:
            t_graph = bench_cuda_graph(query, kv_cache, scales, block_table,
                                       seq_lens, spec, num_splits, Hk, Hq, D, soft_cap)
        except Exception as e:
            t_graph = float('nan')

        speedup = t_triton / t_graph if t_graph > 0 else float('nan')

        print(f"{B:>5} {seq_len:>7} {num_splits:>7} {t_triton:>11.1f} "
              f"{t_cuda:>9.1f} {t_graph:>14.1f} {speedup:>7.2f}x")

    print()
    print("Key: Speedup = Triton / CUDAGraph (higher is better)")
    print("The CUDAGraph column is the target for vLLM integration.")


if __name__ == "__main__":
    main()
