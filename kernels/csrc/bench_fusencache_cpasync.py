#!/usr/bin/env python3
"""Microbench: cp.async prototype vs baseline fusencache decode.

Single config per task:   B=16, seq_len=2048, Hq=16, Hk=8, D=256, page_size=16.

Prints elapsed us, bytes/s, % of 1792 GB/s peak, and a results.tsv line.
"""
import os
import sys
import time
import argparse

# Pin to GPU 1 before torch imports CUDA — GPU 0 is busy with vllm-ad.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))

from build_fusencache_cpasync import build as build_cpasync, load_library as load_cpasync
from build_fusencache import build_kernel as build_baseline, load_library as load_baseline

from kv_cache_gen.spec import PREDEFINED_SPECS


PEAK_GBPS = 1792.0  # RTX PRO 6000 Max-Q theoretical peak


def setup(B, Hq, Hk, D, seq_len, page_size=16, device="cuda", seed=0):
    torch.manual_seed(seed)
    spec = PREDEFINED_SPECS["k4v4b64"]
    slot_bytes = spec.slot_bytes(D)
    max_blocks = (seq_len + page_size - 1) // page_size
    total_blocks = B * max_blocks + 4
    kv_cache = torch.randint(0, 256, (total_blocks, page_size, Hk, slot_bytes),
                             dtype=torch.uint8, device=device)
    block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        for blk in range(max_blocks):
            block_table[b, blk] = b * max_blocks + blk
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
    query = (torch.randn(B, Hq, D, dtype=torch.bfloat16, device=device) * 0.1)
    min_sb = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // min_sb
    max_slots = total_blocks * page_size
    scales = (torch.randn(max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device) * 0.5 + 1.0)
    return query, kv_cache, scales, block_table, seq_lens, spec


def bytes_moved(B, Hq, Hk, D, seq_len, spec):
    # Per decode step: read K+V for every token, scales too.
    # For group_size=2, each KV head is shared by 2 Q heads so we count Hk (not Hq).
    slot_bytes = spec.slot_bytes(D)                 # K + V bytes per tok per kv-head
    scale_elems = 2 * (D // min(spec.k_scale_block, spec.v_scale_block))  # K+V
    scale_bytes = scale_elems * 2                    # fp16
    return B * Hk * seq_len * (slot_bytes + scale_bytes)


def _do_baseline(query, kv_cache, scales, block_table, seq_lens, spec,
                 num_splits, D, Hk, Hq, page_size, sm_scale, soft_cap,
                 mid_out, output):
    kv_group_size = Hq // Hk
    torch.ops.fusencache.decode_attention(
        output, query, kv_cache, scales, block_table, seq_lens, mid_out,
        sm_scale, soft_cap, num_splits, D, Hk, kv_group_size, page_size,
        spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
        spec.k_sym_offset, spec.v_sym_offset,
    )


def _do_cpasync(query, kv_cache, scales, block_table, seq_lens, spec,
                num_splits, D, Hk, Hq, page_size, sm_scale, soft_cap,
                mid_out, output):
    kv_group_size = Hq // Hk
    torch.ops.fusencache_cpasync.decode_attention(
        output, query, kv_cache, scales, block_table, seq_lens, mid_out,
        sm_scale, soft_cap, num_splits, D, Hk, kv_group_size, page_size,
        spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
        spec.k_sym_offset, spec.v_sym_offset,
    )


def timeit(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--Hq", type=int, default=16)
    ap.add_argument("--Hk", type=int, default=8)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--page_size", type=int, default=16)
    ap.add_argument("--num_splits", type=int, default=16)
    ap.add_argument("--soft_cap", type=float, default=50.0)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--skip_baseline", action="store_true")
    ap.add_argument("--results_tsv", type=str, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    print(f"Device: {prop.name} (cc {prop.major}.{prop.minor}, "
          f"smem/SM={prop.shared_memory_per_multiprocessor//1024}KB, SMs={prop.multi_processor_count})")

    print("[load] cpasync ...")
    if not load_cpasync():
        build_cpasync()

    if not args.skip_baseline:
        print("[load] baseline ...")
        if not load_baseline():
            build_baseline()

    B, Hq, Hk, D = args.B, args.Hq, args.Hk, args.D
    seq_len, page_size, num_splits = args.seq_len, args.page_size, args.num_splits

    query, kv_cache, scales, block_table, seq_lens, spec = \
        setup(B, Hq, Hk, D, seq_len, page_size=page_size)
    sm_scale = 1.0 / (D ** 0.5)
    soft_cap = args.soft_cap
    kv_group_size = Hq // Hk

    mid_out = torch.empty(B, Hq, num_splits, D + 1, dtype=torch.float32, device="cuda")
    output  = torch.empty(B, Hq, D, dtype=torch.bfloat16, device="cuda")

    bytes_per_iter = bytes_moved(B, Hq, Hk, D, seq_len, spec)
    print(f"\nConfig: B={B} seq_len={seq_len} Hq={Hq} Hk={Hk} D={D} "
          f"splits={num_splits} page_size={page_size}")
    print(f"Traffic/iter: {bytes_per_iter/1024/1024:.2f} MiB "
          f"(bytes_floor at peak = {bytes_per_iter/(PEAK_GBPS*1e9)*1e6:.1f} us)")

    results = {}

    # cp.async
    try:
        fn = lambda: _do_cpasync(query, kv_cache, scales, block_table, seq_lens,
                                 spec, num_splits, D, Hk, Hq, page_size,
                                 sm_scale, soft_cap, mid_out, output)
        # warmup once with check
        fn()
        torch.cuda.synchronize()
        out = output.clone()
        has_nan = torch.isnan(out).any().item()
        t_us = timeit(fn, iters=args.iters)
        gbps = bytes_per_iter / (t_us * 1e-6) / 1e9
        pct = gbps / PEAK_GBPS * 100.0
        results["cpasync"] = (t_us, gbps, pct, has_nan)
        print(f"cp.async prototype: {t_us:8.2f} us   {gbps:8.1f} GB/s   "
              f"{pct:5.1f}% of peak   NaN={has_nan}")
    except Exception as e:
        print(f"[cpasync FAIL] {e}")
        results["cpasync"] = (float("nan"), 0.0, 0.0, True)

    if not args.skip_baseline:
        try:
            fn = lambda: _do_baseline(query, kv_cache, scales, block_table, seq_lens,
                                      spec, num_splits, D, Hk, Hq, page_size,
                                      sm_scale, soft_cap, mid_out, output)
            fn()
            torch.cuda.synchronize()
            t_us = timeit(fn, iters=args.iters)
            gbps = bytes_per_iter / (t_us * 1e-6) / 1e9
            pct = gbps / PEAK_GBPS * 100.0
            results["baseline"] = (t_us, gbps, pct, False)
            print(f"baseline          : {t_us:8.2f} us   {gbps:8.1f} GB/s   "
                  f"{pct:5.1f}% of peak")
        except Exception as e:
            print(f"[baseline FAIL] {e}")
            results["baseline"] = (float("nan"), 0.0, 0.0, True)
    else:
        results["baseline"] = (float("nan"), 0.0, 0.0, True)

    # Speedup
    if results["cpasync"][0] > 0 and results["baseline"][0] > 0:
        speedup = results["baseline"][0] / results["cpasync"][0]
        print(f"\nSpeedup cp.async vs baseline: {speedup:.2f}x")
    else:
        speedup = 0.0

    # Append results.tsv row if asked
    if args.results_tsv:
        status = "PASS"
        cp_us, _, cp_pct, cp_nan = results["cpasync"]
        if cp_nan or cp_us != cp_us:  # nan
            status = "FAIL"
        line = "\t".join([
            "fusencache_cpasync_prototype",
            "decode_cpasync",
            "kernel_bench",
            "0",
            f"{cp_us:.2f}",
            f"{cp_pct:.1f}",
            f"{speedup:.2f}",
            status,
            "0",
            f"cp.async 3-stage: {cp_pct:.1f}% BW, {speedup:.2f}x vs baseline C++ decode "
            f"(B={B} seq={seq_len} H={Hq})",
        ])
        with open(args.results_tsv, "a") as f:
            f.write(line + "\n")
        print(f"\nAppended to {args.results_tsv}:\n{line}")


if __name__ == "__main__":
    main()
