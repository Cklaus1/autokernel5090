#!/usr/bin/env python3
"""Microbench: warp-spec v2 vs cp.async v1 vs baseline fusencache decode.

Default config:   B=16, seq_len=2048, Hq=16, Hk=8, D=256, page_size=16.

Prints elapsed us, bytes/s, % of 1792 GB/s peak, and a results.tsv line.
"""
import os
import sys
import argparse

# Pin to GPU 1 before torch imports CUDA — GPU 0 may be busy.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))

from build_fusencache_warpspec import build as build_ws, load_library as load_ws
from build_fusencache_cpasync import build as build_cp, load_library as load_cp
from build_fusencache import build_kernel as build_baseline, load_library as load_baseline

from kv_cache_gen.spec import PREDEFINED_SPECS


PEAK_GBPS = 1792.0


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
    slot_bytes = spec.slot_bytes(D)
    scale_elems = 2 * (D // min(spec.k_scale_block, spec.v_scale_block))
    scale_bytes = scale_elems * 2
    return B * Hk * seq_len * (slot_bytes + scale_bytes)


def _do_warpspec(q, kv, sc, bt, sl, spec, n_splits, D, Hk, Hq, ps, sm, cap, mid, out):
    kv_group_size = Hq // Hk
    torch.ops.fusencache_warpspec.decode_attention(
        out, q, kv, sc, bt, sl, mid,
        sm, cap, n_splits, D, Hk, kv_group_size, ps,
        spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
        spec.k_sym_offset, spec.v_sym_offset,
    )


def _do_cpasync(q, kv, sc, bt, sl, spec, n_splits, D, Hk, Hq, ps, sm, cap, mid, out):
    kv_group_size = Hq // Hk
    torch.ops.fusencache_cpasync.decode_attention(
        out, q, kv, sc, bt, sl, mid,
        sm, cap, n_splits, D, Hk, kv_group_size, ps,
        spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
        spec.k_sym_offset, spec.v_sym_offset,
    )


def _do_baseline(q, kv, sc, bt, sl, spec, n_splits, D, Hk, Hq, ps, sm, cap, mid, out):
    kv_group_size = Hq // Hk
    torch.ops.fusencache.decode_attention(
        out, q, kv, sc, bt, sl, mid,
        sm, cap, n_splits, D, Hk, kv_group_size, ps,
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
    ap.add_argument("--results_tsv", type=str, default=None)
    ap.add_argument("--verify_only", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "need CUDA"
    dev = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(dev)
    print(f"Device: {prop.name} (cc {prop.major}.{prop.minor}, SMs={prop.multi_processor_count})")

    print("[load] warpspec ...")
    if not load_ws():
        build_ws()
    print("[load] cpasync ...")
    if not load_cp():
        build_cp()
    print("[load] baseline ...")
    if not load_baseline():
        build_baseline()

    B, Hq, Hk, D = args.B, args.Hq, args.Hk, args.D
    seq_len, page_size, num_splits = args.seq_len, args.page_size, args.num_splits
    soft_cap = args.soft_cap

    query, kv_cache, scales, block_table, seq_lens, spec = \
        setup(B, Hq, Hk, D, seq_len, page_size=page_size)
    sm_scale = 1.0 / (D ** 0.5)

    mid_ws = torch.empty(B, Hq, num_splits, D + 1, dtype=torch.float32, device="cuda")
    out_ws = torch.empty(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    mid_cp = torch.empty_like(mid_ws)
    out_cp = torch.empty_like(out_ws)
    mid_bl = torch.empty_like(mid_ws)
    out_bl = torch.empty_like(out_ws)

    # --- correctness: warpspec vs baseline ---
    print("\n[correctness] running all three kernels ...")
    _do_baseline(query, kv_cache, scales, block_table, seq_lens, spec,
                 num_splits, D, Hk, Hq, page_size, sm_scale, soft_cap, mid_bl, out_bl)
    _do_cpasync(query, kv_cache, scales, block_table, seq_lens, spec,
                num_splits, D, Hk, Hq, page_size, sm_scale, soft_cap, mid_cp, out_cp)
    _do_warpspec(query, kv_cache, scales, block_table, seq_lens, spec,
                 num_splits, D, Hk, Hq, page_size, sm_scale, soft_cap, mid_ws, out_ws)
    torch.cuda.synchronize()

    bl = out_bl.float()
    cp = out_cp.float()
    ws = out_ws.float()
    diff_ws_bl = (ws - bl).abs()
    diff_cp_bl = (cp - bl).abs()
    print(f"  baseline NaN: {torch.isnan(bl).any().item()}")
    print(f"  cpasync  NaN: {torch.isnan(cp).any().item()}, "
          f"max|cp-bl|={diff_cp_bl.max().item():.3e}, "
          f"mean|cp-bl|={diff_cp_bl.mean().item():.3e}")
    print(f"  warpspec NaN: {torch.isnan(ws).any().item()}, "
          f"max|ws-bl|={diff_ws_bl.max().item():.3e}, "
          f"mean|ws-bl|={diff_ws_bl.mean().item():.3e}")
    ws_pass = (not torch.isnan(ws).any().item()) and diff_ws_bl.max().item() < 1e-3

    if args.verify_only:
        print("verify_only: exit")
        return

    bytes_per_iter = bytes_moved(B, Hq, Hk, D, seq_len, spec)
    print(f"\nConfig: B={B} seq_len={seq_len} Hq={Hq} Hk={Hk} D={D} "
          f"splits={num_splits} page_size={page_size}")
    print(f"Traffic/iter: {bytes_per_iter/1024/1024:.2f} MiB "
          f"(floor at peak = {bytes_per_iter/(PEAK_GBPS*1e9)*1e6:.1f} us)")

    results = {}

    for name, fn_impl in [
        ("warpspec v2", lambda: _do_warpspec(query, kv_cache, scales, block_table, seq_lens,
                                             spec, num_splits, D, Hk, Hq, page_size,
                                             sm_scale, soft_cap, mid_ws, out_ws)),
        ("cp.async v1", lambda: _do_cpasync(query, kv_cache, scales, block_table, seq_lens,
                                            spec, num_splits, D, Hk, Hq, page_size,
                                            sm_scale, soft_cap, mid_cp, out_cp)),
        ("baseline C++", lambda: _do_baseline(query, kv_cache, scales, block_table, seq_lens,
                                              spec, num_splits, D, Hk, Hq, page_size,
                                              sm_scale, soft_cap, mid_bl, out_bl)),
    ]:
        try:
            fn_impl()
            torch.cuda.synchronize()
            t_us = timeit(fn_impl, iters=args.iters)
            gbps = bytes_per_iter / (t_us * 1e-6) / 1e9
            pct = gbps / PEAK_GBPS * 100.0
            results[name] = (t_us, gbps, pct)
            print(f"{name:<15}: {t_us:8.2f} us   {gbps:8.1f} GB/s   {pct:5.1f}% of peak")
        except Exception as e:
            print(f"[{name} FAIL] {e}")
            results[name] = (float("nan"), 0.0, 0.0)

    ws_us = results["warpspec v2"][0]
    cp_us = results["cp.async v1"][0]
    bl_us = results["baseline C++"][0]
    ws_pct = results["warpspec v2"][2]

    if ws_us > 0 and cp_us > 0:
        sp_vs_cp = cp_us / ws_us
        print(f"\nSpeedup warpspec v2 vs cp.async v1: {sp_vs_cp:.2f}x")
    else:
        sp_vs_cp = 0.0

    if ws_us > 0 and bl_us > 0:
        sp_vs_bl = bl_us / ws_us
        print(f"Speedup warpspec v2 vs baseline:    {sp_vs_bl:.2f}x")
    else:
        sp_vs_bl = 0.0

    # Kill criteria
    if not ws_pass:
        verdict = "KILL"
    elif ws_pct < 30.0:
        verdict = "KILL"
    elif ws_pct < 50.0:
        verdict = "PARTIAL"
    else:
        verdict = "PASS"
    print(f"\nVerdict: {verdict} (ws correctness {'OK' if ws_pass else 'FAIL'}, BW={ws_pct:.1f}%)")

    if args.results_tsv:
        line = "\t".join([
            "fusencache_warpspec_v2",
            "warpspec_v2",
            "kernel_bench",
            "0",
            f"{ws_us:.2f}",
            f"{ws_pct:.1f}",
            f"{sp_vs_cp:.2f}",
            verdict,
            "0",
            f"warpspec v2: {ws_pct:.1f}% BW, {sp_vs_cp:.2f}x vs cp.async v1 "
            f"(B={B} seq={seq_len} H={Hq}, max|ws-bl|={diff_ws_bl.max().item():.2e})",
        ])
        with open(args.results_tsv, "a") as f:
            f.write(line + "\n")
        print(f"\nAppended to {args.results_tsv}:\n{line}")


if __name__ == "__main__":
    main()
