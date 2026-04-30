"""Correctness + microbench for fused_unshuffle_weightedsum.

Run with:
    CUDA_VISIBLE_DEVICES=1 python kernels/triton/test_fused_unshuffle_weightedsum.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from kernels.triton.fused_unshuffle_weightedsum import (
    fused_unshuffle_weightedsum,
    ref_unshuffle_weightedsum,
)


def _make_inputs(
    M: int, K: int, topk: int, dtype: torch.dtype, device: str = "cuda", seed: int = 0
):
    g = torch.Generator(device=device).manual_seed(seed)
    M_sorted = M * topk
    c3 = torch.randn((M_sorted, K), dtype=dtype, device=device, generator=g)
    # c_map: sorted row indices [0..M_sorted) shuffled.
    perm = torch.randperm(M_sorted, device=device, generator=g).to(torch.int32)
    c_map = perm
    # topk weights: softmax-ish distribution summing to 1 per token.
    raw = torch.randn((M, topk), dtype=torch.float32, device=device, generator=g)
    weights = torch.softmax(raw, dim=1).to(dtype)
    return c3, c_map, weights


def run_correctness(M: int, K: int, topk: int, dtype: torch.dtype):
    c3, c_map, w = _make_inputs(M, K, topk, dtype)
    # vLLM path (matches production): BF16 accumulate via (x * w).sum(dim=1).
    ref = ref_unshuffle_weightedsum(c3, c_map, w, M, topk, out_dtype=dtype)
    got = fused_unshuffle_weightedsum(c3, c_map, w, M, topk, out_dtype=dtype)
    max_abs = (ref.float() - got.float()).abs().max().item()
    mean_abs = (ref.float() - got.float()).abs().mean().item()
    rel = max_abs / (ref.float().abs().max().item() + 1e-6)

    # Ground truth: FP32 accumulate. Our kernel accumulates in FP32 so it
    # should be *at least as accurate* as the vLLM reference. Gate on that.
    unshuffled32 = c3[c_map.long()].float()
    ref_fp32 = (
        unshuffled32.view(M, topk, K)
        * w.view(M, topk, 1).float()
    ).sum(dim=1).to(dtype)
    err_fused_vs_truth = (ref_fp32.float() - got.float()).abs().max().item()
    err_vllm_vs_truth = (ref_fp32.float() - ref.float()).abs().max().item()

    print(
        f"  correctness shape=(M={M}, K={K}, topk={topk}) dtype={dtype} : "
        f"vs_vllm: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e} | "
        f"vs_fp32_truth: fused={err_fused_vs_truth:.3e}, vllm={err_vllm_vs_truth:.3e}"
    )
    return {
        "max_abs": max_abs, "mean_abs": mean_abs, "rel": rel,
        "err_fused_vs_truth": err_fused_vs_truth,
        "err_vllm_vs_truth": err_vllm_vs_truth,
    }


def _bench(fn, warmup=25, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # microseconds


def run_microbench(M: int, K: int, topk: int, dtype: torch.dtype):
    c3, c_map, w = _make_inputs(M, K, topk, dtype)
    out_buf = torch.empty((M, K), dtype=dtype, device="cuda")

    def two_pass():
        # Mirror vLLM's exact code path.
        unshuffled = c3[c_map.long()]  # equivalent of ops.shuffle_rows
        out = (
            unshuffled.view(M, topk, K)
            * w.view(M, topk, 1).to(dtype)
        ).sum(dim=1)
        out_buf.copy_(out, non_blocking=True)

    def fused():
        fused_unshuffle_weightedsum(
            c3, c_map, w, M, topk, out_dtype=dtype, output=out_buf
        )

    us_two = _bench(two_pass)
    us_fused = _bench(fused)
    # HBM bytes (BF16 = 2 bytes):
    # two-pass: write M*topk*K + read M*topk*K + write M*K  = (2*topk+1)*M*K
    # fused:   read M*topk*K + write M*K                    = (topk+1)*M*K
    b_two = (2 * topk + 1) * M * K * 2
    b_fused = (topk + 1) * M * K * 2
    gb_s_two = b_two / (us_two * 1e-6) / 1e9
    gb_s_fused = b_fused / (us_fused * 1e-6) / 1e9
    speedup = us_two / us_fused if us_fused > 0 else 0.0
    print(
        f"  microbench shape=(M={M}, K={K}, topk={topk}) dtype={dtype} : "
        f"two-pass={us_two:.2f}us ({gb_s_two:.0f} GB/s) "
        f"fused={us_fused:.2f}us ({gb_s_fused:.0f} GB/s) "
        f"speedup={speedup:.2f}x "
        f"delta={us_two - us_fused:.2f}us"
    )
    return {
        "M": M, "K": K, "topk": topk, "dtype": str(dtype),
        "us_two_pass": round(us_two, 3),
        "us_fused": round(us_fused, 3),
        "speedup": round(speedup, 3),
        "us_saved": round(us_two - us_fused, 3),
        "gb_s_two": round(gb_s_two, 1),
        "gb_s_fused": round(gb_s_fused, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    ap.add_argument("--big-sweep", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    results: dict = {"correctness": [], "microbench": []}

    # Primary target: Qwen3-30B-A3B at C=1024 serving (M_active ~= 512, topk=8, K=2048).
    print("== Correctness ==")
    for M, K, topk, dt in [
        (512, 2048, 8, torch.bfloat16),
        (256, 2048, 8, torch.bfloat16),
        (128, 2048, 8, torch.bfloat16),
        (32, 2048, 8, torch.bfloat16),
        (512, 2048, 8, torch.float16),
    ]:
        results["correctness"].append(run_correctness(M, K, topk, dt))
    print()

    print("== Microbench ==")
    shapes = [
        (512, 2048, 8, torch.bfloat16),   # Qwen3 primary
        (256, 2048, 8, torch.bfloat16),
        (128, 2048, 8, torch.bfloat16),
        (64, 2048, 8, torch.bfloat16),
        (32, 2048, 8, torch.bfloat16),
        (1024, 2048, 8, torch.bfloat16),  # high concurrency
    ]
    if args.big_sweep:
        shapes.extend([
            (2048, 2048, 8, torch.bfloat16),
            (512, 4096, 8, torch.bfloat16),
            (512, 7168, 2, torch.bfloat16),  # Gemma4 shape
        ])
    for M, K, topk, dt in shapes:
        results["microbench"].append(run_microbench(M, K, topk, dt))

    # Pass/fail gates at Qwen3 shape.
    pri = results["microbench"][0]
    corr = results["correctness"][0]
    correctness_ok = corr["max_abs"] <= 1e-4 or corr["rel"] <= 5e-3
    speedup = pri["speedup"]
    us_saved = pri["us_saved"]

    verdict = "UNKNOWN"
    if not correctness_ok:
        verdict = "FAIL_CORRECTNESS"
    elif speedup >= 2.0:
        verdict = "BIG_WIN"
    elif speedup >= 1.3:
        verdict = "PASS"
    elif speedup >= 1.05:
        verdict = "MARGINAL"
    else:
        verdict = "KILL"
    results["verdict"] = verdict
    results["correctness_ok"] = correctness_ok
    results["primary_speedup"] = speedup
    results["primary_us_saved"] = us_saved

    print()
    print(f"== Verdict at Qwen3 (M=512, K=2048, topk=8): {verdict} ==")
    print(f"  correctness_ok={correctness_ok} speedup={speedup:.2f}x us_saved={us_saved:.2f}us")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
