#!/usr/bin/env python3
"""
Benchmark: Fused RMSNorm+FP4 vs Separate RMSNorm + scaled_fp4_quant.

Usage (inside Docker after build_and_install.py):
    python3 /tmp/csrc/benchmark.py
"""

import time
import torch


def load_fused_kernel():
    """Load the fused kernel if not already loaded."""
    SO_PATH = "/tmp/build_fused_rms_norm_fp4/fused_rms_norm_fp4.so"
    torch.ops.load_library(SO_PATH)


def benchmark_separate(x, w, gs, epsilon, warmup=50, iters=200):
    """Benchmark: separate RMSNorm + scaled_fp4_quant."""
    from vllm._custom_ops import scaled_fp4_quant
    M, N = x.shape

    # Warmup
    for _ in range(warmup):
        # RMSNorm
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = (x.float() * torch.rsqrt(variance + epsilon) * w.float()).to(x.dtype)
        # FP4 quant
        out_fp4, out_sf = scaled_fp4_quant(normed, gs)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = (x.float() * torch.rsqrt(variance + epsilon) * w.float()).to(x.dtype)
        out_fp4, out_sf = scaled_fp4_quant(normed, gs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1e6  # microseconds


def benchmark_separate_vllm_ops(x, w, gs, epsilon, warmup=50, iters=200):
    """Benchmark: vLLM rms_norm C++ + scaled_fp4_quant."""
    M, N = x.shape

    # Use vLLM's C++ RMSNorm
    normed = torch.empty_like(x)

    # Warmup
    for _ in range(warmup):
        torch.ops._C.rms_norm(normed, x, w, epsilon)
        from vllm._custom_ops import scaled_fp4_quant
        out_fp4, out_sf = scaled_fp4_quant(normed, gs)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.ops._C.rms_norm(normed, x, w, epsilon)
        from vllm._custom_ops import scaled_fp4_quant
        out_fp4, out_sf = scaled_fp4_quant(normed, gs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1e6


def benchmark_fused(x, w, gs, epsilon, warmup=50, iters=200):
    """Benchmark: fused RMSNorm + FP4 quant."""
    M, N = x.shape
    out_fp4 = torch.empty(M, N // 2, device="cuda", dtype=torch.uint8)
    out_sf = torch.empty(M, N // 16, device="cuda", dtype=torch.uint8)

    # Warmup
    for _ in range(warmup):
        torch.ops._C.rms_norm_dynamic_fp4_quant(
            out_fp4, out_sf, x, w, gs, epsilon, False
        )

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.ops._C.rms_norm_dynamic_fp4_quant(
            out_fp4, out_sf, x, w, gs, epsilon, False
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters * 1e6


def main():
    load_fused_kernel()
    # Also load vLLM ops
    import vllm._custom_ops

    print("=" * 70)
    print("Benchmark: Fused RMSNorm+FP4 vs Separate")
    print("=" * 70)

    epsilon = 1e-6

    # Test various shapes matching Gemma4 model dimensions
    shapes = [
        (1, 3584),      # decode, hidden_size
        (1, 14336),     # decode, MLP intermediate
        (4, 3584),      # small batch
        (8, 3584),      # medium batch
        (32, 3584),     # larger batch
        (128, 3584),    # prefill chunk
        (1, 4096),      # common hidden size
        (1, 8192),      # larger model
    ]

    print(f"\n{'Shape':>15}  {'Separate (us)':>14}  {'Fused (us)':>12}  {'Speedup':>8}  {'Saved (us)':>11}")
    print("-" * 70)

    for M, N in shapes:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        try:
            t_sep = benchmark_separate_vllm_ops(x, w, gs, epsilon)
        except Exception:
            t_sep = benchmark_separate(x, w, gs, epsilon)

        t_fused = benchmark_fused(x, w, gs, epsilon)
        speedup = t_sep / t_fused
        saved = t_sep - t_fused

        print(f"  ({M:>4}, {N:>5})  {t_sep:>12.1f}us  {t_fused:>10.1f}us  {speedup:>6.2f}x  {saved:>9.1f}us")

    # Estimate model-level impact for Gemma4 26B
    print("\n" + "=" * 70)
    print("Gemma4 26B Impact Estimate")
    print("=" * 70)

    # 60 norm+quant pairs per decode step
    x_gemma = torch.randn(1, 3584, device="cuda", dtype=torch.bfloat16)
    w_gemma = torch.randn(3584, device="cuda", dtype=torch.bfloat16)
    gs_gemma = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    try:
        t_sep = benchmark_separate_vllm_ops(x_gemma, w_gemma, gs_gemma, epsilon, warmup=100, iters=500)
    except Exception:
        t_sep = benchmark_separate(x_gemma, w_gemma, gs_gemma, epsilon, warmup=100, iters=500)
    t_fused = benchmark_fused(x_gemma, w_gemma, gs_gemma, epsilon, warmup=100, iters=500)

    n_layers = 60  # approximate for Gemma4 26B
    total_sep_us = t_sep * n_layers
    total_fused_us = t_fused * n_layers
    saved_per_step_us = total_sep_us - total_fused_us

    print(f"  Per norm+quant: {t_sep:.1f}us (separate) -> {t_fused:.1f}us (fused)")
    print(f"  Speedup: {t_sep/t_fused:.2f}x per invocation")
    print(f"  60 layers/step: {total_sep_us:.0f}us -> {total_fused_us:.0f}us")
    print(f"  Saved per step: {saved_per_step_us:.0f}us ({saved_per_step_us/1000:.1f}ms)")

    # Current decode: ~8.24ms/step = 6615 tok/s
    current_step_us = 8240
    new_step_us = current_step_us - saved_per_step_us
    current_tps = 1e6 / current_step_us
    new_tps = 1e6 / new_step_us
    improvement = (new_tps - current_tps) / current_tps * 100

    print(f"\n  Estimated decode step: {current_step_us}us -> {new_step_us:.0f}us")
    print(f"  Throughput: {current_tps:.0f} tok/s -> {new_tps:.0f} tok/s (+{improvement:.1f}%)")


if __name__ == "__main__":
    main()
