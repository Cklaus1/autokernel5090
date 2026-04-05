#!/usr/bin/env python3
"""FusenCache kernel benchmark — AutoKernel style.

Measures TFLOPS, GB/s, % peak, roofline for the FusenCache Triton decode kernel.
Tests across multiple shapes: single decode, batch decode, various seq lengths.

Usage:
    python bench_fusencache_kernel.py
"""

import math
import time
import torch
from triton.testing import do_bench
from dataclasses import dataclass


# =========================================================================
# GPU Spec
# =========================================================================

@dataclass
class GPUSpec:
    name: str
    peak_tflops_fp16: float
    peak_bandwidth_gb_s: float

    @staticmethod
    def detect():
        props = torch.cuda.get_device_properties(0)
        name = props.name
        # Known GPUs
        if "5090" in name:
            return GPUSpec(name=name, peak_tflops_fp16=419.0,
                           peak_bandwidth_gb_s=1792.0)
        elif "4090" in name:
            return GPUSpec(name=name, peak_tflops_fp16=330.0,
                           peak_bandwidth_gb_s=1008.0)
        else:
            return GPUSpec(name=name, peak_tflops_fp16=200.0,
                           peak_bandwidth_gb_s=1000.0)


# =========================================================================
# FLOPs and Bytes calculation for decode attention
# =========================================================================

def decode_attention_flops(B, Hq, D, seq_len):
    """FLOPs for one decode step: Q@K^T + softmax + P@V."""
    # QK^T: B * Hq * seq_len * D * 2 (multiply-add)
    qk_flops = B * Hq * seq_len * D * 2
    # Softmax: ~5 * B * Hq * seq_len (exp, sum, div, etc.)
    softmax_flops = B * Hq * seq_len * 5
    # PV: B * Hq * seq_len * D * 2
    pv_flops = B * Hq * seq_len * D * 2
    return qk_flops + softmax_flops + pv_flops


def decode_attention_bytes_fp16(B, Hq, Hk, D, seq_len):
    """Bytes loaded/stored for FP16 decode attention."""
    # Read Q: B * Hq * D * 2
    q_bytes = B * Hq * D * 2
    # Read K: B * seq_len * Hk * D * 2
    k_bytes = B * seq_len * Hk * D * 2
    # Read V: B * seq_len * Hk * D * 2
    v_bytes = B * seq_len * Hk * D * 2
    # Write O: B * Hq * D * 2
    o_bytes = B * Hq * D * 2
    return q_bytes + k_bytes + v_bytes + o_bytes


def decode_attention_bytes_fusen(B, Hq, Hk, D, seq_len):
    """Bytes loaded/stored for FusenCache v1 decode (FP8 K + int4 V)."""
    q_bytes = B * Hq * D * 2  # Q is fp16/bf16
    k_bytes = B * seq_len * Hk * D * 1  # K is FP8 (1 byte)
    v_bytes = B * seq_len * Hk * (D // 2) * 1  # V is int4 packed (0.5 bytes)
    v_scale_bytes = B * seq_len * Hk * 2  # V scales (fp16)
    o_bytes = B * Hq * D * 2  # output fp16
    return q_bytes + k_bytes + v_bytes + v_scale_bytes + o_bytes


# =========================================================================
# Benchmark runner
# =========================================================================

def bench_fusencache_kernel(B, Hq, Hk, D, seq_len, device="cuda"):
    """Benchmark the FusenCache Triton decode kernel."""
    from vllm.v1.attention.ops.triton_fusencache_decode import (
        fusencache_decode_attention,
    )

    block_size = 16
    num_blocks = (seq_len + block_size - 1) // block_size + 2
    slot_size = D + D // 2
    max_slots = num_blocks * block_size

    # Create inputs
    query = torch.randn(B, Hq, D, device=device, dtype=torch.bfloat16)
    kv_cache = torch.randint(0, 255, (num_blocks, block_size, Hk, slot_size),
                              device=device, dtype=torch.uint8)
    v_scales = torch.ones(max_slots, Hk, device=device, dtype=torch.float16) * 0.1
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)
    block_table = block_table.unsqueeze(0).expand(B, -1).contiguous()
    seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

    # Warmup
    for _ in range(3):
        fusencache_decode_attention(query, kv_cache, v_scales, block_table,
                                    seq_lens, 1.0 / math.sqrt(D), Hk)
    torch.cuda.synchronize()

    # Benchmark
    ms = do_bench(
        lambda: fusencache_decode_attention(
            query, kv_cache, v_scales, block_table,
            seq_lens, 1.0 / math.sqrt(D), Hk),
        warmup=25, rep=100,
    )

    return ms


def bench_pytorch_reference(B, Hq, Hk, D, seq_len, device="cuda"):
    """Benchmark PyTorch SDPA as reference (FP16)."""
    kv_groups = Hq // Hk
    query = torch.randn(B, Hq, 1, D, device=device, dtype=torch.bfloat16)
    key = torch.randn(B, Hk, seq_len, D, device=device, dtype=torch.bfloat16)
    value = torch.randn(B, Hk, seq_len, D, device=device, dtype=torch.bfloat16)

    # Expand K/V for GQA
    key = key.unsqueeze(2).expand(-1, -1, kv_groups, -1, -1).reshape(B, Hq, seq_len, D)
    value = value.unsqueeze(2).expand(-1, -1, kv_groups, -1, -1).reshape(B, Hq, seq_len, D)

    # Warmup
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(D))
    torch.cuda.synchronize()

    ms = do_bench(
        lambda: torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=1.0 / math.sqrt(D)),
        warmup=25, rep=100,
    )

    return ms


def run_benchmarks():
    gpu = GPUSpec.detect()
    print(f"{'='*80}")
    print(f"FusenCache Kernel Benchmark (AutoKernel-style)")
    print(f"GPU: {gpu.name}")
    print(f"Peak: {gpu.peak_tflops_fp16} TFLOPS, {gpu.peak_bandwidth_gb_s} GB/s")
    print(f"{'='*80}")

    # Gemma 4 31B parameters
    # Sliding: Hq=32, Hk=16, D=256
    # Global:  Hq=32, Hk=4,  D=512

    configs = [
        # (label, B, Hq, Hk, D, seq_len)
        # Single decode, various seq lengths
        ("single B=1 S=128",    1,  32, 16, 256, 128),
        ("single B=1 S=256",    1,  32, 16, 256, 256),
        ("single B=1 S=512",    1,  32, 16, 256, 512),
        ("single B=1 S=1024",   1,  32, 16, 256, 1024),
        ("single B=1 S=2048",   1,  32, 16, 256, 2048),
        ("single B=1 S=4096",   1,  32, 16, 256, 4096),
        # Batch decode
        ("batch B=4 S=512",     4,  32, 16, 256, 512),
        ("batch B=8 S=512",     8,  32, 16, 256, 512),
        ("batch B=16 S=512",    16, 32, 16, 256, 512),
        ("batch B=4 S=2048",    4,  32, 16, 256, 2048),
        ("batch B=8 S=2048",    8,  32, 16, 256, 2048),
        # Global attention heads (D=512)
        ("global B=1 S=1024",   1,  32, 4,  512, 1024),
        ("global B=4 S=1024",   4,  32, 4,  512, 1024),
    ]

    print(f"\n{'Label':<25} {'FusenCache':>10} {'PyTorch':>10} {'Speedup':>8} "
          f"{'TFLOPS':>8} {'GB/s':>8} {'%peak':>7} {'Bound':>8}")
    print(f"{'─'*25} {'─'*10} {'─'*10} {'─'*8} "
          f"{'─'*8} {'─'*8} {'─'*7} {'─'*8}")

    results = []

    for label, B, Hq, Hk, D, seq_len in configs:
        try:
            fc_ms = bench_fusencache_kernel(B, Hq, Hk, D, seq_len)
            ref_ms = bench_pytorch_reference(B, Hq, Hk, D, seq_len)

            flops = decode_attention_flops(B, Hq, D, seq_len)
            fc_bytes = decode_attention_bytes_fusen(B, Hq, Hk, D, seq_len)

            tflops = flops / (fc_ms / 1000.0) / 1e12
            gb_s = fc_bytes / (fc_ms / 1000.0) / 1e9
            speedup = ref_ms / fc_ms

            # Roofline
            ai = flops / fc_bytes
            ridge = (gpu.peak_tflops_fp16 * 1e12) / (gpu.peak_bandwidth_gb_s * 1e9)
            if ai < ridge:
                bound = "mem"
                pct = gb_s / gpu.peak_bandwidth_gb_s * 100
            else:
                bound = "compute"
                pct = tflops / gpu.peak_tflops_fp16 * 100

            print(f"{label:<25} {fc_ms*1000:>8.1f}us {ref_ms*1000:>8.1f}us "
                  f"{speedup:>7.2f}x {tflops:>7.2f} {gb_s:>7.0f} "
                  f"{pct:>6.1f}% {bound:>8}")

            results.append({
                "label": label, "B": B, "Hq": Hq, "Hk": Hk, "D": D,
                "seq_len": seq_len, "fc_us": fc_ms * 1000, "ref_us": ref_ms * 1000,
                "speedup": speedup, "tflops": tflops, "gb_s": gb_s,
                "pct_peak": pct, "bound": bound,
            })

        except Exception as e:
            print(f"{label:<25} ERROR: {e}")

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if results:
        avg_tflops = sum(r["tflops"] for r in results) / len(results)
        avg_pct = sum(r["pct_peak"] for r in results) / len(results)
        mem_bound = sum(1 for r in results if r["bound"] == "mem")
        compute_bound = len(results) - mem_bound

        print(f"Avg TFLOPS: {avg_tflops:.2f}")
        print(f"Avg % peak: {avg_pct:.1f}%")
        print(f"Memory-bound: {mem_bound}/{len(results)}")
        print(f"Compute-bound: {compute_bound}/{len(results)}")

        # Best and worst
        best = max(results, key=lambda r: r["pct_peak"])
        worst = min(results, key=lambda r: r["pct_peak"])
        print(f"Best:  {best['label']} ({best['pct_peak']:.1f}% peak)")
        print(f"Worst: {worst['label']} ({worst['pct_peak']:.1f}% peak)")

    return results


if __name__ == "__main__":
    run_benchmarks()
