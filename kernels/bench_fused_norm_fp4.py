"""
Benchmark: Fused RMSNorm + FP4-E2M1 Quantization

Compares fused Triton kernel vs separate (rms_norm + scaled_fp4_quant) at Gemma4 shapes.
Sweeps batch sizes and hidden dimensions.
Reports: latency (us), bandwidth (GB/s), speedup vs separate ops.
Uses triton.testing.do_bench for accurate GPU kernel timing.
"""

import os
import sys
import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
HIDDEN_SIZES = [256, 512, 2816]  # Gemma4: head norms (256, 512), main dim (2816)
QUANT_BLOCK_SIZE = 16
EPSILON = 1e-6
DO_BENCH_WARMUP = 100
DO_BENCH_REP = 500


def compute_bandwidth(B, H, latency_us, has_residual=False):
    """Compute effective bandwidth in GB/s."""
    bytes_read = B * H * 2 + H * 2 + 4  # x (bf16) + weight (bf16) + global_scale (f32)
    bytes_written = B * (H // 2) + B * (H // 16)  # fp4 (u8) + scales (fp8)
    if has_residual:
        bytes_read += B * H * 2
        bytes_written += B * H * 2
    total = bytes_read + bytes_written
    return total / (latency_us * 1e-6) / 1e9 if latency_us > 0 else 0


def validate_correctness(B, H, dtype=torch.bfloat16):
    """Validate fused kernel output matches vLLM reference."""
    from vllm._custom_ops import rms_norm as vllm_rms_norm
    from vllm._custom_ops import scaled_fp4_quant as vllm_fp4_quant
    from fused_norm_fp4 import fused_rms_norm_fp4_quant

    torch.manual_seed(42)
    x = torch.randn(B, H, device="cuda", dtype=dtype)
    weight = torch.randn(H, device="cuda", dtype=dtype).abs() + 0.1
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    normed = torch.empty_like(x)
    vllm_rms_norm(normed, x, weight, EPSILON)
    ref_fp4, ref_scale = vllm_fp4_quant(normed, gs, is_sf_swizzled_layout=False)

    fused_fp4, fused_scale = fused_rms_norm_fp4_quant(x.clone(), weight, gs, EPSILON)

    fp4_match = (fused_fp4 == ref_fp4).float().mean().item()
    scale_match = (fused_scale.view(torch.uint8) == ref_scale.view(torch.uint8)).float().mean().item()
    return fp4_match, scale_match


def main():
    from vllm._custom_ops import rms_norm as vllm_rms_norm
    from vllm._custom_ops import scaled_fp4_quant as vllm_fp4_quant
    from fused_norm_fp4 import fused_rms_norm_fp4_quant

    print("=" * 90)
    print("Fused RMSNorm + FP4-E2M1 Quantization Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Timing: triton.testing.do_bench (warmup={DO_BENCH_WARMUP}, rep={DO_BENCH_REP})")
    print("=" * 90)

    # Trigger autotune for all H values upfront
    print("\nAutotuning fused kernel...", flush=True)
    for H in HIDDEN_SIZES:
        x = torch.randn(4, H, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(H, device="cuda", dtype=torch.bfloat16)
        gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        for _ in range(3):
            fused_rms_norm_fp4_quant(x.clone(), w, gs, EPSILON)
    print("Done.\n")

    # TSV header
    header = "batch\thidden\tnorm_us\tquant_us\tsep_total_us\tfused_us\tspeedup\tbw_sep\tbw_fused\tfp4_match\tscale_match"
    print(header)

    results = []

    for H in HIDDEN_SIZES:
        for B in BATCH_SIZES:
            torch.manual_seed(42)
            x = torch.randn(B, H, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(H, device="cuda", dtype=torch.bfloat16).abs() + 0.1
            gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
            normed = torch.empty_like(x)

            # Benchmark separate: RMSNorm
            lat_norm_ms = triton.testing.do_bench(
                lambda: vllm_rms_norm(normed, x, weight, EPSILON),
                warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP,
            )
            lat_norm_us = lat_norm_ms * 1000

            # Benchmark separate: FP4 quant
            vllm_rms_norm(normed, x, weight, EPSILON)  # ensure normed is valid
            lat_quant_ms = triton.testing.do_bench(
                lambda: vllm_fp4_quant(normed, gs, is_sf_swizzled_layout=False),
                warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP,
            )
            lat_quant_us = lat_quant_ms * 1000

            lat_sep_us = lat_norm_us + lat_quant_us

            # Benchmark fused kernel
            lat_fused_ms = triton.testing.do_bench(
                lambda: fused_rms_norm_fp4_quant(x.clone(), weight, gs, EPSILON),
                warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP,
            )
            lat_fused_us = lat_fused_ms * 1000

            speedup = lat_sep_us / lat_fused_us if lat_fused_us > 0 else 0
            bw_sep = compute_bandwidth(B, H, lat_sep_us)
            bw_fused = compute_bandwidth(B, H, lat_fused_us)

            # Validate
            try:
                fp4_match, scale_match = validate_correctness(B, H)
            except Exception:
                fp4_match, scale_match = 0, 0

            row = (
                f"{B}\t{H}\t{lat_norm_us:.1f}\t{lat_quant_us:.1f}\t{lat_sep_us:.1f}\t"
                f"{lat_fused_us:.1f}\t{speedup:.2f}x\t{bw_sep:.1f}\t{bw_fused:.1f}\t"
                f"{fp4_match*100:.1f}%\t{scale_match*100:.1f}%"
            )
            print(row, flush=True)
            results.append({
                "batch": B, "hidden": H,
                "norm_us": lat_norm_us, "quant_us": lat_quant_us,
                "sep_us": lat_sep_us, "fused_us": lat_fused_us,
                "speedup": speedup,
                "bw_sep": bw_sep, "bw_fused": bw_fused,
                "fp4_match": fp4_match, "scale_match": scale_match,
            })

    # ---- Summary ----
    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    for H in HIDDEN_SIZES:
        h_results = [r for r in results if r["hidden"] == H]
        if not h_results:
            continue
        avg_speedup = sum(r["speedup"] for r in h_results) / len(h_results)
        best = max(h_results, key=lambda r: r["speedup"])
        worst = min(h_results, key=lambda r: r["speedup"])
        print(f"\nH={H}:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Best:  B={best['batch']:3d} -> {best['speedup']:.2f}x ({best['fused_us']:.1f} us vs {best['sep_us']:.1f} us)")
        print(f"  Worst: B={worst['batch']:3d} -> {worst['speedup']:.2f}x ({worst['fused_us']:.1f} us vs {worst['sep_us']:.1f} us)")

    # Gemma4 decode estimate
    gemma_b1 = [r for r in results if r["hidden"] == 2816 and r["batch"] == 1]
    if gemma_b1:
        r = gemma_b1[0]
        savings_per_call_us = r["sep_us"] - r["fused_us"]
        calls_per_step = 60
        savings_ms = savings_per_call_us * calls_per_step / 1000
        print(f"\n---- Gemma4 26B NVFP4 Decode Projection ----")
        print(f"Per-call savings: {savings_per_call_us:.1f} us")
        print(f"Per-step savings ({calls_per_step} calls): {savings_ms:.2f} ms")
        # At 6615 tok/s, step time = 1000/6615 = 0.151 ms
        step_time_ms = 1000 / 6615
        pct_gain = (savings_ms / step_time_ms) * 100
        print(f"Step time at 6615 tok/s: {step_time_ms:.3f} ms")
        print(f"Projected throughput gain: {pct_gain:.1f}%")

    # ---- Residual variant ----
    print()
    print("=" * 90)
    print("Fused Residual + RMSNorm + FP4 Quantization (H=2816)")
    print("=" * 90)
    print("batch\tfused_us\tfused_res_us\toverhead_us")

    for B in [1, 4, 16, 64, 256]:
        x = torch.randn(B, 2816, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(B, 2816, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2816, device="cuda", dtype=torch.bfloat16).abs() + 0.1
        gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)

        lat_no_res_ms = triton.testing.do_bench(
            lambda: fused_rms_norm_fp4_quant(x.clone(), weight, gs, EPSILON),
            warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP,
        )
        lat_res_ms = triton.testing.do_bench(
            lambda: fused_rms_norm_fp4_quant(x.clone(), weight, gs, EPSILON, residual=residual),
            warmup=DO_BENCH_WARMUP, rep=DO_BENCH_REP,
        )
        overhead = (lat_res_ms - lat_no_res_ms) * 1000
        print(f"{B}\t{lat_no_res_ms*1000:.1f}\t{lat_res_ms*1000:.1f}\t{overhead:+.1f}")


if __name__ == "__main__":
    main()
