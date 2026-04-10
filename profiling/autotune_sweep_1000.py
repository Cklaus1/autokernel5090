#!/usr/bin/env python3
"""
Massive 1000+ config autotune sweep for fused RMSNorm + FP4 quantization kernel.

Targets Gemma4 26B MoE GEMM shapes on RTX 5090 (SM120).
Generates configs that standard @triton.autotune would never try,
including non-power-of-2 BLOCK_H, extreme num_warps, high num_stages.

Also benchmarks CUTLASS FP4 GEMM wrapper at different input shapes/alignments.
"""

import os
import sys
import time
import csv
import json
import itertools
from pathlib import Path
from collections import defaultdict

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# FP4 quantization helper (inlined from fused_norm_fp4.py)
# ---------------------------------------------------------------------------

@triton.jit
def _quantize_to_fp4_code(val_scaled):
    code = (val_scaled * 0).to(tl.int32)
    code = tl.where(val_scaled > 0.25, 1, code)
    code = tl.where(val_scaled > 0.75, 2, code)
    code = tl.where(val_scaled > 1.25, 3, code)
    code = tl.where(val_scaled > 1.75, 4, code)
    code = tl.where(val_scaled > 2.5, 5, code)
    code = tl.where(val_scaled > 3.5, 6, code)
    code = tl.where(val_scaled > 5.0, 7, code)
    return code


# ---------------------------------------------------------------------------
# Kernel: manually launched (no @autotune) so we control config externally
# ---------------------------------------------------------------------------

@triton.jit
def fused_rms_norm_fp4_quant_kernel(
    X_ptr, W_ptr, OUT_fp4_ptr, OUT_scale_ptr, GS_ptr,
    B, H: tl.constexpr,
    stride_x_b, stride_out_b, stride_scale_b,
    eps,
    BLOCK_H: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    HAVE_WEIGHT: tl.constexpr,
    VARIANCE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    global_scale = tl.load(GS_ptr).to(tl.float32)

    VAR_DIM: tl.constexpr = H if VARIANCE_SIZE == 0 else VARIANCE_SIZE
    NUM_ITERS_VAR: tl.constexpr = (VAR_DIM + BLOCK_H - 1) // BLOCK_H
    NUM_ITERS: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    QBLOCKS_PER_ITER: tl.constexpr = BLOCK_H // QUANT_BLOCK_SIZE
    HALF_BLOCK: tl.constexpr = BLOCK_H // 2
    HALF_QBS: tl.constexpr = QUANT_BLOCK_SIZE // 2

    # Pass 1: sum of squares
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for _i in range(NUM_ITERS_VAR):
        offs = _i * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < VAR_DIM
        x = tl.load(X_ptr + row * stride_x_b + offs, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)
    rrms = 1.0 / tl.sqrt(sum_sq / VAR_DIM + eps)

    # Pass 2: normalize + quantize + pack
    for _i in range(NUM_ITERS):
        base = _i * BLOCK_H
        even_offs = base + tl.arange(0, HALF_BLOCK) * 2
        odd_offs = even_offs + 1
        even_mask = even_offs < H
        odd_mask = odd_offs < H

        x_even = tl.load(X_ptr + row * stride_x_b + even_offs, mask=even_mask, other=0.0).to(tl.float32)
        x_odd = tl.load(X_ptr + row * stride_x_b + odd_offs, mask=odd_mask, other=0.0).to(tl.float32)

        xn_even = x_even * rrms
        xn_odd = x_odd * rrms

        if HAVE_WEIGHT:
            w_even = tl.load(W_ptr + even_offs, mask=even_mask, other=1.0).to(tl.float32)
            w_odd = tl.load(W_ptr + odd_offs, mask=odd_mask, other=1.0).to(tl.float32)
            xn_even = xn_even * w_even
            xn_odd = xn_odd * w_odd

        abs_even_2d = tl.abs(tl.reshape(xn_even, [QBLOCKS_PER_ITER, HALF_QBS]))
        abs_odd_2d = tl.abs(tl.reshape(xn_odd, [QBLOCKS_PER_ITER, HALF_QBS]))
        max_even = tl.max(abs_even_2d, axis=1)
        max_odd = tl.max(abs_odd_2d, axis=1)
        block_max = tl.maximum(max_even, max_odd)

        block_scale = block_max / (6.0 * global_scale)
        block_scale = tl.minimum(block_scale, 448.0)

        s_offs = _i * QBLOCKS_PER_ITER + tl.arange(0, QBLOCKS_PER_ITER)
        s_mask = s_offs < (H // QUANT_BLOCK_SIZE)
        tl.store(OUT_scale_ptr + row * stride_scale_b + s_offs, block_scale, mask=s_mask)

        block_scale_fp8 = tl.load(
            OUT_scale_ptr + row * stride_scale_b + s_offs, mask=s_mask, other=0.0
        ).to(tl.float32)

        bs_2d = tl.reshape(block_scale_fp8, [QBLOCKS_PER_ITER, 1])
        bs_2d = tl.broadcast_to(bs_2d, [QBLOCKS_PER_ITER, HALF_QBS])
        denom = tl.reshape(bs_2d, [HALF_BLOCK]) * global_scale
        denom = tl.where(denom > 0.0, denom, 1.0)

        vs_even = tl.abs(xn_even) / denom
        vs_odd = tl.abs(xn_odd) / denom
        code_even = _quantize_to_fp4_code(vs_even)
        code_odd = _quantize_to_fp4_code(vs_odd)

        sign_even = tl.where(xn_even < 0.0, 8, 0).to(tl.int32)
        sign_odd = tl.where(xn_odd < 0.0, 8, 0).to(tl.int32)
        fp4_even = code_even | sign_even
        fp4_odd = code_odd | sign_odd

        packed = (fp4_even & 0xF) | ((fp4_odd & 0xF) << 4)

        byte_offs = _i * HALF_BLOCK + tl.arange(0, HALF_BLOCK)
        byte_mask = byte_offs < (H // 2)
        tl.store(
            OUT_fp4_ptr + row * stride_out_b + byte_offs,
            packed.to(tl.uint8),
            mask=byte_mask,
        )


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_configs():
    """Generate 1000+ unique configurations."""
    configs = []
    seen = set()

    def add(block_h, nw, ns):
        key = (block_h, nw, ns)
        if key not in seen:
            seen.add(key)
            configs.append({"BLOCK_H": block_h, "num_warps": nw, "num_stages": ns})

    # --- Tier 1: Standard power-of-2 sweep (like normal autotune) ---
    for bh in [64, 128, 256, 512, 1024, 2048, 4096]:
        for nw in [1, 2, 4, 8, 16, 32]:
            for ns in [1, 2, 3, 4, 5]:
                add(bh, nw, ns)

    # --- Tier 2: Non-power-of-2 BLOCK_H -- SKIP, Triton requires power-of-2 for tl.arange ---
    # These all fail on Triton 3.6.0. Keep only 8192 which is power-of-2.

    # --- Tier 3: Extreme configs ---
    # Skip very small BLOCK_H (16, 32) -- too many iterations for H=2816

    for bh in [4096, 2816]:
        for nw in [1, 2, 4, 8, 16, 32]:
            for ns in [1, 2, 3, 4, 5, 6, 7]:
                add(bh, nw, ns)

    # --- Tier 4: H-matching configs (BLOCK_H = H exactly) ---
    for bh in [2816, 1408, 704, 4096, 8192]:
        for nw in [1, 2, 4, 8, 16, 32]:
            for ns in [1, 2, 3, 4, 5]:
                add(bh, nw, ns)

    # --- Tier 5: High num_stages with moderate BLOCK_H ---
    for bh in [128, 256, 512, 1024]:
        for nw in [4, 8, 16]:
            for ns in [5, 6, 7, 8]:
                add(bh, nw, ns)

    # --- Tier 6: Large BLOCK_H with varied warps/stages ---
    for bh in [8192, 16384]:
        for nw in [1, 2, 4, 8, 16, 32]:
            for ns in [1, 2, 3, 4, 5, 6, 7, 8]:
                add(bh, nw, ns)

    # Moderate BLOCK_H with extreme stages
    for bh in [64, 128, 256, 512, 1024, 2048, 4096]:
        for nw in [1, 2, 32]:
            for ns in [6, 7, 8, 9, 10]:
                add(bh, nw, ns)

    return configs


def filter_configs_for_h(configs, H, quant_block_size=16):
    """Filter configs that are valid for a given H."""
    valid = []
    for c in configs:
        bh = c["BLOCK_H"]
        # BLOCK_H must be divisible by quant_block_size (16)
        if bh % quant_block_size != 0:
            continue
        # BLOCK_H must be even (for HALF_BLOCK)
        if bh % 2 != 0:
            continue
        # BLOCK_H // 2 must be > 0
        if bh < 2:
            continue
        # For Triton tl.arange, BLOCK_H should ideally be power-of-2
        # but non-pow2 may work on newer Triton. We try and catch errors.
        # BLOCK_H must be >= quant_block_size
        if bh < quant_block_size:
            continue
        valid.append(c)
    return valid


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def bench_config(B, H, config, x, weight, gs, fp4_out, scale_out, warmup=25, rep=100):
    """Benchmark a single config. Returns latency in microseconds or None on error."""
    block_h = config["BLOCK_H"]
    nw = config["num_warps"]
    ns = config["num_stages"]

    grid = (B,)

    def run():
        fused_rms_norm_fp4_quant_kernel[grid](
            x, weight, fp4_out, scale_out, gs,
            B, H,
            x.stride(0), fp4_out.stride(0), scale_out.stride(0),
            1e-6,
            BLOCK_H=block_h,
            QUANT_BLOCK_SIZE=16,
            HAVE_WEIGHT=True,
            VARIANCE_SIZE=0,
            num_warps=nw,
            num_stages=ns,
        )

    try:
        # Warmup / compile
        run()
        torch.cuda.synchronize()
        # Bench
        lat_ms = triton.testing.do_bench(run, warmup=warmup, rep=rep)
        return lat_ms * 1000  # microseconds
    except Exception as e:
        return None


def bench_cutlass_fp4(B, H_in, H_out, warmup=25, rep=200):
    """Benchmark CUTLASS FP4 GEMM (torch._scaled_mm) at given shapes."""
    try:
        from vllm._custom_ops import scaled_fp4_quant, cutlass_scaled_fp4_mm
    except ImportError:
        try:
            # Try direct torch path
            pass
        except:
            return None

    results = {}

    # Create FP4 input (simulating post-quantization)
    x_bf16 = torch.randn(B, H_in, device="cuda", dtype=torch.bfloat16)
    gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    try:
        x_fp4, x_scale = scaled_fp4_quant(x_bf16, gs, is_sf_swizzled_layout=True)
    except Exception:
        return None

    # Create weight in FP4 format
    w_bf16 = torch.randn(H_out, H_in, device="cuda", dtype=torch.bfloat16)
    try:
        w_fp4, w_scale = scaled_fp4_quant(w_bf16, gs, is_sf_swizzled_layout=True)
    except Exception:
        return None

    # Benchmark CUTLASS FP4 GEMM
    try:
        def run_cutlass():
            return cutlass_scaled_fp4_mm(x_fp4, w_fp4, x_scale, w_scale, gs, gs, torch.bfloat16)

        run_cutlass()
        torch.cuda.synchronize()
        lat_ms = triton.testing.do_bench(run_cutlass, warmup=warmup, rep=rep)
        results["cutlass_fp4_us"] = lat_ms * 1000
    except Exception as e:
        results["cutlass_fp4_us"] = None
        results["error"] = str(e)

    # Also benchmark cuBLAS bf16 GEMM for reference
    w_t = w_bf16.t().contiguous()
    def run_cublas():
        return torch.mm(x_bf16, w_t)
    try:
        run_cublas()
        torch.cuda.synchronize()
        lat_ms = triton.testing.do_bench(run_cublas, warmup=warmup, rep=rep)
        results["cublas_bf16_us"] = lat_ms * 1000
    except:
        results["cublas_bf16_us"] = None

    return results


# ---------------------------------------------------------------------------
# Baseline: standard 45-config autotune
# ---------------------------------------------------------------------------

def generate_standard_45_configs():
    """The standard autotune configs from fused_norm_fp4.py."""
    configs = []
    for bh in [256, 512, 1024, 2048, 4096]:
        for nw in [4, 8, 16]:
            for ns in [1, 2, 3]:
                configs.append({"BLOCK_H": bh, "num_warps": nw, "num_stages": ns})
    return configs


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda")

    print("=" * 100)
    print("MASSIVE 1000-CONFIG AUTOTUNE SWEEP")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}")
    print("=" * 100)

    # Generate all configs
    all_configs = generate_configs()
    std_configs = generate_standard_45_configs()
    std_keys = {(c["BLOCK_H"], c["num_warps"], c["num_stages"]) for c in std_configs}

    print(f"\nTotal configs generated: {len(all_configs)}")
    print(f"Standard autotune configs: {len(std_configs)}")
    print(f"Novel configs (not in standard): {sum(1 for c in all_configs if (c['BLOCK_H'], c['num_warps'], c['num_stages']) not in std_keys)}")

    # Target shapes
    H_values = [2816, 1408, 704]  # Main Gemma4 hidden dims
    B_values = [1, 32, 128]       # Decode, small batch, large batch

    # Results storage
    all_results = []
    best_per_shape = {}  # (B, H) -> (latency, config)
    best_std_per_shape = {}

    tsv_path = Path("/tmp/autotune_results.tsv")

    for H in H_values:
        valid_configs = filter_configs_for_h(all_configs, H)
        valid_std = filter_configs_for_h(std_configs, H)
        print(f"\n{'='*80}")
        print(f"H = {H}: {len(valid_configs)} valid configs ({len(valid_std)} standard)")
        print(f"{'='*80}")

        for B in B_values:
            print(f"\n  B={B}, H={H}: sweeping {len(valid_configs)} configs...")
            x = torch.randn(B, H, device=device, dtype=torch.bfloat16)
            weight = torch.randn(H, device=device, dtype=torch.bfloat16).abs() + 0.1
            gs = torch.tensor([1.0], device=device, dtype=torch.float32)
            fp4_out = torch.empty((B, H // 2), device=device, dtype=torch.uint8)
            scale_out = torch.empty((B, H // 16), device=device, dtype=torch.float8_e4m3fn)

            shape_results = []
            n_ok = 0
            n_fail = 0
            t0 = time.time()

            for i, config in enumerate(valid_configs):
                lat = bench_config(B, H, config, x, weight, gs, fp4_out, scale_out,
                                   warmup=10, rep=50)

                is_std = (config["BLOCK_H"], config["num_warps"], config["num_stages"]) in std_keys

                if lat is not None:
                    n_ok += 1
                    result = {
                        "B": B, "H": H,
                        "BLOCK_H": config["BLOCK_H"],
                        "num_warps": config["num_warps"],
                        "num_stages": config["num_stages"],
                        "latency_us": lat,
                        "is_standard": is_std,
                    }
                    shape_results.append(result)
                    all_results.append(result)
                else:
                    n_fail += 1

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"    [{i+1}/{len(valid_configs)}] {n_ok} ok, {n_fail} fail, "
                          f"{elapsed:.1f}s elapsed", flush=True)

            elapsed = time.time() - t0
            print(f"    Done: {n_ok} ok, {n_fail} fail in {elapsed:.1f}s")

            if not shape_results:
                continue

            # Find best overall and best standard
            shape_results.sort(key=lambda r: r["latency_us"])
            best = shape_results[0]
            best_per_shape[(B, H)] = best

            std_results = [r for r in shape_results if r["is_standard"]]
            if std_results:
                best_std = min(std_results, key=lambda r: r["latency_us"])
                best_std_per_shape[(B, H)] = best_std
            else:
                best_std = None
                best_std_per_shape[(B, H)] = None

            # Report
            print(f"\n    BEST OVERALL: {best['latency_us']:.2f} us "
                  f"(BLOCK_H={best['BLOCK_H']}, warps={best['num_warps']}, stages={best['num_stages']})")
            if best_std:
                improvement = (best_std["latency_us"] - best["latency_us"]) / best_std["latency_us"] * 100
                print(f"    BEST STD:     {best_std['latency_us']:.2f} us "
                      f"(BLOCK_H={best_std['BLOCK_H']}, warps={best_std['num_warps']}, stages={best_std['num_stages']})")
                print(f"    IMPROVEMENT:  {improvement:+.1f}% ({best_std['latency_us']:.2f} -> {best['latency_us']:.2f} us)")
            else:
                print(f"    No standard configs succeeded for this shape.")

            # Top 10
            print(f"\n    Top 10 configs:")
            print(f"    {'Rank':>4} {'Lat(us)':>8} {'BLOCK_H':>8} {'warps':>6} {'stg':>4} {'std?':>5}")
            for rank, r in enumerate(shape_results[:10], 1):
                marker = " *" if r["is_standard"] else ""
                print(f"    {rank:>4} {r['latency_us']:>8.2f} {r['BLOCK_H']:>8} "
                      f"{r['num_warps']:>6} {r['num_stages']:>4} {marker}")

            # Surprising configs
            surprising = [r for r in shape_results[:20] if not r["is_standard"]]
            if surprising:
                print(f"\n    Surprising non-standard configs in top 20:")
                for r in surprising[:5]:
                    print(f"      {r['latency_us']:.2f} us: BLOCK_H={r['BLOCK_H']}, "
                          f"warps={r['num_warps']}, stages={r['num_stages']}")

    # ---------------------------------------------------------------------------
    # CUTLASS FP4 GEMM benchmark
    # ---------------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("CUTLASS FP4 GEMM BENCHMARK")
    print(f"{'='*100}")

    gemm_shapes = [
        # (B, H_in, H_out, description)
        (1, 2816, 1408, "Gate/Up B=1"),
        (1, 704, 2816, "Down B=1"),
        (1, 2816, 8192, "QKV sliding B=1"),
        (1, 2816, 4096, "QKV global B=1"),
        (1, 4096, 2816, "O global B=1"),
        (1, 8192, 2816, "O sliding B=1"),
        (32, 2816, 1408, "Gate/Up B=32"),
        (32, 704, 2816, "Down B=32"),
        (32, 2816, 8192, "QKV sliding B=32"),
        (128, 2816, 1408, "Gate/Up B=128"),
        (128, 704, 2816, "Down B=128"),
        (128, 2816, 8192, "QKV sliding B=128"),
        (256, 2816, 1408, "Gate/Up B=256"),
        (512, 2816, 1408, "Gate/Up B=512"),
    ]

    cutlass_results = []
    print(f"\n{'Desc':<22} {'B':>4} {'H_in':>6} {'H_out':>6} {'CUTLASS_us':>11} {'cuBLAS_us':>10} {'Ratio':>7}")
    print("-" * 75)

    for B, H_in, H_out, desc in gemm_shapes:
        r = bench_cutlass_fp4(B, H_in, H_out)
        if r and r.get("cutlass_fp4_us"):
            cut = r["cutlass_fp4_us"]
            cub = r.get("cublas_bf16_us")
            ratio = cut / cub if cub and cub > 0 else float('nan')
            print(f"{desc:<22} {B:>4} {H_in:>6} {H_out:>6} {cut:>10.2f} "
                  f"{cub:>10.2f} {ratio:>6.2f}x")
            cutlass_results.append({
                "desc": desc, "B": B, "H_in": H_in, "H_out": H_out,
                "cutlass_fp4_us": cut, "cublas_bf16_us": cub,
            })
        elif r and r.get("error"):
            print(f"{desc:<22} {B:>4} {H_in:>6} {H_out:>6}  ERROR: {r['error'][:60]}")
        else:
            print(f"{desc:<22} {B:>4} {H_in:>6} {H_out:>6}  SKIPPED (no CUTLASS)")

    # ---------------------------------------------------------------------------
    # Pareto-optimal analysis
    # ---------------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("PARETO-OPTIMAL CONFIGS (across batch sizes)")
    print(f"{'='*100}")

    for H in H_values:
        print(f"\n  H = {H}:")
        # Collect per-config: dict of config_key -> {B: latency}
        config_lats = defaultdict(dict)
        for r in all_results:
            if r["H"] != H:
                continue
            key = (r["BLOCK_H"], r["num_warps"], r["num_stages"])
            config_lats[key][r["B"]] = r["latency_us"]

        # Filter to configs that have all 3 batch sizes
        complete = {k: v for k, v in config_lats.items() if len(v) == len(B_values)}
        if not complete:
            print("    No configs with all batch sizes.")
            continue

        # Pareto front: a config is Pareto-optimal if no other config is <= on all Bs and < on at least one
        pareto = []
        configs_list = list(complete.items())
        for i, (key_i, lats_i) in enumerate(configs_list):
            dominated = False
            for j, (key_j, lats_j) in enumerate(configs_list):
                if i == j:
                    continue
                # Check if j dominates i
                all_leq = all(lats_j[b] <= lats_i[b] for b in B_values)
                any_lt = any(lats_j[b] < lats_i[b] for b in B_values)
                if all_leq and any_lt:
                    dominated = True
                    break
            if not dominated:
                pareto.append((key_i, lats_i))

        pareto.sort(key=lambda x: sum(x[1].values()))
        is_std_set = std_keys

        print(f"    Pareto-optimal configs: {len(pareto)} (out of {len(complete)})")
        print(f"    {'BLOCK_H':>8} {'warps':>6} {'stg':>4} " +
              " ".join(f"{'B='+str(b):>8}" for b in B_values) + "  std?")
        for key, lats in pareto[:15]:
            marker = " *" if key in is_std_set else ""
            lat_str = " ".join(f"{lats[b]:>7.2f}" for b in B_values)
            print(f"    {key[0]:>8} {key[1]:>6} {key[2]:>4} {lat_str} {marker}")

    # ---------------------------------------------------------------------------
    # Summary comparison
    # ---------------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("SUMMARY: 1000-SWEEP vs STANDARD 45-CONFIG AUTOTUNE")
    print(f"{'='*100}")
    print(f"\n{'Shape':>12} {'Best1000(us)':>13} {'BestStd(us)':>13} {'Improvement':>12} {'Best Config':>40}")
    print("-" * 95)

    for H in H_values:
        for B in B_values:
            key = (B, H)
            best = best_per_shape.get(key)
            best_std = best_std_per_shape.get(key)
            if best and best_std:
                imp = (best_std["latency_us"] - best["latency_us"]) / best_std["latency_us"] * 100
                cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
                std_marker = " (std)" if best["is_standard"] else " (NEW)"
                print(f"  B={B:>3},H={H:>4} {best['latency_us']:>12.2f} {best_std['latency_us']:>12.2f} "
                      f"{imp:>+10.1f}% {cfg}{std_marker}")
            elif best:
                cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
                print(f"  B={B:>3},H={H:>4} {best['latency_us']:>12.2f} {'N/A':>13} {'N/A':>12} {cfg}")

    # ---------------------------------------------------------------------------
    # Save TSV
    # ---------------------------------------------------------------------------
    print(f"\nSaving results to {tsv_path}...")
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t",
                                fieldnames=["B", "H", "BLOCK_H", "num_warps", "num_stages",
                                             "latency_us", "is_standard"])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    # Also save CUTLASS results
    cutlass_path = Path("/tmp/cutlass_fp4_results.tsv")
    if cutlass_results:
        with open(cutlass_path, "w", newline="") as f:
            writer = csv.DictWriter(f, delimiter="\t",
                                    fieldnames=["desc", "B", "H_in", "H_out",
                                                 "cutlass_fp4_us", "cublas_bf16_us"])
            writer.writeheader()
            for r in cutlass_results:
                writer.writerow(r)

    # Save JSON summary
    summary_path = Path("/tmp/autotune_summary.json")
    summary = {
        "total_configs": len(all_configs),
        "standard_configs": len(std_configs),
        "shapes_tested": [(B, H) for H in H_values for B in B_values],
        "best_per_shape": {},
        "best_std_per_shape": {},
        "cutlass_results": cutlass_results,
    }
    for key, val in best_per_shape.items():
        summary["best_per_shape"][f"B={key[0]},H={key[1]}"] = val
    for key, val in best_std_per_shape.items():
        if val:
            summary["best_std_per_shape"][f"B={key[0]},H={key[1]}"] = val

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {tsv_path}, {cutlass_path}, {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
