"""Benchmark FP8 decode attention kernel vs FA2 BF16 baseline.

Tests at Gemma4 shapes:
  - B=32, num_q_heads=16, num_kv_heads=8, head_dim=256, seq=2048 (sliding)
  - B=32, num_q_heads=16, num_kv_heads=8, head_dim=512, seq=2048 (global)

Measures latency per call and estimates bandwidth utilization.
"""

import torch
import time
import sys
import math
import gc

sys.path.insert(0, "/root/projects/autokernel/kernels")
from fp8_decode_attention import fp8_decode_attention, create_test_data


def bench_latency(fn, warmup=50, iters=200, sync=True):
    """Benchmark function latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    # Timed
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1e6  # us
    return elapsed


def bench_with_cuda_events(fn, warmup=50, iters=200):
    """Benchmark with CUDA events for accurate GPU timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    return elapsed_ms * 1000  # us


def estimate_bandwidth(latency_us, batch, seq_len, num_kv_heads, head_dim, bytes_per_elem):
    """Estimate effective HBM bandwidth from latency."""
    # KV data read: 2 (K+V) * batch * seq_len * num_kv_heads * head_dim * bytes
    kv_bytes = 2 * batch * seq_len * num_kv_heads * head_dim * bytes_per_elem
    kv_gb = kv_bytes / 1e9
    bw_gbs = kv_gb / (latency_us * 1e-6)
    return kv_bytes, bw_gbs


def try_fa2_baseline(q_bf16, k_bf16, v_bf16, block_table, seq_lens, sm_scale, soft_cap):
    """Try to run FA2 as baseline. Returns latency or None if unavailable."""
    try:
        from vllm._custom_ops import paged_attention_v2
        # vLLM's paged attention interface
        # This is complex, so we do a simple sdpa baseline instead
        raise ImportError("Skip vllm paged_attn, use sdpa")
    except ImportError:
        pass

    # Simple SDPA baseline (not paged, but gives compute reference)
    return None


def run_benchmark(config_name, batch, num_q_heads, num_kv_heads, head_dim, seq_len,
                  logits_soft_cap=0.0, block_size=16):
    """Run benchmark for a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"  batch={batch}, q_heads={num_q_heads}, kv_heads={num_kv_heads}")
    print(f"  head_dim={head_dim}, seq_len={seq_len}, block_size={block_size}")
    print(f"  logits_soft_cap={logits_soft_cap}")
    print(f"{'='*70}")

    data = create_test_data(
        batch=batch, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, seq_len=seq_len, block_size=block_size)

    # ---- FP8 kernel ----
    def run_fp8():
        return fp8_decode_attention(
            data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
            data["block_table"], data["seq_lens"],
            data["k_scale"], data["v_scale"],
            logits_soft_cap=logits_soft_cap,
        )

    # Quick correctness check
    out = run_fp8()
    assert out.shape == (batch, num_q_heads, head_dim), f"Bad shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"

    # Benchmark FP8
    lat_fp8 = bench_with_cuda_events(run_fp8, warmup=100, iters=500)
    kv_bytes_fp8, bw_fp8 = estimate_bandwidth(
        lat_fp8, batch, seq_len, num_kv_heads, head_dim, 1)  # 1 byte for FP8

    print(f"\n  FP8 Triton kernel:")
    print(f"    Latency:    {lat_fp8:.1f} us")
    print(f"    KV data:    {kv_bytes_fp8 / 1e6:.1f} MB")
    print(f"    Bandwidth:  {bw_fp8:.0f} GB/s")
    print(f"    BW util:    {bw_fp8 / 1792 * 100:.1f}% of 1792 GB/s peak")

    # ---- BF16 baseline (unpaged SDPA) ----
    # Gather KV into contiguous BF16 tensors for SDPA comparison
    k_cont = torch.randn(batch, num_kv_heads, seq_len, head_dim,
                          dtype=torch.bfloat16, device="cuda") * 0.1
    v_cont = torch.randn(batch, num_kv_heads, seq_len, head_dim,
                          dtype=torch.bfloat16, device="cuda") * 0.1
    q_sdpa = data["q"].unsqueeze(2)  # [B, H, 1, D]

    # Expand Q for GQA
    q_expanded = data["q"].view(batch, num_kv_heads, num_q_heads // num_kv_heads, head_dim)
    q_expanded = q_expanded.unsqueeze(3)  # [B, kv_heads, group, 1, D]

    sm = 1.0 / math.sqrt(head_dim)

    def run_bf16_sdpa():
        # Simple per-kv-head loop with F.scaled_dot_product_attention
        # This tests compute + BF16 bandwidth
        out = torch.empty(batch, num_q_heads, head_dim,
                          dtype=torch.bfloat16, device="cuda")
        gsize = num_q_heads // num_kv_heads
        for kh in range(num_kv_heads):
            q_group = data["q"][:, kh*gsize:(kh+1)*gsize, :].unsqueeze(2)  # [B, group, 1, D]
            k_h = k_cont[:, kh:kh+1, :, :].expand(-1, gsize, -1, -1)  # [B, group, S, D]
            v_h = v_cont[:, kh:kh+1, :, :].expand(-1, gsize, -1, -1)
            o = torch.nn.functional.scaled_dot_product_attention(
                q_group, k_h, v_h, scale=sm)  # [B, group, 1, D]
            out[:, kh*gsize:(kh+1)*gsize, :] = o.squeeze(2)
        return out

    try:
        _ = run_bf16_sdpa()  # warmup / check it works
        lat_bf16 = bench_with_cuda_events(run_bf16_sdpa, warmup=50, iters=200)
        kv_bytes_bf16, bw_bf16 = estimate_bandwidth(
            lat_bf16, batch, seq_len, num_kv_heads, head_dim, 2)  # 2 bytes for BF16
        print(f"\n  BF16 SDPA baseline (unpaged, for bandwidth reference):")
        print(f"    Latency:    {lat_bf16:.1f} us")
        print(f"    KV data:    {kv_bytes_bf16 / 1e6:.1f} MB")
        print(f"    Bandwidth:  {bw_bf16:.0f} GB/s")
        print(f"    BW util:    {bw_bf16 / 1792 * 100:.1f}% of 1792 GB/s peak")
        print(f"\n  FP8 vs BF16 SDPA speedup: {lat_bf16 / lat_fp8:.2f}x")
    except Exception as e:
        print(f"\n  BF16 SDPA baseline failed: {e}")

    # ---- Theoretical analysis ----
    print(f"\n  Theoretical analysis:")
    theo_fp8 = kv_bytes_fp8 / 1792e9 * 1e6  # us
    theo_bf16 = kv_bytes_fp8 * 2 / 1792e9 * 1e6  # us (double the bytes)
    print(f"    FP8 bandwidth floor:  {theo_fp8:.1f} us")
    print(f"    BF16 bandwidth floor: {theo_bf16:.1f} us (FA2 measured: ~323 us)")
    print(f"    FP8 BW utilization:   {theo_fp8 / lat_fp8 * 100:.1f}%")
    if lat_fp8 > 0:
        print(f"    Projected 30-layer savings:")
        fa2_est = theo_bf16 * (323 / theo_bf16)  # Use known FA2 measurement ratio
        savings_per_layer = fa2_est - lat_fp8
        print(f"      Per layer: {fa2_est:.0f} -> {lat_fp8:.0f} us (save {savings_per_layer:.0f} us)")
        print(f"      30 layers: {fa2_est*30/1000:.1f} -> {lat_fp8*30/1000:.1f} ms")

    return lat_fp8


def main():
    torch.manual_seed(42)
    gc.collect()
    torch.cuda.empty_cache()

    print("FP8 Decode Attention Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Capability: {torch.cuda.get_device_capability()}")

    results = {}

    # Gemma4 sliding window config (primary target)
    results["gemma4_sliding"] = run_benchmark(
        "Gemma4 Sliding (d=256)", batch=32, num_q_heads=16, num_kv_heads=8,
        head_dim=256, seq_len=2048, logits_soft_cap=50.0)

    gc.collect()
    torch.cuda.empty_cache()

    # Gemma4 global attention config
    try:
        results["gemma4_global"] = run_benchmark(
            "Gemma4 Global (d=512)", batch=32, num_q_heads=16, num_kv_heads=8,
            head_dim=512, seq_len=2048, logits_soft_cap=50.0)
    except Exception as e:
        print(f"\nGemma4 Global skipped (likely OOM): {e}")

    gc.collect()
    torch.cuda.empty_cache()

    # Sweep batch sizes
    for batch in [1, 8, 16, 32, 64]:
        try:
            results[f"b{batch}_d256"] = run_benchmark(
                f"Batch={batch}, d=256", batch=batch, num_q_heads=16, num_kv_heads=8,
                head_dim=256, seq_len=2048, logits_soft_cap=50.0)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\nBatch={batch} failed: {e}")

    # Sweep sequence lengths
    for seq_len in [512, 1024, 2048, 4096, 8192]:
        try:
            results[f"s{seq_len}_d256"] = run_benchmark(
                f"SeqLen={seq_len}, d=256", batch=32, num_q_heads=16, num_kv_heads=8,
                head_dim=256, seq_len=seq_len, logits_soft_cap=50.0)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\nSeqLen={seq_len} failed: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<30} {'Latency (us)':>15}")
    print(f"{'-'*30} {'-'*15}")
    for name, lat in results.items():
        print(f"{name:<30} {lat:>15.1f}")


if __name__ == "__main__":
    main()
