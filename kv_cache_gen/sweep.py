"""Sweep all KV cache specs × Triton configs. Find the pareto frontier."""

import torch
import time
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec
from kv_cache_gen.generate import make_decode_fn, make_store_fn


# ===== Config space =====

K_BITS = [2, 4, 8]
V_BITS = [2, 4, 8]
SCALE_BLOCKS = [16, 32, 64]
OFFSET_MAP = {2: 1.5, 4: 7.5, 8: 127.5}

BLOCK_KV_OPTIONS = [8, 16, 32]
BLOCK_H_OPTIONS = [4, 8]
NUM_WARPS_OPTIONS = [2, 4]


def generate_all_specs():
    """Generate all valid KV cache specs."""
    specs = []
    for k_bits in K_BITS:
        for v_bits in V_BITS:
            for k_block in SCALE_BLOCKS:
                for v_block in SCALE_BLOCKS:
                    if k_block < k_bits or v_block < v_bits:
                        continue
                    name = f"k{k_bits}v{v_bits}kb{k_block}vb{v_block}"
                    specs.append(KVCacheSpec(
                        name=name,
                        k_bits=k_bits, k_sym_offset=OFFSET_MAP[k_bits],
                        k_scale_block=k_block,
                        v_bits=v_bits, v_sym_offset=OFFSET_MAP[v_bits],
                        v_scale_block=v_block,
                    ))
    return specs


def python_reference_attention(query, key, value, scale):
    """Reference attention for quality testing."""
    B, Hq, D = query.shape
    _, Hk, _ = key.shape
    groups = Hq // Hk
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    scores = torch.einsum("bhd,shd->bhs", query.float(), key.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,shd->bhd", attn, value.float())


def test_quality(spec, D=256, Hq=16, Hk=8, seq_len=32, block_size=16):
    """Test spec quality, return cosine similarity."""
    B = 1
    device = "cuda"

    decode_fn = make_decode_fn(spec)
    store_fn = make_store_fn(spec)

    key_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    value_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    ref_out = python_reference_attention(query, key_fp, value_fp, scale)

    num_blocks = (seq_len + block_size - 1) // block_size
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)

    class FakeLayer:
        pass
    layer = FakeLayer()

    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
    store_fn(key_fp, value_fp, kv_cache, slot_mapping, layer, Hk)

    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

    out = decode_fn(query, kv_cache, layer._fc_scales, block_table, seq_lens, scale, Hk)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_out[0].reshape(1, -1).float(), out[0].reshape(1, -1).float()
    ).item()

    return cos_sim


def benchmark_decode(spec, block_kv, block_h, num_warps,
                     D=256, Hq=16, Hk=8, seq_len=512, block_size=16,
                     warmup=5, trials=20):
    """Benchmark decode throughput for a specific config."""
    B = 1
    device = "cuda"

    decode_fn = make_decode_fn(spec, block_kv=block_kv, block_h=block_h,
                                num_warps=num_warps)
    store_fn = make_store_fn(spec)

    key_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    value_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    num_blocks = (seq_len + block_size - 1) // block_size
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)

    class FakeLayer:
        pass
    layer = FakeLayer()

    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
    store_fn(key_fp, value_fp, kv_cache, slot_mapping, layer, Hk)

    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    seq_lens_t = torch.tensor([seq_len], device=device, dtype=torch.int32)

    # Warmup
    for _ in range(warmup):
        decode_fn(query, kv_cache, layer._fc_scales, block_table, seq_lens_t, scale, Hk)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(trials):
        decode_fn(query, kv_cache, layer._fc_scales, block_table, seq_lens_t, scale, Hk)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    us_per_call = elapsed / trials * 1e6
    return us_per_call


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-only", action="store_true",
                        help="Only test quality, skip Triton autotune")
    parser.add_argument("--min-quality", type=float, default=0.90,
                        help="Minimum cosine similarity to pass quality gate")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N results by throughput")
    args = parser.parse_args()

    specs = generate_all_specs()
    print(f"Generated {len(specs)} specs")
    print()

    # Phase 1: Quality gate
    print("Phase 1: Quality test (all specs)...")
    print(f"{'Spec':<20} {'Comp':>5} {'CosSim':>8} {'Status'}")
    print("-" * 45)

    passing_specs = []
    for spec in specs:
        try:
            cos_sim = test_quality(spec)
            comp = spec.compression_vs_bf16(256)
            status = "PASS" if cos_sim >= args.min_quality else "FAIL"
            print(f"{spec.name:<20} {comp:>5.1f}x {cos_sim:>7.4f} {status}")
            if cos_sim >= args.min_quality:
                passing_specs.append((spec, cos_sim))
        except Exception as e:
            print(f"{spec.name:<20} ERROR: {str(e)[:50]}")

    print(f"\n{len(passing_specs)}/{len(specs)} passed quality gate (>={args.min_quality})")

    if args.quality_only:
        return

    # Phase 2: Triton autotune on passing specs
    print(f"\nPhase 2: Triton autotune ({len(passing_specs)} specs × "
          f"{len(BLOCK_KV_OPTIONS) * len(BLOCK_H_OPTIONS) * len(NUM_WARPS_OPTIONS)} configs each)...")

    results = []
    total = len(passing_specs) * len(BLOCK_KV_OPTIONS) * len(BLOCK_H_OPTIONS) * len(NUM_WARPS_OPTIONS)
    done = 0

    for spec, quality in passing_specs:
        best_us = float("inf")
        best_config = None

        for block_kv in BLOCK_KV_OPTIONS:
            for block_h in BLOCK_H_OPTIONS:
                for num_warps in NUM_WARPS_OPTIONS:
                    done += 1
                    try:
                        us = benchmark_decode(spec, block_kv, block_h, num_warps)
                        if us < best_us:
                            best_us = us
                            best_config = (block_kv, block_h, num_warps)
                    except Exception:
                        pass

                    if done % 50 == 0:
                        print(f"  [{done}/{total}]", flush=True)

        if best_config:
            comp = spec.compression_vs_bf16(256)
            results.append({
                "spec": spec,
                "quality": quality,
                "compression": comp,
                "latency_us": best_us,
                "block_kv": best_config[0],
                "block_h": best_config[1],
                "num_warps": best_config[2],
            })

    # Sort by latency
    results.sort(key=lambda r: r["latency_us"])

    print(f"\n{'='*85}")
    print(f"Top {args.top} by throughput (lower μs = faster):")
    print(f"{'Spec':<20} {'Comp':>5} {'Qual':>6} {'μs':>8} {'BLK_KV':>6} {'BLK_H':>5} {'Warps':>5}")
    print("-" * 65)
    for r in results[:args.top]:
        print(f"{r['spec'].name:<20} {r['compression']:>5.1f}x {r['quality']:>5.3f} "
              f"{r['latency_us']:>7.0f} {r['block_kv']:>6} {r['block_h']:>5} {r['num_warps']:>5}")

    # Pareto frontier: best throughput at each compression level
    print(f"\nPareto frontier (best speed per compression tier):")
    print(f"{'Spec':<20} {'Comp':>5} {'Qual':>6} {'μs':>8} {'Config'}")
    print("-" * 65)

    tiers = [(1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 7.5)]
    for lo, hi in tiers:
        tier_results = [r for r in results if lo <= r["compression"] < hi]
        if tier_results:
            best = min(tier_results, key=lambda r: r["latency_us"])
            cfg = f"kv={best['block_kv']} h={best['block_h']} w={best['num_warps']}"
            print(f"{best['spec'].name:<20} {best['compression']:>5.1f}x {best['quality']:>5.3f} "
                  f"{best['latency_us']:>7.0f} {cfg}")

    # Save results
    with open("kv_cache_gen/sweep_results.tsv", "w") as f:
        f.write("spec\tcompression\tquality\tlatency_us\tblock_kv\tblock_h\tnum_warps\n")
        for r in results:
            f.write(f"{r['spec'].name}\t{r['compression']:.2f}\t{r['quality']:.4f}\t"
                    f"{r['latency_us']:.0f}\t{r['block_kv']}\t{r['block_h']}\t{r['num_warps']}\n")
    print(f"\nResults saved to kv_cache_gen/sweep_results.tsv")


if __name__ == "__main__":
    main()
