"""Full Gemma4 sweep: all dimensions, seq_lens, batch sizes."""

import torch
import time
import sys
import csv
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec
from kv_cache_gen.generate import make_decode_fn, make_store_fn


# ===== Gemma4 26B-A4B dimensions =====

LAYER_CONFIGS = [
    # (label, D, Hq, Hk, seq_lens)
    ("sliding", 256, 16, 8, [1024]),
    ("global",  512, 16, 2, [8192, 32768, 65536, 131072]),
]

BATCH_SIZES = [1, 8, 32, 64, 128, 240]

# Spec generation
K_BITS = [4, 8]
V_BITS = [4, 8]
SCALE_BLOCKS = [16, 32, 64]
OFFSET_MAP = {2: 1.5, 4: 7.5, 8: 127.5}

# Triton configs to sweep
BLOCK_KV_OPTIONS = [16, 32]
BLOCK_H_OPTIONS = [4, 8]
NUM_WARPS_OPTIONS = [2, 4]
NUM_KV_SPLITS_OPTIONS = [32, 64, 128]


def generate_specs():
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


def quality_test(spec, D, Hq, Hk, seq_len=64, block_size=16):
    """Quick quality check."""
    B = 1
    device = "cuda"

    decode_fn = make_decode_fn(spec)
    store_fn = make_store_fn(spec)

    key_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    value_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)
    scale = 1.0 / (D ** 0.5)

    # Reference
    groups = Hq // Hk
    k_ref = key_fp.repeat_interleave(groups, dim=1) if groups > 1 else key_fp
    v_ref = value_fp.repeat_interleave(groups, dim=1) if groups > 1 else value_fp
    scores = torch.einsum("bhd,shd->bhs", query.float(), k_ref.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bhs,shd->bhd", attn, v_ref.float())

    # Quantized
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

    out = decode_fn(query, kv_cache, layer._fc_scales, block_table, seq_lens_t, scale, Hk)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_out[0].reshape(1, -1).float(), out[0].reshape(1, -1).float()
    ).item()
    return cos_sim


def benchmark_decode(spec, B, D, Hq, Hk, seq_len, block_kv, block_h,
                     num_warps, num_kv_splits, block_size=16, warmup=3, trials=10):
    """Benchmark decode at specific config."""
    device = "cuda"

    decode_fn = make_decode_fn(spec, block_kv=block_kv, block_h=block_h,
                                num_warps=num_warps, num_kv_splits=num_kv_splits)
    store_fn = make_store_fn(spec)

    # Allocate for full batch
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * B
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)

    # Check VRAM before allocating
    cache_bytes = total_blocks * block_size * Hk * cache_per_head
    query_bytes = B * Hq * D * 2
    total_bytes = cache_bytes + query_bytes + total_blocks * block_size * Hk * (D // min(spec.k_scale_block, spec.v_scale_block)) * 2 * 2
    free_mem = torch.cuda.mem_get_info()[0]
    if total_bytes > free_mem * 0.85:
        return None  # skip, would OOM

    try:
        kv_cache = torch.zeros(total_blocks, block_size, Hk, cache_per_head,
                               dtype=torch.uint8, device=device)

        # Store some data (just first seq worth, rest can be zeros for perf test)
        class FakeLayer:
            pass
        layer = FakeLayer()
        key_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        value_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
        store_fn(key_fp, value_fp, kv_cache, slot_mapping, layer, Hk)

        # Scale tensor for all blocks
        min_block = min(spec.k_scale_block, spec.v_scale_block)
        num_sb = D // min_block
        max_slots = total_blocks * block_size
        scales = torch.zeros(max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device)
        # Copy scales from layer
        if hasattr(layer, '_fc_scales'):
            copy_len = min(layer._fc_scales.shape[0], max_slots)
            scales[:copy_len] = layer._fc_scales[:copy_len]

        query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)

        # Block table: each batch element gets its own blocks
        block_table = torch.zeros(B, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(B):
            block_table[b] = torch.arange(num_blocks_per_seq, dtype=torch.int32) + b * num_blocks_per_seq

        seq_lens_t = torch.full((B,), seq_len, device=device, dtype=torch.int32)

        # Warmup
        for _ in range(warmup):
            decode_fn(query, kv_cache, scales, block_table, seq_lens_t, 1.0 / (D ** 0.5), Hk)
        torch.cuda.synchronize()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(trials):
            decode_fn(query, kv_cache, scales, block_table, seq_lens_t, 1.0 / (D ** 0.5), Hk)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        us_per_call = elapsed / trials * 1e6

        # Cleanup
        del kv_cache, scales, query, block_table, seq_lens_t, key_fp, value_fp
        torch.cuda.empty_cache()

        return us_per_call

    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None


def main():
    specs = generate_specs()
    print(f"Generated {len(specs)} specs")

    all_results = []
    total_combos = 0

    for label, D, Hq, Hk, seq_lens in LAYER_CONFIGS:
        # Quality gate
        print(f"\n{'='*80}")
        print(f"Layer type: {label} (D={D}, Hq={Hq}, Hk={Hk})")
        print(f"{'='*80}")

        passing = []
        for spec in specs:
            try:
                q = quality_test(spec, D, Hq, Hk)
                if q >= 0.95:
                    passing.append((spec, q))
            except:
                pass
        print(f"Quality gate: {len(passing)}/{len(specs)} passed (>=0.95)")

        for seq_len in seq_lens:
            for B in BATCH_SIZES:
                print(f"\n--- {label} seq={seq_len} B={B} ---")
                print(f"{'Spec':<20} {'Comp':>5} {'Qual':>6} {'μs':>8} {'splits':>6} {'bkv':>4} {'bh':>3} {'w':>2}")
                print("-" * 65)

                round_results = []
                for spec, quality in passing:
                    best_us = float("inf")
                    best_cfg = None

                    for num_kv_splits in NUM_KV_SPLITS_OPTIONS:
                        for block_kv in BLOCK_KV_OPTIONS:
                            for block_h in BLOCK_H_OPTIONS:
                                for num_warps in NUM_WARPS_OPTIONS:
                                    try:
                                        us = benchmark_decode(
                                            spec, B, D, Hq, Hk, seq_len,
                                            block_kv, block_h, num_warps, num_kv_splits)
                                    except Exception:
                                        us = None
                                    if us is not None and us < best_us:
                                        best_us = us
                                        best_cfg = (block_kv, block_h, num_warps, num_kv_splits)

                    if best_cfg and best_us < float("inf"):
                        comp = spec.compression_vs_bf16(D)
                        round_results.append({
                            "layer": label, "D": D, "Hq": Hq, "Hk": Hk,
                            "seq_len": seq_len, "batch": B,
                            "spec": spec.name, "compression": comp,
                            "quality": quality, "latency_us": best_us,
                            "block_kv": best_cfg[0], "block_h": best_cfg[1],
                            "num_warps": best_cfg[2], "num_kv_splits": best_cfg[3],
                        })

                round_results.sort(key=lambda r: r["latency_us"])
                for r in round_results[:5]:
                    print(f"{r['spec']:<20} {r['compression']:>5.1f}x {r['quality']:>5.3f} "
                          f"{r['latency_us']:>7.0f} {r['num_kv_splits']:>6} "
                          f"{r['block_kv']:>4} {r['block_h']:>3} {r['num_warps']:>2}")

                all_results.extend(round_results)

    # Save all results
    out_path = "kv_cache_gen/sweep_full_results.tsv"
    with open(out_path, "w", newline="") as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys(), delimiter="\t")
            w.writeheader()
            w.writerows(all_results)
    print(f"\n\nAll results saved to {out_path}")
    print(f"Total configs tested: {len(all_results)}")


if __name__ == "__main__":
    main()
