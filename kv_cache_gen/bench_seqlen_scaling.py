"""Sequence length scaling curve: single-layer benchmark to avoid OOM at long contexts."""

import torch
import time
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec
from kv_cache_gen.generate import make_decode_fn, make_store_fn

SPECS = {
    'k4v4kb64vb64': KVCacheSpec(name='k4v4kb64vb64', k_bits=4, k_sym_offset=7.5,
                                 k_scale_block=64, v_bits=4, v_sym_offset=7.5, v_scale_block=64),
    'k8v4kb32vb32': KVCacheSpec(name='k8v4kb32vb32', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=32, v_bits=4, v_sym_offset=7.5, v_scale_block=32),
    'k8v8kb64vb64': KVCacheSpec(name='k8v8kb64vb64', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=64, v_bits=8, v_sym_offset=127.5, v_scale_block=64),
}

# Single-layer benchmarks to fit long sequences
# Test both layer types across sequence lengths
LAYER_CONFIGS = {
    'sliding': (256, 8),   # D, Hk
    'global':  (512, 2),
}

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
BATCH_SIZES = [1, 8, 32]
NUM_KV_SPLITS_MAP = {
    512: 32, 1024: 32, 2048: 32, 4096: 64,
    8192: 64, 16384: 64, 32768: 128, 65536: 128, 131072: 128,
}

WARMUP = 3
TRIALS = 10


def bench_single_layer(spec, B, D, Hk, seq_len, num_kv_splits=64):
    device = 'cuda'
    block_size = 16

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * B
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)

    # Check VRAM
    cache_bytes = total_blocks * block_size * Hk * cache_per_head
    free = torch.cuda.mem_get_info()[0]
    if cache_bytes > free * 0.80:
        return None

    try:
        kv_cache = torch.zeros(total_blocks, block_size, Hk, cache_per_head,
                               dtype=torch.uint8, device=device)

        # Scales
        min_block = min(spec.k_scale_block, spec.v_scale_block)
        num_sb = D // min_block
        max_slots = total_blocks * block_size
        scales = torch.zeros(max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device)

        # Block table
        block_table = torch.zeros(B, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(B):
            block_table[b] = torch.arange(num_blocks_per_seq, dtype=torch.int32) + b * num_blocks_per_seq

        seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
        query = torch.randn(B, 16, D, device=device, dtype=torch.float16)  # Hq=16

        decode_fn = make_decode_fn(spec, block_kv=16, block_h=8, num_warps=2,
                                    num_kv_splits=num_kv_splits)

        # Warmup
        for _ in range(WARMUP):
            decode_fn(query, kv_cache, scales, block_table, seq_lens, 1.0 / (D ** 0.5), Hk)
        torch.cuda.synchronize()

        # Bench
        t0 = time.perf_counter()
        for _ in range(TRIALS):
            decode_fn(query, kv_cache, scales, block_table, seq_lens, 1.0 / (D ** 0.5), Hk)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        us = (elapsed / TRIALS) * 1e6

        del kv_cache, scales, block_table, seq_lens, query
        torch.cuda.empty_cache()

        # Also compute memory bandwidth utilization
        # Bytes read per call: B * seq_len * Hk * cache_per_head (cache) + B * 16 * D * 2 (query)
        bytes_read = B * seq_len * Hk * cache_per_head + B * 16 * D * 2
        bandwidth_gbps = bytes_read / (elapsed / TRIALS) / 1e9

        return us, bandwidth_gbps

    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None


def main():
    print("=" * 90)
    print("Sequence Length Scaling: single-layer decode latency vs context length")
    print("RTX 5090 memory bandwidth: ~1,792 GB/s (theoretical)")
    print("=" * 90)

    for layer_label, (D, Hk) in LAYER_CONFIGS.items():
        for B in BATCH_SIZES:
            print(f"\n--- {layer_label} (D={D}, Hk={Hk}) B={B} ---")
            print(f"{'Seq Len':>8}", end="")
            for name in SPECS:
                comp = SPECS[name].compression_vs_bf16(D)
                print(f" {name+f'({comp:.1f}x)':>22} {'BW':>8}", end="")
            print()
            print("-" * (8 + 31 * len(SPECS)))

            for seq_len in SEQ_LENS:
                row = f"{seq_len:>8}"
                splits = NUM_KV_SPLITS_MAP.get(seq_len, 64)
                for name, spec in SPECS.items():
                    r = bench_single_layer(spec, B, D, Hk, seq_len, splits)
                    if r:
                        us, bw = r
                        row += f" {us:>12.0f} μs {bw:>6.0f} GB/s"
                    else:
                        row += f" {'OOM':>12}    {'':>6}    "
                print(row)

    # 128K context projection
    print("\n" + "=" * 90)
    print("128K Context Projection: estimated per-layer decode latency")
    print("=" * 90)
    print(f"{'Layer':>10} {'Spec':<18} {'B=1 μs':>10} {'B=8 μs':>10} {'B=32 μs':>10}")
    print("-" * 60)

    best_spec_name = 'k4v4kb64vb64'
    best_spec = SPECS[best_spec_name]
    for layer_label, (D, Hk) in LAYER_CONFIGS.items():
        for B in [1, 8, 32]:
            if layer_label == 'sliding':
                seq = 1024  # sliding window caps at 1024
            else:
                seq = 131072  # full 128K for global layers
            r = bench_single_layer(best_spec, B, D, Hk, seq, 128)
            if r:
                us, bw = r
                print(f"{layer_label:>10} {best_spec_name:<18} {'' if B != 1 else ''}{us:>10.0f}", end="")
                if B == 32:
                    print()
            else:
                print(f"{layer_label:>10} {best_spec_name:<18} {'OOM':>10}", end="")
                if B == 32:
                    print()

    # Full model 128K estimate
    print("\n--- Full model 128K decode estimate (k4v4kb64vb64) ---")
    for B in [1, 8]:
        sliding_r = bench_single_layer(best_spec, B, 256, 8, 1024, 32)
        global_r = bench_single_layer(best_spec, B, 512, 2, 131072, 128)
        if sliding_r and global_r:
            total_us = sliding_r[0] * 25 + global_r[0] * 5
            total_ms = total_us / 1000
            # Attention only — MoE adds ~2-3x
            est_full_ms = total_ms * 3  # rough MoE multiplier from simulator
            est_tps = B / (est_full_ms / 1000)
            print(f"  B={B}: attention={total_ms:.1f}ms, est full decode={est_full_ms:.0f}ms, est {est_tps:.0f} tok/s")
        else:
            print(f"  B={B}: OOM at 128K global layers")


if __name__ == "__main__":
    main()
