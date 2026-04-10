"""Benchmark the store (quantize+pack+scatter) kernel for prefill TTFT analysis."""

import torch
import time
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec
from kv_cache_gen.generate import make_store_fn

SPECS = {
    'k4v4kb64vb64': KVCacheSpec(name='k4v4kb64vb64', k_bits=4, k_sym_offset=7.5,
                                 k_scale_block=64, v_bits=4, v_sym_offset=7.5, v_scale_block=64),
    'k8v4kb32vb32': KVCacheSpec(name='k8v4kb32vb32', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=32, v_bits=4, v_sym_offset=7.5, v_scale_block=32),
    'k8v8kb64vb64': KVCacheSpec(name='k8v8kb64vb64', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=64, v_bits=8, v_sym_offset=127.5, v_scale_block=64),
    'k8v4kb16vb16': KVCacheSpec(name='k8v4kb16vb16', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=16, v_bits=4, v_sym_offset=7.5, v_scale_block=16),
}

# Gemma4 dimensions
CONFIGS = [
    ('sliding', 256, 8, [128, 512, 1024]),
    ('global',  512, 2, [512, 2048, 8192]),
]

WARMUP = 3
TRIALS = 20


def bench_store(spec, D, Hk, num_tokens, block_size=16):
    device = 'cuda'
    store_fn = make_store_fn(spec)

    num_blocks = (num_tokens + block_size - 1) // block_size
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)

    key = torch.randn(num_tokens, Hk, D, device=device, dtype=torch.float16)
    value = torch.randn(num_tokens, Hk, D, device=device, dtype=torch.float16)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)

    class FakeLayer:
        pass
    layer = FakeLayer()

    # Warmup
    for _ in range(WARMUP):
        layer._fc_scales = None
        if hasattr(layer, '_fc_scales'):
            del layer._fc_scales
        store_fn(key, value, kv_cache, slot_mapping, layer, Hk)
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(TRIALS):
        store_fn(key, value, kv_cache, slot_mapping, layer, Hk)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    us_per_call = (elapsed / TRIALS) * 1e6
    tokens_per_sec = num_tokens / (elapsed / TRIALS)

    del kv_cache, key, value
    torch.cuda.empty_cache()
    return us_per_call, tokens_per_sec


def main():
    print("=" * 80)
    print("Store Kernel Benchmark (quantize + pack + scatter)")
    print("Impact: runs once per prefill token → bottlenecks time-to-first-token")
    print("=" * 80)

    for layer_label, D, Hk, token_counts in CONFIGS:
        print(f"\n--- {layer_label} (D={D}, Hk={Hk}) ---")
        print(f"{'Spec':<20} {'Comp':>5}", end="")
        for n in token_counts:
            print(f" {f'{n}tok μs':>10} {f'Mtok/s':>7}", end="")
        print()
        print("-" * (27 + 17 * len(token_counts)))

        for name, spec in SPECS.items():
            comp = spec.compression_vs_bf16(D)
            row = f"{name:<20} {comp:>5.1f}x"
            for n in token_counts:
                try:
                    us, tps = bench_store(spec, D, Hk, n)
                    row += f" {us:>10.0f} {tps/1e6:>7.2f}"
                except Exception as e:
                    row += f" {'ERR':>10} {'':>7}"
            print(row)

    # Prefill TTFT estimate
    print("\n" + "=" * 80)
    print("TTFT Estimate: store overhead for 2048-token prefill")
    print("=" * 80)
    print(f"{'Spec':<20} {'Sliding (25L)':>14} {'Global (5L)':>12} {'Total':>10} {'% of 100ms':>10}")
    print("-" * 68)
    for name, spec in SPECS.items():
        try:
            us_s, _ = bench_store(spec, 256, 8, 2048)
            us_g, _ = bench_store(spec, 512, 2, 2048)
            total_us = us_s * 25 + us_g * 5
            total_ms = total_us / 1000
            pct = total_ms / 100 * 100  # vs 100ms TTFT target
            print(f"{name:<20} {us_s*25/1000:>12.1f}ms {us_g*5/1000:>10.1f}ms {total_ms:>8.1f}ms {pct:>9.1f}%")
        except:
            print(f"{name:<20} {'ERR':>14}")


if __name__ == "__main__":
    main()
