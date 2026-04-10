"""Precompile Triton kernels to avoid cold-start latency.

First call to a Triton kernel triggers JIT compilation (~2-5 seconds).
With mixed-spec + adaptive selection, startup could trigger 4+ compilations.
This module precompiles all needed kernels with tiny dummy tensors so the
actual inference path never hits a compile.

Usage:
    from kv_cache_gen.warmup import precompile

    # Precompile for a specific config
    precompile(config)

    # Or precompile common configs
    precompile_common()
"""

import torch
import time
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn, make_store_fn


def _warmup_decode(spec: KVCacheSpec, D: int, Hq: int, Hk: int,
                   block_kv: int = 16, block_h: int = 8,
                   num_warps: int = 2, num_kv_splits: int = 32,
                   device: str = "cuda"):
    """Trigger Triton compilation for a decode kernel config."""
    B, seq_len, block_size = 1, 32, 16
    num_blocks = (seq_len + block_size - 1) // block_size
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)

    kv_cache = torch.zeros(num_blocks, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    if min_block == 0:
        return  # FP8 not yet supported in kernel
    num_sb = D // min_block
    scales = torch.zeros(num_blocks * block_size, Hk, num_sb, 2,
                         dtype=torch.float16, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    query = torch.randn(B, Hq, D, dtype=torch.float16, device=device)

    decode_fn = make_decode_fn(spec, block_kv=block_kv, block_h=block_h,
                                num_warps=num_warps, num_kv_splits=num_kv_splits)
    decode_fn(query, kv_cache, scales, block_table, seq_lens, 1.0 / (D ** 0.5), Hk)
    torch.cuda.synchronize()

    del kv_cache, scales, block_table, seq_lens, query


def _warmup_store(spec: KVCacheSpec, D: int, Hk: int, device: str = "cuda"):
    """Trigger Triton compilation for a store kernel config."""
    N, block_size = 16, 16
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)

    if spec.k_scale_block == 0 or spec.v_scale_block == 0:
        return  # FP8 not yet supported

    kv_cache = torch.zeros(1, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)
    key = torch.randn(N, Hk, D, dtype=torch.float16, device=device)
    value = torch.randn(N, Hk, D, dtype=torch.float16, device=device)
    slot_mapping = torch.arange(N, dtype=torch.int32, device=device)

    class L:
        pass
    layer = L()

    store_fn = make_store_fn(spec)
    store_fn(key, value, kv_cache, slot_mapping, layer, Hk)
    torch.cuda.synchronize()

    del kv_cache, key, value


def precompile(config=None, specs=None, device="cuda", verbose=True):
    """Precompile Triton kernels for given specs or adaptive config.

    Args:
        config: AdaptiveConfig from adaptive.py. If provided, compiles
                exactly the kernels needed for that config.
        specs: List of (spec, D, Hq, Hk, block_kv, block_h, num_warps, num_kv_splits)
               tuples. If provided, compiles these specific configs.
        device: CUDA device.
        verbose: Print compilation progress.
    """
    t0 = time.perf_counter()
    compiled = 0

    if config is not None:
        # Compile from AdaptiveConfig
        from kv_cache_gen.adaptive import AdaptiveConfig
        targets = [
            ("sliding", config.sliding_spec, 256, 16, 8, config.sliding_config),
            ("global", config.global_spec, 512, 16, 2, config.global_config),
        ]
        for label, spec, D, Hq, Hk, cfg in targets:
            if verbose:
                print(f"  Compiling {label} decode ({spec.name})...", end="", flush=True)
            _warmup_decode(spec, D, Hq, Hk,
                           block_kv=cfg.get("block_kv", 16),
                           block_h=cfg.get("block_h", 8),
                           num_warps=cfg.get("num_warps", 2),
                           num_kv_splits=cfg.get("num_kv_splits", 32),
                           device=device)
            compiled += 1
            if verbose:
                print(" done", flush=True)

            if verbose:
                print(f"  Compiling {label} store ({spec.name})...", end="", flush=True)
            _warmup_store(spec, D, Hk, device=device)
            compiled += 1
            if verbose:
                print(" done", flush=True)

    elif specs is not None:
        for spec, D, Hq, Hk, bkv, bh, nw, nks in specs:
            if verbose:
                print(f"  Compiling {spec.name} (D={D})...", end="", flush=True)
            _warmup_decode(spec, D, Hq, Hk, bkv, bh, nw, nks, device)
            _warmup_store(spec, D, Hk, device)
            compiled += 2
            if verbose:
                print(" done", flush=True)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"Precompiled {compiled} kernels in {elapsed:.1f}s")
    return compiled, elapsed


def precompile_common(device="cuda", verbose=True):
    """Precompile the most common kernel configs for Gemma4.

    Covers: k4v4, k8v4, k8v8 × sliding/global dimensions × 3 split-K values.
    Takes ~15-30 seconds but eliminates all cold-start latency.
    """
    if verbose:
        print("Precompiling common Gemma4 kernel configs...")

    common_specs = [
        # (spec_name, D, Hq, Hk, block_kv, block_h, num_warps, num_kv_splits)
        # Sliding layers
        ("k4v4b64", 256, 16, 8, 16, 8, 2, 16),
        ("k4v4b64", 256, 16, 8, 16, 8, 2, 32),
        ("k8v4b32", 256, 16, 8, 16, 8, 2, 32),
        ("k8v8b32", 256, 16, 8, 16, 8, 2, 32),
        # Global layers
        ("k4v4b64", 512, 16, 2, 16, 8, 2, 64),
        ("k4v4b64", 512, 16, 2, 16, 8, 2, 128),
        ("k8v4b32", 512, 16, 2, 16, 8, 2, 64),
        ("k8v8b32", 512, 16, 2, 16, 8, 2, 64),
    ]

    targets = []
    for spec_name, D, Hq, Hk, bkv, bh, nw, nks in common_specs:
        if spec_name in PREDEFINED_SPECS:
            targets.append((PREDEFINED_SPECS[spec_name], D, Hq, Hk, bkv, bh, nw, nks))

    return precompile(specs=targets, device=device, verbose=verbose)


if __name__ == "__main__":
    precompile_common()
