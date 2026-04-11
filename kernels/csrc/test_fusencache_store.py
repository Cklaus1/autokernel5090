#!/usr/bin/env python3
"""Test: verify C++ store_kv matches Triton _universal_store_kernel exactly."""

import sys
import os
sys.path.insert(0, "/root/projects/autokernel")
# Also try workspace mount path
sys.path.insert(0, "/workspace")

import torch
import torch.nn.functional as F

# Load the C++ kernel
SO_PATH = "/tmp/build_fusencache/fusencache_decode.so"
if os.path.exists(SO_PATH):
    torch.ops.load_library(SO_PATH)
    print(f"[OK] Loaded {SO_PATH}")
else:
    print(f"[ERROR] {SO_PATH} not found, run build_fusencache.py first")
    sys.exit(1)

from kv_cache_gen.spec import KVCacheSpec
from kv_cache_gen.generate import make_store_fn


def test_store_kv(spec_name, k_bits, v_bits, k_offset, v_offset,
                  k_scale_block, v_scale_block,
                  head_dim=256, num_kv_heads=8, num_tokens=32,
                  block_size=16, dtype=torch.bfloat16):
    """Compare Triton store vs C++ store for a given spec."""
    print(f"\n{'='*60}")
    print(f"Test: {spec_name}  dtype={dtype}  D={head_dim}  Hk={num_kv_heads}")
    print(f"  k_bits={k_bits} v_bits={v_bits} k_sb={k_scale_block} v_sb={v_scale_block}")
    print(f"{'='*60}")

    device = "cuda"
    D = head_dim
    Hk = num_kv_heads
    N = num_tokens

    # Create input tensors
    torch.manual_seed(42)
    key = torch.randn(N, Hk, D, device=device, dtype=dtype)
    value = torch.randn(N, Hk, D, device=device, dtype=dtype)

    # Slot mapping: sequential, with one padding entry (-1)
    num_blocks = (N + block_size - 1) // block_size + 1  # extra block
    slot_mapping = torch.arange(N, device=device, dtype=torch.int32)
    slot_mapping[N // 2] = -1  # test padding skip

    # Create spec
    spec = KVCacheSpec(
        name=spec_name,
        k_bits=k_bits, k_sym_offset=k_offset, k_scale_block=k_scale_block,
        v_bits=v_bits, v_sym_offset=v_offset, v_scale_block=v_scale_block,
    )

    cache_per_head = spec.slot_bytes(D)
    min_block = min(k_scale_block, v_scale_block)
    num_sb = D // min_block
    max_slots = num_blocks * block_size

    # ---- Triton store ----
    kv_cache_triton = torch.zeros(num_blocks, block_size, Hk, cache_per_head,
                                  dtype=torch.uint8, device=device)

    class FakeLayer:
        pass
    layer_triton = FakeLayer()
    layer_triton._fc_scales = torch.zeros(max_slots, Hk, num_sb, 2,
                                           dtype=torch.float16, device=device)

    store_fn = make_store_fn(spec)
    store_fn(key, value, kv_cache_triton, slot_mapping, layer_triton, Hk)
    torch.cuda.synchronize()

    # ---- C++ store ----
    kv_cache_cpp = torch.zeros_like(kv_cache_triton)
    scales_cpp = torch.zeros(max_slots, Hk, num_sb, 2,
                             dtype=torch.float16, device=device)

    torch.ops.fusencache.store_kv(
        key.contiguous(), value.contiguous(),
        kv_cache_cpp, scales_cpp, slot_mapping,
        D, block_size,
        k_bits, v_bits,
        k_scale_block, v_scale_block,
        k_offset, v_offset,
    )
    torch.cuda.synchronize()

    # ---- Compare ----
    scales_triton = layer_triton._fc_scales

    # Compare KV cache bytes
    cache_match = torch.equal(kv_cache_triton, kv_cache_cpp)
    cache_diff = (kv_cache_triton.int() - kv_cache_cpp.int()).abs()
    cache_max_diff = cache_diff.max().item()
    cache_nonzero = cache_diff.nonzero().shape[0]

    # Compare scales
    scales_match = torch.allclose(scales_triton, scales_cpp, rtol=1e-3, atol=1e-5)
    scales_diff = (scales_triton.float() - scales_cpp.float()).abs()
    scales_max_diff = scales_diff.max().item()

    # Check that padding slot was skipped
    padded_slot = N // 2
    pad_blk = padded_slot // block_size
    pad_off = padded_slot % block_size
    pad_cache_cpp = kv_cache_cpp[pad_blk, pad_off].sum().item()
    pad_scales_cpp = scales_cpp[padded_slot].sum().item()

    status = "PASS" if cache_match and scales_match else "FAIL"

    print(f"  Cache match:  {cache_match}  (max_diff={cache_max_diff}, nonzero={cache_nonzero})")
    print(f"  Scales match: {scales_match}  (max_diff={scales_max_diff:.6f})")
    print(f"  Padding skip: cache_sum={pad_cache_cpp} scales_sum={pad_scales_cpp} "
          f"({'OK' if pad_cache_cpp == 0 and pad_scales_cpp == 0 else 'FAIL'})")
    print(f"  => {status}")

    if not cache_match:
        # Show first few mismatches
        mismatch_locs = cache_diff.nonzero()[:5]
        print(f"  First mismatches at: {mismatch_locs.tolist()}")
        for loc in mismatch_locs:
            t_val = kv_cache_triton[tuple(loc)].item()
            c_val = kv_cache_cpp[tuple(loc)].item()
            print(f"    {loc.tolist()}: triton={t_val} cpp={c_val}")

    return status == "PASS"


def bench_store_kv(k_bits=4, v_bits=4, head_dim=256, num_kv_heads=8,
                   num_tokens=128, block_size=16, dtype=torch.bfloat16):
    """Benchmark Triton vs C++ store kernel."""
    print(f"\n{'='*60}")
    print(f"Benchmark: k{k_bits}v{v_bits} D={head_dim} Hk={num_kv_heads} N={num_tokens}")
    print(f"{'='*60}")

    device = "cuda"
    D = head_dim
    Hk = num_kv_heads
    N = num_tokens
    k_offset = (2**(k_bits-1)) - 0.5
    v_offset = (2**(v_bits-1)) - 0.5
    k_scale_block = 64
    v_scale_block = 64

    spec = KVCacheSpec(
        name=f'k{k_bits}v{v_bits}',
        k_bits=k_bits, k_sym_offset=k_offset, k_scale_block=k_scale_block,
        v_bits=v_bits, v_sym_offset=v_offset, v_scale_block=v_scale_block,
    )

    key = torch.randn(N, Hk, D, device=device, dtype=dtype)
    value = torch.randn(N, Hk, D, device=device, dtype=dtype)
    slot_mapping = torch.arange(N, device=device, dtype=torch.int32)

    num_blocks = (N + block_size - 1) // block_size + 1
    cache_per_head = spec.slot_bytes(D)
    min_block = min(k_scale_block, v_scale_block)
    num_sb = D // min_block
    max_slots = num_blocks * block_size

    # Triton
    kv_cache_t = torch.zeros(num_blocks, block_size, Hk, cache_per_head,
                             dtype=torch.uint8, device=device)
    class L: pass
    layer_t = L()
    layer_t._fc_scales = torch.zeros(max_slots, Hk, num_sb, 2,
                                      dtype=torch.float16, device=device)
    store_fn = make_store_fn(spec)

    # Warmup
    for _ in range(5):
        store_fn(key, value, kv_cache_t, slot_mapping, layer_t, Hk)
    torch.cuda.synchronize()

    # Time Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        store_fn(key, value, kv_cache_t, slot_mapping, layer_t, Hk)
    end.record()
    torch.cuda.synchronize()
    triton_us = start.elapsed_time(end) * 1000 / 100

    # C++
    kv_cache_c = torch.zeros_like(kv_cache_t)
    scales_c = torch.zeros(max_slots, Hk, num_sb, 2,
                           dtype=torch.float16, device=device)

    # Warmup
    for _ in range(5):
        torch.ops.fusencache.store_kv(
            key, value, kv_cache_c, scales_c, slot_mapping,
            D, block_size, k_bits, v_bits, k_scale_block, v_scale_block,
            k_offset, v_offset,
        )
    torch.cuda.synchronize()

    start.record()
    for _ in range(100):
        torch.ops.fusencache.store_kv(
            key, value, kv_cache_c, scales_c, slot_mapping,
            D, block_size, k_bits, v_bits, k_scale_block, v_scale_block,
            k_offset, v_offset,
        )
    end.record()
    torch.cuda.synchronize()
    cpp_us = start.elapsed_time(end) * 1000 / 100

    print(f"  Triton: {triton_us:.1f} us")
    print(f"  C++:    {cpp_us:.1f} us")
    print(f"  Ratio:  {triton_us/cpp_us:.2f}x")


if __name__ == "__main__":
    all_pass = True

    # Test k4v4 (most common config)
    all_pass &= test_store_kv(
        "k4v4", k_bits=4, v_bits=4,
        k_offset=7.5, v_offset=7.5,
        k_scale_block=64, v_scale_block=64,
        head_dim=256, dtype=torch.bfloat16,
    )

    # Test k4v4 with FP16 input
    all_pass &= test_store_kv(
        "k4v4_fp16", k_bits=4, v_bits=4,
        k_offset=7.5, v_offset=7.5,
        k_scale_block=64, v_scale_block=64,
        head_dim=256, dtype=torch.float16,
    )

    # Test k4v4 with head_dim=512 (global attention)
    all_pass &= test_store_kv(
        "k4v4_d512", k_bits=4, v_bits=4,
        k_offset=7.5, v_offset=7.5,
        k_scale_block=64, v_scale_block=64,
        head_dim=512, dtype=torch.bfloat16,
    )

    # Test k8v8
    all_pass &= test_store_kv(
        "k8v8", k_bits=8, v_bits=8,
        k_offset=127.5, v_offset=127.5,
        k_scale_block=64, v_scale_block=64,
        head_dim=256, dtype=torch.bfloat16,
    )

    # Test k8v4
    all_pass &= test_store_kv(
        "k8v4", k_bits=8, v_bits=4,
        k_offset=127.5, v_offset=7.5,
        k_scale_block=64, v_scale_block=64,
        head_dim=256, dtype=torch.bfloat16,
    )

    # Test k2v2
    all_pass &= test_store_kv(
        "k2v2", k_bits=2, v_bits=2,
        k_offset=1.5, v_offset=1.5,
        k_scale_block=64, v_scale_block=64,
        head_dim=256, dtype=torch.bfloat16,
    )

    # Test with different scale blocks
    all_pass &= test_store_kv(
        "k4v4_sb32", k_bits=4, v_bits=4,
        k_offset=7.5, v_offset=7.5,
        k_scale_block=32, v_scale_block=32,
        head_dim=256, dtype=torch.bfloat16,
    )

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")

    # Benchmark if all tests pass
    if all_pass:
        bench_store_kv(k_bits=4, v_bits=4, head_dim=256, num_tokens=1)
        bench_store_kv(k_bits=4, v_bits=4, head_dim=256, num_tokens=32)
        bench_store_kv(k_bits=4, v_bits=4, head_dim=256, num_tokens=128)

    sys.exit(0 if all_pass else 1)
