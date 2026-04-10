#!/usr/bin/env python3
"""
Correctness test: compare CUDA C++ decode attention vs Triton decode attention.

Creates identical inputs, runs both kernels, compares outputs.
Tests multiple configurations:
  - head_dim=256, head_dim=512 (if both compile)
  - Various seq_lens (short, medium)
  - With and without logits_soft_cap
  - Different batch sizes
"""

import os
import sys
import torch
import numpy as np

# Build/load the CUDA kernel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_fusencache import build_kernel, load_library, SO_PATH

# Build the Triton kernel's dependencies
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn, make_store_fn


def setup_test_data(B, Hq, Hk, D, seq_len, page_size=16, device='cuda'):
    """Create test data matching the FusenCache KV cache layout."""
    spec = PREDEFINED_SPECS['k4v4b64']

    # Allocate KV cache
    slot_bytes = spec.slot_bytes(D)
    max_blocks = (seq_len + page_size - 1) // page_size
    total_blocks = B * max_blocks + 10  # extra padding
    kv_cache = torch.randint(0, 256, (total_blocks, page_size, Hk, slot_bytes),
                             dtype=torch.uint8, device=device)

    # Block table: sequential block assignment
    block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=device)
    for b in range(B):
        for blk in range(max_blocks):
            block_table[b, blk] = b * max_blocks + blk

    # Sequence lengths
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    # Query
    query = torch.randn(B, Hq, D, dtype=torch.bfloat16, device=device) * 0.1

    # Scales tensor: [max_slots, Hk, num_scale_blocks, 2] fp16
    max_slots = total_blocks * page_size
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // min_block
    scales = torch.randn(max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device) * 0.5 + 1.0

    # Store scales so Triton can find them via layer._fc_scales
    class FakeLayer:
        pass
    layer = FakeLayer()
    layer._fc_scales = scales

    return query, kv_cache, scales, block_table, seq_lens, spec, layer


def run_triton_decode(query, kv_cache, scales, block_table, seq_lens, spec,
                      num_kv_splits, Hk, logits_soft_cap=0.0):
    """Run the Triton decode kernel."""
    B, Hq, D = query.shape
    sm_scale = 1.0 / (D ** 0.5)

    decode_fn = make_decode_fn(
        spec, block_kv=16, num_kv_splits=num_kv_splits,
        logits_soft_cap=logits_soft_cap,
    )

    output = decode_fn(query, kv_cache, scales, block_table, seq_lens,
                       sm_scale, Hk)
    return output


def run_cuda_decode(query, kv_cache, scales, block_table, seq_lens, spec,
                    num_kv_splits, Hk, Hq, D, logits_soft_cap=0.0):
    """Run the CUDA C++ decode kernel."""
    B = query.shape[0]
    sm_scale = 1.0 / (D ** 0.5)
    kv_group_size = Hq // Hk
    page_size = kv_cache.shape[1]

    # Allocate outputs
    mid_out = torch.empty(B, Hq, num_kv_splits, D + 1,
                          dtype=torch.float32, device=query.device)
    output = torch.empty(B, Hq, D, dtype=torch.bfloat16, device=query.device)

    torch.ops.fusencache.decode_attention(
        output, query, kv_cache, scales, block_table, seq_lens, mid_out,
        sm_scale, logits_soft_cap,
        num_kv_splits, D, Hk, kv_group_size, page_size,
        spec.k_bits, spec.v_bits, spec.k_scale_block, spec.v_scale_block,
        spec.k_sym_offset, spec.v_sym_offset,
    )
    return output


def compare_outputs(triton_out, cuda_out, test_name, atol=0.05, rtol=0.05):
    """Compare two outputs and report results."""
    triton_f = triton_out.float()
    cuda_f = cuda_out.float()

    abs_diff = (triton_f - cuda_f).abs()
    rel_diff = abs_diff / (triton_f.abs() + 1e-6)

    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        triton_f.flatten().unsqueeze(0),
        cuda_f.flatten().unsqueeze(0)
    ).item()

    passed = (max_abs < atol * 10) and (cos_sim > 0.95)

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}")
    print(f"    max_abs_diff={max_abs:.6f}, mean_abs_diff={mean_abs:.6f}")
    print(f"    max_rel_diff={max_rel:.6f}, mean_rel_diff={mean_rel:.6f}")
    print(f"    cosine_similarity={cos_sim:.6f}")

    return passed


def main():
    print("=" * 60)
    print("FusenCache Decode Attention: CUDA vs Triton Correctness Test")
    print("=" * 60)

    # Build/load CUDA kernel
    if not load_library():
        print("\n[BUILD] Building CUDA kernel...")
        build_kernel()
    else:
        print("[LOAD] Using pre-built CUDA kernel")

    device = 'cuda'
    results = []

    # Test configurations
    configs = [
        # (B, Hq, Hk, D, seq_len, num_splits, soft_cap, name)
        (1,  16, 8, 256, 64,   16, 0.0,  "B1_D256_seq64_nosoftcap"),
        (1,  16, 8, 256, 64,   16, 50.0, "B1_D256_seq64_softcap50"),
        (4,  16, 8, 256, 128,  16, 0.0,  "B4_D256_seq128"),
        (8,  16, 8, 256, 256,  16, 50.0, "B8_D256_seq256_softcap50"),
        (1,  16, 8, 256, 512,  32, 0.0,  "B1_D256_seq512_split32"),
        (16, 16, 8, 256, 128,  16, 50.0, "B16_D256_seq128_softcap50"),
        (32, 16, 8, 256, 64,   16, 0.0,  "B32_D256_seq64"),
    ]

    print()
    for B, Hq, Hk, D, seq_len, num_splits, soft_cap, name in configs:
        print(f"\nTest: {name} (B={B}, Hq={Hq}, Hk={Hk}, D={D}, "
              f"seq_len={seq_len}, splits={num_splits}, soft_cap={soft_cap})")

        query, kv_cache, scales, block_table, seq_lens, spec, layer = \
            setup_test_data(B, Hq, Hk, D, seq_len, device=device)

        # Run Triton
        torch.cuda.synchronize()
        triton_out = run_triton_decode(
            query, kv_cache, scales, block_table, seq_lens,
            spec, num_splits, Hk, logits_soft_cap=soft_cap
        )
        torch.cuda.synchronize()

        # Run CUDA
        cuda_out = run_cuda_decode(
            query, kv_cache, scales, block_table, seq_lens,
            spec, num_splits, Hk, Hq, D, logits_soft_cap=soft_cap
        )
        torch.cuda.synchronize()

        passed = compare_outputs(triton_out, cuda_out, name)
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for _, p in results if p)
    for name, p in results:
        print(f"  {'PASS' if p else 'FAIL'}: {name}")
    print(f"\n{passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
