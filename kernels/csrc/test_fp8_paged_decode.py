#!/usr/bin/env python3
"""
Correctness test for FP8 Paged Decode Attention CUDA kernel.

Tests against the Triton FP8 kernel and a PyTorch BF16 reference.
"""

import sys
import os
import math
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.fp8_decode_attention import (
    fp8_decode_attention as triton_fp8_decode,
    bf16_decode_attention_ref,
    create_test_data,
)

# Build and load CUDA kernel
from kernels.csrc.build_fp8_decode import build_kernel, load_library, SO_PATH

if not load_library():
    print("Building CUDA kernel...")
    build_kernel()
else:
    print(f"Loaded pre-built kernel from {SO_PATH}")


def cuda_fp8_decode(
    q, k_cache_fp8, v_cache_fp8, block_table, seq_lens,
    k_scale, v_scale, sm_scale=None, logits_soft_cap=0.0,
    num_kv_splits=0,
):
    """Wrapper around the CUDA kernel matching Triton interface."""
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache_fp8.shape[2]
    block_size = k_cache_fp8.shape[1]
    kv_group_size = num_q_heads // num_kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    max_seq = seq_lens.max().item()
    num_pages = (max_seq + block_size - 1) // block_size

    # Auto-select splits
    if num_kv_splits == 0:
        GQA_H = min(2, kv_group_size) if kv_group_size >= 2 else 1
        num_head_groups = (num_q_heads + GQA_H - 1) // GQA_H
        base_blocks = batch * num_head_groups
        target_blocks = 170 * 4
        if base_blocks >= target_blocks:
            num_kv_splits = 1
        else:
            num_kv_splits = max(1, min(num_pages, target_blocks // max(base_blocks, 1)))
            pages_per_split = num_pages // num_kv_splits
            while pages_per_split < 2 and num_kv_splits > 1:
                num_kv_splits = num_kv_splits // 2
                pages_per_split = num_pages // num_kv_splits

    per_head_scale = 1 if k_scale.numel() > 1 else 0

    # Allocate mid_out and output
    mid_out = torch.zeros(
        (batch, num_q_heads, num_kv_splits, head_dim + 1),
        dtype=torch.float32, device=q.device)
    output = torch.empty_like(q)

    torch.ops.fp8_decode.paged_attention(
        output, q,
        k_cache_fp8, v_cache_fp8,
        k_scale, v_scale,
        block_table, seq_lens,
        mid_out,
        sm_scale, logits_soft_cap,
        num_kv_splits, head_dim, num_kv_heads,
        kv_group_size, block_size, per_head_scale,
    )

    return output


def test_basic_correctness():
    """Test CUDA kernel against BF16 reference."""
    print("\n" + "=" * 60)
    print("TEST: Basic correctness (B=4, seq=256, d=256)")
    print("=" * 60)

    data = create_test_data(
        batch=4, num_q_heads=16, num_kv_heads=8,
        head_dim=256, seq_len=256, block_size=16)

    # BF16 reference (ground truth)
    ref_out = bf16_decode_attention_ref(
        data["q"], data["k_cache_bf16"], data["v_cache_bf16"],
        data["block_table"], data["seq_lens"])

    # CUDA FP8
    cuda_out = cuda_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    # Triton FP8
    triton_out = triton_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    # Compare
    cuda_vs_ref = (cuda_out.float() - ref_out.float()).abs()
    triton_vs_ref = (triton_out.float() - ref_out.float()).abs()

    print(f"  CUDA  vs BF16 ref: max={cuda_vs_ref.max().item():.6f}, "
          f"mean={cuda_vs_ref.mean().item():.6f}")
    print(f"  Triton vs BF16 ref: max={triton_vs_ref.max().item():.6f}, "
          f"mean={triton_vs_ref.mean().item():.6f}")

    # CUDA vs Triton (should be very close since same FP8 data)
    cuda_vs_triton = (cuda_out.float() - triton_out.float()).abs()
    print(f"  CUDA  vs Triton:   max={cuda_vs_triton.max().item():.6f}, "
          f"mean={cuda_vs_triton.mean().item():.6f}")

    # FP8 quantization introduces error, so tolerance is generous
    # But CUDA vs Triton should be very tight
    atol_fp8 = 0.05  # FP8 quantization error
    atol_match = 0.01  # CUDA vs Triton match

    ok_ref = cuda_vs_ref.max().item() < atol_fp8
    ok_triton = cuda_vs_triton.max().item() < atol_match

    print(f"  CUDA vs ref   < {atol_fp8}: {'PASS' if ok_ref else 'FAIL'}")
    print(f"  CUDA vs Triton < {atol_match}: {'PASS' if ok_triton else 'FAIL'}")
    return ok_ref and ok_triton


def test_long_sequence():
    """Test with longer sequence (B=32, seq=2048)."""
    print("\n" + "=" * 60)
    print("TEST: Long sequence (B=32, seq=2048, d=256)")
    print("=" * 60)

    data = create_test_data(
        batch=32, num_q_heads=16, num_kv_heads=8,
        head_dim=256, seq_len=2048, block_size=16)

    # Triton FP8 (reference for FP8 path)
    triton_out = triton_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    # CUDA FP8
    cuda_out = cuda_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    diff = (cuda_out.float() - triton_out.float()).abs()
    print(f"  CUDA vs Triton: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    atol = 0.02
    ok = diff.max().item() < atol
    print(f"  CUDA vs Triton < {atol}: {'PASS' if ok else 'FAIL'}")
    return ok


def test_head_dim_128():
    """Test with head_dim=128."""
    print("\n" + "=" * 60)
    print("TEST: Head dim 128 (B=8, seq=512, d=128)")
    print("=" * 60)

    data = create_test_data(
        batch=8, num_q_heads=32, num_kv_heads=8,
        head_dim=128, seq_len=512, block_size=16)

    triton_out = triton_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    cuda_out = cuda_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    diff = (cuda_out.float() - triton_out.float()).abs()
    print(f"  CUDA vs Triton: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    atol = 0.02
    ok = diff.max().item() < atol
    print(f"  CUDA vs Triton < {atol}: {'PASS' if ok else 'FAIL'}")
    return ok


def test_soft_cap():
    """Test with logits soft cap (Gemma4 style)."""
    print("\n" + "=" * 60)
    print("TEST: Logits soft cap=50.0 (B=4, seq=256, d=256)")
    print("=" * 60)

    data = create_test_data(
        batch=4, num_q_heads=16, num_kv_heads=8,
        head_dim=256, seq_len=256, block_size=16)

    triton_out = triton_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"],
        logits_soft_cap=50.0)

    cuda_out = cuda_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"],
        logits_soft_cap=50.0)

    diff = (cuda_out.float() - triton_out.float()).abs()
    print(f"  CUDA vs Triton: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    atol = 0.02
    ok = diff.max().item() < atol
    print(f"  CUDA vs Triton < {atol}: {'PASS' if ok else 'FAIL'}")
    return ok


def test_gqa_group1():
    """Test with GQA group_size=1 (MHA, no grouping)."""
    print("\n" + "=" * 60)
    print("TEST: GQA group_size=1 (B=4, seq=256, d=256)")
    print("=" * 60)

    data = create_test_data(
        batch=4, num_q_heads=8, num_kv_heads=8,
        head_dim=256, seq_len=256, block_size=16)

    triton_out = triton_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    cuda_out = cuda_fp8_decode(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"])

    diff = (cuda_out.float() - triton_out.float()).abs()
    print(f"  CUDA vs Triton: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    atol = 0.02
    ok = diff.max().item() < atol
    print(f"  CUDA vs Triton < {atol}: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    torch.manual_seed(42)

    results = []
    results.append(("basic_correctness", test_basic_correctness()))
    results.append(("long_sequence", test_long_sequence()))
    results.append(("head_dim_128", test_head_dim_128()))
    results.append(("soft_cap", test_soft_cap()))
    results.append(("gqa_group1", test_gqa_group1()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    sys.exit(0 if all_pass else 1)
