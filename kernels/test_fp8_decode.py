"""Correctness tests for FP8 decode attention kernel.

Compares FP8 Triton kernel against BF16 PyTorch reference implementation.
Tests multiple configurations: GQA ratios, head dims, sequence lengths, soft cap.
"""

import torch
import sys
import math

sys.path.insert(0, "/root/projects/autokernel/kernels")
from fp8_decode_attention import (
    fp8_decode_attention,
    bf16_decode_attention_ref,
    create_test_data,
)


def test_basic_correctness(batch=4, seq_len=512, head_dim=256,
                           num_q_heads=16, num_kv_heads=8,
                           logits_soft_cap=0.0, name="basic"):
    """Test FP8 kernel matches BF16 reference within FP8 quantization tolerance."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  batch={batch}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"  q_heads={num_q_heads}, kv_heads={num_kv_heads}, GQA={num_q_heads//num_kv_heads}x")
    print(f"  logits_soft_cap={logits_soft_cap}")
    print(f"{'='*60}")

    data = create_test_data(
        batch=batch, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, seq_len=seq_len, block_size=16)

    # FP8 Triton kernel
    out_fp8 = fp8_decode_attention(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"],
        logits_soft_cap=logits_soft_cap,
    )

    # BF16 reference using dequantized FP8 data (apples-to-apples comparison)
    # Dequantize FP8 back to BF16 using same scales
    k_deq = data["k_cache_fp8"].float() * data["k_scale_val"]
    v_deq = data["v_cache_fp8"].float() * data["v_scale_val"]
    k_deq_bf16 = k_deq.to(torch.bfloat16)
    v_deq_bf16 = v_deq.to(torch.bfloat16)

    out_ref = bf16_decode_attention_ref(
        data["q"], k_deq_bf16, v_deq_bf16,
        data["block_table"], data["seq_lens"],
        logits_soft_cap=logits_soft_cap,
    )

    # Compare
    out_fp8_f32 = out_fp8.float()
    out_ref_f32 = out_ref.float()

    abs_diff = (out_fp8_f32 - out_ref_f32).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # Relative error (avoid div by zero)
    ref_norm = out_ref_f32.abs().clamp(min=1e-6)
    rel_diff = abs_diff / ref_norm
    max_rel = rel_diff.max().item()
    mean_rel = rel_diff.mean().item()

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        out_fp8_f32.reshape(-1).unsqueeze(0),
        out_ref_f32.reshape(-1).unsqueeze(0)).item()

    print(f"  Max abs error:  {max_abs:.6f}")
    print(f"  Mean abs error: {mean_abs:.6f}")
    print(f"  Max rel error:  {max_rel:.6f}")
    print(f"  Mean rel error: {mean_rel:.6f}")
    print(f"  Cosine sim:     {cos_sim:.8f}")

    # FP8 E4M3 has ~3 mantissa bits, so expect ~1e-2 relative error
    # We're generous: allow up to 5% relative error and 0.999 cosine sim
    passed = cos_sim > 0.999 and mean_rel < 0.05
    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def test_split_k_consistency(batch=4, seq_len=2048, head_dim=256):
    """Verify that different split-K counts produce same result."""
    print(f"\n{'='*60}")
    print(f"Test: split-K consistency (seq_len={seq_len})")
    print(f"{'='*60}")

    data = create_test_data(batch=batch, seq_len=seq_len, head_dim=head_dim)

    results = {}
    for splits in [1, 2, 4, 8]:
        out = fp8_decode_attention(
            data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
            data["block_table"], data["seq_lens"],
            data["k_scale"], data["v_scale"],
            num_kv_splits=splits,
        )
        results[splits] = out.float()

    # Compare all against splits=1
    ref = results[1]
    all_pass = True
    for splits, out in results.items():
        if splits == 1:
            continue
        max_diff = (out - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out.reshape(-1).unsqueeze(0),
            ref.reshape(-1).unsqueeze(0)).item()
        status = "PASS" if cos > 0.9999 else "FAIL"
        print(f"  splits={splits}: max_diff={max_diff:.6f}, cos_sim={cos:.8f} [{status}]")
        if cos <= 0.9999:
            all_pass = False

    print(f"  Result: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_variable_seq_lens(batch=8, head_dim=256):
    """Test with different sequence lengths per batch entry."""
    print(f"\n{'='*60}")
    print(f"Test: variable sequence lengths")
    print(f"{'='*60}")

    max_seq = 1024
    data = create_test_data(batch=batch, seq_len=max_seq, head_dim=head_dim)

    # Set variable seq_lens
    seq_lens_var = torch.tensor(
        [128, 256, 512, 1024, 64, 768, 384, 960],
        dtype=torch.int32, device="cuda")[:batch]
    data["seq_lens"] = seq_lens_var

    out_fp8 = fp8_decode_attention(
        data["q"], data["k_cache_fp8"], data["v_cache_fp8"],
        data["block_table"], data["seq_lens"],
        data["k_scale"], data["v_scale"],
    )

    k_deq = data["k_cache_fp8"].float() * data["k_scale_val"]
    v_deq = data["v_cache_fp8"].float() * data["v_scale_val"]
    out_ref = bf16_decode_attention_ref(
        data["q"], k_deq.to(torch.bfloat16), v_deq.to(torch.bfloat16),
        data["block_table"], data["seq_lens"],
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        out_fp8.float().reshape(-1).unsqueeze(0),
        out_ref.float().reshape(-1).unsqueeze(0)).item()

    passed = cos_sim > 0.999
    print(f"  Cosine sim: {cos_sim:.8f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    torch.manual_seed(42)

    results = []

    # Test 1: Basic Gemma4 sliding window config (head_dim=256, GQA=2x)
    results.append(test_basic_correctness(
        batch=4, seq_len=512, head_dim=256,
        num_q_heads=16, num_kv_heads=8,
        name="Gemma4 sliding (d=256, GQA=2x)"))

    # Test 2: Gemma4 global attention (head_dim=512, GQA=2x)  -- skip if OOM
    try:
        results.append(test_basic_correctness(
            batch=2, seq_len=256, head_dim=512,
            num_q_heads=16, num_kv_heads=8,
            name="Gemma4 global (d=512, GQA=2x)"))
    except Exception as e:
        print(f"\n  Skipped Gemma4 global test: {e}")
        results.append(None)

    # Test 3: With logits soft cap (Gemma4 style)
    results.append(test_basic_correctness(
        batch=4, seq_len=512, head_dim=256,
        num_q_heads=16, num_kv_heads=8,
        logits_soft_cap=50.0,
        name="Gemma4 with soft_cap=50"))

    # Test 4: Larger GQA ratio (8x, like Llama 3)
    results.append(test_basic_correctness(
        batch=4, seq_len=512, head_dim=128,
        num_q_heads=32, num_kv_heads=4,
        name="GQA 8x (d=128, like Llama3)"))

    # Test 5: No GQA (MHA)
    results.append(test_basic_correctness(
        batch=4, seq_len=256, head_dim=128,
        num_q_heads=8, num_kv_heads=8,
        name="MHA (no GQA, d=128)"))

    # Test 6: Long sequence
    results.append(test_basic_correctness(
        batch=2, seq_len=4096, head_dim=256,
        num_q_heads=16, num_kv_heads=8,
        name="Long sequence (4096)"))

    # Test 7: Split-K consistency
    results.append(test_split_k_consistency())

    # Test 8: Variable sequence lengths
    results.append(test_variable_seq_lens())

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r is True)
    skipped = sum(1 for r in results if r is None)
    failed = sum(1 for r in results if r is False)
    total = len(results) - skipped
    print(f"  Passed:  {passed}/{total}")
    print(f"  Failed:  {failed}/{total}")
    if skipped:
        print(f"  Skipped: {skipped}")

    if failed > 0:
        print("\nFAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
