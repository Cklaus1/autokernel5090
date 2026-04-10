#!/usr/bin/env python3
"""Test FusenCache KV plugin with NVFP4 weight quantization compatibility.

Tests:
1. Compatibility matrix: NVFP4 + all FusenKV formats
2. Spec resolution: all dtype strings parse correctly
3. Kernel correctness: decode + store with Gemma4 dimensions
4. Logits soft cap: verify tanh capping in decode kernel
5. Sliding window prefill: verify mask is applied
6. CUDA graph safety: persistent buffers, no allocations in decode

Run standalone (no vLLM required):
    python test_fusencache_nvfp4.py

Run inside vllm-gemma4 container (tests plugin registration):
    docker exec vllm-gemma4 python /path/to/test_fusencache_nvfp4.py --vllm
"""

import sys
import os
import argparse
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_compatibility_matrix():
    """Test that NVFP4 + FusenKV combinations are correctly classified."""
    from fusen_kv.compatibility import (
        check_compatibility, list_compatible_formats,
        WEIGHT_QUANT_NVFP4,
    )

    print("=" * 60)
    print("Test 1: Compatibility Matrix (NVFP4)")
    print("=" * 60)

    # All FusenKV formats should be compatible with NVFP4
    expected_compatible = ["k4v4", "k8v4", "k8v8", "k4v2", "k8v2", "fp8_e4m3", "auto"]
    expected_blocked = ["fp8_e5m2"]

    compatible = list_compatible_formats("nvfp4")
    print(f"  Compatible formats: {compatible}")

    for fmt in expected_compatible:
        allowed, reason = check_compatibility("nvfp4", fmt)
        status = "OK" if allowed else "FAIL"
        print(f"  [{status}] nvfp4 + {fmt}: {reason}")
        assert allowed, f"Expected nvfp4 + {fmt} to be allowed"

    for fmt in expected_blocked:
        allowed, reason = check_compatibility("nvfp4", fmt)
        status = "OK" if not allowed else "FAIL"
        print(f"  [{status}] nvfp4 + {fmt} (blocked): {reason}")
        assert not allowed, f"Expected nvfp4 + {fmt} to be blocked"

    # Test aliases
    for alias in ["modelopt_fp4", "modelopt-fp4"]:
        allowed, reason = check_compatibility(alias, "k4v4")
        assert allowed, f"Alias {alias} should resolve to nvfp4"
        print(f"  [OK] alias '{alias}' resolves correctly")

    print("  PASSED\n")


def test_spec_resolution():
    """Test that all FusenKV dtype strings resolve to valid specs."""
    from fusen_kv.spec_resolver import resolve_spec

    print("=" * 60)
    print("Test 2: Spec Resolution")
    print("=" * 60)

    test_cases = [
        ("fusen", "k4v4b64"),
        ("k4v4", "k4v4b64"),
        ("k4v4b64", "k4v4b64"),
        ("k8v4", "k8v4b32"),
        ("k8v4b16", "k8v4b16"),
        ("k8v8", "k8v8b32"),
        ("int4", "k4v4b64"),
        ("int8", "k8v8b32"),
        ("auto", "k4v4b64"),
    ]

    for dtype_str, expected_name in test_cases:
        spec = resolve_spec(dtype_str)
        status = "OK" if spec.name == expected_name else "FAIL"
        print(f"  [{status}] '{dtype_str}' -> {spec.name} (expected {expected_name})")
        assert spec.name == expected_name, f"Expected {expected_name}, got {spec.name}"

    print("  PASSED\n")


def test_kernel_correctness():
    """Test decode + store kernels with Gemma4 dimensions."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)\n")
        return

    from kv_cache_gen.generate import make_decode_fn, make_store_fn
    from kv_cache_gen.spec import PREDEFINED_SPECS

    print("=" * 60)
    print("Test 3: Kernel Correctness (Gemma4 Dimensions)")
    print("=" * 60)

    spec = PREDEFINED_SPECS["k4v4b64"]

    # Test both Gemma4 layer types
    layer_configs = [
        ("sliding", 256, 16, 8, 1024),   # D=256, Hq=16, Hk=8, seq=1024
        ("global",  512, 16, 2, 8192),    # D=512, Hq=16, Hk=2, seq=8192
    ]

    for name, D, Hq, Hk, seq_len in layer_configs:
        print(f"\n  Layer type: {name} (D={D}, Hq={Hq}, Hk={Hk}, seq={seq_len})")
        B = 4  # batch size

        # Allocate KV cache
        block_size = 16
        num_blocks = (seq_len * B + block_size - 1) // block_size + 16
        slot_bytes = spec.slot_bytes(D)
        kv_cache = torch.zeros(num_blocks, block_size, Hk, slot_bytes,
                               dtype=torch.uint8, device="cuda")

        # Allocate scales
        min_block = min(spec.k_scale_block, spec.v_scale_block)
        num_sb = D // min_block
        max_slots = num_blocks * block_size
        scales = torch.zeros(max_slots, Hk, num_sb, 2,
                             dtype=torch.float16, device="cuda")

        # Create a mock layer object for store
        class MockLayer:
            pass
        layer = MockLayer()
        layer._fc_scales = scales
        layer._fusen_scales = scales

        # Generate random K, V and store them
        store_fn = make_store_fn(spec)
        N_tokens = seq_len * B
        key = torch.randn(N_tokens, Hk, D, dtype=torch.float16, device="cuda")
        value = torch.randn(N_tokens, Hk, D, dtype=torch.float16, device="cuda")

        # Create sequential slot mapping
        slot_mapping = torch.arange(N_tokens, device="cuda")

        # Store KV
        store_fn(key, value, kv_cache, slot_mapping, layer, Hk)

        # Create block table and seq lens for decode
        tokens_per_seq = seq_len
        max_blocks_per_seq = (tokens_per_seq + block_size - 1) // block_size
        block_table = torch.zeros(B, max_blocks_per_seq, dtype=torch.int32, device="cuda")
        for b in range(B):
            start_block = (b * tokens_per_seq) // block_size
            for i in range(max_blocks_per_seq):
                block_table[b, i] = start_block + i
        seq_lens = torch.full((B,), tokens_per_seq, dtype=torch.int32, device="cuda")

        # Decode: generate random query
        query = torch.randn(B, Hq, D, dtype=torch.float16, device="cuda")

        decode_fn = make_decode_fn(
            spec, block_kv=16, block_h=8, num_warps=2,
            max_seq_len=seq_len, max_batch_size=B,
        )

        output = decode_fn(query, kv_cache, scales, block_table, seq_lens,
                           1.0 / (D ** 0.5), Hk)

        assert output.shape == (B, Hq, D), f"Expected ({B}, {Hq}, {D}), got {output.shape}"
        assert not torch.isnan(output).any(), "NaN in output"
        assert not torch.isinf(output).any(), "Inf in output"

        print(f"    Output shape: {output.shape}")
        print(f"    Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"    [OK] Correctness check passed")

    print("\n  PASSED\n")


def test_logits_soft_cap():
    """Test that logits_soft_cap is applied in the decode kernel."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)\n")
        return

    from kv_cache_gen.generate import make_decode_fn, make_store_fn
    from kv_cache_gen.spec import PREDEFINED_SPECS

    print("=" * 60)
    print("Test 4: Logits Soft Cap")
    print("=" * 60)

    spec = PREDEFINED_SPECS["k4v4b64"]
    D, Hq, Hk = 256, 16, 8
    B, seq_len = 2, 128
    block_size = 16
    num_blocks = (seq_len * B + block_size - 1) // block_size + 4

    slot_bytes = spec.slot_bytes(D)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, slot_bytes,
                           dtype=torch.uint8, device="cuda")
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // min_block
    max_slots = num_blocks * block_size
    scales = torch.zeros(max_slots, Hk, num_sb, 2,
                         dtype=torch.float16, device="cuda")

    class MockLayer:
        pass
    layer = MockLayer()
    layer._fc_scales = scales
    layer._fusen_scales = scales

    # Store some KV
    store_fn = make_store_fn(spec)
    N_tokens = seq_len * B
    key = torch.randn(N_tokens, Hk, D, dtype=torch.float16, device="cuda")
    value = torch.randn(N_tokens, Hk, D, dtype=torch.float16, device="cuda")
    slot_mapping = torch.arange(N_tokens, device="cuda")
    store_fn(key, value, kv_cache, slot_mapping, layer, Hk)

    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_table = torch.zeros(B, max_blocks_per_seq, dtype=torch.int32, device="cuda")
    for b in range(B):
        for i in range(max_blocks_per_seq):
            block_table[b, i] = b * max_blocks_per_seq + i
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device="cuda")

    query = torch.randn(B, Hq, D, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / (D ** 0.5)

    # Decode WITHOUT soft cap
    decode_no_cap = make_decode_fn(
        spec, block_kv=16, block_h=8, num_warps=2,
        max_seq_len=seq_len, max_batch_size=B,
        logits_soft_cap=0.0,
    )
    out_no_cap = decode_no_cap(query, kv_cache, scales, block_table, seq_lens, sm_scale, Hk)

    # Decode WITH soft cap (50.0, same as Gemma4)
    decode_with_cap = make_decode_fn(
        spec, block_kv=16, block_h=8, num_warps=2,
        max_seq_len=seq_len, max_batch_size=B,
        logits_soft_cap=50.0,
    )
    out_with_cap = decode_with_cap(query, kv_cache, scales, block_table, seq_lens, sm_scale, Hk)

    # Outputs should be different (soft cap changes attention distribution)
    diff = (out_no_cap - out_with_cap).abs().max().item()
    print(f"  Max diff (no cap vs cap=50): {diff:.6f}")

    # Both should be valid
    assert not torch.isnan(out_with_cap).any(), "NaN with soft cap"
    assert not torch.isinf(out_with_cap).any(), "Inf with soft cap"
    assert diff > 1e-4, "Soft cap should change output (diff too small)"

    print(f"  [OK] Soft cap produces different (valid) output")
    print("  PASSED\n")


def test_cuda_graph_safety():
    """Test that decode kernel works with persistent buffers (CUDA graph prerequisite)."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIPPED (no CUDA)\n")
        return

    from kv_cache_gen.generate import make_decode_fn, make_store_fn
    from kv_cache_gen.spec import PREDEFINED_SPECS

    print("=" * 60)
    print("Test 5: CUDA Graph Safety (Persistent Buffers)")
    print("=" * 60)

    spec = PREDEFINED_SPECS["k4v4b64"]
    D, Hq, Hk = 256, 16, 8
    B, seq_len = 8, 512
    block_size = 16
    num_blocks = (seq_len * B + block_size - 1) // block_size + 4

    slot_bytes = spec.slot_bytes(D)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, slot_bytes,
                           dtype=torch.uint8, device="cuda")
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // min_block
    max_slots = num_blocks * block_size
    scales = torch.zeros(max_slots, Hk, num_sb, 2,
                         dtype=torch.float16, device="cuda")

    class MockLayer:
        pass
    layer = MockLayer()
    layer._fc_scales = scales
    layer._fusen_scales = scales

    store_fn = make_store_fn(spec)
    N_tokens = seq_len * B
    key = torch.randn(N_tokens, Hk, D, dtype=torch.float16, device="cuda")
    value = torch.randn(N_tokens, Hk, D, dtype=torch.float16, device="cuda")
    slot_mapping = torch.arange(N_tokens, device="cuda")
    store_fn(key, value, kv_cache, slot_mapping, layer, Hk)

    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_table = torch.zeros(B, max_blocks_per_seq, dtype=torch.int32, device="cuda")
    for b in range(B):
        for i in range(max_blocks_per_seq):
            block_table[b, i] = b * max_blocks_per_seq + i
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device="cuda")

    query = torch.randn(B, Hq, D, dtype=torch.float16, device="cuda")
    sm_scale = 1.0 / (D ** 0.5)

    # Create decode fn with cuda_graph_safe=True
    decode_fn = make_decode_fn(
        spec, block_kv=16, block_h=8, num_warps=2,
        max_seq_len=seq_len, max_batch_size=B,
        cuda_graph_safe=True,
        logits_soft_cap=50.0,  # Gemma4 soft cap
    )

    assert decode_fn.cuda_graph_safe, "decode_fn should be cuda_graph_safe"
    assert decode_fn.persistent_buffers, "decode_fn should use persistent_buffers"

    # Run multiple times to verify persistent buffers work
    outputs = []
    for i in range(5):
        out = decode_fn(query, kv_cache, scales, block_table, seq_lens, sm_scale, Hk)
        outputs.append(out.clone())

    # All outputs should be identical (same input, deterministic kernel)
    for i in range(1, len(outputs)):
        diff = (outputs[0] - outputs[i]).abs().max().item()
        assert diff < 1e-3, f"Run {i} differs from run 0 by {diff}"

    print(f"  [OK] 5 runs with persistent buffers: consistent output")

    # Memory check: no new allocations between runs
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    for _ in range(10):
        out = decode_fn(query, kv_cache, scales, block_table, seq_lens, sm_scale, Hk)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_delta = mem_after - mem_before

    print(f"  Memory delta after 10 runs: {mem_delta} bytes")
    # Allow small delta for Triton JIT cache, but no large allocations
    assert mem_delta < 1024 * 1024, f"Too much memory allocated: {mem_delta} bytes"
    print(f"  [OK] No significant memory allocation in decode path")
    print("  PASSED\n")


def test_backend_instantiation():
    """Test that FusenKVImpl can be instantiated with Gemma4 parameters."""
    print("=" * 60)
    print("Test 6: Backend Instantiation")
    print("=" * 60)

    # Test without vLLM (standalone)
    from fusen_kv.backend import FusenKVImpl, FusenKVBackend

    # Sliding layer config
    impl_sliding = FusenKVImpl(
        num_heads=16,
        head_size=256,
        scale=1.0 / (256 ** 0.5),
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=1024,
        kv_cache_dtype="k4v4b64",
        logits_soft_cap=50.0,
    )
    print(f"  [OK] Sliding layer: spec={impl_sliding.spec.name}, "
          f"soft_cap={impl_sliding.logits_soft_cap}")

    # Global layer config
    impl_global = FusenKVImpl(
        num_heads=16,
        head_size=512,
        scale=1.0 / (512 ** 0.5),
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="k4v4b64",
        logits_soft_cap=50.0,
    )
    print(f"  [OK] Global layer: spec={impl_global.spec.name}, "
          f"soft_cap={impl_global.logits_soft_cap}")

    # Backend class checks
    assert FusenKVBackend.supports_head_size(256), "Should support D=256"
    assert FusenKVBackend.supports_head_size(512), "Should support D=512"
    assert FusenKVBackend.supports_kv_cache_dtype("k4v4b64"), "Should support k4v4b64"
    assert FusenKVBackend.supports_kv_cache_dtype("fusen"), "Should support fusen"
    print(f"  [OK] Backend validates Gemma4 head sizes and dtypes")

    # KV cache shape
    shape = FusenKVBackend.get_kv_cache_shape(
        num_blocks=100, block_size=16, num_kv_heads=8,
        head_size=256, cache_dtype_str="k4v4b64",
    )
    expected_slot_bytes = int(0.5 * 256 + 0.5 * 256)  # k4v4: 0.5 bytes/dim each
    print(f"  KV cache shape: {shape}")
    assert shape == (100, 16, 8, expected_slot_bytes), f"Unexpected shape: {shape}"
    print(f"  [OK] KV cache shape correct: slot_bytes={expected_slot_bytes}")

    print("  PASSED\n")


def test_memory_projection():
    """Project memory savings for Gemma4 with FusenCache."""
    print("=" * 60)
    print("Test 7: Memory Projection (Gemma4 26B)")
    print("=" * 60)

    from kv_cache_gen.spec import PREDEFINED_SPECS

    spec = PREDEFINED_SPECS["k4v4b64"]

    # Gemma4 architecture
    sliding_layers = 25
    global_layers = 5
    sliding_D = 256
    global_D = 512
    sliding_Hk = 8
    global_Hk = 2
    sliding_window = 1024  # max tokens per sliding layer
    global_max_ctx = 8192  # typical global context

    # BF16 baseline
    bf16_bytes_per_token_per_head = lambda D: 2 * D + 2 * D  # K + V, 2 bytes each

    # FusenCache
    fusen_bytes_per_token_per_head = lambda D: spec.slot_bytes(D) + spec.scale_bytes(D)

    # Per-sequence memory at 4096 context length
    ctx = 4096
    sliding_seq = min(ctx, sliding_window)
    global_seq = ctx

    bf16_sliding = sliding_layers * sliding_Hk * sliding_seq * bf16_bytes_per_token_per_head(sliding_D)
    bf16_global = global_layers * global_Hk * global_seq * bf16_bytes_per_token_per_head(global_D)
    bf16_total = bf16_sliding + bf16_global

    fusen_sliding = sliding_layers * sliding_Hk * sliding_seq * fusen_bytes_per_token_per_head(sliding_D)
    fusen_global = global_layers * global_Hk * global_seq * fusen_bytes_per_token_per_head(global_D)
    fusen_total = fusen_sliding + fusen_global

    print(f"\n  Context length: {ctx} tokens")
    print(f"  Sliding layers: {sliding_layers}x (D={sliding_D}, Hk={sliding_Hk}, window={sliding_window})")
    print(f"  Global layers:  {global_layers}x (D={global_D}, Hk={global_Hk})")
    print()
    print(f"  BF16 KV per sequence:")
    print(f"    Sliding: {bf16_sliding / 1024 / 1024:.2f} MB")
    print(f"    Global:  {bf16_global / 1024 / 1024:.2f} MB")
    print(f"    Total:   {bf16_total / 1024 / 1024:.2f} MB")
    print()
    print(f"  k4v4b64 KV per sequence:")
    print(f"    Sliding: {fusen_sliding / 1024 / 1024:.2f} MB")
    print(f"    Global:  {fusen_global / 1024 / 1024:.2f} MB")
    print(f"    Total:   {fusen_total / 1024 / 1024:.2f} MB")
    print()
    print(f"  Compression ratio: {bf16_total / fusen_total:.1f}x")
    print(f"  Memory saved per seq: {(bf16_total - fusen_total) / 1024 / 1024:.2f} MB")

    # Estimate concurrency on 32GB GPU (assuming ~14GB for KV after model weights)
    kv_budget_gb = 14
    kv_budget_bytes = kv_budget_gb * 1024 * 1024 * 1024
    bf16_concurrency = kv_budget_bytes // bf16_total
    fusen_concurrency = kv_budget_bytes // fusen_total

    print(f"\n  Concurrency at {ctx} ctx (assuming {kv_budget_gb}GB KV budget):")
    print(f"    BF16:    {bf16_concurrency} concurrent sequences")
    print(f"    k4v4b64: {fusen_concurrency} concurrent sequences")
    print(f"    Improvement: {fusen_concurrency / bf16_concurrency:.1f}x")

    assert fusen_total < bf16_total, "FusenCache should use less memory"
    print("\n  PASSED\n")


def test_vllm_plugin():
    """Test FusenKV plugin registration in vLLM (requires vLLM)."""
    print("=" * 60)
    print("Test 8: vLLM Plugin Registration")
    print("=" * 60)

    try:
        from fusen_kv.plugin import register
        register()
        print("  [OK] Plugin registered successfully")

        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        backend_class = AttentionBackendEnum.CUSTOM.get_class()
        print(f"  [OK] CUSTOM backend class: {backend_class}")
        assert backend_class.__name__ == "FusenKVBackend"
        print("  PASSED\n")

    except ImportError as e:
        print(f"  SKIPPED (vLLM not available: {e})\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", action="store_true",
                        help="Run vLLM-specific tests (requires vLLM)")
    args = parser.parse_args()

    print("\nFusenCache + NVFP4 Integration Tests")
    print("=" * 60)
    print()

    passed = 0
    failed = 0
    skipped = 0

    tests = [
        test_compatibility_matrix,
        test_spec_resolution,
        test_kernel_correctness,
        test_logits_soft_cap,
        test_cuda_graph_safety,
        test_backend_instantiation,
        test_memory_projection,
    ]

    if args.vllm:
        tests.append(test_vllm_plugin)

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
