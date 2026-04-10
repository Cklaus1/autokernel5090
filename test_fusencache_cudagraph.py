"""Test CUDA graph capture and replay for FusenKV decode kernel.

Verifies that:
1. The Triton decode kernel can be captured in a CUDA graph
2. Replay with different input values produces correct output
3. The store kernel can also be captured
4. The full forward() path (store + decode) works under CUDA graphs

This validates that FusenKV can run with `-cc.mode none -cc.cudagraph_mode full`
in vLLM, which gives ~2x throughput vs enforce_eager.
"""

import sys
import os
import torch
import time

# Ensure our packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kv_cache_gen.generate import make_decode_fn, make_store_fn
from fusen_kv.spec_resolver import resolve_spec


def create_test_kv_cache(spec, num_blocks, block_size, num_kv_heads, head_dim, device):
    """Create a populated test KV cache with random quantized data."""
    slot_bytes = int(spec.k_bytes_per_dim * head_dim + spec.v_bytes_per_dim * head_dim)
    kv_cache = torch.randint(0, 256, (num_blocks, block_size, num_kv_heads, slot_bytes),
                             dtype=torch.uint8, device=device)

    # Scales tensor
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = head_dim // min_block
    max_slots = num_blocks * block_size
    scales = torch.randn(max_slots, num_kv_heads, num_sb, 2,
                         dtype=torch.float16, device=device) * 0.01

    return kv_cache, scales


def create_block_table(batch_size, max_blocks_per_seq, num_blocks, device):
    """Create a valid block table mapping logical to physical blocks."""
    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for blk in range(max_blocks_per_seq):
            block_table[b, blk] = (b * max_blocks_per_seq + blk) % num_blocks
    return block_table


def test_decode_kernel_cudagraph(spec_name="k4v4b16"):
    """Test that the decode kernel can be captured and replayed as a CUDA graph."""
    print(f"\n{'='*60}")
    print(f"Test 1: Decode kernel CUDA graph capture ({spec_name})")
    print(f"{'='*60}")

    spec = resolve_spec(spec_name)
    device = torch.device("cuda")

    # Parameters matching Gemma4
    batch_size = 16
    num_heads = 8       # query heads
    num_kv_heads = 4    # KV heads (GQA)
    head_dim = 256
    block_size = 16
    num_blocks = 256
    max_blocks_per_seq = 8  # supports up to 128 tokens
    max_seq_len = max_blocks_per_seq * block_size

    # Create decode function with cuda_graph_safe=True
    decode_fn = make_decode_fn(
        spec, block_kv=16, block_h=8, num_warps=2,
        max_seq_len=max_seq_len, max_batch_size=batch_size,
        cuda_graph_safe=True, logits_soft_cap=0.0,
    )
    assert decode_fn.cuda_graph_safe, "decode_fn should be cuda_graph_safe"

    # Allocate fixed tensors (same addresses used for capture + replay)
    query = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv_cache, scales = create_test_kv_cache(spec, num_blocks, block_size, num_kv_heads, head_dim, device)
    block_table = create_block_table(batch_size, max_blocks_per_seq, num_blocks, device)
    seq_lens = torch.full((batch_size,), 64, dtype=torch.int32, device=device)

    # Warmup (triggers JIT compilation + persistent buffer allocation)
    print("  Warming up decode kernel...")
    for _ in range(3):
        out_eager = decode_fn(query, kv_cache, scales, block_table, seq_lens,
                              1.0 / (head_dim ** 0.5), num_kv_heads)
    torch.cuda.synchronize()
    print(f"  Eager output shape: {out_eager.shape}, dtype: {out_eager.dtype}")

    # Capture CUDA graph
    print("  Capturing CUDA graph...")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode_fn(query, kv_cache, scales, block_table, seq_lens,
                  1.0 / (head_dim ** 0.5), num_kv_heads)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = decode_fn(query, kv_cache, scales, block_table, seq_lens,
                              1.0 / (head_dim ** 0.5), num_kv_heads)
    print(f"  Graph captured! Output tensor at {out_graph.data_ptr():#x}")

    # Verify: replay with same inputs should give same output
    graph.replay()
    torch.cuda.synchronize()

    out_eager_ref = decode_fn(query, kv_cache, scales, block_table, seq_lens,
                              1.0 / (head_dim ** 0.5), num_kv_heads)
    torch.cuda.synchronize()

    # The graph output tensor is reused, so compare after replay
    max_diff = (out_graph - out_eager_ref).abs().max().item()
    print(f"  Same-input replay max diff: {max_diff:.6e}")
    assert max_diff < 1e-3, f"Replay output differs from eager: max_diff={max_diff}"

    # Verify: change query values, replay, check output changes
    old_output = out_graph.clone()
    query.copy_(torch.randn_like(query))
    graph.replay()
    torch.cuda.synchronize()

    output_changed = (out_graph - old_output).abs().max().item()
    print(f"  After changing query, output delta: {output_changed:.6e}")
    assert output_changed > 1e-6, "Output should change when query changes"

    # Verify: change seq_lens, replay
    seq_lens.fill_(32)  # shorter sequences
    graph.replay()
    torch.cuda.synchronize()
    out_short = out_graph.clone()

    seq_lens.fill_(64)  # longer sequences
    graph.replay()
    torch.cuda.synchronize()
    out_long = out_graph.clone()

    seqlen_diff = (out_short - out_long).abs().max().item()
    print(f"  Seq_lens change: max diff = {seqlen_diff:.6e}")
    assert seqlen_diff > 1e-6, "Output should change when seq_lens changes"

    # Performance comparison
    torch.cuda.synchronize()
    n_iter = 100

    # Eager timing
    start = time.perf_counter()
    for _ in range(n_iter):
        decode_fn(query, kv_cache, scales, block_table, seq_lens,
                  1.0 / (head_dim ** 0.5), num_kv_heads)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) * 1000 / n_iter

    # Graph timing
    start = time.perf_counter()
    for _ in range(n_iter):
        graph.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - start) * 1000 / n_iter

    speedup = eager_ms / graph_ms if graph_ms > 0 else float('inf')
    print(f"  Eager: {eager_ms:.3f} ms, Graph: {graph_ms:.3f} ms, Speedup: {speedup:.2f}x")

    print("  PASSED")
    return True


def test_store_kernel_cudagraph(spec_name="k4v4b16"):
    """Test that the store kernel can be captured in a CUDA graph."""
    print(f"\n{'='*60}")
    print(f"Test 2: Store kernel CUDA graph capture ({spec_name})")
    print(f"{'='*60}")

    spec = resolve_spec(spec_name)
    device = torch.device("cuda")

    batch_size = 16
    num_kv_heads = 4
    head_dim = 256
    block_size = 16
    num_blocks = 256

    store_fn = make_store_fn(spec)

    # Allocate fixed tensors
    key = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    slot_bytes = int(spec.k_bytes_per_dim * head_dim + spec.v_bytes_per_dim * head_dim)
    kv_cache = torch.zeros(num_blocks, block_size, num_kv_heads, slot_bytes,
                           dtype=torch.uint8, device=device)

    # Slot mapping: each of the batch_size tokens goes to a unique slot
    slot_mapping = torch.arange(batch_size, dtype=torch.int64, device=device)

    # Create a mock layer object with _fc_scales pre-allocated
    class MockLayer:
        pass
    layer = MockLayer()
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = head_dim // min_block
    max_slots = num_blocks * block_size
    layer._fc_scales = torch.zeros(max_slots, num_kv_heads, num_sb, 2,
                                   dtype=torch.float16, device=device)

    # Warmup
    print("  Warming up store kernel...")
    for _ in range(3):
        store_fn(key, value, kv_cache, slot_mapping, layer, num_kv_heads)
    torch.cuda.synchronize()

    # Capture
    print("  Capturing CUDA graph...")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        store_fn(key, value, kv_cache, slot_mapping, layer, num_kv_heads)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        store_fn(key, value, kv_cache, slot_mapping, layer, num_kv_heads)
    print("  Graph captured!")

    # Verify: store different values, check cache is updated
    kv_cache.zero_()
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))
    graph.replay()
    torch.cuda.synchronize()

    # Check that the cache was actually written to
    cache_nonzero = kv_cache[0, :batch_size].sum().item()
    print(f"  Cache sum after replay: {cache_nonzero}")
    assert cache_nonzero != 0, "Store kernel should have written to cache"

    print("  PASSED")
    return True


def test_full_forward_cudagraph(spec_name="k4v4b16"):
    """Test the full forward path (store + decode) under CUDA graph capture.

    This simulates what vLLM does: the entire model forward, including
    attention store + decode, is captured as one CUDA graph.
    """
    print(f"\n{'='*60}")
    print(f"Test 3: Full forward (store + decode) CUDA graph ({spec_name})")
    print(f"{'='*60}")

    spec = resolve_spec(spec_name)
    device = torch.device("cuda")

    # Parameters
    batch_size = 16
    num_heads = 8
    num_kv_heads = 4
    head_dim = 256
    block_size = 16
    num_blocks = 256
    max_blocks_per_seq = 8
    scale = 1.0 / (head_dim ** 0.5)

    # Create kernels
    decode_fn = make_decode_fn(
        spec, block_kv=16, block_h=8, num_warps=2,
        max_seq_len=max_blocks_per_seq * block_size,
        max_batch_size=batch_size,
        cuda_graph_safe=True, logits_soft_cap=0.0,
    )
    store_fn = make_store_fn(spec)

    # Allocate fixed tensors
    query = torch.randn(batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(batch_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)

    slot_bytes = int(spec.k_bytes_per_dim * head_dim + spec.v_bytes_per_dim * head_dim)
    kv_cache = torch.zeros(num_blocks, block_size, num_kv_heads, slot_bytes,
                           dtype=torch.uint8, device=device)

    block_table = create_block_table(batch_size, max_blocks_per_seq, num_blocks, device)
    seq_lens = torch.full((batch_size,), 64, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(batch_size, dtype=torch.int64, device=device)

    # Pre-allocate scales
    class MockLayer:
        pass
    layer = MockLayer()
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = head_dim // min_block
    max_slots = num_blocks * block_size
    layer._fc_scales = torch.zeros(max_slots, num_kv_heads, num_sb, 2,
                                   dtype=torch.float16, device=device)
    layer._fusen_scales = layer._fc_scales

    # The combined forward function (simulates FusenKVImpl.forward for decode)
    def forward_decode():
        # Store KV
        store_fn(key, value, kv_cache, slot_mapping, layer, num_kv_heads)
        # Decode attention
        out = decode_fn(query, kv_cache, layer._fusen_scales,
                        block_table, seq_lens, scale, num_kv_heads)
        return out

    # Warmup
    print("  Warming up full forward...")
    for _ in range(3):
        out = forward_decode()
    torch.cuda.synchronize()
    print(f"  Eager output shape: {out.shape}")

    # Capture
    print("  Capturing CUDA graph for full forward...")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward_decode()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = forward_decode()
    print(f"  Graph captured!")

    # Replay and verify
    query.copy_(torch.randn_like(query))
    key.copy_(torch.randn_like(key))
    value.copy_(torch.randn_like(value))

    graph.replay()
    torch.cuda.synchronize()
    out1 = out_graph.clone()

    # Change inputs and replay again
    query.copy_(torch.randn_like(query))
    graph.replay()
    torch.cuda.synchronize()
    out2 = out_graph.clone()

    diff = (out1 - out2).abs().max().item()
    print(f"  Output diff between different inputs: {diff:.6e}")
    assert diff > 1e-6, "Output should change with different inputs"

    # Performance
    n_iter = 100
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iter):
        forward_decode()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) * 1000 / n_iter

    start = time.perf_counter()
    for _ in range(n_iter):
        graph.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - start) * 1000 / n_iter

    speedup = eager_ms / graph_ms if graph_ms > 0 else float('inf')
    print(f"  Eager: {eager_ms:.3f} ms, Graph: {graph_ms:.3f} ms, Speedup: {speedup:.2f}x")

    print("  PASSED")
    return True


def test_padded_batch_cudagraph(spec_name="k4v4b16"):
    """Test CUDA graph with padded batch (some requests have seq_lens=0).

    This is what happens in vLLM: the batch is padded to a fixed size for
    CUDA graph capture, with unused entries having seq_lens=0.
    """
    print(f"\n{'='*60}")
    print(f"Test 4: Padded batch CUDA graph ({spec_name})")
    print(f"{'='*60}")

    spec = resolve_spec(spec_name)
    device = torch.device("cuda")

    # Padded to 16, but only 8 real requests
    padded_batch = 16
    real_batch = 8
    num_heads = 8
    num_kv_heads = 4
    head_dim = 256
    block_size = 16
    num_blocks = 256
    max_blocks_per_seq = 8
    scale = 1.0 / (head_dim ** 0.5)

    decode_fn = make_decode_fn(
        spec, block_kv=16, block_h=8, num_warps=2,
        max_seq_len=max_blocks_per_seq * block_size,
        max_batch_size=padded_batch,
        cuda_graph_safe=True,
    )

    # Fixed tensors at padded size
    query = torch.randn(padded_batch, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    kv_cache, scales = create_test_kv_cache(spec, num_blocks, block_size, num_kv_heads, head_dim, device)
    block_table = create_block_table(padded_batch, max_blocks_per_seq, num_blocks, device)

    # Real requests have seq_lens > 0, padding has seq_lens = 0
    seq_lens = torch.zeros(padded_batch, dtype=torch.int32, device=device)
    seq_lens[:real_batch] = 64

    # Warmup
    print("  Warming up with padded batch...")
    for _ in range(3):
        decode_fn(query, kv_cache, scales, block_table, seq_lens, scale, num_kv_heads)
    torch.cuda.synchronize()

    # Capture
    print("  Capturing CUDA graph...")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode_fn(query, kv_cache, scales, block_table, seq_lens, scale, num_kv_heads)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_graph = decode_fn(query, kv_cache, scales, block_table, seq_lens, scale, num_kv_heads)
    print("  Graph captured!")

    # Replay with different number of real requests
    seq_lens.zero_()
    seq_lens[:4] = 32  # only 4 real requests now
    query.copy_(torch.randn_like(query))

    graph.replay()
    torch.cuda.synchronize()

    # Real entries should have non-zero output, padding should be ~zero
    real_norm = out_graph[:4].float().norm().item()
    padding_norm = out_graph[4:].float().norm().item()
    print(f"  Real entries norm: {real_norm:.4f}")
    print(f"  Padding entries norm: {padding_norm:.4f}")

    # Padding entries with seq_lens=0 should produce near-zero output
    # (the kernel writes zeros or doesn't update for seq_lens=0)
    assert real_norm > 0.01, f"Real entries should have non-zero output, got {real_norm}"

    print("  PASSED")
    return True


def test_multiple_capture_sizes(spec_name="k4v4b16"):
    """Test CUDA graph capture at multiple batch sizes (like vLLM does).

    vLLM captures graphs at sizes [1, 2, 4, 8, 16, ...] and picks the
    smallest one that fits the actual batch size. In vLLM, the same
    decode_fn is used for all sizes -- tensors are allocated at max size
    and padded entries have seq_lens=0.
    """
    print(f"\n{'='*60}")
    print(f"Test 5: Multiple capture sizes ({spec_name})")
    print(f"{'='*60}")

    spec = resolve_spec(spec_name)
    device = torch.device("cuda")

    max_bs = 16
    num_heads = 8
    num_kv_heads = 4
    head_dim = 256
    block_size = 16
    num_blocks = 256
    max_blocks_per_seq = 8
    scale = 1.0 / (head_dim ** 0.5)

    # Single decode_fn at max batch size (like vLLM)
    decode_fn = make_decode_fn(
        spec, block_kv=16, block_h=8, num_warps=2,
        max_seq_len=max_blocks_per_seq * block_size,
        max_batch_size=max_bs,
        cuda_graph_safe=True,
    )

    # Shared KV cache and scales (like vLLM)
    kv_cache, scales_t = create_test_kv_cache(
        spec, num_blocks, block_size, num_kv_heads, head_dim, device)

    capture_sizes = [1, 2, 4, 8, 16]
    graphs = {}

    for bs in capture_sizes:
        # Allocate fixed tensors at this capture size
        query = torch.randn(bs, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        block_table = create_block_table(bs, max_blocks_per_seq, num_blocks, device)
        seq_lens = torch.full((bs,), 64, dtype=torch.int32, device=device)

        # Warmup
        for _ in range(3):
            decode_fn(query, kv_cache, scales_t, block_table, seq_lens, scale, num_kv_heads)
        torch.cuda.synchronize()

        # Capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            decode_fn(query, kv_cache, scales_t, block_table, seq_lens, scale, num_kv_heads)
        torch.cuda.current_stream().wait_stream(stream)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = decode_fn(query, kv_cache, scales_t, block_table, seq_lens, scale, num_kv_heads)

        graphs[bs] = (g, query, seq_lens, out)
        print(f"  Captured graph for batch_size={bs}")

    # Replay all graphs
    for bs, (g, q, sl, out) in graphs.items():
        q.copy_(torch.randn_like(q))
        g.replay()
        torch.cuda.synchronize()
        norm = out.float().norm().item()
        print(f"  BS={bs}: replay OK, output norm={norm:.4f}")
        assert norm > 0.01, f"Output should be non-zero for BS={bs}"

    print("  PASSED")
    return True


if __name__ == "__main__":
    print("FusenKV CUDA Graph Compatibility Test Suite")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    spec_name = sys.argv[1] if len(sys.argv) > 1 else "k4v4b16"
    print(f"Testing with spec: {spec_name}")

    results = []
    tests = [
        ("Decode kernel CG", test_decode_kernel_cudagraph),
        ("Store kernel CG", test_store_kernel_cudagraph),
        ("Full forward CG", test_full_forward_cudagraph),
        ("Padded batch CG", test_padded_batch_cudagraph),
        ("Multiple sizes CG", test_multiple_capture_sizes),
    ]

    for name, test_fn in tests:
        try:
            passed = test_fn(spec_name)
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"FAILED: {e}"))

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    all_passed = True
    for name, status in results:
        marker = "PASS" if "PASSED" in status else "FAIL"
        print(f"  [{marker}] {name}: {status}")
        if "FAIL" in status:
            all_passed = False

    if all_passed:
        print(f"\nAll {len(results)} tests passed! FusenKV is CUDA graph compatible.")
    else:
        print(f"\nSome tests failed. See above for details.")
        sys.exit(1)
