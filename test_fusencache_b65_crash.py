#!/usr/bin/env python3
"""Reproduce FusenCache CUDA assertion crash at batch size 65.

Tests the decode kernel directly (without vLLM) to isolate whether the
crash is in the Triton kernel, the persistent buffer logic, or the
vLLM integration layer (block table / slot mapping).

Root cause candidates:
  1. Block table overflow
  2. Slot mapping out of bounds
  3. Triton kernel grid calculation (cdiv issue with BLOCK_H)
  4. CUDA graph replay with wrong tensor sizes
  5. Persistent buffer keyed by exact B — mismatch with padded B
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton

from kv_cache_gen.spec import PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn, make_store_fn


def make_test_cache(spec, num_blocks, block_size, num_kv_heads, head_size, device):
    """Create a realistic KV cache with stored data."""
    slot_bytes = int(spec.k_bytes_per_dim * head_size + spec.v_bytes_per_dim * head_size)
    kv_cache = torch.zeros(num_blocks, block_size, num_kv_heads, slot_bytes,
                           dtype=torch.uint8, device=device)

    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = head_size // min_block
    max_slots = num_blocks * block_size
    scales = torch.ones(max_slots, num_kv_heads, num_sb, 2,
                        dtype=torch.float16, device=device) * 0.1

    return kv_cache, scales


def store_kv_data(spec, kv_cache, scales, block_table, seq_lens,
                  num_kv_heads, head_size, device):
    """Store random KV data into the cache for all sequences."""
    store_fn = make_store_fn(spec)
    block_size = kv_cache.shape[1]

    B = block_table.shape[0]

    class FakeLayer:
        pass
    layer = FakeLayer()
    layer._fc_scales = scales

    for b in range(B):
        sl = seq_lens[b].item()
        if sl <= 0:
            continue
        key = torch.randn(sl, num_kv_heads, head_size, device=device, dtype=torch.float16)
        value = torch.randn(sl, num_kv_heads, head_size, device=device, dtype=torch.float16)

        # Build slot mapping from block table
        slots = []
        for pos in range(sl):
            blk_idx = pos // block_size
            blk_off = pos % block_size
            phys_block = block_table[b, blk_idx].item()
            slots.append(phys_block * block_size + blk_off)
        slot_mapping = torch.tensor(slots, dtype=torch.int64, device=device)

        store_fn(key, value, kv_cache, slot_mapping, layer, num_kv_heads)


def test_decode_basic(B, seq_len, num_heads, num_kv_heads, head_size, spec_name,
                      cuda_graph_safe=False, label=""):
    """Test decode kernel at given batch size."""
    device = "cuda"
    spec = PREDEFINED_SPECS[spec_name]
    block_size = 16
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = B * blocks_per_seq + 64  # extra headroom

    print(f"\n{'='*60}")
    print(f"Test: {label or f'B={B}'}")
    print(f"  B={B}, seq_len={seq_len}, heads={num_heads}/{num_kv_heads}, "
          f"D={head_size}, spec={spec_name}")
    print(f"  blocks_per_seq={blocks_per_seq}, total_blocks={num_blocks}")
    print(f"  cuda_graph_safe={cuda_graph_safe}")

    kv_cache, scales = make_test_cache(spec, num_blocks, block_size,
                                        num_kv_heads, head_size, device)

    # Build block table: sequential allocation
    block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(B):
        for blk in range(blocks_per_seq):
            block_table[b, blk] = b * blocks_per_seq + blk

    seq_lens_t = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    # Store KV data (just first few sequences to keep test fast)
    store_count = min(B, 4)
    store_kv_data(spec, kv_cache, scales, block_table[:store_count],
                  seq_lens_t[:store_count], num_kv_heads, head_size, device)

    # Build decode function
    max_seq = seq_len
    decode_fn = make_decode_fn(
        spec,
        block_kv=16, block_h=8, num_warps=2,
        max_seq_len=max_seq,
        max_batch_size=256,
        cuda_graph_safe=cuda_graph_safe,
    )

    query = torch.randn(B, num_heads, head_size, device=device, dtype=torch.float16)

    # Synchronous test with error checking
    torch.cuda.synchronize()
    try:
        output = decode_fn(query, kv_cache, scales, block_table, seq_lens_t,
                          1.0 / (head_size ** 0.5), num_kv_heads)
        torch.cuda.synchronize()

        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"  Has NaN: {output.isnan().any().item()}")
        print(f"  Has Inf: {output.isinf().any().item()}")
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  CRASH: {e}")
        return False


def test_padded_batch(actual_B, padded_B, seq_len, num_heads, num_kv_heads,
                      head_size, spec_name, label=""):
    """Test decode with padded batch (simulating CUDA graph replay).

    The query tensor has padded_B rows, but only actual_B have valid seq_lens.
    Remaining rows have seq_len=0 (what vLLM does for CUDA graph padding).
    """
    device = "cuda"
    spec = PREDEFINED_SPECS[spec_name]
    block_size = 16
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = padded_B * blocks_per_seq + 64

    print(f"\n{'='*60}")
    print(f"Test: {label or f'actual_B={actual_B}, padded_B={padded_B}'}")
    print(f"  actual_B={actual_B}, padded_B={padded_B}, seq_len={seq_len}")
    print(f"  heads={num_heads}/{num_kv_heads}, D={head_size}, spec={spec_name}")

    kv_cache, scales = make_test_cache(spec, num_blocks, block_size,
                                        num_kv_heads, head_size, device)

    # Block table: padded_B rows, but only actual_B are valid
    block_table = torch.zeros(padded_B, blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(actual_B):
        for blk in range(blocks_per_seq):
            block_table[b, blk] = b * blocks_per_seq + blk
    # Padded rows: block_table stays 0 (block 0 is valid, just unused)

    seq_lens_t = torch.zeros(padded_B, dtype=torch.int32, device=device)
    seq_lens_t[:actual_B] = seq_len
    # Padded entries have seq_len=0

    # Store some KV data
    store_count = min(actual_B, 4)
    store_kv_data(spec, kv_cache, scales, block_table[:store_count],
                  seq_lens_t[:store_count], num_kv_heads, head_size, device)

    decode_fn = make_decode_fn(
        spec,
        block_kv=16, block_h=8, num_warps=2,
        max_seq_len=seq_len,
        max_batch_size=256,
        cuda_graph_safe=True,
    )

    query = torch.randn(padded_B, num_heads, head_size, device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    try:
        output = decode_fn(query, kv_cache, scales, block_table, seq_lens_t,
                          1.0 / (head_size ** 0.5), num_kv_heads)
        torch.cuda.synchronize()

        # Check valid outputs
        valid_out = output[:actual_B]
        padded_out = output[actual_B:]

        print(f"  Output shape: {output.shape}")
        print(f"  Valid range: [{valid_out.min().item():.4f}, {valid_out.max().item():.4f}]")
        print(f"  Valid NaN: {valid_out.isnan().any().item()}")
        print(f"  Padded range: [{padded_out.min().item():.4f}, {padded_out.max().item():.4f}]")
        print(f"  Padded NaN: {padded_out.isnan().any().item()}")

        # Padded outputs should be all zeros (seq_len=0)
        padded_nonzero = (padded_out != 0).any().item()
        if padded_nonzero:
            print(f"  WARNING: Padded entries are non-zero!")

        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  CRASH: {e}")
        return False


def test_persistent_buffer_reuse(spec_name, head_size, num_heads, num_kv_heads):
    """Test that persistent buffers work when B changes between calls.

    Under CUDA graphs, each captured batch size gets its own buffer.
    But if the closure incorrectly shares state, this could cause issues.
    """
    device = "cuda"
    spec = PREDEFINED_SPECS[spec_name]
    block_size = 16
    seq_len = 512
    blocks_per_seq = seq_len // block_size
    num_blocks = 256 * blocks_per_seq + 64

    print(f"\n{'='*60}")
    print(f"Test: Persistent buffer reuse across batch sizes")

    kv_cache, scales = make_test_cache(spec, num_blocks, block_size,
                                        num_kv_heads, head_size, device)

    decode_fn = make_decode_fn(
        spec,
        block_kv=16, block_h=8, num_warps=2,
        max_seq_len=seq_len,
        max_batch_size=256,
        cuda_graph_safe=True,
    )

    # Test sequence: small, medium, large, then back to medium
    batch_sizes = [1, 8, 32, 64, 65, 72, 128, 65, 72, 65]

    for B in batch_sizes:
        block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(B):
            for blk in range(blocks_per_seq):
                block_table[b, blk] = (b * blocks_per_seq + blk) % num_blocks

        seq_lens_t = torch.full((B,), seq_len, dtype=torch.int32, device=device)
        query = torch.randn(B, num_heads, head_size, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        try:
            output = decode_fn(query, kv_cache, scales, block_table, seq_lens_t,
                              1.0 / (head_size ** 0.5), num_kv_heads)
            torch.cuda.synchronize()
            print(f"  B={B:>4}: shape={output.shape}, nan={output.isnan().any().item()}")
        except Exception as e:
            print(f"  B={B:>4}: CRASH: {e}")
            return False

    print(f"  PASS")
    return True


def test_bounds_checking(spec_name, head_size, num_heads, num_kv_heads):
    """Test with deliberately tight block allocation to find OOB."""
    device = "cuda"
    spec = PREDEFINED_SPECS[spec_name]
    block_size = 16
    seq_len = 4096
    blocks_per_seq = seq_len // block_size  # 256
    B = 65

    # TIGHT allocation: exactly enough blocks, no headroom
    num_blocks = B * blocks_per_seq

    print(f"\n{'='*60}")
    print(f"Test: Bounds checking with tight allocation")
    print(f"  B={B}, seq_len={seq_len}, blocks_per_seq={blocks_per_seq}")
    print(f"  num_blocks={num_blocks} (exactly B*blocks_per_seq)")

    kv_cache, scales = make_test_cache(spec, num_blocks, block_size,
                                        num_kv_heads, head_size, device)

    # Block table: sequential, uses ALL blocks
    block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(B):
        for blk in range(blocks_per_seq):
            block_table[b, blk] = b * blocks_per_seq + blk

    # Verify: max block index should be num_blocks - 1
    max_block = block_table.max().item()
    print(f"  Max block index: {max_block} (should be < {num_blocks})")
    assert max_block < num_blocks, f"Block table OOB: {max_block} >= {num_blocks}"

    # Verify: max slot index
    max_slot = max_block * block_size + (block_size - 1)
    max_scales_slot = scales.shape[0] - 1
    print(f"  Max slot index: {max_slot} (scales capacity: {max_scales_slot})")
    assert max_slot <= max_scales_slot, f"Slot OOB: {max_slot} > {max_scales_slot}"

    seq_lens_t = torch.full((B,), seq_len, dtype=torch.int32, device=device)

    decode_fn = make_decode_fn(
        spec,
        block_kv=16, block_h=8, num_warps=2,
        max_seq_len=seq_len,
        max_batch_size=256,
        cuda_graph_safe=True,
    )

    query = torch.randn(B, num_heads, head_size, device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    try:
        output = decode_fn(query, kv_cache, scales, block_table, seq_lens_t,
                          1.0 / (head_size ** 0.5), num_kv_heads)
        torch.cuda.synchronize()
        print(f"  Output: shape={output.shape}, nan={output.isnan().any().item()}")
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  CRASH: {e}")
        return False


def test_grid_calculation(spec_name, head_size, num_heads, num_kv_heads):
    """Verify grid calculation matches mid_out buffer dimensions.

    The bug might be in cdiv(Hq, VALID_BLOCK_H) vs the mid_out allocation.
    mid_out is [B, Hq, NUM_KV_SPLITS, D+1] but the grid uses num_head_groups.
    If num_head_groups * VALID_BLOCK_H > Hq, the kernel writes past mid_out bounds.
    """
    device = "cuda"
    spec = PREDEFINED_SPECS[spec_name]

    Hq = num_heads
    Hk = num_kv_heads
    kv_group_size = Hq // Hk
    BLOCK_H = min(8, kv_group_size)
    BLOCK_H = max(1, BLOCK_H)
    VALID_BLOCK_H = BLOCK_H if kv_group_size > BLOCK_H else kv_group_size
    num_head_groups = triton.cdiv(Hq, VALID_BLOCK_H)

    # The maximum head index written by the kernel
    max_head_idx = (num_head_groups - 1) * VALID_BLOCK_H + BLOCK_H - 1

    print(f"\n{'='*60}")
    print(f"Test: Grid calculation analysis")
    print(f"  Hq={Hq}, Hk={Hk}, kv_group_size={kv_group_size}")
    print(f"  BLOCK_H={BLOCK_H}, VALID_BLOCK_H={VALID_BLOCK_H}")
    print(f"  num_head_groups={num_head_groups}")
    print(f"  Max head index in kernel: {max_head_idx}")
    print(f"  mid_out head dim: {Hq}")

    if max_head_idx >= Hq:
        print(f"  BUG: max_head_idx ({max_head_idx}) >= Hq ({Hq})!")
        print(f"  The kernel can write out-of-bounds into mid_out!")
        print(f"  The mask guards stores, but addresses are still computed.")
        # Check if the mask properly guards
        # head_indices = cur_head_id * VALID_BLOCK_H + arange(0, BLOCK_H)
        # head_mask = (head_indices < Q_HEAD_NUM) & mask_h
        # So writes at index >= Hq are masked. But mid_out is [B, Hq, ...],
        # so stride_mid_h = num_kv_splits * (D+1). A write to head index Hq
        # would be at offset Hq * stride_mid_h, which is past the end of
        # the head dimension but might alias into the next batch entry.
        print(f"  However, stores are masked by head_indices < Q_HEAD_NUM.")
        print(f"  The mask should prevent OOB writes. SAFE but wasteful.")
    else:
        print(f"  Grid is within bounds.")

    print(f"  PASS (analysis only)")
    return True


def test_cuda_graph_capture_and_replay(spec_name, head_size, num_heads, num_kv_heads):
    """Simulate CUDA graph capture at B=72, then replay with actual B=65 (padded to 72).

    This is the exact scenario that crashes in production.
    """
    device = "cuda"
    spec = PREDEFINED_SPECS[spec_name]
    block_size = 16
    seq_len = 256  # shorter for test speed
    blocks_per_seq = seq_len // block_size
    padded_B = 72  # vLLM capture size
    actual_B = 65
    num_blocks = padded_B * blocks_per_seq + 64

    print(f"\n{'='*60}")
    print(f"Test: CUDA graph capture at B={padded_B}, replay with actual B={actual_B}")

    kv_cache, scales = make_test_cache(spec, num_blocks, block_size,
                                        num_kv_heads, head_size, device)

    decode_fn = make_decode_fn(
        spec,
        block_kv=16, block_h=8, num_warps=2,
        max_seq_len=seq_len,
        max_batch_size=256,
        cuda_graph_safe=True,
    )

    # Pre-allocate tensors at padded size (as vLLM does)
    query_buf = torch.randn(padded_B, num_heads, head_size, device=device, dtype=torch.float16)
    block_table_buf = torch.zeros(padded_B, blocks_per_seq, dtype=torch.int32, device=device)
    seq_lens_buf = torch.zeros(padded_B, dtype=torch.int32, device=device)
    output_buf = torch.empty(padded_B, num_heads, head_size, device=device, dtype=torch.float16)

    # Fill block table for all padded_B entries (capture uses these)
    for b in range(padded_B):
        for blk in range(blocks_per_seq):
            block_table_buf[b, blk] = (b * blocks_per_seq + blk) % num_blocks

    # --- Phase 1: Warm up (pre-compile Triton kernels) ---
    seq_lens_buf.fill_(1)  # minimal seq_len for fast capture
    print(f"  Warmup...")
    _ = decode_fn(query_buf, kv_cache, scales, block_table_buf, seq_lens_buf,
                  1.0 / (head_size ** 0.5), num_kv_heads)
    torch.cuda.synchronize()

    # --- Phase 2: Capture CUDA graph ---
    print(f"  Capturing CUDA graph at B={padded_B}...")
    graph = torch.cuda.CUDAGraph()

    # Set capture-time seq_lens (minimal, as vLLM does)
    seq_lens_buf.fill_(1)

    with torch.cuda.graph(graph):
        output_buf = decode_fn(query_buf, kv_cache, scales, block_table_buf, seq_lens_buf,
                              1.0 / (head_size ** 0.5), num_kv_heads)
    torch.cuda.synchronize()
    print(f"  Graph captured successfully")

    # --- Phase 3: Replay with actual B=65 (padded to 72) ---
    print(f"  Replaying with actual_B={actual_B} (padded to {padded_B})...")

    # Fill actual data: 65 real requests, 7 padding entries
    seq_lens_buf[:actual_B] = seq_len
    seq_lens_buf[actual_B:] = 0  # padding

    # Re-fill query with new data
    query_buf.normal_()

    try:
        graph.replay()
        torch.cuda.synchronize()

        valid_out = output_buf[:actual_B]
        padded_out = output_buf[actual_B:]

        print(f"  Output shape: {output_buf.shape}")
        print(f"  Valid NaN: {valid_out.isnan().any().item()}")
        print(f"  Padded NaN: {padded_out.isnan().any().item()}")
        print(f"  PASS")
        return True
    except Exception as e:
        print(f"  CRASH during replay: {e}")
        return False


def test_split_kv_boundary(spec_name, head_size, num_heads, num_kv_heads):
    """Test split-KV at boundary where seq_len is not evenly divisible.

    With NUM_KV_SPLITS=32 and seq_len=4096: kv_len = 128, evenly divisible.
    With NUM_KV_SPLITS=16 and seq_len=4096: kv_len = 256, evenly divisible.

    But what about seq_len=100? kv_len = cdiv(100, 32) = 4.
    split 0: [0, 4), split 1: [4, 8), ..., split 24: [96, 100), split 25+: empty.

    The kernel iterates by BLOCK_KV=16, so start_n=0 loads positions 0..15.
    But split_end=4, so only 0..3 are valid. The rest are masked.
    """
    device = "cuda"
    spec = PREDEFINED_SPECS[spec_name]
    block_size = 16
    B = 65

    # Test problematic seq_lens
    test_seq_lens = [1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256,
                     512, 1024, 2048, 4096]

    print(f"\n{'='*60}")
    print(f"Test: Split-KV boundary conditions at B={B}")

    all_pass = True
    for seq_len in test_seq_lens:
        blocks_per_seq = (seq_len + block_size - 1) // block_size
        num_blocks = B * blocks_per_seq + 64

        kv_cache, scales = make_test_cache(spec, num_blocks, block_size,
                                            num_kv_heads, head_size, device)

        block_table = torch.zeros(B, blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(B):
            for blk in range(blocks_per_seq):
                block_table[b, blk] = (b * blocks_per_seq + blk) % num_blocks

        seq_lens_t = torch.full((B,), seq_len, dtype=torch.int32, device=device)

        decode_fn = make_decode_fn(
            spec,
            block_kv=16, block_h=8, num_warps=2,
            max_seq_len=max(seq_len, 512),
            max_batch_size=256,
            cuda_graph_safe=True,
        )

        query = torch.randn(B, num_heads, head_size, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        try:
            output = decode_fn(query, kv_cache, scales, block_table, seq_lens_t,
                              1.0 / (head_size ** 0.5), num_kv_heads)
            torch.cuda.synchronize()
            nan = output.isnan().any().item()
            status = "OK" if not nan else "NaN!"
            print(f"  seq_len={seq_len:>5}: {status}")
            if nan:
                all_pass = False
        except Exception as e:
            print(f"  seq_len={seq_len:>5}: CRASH: {e}")
            all_pass = False

    print(f"  {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_gemma4_config():
    """Test with Gemma 4 26B-A4B architecture.

    Gemma4 has:
    - Sliding window layers: head_size=256, num_heads=16, num_kv_heads=8
    - Global layers: head_size=512, num_heads=16, num_kv_heads=2
    """
    print(f"\n{'#'*60}")
    print(f"# Gemma 4 26B-A4B configuration tests")
    print(f"{'#'*60}")

    results = []

    # Sliding window layers
    for B in [1, 32, 64, 65, 72, 128]:
        ok = test_decode_basic(
            B=B, seq_len=4096, num_heads=16, num_kv_heads=8,
            head_size=256, spec_name="k4v4b64",
            cuda_graph_safe=True,
            label=f"Sliding window B={B}",
        )
        results.append(("sliding", B, ok))

    # Global layers
    for B in [1, 32, 64, 65, 72, 128]:
        ok = test_decode_basic(
            B=B, seq_len=4096, num_heads=16, num_kv_heads=2,
            head_size=512, spec_name="k4v4b64",
            cuda_graph_safe=True,
            label=f"Global layer B={B}",
        )
        results.append(("global", B, ok))

    return results


def main():
    print("FusenCache B=65 Crash Reproduction Test")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    all_results = []

    # === Test 1: Basic decode at various batch sizes ===
    for B in [1, 8, 32, 64, 65, 72, 128, 256]:
        ok = test_decode_basic(
            B=B, seq_len=512, num_heads=16, num_kv_heads=8,
            head_size=256, spec_name="k4v4b64",
            cuda_graph_safe=True,
            label=f"Basic B={B}",
        )
        all_results.append(("basic", B, ok))

    # === Test 2: Padded batch (CUDA graph simulation) ===
    ok = test_padded_batch(
        actual_B=65, padded_B=72, seq_len=512,
        num_heads=16, num_kv_heads=8, head_size=256,
        spec_name="k4v4b64",
        label="Padded B=65->72",
    )
    all_results.append(("padded", 65, ok))

    # === Test 3: Grid calculation analysis ===
    for num_heads, num_kv_heads in [(16, 8), (16, 2), (8, 1), (32, 8)]:
        ok = test_grid_calculation("k4v4b64", 256, num_heads, num_kv_heads)
        all_results.append(("grid", num_heads, ok))

    # === Test 4: Persistent buffer reuse ===
    ok = test_persistent_buffer_reuse("k4v4b64", 256, 16, 8)
    all_results.append(("persistent", 0, ok))

    # === Test 5: Bounds checking with tight allocation ===
    ok = test_bounds_checking("k4v4b64", 256, 16, 8)
    all_results.append(("bounds", 65, ok))

    # === Test 6: CUDA graph capture + replay ===
    ok = test_cuda_graph_capture_and_replay("k4v4b64", 256, 16, 8)
    all_results.append(("cudagraph", 65, ok))

    # === Test 7: Split-KV boundary conditions ===
    ok = test_split_kv_boundary("k4v4b64", 256, 16, 8)
    all_results.append(("split_kv", 65, ok))

    # === Test 8: Gemma 4 config ===
    gemma_results = test_gemma4_config()
    all_results.extend(gemma_results)

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    failures = [r for r in all_results if not r[2]]
    for name, param, ok in all_results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:>12} {param:>5}: {status}")

    if failures:
        print(f"\n{len(failures)} FAILURES detected!")
        return 1
    else:
        print(f"\nAll tests passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
