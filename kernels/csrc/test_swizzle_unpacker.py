#!/usr/bin/env python3
"""CUTLASS 128×4 blockscale swizzle unpacker — CPU reference for mega-graph v4b.

Implements the CUTLASS NVFP4 scale swizzle in pure Python, matching the
formula in fused_shuffle_quant.cu:72 and rms_norm_dynamic_fp4_quant.cu::swizzled_sf_offset.

Formula:
    mTileIdx  = row_in_buf >> 7           # row / 128
    outerMIdx = row_in_buf & 31           # row % 32
    innerMIdx = (row_in_buf >> 5) & 3     # (row / 32) % 4
    kTileIdx  = block_idx >> 2            # block_idx / 4
    innerKIdx = block_idx & 3             # block_idx % 4
    byte_off  = ((mTileIdx * numKTiles + kTileIdx) << 9)
                | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx

Layout:
    Logical: [M, numKBlocks]   where numKBlocks = K / 16
    Physical 5D: [numMTiles, numKTiles, 32, 4, 4]
    numMTiles = ceil(M / 128)
    numKTiles = ceil(numKBlocks / 4) = ceil(K / 64)

Usage (CPU only):
    python test_swizzle_unpacker.py
    python test_swizzle_unpacker.py --M 256 --K 4096  # custom dims
"""

import argparse
import torch


# ---------------------------------------------------------------------------
# Swizzle forward: (row, block_idx) -> flat_byte_offset
# ---------------------------------------------------------------------------

def cutlass_sf_offset(row_in_buf: int, block_idx: int, numKTiles: int) -> int:
    """Compute CUTLASS 128×4 blockscale swizzled byte offset.

    Matches fused_shuffle_quant.cu::compute_cutlass_sf_offset and
    rms_norm_dynamic_fp4_quant.cu::swizzled_sf_offset.

    Args:
        row_in_buf: row index in the scale buffer (0-based).
        block_idx:  K-block index (0-based). Each block covers 16 FP4 elements.
        numKTiles:  ceil(K / 64) — number of 64-element K tiles.

    Returns:
        Flat byte offset into the swizzled scale buffer.
    """
    mTileIdx  = row_in_buf >> 7           # row / 128
    outerMIdx = row_in_buf & 31           # row % 32
    innerMIdx = (row_in_buf >> 5) & 3     # (row / 32) % 4
    kTileIdx  = block_idx >> 2            # block_idx / 4
    innerKIdx = block_idx & 3            # block_idx % 4
    return (
        ((mTileIdx * numKTiles + kTileIdx) << 9)
        | (outerMIdx << 4)
        | (innerMIdx << 2)
        | innerKIdx
    )


def cutlass_sf_offset_batch(
    rows: torch.Tensor,   # [M] int64 row indices
    blocks: torch.Tensor, # [B] int64 block indices
    numKTiles: int,
) -> torch.Tensor:
    """Vectorized version of cutlass_sf_offset.

    Returns:
        [M, B] int64 tensor of byte offsets.
    """
    rows = rows.long().unsqueeze(1)    # [M, 1]
    blocks = blocks.long().unsqueeze(0) # [1, B]

    mTileIdx  = rows >> 7
    outerMIdx = rows & 31
    innerMIdx = (rows >> 5) & 3
    kTileIdx  = blocks >> 2
    innerKIdx = blocks & 3

    return (
        ((mTileIdx * numKTiles + kTileIdx) << 9)
        | (outerMIdx << 4)
        | (innerMIdx << 2)
        | innerKIdx
    )  # [M, B]


# ---------------------------------------------------------------------------
# Swizzle inverse: flat_byte_offset -> (row, block_idx)
# ---------------------------------------------------------------------------

def cutlass_sf_offset_inverse(byte_off: int, numKTiles: int) -> tuple[int, int]:
    """Invert the CUTLASS swizzle: flat byte offset -> (row_in_buf, block_idx).

    Useful for debugging: given a corrupted or unexpected byte offset, this
    tells you which (row, block) it was supposed to come from.

    Layout derivation (reverse of forward formula):
        byte_off = ((mTileIdx * numKTiles + kTileIdx) << 9)
                   | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx

    Extraction:
        innerKIdx = byte_off & 3
        innerMIdx = (byte_off >> 2) & 3
        outerMIdx = (byte_off >> 4) & 31
        tile_index = byte_off >> 9         # = mTileIdx * numKTiles + kTileIdx
        kTileIdx  = tile_index % numKTiles
        mTileIdx  = tile_index // numKTiles

    Reconstruct row_in_buf and block_idx:
        row_in_buf = mTileIdx * 128 + innerMIdx * 32 + outerMIdx
        block_idx  = kTileIdx * 4 + innerKIdx

    Args:
        byte_off:  flat byte offset into swizzled buffer.
        numKTiles: ceil(K / 64), same as in the forward pass.

    Returns:
        (row_in_buf, block_idx) tuple.
    """
    innerKIdx = byte_off & 3
    innerMIdx = (byte_off >> 2) & 3
    outerMIdx = (byte_off >> 4) & 31
    tile_index = byte_off >> 9         # mTileIdx * numKTiles + kTileIdx

    kTileIdx = tile_index % numKTiles
    mTileIdx = tile_index // numKTiles

    row_in_buf = mTileIdx * 128 + innerMIdx * 32 + outerMIdx
    block_idx  = kTileIdx * 4 + innerKIdx

    return row_in_buf, block_idx


# ---------------------------------------------------------------------------
# Apply swizzle to a row-major scale tensor
# ---------------------------------------------------------------------------

def apply_swizzle(scale_rowmajor: torch.Tensor) -> torch.Tensor:
    """Reorder a row-major scale tensor into CUTLASS swizzled layout.

    Args:
        scale_rowmajor: [M, numKBlocks] tensor, where numKBlocks = K // 16.
                        Each element is a scale factor (fp8, fp32, or any type).

    Returns:
        Flat tensor of length M * numKBlocks in CUTLASS swizzled order.
        The physical 5D layout is [numMTiles, numKTiles, 32, 4, 4] flattened.
    """
    M, numKBlocks = scale_rowmajor.shape
    numKTiles = (numKBlocks + 3) // 4   # ceil(numKBlocks / 4)
    numMTiles = (M + 127) // 128        # ceil(M / 128)

    # Total buffer size: numMTiles * numKTiles * 128 (32 * 4) * 4 (inner K)
    # = numMTiles * numKTiles * 512 bytes
    buf_size = numMTiles * numKTiles * 512
    out = torch.zeros(buf_size, dtype=scale_rowmajor.dtype)

    rows = torch.arange(M, dtype=torch.long)
    blocks = torch.arange(numKBlocks, dtype=torch.long)

    # Vectorized offset computation
    offsets = cutlass_sf_offset_batch(rows, blocks, numKTiles)  # [M, numKBlocks]

    # Scatter into output buffer
    flat_scale = scale_rowmajor.reshape(-1)
    flat_offsets = offsets.reshape(-1)

    out.scatter_(0, flat_offsets, flat_scale.to(out.dtype))
    return out


def unapply_swizzle(
    swizzled: torch.Tensor,
    M: int,
    numKBlocks: int,
) -> torch.Tensor:
    """Invert the swizzle: swizzled flat buffer -> row-major [M, numKBlocks].

    Args:
        swizzled:   flat 1D tensor in CUTLASS swizzled order.
        M:          number of rows.
        numKBlocks: K // 16.

    Returns:
        [M, numKBlocks] tensor in row-major order.
    """
    numKTiles = (numKBlocks + 3) // 4
    out = torch.zeros(M, numKBlocks, dtype=swizzled.dtype)

    rows = torch.arange(M, dtype=torch.long)
    blocks = torch.arange(numKBlocks, dtype=torch.long)
    offsets = cutlass_sf_offset_batch(rows, blocks, numKTiles)  # [M, numKBlocks]

    flat_offsets = offsets.reshape(-1)
    out.reshape(-1).copy_(swizzled[flat_offsets])
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_scalar():
    """Scalar formula smoke test: verify a few known offsets."""
    print("[Test 1] Scalar formula spot checks")

    # With numKTiles = 2 (K=128, numKBlocks=8)
    numKTiles = 2

    # Row 0, block 0: mTile=0, outer=0, inner=0, kTile=0, innerK=0
    # byte_off = (0*2+0)<<9 | 0<<4 | 0<<2 | 0 = 0
    off = cutlass_sf_offset(0, 0, numKTiles)
    assert off == 0, f"Expected 0, got {off}"
    print(f"  (row=0, blk=0) -> off={off}  [PASS]")

    # Row 0, block 1: innerK=1
    # byte_off = 0 | 0 | 0 | 1 = 1
    off = cutlass_sf_offset(0, 1, numKTiles)
    assert off == 1, f"Expected 1, got {off}"
    print(f"  (row=0, blk=1) -> off={off}  [PASS]")

    # Row 1, block 0: mTile=0, outerM=1, innerM=0
    # byte_off = 0 | (1<<4) | 0 | 0 = 16
    off = cutlass_sf_offset(1, 0, numKTiles)
    assert off == 16, f"Expected 16, got {off}"
    print(f"  (row=1, blk=0) -> off={off}  [PASS]")

    # Row 32, block 0: mTile=0, outerM=0, innerM=1
    # byte_off = 0 | 0 | (1<<2) | 0 = 4
    off = cutlass_sf_offset(32, 0, numKTiles)
    assert off == 4, f"Expected 4, got {off}"
    print(f"  (row=32, blk=0) -> off={off}  [PASS]")

    # Row 128, block 0: mTile=1
    # byte_off = (1*2+0)<<9 = 1024
    off = cutlass_sf_offset(128, 0, numKTiles)
    assert off == 1024, f"Expected 1024, got {off}"
    print(f"  (row=128, blk=0) -> off={off}  [PASS]")

    # Row 0, block 4: kTile=1
    # byte_off = (0*2+1)<<9 = 512
    off = cutlass_sf_offset(0, 4, numKTiles)
    assert off == 512, f"Expected 512, got {off}"
    print(f"  (row=0, blk=4) -> off={off}  [PASS]")


def test_inverse_consistency(M=256, K=4096):
    """Test that inverse(forward(r, b)) == (r, b) for all valid (r, b)."""
    print(f"\n[Test 2] Forward/inverse consistency (M={M}, K={K})")
    numKBlocks = K // 16
    numKTiles = (numKBlocks + 3) // 4

    mismatches = 0
    for r in range(M):
        for b in range(numKBlocks):
            off = cutlass_sf_offset(r, b, numKTiles)
            r2, b2 = cutlass_sf_offset_inverse(off, numKTiles)
            if r2 != r or b2 != b:
                print(f"  MISMATCH: ({r},{b}) -> off={off} -> ({r2},{b2})")
                mismatches += 1
                if mismatches >= 5:
                    break
        if mismatches >= 5:
            break

    if mismatches == 0:
        print(f"  All {M * numKBlocks} offsets round-trip correctly.  [PASS]")
    else:
        print(f"  {mismatches} mismatches found.  [FAIL]")
    assert mismatches == 0


def test_bijectivity(M=128, K=64):
    """Test that the swizzle is a bijection: all offsets are unique."""
    print(f"\n[Test 3] Bijectivity (M={M}, K={K})")
    numKBlocks = K // 16
    numKTiles = (numKBlocks + 3) // 4

    offsets = set()
    for r in range(M):
        for b in range(numKBlocks):
            off = cutlass_sf_offset(r, b, numKTiles)
            offsets.add(off)

    expected = M * numKBlocks
    actual = len(offsets)
    if actual == expected:
        print(f"  All {expected} offsets are unique.  [PASS]")
    else:
        print(f"  Only {actual} / {expected} unique offsets — collision!  [FAIL]")
    assert actual == expected, f"Swizzle is not bijective: {actual} != {expected}"


def test_roundtrip_scale_tensor(M=256, K=4096):
    """Apply swizzle then unapply and verify exact reconstruction."""
    print(f"\n[Test 4] Swizzle round-trip (M={M}, K={K})")
    numKBlocks = K // 16

    # Create a row-major scale tensor with unique values for identification
    scale = torch.arange(M * numKBlocks, dtype=torch.float32).reshape(M, numKBlocks)

    swizzled = apply_swizzle(scale)
    recovered = unapply_swizzle(swizzled, M, numKBlocks)

    match = torch.equal(scale, recovered)
    if match:
        print(f"  Recovered scale tensor is identical to original.  [PASS]")
    else:
        diff = (scale - recovered).abs()
        print(f"  Mismatch: max_diff={diff.max().item()}, "
              f"num_wrong={(diff > 0).sum().item()}  [FAIL]")
    assert match


def test_sequential_access_pattern(M=128, K=4096):
    """Show that sequential row access in a single SM's stripe is nearly contiguous.

    For v4b: each SM owns rows [sm*stripe .. (sm+1)*stripe-1]. This test
    measures the 'stride spread' of offsets for a contiguous row slice,
    helping size the swizzled smem staging buffer.
    """
    print(f"\n[Test 5] Access pattern for single-SM stripe (M={M}, K={K})")
    numKBlocks = K // 16
    numKTiles = (numKBlocks + 3) // 4

    # Simulate SM 0 owning rows 0..15 (a 16-row stripe)
    stripe_size = 16
    rows = list(range(stripe_size))
    blocks = list(range(numKBlocks))

    all_offsets = []
    for r in rows:
        for b in blocks:
            all_offsets.append(cutlass_sf_offset(r, b, numKTiles))

    all_offsets.sort()
    min_off = min(all_offsets)
    max_off = max(all_offsets)
    span = max_off - min_off + 1
    density = len(all_offsets) / span * 100

    print(f"  Stripe rows 0..{stripe_size-1}, all {numKBlocks} K-blocks")
    print(f"  Offset range: [{min_off}, {max_off}], span={span} bytes")
    print(f"  Density: {density:.1f}%  ({len(all_offsets)} unique offsets in span)")
    print(f"  Smem budget for swizzled scale staging: {span} bytes")
    print(f"  (For v4b: this is the smem required to stage a {stripe_size}-row scale tile)")


def test_large_model_dims():
    """Test with Gemma4 26B actual dims: M=128 (one M-tile), K=4096."""
    print("\n[Test 6] Gemma4 26B expert dims (M=128, K=4096, num_experts=128)")
    M = 128    # one M-tile (fits in one mTileIdx)
    K = 4096   # gate_proj input dim for MoE experts

    numKBlocks = K // 16   # = 256
    numKTiles = (numKBlocks + 3) // 4  # = 64

    # Verify numKTiles
    assert numKTiles == 64, f"Expected numKTiles=64, got {numKTiles}"

    # The 5D layout dimensions for one expert:
    numMTiles = (M + 127) // 128  # = 1
    buf_size = numMTiles * numKTiles * 512
    print(f"  numMTiles={numMTiles}, numKTiles={numKTiles}")
    print(f"  5D layout: [{numMTiles}, {numKTiles}, 32, 4, 4]")
    print(f"  Total swizzled buffer: {buf_size} bytes = {buf_size // 1024} KB")
    print(f"  Row-major size: {M * numKBlocks} bytes")
    print(f"  Buffer overhead: {buf_size / (M * numKBlocks):.2f}×")

    # Confirm bijectivity at this scale
    offsets = set()
    for r in range(M):
        for b in range(numKBlocks):
            offsets.add(cutlass_sf_offset(r, b, numKTiles))
    assert len(offsets) == M * numKBlocks, "Swizzle collision at Gemma4 dims!"
    print(f"  Bijectivity at M={M}, K={K}: PASS  ({len(offsets)} unique offsets)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CUTLASS swizzle unpacker tests")
    parser.add_argument("--M", type=int, default=256, help="Rows for round-trip test")
    parser.add_argument("--K", type=int, default=4096, help="Cols (FP4 elements) for round-trip test")
    args = parser.parse_args()

    print("=" * 70)
    print("CUTLASS 128×4 Blockscale Swizzle Unpacker — CPU reference (v4b prep)")
    print("=" * 70)
    print()
    print("Formula (from fused_shuffle_quant.cu:72 + rms_norm_dynamic_fp4_quant.cu):")
    print("  mTileIdx  = row >> 7")
    print("  outerMIdx = row & 31")
    print("  innerMIdx = (row >> 5) & 3")
    print("  kTileIdx  = block_idx >> 2")
    print("  innerKIdx = block_idx & 3")
    print("  byte_off  = ((mTileIdx*numKTiles + kTileIdx) << 9)")
    print("              | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx")
    print()

    test_forward_scalar()
    test_inverse_consistency(M=args.M, K=args.K)
    test_bijectivity(M=128, K=64)
    test_roundtrip_scale_tensor(M=args.M, K=args.K)
    test_sequential_access_pattern(M=128, K=4096)
    test_large_model_dims()

    print("\n" + "=" * 70)
    print("All tests PASSED.")
    print()
    print("v4b integration notes:")
    print("  1. Checkpoint scales are row-major (NOT swizzled on disk).")
    print("  2. CUTLASS MoE kernel expects swizzled layout at runtime.")
    print("  3. For mega-graph cooperative kernel (v4b), each SM reads a")
    print("     contiguous row-stripe — use row-major (not swizzled) access")
    print("     for scale staging in smem. Swizzle only needed for the")
    print("     cutlass_fp4_moe_mm codepath, not the cooperative WMMA path.")
    print("  4. apply_swizzle() above is the Python reference for the")
    print("     fused_shuffle_quant.cu swizzle output — use for bit-compare.")


if __name__ == "__main__":
    main()
