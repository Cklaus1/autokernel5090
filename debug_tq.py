"""Debug TurboQuant quality issue on Gemma 4.

Test: quantize a known K vector, store it, read it back, check if
the attention score is preserved. This isolates the store/load path
from the full model.
"""
import math
import torch
from vllm.turboquant.config import TurboQuantConfig
from vllm.turboquant.quantizer import (
    TurboQuantizer,
    generate_rotation_matrix,
    generate_qjl_matrix,
)
from vllm.turboquant.centroids import get_centroids


def test_roundtrip(head_dim, total_bits=4, label=""):
    """Test quantize → pack → unpack → dequantize roundtrip."""
    config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits)
    quantizer = TurboQuantizer(config, layer_idx=0)

    # Random K vector
    torch.manual_seed(42)
    k = torch.randn(1, head_dim)
    q = torch.randn(1, head_dim)

    # True dot product
    true_score = (q @ k.T).item()

    # Quantize
    compressed = quantizer.quantize(k)

    # Pack → unpack roundtrip
    packed = quantizer.pack_cache(compressed)
    unpacked = quantizer.unpack_cache(packed)

    # Check indices survived packing
    idx_match = (compressed["mse_indices"] == unpacked["mse_indices"]).all()
    sign_match = (compressed["qjl_signs"] == unpacked["qjl_signs"]).all()
    norm_close = torch.allclose(compressed["vec_norm"].float(),
                                 unpacked["vec_norm"].float(), rtol=1e-3)

    # Dequantize and compute score
    k_recon = quantizer.dequantize(unpacked)
    recon_score = (q @ k_recon.T).item()

    # Attention score via TQ estimator
    tq_score = quantizer.attention_scores(
        q.unsqueeze(0).unsqueeze(0),  # (1,1,1,D)
        {k: v.unsqueeze(0).unsqueeze(0) for k, v in compressed.items()},
    ).item()

    print(f"\n{'='*60}")
    print(f"  {label} head_dim={head_dim}, tq{total_bits}")
    print(f"{'='*60}")
    print(f"  True score:      {true_score:.6f}")
    print(f"  Recon score:     {recon_score:.6f} (err={abs(recon_score-true_score):.6f})")
    print(f"  TQ estimator:    {tq_score:.6f} (err={abs(tq_score-true_score):.6f})")
    print(f"  Idx pack OK:     {idx_match.item()}")
    print(f"  Signs pack OK:   {sign_match.item()}")
    print(f"  Norms close:     {norm_close}")
    print(f"  Key packed size: {config.key_packed_size}")
    print(f"  Val packed size: {config.value_packed_size}")
    print(f"  Slot size:       {config.slot_size}")
    print(f"  Padded slot:     {config.padded_slot_size}")

    return idx_match.item() and sign_match.item()


def test_store_load_path(head_dim, total_bits=4, label=""):
    """Test the actual _store_kv → _decode_attention_python path."""
    config = TurboQuantConfig(head_dim=head_dim, total_bits=total_bits,
                              value_quant_bits=4)

    kps = config.key_packed_size
    vps = config.value_packed_size
    slot_size = config.padded_slot_size

    print(f"\n{'='*60}")
    print(f"  Store/Load test: {label} head_dim={head_dim}, tq{total_bits}")
    print(f"{'='*60}")
    print(f"  key_packed_size:  {kps}")
    print(f"  value_packed_size: {vps}")
    print(f"  slot_size (raw):  {config.slot_size}")
    print(f"  padded_slot_size: {slot_size}")

    # The cache shape uses head_size = padded_slot_size // 2
    # This is what TurboQuantAttentionBackend.get_kv_cache_shape does
    effective_head_size = slot_size // 2

    print(f"  effective_head_size (for cache): {effective_head_size}")
    print(f"  actual head_dim: {head_dim}")
    print(f"  MISMATCH: {'YES - THIS IS THE BUG!' if effective_head_size != head_dim else 'no'}")

    # Now simulate what TurboQuantAttentionImpl.__init__ does:
    # self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
    # where head_size = effective_head_size (padded_slot_size // 2)
    impl_config = TurboQuantConfig.from_cache_dtype(
        f"tq{total_bits}", effective_head_size)

    print(f"\n  Config from actual head_dim={head_dim}:")
    print(f"    key_packed_size: {config.key_packed_size}")
    print(f"    slot_size:       {config.slot_size}")
    print(f"  Config from effective head_size={effective_head_size}:")
    print(f"    key_packed_size: {impl_config.key_packed_size}")
    print(f"    slot_size:       {impl_config.slot_size}")

    if config.key_packed_size != impl_config.key_packed_size:
        print(f"\n  *** BUG: key_packed_size MISMATCH! ***")
        print(f"  Store uses head_dim={head_dim} → kps={config.key_packed_size}")
        print(f"  Load uses head_dim={effective_head_size} → kps={impl_config.key_packed_size}")
        print(f"  The decode path will read bytes at wrong offsets!")


print("=" * 60)
print("TurboQuant Roundtrip Diagnostics")
print("=" * 60)

# Test standard head dims
for hd in [64, 128, 256]:
    test_roundtrip(hd, total_bits=4, label="standard")

# Test Gemma 4 head dims
test_roundtrip(256, total_bits=4, label="Gemma4 sliding")
test_roundtrip(512, total_bits=4, label="Gemma4 global")

print("\n\n" + "=" * 60)
print("Store/Load Path Analysis")
print("=" * 60)

# Standard models
test_store_load_path(128, 4, "standard (Llama)")
test_store_load_path(256, 4, "Gemma4 sliding")
test_store_load_path(512, 4, "Gemma4 global")

# Also check tq3
test_store_load_path(256, 3, "Gemma4 sliding tq3")
test_store_load_path(512, 3, "Gemma4 global tq3")
