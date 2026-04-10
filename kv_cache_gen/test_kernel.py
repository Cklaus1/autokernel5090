"""Test the universal kernel against Python reference for all predefined specs."""

import torch
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn, make_store_fn


def python_reference_attention(query, key, value, scale):
    """Reference attention in pure PyTorch for correctness validation."""
    B, Hq, D = query.shape
    _, Hk, _ = key.shape
    groups = Hq // Hk

    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    scores = torch.einsum("bhd,shd->bhs", query.float(), key.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,shd->bhd", attn, value.float())
    return out


def test_spec(spec_name, spec, D=256, Hq=16, Hk=8, seq_len=32, block_size=16):
    """Test one spec: store → decode → compare with reference."""
    B = 1
    device = "cuda"

    # Generate functions
    decode_fn = make_decode_fn(spec)
    store_fn = make_store_fn(spec)

    # Random K, V in float (what the model produces)
    key_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    value_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
    query = torch.randn(B, Hq, D, device=device, dtype=torch.float16)

    # Reference attention (no quantization)
    ref_out = python_reference_attention(query, key_fp, value_fp, 1.0 / (D ** 0.5))

    # Allocate cache
    num_blocks = (seq_len + block_size - 1) // block_size
    slot_bytes = spec.slot_bytes(D)
    # Cache shape: (num_blocks, block_size, Hk, slot_bytes_per_head)
    cache_last_dim = slot_bytes // Hk if slot_bytes % Hk == 0 else slot_bytes
    # Simpler: use the raw slot bytes per head
    k_bytes = int(spec.k_bytes_per_dim * D)
    v_bytes = int(spec.v_bytes_per_dim * D)
    cache_per_head = k_bytes + v_bytes
    kv_cache = torch.zeros(num_blocks, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)

    # Store K/V one token at a time
    class FakeLayer:
        pass
    layer = FakeLayer()

    slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32)
    store_fn(key_fp, value_fp, kv_cache, slot_mapping, layer, Hk)

    # Build block table
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

    # Decode
    scale_block = max(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // scale_block

    out = decode_fn(query, kv_cache, layer._fc_scales, block_table, seq_lens,
                    1.0 / (D ** 0.5), Hk)

    # Compare
    ref = ref_out[0].to(torch.float16)
    got = out[0].to(torch.float16)

    # Cosine similarity (quantization means we won't match exactly)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref.reshape(1, -1).float(), got.reshape(1, -1).float()
    ).item()

    # Max absolute error
    max_err = (ref.float() - got.float()).abs().max().item()

    # Relative error
    rel_err = ((ref.float() - got.float()).abs() / (ref.float().abs() + 1e-6)).mean().item()

    return cos_sim, max_err, rel_err


def main():
    print(f"{'Spec':<12} {'Comp':>5} {'CosSim':>8} {'MaxErr':>8} {'RelErr':>8} {'Status'}")
    print("-" * 60)

    for name, spec in PREDEFINED_SPECS.items():
        try:
            cos_sim, max_err, rel_err = test_spec(name, spec)
            comp = spec.compression_vs_bf16(256)
            status = "PASS" if cos_sim > 0.90 else "FAIL"
            print(f"{name:<12} {comp:>5.1f}x {cos_sim:>7.4f} {max_err:>8.4f} {rel_err:>8.4f} {status}")
        except Exception as e:
            print(f"{name:<12}  ERROR: {e}")

    print()
    print("CosSim > 0.95 = good quality, > 0.99 = near-lossless")


if __name__ == "__main__":
    main()
