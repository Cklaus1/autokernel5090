"""Test suite for the FusenKV vLLM plugin.

Covers plugin registration, spec resolution, CacheDType monkey-patching,
backend class properties, forward pass, and kernel correctness.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
import torch

from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
from kv_cache_gen.config import parse_spec
from fusen_kv.plugin import register, FUSEN_DTYPES, _patch_cache_dtype

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required",
)


# ============================================================
# 1. Plugin Registration
# ============================================================

class TestRegistration:
    """Test that the plugin registers correctly with vLLM."""

    def test_register_succeeds(self):
        """register() completes without raising."""
        register()

    def test_custom_backend_resolves(self):
        """After register(), AttentionBackendEnum.CUSTOM resolves to FusenKVBackend."""
        register()
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        path = AttentionBackendEnum.CUSTOM.get_path()
        assert "FusenKVBackend" in path, f"CUSTOM path is '{path}', expected FusenKVBackend"

    def test_backend_name(self):
        """Backend name is 'FUSEN_KV'."""
        from fusen_kv.backend import FusenKVBackend
        assert FusenKVBackend.get_name() == "FUSEN_KV"


# ============================================================
# 2. Spec Resolution
# ============================================================

class TestSpecResolution:
    """Test parse_spec and resolve_spec for various inputs."""

    def test_parse_k4v4(self):
        """parse_spec('k4v4') returns a KVCacheSpec with k_bits=4, v_bits=4."""
        spec = parse_spec("k4v4")
        assert isinstance(spec, KVCacheSpec)
        assert spec.k_bits == 4
        assert spec.v_bits == 4

    def test_parse_fusen_returns_default(self):
        """parse_spec('fusen') is an alias; resolve_spec('fusen') returns default spec."""
        from fusen_kv.spec_resolver import resolve_spec
        spec = resolve_spec("fusen")
        assert isinstance(spec, KVCacheSpec)
        # Default is k4v4b64
        assert spec.name == "k4v4b64"

    def test_parse_auto_returns_none(self):
        """parse_spec('auto') returns None (sentinel for adaptive selection)."""
        result = parse_spec("auto")
        assert result is None

    def test_parse_invalid_raises(self):
        """parse_spec('invalid_garbage') raises ValueError."""
        with pytest.raises(ValueError, match="Unknown KV cache spec"):
            parse_spec("invalid_garbage")

    def test_all_predefined_specs_resolve(self):
        """Every entry in PREDEFINED_SPECS is a valid KVCacheSpec."""
        for name, spec in PREDEFINED_SPECS.items():
            assert isinstance(spec, KVCacheSpec), f"PREDEFINED_SPECS['{name}'] is not a KVCacheSpec"
            assert spec.k_bits > 0
            assert spec.v_bits > 0
            # Round-trip: parse_spec(name) should return same spec
            resolved = parse_spec(name)
            assert resolved is not None, f"parse_spec('{name}') returned None"
            assert resolved.name == spec.name, (
                f"parse_spec('{name}').name = '{resolved.name}', expected '{spec.name}'"
            )


# ============================================================
# 3. CacheDType Monkey-Patch
# ============================================================

class TestCacheDTypePatch:
    """Test that the CacheDType monkey-patch accepts FusenKV dtypes."""

    @pytest.fixture(autouse=True)
    def _register(self):
        register()

    def test_validator_accepts_k4v4(self):
        """After register(), CacheConfig validator accepts 'k4v4'."""
        from vllm.config.cache import CacheConfig
        # The patched validator should not raise on our dtypes
        result = CacheConfig._validate_cache_dtype("k4v4")
        assert result == "k4v4"

    def test_validator_accepts_fusen(self):
        """After register(), CacheConfig validator accepts 'fusen'."""
        from vllm.config.cache import CacheConfig
        result = CacheConfig._validate_cache_dtype("fusen")
        assert result == "fusen"

    def test_original_auto_still_works(self):
        """Original dtype 'auto' still passes validation."""
        from vllm.config.cache import CacheConfig
        result = CacheConfig._validate_cache_dtype("auto")
        assert result == "auto"

    def test_original_fp8_still_works(self):
        """Original dtype 'fp8_e4m3' still passes validation."""
        from vllm.config.cache import CacheConfig
        result = CacheConfig._validate_cache_dtype("fp8_e4m3")
        assert result == "fp8_e4m3"


# ============================================================
# 4. Backend Class Properties
# ============================================================

class TestBackendProperties:
    """Test static/class properties of FusenKVBackend."""

    def test_supported_kv_cache_dtypes_contains_formats(self):
        """supported_kv_cache_dtypes contains our key formats."""
        from fusen_kv.backend import FusenKVBackend
        dtypes = FusenKVBackend.supported_kv_cache_dtypes
        for fmt in ["fusen", "k4v4", "k8v8", "k8v4"]:
            assert fmt in dtypes, f"'{fmt}' not in supported_kv_cache_dtypes"

    def test_supports_head_size_256(self):
        from fusen_kv.backend import FusenKVBackend
        assert FusenKVBackend.supports_head_size(256) is True

    def test_supports_head_size_512(self):
        from fusen_kv.backend import FusenKVBackend
        assert FusenKVBackend.supports_head_size(512) is True

    def test_does_not_support_head_size_32(self):
        from fusen_kv.backend import FusenKVBackend
        assert FusenKVBackend.supports_head_size(32) is False

    def test_kv_cache_shape_k4v4(self):
        """get_kv_cache_shape returns correct dimensions for k4v4."""
        from fusen_kv.backend import FusenKVBackend
        num_blocks, block_size, num_kv_heads, head_size = 128, 16, 8, 256
        shape = FusenKVBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, "k4v4"
        )
        # k4v4 → k4v4b64: k_bytes=0.5/dim, v_bytes=0.5/dim → slot_bytes = 256
        spec = parse_spec("k4v4")
        expected_slot_bytes = int(spec.k_bytes_per_dim * head_size
                                  + spec.v_bytes_per_dim * head_size)
        assert shape == (num_blocks, block_size, num_kv_heads, expected_slot_bytes)

    def test_kv_cache_shape_k8v8(self):
        """get_kv_cache_shape returns correct dimensions for k8v8."""
        from fusen_kv.backend import FusenKVBackend
        num_blocks, block_size, num_kv_heads, head_size = 64, 16, 4, 128
        shape = FusenKVBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, "k8v8"
        )
        spec = parse_spec("k8v8")
        expected_slot_bytes = int(spec.k_bytes_per_dim * head_size
                                  + spec.v_bytes_per_dim * head_size)
        assert shape == (num_blocks, block_size, num_kv_heads, expected_slot_bytes)


# ============================================================
# 5. Impl Forward Pass (GPU Required)
# ============================================================

@requires_gpu
class TestForwardPass:
    """Test FusenKVImpl forward pass on GPU."""

    @pytest.fixture
    def impl_k4v4(self):
        """Create a FusenKVImpl with k4v4 spec using GQA layout."""
        from fusen_kv.backend import FusenKVImpl, AttentionType
        return FusenKVImpl(
            num_heads=8,
            head_size=256,
            scale=1.0 / (256 ** 0.5),
            num_kv_heads=1,  # GQA: 8 query heads share 1 KV head
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="k4v4",
            attn_type=AttentionType.DECODER,
        )

    @pytest.fixture
    def kv_cache_and_scales(self, impl_k4v4):
        """Allocate KV cache and scales tensors."""
        num_blocks = 64
        block_size = 16
        num_kv_heads = 1
        head_size = 256
        spec = impl_k4v4.spec
        slot_bytes = int(spec.k_bytes_per_dim * head_size
                         + spec.v_bytes_per_dim * head_size)
        kv_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, slot_bytes,
            dtype=torch.uint8, device="cuda",
        )
        return kv_cache, num_blocks, block_size

    def test_forward_decode_nonzero(self, impl_k4v4, kv_cache_and_scales):
        """Decode forward produces non-zero output."""
        from fusen_kv.backend import FusenKVMetadata
        kv_cache, num_blocks, block_size = kv_cache_and_scales
        B = 1
        num_heads, num_kv_heads, head_size = 8, 1, 256

        # Create a simple layer object to hold scales
        layer = torch.nn.Module()

        query = torch.randn(B, num_heads, head_size, device="cuda", dtype=torch.float16)
        key = torch.randn(B, num_kv_heads, head_size, device="cuda", dtype=torch.float16)
        value = torch.randn(B, num_kv_heads, head_size, device="cuda", dtype=torch.float16)

        # First store some KV data
        slot_mapping = torch.zeros(B, dtype=torch.int64, device="cuda")
        store_meta = FusenKVMetadata(
            block_table=torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda"),
            seq_lens=torch.ones(B, dtype=torch.int32, device="cuda"),
            slot_mapping=slot_mapping,
            is_prefill=True,
            num_prefill_tokens=B,
            num_decode_tokens=0,
        )
        impl_k4v4.forward(layer, query, key, value, kv_cache, store_meta)

        # Now decode
        block_table = torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda")
        seq_lens = torch.ones(B, dtype=torch.int32, device="cuda")
        decode_meta = FusenKVMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            slot_mapping=None,
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=B,
        )

        output = impl_k4v4.forward(layer, query, key, value, kv_cache, decode_meta)
        assert output.shape == (B, num_heads * head_size)
        assert output.abs().sum().item() > 0, "Decode output is all zeros"

    def test_forward_prefill_nonzero(self, impl_k4v4, kv_cache_and_scales):
        """Prefill forward produces non-zero output."""
        from fusen_kv.backend import FusenKVMetadata
        kv_cache, num_blocks, block_size = kv_cache_and_scales
        num_tokens = 4
        num_heads, num_kv_heads, head_size = 8, 1, 256

        layer = torch.nn.Module()
        query = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16)
        key = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda", dtype=torch.float16)
        value = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda", dtype=torch.float16)

        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
        meta = FusenKVMetadata(
            block_table=torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
            slot_mapping=slot_mapping,
            is_prefill=True,
            num_prefill_tokens=num_tokens,
            num_decode_tokens=0,
        )

        output = impl_k4v4.forward(layer, query, key, value, kv_cache, meta)
        assert output.shape == (num_tokens, num_heads * head_size)
        assert output.abs().sum().item() > 0, "Prefill output is all zeros"

    def test_output_shape(self, impl_k4v4, kv_cache_and_scales):
        """Output shape is [num_tokens, num_heads * head_size]."""
        from fusen_kv.backend import FusenKVMetadata
        kv_cache, num_blocks, block_size = kv_cache_and_scales
        num_heads, num_kv_heads, head_size = 8, 1, 256

        for num_tokens in [1, 4]:
            layer = torch.nn.Module()
            query = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16)
            key = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda", dtype=torch.float16)
            value = torch.randn(num_tokens, num_kv_heads, head_size, device="cuda", dtype=torch.float16)

            slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")
            meta = FusenKVMetadata(
                block_table=torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda"),
                seq_lens=torch.tensor([num_tokens], dtype=torch.int32, device="cuda"),
                slot_mapping=slot_mapping,
                is_prefill=True,
                num_prefill_tokens=num_tokens,
                num_decode_tokens=0,
            )

            output = impl_k4v4.forward(layer, query, key, value, kv_cache, meta)
            assert output.shape == (num_tokens, num_heads * head_size), (
                f"Expected ({num_tokens}, {num_heads * head_size}), got {output.shape}"
            )


# ============================================================
# 6. Kernel Correctness (GPU Required)
# ============================================================

@requires_gpu
class TestKernelCorrectness:
    """Test store + decode roundtrip fidelity."""

    @pytest.mark.parametrize("head_size", [256, 512])
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_store_decode_roundtrip(self, head_size, batch_size):
        """Store KV then decode; cosine similarity > 0.95 with FP16 reference.

        Uses GQA layout (8 query heads, 1 KV head) matching the kernel's
        design target (Gemma4-style). The block_h=8 parameter in the decode
        kernel groups query heads that share one KV head, so GQA is required
        for correct operation.
        """
        from fusen_kv.backend import FusenKVImpl, FusenKVMetadata, AttentionType

        num_heads = 8
        num_kv_heads = 1  # GQA: 8 query heads share 1 KV head
        num_blocks = 64
        block_size = 16

        impl = FusenKVImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / (head_size ** 0.5),
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="k4v4",
            attn_type=AttentionType.DECODER,
        )

        spec = impl.spec
        slot_bytes = int(spec.k_bytes_per_dim * head_size
                         + spec.v_bytes_per_dim * head_size)
        kv_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, slot_bytes,
            dtype=torch.uint8, device="cuda",
        )
        layer = torch.nn.Module()

        # Store a single sequence of length seq_len into block 0
        seq_len = 8
        torch.manual_seed(42)
        key = torch.randn(seq_len, num_kv_heads, head_size, device="cuda", dtype=torch.float16)
        value = torch.randn(seq_len, num_kv_heads, head_size, device="cuda", dtype=torch.float16)
        query_store = torch.randn(seq_len, num_heads, head_size, device="cuda", dtype=torch.float16)

        slot_mapping = torch.arange(seq_len, dtype=torch.int64, device="cuda")
        store_meta = FusenKVMetadata(
            block_table=torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
            slot_mapping=slot_mapping,
            is_prefill=True,
            num_prefill_tokens=seq_len,
            num_decode_tokens=0,
        )
        impl.forward(layer, query_store, key, value, kv_cache, store_meta)

        # Decode: all batch entries read the same cached sequence (block 0)
        query_decode = torch.randn(batch_size, num_heads, head_size, device="cuda", dtype=torch.float16)

        # Reference: FP16 attention with GQA expansion
        groups = num_heads // num_kv_heads
        k_exp = key.repeat_interleave(groups, dim=1)  # [S, Hq, D]
        v_exp = value.repeat_interleave(groups, dim=1)  # [S, Hq, D]
        scores = torch.einsum("bhd,shd->bhs", query_decode.float(), k_exp.float()) * impl.scale
        weights = torch.softmax(scores, dim=-1)
        ref_out = torch.einsum("bhs,shd->bhd", weights, v_exp.float())  # [B, Hq, D]
        ref_out = ref_out.reshape(batch_size, -1).half()

        # All batch entries point to block 0 (same cached KV)
        block_table = torch.zeros(batch_size, num_blocks, dtype=torch.int32, device="cuda")
        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
        decode_meta = FusenKVMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            slot_mapping=None,
            is_prefill=False,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
        )

        output = impl.forward(layer, query_decode, key[:batch_size], value[:batch_size],
                               kv_cache, decode_meta)
        assert output.shape == (batch_size, num_heads * head_size)

        # Per-sample cosine similarity: quantized decode vs FP16 reference
        # k4v4 with 4-bit quantization still achieves > 0.95 on these shapes.
        for b in range(batch_size):
            cos_sim = torch.nn.functional.cosine_similarity(
                output[b:b+1].float(),
                ref_out[b:b+1].float(),
            ).item()
            assert cos_sim > 0.95, (
                f"Cosine similarity {cos_sim:.4f} < 0.95 for sample {b}, "
                f"head_size={head_size}, batch_size={batch_size}"
            )
