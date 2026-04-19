"""
Standalone tests for FP8 decode attention backend.

Verifies the Triton kernel produces correct output vs a PyTorch reference
for FP8 KV cache across Gemma4-relevant configurations:
  - GQA with various group sizes
  - head_dim=256 and head_dim=512
  - logits_soft_cap=50.0 (Gemma4) and disabled
  - Variable sequence lengths and batch sizes
"""

import math
import pytest
import torch

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def device():
    return "cuda"


def _run_fp8_vs_ref(
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    logits_soft_cap: float,
    device: str,
    atol_cos: float = 0.98,
):
    """Run FP8 kernel and compare against BF16 PyTorch reference."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from kernels.fp8_decode_attention import (
        bf16_decode_attention_ref,
        create_test_data,
        fp8_decode_attention,
    )

    data = create_test_data(
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        device=device,
    )

    out_fp8 = fp8_decode_attention(
        q=data["q"],
        k_cache=data["k_cache_fp8"],
        v_cache=data["v_cache_fp8"],
        block_table=data["block_table"],
        seq_lens=data["seq_lens"],
        k_scale=data["k_scale"],
        v_scale=data["v_scale"],
        logits_soft_cap=logits_soft_cap,
    )

    out_ref = bf16_decode_attention_ref(
        q=data["q"],
        k_cache=data["k_cache_bf16"],
        v_cache=data["v_cache_bf16"],
        block_table=data["block_table"],
        seq_lens=data["seq_lens"],
        logits_soft_cap=logits_soft_cap,
    )

    assert out_fp8.shape == out_ref.shape, (
        f"Shape mismatch: {out_fp8.shape} vs {out_ref.shape}"
    )

    # Cosine similarity (FP8 quant introduces bounded error)
    cos = torch.nn.functional.cosine_similarity(
        out_fp8.float().flatten(), out_ref.float().flatten(), dim=0
    ).item()

    assert cos >= atol_cos, (
        f"Cosine similarity {cos:.4f} < {atol_cos} "
        f"(batch={batch}, q_heads={num_q_heads}, kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}, seq_len={seq_len}, cap={logits_soft_cap})"
    )
    return cos


def _run_dispatch_wrapper(
    batch: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    logits_soft_cap: float,
    device: str,
):
    """Test the fp8_decode_dispatch wrapper handles 4D input correctly."""
    from kernels.fp8_decode_attention import create_test_data

    from .fp8_decode_backend import fp8_decode_dispatch

    data = create_test_data(
        batch=batch,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        device=device,
    )

    # Simulate vLLM's 4D query: [batch, 1, num_q_heads, head_dim]
    q_4d = data["q"].unsqueeze(1)

    out = fp8_decode_dispatch(
        q=q_4d,
        k_cache=data["k_cache_fp8"],
        v_cache=data["v_cache_fp8"],
        block_table=data["block_table"],
        seq_lens=data["seq_lens"],
        k_scale=data["k_scale"],
        v_scale=data["v_scale"],
        logits_soft_cap=logits_soft_cap,
    )

    assert out.shape == (batch, num_q_heads, head_dim), (
        f"Expected shape ({batch}, {num_q_heads}, {head_dim}), got {out.shape}"
    )
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"


# ---- Gemma4 sliding window config: head_dim=256, GQA 8:1 ----

class TestGemma4Sliding:
    """Gemma4 sliding window attention: head_dim=256, 16 Q heads, 8 KV heads."""

    def test_basic(self, device):
        cos = _run_fp8_vs_ref(
            batch=4, num_q_heads=16, num_kv_heads=8,
            head_dim=256, seq_len=1024, logits_soft_cap=0.0,
            device=device,
        )
        assert cos > 0.99

    def test_with_soft_cap(self, device):
        cos = _run_fp8_vs_ref(
            batch=4, num_q_heads=16, num_kv_heads=8,
            head_dim=256, seq_len=1024, logits_soft_cap=50.0,
            device=device,
        )
        assert cos > 0.98

    def test_large_batch(self, device):
        _run_fp8_vs_ref(
            batch=32, num_q_heads=16, num_kv_heads=8,
            head_dim=256, seq_len=2048, logits_soft_cap=50.0,
            device=device,
        )

    def test_dispatch_wrapper(self, device):
        _run_dispatch_wrapper(
            batch=4, num_q_heads=16, num_kv_heads=8,
            head_dim=256, seq_len=512, logits_soft_cap=50.0,
            device=device,
        )


# ---- Gemma4 global attention config: head_dim=512 ----

class TestGemma4Global:
    """Gemma4 global attention: head_dim=512 (double-width KV)."""

    def test_basic_512(self, device):
        _run_fp8_vs_ref(
            batch=4, num_q_heads=16, num_kv_heads=8,
            head_dim=512, seq_len=512, logits_soft_cap=0.0,
            device=device,
        )

    def test_with_soft_cap_512(self, device):
        _run_fp8_vs_ref(
            batch=4, num_q_heads=16, num_kv_heads=8,
            head_dim=512, seq_len=512, logits_soft_cap=50.0,
            device=device,
        )


# ---- GQA edge cases ----

class TestGQAVariants:
    """Various GQA ratios."""

    def test_mha_no_grouping(self, device):
        """MHA: num_q_heads == num_kv_heads (group_size=1)."""
        _run_fp8_vs_ref(
            batch=4, num_q_heads=8, num_kv_heads=8,
            head_dim=256, seq_len=512, logits_soft_cap=0.0,
            device=device,
        )

    def test_gqa_4x(self, device):
        """GQA 4:1."""
        _run_fp8_vs_ref(
            batch=4, num_q_heads=32, num_kv_heads=8,
            head_dim=256, seq_len=512, logits_soft_cap=50.0,
            device=device,
        )

    def test_mqa(self, device):
        """MQA: single KV head."""
        _run_fp8_vs_ref(
            batch=4, num_q_heads=16, num_kv_heads=1,
            head_dim=256, seq_len=512, logits_soft_cap=0.0,
            device=device,
        )


# ---- Sequence length edge cases ----

class TestSeqLenEdges:
    """Edge cases for sequence lengths."""

    def test_short_seq(self, device):
        """Very short sequence (less than one page)."""
        _run_fp8_vs_ref(
            batch=4, num_q_heads=16, num_kv_heads=8,
            head_dim=256, seq_len=8, logits_soft_cap=0.0,
            device=device, atol_cos=0.95,
        )

    def test_single_batch(self, device):
        _run_fp8_vs_ref(
            batch=1, num_q_heads=16, num_kv_heads=8,
            head_dim=256, seq_len=2048, logits_soft_cap=50.0,
            device=device,
        )


# ---- Plugin integration ----

class TestPluginIntegration:
    """Test FP8DecodeBackendPlugin interface."""

    def test_applies_to_fp8_gpu(self):
        from ..types import GPUInfo, OpCategory, OpProfile, ProfileResult

        from .fp8_decode_backend import FP8DecodeBackendPlugin

        gpu = GPUInfo(
            name="PRO 6000", compute_capability=(12, 0),
            has_fp8=True, memory_gb=96.0,
        )
        profile = ProfileResult(
            model_name="gemma4-26b",
            gpu=gpu,
            total_time_us=1000.0,
            ops=[OpProfile(
                name="attention", category=OpCategory.ATTENTION,
                time_fraction=0.4,
            )],
            metadata={"kv_cache_dtype": "fp8_e4m3", "logits_soft_cap": 50.0},
        )
        plugin = FP8DecodeBackendPlugin()
        assert plugin.applies_to(profile, gpu)

    def test_not_applies_bf16_kv(self):
        from ..types import GPUInfo, OpCategory, OpProfile, ProfileResult

        from .fp8_decode_backend import FP8DecodeBackendPlugin

        gpu = GPUInfo(has_fp8=True)
        profile = ProfileResult(
            model_name="test", gpu=gpu,
            ops=[OpProfile(name="attn", category=OpCategory.ATTENTION, time_fraction=0.3)],
            metadata={"kv_cache_dtype": "bfloat16"},
        )
        plugin = FP8DecodeBackendPlugin()
        assert not plugin.applies_to(profile, gpu)

    def test_not_applies_no_fp8_gpu(self):
        from ..types import GPUInfo, OpCategory, OpProfile, ProfileResult

        from .fp8_decode_backend import FP8DecodeBackendPlugin

        gpu = GPUInfo(has_fp8=False)
        profile = ProfileResult(
            model_name="test", gpu=gpu,
            ops=[OpProfile(name="attn", category=OpCategory.ATTENTION, time_fraction=0.3)],
            metadata={"kv_cache_dtype": "fp8_e4m3"},
        )
        plugin = FP8DecodeBackendPlugin()
        assert not plugin.applies_to(profile, gpu)
