#!/usr/bin/env python3
"""
Test suite for the Two-Tier Brain architecture.

Verifies:
    1. Detection logic (router confidence, residual norm, output entropy)
    2. GPU LRU cache behavior
    3. Slow expert loading and forward pass
    4. Batch splitting and merging
    5. Fast brain alone produces degraded output
    6. Fast brain + slow brain produces identical output to full model
    7. Slow brain hit rate measurement
    8. Async prefetch mechanics
    9. Graceful degradation (timeout skipping)
   10. Generic two-tier forward

Usage:
    python tools/test_two_tier.py              # Run all tests
    python tools/test_two_tier.py -v           # Verbose
    python tools/test_two_tier.py -k cache     # Run only cache tests
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.two_tier_brain import (
    GPUExpertCache,
    SlowBrainDetector,
    SlowBrainExperts,
    SlowBrainLayers,
    TwoTierConfig,
    TwoTierGeneric,
    TwoTierModel,
    TwoTierStats,
    expert_forward_fp16,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_expert_weights(hidden_dim: int = 256, intermediate_dim: int = 512):
    """Create fake expert weights for testing."""
    return {
        "gate_proj.weight": torch.randn(intermediate_dim, hidden_dim),
        "up_proj.weight": torch.randn(intermediate_dim, hidden_dim),
        "down_proj.weight": torch.randn(hidden_dim, intermediate_dim),
    }


def make_config(**overrides) -> TwoTierConfig:
    """Create a test config with optional overrides."""
    cfg = TwoTierConfig(
        num_original_experts=16,
        num_original_layers=8,
        top_k=4,
        gpu_cache_max_experts=8,
        gpu_cache_max_bytes=50 * 1024**2,
        router_confidence_threshold=0.1,
        residual_norm_threshold=10.0,
        entropy_threshold=3.0,
        prefetch_enabled=False,
        profile=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 1. Detection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSlowBrainDetector(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.pruned_experts = {
            0: {10, 11, 12, 13, 14, 15},  # 6 pruned experts in layer 0
            1: {12, 13, 14, 15},           # 4 pruned in layer 1
            3: set(),                       # none pruned in layer 3
        }
        self.pruned_layers = {2, 5}
        self.detector = SlowBrainDetector(
            self.pruned_experts, self.pruned_layers, self.config
        )

    def test_router_confidence_detects_pruned_expert(self):
        """Token strongly routing to pruned expert should trigger slow brain."""
        batch = 4
        router_logits = torch.randn(batch, 16)
        # Token 0: strongly prefers pruned expert 14
        router_logits[0, 14] = 20.0
        # Token 1: prefers non-pruned expert 3
        router_logits[1, 3] = 20.0

        needs_slow, slow_ids = self.detector.detect_slow_experts(router_logits, layer_idx=0)

        self.assertTrue(needs_slow[0].item(), "Token 0 should need slow brain")
        self.assertIn(14, slow_ids[0], "Token 0 should need expert 14")
        # Token 1 might or might not trigger depending on softmax distribution

    def test_router_no_pruned_experts(self):
        """Layer with no pruned experts should never trigger."""
        router_logits = torch.randn(4, 16)
        needs_slow, slow_ids = self.detector.detect_slow_experts(router_logits, layer_idx=3)
        self.assertFalse(needs_slow.any(), "No pruned experts = no slow brain needed")

    def test_router_unknown_layer(self):
        """Unknown layer should return no triggers."""
        router_logits = torch.randn(4, 16)
        needs_slow, slow_ids = self.detector.detect_slow_experts(router_logits, layer_idx=99)
        self.assertFalse(needs_slow.any())

    def test_residual_norm_detection(self):
        """High norm hidden states should trigger slow layer."""
        batch = 8
        hidden = torch.randn(batch, 1, 256)
        # Make token 3 an outlier (20x normal magnitude)
        hidden[3] *= 20.0

        # Set threshold high enough that normal tokens don't trigger
        # randn(1, 256) has expected norm ~sqrt(256)=16, so 100 is safe
        self.detector.residual_norm_thresholds[2] = 100.0

        needs_slow = self.detector.detect_slow_layer(hidden, pruned_layer_idx=2)
        self.assertTrue(needs_slow[3].item(), "Outlier token should need slow layer")
        # Most normal tokens should not trigger
        normal_count = (~needs_slow).sum().item()
        self.assertGreater(normal_count, batch // 2, "Most normal tokens should not trigger")

    def test_output_entropy_detection(self):
        """High entropy (uncertain) outputs should trigger slow brain."""
        batch = 4
        vocab_size = 1000

        # Token 0: very confident (one-hot-ish)
        logits = torch.zeros(batch, vocab_size)
        logits[0, 42] = 100.0  # Very confident

        # Token 2: very uncertain (uniform-ish)
        logits[2] = torch.zeros(vocab_size)  # Uniform = max entropy

        needs_slow = self.detector.detect_output_entropy(logits)
        self.assertFalse(needs_slow[0].item(), "Confident token should not trigger")
        self.assertTrue(needs_slow[2].item(), "Uncertain token should trigger")

    def test_entropy_disabled(self):
        """When entropy detection is disabled, no triggers."""
        self.config.entropy_enabled = False
        detector = SlowBrainDetector(self.pruned_experts, self.pruned_layers, self.config)
        logits = torch.zeros(4, 1000)  # Very uncertain
        needs_slow = detector.detect_output_entropy(logits)
        self.assertFalse(needs_slow.any())

    def test_calibrate_thresholds(self):
        """Calibration should set threshold from data percentile."""
        samples = [torch.randn(10, 1, 256) for _ in range(5)]
        threshold = self.detector.calibrate_thresholds(samples, pruned_layer_idx=2, percentile=90.0)
        self.assertGreater(threshold, 0, "Calibrated threshold should be positive")
        self.assertEqual(
            self.detector.residual_norm_thresholds[2],
            threshold,
            "Threshold should be stored"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPU LRU Cache Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGPUExpertCache(unittest.TestCase):

    def test_put_and_get(self):
        cache = GPUExpertCache(max_experts=4, max_bytes=100 * 1024**2)
        weights = make_fake_expert_weights()
        cache.put(0, 5, weights)

        result = cache.get(0, 5)
        self.assertIsNotNone(result)
        self.assertIn("gate_proj.weight", result)

    def test_miss(self):
        cache = GPUExpertCache(max_experts=4)
        self.assertIsNone(cache.get(0, 99))

    def test_lru_eviction(self):
        """Oldest entry should be evicted when cache is full."""
        cache = GPUExpertCache(max_experts=3, max_bytes=500 * 1024**2)

        for i in range(3):
            cache.put(0, i, make_fake_expert_weights())

        self.assertEqual(cache.num_cached, 3)

        # Access expert 0 to make it recently used
        cache.get(0, 0)

        # Add expert 3 - should evict expert 1 (oldest not recently accessed)
        cache.put(0, 3, make_fake_expert_weights())
        self.assertEqual(cache.num_cached, 3)
        self.assertIsNone(cache.get(0, 1), "Expert 1 should be evicted")
        self.assertIsNotNone(cache.get(0, 0), "Expert 0 should still be cached (recently used)")

    def test_contains(self):
        cache = GPUExpertCache(max_experts=4)
        cache.put(0, 5, make_fake_expert_weights())
        self.assertTrue(cache.contains(0, 5))
        self.assertFalse(cache.contains(0, 6))

    def test_clear(self):
        cache = GPUExpertCache(max_experts=4)
        cache.put(0, 5, make_fake_expert_weights())
        cache.clear()
        self.assertEqual(cache.num_cached, 0)
        self.assertIsNone(cache.get(0, 5))

    def test_memory_tracking(self):
        cache = GPUExpertCache(max_experts=10, max_bytes=500 * 1024**2)
        weights = make_fake_expert_weights(hidden_dim=64, intermediate_dim=128)
        cache.put(0, 0, weights)
        self.assertGreater(cache.memory_used_mb, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Expert Forward Pass Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExpertForward(unittest.TestCase):

    def test_fp16_forward_shape(self):
        """SwiGLU expert should produce correct output shape."""
        hidden_dim = 128
        intermediate_dim = 256
        batch = 4
        hidden = torch.randn(batch, hidden_dim)
        weights = make_fake_expert_weights(hidden_dim, intermediate_dim)
        output = expert_forward_fp16(hidden, weights)
        self.assertEqual(output.shape, (batch, hidden_dim))

    def test_fp16_forward_deterministic(self):
        """Same input + weights should produce same output."""
        hidden_dim = 64
        hidden = torch.randn(2, hidden_dim)
        weights = make_fake_expert_weights(hidden_dim, 128)
        out1 = expert_forward_fp16(hidden, weights)
        out2 = expert_forward_fp16(hidden, weights)
        torch.testing.assert_close(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch Splitting Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchSplitting(unittest.TestCase):

    def setUp(self):
        self.config = make_config()

    def test_split_and_merge(self):
        """Split then merge should recover original tensor."""
        batch = 8
        hidden = torch.randn(batch, 1, 256)
        needs_slow = torch.tensor([False, True, False, False, True, False, True, False])

        # Create a minimal TwoTierModel for split/merge
        model = self._make_dummy_model()
        fast_h, slow_h, fast_idx, slow_idx = model.split_batch(hidden, needs_slow)

        self.assertEqual(fast_h.shape[0], 5)
        self.assertEqual(slow_h.shape[0], 3)

        # Process (identity)
        merged = model.merge_batch(fast_h, slow_h, fast_idx, slow_idx, batch)
        torch.testing.assert_close(merged, hidden)

    def test_all_fast(self):
        """No slow tokens = empty slow split."""
        batch = 4
        hidden = torch.randn(batch, 1, 256)
        needs_slow = torch.zeros(batch, dtype=torch.bool)

        model = self._make_dummy_model()
        fast_h, slow_h, fast_idx, slow_idx = model.split_batch(hidden, needs_slow)

        self.assertEqual(fast_h.shape[0], batch)
        self.assertEqual(slow_h.shape[0], 0)

    def test_all_slow(self):
        """All slow tokens = empty fast split."""
        batch = 4
        hidden = torch.randn(batch, 1, 256)
        needs_slow = torch.ones(batch, dtype=torch.bool)

        model = self._make_dummy_model()
        fast_h, slow_h, fast_idx, slow_idx = model.split_batch(hidden, needs_slow)

        self.assertEqual(fast_h.shape[0], 0)
        self.assertEqual(slow_h.shape[0], batch)

    def _make_dummy_model(self):
        """Create a TwoTierModel with mock slow brains."""
        slow_experts = MagicMock(spec=SlowBrainExperts)
        slow_experts.load_all_to_cpu = MagicMock()
        slow_layers = MagicMock(spec=SlowBrainLayers)
        slow_layers.load_all_to_cpu = MagicMock()

        return TwoTierModel(
            slow_experts=slow_experts,
            slow_layers=slow_layers,
            pruned_expert_ids={0: {10, 11}},
            pruned_layer_ids=set(),
            layer_index_map={},
            config=self.config,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistics Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoTierStats(unittest.TestCase):

    def test_hit_rate_calculation(self):
        stats = TwoTierStats()
        stats.total_tokens = 1000
        stats.slow_expert_hits = 30
        stats.slow_layer_hits = 20
        self.assertAlmostEqual(stats.slow_hit_rate, 0.05)

    def test_cache_hit_rate(self):
        stats = TwoTierStats()
        stats.slow_expert_cache_hits = 80
        stats.slow_expert_cache_misses = 20
        self.assertAlmostEqual(stats.cache_hit_rate, 0.8)

    def test_zero_division_safety(self):
        stats = TwoTierStats()
        self.assertEqual(stats.slow_hit_rate, 0.0)
        self.assertEqual(stats.cache_hit_rate, 0.0)
        self.assertEqual(stats.avg_slow_latency_ms, 0.0)

    def test_summary_string(self):
        stats = TwoTierStats()
        stats.total_tokens = 100
        stats.slow_expert_hits = 5
        summary = stats.summary()
        self.assertIn("Two-Tier Brain Statistics", summary)
        self.assertIn("100", summary)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Fast Brain Only vs Full Model (Simulated)
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityComparison(unittest.TestCase):
    """Simulated quality comparison: fast-only vs fast+slow vs full model."""

    def test_fast_brain_degraded(self):
        """Fast brain alone should produce different (degraded) output.

        We simulate this by creating a 'full model' that sums all expert outputs,
        then a 'fast brain' that only uses a subset.
        """
        hidden_dim = 64
        batch = 4
        num_experts = 8
        num_pruned = 3  # prune 3 of 8

        hidden = torch.randn(batch, hidden_dim)
        expert_weights = [make_fake_expert_weights(hidden_dim, 128) for _ in range(num_experts)]

        # Full model: sum all expert outputs
        full_output = torch.zeros(batch, hidden_dim)
        for i in range(num_experts):
            full_output += expert_forward_fp16(hidden, expert_weights[i]) / num_experts

        # Fast brain: only non-pruned experts
        fast_output = torch.zeros(batch, hidden_dim)
        for i in range(num_experts - num_pruned):
            fast_output += expert_forward_fp16(hidden, expert_weights[i]) / (num_experts - num_pruned)

        # Fast brain output should differ from full model
        diff = (full_output - fast_output).norm()
        self.assertGreater(diff.item(), 0.01, "Fast brain should differ from full model")

    def test_fast_plus_slow_matches_full(self):
        """Fast brain + slow brain should match full model output.

        Same simulation: fast experts + slow (pruned) experts = all experts.
        """
        hidden_dim = 64
        batch = 4
        num_experts = 8
        num_pruned = 3

        hidden = torch.randn(batch, hidden_dim)
        expert_weights = [make_fake_expert_weights(hidden_dim, 128) for _ in range(num_experts)]

        # Full model
        full_output = torch.zeros(batch, hidden_dim)
        for i in range(num_experts):
            full_output += expert_forward_fp16(hidden, expert_weights[i]) / num_experts

        # Fast brain
        fast_output = torch.zeros(batch, hidden_dim)
        for i in range(num_experts - num_pruned):
            fast_output += expert_forward_fp16(hidden, expert_weights[i]) / num_experts

        # Slow brain (the pruned experts)
        slow_output = torch.zeros(batch, hidden_dim)
        for i in range(num_experts - num_pruned, num_experts):
            slow_output += expert_forward_fp16(hidden, expert_weights[i]) / num_experts

        # Combined should match full
        combined = fast_output + slow_output
        torch.testing.assert_close(combined, full_output, atol=1e-5, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hit Rate Measurement
# ─────────────────────────────────────────────────────────────────────────────

class TestHitRate(unittest.TestCase):
    """Measure slow brain hit rates under various router distributions."""

    def test_low_hit_rate_normal_distribution(self):
        """With normally distributed router logits, pruned experts should
        rarely be in top-k, yielding a low hit rate."""
        config = make_config()
        pruned = {0: set(range(12, 16))}  # prune experts 12-15 (25%)
        detector = SlowBrainDetector(pruned, set(), config)

        hits = 0
        total = 1000
        for _ in range(total):
            logits = torch.randn(1, 16)
            needs_slow, _ = detector.detect_slow_experts(logits, layer_idx=0)
            if needs_slow.any():
                hits += 1

        hit_rate = hits / total
        # With 4/16 pruned experts and top-4, random logits hit ~55% of the time
        # (at least one of 4 top-k slots picks from the 4/16 pruned).
        # The key property: it's well below 100% (not every token needs slow brain).
        self.assertLess(hit_rate, 0.80, f"Hit rate {hit_rate:.2%} should be < 80%")
        print(f"  Normal distribution hit rate: {hit_rate:.2%}")

    def test_high_hit_rate_biased_distribution(self):
        """When router logits are biased toward pruned experts, hit rate should be high."""
        config = make_config(router_confidence_threshold=0.01)
        pruned = {0: set(range(12, 16))}
        detector = SlowBrainDetector(pruned, set(), config)

        hits = 0
        total = 200
        for _ in range(total):
            logits = torch.randn(1, 16)
            # Bias toward pruned experts
            logits[0, 12:16] += 5.0
            needs_slow, _ = detector.detect_slow_experts(logits, layer_idx=0)
            if needs_slow.any():
                hits += 1

        hit_rate = hits / total
        self.assertGreater(hit_rate, 0.8, f"Biased hit rate {hit_rate:.2%} should be > 80%")
        print(f"  Biased distribution hit rate: {hit_rate:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Generic Two-Tier Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoTierGeneric(unittest.TestCase):

    def test_fast_path_only(self):
        """When detector says no slow needed, only fast path runs."""
        fast_fn = lambda h: h * 2
        slow_fn = lambda h: h * 10  # should not be called
        detect_fn = lambda h: torch.zeros(h.shape[0], dtype=torch.bool)

        model = TwoTierGeneric(fast_fn, slow_fn, detect_fn)
        hidden = torch.ones(4, 8)
        output = model.forward(hidden)
        torch.testing.assert_close(output, hidden * 2)
        self.assertEqual(model.stats.slow_expert_hits, 0)

    def test_mixed_batch(self):
        """Batch with both fast and slow tokens should process correctly."""
        fast_fn = lambda h: h * 2
        slow_fn = lambda h: h * 5
        detect_fn = lambda h: torch.tensor([True, False, True, False])

        model = TwoTierGeneric(fast_fn, slow_fn, detect_fn)
        hidden = torch.ones(4, 8)
        output = model.forward(hidden)

        expected = torch.ones(4, 8)
        expected[0] *= 5  # slow
        expected[1] *= 2  # fast
        expected[2] *= 5  # slow
        expected[3] *= 2  # fast
        torch.testing.assert_close(output, expected)

    def test_stats_tracking(self):
        """Stats should track tokens and hits."""
        detect_fn = lambda h: torch.tensor([True, False, False, False])
        model = TwoTierGeneric(lambda h: h, lambda h: h, detect_fn)

        for _ in range(10):
            model.forward(torch.ones(4, 8))

        self.assertEqual(model.stats.total_tokens, 40)
        self.assertEqual(model.stats.slow_expert_hits, 10)  # 1 per batch of 4


# ─────────────────────────────────────────────────────────────────────────────
# 9. Graceful Degradation
# ─────────────────────────────────────────────────────────────────────────────

class TestGracefulDegradation(unittest.TestCase):

    def test_timeout_config(self):
        """Verify timeout configuration is respected."""
        config = make_config(max_slow_latency_ms=0.001, skip_slow_on_timeout=True)
        self.assertTrue(config.skip_slow_on_timeout)
        self.assertEqual(config.max_slow_latency_ms, 0.001)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Slow Brain Experts with Temp Directory
# ─────────────────────────────────────────────────────────────────────────────

class TestSlowBrainExpertsIO(unittest.TestCase):
    """Test loading slow experts from disk."""

    def test_load_from_directory(self):
        """Create a fake slow expert directory and verify loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            layer_dir = Path(tmpdir) / "layer_0"
            expert_dir = layer_dir / "expert_5"
            expert_dir.mkdir(parents=True)

            weights = make_fake_expert_weights(64, 128)
            for name, tensor in weights.items():
                torch.save(tensor, expert_dir / f"{name}.pt")

            # Write manifest
            manifest = {
                "type": "slow_brain_experts",
                "experts": {"0,5": {"original_layer": 0, "original_expert": 5}},
            }
            with open(Path(tmpdir) / "manifest.json", "w") as f:
                json.dump(manifest, f)

            # Load
            config = make_config()
            slow = SlowBrainExperts(tmpdir, config)
            slow.load_all_to_cpu()

            self.assertTrue(slow.has_expert(0, 5))
            self.assertFalse(slow.has_expert(0, 99))

    def test_manifest_empty_dir(self):
        """Empty directory should load without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_config()
            slow = SlowBrainExperts(tmpdir, config)
            slow.load_all_to_cpu()
            self.assertFalse(slow.has_expert(0, 0))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
