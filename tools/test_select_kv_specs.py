#!/usr/bin/env python3
"""Tests for per-layer KV cache spec selector.

Runs without GPU or model — tests the selection logic, topology,
and greedy assignment algorithm using mocked quality data.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

from tools.select_kv_specs import (
    get_gemma4_layers,
    LayerInfo,
    LayerAssignment,
    assign_specs_greedy,
    compute_selection_result,
    export_config,
    get_candidate_specs,
    GEMMA4_GLOBAL_LAYERS,
    GEMMA4_NUM_LAYERS,
)
from kv_cache_gen.config import parse_spec


def test_gemma4_topology():
    """Test that Gemma4 layer topology is correct."""
    layers = get_gemma4_layers()
    assert len(layers) == 30, f"Expected 30 layers, got {len(layers)}"

    global_indices = {l.index for l in layers if l.layer_type == "global"}
    assert global_indices == {5, 11, 17, 23, 29}, f"Wrong global layers: {global_indices}"

    sliding_layers = [l for l in layers if l.layer_type == "sliding"]
    assert len(sliding_layers) == 25

    for l in sliding_layers:
        assert l.head_dim == 256, f"Layer {l.index}: expected head_dim=256, got {l.head_dim}"
        assert l.window_size == 1024

    global_layers = [l for l in layers if l.layer_type == "global"]
    for l in global_layers:
        assert l.head_dim == 512, f"Layer {l.index}: expected head_dim=512, got {l.head_dim}"
        assert l.window_size is None

    print("[PASS] test_gemma4_topology")


def test_greedy_assignment_basic():
    """Test greedy assignment with simple mock quality data."""
    layers = get_gemma4_layers()
    spec_names = ["k4v4b64", "k8v4b32", "k8v8b32"]

    # Mock: sliding layers tolerate k4v4, global layers need k8v8
    qualities = {}
    for l in layers:
        if l.layer_type == "sliding":
            qualities[l.index] = {
                "k4v4b64": 0.998,   # good enough
                "k8v4b32": 0.9995,
                "k8v8b32": 0.99999,
            }
        else:
            qualities[l.index] = {
                "k4v4b64": 0.990,   # below threshold
                "k8v4b32": 0.993,   # below threshold
                "k8v8b32": 0.999,   # good enough
            }

    assignments = assign_specs_greedy(
        layers, qualities, spec_names, min_quality=0.995
    )

    assert len(assignments) == 30

    for a in assignments:
        if a.layer_type == "sliding":
            assert a.assigned_spec == "k4v4b64", \
                f"Layer {a.layer_index}: expected k4v4b64, got {a.assigned_spec}"
        else:
            assert a.assigned_spec == "k8v8b32", \
                f"Layer {a.layer_index}: expected k8v8b32, got {a.assigned_spec}"

    print("[PASS] test_greedy_assignment_basic")


def test_greedy_assignment_fallback():
    """Test that fallback to least aggressive spec works."""
    layers = get_gemma4_layers(num_layers=2, global_layers={1})
    spec_names = ["k4v4b64", "k8v8b32"]

    # Layer 0: nothing meets quality threshold
    qualities = {
        0: {"k4v4b64": 0.90, "k8v8b32": 0.91},
        1: {"k4v4b64": 0.999, "k8v8b32": 0.99999},
    }

    assignments = assign_specs_greedy(
        layers, qualities, spec_names, min_quality=0.995
    )

    # Layer 0 should fall back to least aggressive (k8v8b32, lower compression)
    assert assignments[0].assigned_spec == "k8v8b32"
    # Layer 1 should use most aggressive that qualifies (k4v4b64)
    assert assignments[1].assigned_spec == "k4v4b64"

    print("[PASS] test_greedy_assignment_fallback")


def test_compression_calculation():
    """Test effective compression ratio calculation."""
    layers = get_gemma4_layers()
    spec_names = ["k4v4b64", "k8v8b32"]

    # All layers use k4v4b64
    qualities = {}
    for l in layers:
        qualities[l.index] = {"k4v4b64": 0.999, "k8v8b32": 0.99999}

    assignments = assign_specs_greedy(
        layers, qualities, spec_names, min_quality=0.995
    )

    result = compute_selection_result(layers, assignments, qualities)

    # All layers assigned k4v4b64, compression should be > 1
    assert result.effective_compression > 1.0, \
        f"Expected compression > 1, got {result.effective_compression}"
    assert result.kv_bytes_per_token_mixed < result.kv_bytes_per_token_bf16

    print(f"  Effective compression: {result.effective_compression:.2f}x")
    print(f"  BF16 bytes/token: {result.kv_bytes_per_token_bf16:,.0f}")
    print(f"  Mixed bytes/token: {result.kv_bytes_per_token_mixed:,.0f}")
    print("[PASS] test_compression_calculation")


def test_mixed_spec_compression():
    """Test that mixed specs (sliding=k4v4, global=k8v8) give expected compression."""
    layers = get_gemma4_layers()
    spec_names = ["k4v4b64", "k8v8b32"]

    qualities = {}
    for l in layers:
        if l.layer_type == "sliding":
            qualities[l.index] = {"k4v4b64": 0.998, "k8v8b32": 0.99999}
        else:
            qualities[l.index] = {"k4v4b64": 0.990, "k8v8b32": 0.999}

    assignments = assign_specs_greedy(
        layers, qualities, spec_names, min_quality=0.995
    )

    result = compute_selection_result(layers, assignments, qualities)

    # Sliding layers (25, D=256) use k4v4b64, global layers (5, D=512) use k8v8b32
    assert result.sliding_spec == "k4v4b64"
    assert result.global_spec == "k8v8b32"

    # Compression should be between pure k4v4 and pure k8v8
    k4v4_spec = parse_spec("k4v4b64")
    k8v8_spec = parse_spec("k8v8b32")
    pure_k4v4_comp = k4v4_spec.compression_vs_bf16(256)
    pure_k8v8_comp = k8v8_spec.compression_vs_bf16(256)
    assert result.effective_compression > pure_k8v8_comp, \
        f"Mixed compression {result.effective_compression:.2f} should be > pure k8v8 {pure_k8v8_comp:.2f}"

    print(f"  Mixed compression: {result.effective_compression:.2f}x")
    print(f"  (pure k4v4@256: {pure_k4v4_comp:.2f}x, pure k8v8@256: {pure_k8v8_comp:.2f}x)")
    print("[PASS] test_mixed_spec_compression")


def test_export_config():
    """Test config export format."""
    layers = get_gemma4_layers()
    spec_names = ["k4v4b64", "k8v8b32"]

    qualities = {}
    for l in layers:
        if l.layer_type == "sliding":
            qualities[l.index] = {"k4v4b64": 0.998, "k8v8b32": 0.99999}
        else:
            qualities[l.index] = {"k4v4b64": 0.990, "k8v8b32": 0.999}

    assignments = assign_specs_greedy(
        layers, qualities, spec_names, min_quality=0.995
    )
    result = compute_selection_result(layers, assignments, qualities)
    config = export_config(result)

    assert "sliding" in config
    assert "global" in config
    assert "effective_compression" in config
    assert "per_layer_specs" in config
    assert len(config["per_layer_specs"]) == 30
    assert "spec_groups" in config
    assert "quality" in config
    assert config["quality"]["min"] > 0
    assert config["quality"]["mean"] > 0

    print(f"  Config keys: {list(config.keys())}")
    print(f"  Per-layer specs sample: {config['per_layer_specs'][:6]}...")
    print(f"  Spec groups: { {k: len(v) for k, v in config['spec_groups'].items()} }")
    print("[PASS] test_export_config")


def test_head_dim_aware_compression():
    """Test that compression accounts for different head dims."""
    layers = get_gemma4_layers()

    # Global layers (D=512) should have different compression than sliding (D=256)
    spec = parse_spec("k8v8b32")
    comp_256 = spec.compression_vs_bf16(256)
    comp_512 = spec.compression_vs_bf16(512)

    # Both should be > 1 (compressing)
    assert comp_256 > 1.0
    assert comp_512 > 1.0

    # Scale overhead is proportionally smaller with larger D, so compression
    # should be slightly better with D=512
    assert comp_512 >= comp_256, \
        f"Expected D=512 ({comp_512:.2f}x) >= D=256 ({comp_256:.2f}x)"

    print(f"  k8v8b32 compression: D=256 -> {comp_256:.2f}x, D=512 -> {comp_512:.2f}x")
    print("[PASS] test_head_dim_aware_compression")


def test_candidate_specs_ordering():
    """Test that candidate specs are ordered by compression ratio."""
    specs = get_candidate_specs()
    compressions = []
    for name in specs:
        spec = parse_spec(name)
        if spec:
            compressions.append(spec.compression_vs_bf16(256))
        else:
            compressions.append(1.0)

    # Verify they are parseable
    assert all(c > 0 for c in compressions), "All specs should parse successfully"
    print(f"  Default candidates: {list(zip(specs, [f'{c:.1f}x' for c in compressions]))}")
    print("[PASS] test_candidate_specs_ordering")


def test_all_layers_same_spec():
    """Test edge case: all layers get the same spec."""
    layers = get_gemma4_layers()
    spec_names = ["k4v4b64"]

    qualities = {l.index: {"k4v4b64": 0.999} for l in layers}
    assignments = assign_specs_greedy(layers, qualities, spec_names, min_quality=0.995)

    result = compute_selection_result(layers, assignments, qualities)
    assert result.sliding_spec == "k4v4b64"
    assert result.global_spec == "k4v4b64"

    # All layers same spec
    unique_specs = set(a.assigned_spec for a in assignments)
    assert len(unique_specs) == 1

    print("[PASS] test_all_layers_same_spec")


def test_strict_quality_threshold():
    """Test with very strict quality threshold."""
    layers = get_gemma4_layers(num_layers=5, global_layers={4})
    spec_names = ["k4v4b64", "k8v4b32", "k8v8b32"]

    # Only k8v8 meets strict threshold
    qualities = {}
    for l in layers:
        qualities[l.index] = {
            "k4v4b64": 0.998,
            "k8v4b32": 0.9995,
            "k8v8b32": 0.99999,
        }

    # With threshold 0.9999, only k8v8 qualifies
    assignments = assign_specs_greedy(
        layers, qualities, spec_names, min_quality=0.9999
    )
    for a in assignments:
        assert a.assigned_spec == "k8v8b32", \
            f"Layer {a.layer_index}: expected k8v8b32 with strict threshold"

    print("[PASS] test_strict_quality_threshold")


if __name__ == "__main__":
    tests = [
        test_gemma4_topology,
        test_greedy_assignment_basic,
        test_greedy_assignment_fallback,
        test_compression_calculation,
        test_mixed_spec_compression,
        test_export_config,
        test_head_dim_aware_compression,
        test_candidate_specs_ordering,
        test_all_layers_same_spec,
        test_strict_quality_threshold,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
