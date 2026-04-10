"""Tests for auto_config.py profiler logic.

Validates config generation, selection logic, and output generation
without launching Docker containers.

Usage:
    python3 tools/auto_config_test.py
    python3 -m pytest tools/auto_config_test.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure tools/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from auto_config import (
    GPUInfo,
    ModelInfo,
    ConfigCandidate,
    BenchResult,
    detect_gpu_mock,
    detect_model_mock,
    generate_configs,
    select_optimal,
    config_to_cli_flags,
    config_to_env_vars,
    generate_serve_script,
    mock_profile_config,
    print_results_table,
    build_docker_cmd,
    MOCK_RESULTS,
)


# ---------------------------------------------------------------------------
# GPU detection tests
# ---------------------------------------------------------------------------

def test_gpu_info_properties():
    gpu = GPUInfo(name="RTX 5090", vram_mb=32768, compute_capability="12.0")
    assert gpu.vram_gb == 32.0
    assert gpu.supports_fp8 is True
    assert gpu.supports_fp4 is True

    gpu_4090 = GPUInfo(name="RTX 4090", vram_mb=24576, compute_capability="8.9")
    assert gpu_4090.supports_fp8 is False  # SM89 < 9
    assert gpu_4090.supports_fp4 is False

    gpu_h100 = GPUInfo(name="H100", vram_mb=81920, compute_capability="9.0")
    assert gpu_h100.supports_fp8 is True
    assert gpu_h100.supports_fp4 is False  # SM90 < 12


def test_gpu_to_dict():
    gpu = detect_gpu_mock()
    d = gpu.to_dict()
    assert d["name"] == "NVIDIA GeForce RTX 5090"
    assert d["vram_mb"] == 32607
    assert d["supports_fp8"] is True
    assert "vram_gb" in d


def test_detect_gpu_mock():
    gpu = detect_gpu_mock()
    assert "5090" in gpu.name
    assert gpu.vram_mb > 30000
    assert gpu.compute_capability == "12.0"
    assert gpu.bandwidth_gbps > 0


# ---------------------------------------------------------------------------
# Model detection tests
# ---------------------------------------------------------------------------

def test_detect_model_mock():
    model = detect_model_mock("/models/test", "modelopt")
    assert model.num_layers == 34
    assert model.architecture == "moe"
    assert model.quantization == "modelopt"
    assert model.num_params_b == 26.0


def test_model_to_dict():
    model = detect_model_mock("/models/test", "awq")
    d = model.to_dict()
    assert d["quantization"] == "awq"
    assert d["num_layers"] == 34
    assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# Config generation tests
# ---------------------------------------------------------------------------

def test_generate_configs_basic():
    gpu = detect_gpu_mock()
    model = detect_model_mock("/models/test", "modelopt")
    configs = generate_configs(gpu, model)

    assert len(configs) >= 6
    names = [c.name for c in configs]

    # Must have baseline
    assert "eager_baseline" in names
    # Must have cudagraph variants
    assert "cudagraph_no_inductor" in names
    assert "cudagraph_inductor" in names


def test_generate_configs_fp8_on_capable_gpu():
    gpu = GPUInfo(name="RTX 5090", vram_mb=32768, compute_capability="12.0")
    model = detect_model_mock("/models/test", "modelopt")
    configs = generate_configs(gpu, model)
    names = [c.name for c in configs]

    assert "cudagraph_fp8kv" in names
    assert "inductor_fp8kv" in names


def test_generate_configs_no_fp8_on_old_gpu():
    gpu = GPUInfo(name="RTX 4090", vram_mb=24576, compute_capability="8.9")
    model = detect_model_mock("/models/test", "modelopt")
    configs = generate_configs(gpu, model)
    names = [c.name for c in configs]

    assert "cudagraph_fp8kv" not in names
    assert "inductor_fp8kv" not in names


def test_config_max_model_len_from_model():
    gpu = detect_gpu_mock()
    model = ModelInfo(max_model_len=2048)
    configs = generate_configs(gpu, model)

    for c in configs:
        assert c.max_model_len == 2048


def test_config_properties():
    """Verify eager baseline has correct flags."""
    gpu = detect_gpu_mock()
    model = detect_model_mock("/models/test", "none")
    configs = generate_configs(gpu, model)

    eager = next(c for c in configs if c.name == "eager_baseline")
    assert eager.enforce_eager is True
    assert eager.cuda_graph is False
    assert eager.inductor is False

    inductor = next(c for c in configs if c.name == "cudagraph_inductor")
    assert inductor.enforce_eager is False
    assert inductor.cuda_graph is True
    assert inductor.inductor is True


def test_memory_utilization_variants():
    gpu = detect_gpu_mock()
    model = detect_model_mock("/models/test", "none")
    configs = generate_configs(gpu, model)

    mem_values = set(c.gpu_memory_utilization for c in configs)
    assert 0.85 in mem_values
    assert 0.90 in mem_values
    assert 0.92 in mem_values


# ---------------------------------------------------------------------------
# Selection logic tests
# ---------------------------------------------------------------------------

def test_select_latency():
    results = [
        BenchResult(config_name="slow", latency_tok_s=10, throughput_tok_s=5000),
        BenchResult(config_name="fast_lat", latency_tok_s=100, throughput_tok_s=2000),
        BenchResult(config_name="fast_thr", latency_tok_s=50, throughput_tok_s=8000),
    ]
    best, reason = select_optimal(results, "latency")
    assert best.config_name == "fast_lat"
    assert "100.0" in reason


def test_select_throughput():
    results = [
        BenchResult(config_name="slow", latency_tok_s=10, throughput_tok_s=5000),
        BenchResult(config_name="fast_lat", latency_tok_s=100, throughput_tok_s=2000),
        BenchResult(config_name="fast_thr", latency_tok_s=50, throughput_tok_s=8000),
    ]
    best, reason = select_optimal(results, "throughput")
    assert best.config_name == "fast_thr"
    assert "8000.0" in reason


def test_select_balanced():
    results = [
        BenchResult(config_name="balanced", latency_tok_s=80, throughput_tok_s=7000),
        BenchResult(config_name="lat_only", latency_tok_s=100, throughput_tok_s=1000),
        BenchResult(config_name="thr_only", latency_tok_s=10, throughput_tok_s=8000),
    ]
    best, reason = select_optimal(results, "balanced")
    # "balanced" should win: 80/100*0.5 + 7000/8000*0.5 = 0.4 + 0.4375 = 0.8375
    # vs lat_only: 1.0*0.5 + 1000/8000*0.5 = 0.5 + 0.0625 = 0.5625
    # vs thr_only: 10/100*0.5 + 1.0*0.5 = 0.05 + 0.5 = 0.55
    assert best.config_name == "balanced"


def test_select_skips_errors():
    results = [
        BenchResult(config_name="errored", latency_tok_s=999, throughput_tok_s=999,
                    error="Server crashed"),
        BenchResult(config_name="ok", latency_tok_s=50, throughput_tok_s=3000),
    ]
    best, reason = select_optimal(results, "throughput")
    assert best.config_name == "ok"


def test_select_skips_skipped():
    results = [
        BenchResult(config_name="skipped", latency_tok_s=999, throughput_tok_s=999,
                    skipped=True),
        BenchResult(config_name="ok", latency_tok_s=50, throughput_tok_s=3000),
    ]
    best, reason = select_optimal(results, "latency")
    assert best.config_name == "ok"


def test_select_all_errors():
    results = [
        BenchResult(config_name="err1", error="fail"),
        BenchResult(config_name="err2", error="fail"),
    ]
    best, reason = select_optimal(results, "throughput")
    assert "No valid" in reason


def test_select_empty():
    best, reason = select_optimal([], "throughput")
    assert best.config_name == ""


# ---------------------------------------------------------------------------
# Mock profiling tests
# ---------------------------------------------------------------------------

def test_mock_profile_known_config():
    config = ConfigCandidate(name="cudagraph_inductor")
    result = mock_profile_config(config)
    assert result.config_name == "cudagraph_inductor"
    assert result.latency_tok_s == 127.0
    assert result.throughput_tok_s == 3112.0
    assert result.startup_time_s > 0


def test_mock_profile_unknown_config():
    config = ConfigCandidate(name="totally_new_config")
    result = mock_profile_config(config)
    assert result.config_name == "totally_new_config"
    assert result.latency_tok_s > 0
    assert result.throughput_tok_s > 0


def test_mock_results_consistency():
    """Verify mock data reflects known empirical results."""
    eager = MOCK_RESULTS["eager_baseline"]
    cudagraph = MOCK_RESULTS["cudagraph_no_inductor"]
    inductor = MOCK_RESULTS["cudagraph_inductor"]

    # CUDA graphs should be much faster than eager
    assert cudagraph.latency_tok_s > eager.latency_tok_s * 3

    # Inductor should have higher single-request latency than cudagraph
    assert inductor.latency_tok_s > cudagraph.latency_tok_s

    # Cudagraph (no inductor) should have higher throughput than inductor
    assert cudagraph.throughput_tok_s > inductor.throughput_tok_s

    # FP8 KV should have 2x capacity
    fp8 = MOCK_RESULTS["cudagraph_fp8kv"]
    assert fp8.kv_capacity == eager.kv_capacity * 2


# ---------------------------------------------------------------------------
# CLI flag generation tests
# ---------------------------------------------------------------------------

def test_cli_flags_basic():
    config = ConfigCandidate(
        name="test", enforce_eager=False, kv_cache_dtype="auto",
        gpu_memory_utilization=0.90, max_model_len=4096,
    )
    flags = config_to_cli_flags(config, "/models/test-model", "modelopt")
    assert "--model /models/test-model" in flags
    assert "--quantization modelopt" in flags
    assert "--gpu-memory-utilization 0.9" in flags
    assert "--enforce-eager" not in flags


def test_cli_flags_eager():
    config = ConfigCandidate(name="test", enforce_eager=True)
    flags = config_to_cli_flags(config, "/models/test", "none")
    assert "--enforce-eager" in flags
    assert "--quantization" not in flags


def test_cli_flags_fp8kv():
    config = ConfigCandidate(name="test", kv_cache_dtype="fp8_e5m2")
    flags = config_to_cli_flags(config, "/models/test", "none")
    assert "--kv-cache-dtype fp8_e5m2" in flags


def test_env_vars_inductor():
    config = ConfigCandidate(name="test", inductor=True)
    env = config_to_env_vars(config)
    assert env["VLLM_TORCH_COMPILE_LEVEL"] == "1"


def test_env_vars_no_inductor():
    config = ConfigCandidate(name="test", inductor=False)
    env = config_to_env_vars(config)
    assert "VLLM_TORCH_COMPILE_LEVEL" not in env


def test_env_vars_extra():
    config = ConfigCandidate(name="test", inductor=True,
                             extra_env={"CUDA_VISIBLE_DEVICES": "0"})
    env = config_to_env_vars(config)
    assert env["VLLM_TORCH_COMPILE_LEVEL"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


# ---------------------------------------------------------------------------
# Docker command generation tests
# ---------------------------------------------------------------------------

def test_build_docker_cmd_basic():
    config = ConfigCandidate(
        name="test", enforce_eager=True, inductor=False,
        kv_cache_dtype="auto", gpu_memory_utilization=0.90,
        max_model_len=4096,
    )
    cmd = build_docker_cmd(
        config, "/models/test-model", "modelopt",
        "vllm-built", "test_container", 8100, "/root/models",
    )
    assert "docker" in cmd[0]
    assert "--name" in cmd
    assert "test_container" in cmd
    assert "--enforce-eager" in cmd
    assert "--quantization" in cmd
    assert "modelopt" in cmd


def test_build_docker_cmd_inductor_env():
    config = ConfigCandidate(name="test", inductor=True, enforce_eager=False)
    cmd = build_docker_cmd(
        config, "/models/test", "none",
        "vllm-built", "test_c", 8100, "/root/models",
    )
    # Check env is set
    env_idx = cmd.index("VLLM_TORCH_COMPILE_LEVEL=1") - 1
    assert cmd[env_idx] == "-e"


def test_build_docker_cmd_fp8kv():
    config = ConfigCandidate(name="test", kv_cache_dtype="fp8_e5m2")
    cmd = build_docker_cmd(
        config, "/models/test", "none",
        "vllm-built", "test_c", 8100, "/root/models",
    )
    assert "--kv-cache-dtype" in cmd
    assert "fp8_e5m2" in cmd


def test_build_docker_cmd_no_quant():
    config = ConfigCandidate(name="test")
    cmd = build_docker_cmd(
        config, "/models/test", "none",
        "vllm-built", "test_c", 8100, "/root/models",
    )
    assert "--quantization" not in cmd


# ---------------------------------------------------------------------------
# Serve script generation tests
# ---------------------------------------------------------------------------

def test_generate_serve_script():
    config = ConfigCandidate(
        name="cudagraph_inductor", inductor=True,
        gpu_memory_utilization=0.90, max_model_len=4096,
    )
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="w") as f:
        tmppath = f.name

    try:
        generate_serve_script(
            config, "/models/test-model", "modelopt",
            "vllm-built", "/root/models", tmppath,
        )
        with open(tmppath) as f:
            content = f.read()

        assert "#!/bin/bash" in content
        assert "cudagraph_inductor" in content
        assert "VLLM_TORCH_COMPILE_LEVEL=1" in content
        assert "--quantization modelopt" in content
        assert "docker run" in content
        assert "health" in content.lower()

        # Check executable
        assert os.access(tmppath, os.X_OK)
    finally:
        os.unlink(tmppath)


def test_serve_script_no_inductor():
    config = ConfigCandidate(name="eager", enforce_eager=True, inductor=False)
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
        tmppath = f.name

    try:
        generate_serve_script(
            config, "/models/test", "none",
            "vllm-built", "/root/models", tmppath,
        )
        with open(tmppath) as f:
            content = f.read()

        assert "VLLM_TORCH_COMPILE_LEVEL" not in content
        assert "--enforce-eager" in content
        assert "--quantization" not in content
    finally:
        os.unlink(tmppath)


# ---------------------------------------------------------------------------
# JSON output tests
# ---------------------------------------------------------------------------

def test_full_json_output():
    """Simulate a full run and verify JSON structure."""
    gpu = detect_gpu_mock()
    model = detect_model_mock("/models/test", "modelopt")
    configs = generate_configs(gpu, model)

    results = [mock_profile_config(c) for c in configs]
    best, reason = select_optimal(results, "throughput")
    best_config = next(c for c in configs if c.name == best.config_name)

    output = {
        "target": "throughput",
        "gpu": gpu.to_dict(),
        "model": model.to_dict(),
        "results": [r.to_dict() for r in results],
        "recommended": {
            "config_name": best.config_name,
            "config": best_config.to_dict(),
            "reason": reason,
        },
    }

    # Verify serializable
    json_str = json.dumps(output, indent=2)
    parsed = json.loads(json_str)

    assert parsed["target"] == "throughput"
    assert len(parsed["results"]) == len(configs)
    assert parsed["recommended"]["config_name"] == best.config_name
    assert parsed["gpu"]["name"] == "NVIDIA GeForce RTX 5090"


# ---------------------------------------------------------------------------
# Integration: end-to-end mock test
# ---------------------------------------------------------------------------

def test_end_to_end_mock():
    """Full pipeline in mock mode."""
    gpu = detect_gpu_mock()
    model = detect_model_mock("/models/gemma-test", "modelopt")
    configs = generate_configs(gpu, model)

    assert len(configs) >= 6

    results = []
    for config in configs:
        result = mock_profile_config(config)
        results.append(result)

    # Test all three targets
    for target in ["latency", "throughput", "balanced"]:
        best, reason = select_optimal(results, target)
        assert best.config_name != ""
        assert not best.error
        assert best.latency_tok_s > 0
        assert best.throughput_tok_s > 0

    # Verify latency winner
    best_lat, _ = select_optimal(results, "latency")
    assert best_lat.config_name == "cudagraph_inductor"  # 127 tok/s

    # Verify throughput winner
    best_thr, _ = select_optimal(results, "throughput")
    assert best_thr.config_name == "cudagraph_highmem"  # 6800 tok/s


def test_results_table_no_crash():
    """Verify print_results_table doesn't crash."""
    results = [mock_profile_config(ConfigCandidate(name=n)) for n in MOCK_RESULTS]
    results.append(BenchResult(config_name="errored", error="fail"))
    results.append(BenchResult(config_name="skipped", skipped=True))
    # Should not raise
    print_results_table(results)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    """Simple test runner - runs all test_* functions."""
    import traceback

    test_funcs = [
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    test_funcs.sort(key=lambda x: x[0])

    passed = 0
    failed = 0
    errors = []

    print(f"\nRunning {len(test_funcs)} tests...\n")

    for name, func in test_funcs:
        try:
            func()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, tb))
            print(f"  FAIL  {name}: {e}")

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")

    if errors:
        print(f"\nFailures:")
        for name, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
