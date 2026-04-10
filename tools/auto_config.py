"""Auto-Config Profiler for vLLM serving optimization.

Profiles multiple serving configurations and selects the best one
for the detected GPU, model, and target workload.

Usage:
    python3 tools/auto_config.py --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
        --quantization modelopt --target throughput

    python3 tools/auto_config.py --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
        --quantization modelopt --target latency

    # Dry run (no Docker, just show configs):
    python3 tools/auto_config.py --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
        --quantization modelopt --dry-run

    # Mock mode for testing:
    python3 tools/auto_config.py --model /models/test-model --mock
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    name: str = "Unknown"
    vram_mb: int = 0
    compute_capability: str = "0.0"
    bandwidth_gbps: float = 0.0

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024

    @property
    def supports_fp8(self) -> bool:
        major = int(self.compute_capability.split(".")[0])
        return major >= 9  # Hopper+ or Blackwell (SM89/90/100+)

    @property
    def supports_fp4(self) -> bool:
        major = int(self.compute_capability.split(".")[0])
        return major >= 12  # Blackwell SM100+

    def to_dict(self):
        return {
            "name": self.name,
            "vram_mb": self.vram_mb,
            "vram_gb": round(self.vram_gb, 1),
            "compute_capability": self.compute_capability,
            "bandwidth_gbps": self.bandwidth_gbps,
            "supports_fp8": self.supports_fp8,
            "supports_fp4": self.supports_fp4,
        }


@dataclass
class ModelInfo:
    path: str = ""
    name: str = ""
    architecture: str = "unknown"  # "dense" or "moe"
    num_layers: int = 0
    hidden_size: int = 0
    num_params_b: float = 0.0
    max_model_len: int = 4096
    quantization: str = "none"
    vocab_size: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ConfigCandidate:
    """A single vLLM serving configuration to test."""
    name: str
    enforce_eager: bool = False
    cuda_graph: bool = True
    inductor: bool = False  # VLLM_TORCH_COMPILE_LEVEL=1
    kv_cache_dtype: str = "auto"  # "auto" (bf16), "fp8", "fp8_e5m2", "fp8_e4m3"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    max_num_seqs: int = 256
    # extra vllm args
    extra_args: list = field(default_factory=list)
    extra_env: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchResult:
    """Benchmark result for a single configuration."""
    config_name: str = ""
    startup_time_s: float = 0.0
    # latency benchmark (C=1)
    latency_tok_s: float = 0.0
    latency_ttft_ms: float = 0.0
    # throughput benchmark (C=64)
    throughput_tok_s: float = 0.0
    throughput_req_s: float = 0.0
    # capacity
    kv_capacity: int = 0
    # errors
    error: str = ""
    skipped: bool = False

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_gpu() -> GPUInfo:
    """Detect GPU using nvidia-smi."""
    gpu = GPUInfo()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 3:
            gpu.name = parts[0]
            gpu.vram_mb = int(float(parts[1]))
            gpu.compute_capability = parts[2]
    except Exception as e:
        print(f"[WARN] nvidia-smi failed: {e}")

    # Estimate bandwidth from known GPUs
    bw_map = {
        "5090": 1568, "5080": 960, "5070": 672,
        "4090": 1008, "4080": 717, "4070": 504,
        "A100": 2039, "H100": 3352, "H200": 4800,
        "A6000": 768, "L40": 864, "B200": 8000,
    }
    for pattern, bw in bw_map.items():
        if pattern.lower() in gpu.name.lower():
            gpu.bandwidth_gbps = bw
            break

    return gpu


def detect_gpu_mock() -> GPUInfo:
    return GPUInfo(
        name="NVIDIA GeForce RTX 5090",
        vram_mb=32607,
        compute_capability="12.0",
        bandwidth_gbps=1568,
    )


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def detect_model(model_path: str, quantization: str = "none") -> ModelInfo:
    """Detect model info from config.json at model_path."""
    info = ModelInfo(path=model_path, name=Path(model_path).name, quantization=quantization)

    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        print(f"[WARN] No config.json at {config_path}")
        return info

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read config.json: {e}")
        return info

    info.architecture = "moe" if "num_local_experts" in cfg else "dense"
    info.num_layers = cfg.get("num_hidden_layers", cfg.get("num_layers", 0))
    info.hidden_size = cfg.get("hidden_size", 0)
    info.vocab_size = cfg.get("vocab_size", 0)
    info.max_model_len = min(
        cfg.get("max_position_embeddings", 8192),
        8192  # cap for profiling speed
    )

    # Estimate param count
    num_params = cfg.get("num_parameters", 0)
    if num_params == 0:
        # Rough estimate from architecture
        h = info.hidden_size
        L = info.num_layers
        V = info.vocab_size
        if h > 0 and L > 0:
            if info.architecture == "moe":
                num_experts = cfg.get("num_local_experts", 8)
                expert_size = cfg.get("intermediate_size", 4 * h)
                num_params = L * (12 * h * h + num_experts * 3 * h * expert_size) + V * h * 2
            else:
                num_params = L * 12 * h * h + V * h * 2

    info.num_params_b = round(num_params / 1e9, 1) if num_params > 0 else 0

    return info


def detect_model_mock(model_path: str, quantization: str = "none") -> ModelInfo:
    return ModelInfo(
        path=model_path,
        name="gemma-4-26B-A4B-it-NVFP4-modelopt",
        architecture="moe",
        num_layers=34,
        hidden_size=3840,
        num_params_b=26.0,
        max_model_len=4096,
        quantization=quantization,
        vocab_size=262144,
    )


# ---------------------------------------------------------------------------
# Configuration generation
# ---------------------------------------------------------------------------

def generate_configs(gpu: GPUInfo, model: ModelInfo) -> list[ConfigCandidate]:
    """Generate a set of config candidates based on hardware + model."""
    configs = []
    max_len = model.max_model_len if model.max_model_len > 0 else 4096

    # Base: enforce_eager (baseline, fastest startup)
    configs.append(ConfigCandidate(
        name="eager_baseline",
        enforce_eager=True,
        cuda_graph=False,
        inductor=False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=max_len,
    ))

    # CUDA graphs without inductor
    configs.append(ConfigCandidate(
        name="cudagraph_no_inductor",
        enforce_eager=False,
        cuda_graph=True,
        inductor=False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=max_len,
    ))

    # CUDA graphs with inductor (torch.compile)
    configs.append(ConfigCandidate(
        name="cudagraph_inductor",
        enforce_eager=False,
        cuda_graph=True,
        inductor=True,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=max_len,
    ))

    # FP8 KV cache (if GPU supports it)
    if gpu.supports_fp8:
        configs.append(ConfigCandidate(
            name="cudagraph_fp8kv",
            enforce_eager=False,
            cuda_graph=True,
            inductor=False,
            kv_cache_dtype="fp8_e5m2",
            gpu_memory_utilization=0.90,
            max_model_len=max_len,
        ))

        configs.append(ConfigCandidate(
            name="inductor_fp8kv",
            enforce_eager=False,
            cuda_graph=True,
            inductor=True,
            kv_cache_dtype="fp8_e5m2",
            gpu_memory_utilization=0.90,
            max_model_len=max_len,
        ))

    # Higher memory utilization (squeeze more KV capacity)
    configs.append(ConfigCandidate(
        name="cudagraph_highmem",
        enforce_eager=False,
        cuda_graph=True,
        inductor=False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.92,
        max_model_len=max_len,
    ))

    # Lower memory utilization (safer, avoids OOM)
    configs.append(ConfigCandidate(
        name="cudagraph_safemem",
        enforce_eager=False,
        cuda_graph=True,
        inductor=False,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.85,
        max_model_len=max_len,
    ))

    # Inductor + higher mem
    configs.append(ConfigCandidate(
        name="inductor_highmem",
        enforce_eager=False,
        cuda_graph=True,
        inductor=True,
        kv_cache_dtype="auto",
        gpu_memory_utilization=0.92,
        max_model_len=max_len,
    ))

    return configs


# ---------------------------------------------------------------------------
# Docker container management
# ---------------------------------------------------------------------------

def build_docker_cmd(
    config: ConfigCandidate,
    model_path: str,
    quantization: str,
    docker_image: str,
    container_name: str,
    port: int,
    model_mount: str,
) -> list[str]:
    """Build docker run command for a config."""
    cmd = [
        "docker", "run", "--rm", "-d",
        "--name", container_name,
        "--runtime=nvidia", "--gpus", "all",
        "-v", f"{model_mount}:/models:ro",
        "-p", f"{port}:8000",
        "--shm-size=2g",
    ]

    # Environment variables
    env = {}
    if config.inductor:
        env["VLLM_TORCH_COMPILE_LEVEL"] = "1"

    env.update(config.extra_env)

    for k, v in env.items():
        cmd.extend(["-e", f"{k}={v}"])

    # vLLM serve command
    cmd.extend([
        docker_image,
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", f"/models/{Path(model_path).name}",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ])

    if quantization != "none":
        cmd.extend(["--quantization", quantization])

    if config.enforce_eager:
        cmd.append("--enforce-eager")

    if config.kv_cache_dtype != "auto":
        cmd.extend(["--kv-cache-dtype", config.kv_cache_dtype])

    if config.max_num_seqs != 256:
        cmd.extend(["--max-num-seqs", str(config.max_num_seqs)])

    cmd.extend(config.extra_args)

    return cmd


def wait_for_health(port: int, timeout: int = 300) -> tuple[bool, float]:
    """Wait for vLLM to be ready. Returns (success, startup_seconds)."""
    import urllib.request
    import urllib.error

    t0 = time.time()
    url = f"http://localhost:{port}/health"

    while time.time() - t0 < timeout:
        try:
            req = urllib.request.urlopen(url, timeout=5)
            if req.status == 200:
                return True, time.time() - t0
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)

    return False, time.time() - t0


def stop_container(container_name: str, timeout: int = 30):
    """Stop and remove a Docker container."""
    subprocess.run(
        ["docker", "stop", "-t", str(timeout), container_name],
        capture_output=True, timeout=timeout + 10,
    )
    # --rm flag should auto-remove, but be safe
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, timeout=10,
    )


def get_kv_capacity(port: int) -> int:
    """Try to extract KV cache capacity from vLLM metrics."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=10)
        text = resp.read().decode()
        # Look for gpu_cache_usage or num_gpu_blocks
        for line in text.split("\n"):
            if "vllm:num_gpu_blocks_total" in line and not line.startswith("#"):
                # gauge value
                val = line.strip().split()[-1]
                return int(float(val))
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

BENCH_PROMPTS = [
    "What is 2+2?",
    "Explain photosynthesis in one sentence.",
    "Capital of Japan?",
    "Write a haiku about coding.",
    "What causes rain?",
    "List three prime numbers.",
    "Define machine learning briefly.",
    "Who wrote Hamlet?",
    "What is the speed of light?",
    "Explain gravity in simple terms.",
    "What is DNA?",
    "Name three programming languages.",
    "What is the boiling point of water?",
    "Describe the internet in one sentence.",
    "What year was Python created?",
    "Explain entropy in one sentence.",
    "What is an algorithm?",
    "Name the planets in order.",
    "What is a neural network?",
    "Explain the Pythagorean theorem.",
]


async def _send_request(session, url: str, model_name: str, prompt: str,
                        max_tokens: int = 64) -> dict:
    """Send one chat completion request."""
    import aiohttp
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=120)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text[:200]}",
                        "latency": time.perf_counter() - t0}
            data = await resp.json()
            latency = time.perf_counter() - t0
            usage = data.get("usage", {})
            return {
                "latency": latency,
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
            }
    except Exception as e:
        return {"error": str(e), "latency": time.perf_counter() - t0}


async def run_bench(port: int, model_name: str, concurrency: int,
                    num_requests: int = 15, max_tokens: int = 64) -> dict:
    """Run a quick benchmark at given concurrency. Returns aggregate stats."""
    import aiohttp
    url = f"http://localhost:{port}/v1/chat/completions"

    # Warmup: 2 requests serial
    async with aiohttp.ClientSession() as session:
        for i in range(2):
            await _send_request(session, url, model_name, BENCH_PROMPTS[i], max_tokens)

    # Actual benchmark
    results = []
    t0_total = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_req(prompt):
            async with sem:
                return await _send_request(session, url, model_name, prompt, max_tokens)

        tasks = [bounded_req(BENCH_PROMPTS[i % len(BENCH_PROMPTS)])
                 for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - t0_total

    # Compute stats
    errors = [r for r in results if "error" in r]
    successes = [r for r in results if "error" not in r]

    if not successes:
        return {"error": f"All {len(errors)} requests failed",
                "sample_error": errors[0]["error"] if errors else ""}

    total_tokens = sum(r["completion_tokens"] for r in successes)
    avg_latency = sum(r["latency"] for r in successes) / len(successes)
    tok_per_s = total_tokens / wall_time if wall_time > 0 else 0

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "num_success": len(successes),
        "num_errors": len(errors),
        "wall_time_s": round(wall_time, 2),
        "total_tokens": total_tokens,
        "tok_per_s": round(tok_per_s, 1),
        "avg_latency_s": round(avg_latency, 3),
        "req_per_s": round(len(successes) / wall_time, 2) if wall_time > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Mock benchmark (for testing)
# ---------------------------------------------------------------------------

MOCK_RESULTS = {
    "eager_baseline": BenchResult(
        config_name="eager_baseline", startup_time_s=15.0,
        latency_tok_s=18.0, latency_ttft_ms=85.0,
        throughput_tok_s=450.0, throughput_req_s=7.0,
        kv_capacity=3200,
    ),
    "cudagraph_no_inductor": BenchResult(
        config_name="cudagraph_no_inductor", startup_time_s=45.0,
        latency_tok_s=89.0, latency_ttft_ms=42.0,
        throughput_tok_s=6615.0, throughput_req_s=103.0,
        kv_capacity=3200,
    ),
    "cudagraph_inductor": BenchResult(
        config_name="cudagraph_inductor", startup_time_s=120.0,
        latency_tok_s=127.0, latency_ttft_ms=35.0,
        throughput_tok_s=3112.0, throughput_req_s=48.0,
        kv_capacity=3200,
    ),
    "cudagraph_fp8kv": BenchResult(
        config_name="cudagraph_fp8kv", startup_time_s=50.0,
        latency_tok_s=22.0, latency_ttft_ms=120.0,
        throughput_tok_s=1600.0, throughput_req_s=25.0,
        kv_capacity=6400,
    ),
    "inductor_fp8kv": BenchResult(
        config_name="inductor_fp8kv", startup_time_s=130.0,
        latency_tok_s=30.0, latency_ttft_ms=100.0,
        throughput_tok_s=2000.0, throughput_req_s=31.0,
        kv_capacity=6400,
    ),
    "cudagraph_highmem": BenchResult(
        config_name="cudagraph_highmem", startup_time_s=48.0,
        latency_tok_s=88.0, latency_ttft_ms=43.0,
        throughput_tok_s=6800.0, throughput_req_s=106.0,
        kv_capacity=3500,
    ),
    "cudagraph_safemem": BenchResult(
        config_name="cudagraph_safemem", startup_time_s=44.0,
        latency_tok_s=87.0, latency_ttft_ms=44.0,
        throughput_tok_s=6100.0, throughput_req_s=95.0,
        kv_capacity=2800,
    ),
    "inductor_highmem": BenchResult(
        config_name="inductor_highmem", startup_time_s=125.0,
        latency_tok_s=125.0, latency_ttft_ms=36.0,
        throughput_tok_s=3200.0, throughput_req_s=50.0,
        kv_capacity=3500,
    ),
}


def mock_profile_config(config: ConfigCandidate) -> BenchResult:
    """Return mock results for testing."""
    if config.name in MOCK_RESULTS:
        return MOCK_RESULTS[config.name]
    # Generate plausible defaults
    return BenchResult(
        config_name=config.name, startup_time_s=60.0,
        latency_tok_s=50.0, latency_ttft_ms=60.0,
        throughput_tok_s=2000.0, throughput_req_s=30.0,
        kv_capacity=3200,
    )


# ---------------------------------------------------------------------------
# Profiling orchestration
# ---------------------------------------------------------------------------

def profile_config(
    config: ConfigCandidate,
    model_path: str,
    quantization: str,
    docker_image: str,
    model_mount: str,
    port: int = 8100,
    container_name: str = "autoconfig_bench",
) -> BenchResult:
    """Profile a single configuration. Launches container, benchmarks, tears down."""
    result = BenchResult(config_name=config.name)

    # Build model name for API calls
    model_api_name = f"/models/{Path(model_path).name}"

    # Stop any existing container with this name
    stop_container(container_name)
    time.sleep(1)

    # Launch
    cmd = build_docker_cmd(
        config, model_path, quantization, docker_image,
        container_name, port, model_mount,
    )
    print(f"\n  Launching: {config.name}")
    print(f"  Command: {' '.join(cmd[:10])}...")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode != 0:
            result.error = f"Docker launch failed: {proc.stderr[:300]}"
            print(f"  ERROR: {result.error}")
            return result
    except Exception as e:
        result.error = f"Docker launch exception: {e}"
        print(f"  ERROR: {result.error}")
        return result

    # Wait for health
    print(f"  Waiting for server...")
    healthy, startup_s = wait_for_health(port, timeout=360)
    result.startup_time_s = round(startup_s, 1)

    if not healthy:
        result.error = "Server failed to start within timeout"
        print(f"  ERROR: Server not healthy after {startup_s:.0f}s")
        # Grab logs
        try:
            logs = subprocess.check_output(
                ["docker", "logs", "--tail", "30", container_name],
                text=True, timeout=10, stderr=subprocess.STDOUT,
            )
            print(f"  Last logs:\n{logs[-500:]}")
        except Exception:
            pass
        stop_container(container_name)
        return result

    print(f"  Server ready in {startup_s:.1f}s")

    # Get KV capacity
    result.kv_capacity = get_kv_capacity(port)
    if result.kv_capacity > 0:
        print(f"  KV blocks: {result.kv_capacity}")

    # Latency benchmark (C=1)
    try:
        print(f"  Running latency bench (C=1, 15 requests)...")
        lat_result = asyncio.run(
            run_bench(port, model_api_name, concurrency=1, num_requests=15, max_tokens=64)
        )
        if "error" not in lat_result:
            result.latency_tok_s = lat_result["tok_per_s"]
            result.latency_ttft_ms = lat_result["avg_latency_s"] * 1000
            print(f"  Latency: {result.latency_tok_s:.1f} tok/s, "
                  f"avg {result.latency_ttft_ms:.0f}ms")
        else:
            print(f"  Latency bench error: {lat_result['error']}")
    except Exception as e:
        print(f"  Latency bench exception: {e}")

    # Throughput benchmark (C=64)
    try:
        print(f"  Running throughput bench (C=64, 20 requests)...")
        thr_result = asyncio.run(
            run_bench(port, model_api_name, concurrency=64, num_requests=20, max_tokens=64)
        )
        if "error" not in thr_result:
            result.throughput_tok_s = thr_result["tok_per_s"]
            result.throughput_req_s = thr_result["req_per_s"]
            print(f"  Throughput: {result.throughput_tok_s:.1f} tok/s, "
                  f"{result.throughput_req_s:.1f} req/s")
        else:
            print(f"  Throughput bench error: {thr_result['error']}")
    except Exception as e:
        print(f"  Throughput bench exception: {e}")

    # Cleanup
    print(f"  Stopping container...")
    stop_container(container_name)
    time.sleep(2)

    return result


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def select_optimal(results: list[BenchResult], target: str) -> tuple[BenchResult, str]:
    """Select the best config based on target. Returns (best_result, reason)."""
    # Filter out errors and skipped
    valid = [r for r in results if not r.error and not r.skipped]
    if not valid:
        return results[0] if results else BenchResult(), "No valid results"

    if target == "latency":
        best = max(valid, key=lambda r: r.latency_tok_s)
        reason = (f"Best single-request latency: {best.latency_tok_s:.1f} tok/s "
                  f"(avg response {best.latency_ttft_ms:.0f}ms)")

    elif target == "throughput":
        best = max(valid, key=lambda r: r.throughput_tok_s)
        reason = (f"Best aggregate throughput: {best.throughput_tok_s:.1f} tok/s "
                  f"at C=64 ({best.throughput_req_s:.1f} req/s)")

    elif target == "balanced":
        # Normalize both metrics to [0,1] and weight equally
        max_lat = max(r.latency_tok_s for r in valid) or 1
        max_thr = max(r.throughput_tok_s for r in valid) or 1

        def score(r):
            lat_norm = r.latency_tok_s / max_lat
            thr_norm = r.throughput_tok_s / max_thr
            return 0.5 * lat_norm + 0.5 * thr_norm

        best = max(valid, key=score)
        s = score(best)
        reason = (f"Best balanced score: {s:.3f} "
                  f"(latency={best.latency_tok_s:.1f}, "
                  f"throughput={best.throughput_tok_s:.1f})")

    else:
        best = max(valid, key=lambda r: r.throughput_tok_s)
        reason = f"Unknown target '{target}', defaulting to throughput"

    return best, reason


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def config_to_cli_flags(config: ConfigCandidate, model_path: str,
                        quantization: str) -> str:
    """Convert a ConfigCandidate to vLLM CLI flags."""
    parts = [
        f"--model /models/{Path(model_path).name}",
        f"--dtype bfloat16",
        f"--gpu-memory-utilization {config.gpu_memory_utilization}",
        f"--max-model-len {config.max_model_len}",
        "--trust-remote-code",
        "--host 0.0.0.0 --port 8000",
    ]
    if quantization != "none":
        parts.append(f"--quantization {quantization}")
    if config.enforce_eager:
        parts.append("--enforce-eager")
    if config.kv_cache_dtype != "auto":
        parts.append(f"--kv-cache-dtype {config.kv_cache_dtype}")
    if config.max_num_seqs != 256:
        parts.append(f"--max-num-seqs {config.max_num_seqs}")
    for arg in config.extra_args:
        parts.append(arg)
    return " \\\n    ".join(parts)


def config_to_env_vars(config: ConfigCandidate) -> dict:
    """Return environment variables needed for this config."""
    env = {}
    if config.inductor:
        env["VLLM_TORCH_COMPILE_LEVEL"] = "1"
    env.update(config.extra_env)
    return env


def generate_serve_script(
    config: ConfigCandidate,
    model_path: str,
    quantization: str,
    docker_image: str,
    model_mount: str,
    output_path: str,
):
    """Generate a serve_optimal.sh script."""
    env_vars = config_to_env_vars(config)
    env_flags = " ".join(f"-e {k}={v}" for k, v in env_vars.items())
    if env_flags:
        env_flags = " " + env_flags

    cli_flags = config_to_cli_flags(config, model_path, quantization)

    script = f"""#!/bin/bash
# Auto-generated by auto_config.py
# Optimal configuration: {config.name}
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

set -euo pipefail

CONTAINER_NAME="vllm-optimal"
IMAGE="{docker_image}"
MODEL_MOUNT="{model_mount}"

# Stop existing container if running
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "Starting vLLM with optimal config: {config.name}"

docker run -d \\
    --name "$CONTAINER_NAME" \\
    --runtime=nvidia --gpus all \\
    -v "$MODEL_MOUNT:/models:ro" \\
    -p 8000:8000 \\
    --shm-size=2g{env_flags} \\
    "$IMAGE" \\
    python3 -m vllm.entrypoints.openai.api_server \\
    {cli_flags}

echo "Container started. Waiting for health check..."

for i in $(seq 1 180); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server ready! (took ${{i}}x2 seconds)"
        echo "API endpoint: http://localhost:8000/v1"
        exit 0
    fi
    sleep 2
done

echo "ERROR: Server did not become healthy within 360 seconds"
docker logs --tail 30 "$CONTAINER_NAME"
exit 1
"""

    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)


def print_results_table(results: list[BenchResult]):
    """Print a formatted results table."""
    print("\n" + "=" * 90)
    print(f"{'Config':<25} {'Startup':>8} {'C=1 tok/s':>10} {'C=64 tok/s':>11} "
          f"{'C=64 req/s':>11} {'KV Blocks':>10} {'Status':>8}")
    print("-" * 90)
    for r in results:
        if r.error:
            status = "FAIL"
        elif r.skipped:
            status = "SKIP"
        else:
            status = "OK"

        print(f"{r.config_name:<25} {r.startup_time_s:>7.1f}s "
              f"{r.latency_tok_s:>10.1f} {r.throughput_tok_s:>11.1f} "
              f"{r.throughput_req_s:>11.1f} {r.kv_capacity:>10} {status:>8}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-Config Profiler for vLLM serving optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", required=True,
                        help="Path to model directory (inside Docker: /models/...)")
    parser.add_argument("--quantization", default="none",
                        help="Quantization method (none, modelopt, awq, gptq, fp8)")
    parser.add_argument("--target", choices=["latency", "throughput", "balanced"],
                        default="throughput",
                        help="Optimization target (default: throughput)")
    parser.add_argument("--docker-image", default="vllm-built",
                        help="Docker image to use (default: vllm-built)")
    parser.add_argument("--model-mount", default="/root/models",
                        help="Host path to model directory (default: /root/models)")
    parser.add_argument("--port", type=int, default=8100,
                        help="Host port for benchmark containers (default: 8100)")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for output files (default: .)")
    parser.add_argument("--configs", nargs="*",
                        help="Specific config names to test (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show configs, don't launch anything")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock results (for testing)")
    parser.add_argument("--container-prefix", default="autoconfig",
                        help="Docker container name prefix")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  vLLM Auto-Config Profiler")
    print("=" * 60)

    # --- Detect hardware ---
    print("\n[1/5] Detecting hardware...")
    if args.mock:
        gpu = detect_gpu_mock()
    else:
        gpu = detect_gpu()
    print(f"  GPU: {gpu.name}")
    print(f"  VRAM: {gpu.vram_gb:.1f} GB")
    print(f"  Compute: SM {gpu.compute_capability}")
    print(f"  Bandwidth: {gpu.bandwidth_gbps} GB/s")
    print(f"  FP8 support: {gpu.supports_fp8}")

    # --- Detect model ---
    print("\n[2/5] Detecting model...")
    if args.mock:
        model = detect_model_mock(args.model, args.quantization)
    else:
        model = detect_model(args.model, args.quantization)
    print(f"  Model: {model.name}")
    print(f"  Architecture: {model.architecture}")
    print(f"  Layers: {model.num_layers}, Hidden: {model.hidden_size}")
    print(f"  Parameters: ~{model.num_params_b}B")
    print(f"  Quantization: {model.quantization}")
    print(f"  Max seq len: {model.max_model_len}")

    # --- Generate configs ---
    print("\n[3/5] Generating configurations...")
    configs = generate_configs(gpu, model)

    if args.configs:
        configs = [c for c in configs if c.name in args.configs]
        if not configs:
            print(f"  ERROR: No matching configs. Available: "
                  f"{[c.name for c in generate_configs(gpu, model)]}")
            sys.exit(1)

    for c in configs:
        eager_str = "eager" if c.enforce_eager else "cudagraph"
        inductor_str = "+inductor" if c.inductor else ""
        kv_str = f" kv={c.kv_cache_dtype}" if c.kv_cache_dtype != "auto" else ""
        mem_str = f" mem={c.gpu_memory_utilization}"
        print(f"  - {c.name}: {eager_str}{inductor_str}{kv_str}{mem_str}")

    print(f"\n  Total configs to profile: {len(configs)}")

    if args.dry_run:
        print("\n[DRY RUN] Exiting without profiling.")
        # Save config list
        configs_out = output_dir / "auto_config_plans.json"
        with open(configs_out, "w") as f:
            json.dump({
                "gpu": gpu.to_dict(),
                "model": model.to_dict(),
                "configs": [c.to_dict() for c in configs],
            }, f, indent=2)
        print(f"  Saved plan to {configs_out}")
        return

    # --- Profile each config ---
    print("\n[4/5] Profiling configurations...")
    results = []

    for i, config in enumerate(configs):
        print(f"\n--- Config {i+1}/{len(configs)}: {config.name} ---")

        if args.mock:
            result = mock_profile_config(config)
            results.append(result)
            print(f"  [MOCK] latency={result.latency_tok_s} tok/s, "
                  f"throughput={result.throughput_tok_s} tok/s")
        else:
            container_name = f"{args.container_prefix}_{config.name}"
            result = profile_config(
                config=config,
                model_path=args.model,
                quantization=args.quantization,
                docker_image=args.docker_image,
                model_mount=args.model_mount,
                port=args.port,
                container_name=container_name,
            )
            results.append(result)

    # --- Select optimal ---
    print("\n[5/5] Selecting optimal configuration...")
    print_results_table(results)

    best, reason = select_optimal(results, args.target)
    best_config = next((c for c in configs if c.name == best.config_name), configs[0])

    print(f"\n  RECOMMENDED: {best.config_name}")
    print(f"  Reason: {reason}")
    print(f"  Startup time: {best.startup_time_s:.1f}s")

    # Print CLI flags
    print(f"\n  vLLM flags:")
    env_vars = config_to_env_vars(best_config)
    if env_vars:
        for k, v in env_vars.items():
            print(f"    export {k}={v}")
    cli_flags = config_to_cli_flags(best_config, args.model, args.quantization)
    for line in cli_flags.split("\n"):
        print(f"    {line.strip()}")

    # --- Save outputs ---
    # JSON results
    json_path = output_dir / "auto_config_results.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "target": args.target,
            "gpu": gpu.to_dict(),
            "model": model.to_dict(),
            "results": [r.to_dict() for r in results],
            "recommended": {
                "config_name": best.config_name,
                "config": best_config.to_dict(),
                "reason": reason,
            },
        }, f, indent=2)
    print(f"\n  Results saved to {json_path}")

    # Generate serve script
    script_path = output_dir / "serve_optimal.sh"
    generate_serve_script(
        config=best_config,
        model_path=args.model,
        quantization=args.quantization,
        docker_image=args.docker_image,
        model_mount=args.model_mount,
        output_path=str(script_path),
    )
    print(f"  Serve script saved to {script_path}")

    print(f"\n  Done! Run: bash {script_path}")


if __name__ == "__main__":
    main()
