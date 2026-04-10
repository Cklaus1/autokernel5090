"""Shared data types for AutoKernel v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# GPU specification
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Hardware specification for a GPU."""
    name: str = "Unknown"
    sm_count: int = 0
    memory_gb: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_tflops_fp4: float = 0.0  # Blackwell+
    peak_bandwidth_gb_s: float = 0.0
    l2_cache_mb: float = 0.0
    compute_capability: tuple[int, int] = (0, 0)
    has_fp8: bool = False
    has_fp4: bool = False
    has_cuda_graphs: bool = True

    @property
    def arch_family(self) -> str:
        major = self.compute_capability[0]
        return {
            7: "volta_turing",
            8: "ampere",
            9: "hopper_blackwell",
            10: "hopper_blackwell",
        }.get(major, "unknown")


# Known GPU database
KNOWN_GPUS: dict[str, GPUInfo] = {
    "rtx-5090": GPUInfo(
        name="NVIDIA GeForce RTX 5090",
        sm_count=170,
        memory_gb=32.0,
        peak_tflops_fp16=419.0,
        peak_tflops_bf16=419.0,
        peak_tflops_fp32=104.8,
        peak_tflops_fp4=3352.0,
        peak_bandwidth_gb_s=1792.0,
        l2_cache_mb=96.0,
        compute_capability=(10, 0),
        has_fp8=True,
        has_fp4=True,
    ),
    "rtx-4090": GPUInfo(
        name="NVIDIA GeForce RTX 4090",
        sm_count=128,
        memory_gb=24.0,
        peak_tflops_fp16=330.0,
        peak_tflops_bf16=330.0,
        peak_tflops_fp32=82.6,
        peak_bandwidth_gb_s=1008.0,
        l2_cache_mb=72.0,
        compute_capability=(8, 9),
        has_fp8=True,
        has_fp4=False,
    ),
    "h100-sxm": GPUInfo(
        name="NVIDIA H100 SXM",
        sm_count=132,
        memory_gb=80.0,
        peak_tflops_fp16=989.5,
        peak_tflops_bf16=989.5,
        peak_tflops_fp32=66.9,
        peak_bandwidth_gb_s=3352.0,
        l2_cache_mb=50.0,
        compute_capability=(9, 0),
        has_fp8=True,
        has_fp4=False,
    ),
    "a100-sxm": GPUInfo(
        name="NVIDIA A100 SXM",
        sm_count=108,
        memory_gb=80.0,
        peak_tflops_fp16=312.0,
        peak_tflops_bf16=312.0,
        peak_tflops_fp32=19.5,
        peak_bandwidth_gb_s=2039.0,
        l2_cache_mb=40.0,
        compute_capability=(8, 0),
        has_fp8=False,
        has_fp4=False,
    ),
    "l40s": GPUInfo(
        name="NVIDIA L40S",
        sm_count=142,
        memory_gb=48.0,
        peak_tflops_fp16=362.05,
        peak_tflops_bf16=362.05,
        peak_tflops_fp32=90.5,
        peak_bandwidth_gb_s=864.0,
        l2_cache_mb=48.0,
        compute_capability=(8, 9),
        has_fp8=True,
        has_fp4=False,
    ),
}


def detect_gpu_runtime() -> GPUInfo:
    """Detect GPU at runtime via torch.cuda."""
    try:
        import torch
        if not torch.cuda.is_available():
            return GPUInfo()
        props = torch.cuda.get_device_properties(0)
        name = props.name
        cc = (props.major, props.minor)
        memory_gb = round(props.total_memory / (1024 ** 3), 1)
        sm_count = props.multi_processor_count

        # Match against known GPUs
        for key, gpu in KNOWN_GPUS.items():
            # Check if any part of the known name matches
            for fragment in key.replace("-", " ").split():
                if fragment.isdigit() and fragment in name:
                    return GPUInfo(
                        name=name,
                        sm_count=sm_count,
                        memory_gb=memory_gb,
                        peak_tflops_fp16=gpu.peak_tflops_fp16,
                        peak_tflops_bf16=gpu.peak_tflops_bf16,
                        peak_tflops_fp32=gpu.peak_tflops_fp32,
                        peak_tflops_fp4=gpu.peak_tflops_fp4,
                        peak_bandwidth_gb_s=gpu.peak_bandwidth_gb_s,
                        l2_cache_mb=gpu.l2_cache_mb,
                        compute_capability=cc,
                        has_fp8=gpu.has_fp8,
                        has_fp4=gpu.has_fp4,
                    )

        # Fallback: estimate from SM count and clock
        ops_per_clock = 256 if cc[0] >= 8 else 128
        clock_ghz = props.clock_rate / 1e6
        peak_fp16 = sm_count * ops_per_clock * clock_ghz * 2 / 1e3

        return GPUInfo(
            name=name,
            sm_count=sm_count,
            memory_gb=memory_gb,
            peak_tflops_fp16=peak_fp16,
            peak_tflops_bf16=peak_fp16,
            peak_tflops_fp32=peak_fp16 / 2,
            peak_bandwidth_gb_s=500.0,
            l2_cache_mb=0.0,
            compute_capability=cc,
            has_fp8=cc[0] >= 9,
            has_fp4=cc[0] >= 10,
        )
    except Exception:
        return GPUInfo()


# ---------------------------------------------------------------------------
# Profile data types
# ---------------------------------------------------------------------------

class OpCategory(str, Enum):
    """Categories of GPU operations."""
    ATTENTION = "attention"
    LINEAR = "linear"
    NORM = "norm"
    ACTIVATION = "activation"
    EMBEDDING = "embedding"
    MOE_ROUTING = "moe_routing"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    QUANTIZATION = "quantization"
    SAMPLING = "sampling"
    OTHER = "other"


@dataclass
class OpProfile:
    """Profile of a single operation in the model."""
    name: str
    category: OpCategory
    time_us: float = 0.0
    time_fraction: float = 0.0
    flops: float = 0.0
    bytes_accessed: float = 0.0
    utilization: float = 0.0  # 0.0 to 1.0 (actual / theoretical peak)
    memory_mb: float = 0.0
    call_count: int = 1
    shapes: dict[str, Any] = field(default_factory=dict)
    dtype: str = "float16"
    kernel_names: list[str] = field(default_factory=list)
    # Derived
    arithmetic_intensity: float = 0.0  # flops / bytes


@dataclass
class ProfileResult:
    """Complete profile of a model's inference step."""
    model_name: str
    gpu: GPUInfo
    total_time_us: float = 0.0
    ops: list[OpProfile] = field(default_factory=list)
    memory_total_mb: float = 0.0
    memory_kv_cache_mb: float = 0.0
    memory_weights_mb: float = 0.0
    memory_activations_mb: float = 0.0
    batch_size: int = 1
    sequence_length: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def throughput_tokens_per_sec(self) -> float:
        if self.total_time_us <= 0:
            return 0.0
        return self.batch_size * 1e6 / self.total_time_us

    def top_ops(self, n: int = 10) -> list[OpProfile]:
        return sorted(self.ops, key=lambda o: o.time_fraction, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Optimization target and candidate types
# ---------------------------------------------------------------------------

@dataclass
class OptimizationTarget:
    """A bottleneck identified for optimization."""
    op_name: str
    category: OpCategory
    time_fraction: float  # fraction of total decode time
    utilization: float    # current hardware utilization (0-1)
    headroom: float       # time_fraction * (1 - utilization) = improvement potential
    ceiling_speedup: float  # max speedup if this op went to 100% utilization
    amdahl_max_speedup: float  # max end-to-end speedup (Amdahl's law)
    shapes: dict[str, Any] = field(default_factory=dict)
    dtype: str = "float16"
    memory_mb: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def priority_score(self) -> float:
        """Higher = more worth optimizing."""
        return self.headroom * self.ceiling_speedup


@dataclass
class Candidate:
    """A proposed optimization for a target."""
    name: str
    description: str
    target: OptimizationTarget
    strategy: str  # fusion, quantization, algorithm, config, memory
    expected_impact: float  # estimated speedup multiplier (e.g. 1.5 = 50% faster)
    implementation_plan: list[str]  # step-by-step
    effort: str  # low, medium, high
    confidence: float = 0.5  # 0-1, how sure we are this will work
    prerequisites: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of benchmarking a candidate optimization."""
    candidate: Candidate
    baseline_time_us: float = 0.0
    optimized_time_us: float = 0.0
    speedup: float = 1.0
    correctness: bool = False
    memory_delta_mb: float = 0.0
    notes: str = ""
    applied: bool = False


@dataclass
class OptimizationRound:
    """Record of one optimization round."""
    round_number: int
    target: OptimizationTarget
    candidates: list[Candidate] = field(default_factory=list)
    results: list[BenchmarkResult] = field(default_factory=list)
    best_result: Optional[BenchmarkResult] = None
    cumulative_speedup: float = 1.0
