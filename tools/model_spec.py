"""
ModelSpec: Declarative model configuration DSL for GPU serving optimization.

Captures all optimization decisions (inductor, KV cache, CUDA graphs, fused kernels,
serving parameters) in one spec file. Enables reproducible and portable serving configs.

Usage:
    spec = ModelSpec.load("specs/gemma4_nvfp4_rtx5090_throughput.yaml")
    spec.validate()
    print(spec.to_docker_command())
    spec.to_launch_script("serve.sh")
    diff = ModelSpec.compare(spec_a, spec_b)
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class ModelSpec:
    """Complete optimization specification for a model+GPU combination."""

    # --- Model identity ---
    model_name: str = ""              # e.g. "gemma-4-26B-A4B-it-NVFP4"
    model_path: str = ""              # e.g. "/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
    architecture: str = ""            # e.g. "gemma4_moe"
    quantization: str = ""            # e.g. "modelopt" | "awq" | "gptq" | "none"

    # --- Hardware ---
    gpu: str = ""                     # e.g. "rtx-5090"
    gpu_memory_gb: int = 0            # e.g. 32
    num_gpus: int = 1
    tensor_parallel: int = 1

    # --- Serving config ---
    serving_mode: str = "throughput"  # "throughput" | "latency" | "balanced"
    inductor: bool = False
    cudagraph_mode: str = "full"      # "full" | "piecewise" | "none"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    max_num_seqs: int = 256
    dtype: str = "auto"               # "auto" | "bfloat16" | "float16"

    # --- KV cache ---
    kv_cache_dtype: str = "auto"      # "auto" | "fp8" | "k4v4b64" | "k8v4b32"
    per_layer_kv: dict | None = None  # e.g. {"sliding": "k4v4b64", "global": "k8v8"}

    # --- Kernel optimizations ---
    fused_norm_quant: bool = False
    fused_add_residual: bool = True
    shuffle_quant_fusion: bool = False

    # --- Docker / deployment ---
    docker_image: str = "vllm-built"
    docker_port: int = 8000
    docker_extra_args: list[str] = field(default_factory=list)
    vllm_extra_args: list[str] = field(default_factory=list)

    # --- Performance profile (from benchmarks) ---
    measured_throughput: dict[str, float] = field(default_factory=dict)  # {"C1": 89, "C32": 1738}
    measured_kv_capacity: int = 0
    profiled_date: str = ""

    # --- Metadata ---
    description: str = ""
    notes: list[str] = field(default_factory=list)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dict matching the YAML structure."""
        d: dict[str, Any] = {
            "model": {
                "name": self.model_name,
                "path": self.model_path,
                "architecture": self.architecture,
                "quantization": self.quantization,
            },
            "hardware": {
                "gpu": self.gpu,
                "memory_gb": self.gpu_memory_gb,
                "num_gpus": self.num_gpus,
                "tensor_parallel": self.tensor_parallel,
            },
            "serving": {
                "mode": self.serving_mode,
                "inductor": self.inductor,
                "cudagraph_mode": self.cudagraph_mode,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "enforce_eager": self.enforce_eager,
                "max_num_seqs": self.max_num_seqs,
                "dtype": self.dtype,
            },
            "kv_cache": {
                "dtype": self.kv_cache_dtype,
                "per_layer": self.per_layer_kv,
            },
            "kernels": {
                "fused_norm_quant": self.fused_norm_quant,
                "fused_add_residual": self.fused_add_residual,
                "shuffle_quant_fusion": self.shuffle_quant_fusion,
            },
            "docker": {
                "image": self.docker_image,
                "port": self.docker_port,
                "extra_args": self.docker_extra_args,
                "vllm_extra_args": self.vllm_extra_args,
            },
            "performance": {
                **{k: v for k, v in self.measured_throughput.items()},
                "kv_tokens": self.measured_kv_capacity,
                "profiled": self.profiled_date,
            },
        }
        if self.description:
            d["description"] = self.description
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelSpec:
        """Construct a ModelSpec from a nested dict (as loaded from YAML/JSON)."""
        model = d.get("model", {})
        hw = d.get("hardware", {})
        srv = d.get("serving", {})
        kv = d.get("kv_cache", {})
        kern = d.get("kernels", {})
        dock = d.get("docker", {})
        perf = d.get("performance", {})

        # Extract throughput entries (keys starting with C)
        throughput = {k: v for k, v in perf.items() if k not in ("kv_tokens", "profiled")}

        return cls(
            model_name=model.get("name", ""),
            model_path=model.get("path", ""),
            architecture=model.get("architecture", ""),
            quantization=model.get("quantization", ""),
            gpu=hw.get("gpu", ""),
            gpu_memory_gb=hw.get("memory_gb", 0),
            num_gpus=hw.get("num_gpus", 1),
            tensor_parallel=hw.get("tensor_parallel", 1),
            serving_mode=srv.get("mode", "throughput"),
            inductor=srv.get("inductor", False),
            cudagraph_mode=srv.get("cudagraph_mode", "full"),
            max_model_len=srv.get("max_model_len", 4096),
            gpu_memory_utilization=srv.get("gpu_memory_utilization", 0.90),
            enforce_eager=srv.get("enforce_eager", False),
            max_num_seqs=srv.get("max_num_seqs", 256),
            dtype=srv.get("dtype", "auto"),
            kv_cache_dtype=kv.get("dtype", "auto"),
            per_layer_kv=kv.get("per_layer"),
            fused_norm_quant=kern.get("fused_norm_quant", False),
            fused_add_residual=kern.get("fused_add_residual", True),
            shuffle_quant_fusion=kern.get("shuffle_quant_fusion", False),
            docker_image=dock.get("image", "vllm-built"),
            docker_port=dock.get("port", 8000),
            docker_extra_args=dock.get("extra_args", []),
            vllm_extra_args=dock.get("vllm_extra_args", []),
            measured_throughput=throughput,
            measured_kv_capacity=perf.get("kv_tokens", 0),
            profiled_date=perf.get("profiled", ""),
            description=d.get("description", ""),
            notes=d.get("notes", []),
        )

    # -------------------------------------------------------------------------
    # File I/O
    # -------------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save spec to YAML or JSON (determined by extension)."""
        path = Path(path)
        d = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML output: pip install pyyaml")
            with open(path, "w") as f:
                yaml.dump(d, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(d, f, indent=2)
        else:
            raise ValueError(f"Unsupported extension: {path.suffix} (use .yaml or .json)")

    @classmethod
    def load(cls, path: str | Path) -> ModelSpec:
        """Load a spec from YAML or JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")

        if path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML input: pip install pyyaml")
            with open(path) as f:
                d = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                d = json.load(f)
        else:
            raise ValueError(f"Unsupported extension: {path.suffix}")

        return cls.from_dict(d)

    # -------------------------------------------------------------------------
    # Docker command generation
    # -------------------------------------------------------------------------

    def to_docker_command(self, detach: bool = True, name: str | None = None) -> str:
        """Generate a complete docker run command for this spec."""
        parts = ["docker run"]
        if detach:
            parts.append("-d")
        parts.append("--gpus all --ipc=host --privileged")

        container_name = name or f"vllm-{self.model_name.lower().replace(' ', '-')}"
        parts.append(f"--name {container_name}")
        parts.append(f"-p {self.docker_port}:8000")
        parts.append(f"-v {self.model_path}:{self.model_path}")

        for extra in self.docker_extra_args:
            parts.append(extra)

        parts.append(self.docker_image)

        # vLLM command
        vllm_args = self._build_vllm_args()
        parts.append(f"python3 -m vllm.entrypoints.openai.api_server {vllm_args}")

        return " \\\n  ".join(parts)

    def _build_vllm_args(self) -> str:
        """Build vLLM server CLI arguments from spec."""
        args = [
            f"--model {self.model_path}",
            f"--max-model-len {self.max_model_len}",
            f"--gpu-memory-utilization {self.gpu_memory_utilization}",
            f"--max-num-seqs {self.max_num_seqs}",
        ]

        if self.tensor_parallel > 1:
            args.append(f"--tensor-parallel-size {self.tensor_parallel}")

        if self.dtype != "auto":
            args.append(f"--dtype {self.dtype}")

        if self.quantization and self.quantization != "none":
            args.append(f"--quantization {self.quantization}")

        if self.kv_cache_dtype != "auto":
            args.append(f"--kv-cache-dtype {self.kv_cache_dtype}")

        if self.enforce_eager:
            args.append("--enforce-eager")

        if not self.inductor:
            args.append("--no-enable-torch-compile")

        for extra in self.vllm_extra_args:
            args.append(extra)

        return " \\\n    ".join(args)

    # -------------------------------------------------------------------------
    # Launch script generation
    # -------------------------------------------------------------------------

    def to_launch_script(self, path: str | Path) -> None:
        """Generate a self-contained bash launch script."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        env_lines = []
        if self.cudagraph_mode == "none":
            env_lines.append("export VLLM_TORCH_COMPILE_LEVEL=0")
        elif self.cudagraph_mode == "piecewise":
            env_lines.append("export VLLM_TORCH_COMPILE_LEVEL=2")
        if self.inductor:
            env_lines.append("export VLLM_TORCH_COMPILE_LEVEL=3")

        env_block = "\n".join(env_lines) if env_lines else "# No special environment variables needed"
        docker_cmd = self.to_docker_command(detach=True)
        container = f"vllm-{self.model_name.lower().replace(' ', '-')}"
        usage_json = f'{{"model": "{self.model_path}", "prompt": "Hello", "max_tokens": 64}}'

        lines = [
            "#!/usr/bin/env bash",
            f"# Auto-generated serving script for: {self.model_name}",
            f"# Spec: {self.serving_mode} mode on {self.gpu} x{self.num_gpus}",
            "# Generated by ModelSpec -- do not edit manually",
            "#",
            f"# Description: {self.description or 'N/A'}",
            "set -euo pipefail",
            "",
            "# ---- Environment ----",
            env_block,
            "",
            "# ---- Stop existing container ----",
            f"docker rm -f {container} 2>/dev/null || true",
            "",
            "# ---- Launch ----",
            docker_cmd,
            "",
            'echo ""',
            f'echo "Server starting on port {self.docker_port}..."',
            f'echo "Model: {self.model_name}"',
            f'echo "Mode:  {self.serving_mode} | Inductor: {self.inductor} | CUDA graphs: {self.cudagraph_mode}"',
            f'echo "KV:    {self.kv_cache_dtype}"',
            'echo ""',
            f'echo "Health check: curl http://localhost:{self.docker_port}/health"',
            f"echo \"Usage:        curl http://localhost:{self.docker_port}/v1/completions -d '{usage_json}'\"",
        ]

        path.write_text("\n".join(lines) + "\n")
        path.chmod(0o755)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Validate internal consistency. Returns list of issues (empty = valid).
        Raises ValueError if critical issues are found.
        """
        issues: list[str] = []
        warnings: list[str] = []

        # Required fields
        if not self.model_name:
            issues.append("model_name is required")
        if not self.model_path:
            issues.append("model_path is required")

        # Serving mode
        if self.serving_mode not in ("throughput", "latency", "balanced"):
            issues.append(f"Invalid serving_mode: {self.serving_mode!r} (expected throughput|latency|balanced)")

        # CUDA graph mode
        if self.cudagraph_mode not in ("full", "piecewise", "none"):
            issues.append(f"Invalid cudagraph_mode: {self.cudagraph_mode!r} (expected full|piecewise|none)")

        # Inductor + eager conflict
        if self.inductor and self.enforce_eager:
            issues.append("inductor=True conflicts with enforce_eager=True")

        # GPU memory utilization range
        if not (0.1 <= self.gpu_memory_utilization <= 1.0):
            issues.append(f"gpu_memory_utilization={self.gpu_memory_utilization} out of range [0.1, 1.0]")

        # Tensor parallel vs num_gpus
        if self.tensor_parallel > self.num_gpus:
            issues.append(f"tensor_parallel={self.tensor_parallel} exceeds num_gpus={self.num_gpus}")

        # KV cache dtype
        valid_kv = ("auto", "fp8", "fp8_e5m2", "fp8_e4m3", "k4v4b64", "k8v4b32", "k8v4b16", "k8v8")
        if self.kv_cache_dtype not in valid_kv:
            warnings.append(f"Non-standard kv_cache_dtype: {self.kv_cache_dtype!r}")

        # FusenCache requires custom vLLM build
        if self.kv_cache_dtype.startswith("k") and self.kv_cache_dtype != "auto":
            warnings.append(f"KV dtype {self.kv_cache_dtype!r} requires FusenCache-patched vLLM build")

        # Throughput mode advice
        if self.serving_mode == "throughput" and self.inductor:
            warnings.append("Inductor typically reduces throughput (batch serving); consider inductor=False")

        # Latency mode advice
        if self.serving_mode == "latency" and not self.inductor:
            warnings.append("Inductor typically improves single-request latency; consider inductor=True")

        # max_model_len sanity
        if self.max_model_len < 128:
            warnings.append(f"max_model_len={self.max_model_len} is unusually low")
        if self.max_model_len > 131072:
            warnings.append(f"max_model_len={self.max_model_len} is very large; check GPU memory")

        all_issues = [f"ERROR: {i}" for i in issues] + [f"WARNING: {w}" for w in warnings]

        if issues:
            raise ValueError(
                f"Spec validation failed with {len(issues)} error(s):\n"
                + "\n".join(f"  - {i}" for i in issues)
            )

        return all_issues

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------

    @staticmethod
    def compare(a: ModelSpec, b: ModelSpec) -> str:
        """
        Compare two specs and produce a human-readable diff with impact analysis.
        """
        lines: list[str] = []
        lines.append(f"Comparing: [{a.model_name}] vs [{b.model_name}]")
        lines.append("=" * 60)

        diffs: list[tuple[str, Any, Any, str]] = []

        # Helper to record differences
        def check(label: str, va: Any, vb: Any, impact: str = ""):
            if va != vb:
                diffs.append((label, va, vb, impact))

        check("model_name", a.model_name, b.model_name)
        check("model_path", a.model_path, b.model_path)
        check("architecture", a.architecture, b.architecture)
        check("quantization", a.quantization, b.quantization)
        check("gpu", a.gpu, b.gpu)
        check("gpu_memory_gb", a.gpu_memory_gb, b.gpu_memory_gb)
        check("num_gpus", a.num_gpus, b.num_gpus)
        check("tensor_parallel", a.tensor_parallel, b.tensor_parallel)
        check("serving_mode", a.serving_mode, b.serving_mode)

        # Inductor has a big impact
        check(
            "inductor", a.inductor, b.inductor,
            "Inductor ON: better single-request latency. OFF: ~2x batch throughput."
        )
        check(
            "cudagraph_mode", a.cudagraph_mode, b.cudagraph_mode,
            "CUDA graphs reduce kernel launch overhead; 'full' is fastest for fixed shapes."
        )
        check("max_model_len", a.max_model_len, b.max_model_len)
        check("gpu_memory_utilization", a.gpu_memory_utilization, b.gpu_memory_utilization)
        check("enforce_eager", a.enforce_eager, b.enforce_eager)
        check("max_num_seqs", a.max_num_seqs, b.max_num_seqs)
        check("dtype", a.dtype, b.dtype)

        check(
            "kv_cache_dtype", a.kv_cache_dtype, b.kv_cache_dtype,
            "FP8 KV: 2x capacity vs BF16. FusenCache k4v4b64: 4x capacity."
        )
        check("per_layer_kv", a.per_layer_kv, b.per_layer_kv)

        check("fused_norm_quant", a.fused_norm_quant, b.fused_norm_quant,
              "Fused norm+quant kernel reduces memory traffic.")
        check("fused_add_residual", a.fused_add_residual, b.fused_add_residual)
        check("shuffle_quant_fusion", a.shuffle_quant_fusion, b.shuffle_quant_fusion)

        if not diffs:
            lines.append("Specs are identical.")
            return "\n".join(lines)

        lines.append(f"{len(diffs)} difference(s) found:\n")
        for label, va, vb, impact in diffs:
            lines.append(f"  {label}:")
            lines.append(f"    A: {va}")
            lines.append(f"    B: {vb}")
            if impact:
                lines.append(f"    Impact: {impact}")
            lines.append("")

        # Throughput comparison if both have measured data
        if a.measured_throughput and b.measured_throughput:
            lines.append("Performance comparison:")
            all_keys = sorted(set(a.measured_throughput) | set(b.measured_throughput))
            for k in all_keys:
                va = a.measured_throughput.get(k, "N/A")
                vb = b.measured_throughput.get(k, "N/A")
                if va != "N/A" and vb != "N/A":
                    ratio = vb / va if va else float("inf")
                    lines.append(f"  {k}: {va} -> {vb} tok/s ({ratio:.2f}x)")
                else:
                    lines.append(f"  {k}: {va} -> {vb} tok/s")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """One-line summary of the spec."""
        kv = self.kv_cache_dtype if self.kv_cache_dtype != "auto" else "BF16"
        ind = "inductor" if self.inductor else "no-inductor"
        return (
            f"{self.model_name} | {self.gpu} x{self.num_gpus} | "
            f"{self.serving_mode} | {ind} | cuda_graph={self.cudagraph_mode} | "
            f"KV={kv} | max_len={self.max_model_len}"
        )

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Preset specs for known configurations
# ---------------------------------------------------------------------------

def gemma4_nvfp4_rtx5090_throughput() -> ModelSpec:
    """Best throughput config: no inductor, full CUDA graphs, BF16 KV."""
    return ModelSpec(
        model_name="gemma-4-26B-A4B-it-NVFP4",
        model_path="/models/gemma-4-26B-A4B-it-NVFP4-modelopt",
        architecture="gemma4_moe",
        quantization="modelopt",
        gpu="rtx-5090",
        gpu_memory_gb=32,
        num_gpus=1,
        tensor_parallel=1,
        serving_mode="throughput",
        inductor=False,
        cudagraph_mode="full",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        max_num_seqs=256,
        dtype="auto",
        kv_cache_dtype="auto",
        per_layer_kv=None,
        fused_norm_quant=False,
        fused_add_residual=True,
        shuffle_quant_fusion=False,
        docker_image="vllm-built",
        docker_port=8000,
        measured_throughput={"C1": 89, "C32": 1738, "C256": 6615},
        measured_kv_capacity=43760,
        profiled_date="2026-04-10",
        description="Best throughput: no inductor, full CUDA graphs, BF16 KV cache",
        notes=[
            "Inductor OFF gives ~2x throughput for batched serving",
            "CUDA graphs reduce kernel launch overhead significantly",
            "BF16 KV is default; FP8 saves memory but adds conversion overhead",
        ],
    )


def gemma4_nvfp4_rtx5090_latency() -> ModelSpec:
    """Best latency config: inductor on, optimized for single-request."""
    return ModelSpec(
        model_name="gemma-4-26B-A4B-it-NVFP4",
        model_path="/models/gemma-4-26B-A4B-it-NVFP4-modelopt",
        architecture="gemma4_moe",
        quantization="modelopt",
        gpu="rtx-5090",
        gpu_memory_gb=32,
        num_gpus=1,
        tensor_parallel=1,
        serving_mode="latency",
        inductor=True,
        cudagraph_mode="full",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        max_num_seqs=32,
        dtype="auto",
        kv_cache_dtype="auto",
        per_layer_kv=None,
        fused_norm_quant=False,
        fused_add_residual=True,
        shuffle_quant_fusion=False,
        docker_image="vllm-built",
        docker_port=8000,
        measured_throughput={"C1": 45, "C4": 170},
        measured_kv_capacity=43760,
        profiled_date="2026-04-10",
        description="Best latency: inductor ON for torch.compile optimizations",
        notes=[
            "Inductor ON improves single-request latency ~30%",
            "Lower max_num_seqs to reduce scheduling overhead",
            "Throughput is ~50% of no-inductor config for large batches",
        ],
    )


def gemma4_nvfp4_rtx5090_fusencache() -> ModelSpec:
    """FusenCache config: 4x KV capacity via k4v4b64 quantization."""
    return ModelSpec(
        model_name="gemma-4-26B-A4B-it-NVFP4",
        model_path="/models/gemma-4-26B-A4B-it-NVFP4-modelopt",
        architecture="gemma4_moe",
        quantization="modelopt",
        gpu="rtx-5090",
        gpu_memory_gb=32,
        num_gpus=1,
        tensor_parallel=1,
        serving_mode="throughput",
        inductor=False,
        cudagraph_mode="full",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        max_num_seqs=512,
        dtype="auto",
        kv_cache_dtype="k4v4b64",
        per_layer_kv={"sliding": "k4v4b64", "global": "k8v4b16"},
        fused_norm_quant=False,
        fused_add_residual=True,
        shuffle_quant_fusion=False,
        docker_image="vllm-fusencache",
        docker_port=8000,
        measured_throughput={"C1": 85, "C32": 1650, "C512": 9800},
        measured_kv_capacity=175040,
        profiled_date="2026-04-10",
        description="FusenCache: 4x KV capacity via k4v4b64 quantization (175K tokens)",
        notes=[
            "FusenCache k4v4b64 gives ~4x KV compression vs BF16",
            "Per-layer: sliding attention uses aggressive k4v4, global uses k8v4b16",
            "Requires FusenCache-patched vLLM build (docker image: vllm-fusencache)",
            "Higher max_num_seqs to exploit larger KV capacity",
        ],
    )


def gemma4_nvfp4_pro6000_tp2() -> ModelSpec:
    """Projected config for PRO 6000 with TP=2."""
    return ModelSpec(
        model_name="gemma-4-26B-A4B-it-NVFP4",
        model_path="/models/gemma-4-26B-A4B-it-NVFP4-modelopt",
        architecture="gemma4_moe",
        quantization="modelopt",
        gpu="pro-6000",
        gpu_memory_gb=96,
        num_gpus=2,
        tensor_parallel=2,
        serving_mode="throughput",
        inductor=False,
        cudagraph_mode="full",
        max_model_len=16384,
        gpu_memory_utilization=0.92,
        enforce_eager=False,
        max_num_seqs=512,
        dtype="auto",
        kv_cache_dtype="fp8",
        per_layer_kv=None,
        fused_norm_quant=True,
        fused_add_residual=True,
        shuffle_quant_fusion=True,
        docker_image="vllm-built",
        docker_port=8000,
        measured_throughput={},
        measured_kv_capacity=0,
        profiled_date="",
        description="Projected PRO 6000 TP=2: 192GB VRAM, FP8 KV, 16K context",
        notes=[
            "PRO 6000 has 96GB per GPU, 192GB total with TP=2",
            "FP8 KV is safe bet: 2x capacity, well-supported",
            "16K context feasible with 192GB VRAM",
            "Fused kernels expected available for Blackwell",
            "Performance numbers TBD -- not yet benchmarked",
        ],
    )


PRESETS = {
    "gemma4_nvfp4_rtx5090_throughput": gemma4_nvfp4_rtx5090_throughput,
    "gemma4_nvfp4_rtx5090_latency": gemma4_nvfp4_rtx5090_latency,
    "gemma4_nvfp4_rtx5090_fusencache": gemma4_nvfp4_rtx5090_fusencache,
    "gemma4_nvfp4_pro6000_tp2": gemma4_nvfp4_pro6000_tp2,
}
