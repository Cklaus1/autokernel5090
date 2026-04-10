"""
AutoKernel v2 Profiler -- Automatic model profiling and operation decomposition.

Profiles a model's inference step and decomposes it into per-operation timing,
memory usage, and compute utilization. Supports both torch.profiler-based
profiling and API-level timing for serving frameworks.

Usage:
    from autokernel_v2.profiler import ModelProfiler
    profiler = ModelProfiler()
    result = profiler.profile("path/to/model", gpu_info)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .types import (
    GPUInfo,
    OpCategory,
    OpProfile,
    ProfileResult,
    detect_gpu_runtime,
)

# ---------------------------------------------------------------------------
# Operation classification rules
# ---------------------------------------------------------------------------

# (pattern_fragments, category) -- checked in order, first match wins
_OP_CLASSIFICATION: list[tuple[list[str], OpCategory]] = [
    (["flash", "fmha", "sdpa", "attention", "attn"],   OpCategory.ATTENTION),
    (["gemm", "matmul", "cublas", "linear", "mm_"],     OpCategory.LINEAR),
    (["layer_norm", "layernorm", "rms_norm", "rmsnorm"], OpCategory.NORM),
    (["gelu", "silu", "relu", "swiglu", "activation"],  OpCategory.ACTIVATION),
    (["embed", "rotary", "rope", "position"],            OpCategory.EMBEDDING),
    (["moe", "topk", "routing", "expert_select"],        OpCategory.MOE_ROUTING),
    (["allreduce", "allgather", "nccl", "broadcast"],    OpCategory.COMMUNICATION),
    (["copy", "memcpy", "memset", "transpose", "cat"],   OpCategory.MEMORY),
    (["quant", "dequant", "scale", "fp4", "fp8"],        OpCategory.QUANTIZATION),
    (["sample", "argmax", "topk_sample", "logits"],      OpCategory.SAMPLING),
]


def classify_op(kernel_name: str) -> OpCategory:
    """Classify a CUDA kernel name into an operation category."""
    name_lower = kernel_name.lower()
    for fragments, category in _OP_CLASSIFICATION:
        for frag in fragments:
            if frag in name_lower:
                return category
    return OpCategory.OTHER


def _estimate_flops(category: OpCategory, shapes: dict[str, Any], dtype: str = "float16") -> float:
    """Estimate FLOPs for an operation based on category and shapes."""
    if category == OpCategory.LINEAR:
        m = shapes.get("M", shapes.get("batch", 1))
        n = shapes.get("N", shapes.get("out_features", 1))
        k = shapes.get("K", shapes.get("in_features", 1))
        return 2.0 * m * n * k
    elif category == OpCategory.ATTENTION:
        b = shapes.get("B", shapes.get("batch", 1))
        h = shapes.get("H", shapes.get("heads", 1))
        s = shapes.get("S", shapes.get("seq_len", 1))
        d = shapes.get("D", shapes.get("head_dim", 128))
        # QK^T + softmax + AV
        return b * h * (2 * s * s * d + 2 * s * d * s)
    elif category in (OpCategory.NORM, OpCategory.ACTIVATION):
        m = shapes.get("M", shapes.get("batch", 1))
        n = shapes.get("N", shapes.get("dim", 1))
        return 5.0 * m * n  # approximate
    return 0.0


def _estimate_bytes(category: OpCategory, shapes: dict[str, Any], dtype: str = "float16") -> float:
    """Estimate bytes accessed for an operation."""
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4, "float8": 1, "fp4": 0.5}.get(dtype, 2)

    if category == OpCategory.LINEAR:
        m = shapes.get("M", shapes.get("batch", 1))
        n = shapes.get("N", shapes.get("out_features", 1))
        k = shapes.get("K", shapes.get("in_features", 1))
        # Read A (M*K) + B (K*N) + write C (M*N)
        return (m * k + k * n + m * n) * dtype_bytes
    elif category == OpCategory.ATTENTION:
        b = shapes.get("B", shapes.get("batch", 1))
        h = shapes.get("H", shapes.get("heads", 1))
        s = shapes.get("S", shapes.get("seq_len", 1))
        d = shapes.get("D", shapes.get("head_dim", 128))
        return b * h * (3 * s * d + s * d) * dtype_bytes  # Q,K,V read + output write
    elif category in (OpCategory.NORM, OpCategory.ACTIVATION):
        m = shapes.get("M", shapes.get("batch", 1))
        n = shapes.get("N", shapes.get("dim", 1))
        return 2 * m * n * dtype_bytes  # read + write
    return 0.0


# ---------------------------------------------------------------------------
# ModelProfiler
# ---------------------------------------------------------------------------

class ModelProfiler:
    """
    Profile a model and decompose its inference step into per-operation metrics.

    Supports three profiling modes:
    1. torch_profiler: Uses torch.profiler to trace CUDA kernels (requires model on GPU)
    2. profile_json: Parses an existing profile_report.json from AutoKernel v1
    3. manual: Accepts manually specified operation breakdowns

    For serving frameworks (vLLM, SGLang), use the API-level timing mode
    which measures end-to-end latency and estimates component breakdown
    from known model architectures.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def profile(
        self,
        model_path: str,
        gpu_info: Optional[GPUInfo] = None,
        mode: str = "auto",
        batch_size: int = 1,
        seq_len: int = 1,
        **kwargs: Any,
    ) -> ProfileResult:
        """
        Profile a model and return per-operation breakdown.

        Args:
            model_path: Path to model directory, HF model ID, or profile JSON
            gpu_info: GPU specification (auto-detected if None)
            mode: "auto", "torch_profiler", "profile_json", "architecture", "manual"
            batch_size: Batch size for profiling
            seq_len: Sequence length for profiling

        Returns:
            ProfileResult with per-operation timing and utilization
        """
        if gpu_info is None:
            gpu_info = detect_gpu_runtime()

        # Auto-detect profiling mode
        if mode == "auto":
            if model_path.endswith(".json"):
                mode = "profile_json"
            elif os.path.isdir(model_path):
                # Check if it has a config.json (HF model)
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    mode = "architecture"
                else:
                    mode = "torch_profiler"
            else:
                mode = "architecture"

        if mode == "profile_json":
            return self._profile_from_json(model_path, gpu_info)
        elif mode == "architecture":
            return self._profile_from_architecture(model_path, gpu_info, batch_size, seq_len)
        elif mode == "torch_profiler":
            return self._profile_with_torch(model_path, gpu_info, batch_size, seq_len, **kwargs)
        elif mode == "manual":
            return self._profile_manual(model_path, gpu_info, **kwargs)
        else:
            raise ValueError(f"Unknown profiling mode: {mode}")

    # ------------------------------------------------------------------
    # Mode: Parse existing profile JSON
    # ------------------------------------------------------------------

    def _profile_from_json(self, json_path: str, gpu_info: GPUInfo) -> ProfileResult:
        """Parse an AutoKernel v1 profile_report.json."""
        with open(json_path) as f:
            data = json.load(f)

        model_name = data.get("model_name", os.path.basename(json_path))
        ops = []
        total_time = 0.0

        for kernel in data.get("kernels", []):
            name = kernel.get("name", "unknown")
            time_us = kernel.get("cuda_time_us", kernel.get("time_us", 0.0))
            total_time += time_us

            category = classify_op(name)
            shapes = kernel.get("shapes", {})
            flops = _estimate_flops(category, shapes)
            bytes_accessed = _estimate_bytes(category, shapes)

            ops.append(OpProfile(
                name=name,
                category=category,
                time_us=time_us,
                flops=flops,
                bytes_accessed=bytes_accessed,
                shapes=shapes,
                kernel_names=[name],
            ))

        # Compute derived fields
        for op in ops:
            op.time_fraction = op.time_us / total_time if total_time > 0 else 0.0
            if op.flops > 0 and op.time_us > 0:
                achieved_tflops = op.flops / (op.time_us * 1e6)
                op.utilization = min(achieved_tflops / gpu_info.peak_tflops_fp16, 1.0) if gpu_info.peak_tflops_fp16 > 0 else 0.0
            elif op.bytes_accessed > 0 and op.time_us > 0:
                achieved_bw = op.bytes_accessed / (op.time_us * 1e3)  # GB/s
                op.utilization = min(achieved_bw / gpu_info.peak_bandwidth_gb_s, 1.0) if gpu_info.peak_bandwidth_gb_s > 0 else 0.0
            if op.bytes_accessed > 0:
                op.arithmetic_intensity = op.flops / op.bytes_accessed

        return ProfileResult(
            model_name=model_name,
            gpu=gpu_info,
            total_time_us=total_time,
            ops=ops,
        )

    # ------------------------------------------------------------------
    # Mode: Architecture-based analytical profiling
    # ------------------------------------------------------------------

    def _profile_from_architecture(
        self,
        model_path: str,
        gpu_info: GPUInfo,
        batch_size: int = 1,
        seq_len: int = 1,
    ) -> ProfileResult:
        """
        Analytically profile a model based on its architecture config.

        This mode does not require a GPU -- it estimates timing from the model
        architecture, known operation costs, and GPU peak specs. Useful for
        planning optimization before running the actual model.
        """
        config = self._load_model_config(model_path)
        model_name = config.get("model_type", os.path.basename(model_path))

        # Extract architecture params
        hidden = config.get("hidden_size", config.get("d_model", 4096))
        num_layers = config.get("num_hidden_layers", config.get("n_layer", 32))
        num_heads = config.get("num_attention_heads", config.get("n_head", 32))
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden // num_heads)
        intermediate = config.get("intermediate_size", config.get("n_inner", hidden * 4))
        vocab_size = config.get("vocab_size", 32000)
        num_experts = config.get("num_local_experts", config.get("num_experts", 0))
        experts_per_token = config.get("num_experts_per_tok", 2)
        rope_dim = config.get("rope_dim", head_dim)

        # Determine if MoE
        is_moe = num_experts > 0

        # Dtype info
        dtype_str = config.get("torch_dtype", "bfloat16")
        dtype_bytes = 2 if "16" in dtype_str else (1 if "8" in dtype_str else (0.5 if "4" in dtype_str else 4))

        # Compute weight memory
        # Per layer: Q, K, V, O projections + MLP
        qkv_params = hidden * (num_heads + 2 * num_kv_heads) * head_dim
        o_params = num_heads * head_dim * hidden
        if is_moe:
            mlp_params = num_experts * 3 * hidden * intermediate  # gate, up, down per expert
        else:
            mlp_params = 3 * hidden * intermediate  # gate, up, down
        norm_params = 2 * hidden  # 2 norms per layer
        params_per_layer = qkv_params + o_params + mlp_params + norm_params
        embed_params = vocab_size * hidden
        total_params = num_layers * params_per_layer + embed_params

        weights_mb = total_params * dtype_bytes / (1024 ** 2)

        # KV cache memory for decode (single token generation)
        kv_per_layer = 2 * batch_size * num_kv_heads * seq_len * head_dim * dtype_bytes
        kv_cache_mb = num_layers * kv_per_layer / (1024 ** 2)

        # Build per-operation profiles for ONE decode step
        ops = []
        total_time_us = 0.0

        for layer_idx in range(num_layers):
            prefix = f"layer_{layer_idx}"

            # --- RMSNorm (pre-attention) ---
            norm_shapes = {"M": batch_size, "N": hidden}
            norm_flops = 5.0 * batch_size * hidden
            norm_bytes = 2 * batch_size * hidden * dtype_bytes
            norm_time = self._estimate_time_us(norm_flops, norm_bytes, gpu_info)
            ops.append(OpProfile(
                name=f"{prefix}.input_norm",
                category=OpCategory.NORM,
                time_us=norm_time,
                flops=norm_flops,
                bytes_accessed=norm_bytes,
                shapes=norm_shapes,
                dtype=dtype_str,
            ))
            total_time_us += norm_time

            # --- QKV projection (linear) ---
            # For decode: M=batch_size (single token), K=hidden, N=qkv_out
            qkv_n = (num_heads + 2 * num_kv_heads) * head_dim
            qkv_shapes = {"M": batch_size, "N": qkv_n, "K": hidden}
            qkv_flops = 2.0 * batch_size * qkv_n * hidden
            qkv_bytes = (batch_size * hidden + hidden * qkv_n + batch_size * qkv_n) * dtype_bytes
            qkv_time = self._estimate_time_us(qkv_flops, qkv_bytes, gpu_info)
            ops.append(OpProfile(
                name=f"{prefix}.qkv_proj",
                category=OpCategory.LINEAR,
                time_us=qkv_time,
                flops=qkv_flops,
                bytes_accessed=qkv_bytes,
                shapes=qkv_shapes,
                dtype=dtype_str,
            ))
            total_time_us += qkv_time

            # --- Rotary embedding ---
            rope_shapes = {"B": batch_size, "H": num_heads, "D": rope_dim}
            rope_flops = batch_size * num_heads * rope_dim * 6
            rope_bytes = 2 * batch_size * num_heads * rope_dim * dtype_bytes
            rope_time = self._estimate_time_us(rope_flops, rope_bytes, gpu_info)
            ops.append(OpProfile(
                name=f"{prefix}.rotary",
                category=OpCategory.EMBEDDING,
                time_us=rope_time,
                flops=rope_flops,
                bytes_accessed=rope_bytes,
                shapes=rope_shapes,
                dtype=dtype_str,
            ))
            total_time_us += rope_time

            # --- Attention (decode = single query against KV cache) ---
            attn_shapes = {"B": batch_size, "H": num_heads, "S": seq_len, "D": head_dim}
            # QK^T: batch * heads * 1 * seq_len * head_dim (dot product per head)
            # AV: batch * heads * 1 * head_dim * seq_len
            attn_flops = 2.0 * batch_size * num_heads * seq_len * head_dim * 2
            # Read K cache + V cache + Q
            attn_bytes = batch_size * (
                num_kv_heads * seq_len * head_dim * 2 +  # K+V cache read
                num_heads * head_dim +  # Q read
                num_heads * head_dim  # output write
            ) * dtype_bytes
            attn_time = self._estimate_time_us(attn_flops, attn_bytes, gpu_info, op_type="attention")
            ops.append(OpProfile(
                name=f"{prefix}.attention",
                category=OpCategory.ATTENTION,
                time_us=attn_time,
                flops=attn_flops,
                bytes_accessed=attn_bytes,
                shapes=attn_shapes,
                dtype=dtype_str,
            ))
            total_time_us += attn_time

            # --- Output projection ---
            o_n = hidden
            o_k = num_heads * head_dim
            o_shapes = {"M": batch_size, "N": o_n, "K": o_k}
            o_flops = 2.0 * batch_size * o_n * o_k
            o_bytes = (batch_size * o_k + o_k * o_n + batch_size * o_n) * dtype_bytes
            o_time = self._estimate_time_us(o_flops, o_bytes, gpu_info)
            ops.append(OpProfile(
                name=f"{prefix}.o_proj",
                category=OpCategory.LINEAR,
                time_us=o_time,
                flops=o_flops,
                bytes_accessed=o_bytes,
                shapes=o_shapes,
                dtype=dtype_str,
            ))
            total_time_us += o_time

            # --- RMSNorm (post-attention) ---
            ops.append(OpProfile(
                name=f"{prefix}.post_attn_norm",
                category=OpCategory.NORM,
                time_us=norm_time,
                flops=norm_flops,
                bytes_accessed=norm_bytes,
                shapes=norm_shapes,
                dtype=dtype_str,
            ))
            total_time_us += norm_time

            # --- MLP ---
            if is_moe:
                # MoE routing
                route_shapes = {"M": batch_size, "N": num_experts}
                route_flops = 2.0 * batch_size * hidden * num_experts
                route_bytes = (batch_size * hidden + hidden * num_experts) * dtype_bytes
                route_time = self._estimate_time_us(route_flops, route_bytes, gpu_info)
                ops.append(OpProfile(
                    name=f"{prefix}.moe_routing",
                    category=OpCategory.MOE_ROUTING,
                    time_us=route_time,
                    flops=route_flops,
                    bytes_accessed=route_bytes,
                    shapes=route_shapes,
                    dtype=dtype_str,
                ))
                total_time_us += route_time

                # Expert MLP (gate + up + down, for experts_per_token experts)
                expert_mlp_flops = experts_per_token * (
                    2.0 * batch_size * intermediate * hidden +  # gate
                    2.0 * batch_size * intermediate * hidden +  # up
                    2.0 * batch_size * hidden * intermediate    # down
                )
                expert_mlp_bytes = experts_per_token * (
                    (hidden * intermediate * 2 + batch_size * hidden + batch_size * intermediate) * 2 +
                    (intermediate * hidden + batch_size * intermediate + batch_size * hidden)
                ) * dtype_bytes
                expert_time = self._estimate_time_us(expert_mlp_flops, expert_mlp_bytes, gpu_info)
                ops.append(OpProfile(
                    name=f"{prefix}.moe_experts",
                    category=OpCategory.LINEAR,
                    time_us=expert_time,
                    flops=expert_mlp_flops,
                    bytes_accessed=expert_mlp_bytes,
                    shapes={"M": batch_size, "N": intermediate, "K": hidden,
                            "num_experts": num_experts, "active_experts": experts_per_token},
                    dtype=dtype_str,
                ))
                total_time_us += expert_time
            else:
                # Standard MLP: gate_proj, up_proj, down_proj
                # gate + up: 2 * (M * intermediate * hidden)
                # down: M * hidden * intermediate
                mlp_flops = 2.0 * batch_size * intermediate * hidden * 3
                mlp_bytes = (
                    2 * (hidden * intermediate + batch_size * hidden + batch_size * intermediate) +
                    (intermediate * hidden + batch_size * intermediate + batch_size * hidden)
                ) * dtype_bytes
                mlp_time = self._estimate_time_us(mlp_flops, mlp_bytes, gpu_info)
                mlp_shapes = {"M": batch_size, "N": intermediate, "K": hidden}
                ops.append(OpProfile(
                    name=f"{prefix}.mlp",
                    category=OpCategory.LINEAR,
                    time_us=mlp_time,
                    flops=mlp_flops,
                    bytes_accessed=mlp_bytes,
                    shapes=mlp_shapes,
                    dtype=dtype_str,
                ))
                total_time_us += mlp_time

        # Recompute time fractions
        for op in ops:
            op.time_fraction = op.time_us / total_time_us if total_time_us > 0 else 0.0
            if op.flops > 0 and op.time_us > 0:
                achieved_tflops = op.flops / (op.time_us * 1e6)
                peak = gpu_info.peak_tflops_fp16 if gpu_info.peak_tflops_fp16 > 0 else 1.0
                op.utilization = min(achieved_tflops / peak, 1.0)
            if op.bytes_accessed > 0:
                op.arithmetic_intensity = op.flops / op.bytes_accessed

        # Aggregate ops by category for cleaner profile
        result = ProfileResult(
            model_name=model_name,
            gpu=gpu_info,
            total_time_us=total_time_us,
            ops=self._aggregate_ops(ops),
            memory_total_mb=weights_mb + kv_cache_mb,
            memory_kv_cache_mb=kv_cache_mb,
            memory_weights_mb=weights_mb,
            batch_size=batch_size,
            sequence_length=seq_len,
            metadata={
                "mode": "architecture",
                "num_layers": num_layers,
                "hidden_size": hidden,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "intermediate_size": intermediate,
                "is_moe": is_moe,
                "num_experts": num_experts,
                "total_params": total_params,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Mode: torch.profiler (requires GPU + model loaded)
    # ------------------------------------------------------------------

    def _profile_with_torch(
        self,
        model_path: str,
        gpu_info: GPUInfo,
        batch_size: int = 1,
        seq_len: int = 1,
        **kwargs: Any,
    ) -> ProfileResult:
        """
        Profile using torch.profiler. Requires the model to be loadable.
        Falls back to architecture mode if torch.profiler is unavailable.
        """
        try:
            import torch
            from torch.profiler import profile, ProfilerActivity
        except ImportError:
            if self.verbose:
                print("[profiler] torch.profiler unavailable, falling back to architecture mode")
            return self._profile_from_architecture(model_path, gpu_info, batch_size, seq_len)

        # Try to load model
        model = self._load_model(model_path)
        if model is None:
            if self.verbose:
                print("[profiler] Could not load model, falling back to architecture mode")
            return self._profile_from_architecture(model_path, gpu_info, batch_size, seq_len)

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create dummy input
        dummy_input = self._create_dummy_input(model, batch_size, seq_len, device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(**dummy_input) if isinstance(dummy_input, dict) else model(dummy_input)

        # Profile
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof:
                for _ in range(5):
                    model(**dummy_input) if isinstance(dummy_input, dict) else model(dummy_input)

        # Parse profiler output
        ops = []
        total_time = 0.0

        for event in prof.key_averages():
            if event.device_type is not None and "cuda" in str(event.device_type).lower():
                time_us = event.cuda_time_total / 5.0  # average over iterations
                if time_us < 0.1:  # skip trivial ops
                    continue

                category = classify_op(event.key)
                flops = event.flops or 0
                shapes = {}
                if event.input_shapes:
                    shapes["input_shapes"] = [list(s) for s in event.input_shapes if s]

                ops.append(OpProfile(
                    name=event.key,
                    category=category,
                    time_us=time_us,
                    flops=flops / 5.0,
                    call_count=event.count // 5,
                    shapes=shapes,
                    kernel_names=[event.key],
                ))
                total_time += time_us

        # Compute derived fields
        for op in ops:
            op.time_fraction = op.time_us / total_time if total_time > 0 else 0.0
            if op.flops > 0 and op.time_us > 0:
                achieved_tflops = op.flops / (op.time_us * 1e6)
                peak = gpu_info.peak_tflops_fp16 if gpu_info.peak_tflops_fp16 > 0 else 1.0
                op.utilization = min(achieved_tflops / peak, 1.0)

        return ProfileResult(
            model_name=os.path.basename(model_path),
            gpu=gpu_info,
            total_time_us=total_time,
            ops=self._aggregate_ops(ops),
            batch_size=batch_size,
            sequence_length=seq_len,
            metadata={"mode": "torch_profiler"},
        )

    # ------------------------------------------------------------------
    # Mode: Manual specification
    # ------------------------------------------------------------------

    def _profile_manual(
        self,
        model_path: str,
        gpu_info: GPUInfo,
        ops: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> ProfileResult:
        """Create a profile from manually specified operation data."""
        if ops is None:
            ops = kwargs.get("ops_data", [])

        profiles = []
        total_time = 0.0

        for op_data in ops:
            time_us = op_data.get("time_us", 0.0)
            total_time += time_us
            category = OpCategory(op_data.get("category", "other"))

            profiles.append(OpProfile(
                name=op_data.get("name", "unknown"),
                category=category,
                time_us=time_us,
                flops=op_data.get("flops", 0.0),
                bytes_accessed=op_data.get("bytes_accessed", 0.0),
                memory_mb=op_data.get("memory_mb", 0.0),
                shapes=op_data.get("shapes", {}),
                dtype=op_data.get("dtype", "float16"),
            ))

        for op in profiles:
            op.time_fraction = op.time_us / total_time if total_time > 0 else 0.0

        return ProfileResult(
            model_name=os.path.basename(model_path),
            gpu=gpu_info,
            total_time_us=total_time,
            ops=profiles,
            metadata={"mode": "manual"},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_time_us(
        self,
        flops: float,
        bytes_accessed: float,
        gpu_info: GPUInfo,
        op_type: str = "compute",
    ) -> float:
        """
        Estimate operation time using the roofline model.

        Time = max(compute_time, memory_time)
        For decode steps, most operations are memory-bound.
        """
        # Compute time (assuming peak throughput)
        peak_tflops = gpu_info.peak_tflops_fp16 if gpu_info.peak_tflops_fp16 > 0 else 100.0
        compute_time_us = flops / (peak_tflops * 1e6) if flops > 0 else 0.0

        # Memory time
        peak_bw = gpu_info.peak_bandwidth_gb_s if gpu_info.peak_bandwidth_gb_s > 0 else 1000.0
        memory_time_us = bytes_accessed / (peak_bw * 1e3) if bytes_accessed > 0 else 0.0

        # Efficiency factors (real hardware never hits peak)
        compute_efficiency = 0.70  # typical for small-batch GEMM
        memory_efficiency = 0.80   # typical for sequential reads

        if op_type == "attention":
            # Attention decode is heavily memory-bound (reading KV cache)
            memory_efficiency = 0.75
            compute_efficiency = 0.50

        adjusted_compute = compute_time_us / compute_efficiency if compute_efficiency > 0 else compute_time_us
        adjusted_memory = memory_time_us / memory_efficiency if memory_efficiency > 0 else memory_time_us

        # Roofline: time is the max of compute and memory
        return max(adjusted_compute, adjusted_memory)

    def _aggregate_ops(self, ops: list[OpProfile]) -> list[OpProfile]:
        """Aggregate per-layer ops into category-level summaries."""
        by_category: dict[str, list[OpProfile]] = {}
        for op in ops:
            key = f"{op.category.value}"
            if key not in by_category:
                by_category[key] = []
            by_category[key].append(op)

        aggregated = []
        total_time = sum(op.time_us for op in ops)

        for cat_key, cat_ops in by_category.items():
            total_us = sum(o.time_us for o in cat_ops)
            total_flops = sum(o.flops for o in cat_ops)
            total_bytes = sum(o.bytes_accessed for o in cat_ops)
            total_mem = sum(o.memory_mb for o in cat_ops)
            avg_util = (
                sum(o.utilization * o.time_us for o in cat_ops) / total_us
                if total_us > 0 else 0.0
            )

            # Use a representative shape from the first op
            rep_shapes = cat_ops[0].shapes if cat_ops else {}

            aggregated.append(OpProfile(
                name=cat_key,
                category=cat_ops[0].category,
                time_us=total_us,
                time_fraction=total_us / total_time if total_time > 0 else 0.0,
                flops=total_flops,
                bytes_accessed=total_bytes,
                utilization=avg_util,
                memory_mb=total_mem,
                call_count=len(cat_ops),
                shapes=rep_shapes,
                dtype=cat_ops[0].dtype,
                arithmetic_intensity=total_flops / total_bytes if total_bytes > 0 else 0.0,
            ))

        return sorted(aggregated, key=lambda o: o.time_us, reverse=True)

    def _load_model_config(self, model_path: str) -> dict[str, Any]:
        """Load model config from a directory or HF model ID."""
        # Direct path to config.json
        config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else model_path
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)

        # Try HF-style model ID by looking for common config locations
        for candidate in [
            os.path.join(model_path, "config.json"),
            model_path + ".json",
        ]:
            if os.path.exists(candidate):
                with open(candidate) as f:
                    return json.load(f)

        # Return a default config for unknown models
        return {
            "model_type": "unknown",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "vocab_size": 32000,
        }

    def _load_model(self, model_path: str) -> Any:
        """Try to load a PyTorch model from path."""
        try:
            import torch

            # Try loading as a state dict / checkpoint
            if model_path.endswith((".pt", ".pth", ".bin")):
                return torch.load(model_path, map_location="cpu", weights_only=False)

            # Try as a HF model
            try:
                from transformers import AutoModelForCausalLM
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            except ImportError:
                pass

            return None
        except Exception:
            return None

    def _create_dummy_input(
        self,
        model: Any,
        batch_size: int,
        seq_len: int,
        device: Any,
    ) -> Any:
        """Create dummy input for profiling."""
        import torch
        return {
            "input_ids": torch.randint(0, 32000, (batch_size, seq_len), device=device),
        }

    def to_json(self, result: ProfileResult) -> dict[str, Any]:
        """Serialize a ProfileResult to a JSON-compatible dict."""
        return {
            "model_name": result.model_name,
            "gpu": {
                "name": result.gpu.name,
                "memory_gb": result.gpu.memory_gb,
                "peak_tflops_fp16": result.gpu.peak_tflops_fp16,
                "peak_bandwidth_gb_s": result.gpu.peak_bandwidth_gb_s,
            },
            "total_time_us": result.total_time_us,
            "throughput_tokens_per_sec": result.throughput_tokens_per_sec,
            "memory": {
                "total_mb": result.memory_total_mb,
                "weights_mb": result.memory_weights_mb,
                "kv_cache_mb": result.memory_kv_cache_mb,
            },
            "ops": [
                {
                    "name": op.name,
                    "category": op.category.value,
                    "time_us": round(op.time_us, 2),
                    "time_fraction": round(op.time_fraction, 4),
                    "utilization": round(op.utilization, 4),
                    "flops": op.flops,
                    "bytes_accessed": op.bytes_accessed,
                    "arithmetic_intensity": round(op.arithmetic_intensity, 2),
                    "call_count": op.call_count,
                }
                for op in result.ops
            ],
            "metadata": result.metadata,
        }
