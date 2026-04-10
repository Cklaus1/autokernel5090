"""
AutoKernel v2 Knowledge Base -- Learned optimization patterns from real experiments.

Stores what worked for which model/GPU/shape combinations and enables transfer
learning: if optimization X helped model A, try it on model B with similar ops.

Seeded with real data from AutoKernel v1 experiments (90+ experiments on W4A16,
FusenCache KV cache compression, NVFP4 Blackwell optimizations).

Usage:
    from autokernel_v2.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    matches = kb.lookup(category=OpCategory.LINEAR, gpu_arch="hopper_blackwell")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import OpCategory


@dataclass
class Pattern:
    """A learned optimization pattern."""
    name: str
    description: str
    category: OpCategory
    strategy: str  # fusion, quantization, algorithm, config, memory
    condition: str  # human-readable condition for when to apply
    expected_impact: float  # speedup multiplier
    confidence: float  # 0-1
    effort: str  # low, medium, high
    implementation_steps: list[str] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    validated_on: list[str] = field(default_factory=list)  # model/GPU combos
    gpu_archs: list[str] = field(default_factory=list)  # compatible GPU architectures
    shape_constraints: dict[str, Any] = field(default_factory=dict)
    experiment_ids: list[int] = field(default_factory=list)  # source experiments


class KnowledgeBase:
    """
    Store and retrieve optimization patterns.

    The knowledge base is seeded with patterns discovered during AutoKernel v1
    experiments (90+ experiments across W4A16 matmul, NVFP4, FusenCache, and
    serving optimization sessions).
    """

    def __init__(self):
        self.patterns: list[Pattern] = list(_SEED_PATTERNS)

    def lookup(
        self,
        category: Optional[OpCategory] = None,
        gpu_arch: Optional[str] = None,
        shapes: Optional[dict[str, Any]] = None,
        strategy: Optional[str] = None,
    ) -> list[Pattern]:
        """
        Find patterns matching the given criteria.

        Returns patterns sorted by (confidence * expected_impact), highest first.
        """
        matches = []

        for p in self.patterns:
            # Category filter
            if category is not None and p.category != category:
                continue

            # GPU architecture filter
            if gpu_arch is not None and p.gpu_archs:
                if not any(gpu_arch.startswith(a) or a in gpu_arch for a in p.gpu_archs):
                    continue

            # Strategy filter
            if strategy is not None and p.strategy != strategy:
                continue

            # Shape constraints
            if shapes and p.shape_constraints:
                if not self._shapes_match(shapes, p.shape_constraints):
                    continue

            matches.append(p)

        matches.sort(key=lambda p: p.confidence * p.expected_impact, reverse=True)
        return matches

    def add_pattern(self, pattern: Pattern) -> None:
        """Add a new pattern to the knowledge base."""
        # Check for duplicate by name
        for i, existing in enumerate(self.patterns):
            if existing.name == pattern.name:
                self.patterns[i] = pattern
                return
        self.patterns.append(pattern)

    def update_confidence(self, name: str, success: bool, model_gpu: str = "") -> None:
        """Update pattern confidence based on experiment result."""
        for p in self.patterns:
            if p.name == name:
                # Bayesian-style update
                if success:
                    p.confidence = min(p.confidence * 1.1 + 0.05, 0.99)
                    if model_gpu and model_gpu not in p.validated_on:
                        p.validated_on.append(model_gpu)
                else:
                    p.confidence = max(p.confidence * 0.8 - 0.05, 0.01)
                return

    def _shapes_match(self, actual: dict, constraints: dict) -> bool:
        """Check if actual shapes satisfy constraints."""
        for key, constraint in constraints.items():
            if key not in actual:
                continue
            val = actual[key]
            if isinstance(constraint, dict):
                if "min" in constraint and val < constraint["min"]:
                    return False
                if "max" in constraint and val > constraint["max"]:
                    return False
            elif isinstance(constraint, (int, float)):
                if val != constraint:
                    return False
        return True

    def summary(self) -> str:
        """Format a summary of all patterns in the knowledge base."""
        lines = [
            "=" * 90,
            f"KNOWLEDGE BASE: {len(self.patterns)} patterns",
            "=" * 90,
            f"{'Name':<35} {'Category':<14} {'Strategy':<12} {'Impact':<8} {'Conf':<6} {'Validated On'}",
            "-" * 90,
        ]

        for p in sorted(self.patterns, key=lambda x: x.confidence * x.expected_impact, reverse=True):
            validated = ", ".join(p.validated_on[:3])
            if len(p.validated_on) > 3:
                validated += f" (+{len(p.validated_on)-3})"
            lines.append(
                f"{p.name:<35} {p.category.value:<14} {p.strategy:<12} "
                f"{p.expected_impact:>5.1f}x  {p.confidence:>4.0%}  {validated}"
            )

        lines.append("=" * 90)
        return "\n".join(lines)

    def to_dict(self) -> list[dict]:
        """Serialize all patterns to a list of dicts."""
        result = []
        for p in self.patterns:
            result.append({
                "name": p.name,
                "description": p.description,
                "category": p.category.value,
                "strategy": p.strategy,
                "condition": p.condition,
                "expected_impact": p.expected_impact,
                "confidence": p.confidence,
                "effort": p.effort,
                "implementation_steps": p.implementation_steps,
                "prerequisites": p.prerequisites,
                "risks": p.risks,
                "validated_on": p.validated_on,
                "gpu_archs": p.gpu_archs,
                "shape_constraints": p.shape_constraints,
                "experiment_ids": p.experiment_ids,
            })
        return result


# ---------------------------------------------------------------------------
# Seed patterns from real AutoKernel v1 experiments
# ---------------------------------------------------------------------------

_SEED_PATTERNS: list[Pattern] = [

    # --- W4A16 Matmul patterns (from 90 experiments) ---

    Pattern(
        name="split_dequant_cublas",
        description=(
            "Split quantized matmul into separate Triton dequant kernel + cuBLAS FP16 matmul. "
            "cuBLAS is extremely hard to beat for the matmul itself, so let Triton handle only "
            "the dequantization and use F.linear for the GEMM."
        ),
        category=OpCategory.LINEAR,
        strategy="algorithm",
        condition="quantized weight matmul (W4A16, W8A16, NVFP4)",
        expected_impact=1.55,
        confidence=0.90,
        effort="medium",
        implementation_steps=[
            "Write Triton kernel that dequantizes INT4 -> FP16 (unpack + scale + zero)",
            "Cache dequantized FP16 weights by tensor identity (avoid redundant dequant)",
            "Use F.linear (NT cuBLAS GEMM) instead of torch.mm (4% faster)",
            "Autotune dequant block sizes (BLOCK_N, BLOCK_K)",
        ],
        prerequisites=["Weight quantization with scales and zeros"],
        risks=["Extra memory for dequantized weight cache"],
        validated_on=["w4a16-rtx5090", "w4a16-a100"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[60, 61, 62, 74],
    ),

    Pattern(
        name="fp16_accumulator",
        description=(
            "Use FP16 accumulator in Triton matmul instead of FP32. Doubles tensor core "
            "throughput at the cost of some numerical precision. Safe for inference."
        ),
        category=OpCategory.LINEAR,
        strategy="config",
        condition="large GEMM (M*N*K > 1e8) where accuracy loss is acceptable",
        expected_impact=1.8,
        confidence=0.75,
        effort="low",
        implementation_steps=[
            "Set out_dtype=tl.float16 in Triton tl.dot",
            "Validate numerical accuracy on representative inputs",
            "Focus autotune on BK=128 configs (best for FP16 acc)",
        ],
        risks=["Numerical accuracy loss (typically < 0.1% relative error)"],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[75, 76],
    ),

    Pattern(
        name="flat_k_loop",
        description=(
            "Use a flat (single) K-reduction loop instead of nested loops. "
            "Triton's compiler pipelines single loops much better than nested ones."
        ),
        category=OpCategory.LINEAR,
        strategy="algorithm",
        condition="any Triton matmul kernel with nested K loop",
        expected_impact=1.25,
        confidence=0.85,
        effort="low",
        implementation_steps=[
            "Replace nested group/tile K loops with single flat loop",
            "Simplify mask computation",
            "Remove branch on group boundaries",
        ],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[36],
    ),

    Pattern(
        name="constexpr_group_size",
        description=(
            "Make quantization group_size a tl.constexpr parameter. Enables "
            "compile-time division optimization, eliminating expensive integer divides."
        ),
        category=OpCategory.LINEAR,
        strategy="config",
        condition="quantized kernel with group_size parameter",
        expected_impact=1.05,
        confidence=0.90,
        effort="low",
        implementation_steps=[
            "Change group_size function parameter to QUANT_GROUP_SIZE: tl.constexpr",
            "All group_size divisions become compile-time shifts",
        ],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[39],
    ),

    Pattern(
        name="nt_layout_cublas",
        description=(
            "Use F.linear (NT GEMM) instead of torch.mm (NN GEMM) for cuBLAS. "
            "NT layout is ~4% faster because of better memory access patterns."
        ),
        category=OpCategory.LINEAR,
        strategy="config",
        condition="any cuBLAS GEMM call",
        expected_impact=1.04,
        confidence=0.95,
        effort="low",
        implementation_steps=[
            "Store dequantized weights in transposed layout",
            "Use F.linear(input, weight) instead of torch.mm(input, weight.T)",
        ],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[61],
    ),

    Pattern(
        name="autotune_broad_then_narrow",
        description=(
            "Start with 20+ autotune configs, benchmark all, then trim to the top 4. "
            "The broad sweep finds unexpected winners; the trim reduces compilation time."
        ),
        category=OpCategory.LINEAR,
        strategy="config",
        condition="any Triton kernel with @triton.autotune",
        expected_impact=1.15,
        confidence=0.85,
        effort="low",
        implementation_steps=[
            "Start with configs: BM={64,128}, BN={64,128,256}, BK={32,64,128}, stages={2,3,4}, warps={4,8}",
            "Run full benchmark sweep",
            "Keep only top 4 configs by throughput",
            "Re-benchmark with trimmed set to confirm",
        ],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[21, 86],
    ),

    Pattern(
        name="bk128_triton36",
        description=(
            "BLOCK_SIZE_K=128 with Triton 3.6.0+ gives best throughput for FP16-accumulate "
            "matmuls. Earlier Triton versions had register spill issues with BK=128."
        ),
        category=OpCategory.LINEAR,
        strategy="config",
        condition="Triton >= 3.6.0 and FP16 accumulation enabled",
        expected_impact=1.15,
        confidence=0.80,
        effort="low",
        implementation_steps=[
            "Add BK=128 configs to autotune sweep",
            "Ensure Triton >= 3.6.0 (older versions spill registers)",
            "Pair with num_stages=2 or 3 (stages>4 causes shmem overflow)",
        ],
        risks=["Requires Triton >= 3.6.0"],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[82],
    ),

    Pattern(
        name="aligned_boundary_skip",
        description=(
            "Skip boundary checks when matrix dimensions are aligned to block sizes. "
            "Saves branch instructions and enables vectorized loads."
        ),
        category=OpCategory.LINEAR,
        strategy="algorithm",
        condition="matrix dims divisible by block sizes",
        expected_impact=1.02,
        confidence=0.90,
        effort="low",
        implementation_steps=[
            "Add ALIGNED: tl.constexpr parameter",
            "When ALIGNED, skip tl.where masks on loads and stores",
            "Set ALIGNED=True in autotune configs when shapes are aligned",
        ],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[89],
    ),

    Pattern(
        name="dequant_weight_caching",
        description=(
            "Cache dequantized FP16 weights by tensor identity (id(tensor)). "
            "Avoids redundant dequantization when the same weight is reused."
        ),
        category=OpCategory.QUANTIZATION,
        strategy="memory",
        condition="quantized model with repeated forward passes",
        expected_impact=1.10,
        confidence=0.85,
        effort="low",
        implementation_steps=[
            "Use a dict mapping id(quantized_weight) -> dequantized_fp16",
            "Check cache before dequantizing",
            "Invalidate cache on weight update (rare in inference)",
        ],
        risks=["Extra VRAM for cached weights"],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[74],
    ),

    # --- NVFP4 patterns (from Blackwell experiments) ---

    Pattern(
        name="nvfp4_scaled_mm",
        description=(
            "Use torch._scaled_mm or torch._scaled_mm_v2 for native FP4 tensor core "
            "matmul on Blackwell GPUs. Achieves 1270 TFLOPS (3.9x best Triton kernel)."
        ),
        category=OpCategory.LINEAR,
        strategy="quantization",
        condition="Blackwell GPU (sm_100+) with NVFP4 quantized weights",
        expected_impact=3.9,
        confidence=0.85,
        effort="medium",
        implementation_steps=[
            "Convert weights to NVFP4 format using modelopt or manual quantization",
            "Use torch._scaled_mm_v2(a_fp4, b_fp4, scale_a, scale_b) for GEMM",
            "Handle per-block scaling factors correctly",
            "Validate numerical accuracy vs FP16 reference",
        ],
        prerequisites=["Blackwell GPU (RTX 5090, B200)", "PyTorch 2.6+", "CUDA 12.8+"],
        risks=["API is private and may change", "Quantization accuracy depends on calibration"],
        validated_on=["gemma4-27b-nvfp4-rtx5090"],
        gpu_archs=["hopper_blackwell"],
    ),

    # --- Serving optimization patterns ---

    Pattern(
        name="disable_inductor_for_moe",
        description=(
            "Disable torch.compile inductor for models with Mixture-of-Experts layers. "
            "Use CUDA graphs instead. Inductor handles dynamic expert routing poorly, "
            "causing 1.5-2.1x slowdown."
        ),
        category=OpCategory.MOE_ROUTING,
        strategy="config",
        condition="model has MoE layers and using torch.compile",
        expected_impact=2.0,
        confidence=0.90,
        effort="low",
        implementation_steps=[
            "Set compilation config: mode=None (disable inductor)",
            "Set cudagraph_mode=full (capture entire forward as CUDA graph)",
            "For vLLM: --compilation-config '{\"level\": 0}' disables inductor",
            "For SGLang: use --disable-torch-compile",
        ],
        risks=["CUDA graphs require static shapes (fixed batch size)"],
        validated_on=["gemma4-27b-nvfp4-vllm", "gemma4-27b-nvfp4-sglang"],
        gpu_archs=["ampere", "hopper_blackwell"],
    ),

    Pattern(
        name="fusencache_kv_compression",
        description=(
            "FusenCache: quantize KV cache to FP8 or INT4 with block-wise scaling. "
            "Achieves 2-4x KV memory compression with minimal quality loss, enabling "
            "larger batch sizes or longer sequences."
        ),
        category=OpCategory.ATTENTION,
        strategy="quantization",
        condition="KV cache memory > 50% of total at target batch size",
        expected_impact=1.36,
        confidence=0.75,
        effort="high",
        implementation_steps=[
            "Implement K quantization (FP8 E4M3 with per-channel scales)",
            "Implement V quantization (FP8 or INT4 with block-wise scales)",
            "Modify attention kernel to dequantize K/V on load",
            "Tune block size for quantization (64 or 128)",
            "Validate quality on representative prompts",
        ],
        risks=[
            "Accuracy loss on long contexts",
            "Extra compute for dequantization",
            "Requires custom attention kernel",
        ],
        validated_on=["gemma4-27b-nvfp4-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
    ),

    Pattern(
        name="eager_attention_for_quantized",
        description=(
            "Use eager (non-compiled) attention for quantized models. Flash attention "
            "may not support quantized KV cache formats natively."
        ),
        category=OpCategory.ATTENTION,
        strategy="config",
        condition="quantized model with KV cache compression",
        expected_impact=1.1,
        confidence=0.70,
        effort="low",
        implementation_steps=[
            "Set attn_implementation='eager' in model config",
            "Or use custom attention kernel that handles quantized KV",
        ],
        validated_on=["gemma4-27b-nvfp4-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
    ),

    # --- General GPU optimization patterns ---

    Pattern(
        name="l2_swizzle_grouping",
        description=(
            "Apply L2 cache-aware tile ordering (swizzle) in Triton matmul. "
            "Groups output tiles to improve L2 hit rate on shared input tiles."
        ),
        category=OpCategory.LINEAR,
        strategy="algorithm",
        condition="GEMM with large N dimension (N > 1024)",
        expected_impact=1.10,
        confidence=0.80,
        effort="low",
        implementation_steps=[
            "Add GROUP_SIZE_M parameter to Triton kernel",
            "Reorder pid mapping: group adjacent M-tiles together",
            "Autotune GROUP_SIZE_M in {4, 8, 16}",
        ],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[21],
    ),

    Pattern(
        name="persistent_kernel",
        description=(
            "Use persistent Triton kernel that streams tiles across SMs without relaunching. "
            "Reduces kernel launch overhead for large matrices."
        ),
        category=OpCategory.LINEAR,
        strategy="algorithm",
        condition="large GEMM (M*N > 1e6) and high SM count GPU",
        expected_impact=1.10,
        confidence=0.60,
        effort="high",
        implementation_steps=[
            "Set NUM_SMS = gpu.sm_count",
            "Each program loops over multiple output tiles",
            "Use flat tile indexing with pid remapping",
            "Be careful with num_stages (>4 causes shmem overflow)",
        ],
        risks=["Complex implementation", "May regress on small shapes"],
        validated_on=["w4a16-rtx5090"],
        gpu_archs=["ampere", "hopper_blackwell"],
        experiment_ids=[32, 34],
    ),

    Pattern(
        name="fused_residual_norm",
        description=(
            "Fuse residual addition with layer normalization into a single kernel. "
            "Saves one global memory read+write round trip per layer."
        ),
        category=OpCategory.NORM,
        strategy="fusion",
        condition="model uses residual connections followed by normalization",
        expected_impact=1.3,
        confidence=0.80,
        effort="medium",
        implementation_steps=[
            "Write Triton kernel: output = norm(x + residual)",
            "Load x and residual, add in registers, normalize, write output",
            "Handle weight and bias parameters",
        ],
        validated_on=["llama-7b-fp16", "gemma4-27b"],
        gpu_archs=["ampere", "hopper_blackwell"],
    ),

    Pattern(
        name="fused_gate_up_silu",
        description=(
            "Fuse gate projection + SiLU activation + up projection multiplication "
            "into a single kernel, halving memory traffic for the MLP gate."
        ),
        category=OpCategory.ACTIVATION,
        strategy="fusion",
        condition="SwiGLU or GeGLU MLP architecture",
        expected_impact=1.3,
        confidence=0.70,
        effort="medium",
        implementation_steps=[
            "Write Triton kernel: output = silu(x @ gate_weight) * (x @ up_weight)",
            "Tile over output dimension, accumulate both GEMMs in registers",
            "Apply SiLU and multiply before writing to global memory",
        ],
        validated_on=["llama-7b-fp16"],
        gpu_archs=["ampere", "hopper_blackwell"],
    ),
]
