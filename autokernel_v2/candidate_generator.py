"""
AutoKernel v2 Candidate Generator -- Generate optimization candidates for bottlenecks.

For each optimization target, generates 5-10 candidate optimizations drawn from:
  - Knowledge base (proven patterns from previous sessions)
  - Category-specific strategies (fusion, quantization, algorithm changes)
  - GPU-specific features (FP8/FP4, CUDA graphs, tensor cores)

Each candidate has an expected impact, implementation plan, and confidence score.

Usage:
    from autokernel_v2.candidate_generator import CandidateGenerator
    generator = CandidateGenerator(gpu_info)
    candidates = generator.generate(target)
"""

from __future__ import annotations

from .types import (
    Candidate,
    GPUInfo,
    OpCategory,
    OptimizationTarget,
)
from .knowledge_base import KnowledgeBase, Pattern


class CandidateGenerator:
    """
    Generate optimization candidates for a given bottleneck target.

    The generator combines:
    1. Knowledge base lookup (what worked before on similar ops/shapes)
    2. Category-specific strategies (different approaches for attention vs. linear vs. norm)
    3. GPU-specific optimizations (leverage FP8, FP4, CUDA graphs, etc.)
    """

    def __init__(self, gpu_info: GPUInfo, knowledge_base: KnowledgeBase | None = None):
        self.gpu = gpu_info
        self.kb = knowledge_base or KnowledgeBase()

    def generate(self, target: OptimizationTarget) -> list[Candidate]:
        """
        Generate candidate optimizations for a bottleneck target.

        Returns candidates sorted by (confidence * expected_impact), highest first.
        """
        candidates: list[Candidate] = []

        # 1. Knowledge base matches
        candidates.extend(self._from_knowledge_base(target))

        # 2. Category-specific strategies
        category_generators = {
            OpCategory.LINEAR: self._linear_candidates,
            OpCategory.ATTENTION: self._attention_candidates,
            OpCategory.NORM: self._norm_candidates,
            OpCategory.ACTIVATION: self._activation_candidates,
            OpCategory.MOE_ROUTING: self._moe_candidates,
            OpCategory.QUANTIZATION: self._quantization_candidates,
            OpCategory.MEMORY: self._memory_candidates,
            OpCategory.EMBEDDING: self._embedding_candidates,
        }

        gen_fn = category_generators.get(target.category)
        if gen_fn:
            candidates.extend(gen_fn(target))

        # 3. Universal strategies (applicable to any op)
        candidates.extend(self._universal_candidates(target))

        # Deduplicate by name
        seen = set()
        unique = []
        for c in candidates:
            if c.name not in seen:
                seen.add(c.name)
                unique.append(c)

        # Sort by expected value (confidence * impact)
        unique.sort(key=lambda c: c.confidence * c.expected_impact, reverse=True)

        return unique[:10]  # cap at 10 candidates

    # ------------------------------------------------------------------
    # Knowledge base candidates
    # ------------------------------------------------------------------

    def _from_knowledge_base(self, target: OptimizationTarget) -> list[Candidate]:
        """Look up proven optimization patterns from the knowledge base."""
        candidates = []
        matches = self.kb.lookup(
            category=target.category,
            gpu_arch=self.gpu.arch_family,
            shapes=target.shapes,
        )

        for pattern in matches:
            # Adjust confidence based on how well the pattern matches
            confidence = pattern.confidence
            if self.gpu.name not in " ".join(pattern.validated_on):
                confidence *= 0.7  # not validated on this exact GPU

            candidates.append(Candidate(
                name=f"kb_{pattern.name}",
                description=f"[Knowledge Base] {pattern.description}",
                target=target,
                strategy=pattern.strategy,
                expected_impact=pattern.expected_impact,
                implementation_plan=pattern.implementation_steps,
                effort=pattern.effort,
                confidence=confidence,
                prerequisites=pattern.prerequisites,
                risks=pattern.risks,
            ))

        return candidates

    # ------------------------------------------------------------------
    # Category-specific candidate generators
    # ------------------------------------------------------------------

    def _linear_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for linear/GEMM operations."""
        candidates = []
        m = target.shapes.get("M", target.shapes.get("batch", 1))
        n = target.shapes.get("N", 4096)
        k = target.shapes.get("K", 4096)

        # Candidate: Split dequant + cuBLAS
        if "quant" in target.dtype or "int" in target.dtype or "fp4" in target.dtype:
            candidates.append(Candidate(
                name="split_dequant_cublas",
                description="Separate Triton dequantization kernel + cuBLAS FP16 matmul via F.linear",
                target=target,
                strategy="algorithm",
                expected_impact=1.5,
                implementation_plan=[
                    "Write Triton kernel that dequantizes weight tile to FP16",
                    "Cache dequantized weights by tensor identity",
                    "Use F.linear (NT cuBLAS GEMM) for the matmul",
                    "Autotune dequant block sizes",
                ],
                effort="medium",
                confidence=0.8,
                risks=["Extra memory for dequantized cache"],
            ))

        # Candidate: FP16 accumulation (if shapes allow)
        if m * n * k > 1e9:  # large GEMM
            candidates.append(Candidate(
                name="fp16_accumulation",
                description="Use FP16 accumulator instead of FP32 for large GEMMs (2x throughput on tensor cores)",
                target=target,
                strategy="config",
                expected_impact=1.8,
                implementation_plan=[
                    "Set out_dtype=tl.float16 in Triton tl.dot",
                    "Validate numerical accuracy on representative inputs",
                    "Add autotune configs focused on BK=128",
                ],
                effort="low",
                confidence=0.6,
                risks=["Possible numerical accuracy loss for sensitive layers"],
            ))

        # Candidate: Triton kernel with autotuning
        candidates.append(Candidate(
            name="triton_autotune_gemm",
            description="Custom Triton GEMM with broad autotune sweep targeting this specific shape",
            target=target,
            strategy="algorithm",
            expected_impact=1.3,
            implementation_plan=[
                f"Write Triton matmul for M={m}, N={n}, K={k}",
                "Start with 20+ autotune configs covering BM={64,128}, BN={64,128,256}, BK={32,64,128}",
                "Add L2 swizzle grouping (GROUP_SIZE_M=8)",
                "Benchmark and narrow to top 4 configs",
            ],
            effort="medium",
            confidence=0.7,
        ))

        # Candidate: Persistent kernel (large shapes only)
        if m >= 32 and n >= 1024:
            candidates.append(Candidate(
                name="persistent_gemm",
                description="Persistent Triton GEMM kernel that reuses SMs across output tiles",
                target=target,
                strategy="algorithm",
                expected_impact=1.15,
                implementation_plan=[
                    "Implement tile-streaming persistent kernel",
                    f"Set NUM_SMS={self.gpu.sm_count}",
                    "Use flat K loop for better pipelining",
                    "Autotune num_stages and block sizes",
                ],
                effort="high",
                confidence=0.5,
                risks=["Complex implementation", "BK=128 may cause register spill"],
            ))

        # Candidate: Batch small GEMMs
        if m <= 4:  # very small batch (decode)
            candidates.append(Candidate(
                name="weight_prefetch_small_m",
                description="Optimize for small-M decode: prefetch weights, use vectorized loads",
                target=target,
                strategy="algorithm",
                expected_impact=1.2,
                implementation_plan=[
                    "Use memory-bound optimized kernel for M<=4",
                    "Maximize memory bandwidth utilization",
                    "Consider using cuBLAS directly (hard to beat for small M)",
                ],
                effort="low",
                confidence=0.5,
            ))

        return candidates

    def _attention_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for attention operations."""
        candidates = []
        seq_len = target.shapes.get("S", target.shapes.get("seq_len", 2048))
        num_heads = target.shapes.get("H", target.shapes.get("heads", 32))

        # FlashAttention / FlashDecoding
        candidates.append(Candidate(
            name="flash_attention_v2",
            description="Use FlashAttention-2 for fused QKV attention with O(N) memory",
            target=target,
            strategy="algorithm",
            expected_impact=2.0 if seq_len > 512 else 1.3,
            implementation_plan=[
                "Replace standard attention with flash_attn_func",
                "Ensure causal mask is handled natively",
                "For decode: use flash_attn_with_kvcache for paged KV",
            ],
            effort="low",
            confidence=0.9,
            prerequisites=["flash-attn package installed"],
        ))

        # KV cache compression
        candidates.append(Candidate(
            name="kv_cache_quantization",
            description="Quantize KV cache to FP8 or INT4 to reduce memory bandwidth",
            target=target,
            strategy="quantization",
            expected_impact=1.5,
            implementation_plan=[
                "Quantize K cache to FP8 (per-channel scales)",
                "Quantize V cache to FP8 or INT4",
                "Modify attention kernel to dequantize on-the-fly",
                "Validate accuracy on representative prompts",
            ],
            effort="high",
            confidence=0.7,
            risks=["Accuracy loss on long sequences"],
        ))

        # FlashInfer / PagedAttention
        candidates.append(Candidate(
            name="paged_attention",
            description="Use paged KV cache for better memory utilization in serving",
            target=target,
            strategy="memory",
            expected_impact=1.3,
            implementation_plan=[
                "Implement paged KV cache layout",
                "Use FlashInfer decode attention kernel",
                "Tune page size for this model's head dim",
            ],
            effort="medium",
            confidence=0.6,
        ))

        # Multi-head latent attention (for compatible architectures)
        if num_heads > 16:
            candidates.append(Candidate(
                name="grouped_query_attention",
                description="Switch to GQA if model supports it (fewer KV heads = less memory)",
                target=target,
                strategy="algorithm",
                expected_impact=1.4,
                implementation_plan=[
                    "Check if model config supports GQA",
                    "Reduce num_kv_heads while maintaining quality",
                    "Retune attention kernel for asymmetric heads",
                ],
                effort="high",
                confidence=0.4,
                prerequisites=["Model must support GQA architecture"],
            ))

        return candidates

    def _norm_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for normalization operations."""
        candidates = []

        # Fuse norm with adjacent op
        candidates.append(Candidate(
            name="fused_norm_linear",
            description="Fuse RMSNorm/LayerNorm with the following linear projection",
            target=target,
            strategy="fusion",
            expected_impact=1.5,
            implementation_plan=[
                "Write fused Triton kernel: norm + linear in one pass",
                "Load input once, normalize in registers, multiply by weight",
                "Avoids extra global memory round-trip",
            ],
            effort="medium",
            confidence=0.7,
        ))

        # Fuse residual add + norm
        candidates.append(Candidate(
            name="fused_residual_norm",
            description="Fuse residual addition with normalization",
            target=target,
            strategy="fusion",
            expected_impact=1.3,
            implementation_plan=[
                "Combine residual = x + attn_output and norm(residual) into one kernel",
                "Single memory read for both operations",
            ],
            effort="low",
            confidence=0.8,
        ))

        return candidates

    def _activation_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for activation function operations."""
        candidates = []

        candidates.append(Candidate(
            name="fused_gate_activation",
            description="Fuse gate projection + activation (SiLU/GELU) + up projection",
            target=target,
            strategy="fusion",
            expected_impact=1.4,
            implementation_plan=[
                "Combine gate_proj, silu(gate) * up_proj into single kernel",
                "Reduces memory traffic by 2x",
            ],
            effort="medium",
            confidence=0.7,
        ))

        return candidates

    def _moe_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for Mixture-of-Experts routing."""
        candidates = []

        # Disable torch.compile for MoE
        candidates.append(Candidate(
            name="disable_inductor_moe",
            description="Disable torch.compile inductor for MoE layers, use CUDA graphs instead",
            target=target,
            strategy="config",
            expected_impact=1.8,
            implementation_plan=[
                "Set torch._inductor.config.mode = None",
                "Enable CUDA graph capture for the full model",
                "This avoids inductor's poor handling of dynamic expert routing",
            ],
            effort="low",
            confidence=0.85,
            risks=["Only works with static batch sizes in CUDA graph mode"],
        ))

        # Expert parallelism
        candidates.append(Candidate(
            name="expert_parallelism",
            description="Batch expert computations and overlap routing with expert forward",
            target=target,
            strategy="algorithm",
            expected_impact=1.3,
            implementation_plan=[
                "Group tokens by selected expert",
                "Batch all tokens for same expert into single GEMM",
                "Overlap routing computation with expert dispatch",
            ],
            effort="high",
            confidence=0.5,
        ))

        return candidates

    def _quantization_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for quantization-related operations."""
        candidates = []

        if self.gpu.has_fp4:
            candidates.append(Candidate(
                name="nvfp4_native",
                description="Use NVFP4 with torch._scaled_mm for native FP4 tensor core ops",
                target=target,
                strategy="quantization",
                expected_impact=3.0,
                implementation_plan=[
                    "Convert weights to NVFP4 format with per-block scales",
                    "Use torch._scaled_mm (Blackwell FP4 tensor cores)",
                    "Validate accuracy with representative calibration data",
                ],
                effort="medium",
                confidence=0.8 if self.gpu.compute_capability[0] >= 10 else 0.2,
                prerequisites=["Blackwell GPU (sm_100+)", "PyTorch 2.6+"],
            ))

        if self.gpu.has_fp8:
            candidates.append(Candidate(
                name="fp8_quantization",
                description="Quantize weights and/or activations to FP8 for 2x throughput",
                target=target,
                strategy="quantization",
                expected_impact=1.8,
                implementation_plan=[
                    "Quantize weights to FP8 E4M3 with per-channel scales",
                    "Use torch._scaled_mm for FP8 GEMM",
                    "Optionally quantize activations (dynamic per-tensor)",
                ],
                effort="medium",
                confidence=0.7,
            ))

        candidates.append(Candidate(
            name="w4a16_split",
            description="W4A16 quantization with split dequant + cuBLAS matmul",
            target=target,
            strategy="quantization",
            expected_impact=1.5,
            implementation_plan=[
                "Pack weights into INT4 with group-wise scales and zeros",
                "Triton dequant kernel: unpack + scale in registers",
                "F.linear for the actual GEMM (NT cuBLAS)",
            ],
            effort="medium",
            confidence=0.75,
        ))

        return candidates

    def _memory_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for memory-bound operations."""
        candidates = []

        candidates.append(Candidate(
            name="operator_fusion",
            description="Fuse this memory op with adjacent compute ops to reduce traffic",
            target=target,
            strategy="fusion",
            expected_impact=1.3,
            implementation_plan=[
                "Identify adjacent ops that can share memory loads",
                "Write fused Triton kernel combining both operations",
            ],
            effort="medium",
            confidence=0.6,
        ))

        return candidates

    def _embedding_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Candidates for embedding / positional encoding operations."""
        candidates = []

        candidates.append(Candidate(
            name="fused_rope",
            description="Fuse rotary position embedding with QKV projection",
            target=target,
            strategy="fusion",
            expected_impact=1.2,
            implementation_plan=[
                "Apply RoPE inside the QKV kernel after projection",
                "Avoids separate kernel launch and memory round-trip",
            ],
            effort="low",
            confidence=0.7,
        ))

        return candidates

    # ------------------------------------------------------------------
    # Universal strategies
    # ------------------------------------------------------------------

    def _universal_candidates(self, target: OptimizationTarget) -> list[Candidate]:
        """Strategies applicable to any operation category."""
        candidates = []

        # CUDA graphs (if not already using)
        if self.gpu.has_cuda_graphs:
            candidates.append(Candidate(
                name="cuda_graphs",
                description="Capture the operation in a CUDA graph to eliminate kernel launch overhead",
                target=target,
                strategy="config",
                expected_impact=1.1,
                implementation_plan=[
                    "Wrap forward pass in torch.cuda.CUDAGraph",
                    "Replay graph instead of launching individual kernels",
                    "Requires static tensor shapes",
                ],
                effort="low",
                confidence=0.5,
                risks=["Requires static shapes", "Incompatible with dynamic control flow"],
            ))

        # torch.compile
        candidates.append(Candidate(
            name="torch_compile",
            description="Use torch.compile with max-autotune backend",
            target=target,
            strategy="config",
            expected_impact=1.2,
            implementation_plan=[
                "Apply @torch.compile(mode='max-autotune') to the model or submodule",
                "Test with fullgraph=True for best performance",
                "Warm up with representative inputs",
            ],
            effort="low",
            confidence=0.4,
            risks=["Compilation time can be very long", "May not help if already using custom kernels"],
        ))

        return candidates

    def summary(self, candidates: list[Candidate]) -> str:
        """Format a human-readable summary of candidates."""
        lines = [
            "=" * 90,
            f"OPTIMIZATION CANDIDATES for: {candidates[0].target.op_name}" if candidates else "NO CANDIDATES",
            "=" * 90,
            f"{'#':<3} {'Name':<30} {'Strategy':<14} {'Impact':<8} {'Conf':<6} {'Effort':<8} {'Description'}",
            "-" * 90,
        ]

        for i, c in enumerate(candidates):
            lines.append(
                f"{i+1:<3} {c.name:<30} {c.strategy:<14} {c.expected_impact:>5.1f}x  "
                f"{c.confidence:>4.0%}  {c.effort:<8} {c.description[:50]}"
            )

        lines.append("=" * 90)
        return "\n".join(lines)
