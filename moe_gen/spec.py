"""MoE execution spec — describes an expert dispatch strategy declaratively."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoESpec:
    """Fully describes an MoE execution strategy.

    From this spec, the system selects:
    - How tokens are grouped and dispatched to experts
    - How many CUDA streams run in parallel
    - Whether to prefetch next expert weights
    - Whether to use native FP4 tensor cores
    """

    name: str

    # Architecture (from model config)
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    activation: str = "gelu"

    # Weight format
    weight_dtype: str = "fp16"  # "fp16", "bf16", "fp8", "nvfp4", "int4_awq"
    has_gate_up_fused: bool = True

    # Execution strategy
    strategy: str = "grouped_gemm"  # "grouped_gemm", "stream_parallel", "batched_expert"
    num_streams: int = 1            # CUDA streams for expert parallelism
    prefetch: bool = False          # prefetch next expert weights
    use_native_fp4: bool = False    # SM120 FP4 tensor cores

    @property
    def expert_flops(self) -> int:
        """FLOPs per token per expert (gate + up + down)."""
        H, I = self.hidden_size, self.intermediate_size
        # gate: [I, H] × [H, 1] = 2*I*H FLOPs
        # up:   [I, H] × [H, 1] = 2*I*H FLOPs
        # down: [H, I] × [I, 1] = 2*H*I FLOPs
        return 3 * 2 * H * I

    @property
    def total_flops_per_token(self) -> int:
        """Total MoE FLOPs per token (top_k experts)."""
        return self.top_k * self.expert_flops

    @property
    def expert_weight_bytes(self) -> int:
        """Bytes per expert (gate + up + down weights)."""
        H, I = self.hidden_size, self.intermediate_size
        bytes_per_element = {
            "fp16": 2, "bf16": 2, "fp8": 1, "nvfp4": 0.5, "int4_awq": 0.5,
        }.get(self.weight_dtype, 2)
        if self.has_gate_up_fused:
            return int((2 * I * H + H * I) * bytes_per_element)  # gate_up [2I, H] + down [H, I]
        return int(3 * I * H * bytes_per_element)

    @property
    def all_experts_bytes(self) -> int:
        """Total weight bytes for all experts."""
        return self.num_experts * self.expert_weight_bytes

    @property
    def avg_tokens_per_expert(self) -> float:
        """Average tokens routed to each expert at a given batch size."""
        # Assuming uniform routing (worst case for grouped_gemm)
        return lambda batch_size: batch_size * self.top_k / self.num_experts


# Predefined specs for Gemma4 26B-A4B
GEMMA4_MOE_SPECS = {
    "baseline": MoESpec(
        name="baseline",
        num_experts=128, top_k=8,
        hidden_size=2816, intermediate_size=704,
        weight_dtype="nvfp4", has_gate_up_fused=True,
        strategy="grouped_gemm", num_streams=1,
    ),
    "stream_2": MoESpec(
        name="stream_2",
        num_experts=128, top_k=8,
        hidden_size=2816, intermediate_size=704,
        weight_dtype="nvfp4", has_gate_up_fused=True,
        strategy="stream_parallel", num_streams=2,
    ),
    "stream_4": MoESpec(
        name="stream_4",
        num_experts=128, top_k=8,
        hidden_size=2816, intermediate_size=704,
        weight_dtype="nvfp4", has_gate_up_fused=True,
        strategy="stream_parallel", num_streams=4,
    ),
    "stream_4_prefetch": MoESpec(
        name="stream_4_prefetch",
        num_experts=128, top_k=8,
        hidden_size=2816, intermediate_size=704,
        weight_dtype="nvfp4", has_gate_up_fused=True,
        strategy="stream_parallel", num_streams=4, prefetch=True,
    ),
    "native_fp4": MoESpec(
        name="native_fp4",
        num_experts=128, top_k=8,
        hidden_size=2816, intermediate_size=704,
        weight_dtype="nvfp4", has_gate_up_fused=True,
        strategy="grouped_gemm", num_streams=1, use_native_fp4=True,
    ),
    "stream_4_fp4": MoESpec(
        name="stream_4_fp4",
        num_experts=128, top_k=8,
        hidden_size=2816, intermediate_size=704,
        weight_dtype="nvfp4", has_gate_up_fused=True,
        strategy="stream_parallel", num_streams=4, use_native_fp4=True,
    ),
}
