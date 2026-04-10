# Data-Driven MoE Kernel Plan

## The Problem

Gemma4 26B-A4B MoE decode at B=128:
- 128 experts, top-8 routing per token
- Per layer: 128 × 8 = 1,024 token-expert pairs (B=128 × top_k=8)
- Per token: gate_proj [704, 2816] + up_proj [704, 2816] + down_proj [2816, 704]
- Total: 1,024 × 3 matmuls = 3,072 small GEMMs per layer × 30 layers = 92,160 GEMMs per decode step
- This takes ~35ms (85% of decode time)

vLLM's current approach: FusedMoE Triton kernel groups tokens by expert and runs batched GEMMs.
The issue: suboptimal for small per-expert batch sizes (B=128 / 128 experts ≈ 1 token per expert on average).

## The Spec

```python
@dataclass
class MoESpec:
    """Describes an MoE execution strategy."""

    name: str

    # Architecture
    num_experts: int           # 128
    top_k: int                 # 8
    hidden_size: int           # 2816
    intermediate_size: int     # 704
    activation: str            # "gelu"

    # Weight format
    weight_dtype: str          # "nvfp4", "fp8", "fp16", "int4_awq"
    has_gate_up_fused: bool    # True = gate+up in one [2*I, H] tensor

    # Execution strategy (the tunable part)
    strategy: str              # "grouped_gemm", "batched_expert", "megablocks", "stream_parallel"
    group_size: int            # tokens per expert group before GEMM dispatch
    num_streams: int           # CUDA streams for expert parallelism (1=serial, 2-4=parallel)
    prefetch: bool             # prefetch next expert weights during current expert compute
    use_native_fp4: bool       # use SM120 FP4 tensor cores (Blackwell only)
```

## Four Execution Strategies

### 1. grouped_gemm (vLLM's current approach)
- Sort tokens by routed expert
- Run one batched GEMM per expert with all its tokens
- Good at high batch (many tokens per expert), bad at low batch

### 2. batched_expert (new)
- Don't sort — run all experts for all tokens as one large GEMM
- Pad to uniform size, mask inactive expert-token pairs
- Wastes compute but eliminates the sort + scatter overhead
- Good when per-expert batch is tiny (B < num_experts)

### 3. stream_parallel (new, the big win)
- Partition active experts across 2-4 CUDA streams
- Each stream runs its experts serially, but streams run in parallel
- Expert 0-3 on stream 0, expert 4-7 on stream 1, etc.
- Sync after all streams complete
- The GPU has enough SMs for 2-4 concurrent small GEMMs

### 4. megablocks (existing research)
- Reshape all expert GEMMs into block-sparse operations
- Uses Triton dsd (dense × sparse → dense) kernels
- Best theoretical throughput but complex implementation

## Build Phases (ASI timeline)

### Phase 1: MoE Spec + Benchmark Harness (~30 min)
- `moe_spec.py` — MoESpec dataclass with predefined configs
- `moe_bench.py` — Benchmark current approaches vs strategies
- Measure: tokens/sec, FLOPS utilization, memory bandwidth

### Phase 2: Stream-Parallel MoE (~1 hour)
- Wrap expert matmuls in multi-stream execution
- No new kernels — just CUDA stream management + token grouping
- Expected: 1.5-2x on MoE portion

### Phase 3: NVFP4 Native Tensor Core (~1 hour)
- Replace dequant+FP16 matmul with `torch._scaled_mm` using FP4 inputs
- SM120 specific — won't work on older GPUs
- Expected: 1.5-2x on every expert matmul

### Phase 4: Grouped GEMM with Prefetch (~2 hours)
- Triton kernel that processes experts in groups
- While computing expert[i], prefetch weights for expert[i+1]
- Expected: 1.1-1.3x from hidden memory latency

### Full sweep + integration: ~1 session (half day)

## Expected Combined Impact

| Optimization | Speedup on MoE (85%) | Speedup on total |
|-------------|---------------------|-----------------|
| Stream parallel (4 streams) | 1.5-2.0x | 1.4-1.7x |
| NVFP4 native tensor core | 1.5-2.0x | 1.4-1.7x |
| Weight prefetch | 1.1-1.3x | 1.1-1.25x |
| Combined | 2.5-5.0x | 2.1-4.3x |

At B=128: 3,032 tok/s → **6,300-13,000 tok/s**

## Relationship to KV Cache Kernel

Same design pattern:
- KV cache: KVCacheSpec → kernel generator → autotune sweep
- MoE: MoESpec → strategy selector → autotune sweep

The sweep infrastructure (bench harness, config parser, adaptive selector) is reusable.
The plugin architecture extends: `fusen_kv` becomes `fusen` with both attention and MoE backends.
