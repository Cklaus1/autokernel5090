# Gemma4 26B NVFP4 Decode — CUDA Kernel Profile

**Date:** 2026-04-09  
**GPU:** RTX 5090  
**Method:** `torch.profiler` via standalone Docker (`vllm-built`) — no server dependency  
**Model config:** Gemma4 27B, `hidden_size=2816`, `num_attention_heads=16`, `num_key_value_heads=8`, `head_dim=256`, `num_experts=16`, `top_k=2`, `intermediate_size=4608` per expert  

---

## 1. CUTLASS FP4 GEMM (`cutlass_scaled_fp4_mm`)

**CUDA kernel:** `_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniv...`  
**Used for:** All dense linear projections (Q/K/V/O), shared-expert gate+up+down

### All Gemma4 Linear Shapes — Average Latency (us) and TFLOPS

| Layer            | Shape (BxKxN)       | B=1 (us) | B=8 (us) | B=32 (us) | B=32 TFLOPS |
|------------------|---------------------|-----------|-----------|-----------|-------------|
| Q_proj           | Bx2816x4096         | ~818      | ~466      | **12.5**  | 39          |
| K_proj           | Bx2816x2048         | ~291      | ~114      | **16.9**  | 22          |
| V_proj           | Bx2816x2048         | ~646      | ~18       | **20.1**  | 18          |
| O_proj           | Bx4096x2816         | ~326      | ~22       | **18.9**  | 39          |
| gate_up_proj     | Bx2816x9216         | ~190      | ~32       | **28.6**  | 58          |
| down_proj        | Bx4608x2816         | ~660      | ~22       | **20.0**  | 42          |

> Note: B=1,8 latencies are noisy due to CUTLASS initialization overhead captured by profiler on first few iterations. B=32 is the reliable decode batch size measurement.

**Key observations:**
- CUTLASS FP4 GEMM at B=32 achieves 18–58 TFLOPS (RTX 5090 FP4 peak is ~3,352 TFLOPS via `torch._scaled_mm_v2`; these small-M shapes are heavily memory-bandwidth-bound)
- The `gate_up_proj` (largest N=9216) achieves best TFLOPS (58) due to better roofline utilization
- Square/larger-N shapes benefit most from FP4 GEMM

### MoE FP4 GEMM Timing (full `run_cutlass_moe_fp4` pipeline)

Config: `E=16, K=2, H=2816, N=4608, gate_up=9216`

| Batch | Latency (us) | TFLOPS (approx) |
|-------|-------------|-----------------|
| B=1   | 306         | 0.5             |
| B=8   | 690         | 1.8             |
| B=32  | 941         | 5.3             |
| B=128 | 648         | 30.7            |

**MoE full-pipeline kernel breakdown (B=32, 20 iterations):**

| Kernel | CUDA Time | % of Total |
|--------|-----------|------------|
| `cutlass::device_kernel<GemmUniversal...>` (×2 GEMMs) | 22.66 ms / 40 calls | **98.2%** |
| `at::native::reduce_kernel` (topk weight reduce) | 76.8 us / 20 calls | 0.33% |
| `shuffleInputRowsKernel<bfloat16>` | 67.6 us / 40 calls | 0.29% |
| `vllm::cvt_fp16_to_fp4<bfloat16, true>` (experts quant) | 45.5 us / 20 calls | 0.20% |
| `__get_group_gemm_starts` (setup) | 42.7 us / 40 calls | 0.19% |
| `compute_problem_sizes` | 25.3 us / 20 calls | 0.11% |
| `compute_expert_blockscale_offsets` | 23.7 us / 20 calls | 0.10% |
| `compute_arg_sorts` (permutation) | 23.1 us / 20 calls | 0.10% |

**Conclusion:** MoE is 98.2% GEMM-bound. The two `cutlass_fp4_moe_mm` calls (gate+up and down) average **566 us each** at B=32.

---

## 2. RMSNorm (`rms_norm` / `fused_add_rms_norm`)

**CUDA kernels:**
- `void vllm::rms_norm_kernel<c10::BFloat16, 8, 2>(...)`
- `std::enable_if<...>vllm::_typeConvert<c10::BFloat16>...` (fused_add_rms_norm)

### Latency vs Batch Size (H=2816)

| Batch | rms_norm (us) | fused_add_rms_norm (us) |
|-------|---------------|------------------------|
| B=1   | **1.50**      | **1.61**               |
| B=8   | **1.54**      | **1.66**               |
| B=32  | **1.59**      | **1.71**               |
| B=128 | **1.88**      | **2.37**               |

**Key observations:**
- Extremely fast and nearly batch-invariant at small batch (memory-bandwidth bound)
- `fused_add_rms_norm` is only ~7% slower than plain `rms_norm` — residual add is essentially free
- At B=32, contributes ~1.6 us per norm call. Gemma4 has ~62 layers × 2 norms = ~100 us total per token step — negligible vs GEMM

---

## 3. FP4 Quantization (`scaled_fp4_quant`)

**CUDA kernels:**
1. `void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false>(...)` — main conversion (57.6% of time)
2. `void at::native::vectorized_elementwise_kernel<4,...>` — scale computation (42.4%)

### Latency vs Batch/Hidden

| Config | Latency (us) | Effective BW |
|--------|-------------|--------------|
| B=32, H=2816 | **~1.6 us** (per kernel call; ~19 us total measured with overhead) | ~9 GB/s |
| B=32, H=4608 | **~1.4 us** | ~16 GB/s |
| B=128, H=2816 | **~0.95 us** | ~54 GB/s |
| B=128, H=4608 | **~1.0 us** | ~72 GB/s |

> Note: Single-call profiler shows ~0.95–1.6 us CUDA time. The 19–26 us wall-clock includes Python/CUDA dispatch overhead. In a CUDA graph the actual kernel time dominates.

**Two-kernel structure:**
- First kernel computes per-16-element block max and converts bf16→fp4
- Second kernel is a PyTorch elementwise for the global scale computation

---

## 4. FlashInfer Paged Decode Attention

**Config:** `H=16 heads, KV_H=8, head_dim=256, page_size=16`  
**Note:** `head_dim=176` (exact Gemma4: 2816/16) is unsupported by FlashInfer's merge-states kernel; tested at `head_dim=256`.

**CUDA kernels:**
1. `void flashinfer::BatchDecodeWithPagedKVCacheKernel<...>` — main decode (~83-99% of time)
2. `void flashinfer::PersistentVariableLengthMergeStates<...>` — split-K reduction (~1-17%)

### Latency vs Batch/Context

| Batch | Context | Decode Kernel (us) | MergeStates (us) | Total (us) |
|-------|---------|-------------------|-------------------|------------|
| B=1   | 2048    | 8.35              | 1.68              | **10.0**   |
| B=8   | 2048    | 83.1              | 1.98              | **85.1**   |
| B=32  | 2048    | 321.2             | 2.20              | **323.4**  |
| B=1   | 8192    | 14.0              | 2.81              | **16.8**   |

**Key observations:**
- Attention decode scales linearly with batch (memory bandwidth bound — reading the entire KV cache)
- At B=32, seq=2048: **323 us per layer**, which at 62 layers = ~20 ms attention-only → dominant cost for long contexts
- MergeStates (split-K combine) is only ~0.7-2% overhead
- Context 4× longer (8192 vs 2048) with B=1: 1.68× slower (sub-linear — better GPU utilization at longer seq)

---

## 5. Memory Bandwidth Baseline

Measured copy throughput: **0.635 TB/s** (theoretical peak RTX 5090: 1.79 TB/s)  
→ Kernels are achieving ~35% of theoretical peak bandwidth in isolation.

---

## Roofline Summary (B=32, Gemma4 27B)

| Kernel | Latency/call (us) | % of decode step (est.) | Bottleneck |
|--------|------------------|------------------------|------------|
| MoE CUTLASS FP4 GEMM ×2 | ~566 each | **~60%** | Compute (small-M GEMM) |
| FlashInfer Decode Attn | ~323 | **~20%** | Memory BW (KV cache read) |
| FP4 Quant (per-layer) | ~1.6 (kernel) | ~5% | Memory BW |
| RMSNorm (per-layer) | ~1.6 | ~3% | Memory BW |
| CUTLASS FP4 GEMM (QKV/O) | ~12–29 each | ~10% | Compute |
| MoE overhead (shuffle, setup) | ~0.3 | <1% | Negligible |

**Primary optimization targets:**
1. **MoE GEMMs** — 98% of MoE time; small-M GEMM is bottleneck. At B=32 with 16 experts each getting ~4 tokens, CUTLASS is running 4×2816 vs 4608 batches (very small M). Batching more requests or using speculative decoding to increase effective batch size helps.
2. **Attention decode** — scales with context length; consider FP8 KV cache to halve bandwidth.
3. **FP4 quant** — fast at B≥32; noisy/slow at B<8 (CUDA launch overhead dominates).
