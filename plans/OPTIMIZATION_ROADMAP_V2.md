# Optimization Roadmap V2: Definitive Prioritized Build Plan

**Target:** Gemma4 26B-A4B MoE (128 experts, top-8), NVFP4, RTX 5090 (SM120, 32GB GDDR7, 1792 GB/s BW, ~48MB L2)
**Current state:** 6,615 tok/s batch throughput, 186 tok/s single-request decode (MTP3)
**Date:** April 9, 2026

---

## Part 1: Master Ranking of All 120+ Techniques

### Scoring Methodology

Each technique is scored on three axes:
- **Impact (I):** Expected throughput/latency/capacity gain, 1-10 scale
- **Feasibility (F):** Inverse of effort/risk, 1-10 scale (10 = hours, 1 = months/impossible)
- **Confidence (C):** How certain is the impact? 1-10 (10 = measured, 1 = theoretical)

**Priority Score = I x F x C / 100** (max 10.0)

### 1.1 Master Ranking Table

| Rank | ID | Technique | Tier | Impact | Effort | Dependencies | Score | Compounds With | Automatable? |
|------|-----|-----------|------|--------|--------|--------------|-------|----------------|-------------|
| 1 | F1/F3/F9 | Fused MoE dispatch kernel (persistent, all-expert) | Critical | 2-5x on MoE (85% of decode) | Days-Weeks | None | 8.1 | Everything | Partially (autotune config) |
| 2 | D2 | Expert pruning (remove underused experts) | Critical | 10-30% model size + proportional speedup | Days | Expert usage profiling (J2-3) | 7.2 | Quantization, TP, fused MoE | Yes (profile + threshold) |
| 3 | E3 | Tensor parallelism TP=2 | Critical | ~1.8x throughput, 2x KV capacity | Hours-Days | Second GPU + NVLink | 7.0 | FusenCache, expert pruning, MTP | No (hardware decision) |
| 4 | C3-int | FusenCache vLLM v1 production integration | Critical | 4x KV capacity, higher batch sizes | Days | vLLM per-layer allocator PR | 6.5 | TP, batch scheduling, prefix cache | Partially (integration tests) |
| 5 | J4-1 | C++ minimal inference server | High | Up to 1.65x (eliminate 5.38ms Python overhead) | Weeks | None | 5.6 | All kernel-level opts | No (architecture decision) |
| 6 | E7 | Disaggregated prefill/decode | High | 20-40% for mixed workloads | Days | Multi-GPU or separate processes | 5.4 | TP, FusenCache, continuous batch | Partially (workload-dependent) |
| 7 | J2-4 | Nsight Compute profiling of hot kernels | High | Reveals true bottleneck, guides everything | Hours | None | 5.3 | All (meta-optimization) | Yes (scripted profiling) |
| 8 | D3 | Expert merging (128 -> 64 experts) | High | ~2x expert compute reduction | Weeks | Expert similarity analysis | 4.8 | Expert pruning, quantization | Partially (similarity metric) |
| 9 | D4 | Layer pruning (remove 3-5 layers) | High | 10-17% compute reduction | Days | Quality validation suite | 4.5 | All | Partially (prune + eval loop) |
| 10 | C5 | QAT checkpoint (if Google releases) | High | Better quality at same speed | Hours (swap) | Google releasing checkpoint | 4.2 | Everything (quality floor rises) | No (external dependency) |
| 11 | A5 | EAGLE speculative decoding | High | 2-3x single-user decode | Weeks | vLLM Eagle3 fix | 4.0 | MTP, CUDA graphs (with caveats) | No (architecture choice) |
| 12 | C12-str | 2:4 structured sparsity on expert weights | High | Up to 2x GEMM if TC support | Weeks | Verify SM120 FP4+sparsity compat | 3.8 | NVFP4, fused MoE | Partially (pruning sweep) |
| 13 | G8/G9 | CUTLASS 3.x SM120 audit + CUDA 12.8 features | Medium | 5-15% kernel improvement | Days | None | 3.6 | All kernel paths | Yes (benchmark sweep) |
| 14 | B12 | Cross-layer KV sharing | Medium | 30-50% KV memory reduction | Weeks | Quality validation per layer group | 3.2 | FusenCache, TP | Partially (attention pattern analysis) |
| 15 | E8 | KV cache offloading (CPU/SSD) | Medium | Extends capacity for preempted requests | Hours | None (vLLM flag) | 3.0 | FusenCache (offload compressed) | Yes (config flag) |
| 16 | E9 | Request scheduling (SJF, priority) | Medium | 20-30% latency improvement for mixed workloads | Hours | None (vLLM config) | 2.8 | Continuous batching | Yes (A/B test policies) |
| 17 | B2 | FlashAttention-3 for SM120 | Medium | 10-30% attention speedup (~3% system) | Days | FA3 SM120 release | 2.7 | FusenCache, KV quant | No (external dependency) |
| 18 | F1-moe | MoE shuffle + quantization fusion | Medium | 2.3% system gain | Hours | None | 2.6 | Fused MoE dispatch | Yes (profile + fuse) |
| 19 | C2 | FP8 activation quantization (inter-expert) | Medium | Reduces inter-expert memory traffic | Days | Profiling activation patterns | 2.5 | NVFP4, fused MoE | Yes (sweep quant strategies) |
| 20 | E6 | Data parallelism (multiple vLLM instances) | Medium | Linear scaling with GPUs | Hours | Additional GPUs | 2.4 | Independent of all | Yes (trivial) |
| 21 | I3 | Mixture of Agents (Fusen Engine) | Medium | Better quality-throughput tradeoff | Weeks | Multiple model checkpoints | 2.3 | Per-model optimization | Partially (router design) |
| 22 | C4 | Mixed precision per-module (FP8 gate, INT8 norm) | Medium | <5% (non-GEMM ops are tiny) | Days | None | 2.0 | Everything | Yes (sweep precision per op) |
| 23 | G5 | L2 cache partitioning | Medium | Unknown (API uncertain on SM120) | Days | Verify cudaAccessPolicyWindow | 1.8 | Expert weight caching | Yes (sweep partition ratios) |
| 24 | E14-prefetch | Expert weight prefetch (L2 strategy) | Medium | 1.1-1.3x on MoE | Hours | L2 geometry knowledge | 1.7 | Fused MoE | Yes (prefetch distance sweep) |
| 25 | J2-1 | Task-specific accuracy benchmarks | Medium | Validates quality claims | Hours | Benchmark datasets (HumanEval, GSM8K) | 1.6 | All quality-affecting opts | Yes (automated eval) |
| 26 | J2-2 | P99 latency under load testing | Medium | Reveals production bottlenecks | Hours | Load testing tool | 1.5 | Scheduling, disaggregated | Yes (load gen + measure) |
| 27 | A6 | Lookahead decoding (n-gram) | Low | 1.1-1.3x for repetitive workloads | Days | None | 1.4 | Continuous batching | Yes (n-gram table auto-built) |
| 28 | A13 | REST retrieval-based spec decode | Low | Domain-specific only | Days-Weeks | Retrieval index | 1.3 | Continuous batching | Partially (index auto-built) |
| 29 | B9 | KV cache eviction (H2O, SnapKV) | Low | Minimal at short sequence lengths | Days | None | 1.2 | FusenCache | Partially (importance scoring) |
| 30 | F3 | Persistent kernels (per-shape analysis) | Low | Shape-dependent, mixed results | Days | None | 1.1 | Autotuning | Yes (sweep) |
| 31 | B8 | Sparse attention within sliding window | Low | Minimal (window=1024 too short) | Days | None | 1.0 | FusenCache | Partially |
| 32 | C14 | Low-rank approximation for expert weights | Low | Uncertain (SVD + 2 matmuls vs 1) | Weeks | None | 0.9 | Expert merging | Yes (SVD sweep) |
| 33 | A12 | Staged speculative decoding | Low | Diminishing returns from more stages | Weeks | Working spec decode (A5) | 0.8 | EAGLE | No (design decision) |
| 34 | A14 | Cascade speculative decoding | Low | Tree verification expensive for MoE | Weeks | Working spec decode (A5) | 0.7 | EAGLE, n-gram, REST | No |
| 35 | H8 | Tensor decomposition (shared expert factors) | Low | Speculative | Weeks | None | 0.7 | Expert merging | Partially |
| 36 | B13 | Attention sink / StreamingLLM | Low | Only for infinite generation | Hours | None | 0.6 | Sliding window | Yes (config) |
| 37 | E10 | Preemption policy tuning | Low | Marginal | Hours | None | 0.5 | KV offloading | Yes (A/B test) |
| 38 | G7 | Dynamic parallelism (kernels launching kernels) | Low | High overhead on consumer GPUs | Days | None | 0.5 | Fused MoE | No |
| 39 | C7 | AQLM additive quantization | Low | Loses tensor core accel | Days | None | 0.4 | Nothing (standalone) | Yes (benchmark) |
| 40 | C8 | QuIP# incoherence processing | Low | Rotation overhead at 4-bit | Days-Weeks | None | 0.4 | Nothing (standalone) | Yes (benchmark) |
| 41 | C9 | SqueezeLLM non-uniform quant | Low | Slower than NVFP4 on Blackwell | Days | None | 0.4 | Nothing (standalone) | Yes (benchmark) |
| 42 | C10 | Microscaling MXFP4/MXFP8 | Low | NVFP4 is equivalent concept | Days | None | 0.4 | Nothing | Yes (test MX intrinsics) |
| 43 | H3 | ANN for attention (LSH, PQ) | Low | Decode not attention-bottlenecked | Weeks | None | 0.3 | KV eviction | No |
| 44 | H5 | Polynomial approx for softmax/GELU | Low | <0.1% system impact | Hours | None | 0.3 | Nothing meaningful | Yes (trivial) |
| 45 | H13 | Winograd for Mamba conv | Low | Negligible impact | Days | None | 0.2 | Nothing | Yes |
| 46 | B10 | Dynamic sparse attention (learned masks) | Research | Predictor overhead > savings at short seq | Weeks | Training predictor | 0.2 | KV eviction | No (requires training) |
| 47 | D5 | Early exit | Research | CUDA graph incompatible | Weeks | Exit classifier training | 0.2 | Breaks CUDA graphs | No |
| 48 | D6 | Dynamic depth | Research | Same CUDA graph problem | Weeks | Router training | 0.2 | Breaks CUDA graphs | No |
| 49 | I2 | Dynamic compute per token | Research | CUDA graph incompatible | Weeks | Classifier | 0.2 | Breaks CUDA graphs | No |
| 50 | I1 | Test-time compute (thinking tokens) | Research | Orthogonal to throughput | Hours | Model capability | 0.2 | All throughput opts | N/A |

**Not Applicable / Impossible (requires model retraining or different hardware):**

| ID | Technique | Reason |
|----|-----------|--------|
| A4 | Medusa heads | Dominated by native MTP; requires fine-tuning |
| A7 | Jacobi decoding | Negative expected value (convergence too slow) |
| A8 | Diffusion text generation | Different model architecture entirely |
| A9 | Consistency models | Different model architecture |
| A10 | Non-autoregressive generation | Incompatible with Gemma4 |
| A11 | Blockwise parallel decoding | Already covered by MTP3 |
| B4 | Ring attention | Single GPU, short sequences |
| B6 | MLA (Multi-Head Latent Attention) | Requires model retraining |
| B7 | Linear attention (Mamba/RWKV replace) | Cannot retrofit into Gemma4 attention layers |
| B14 | Hash-based attention (Reformer) | Decode is already linear in sequence length |
| B15 | Differential attention | Requires retraining |
| C11 | 2-bit/BitNet | Requires training from scratch |
| C13 | Knowledge distillation | Orthogonal; weeks-months; uncertain quality |
| D7 | ALBERT weight sharing | Requires retraining |
| D8 | Activation checkpointing | Training only, not inference |
| D9 | Mixture of Depths | Requires retraining |
| D10 | Matryoshka representations | Requires retraining |
| E4 | Pipeline parallelism | Model fits on one GPU |
| E5 | Expert parallelism | vLLM lacks Gemma4 EP support |
| H4 | Matrix sketching | Uncontrolled quality loss |
| H6 | FFT attention replacement | Requires retraining |
| H7 | Structured matrices | Requires retraining |
| H9 | Randomized GEMM | Uncontrolled error accumulation |
| H10 | Cache-oblivious algorithms | Dominated by autotuning |
| H11 | Toeplitz/circulant structure | Not present in attention |
| H12 | Newton-Schulz iteration | Slower than rsqrt() for RMSNorm |
| I4 | torch.compile whole-model | Tested, 3x slower than CUDA graphs |
| I5 | FPGA/ASIC accelerators | Different hardware platform |
| I6 | Photonic computing | Lab-stage, 5-10 years out |
| I7 | Analog computing | Lab-stage, 3-7 years out |
| I8 | Neuromorphic | Incompatible with transformers |
| I9 | In-memory computing (PIM) | Hardware not in RTX 5090 |

### Already Deployed (baseline):

| ID | Technique | Status |
|----|-----------|--------|
| A1 | Autoregressive decoding | Baseline |
| A3 | MTP3 | Deployed, +54% single-user |
| B1 | FlashAttention 2 | Deployed |
| B3 | PagedAttention | Deployed via vLLM |
| B5 | GQA | Inherent to Gemma4 |
| B11 | Sliding window + global hybrid | Inherent, per-layer spec exploited |
| C1 | NVFP4 weight quantization | Deployed, 1261 TFLOPS |
| C3 | KV cache quantization (FusenCache) | Proven, integration pending |
| C6 | PTQ with calibration (modelopt) | Deployed |
| E1 | Continuous batching | Deployed via vLLM |
| E2 | Prefix caching | Deployed (workload-dependent) |
| E11 | Chunked prefill | Deployed |
| E12 | Memory pool management | Inherent |
| F2 | cuBLAS/CUTLASS/Triton selection | Optimized per-op |
| F5 | FP4 tensor core utilization | Deployed |
| F6 | Memory access optimization | Deployed |
| F7 | Register pressure tuning | Optimized |
| F8 | Shared memory optimization | Optimized |
| F12 | Software pipelining | Optimized |
| F13 | Kernel autotuning | Deployed |
| G1 | FP4 tensor cores | Deployed |
| G2 | TMA | Implicit via CUTLASS |
| H1 | Flash-decoding (split-K) | Deployed |
| H2 | Online softmax | Deployed |

---

## Part 1.2: Grouped Rankings

### Top 10 Highest-Impact (ordered by Impact x Feasibility)

| # | Technique | Expected Impact | Effort | Why Top 10 |
|---|-----------|----------------|--------|------------|
| 1 | **Fused MoE dispatch kernel** | 2-5x on 85% of decode time | Days-Weeks | Eliminates launch overhead for 128-expert dispatch. Single biggest compute bottleneck. |
| 2 | **Expert pruning** | 10-30% speedup + size reduction | Days | MoE models empirically have underutilized experts. Profiling is cheap, pruning is immediate. |
| 3 | **TP=2 (two GPUs)** | ~1.8x throughput, 2x capacity | Hours-Days | Well-understood, vLLM native. Doubles the hardware budget. |
| 4 | **FusenCache production integration** | 4x KV capacity, enables B=800+ | Days | Proven tech, just needs vLLM v1 API plumbing. Unlocks batch sizes limited by KV memory. |
| 5 | **C++ minimal server** | Up to 1.65x (eliminates 5.38ms overhead) | Weeks | 65% of per-token time is vLLM/Python overhead. C++ goes straight to CUDA graph replay. |
| 6 | **Disaggregated prefill/decode** | 20-40% mixed workload improvement | Days | Eliminates prefill-decode interference. Proven by major serving deployments. |
| 7 | **Nsight Compute hot-kernel profiling** | Guides all future optimization | Hours | We have never measured HBM bandwidth utilization inside the hot kernels. This changes everything. |
| 8 | **Expert merging (128->64)** | ~2x expert compute | Weeks | Halves expert dispatch count. Quality impact is the key risk. |
| 9 | **Layer pruning** | 10-17% compute reduction | Days | Each removed layer is proportional speedup. Middle layers are most redundant in practice. |
| 10 | **QAT checkpoint** | Quality uplift at zero speed cost | Hours (to swap) | If Google releases QAT Gemma4, we get better quality for free. |

### Quick Wins (< 1 day, > 5% impact)

| Technique | Impact | Effort | Notes |
|-----------|--------|--------|-------|
| Nsight Compute profiling | Reveals true bottleneck | 2-4 hours | Run `ncu` on top 5 kernels by time |
| KV cache offloading flag | Extends capacity for preempted reqs | 1 hour | `--swap-space` in vLLM |
| Request scheduling policy | 20-30% latency for mixed loads | 1-2 hours | vLLM scheduling config |
| Task-specific accuracy benchmarks | Validates quality | 4-8 hours | Run HumanEval, GSM8K, MMLU |
| P99 latency measurement | Reveals production readiness | 4-8 hours | Load test with variable arrival rates |
| Expert usage profiling | Informs pruning/merging | 2-4 hours | Run 1000 prompts, count expert activations |
| MoE shuffle+quant fusion | 2.3% system gain | 4-8 hours | ~5-line CUDA change, already analyzed |

### Strategic Investments (weeks, > 50% cumulative impact)

| Technique | Impact | Effort | ROI Rationale |
|-----------|--------|--------|---------------|
| Fused MoE dispatch kernel | 2-5x on MoE dispatch | 2-4 weeks | The single kernel that matters most. 85% of decode is MoE. |
| C++ inference server | 1.65x system throughput | 2-3 weeks | Eliminates the entire Python serving stack overhead. |
| Expert merging + pruning pipeline | 1.3-2x | 2-3 weeks | Reduce the model to its essential experts. |
| TP=2 optimized deployment | 1.8x | 1-2 weeks | Hardware scaling, the "just buy more GPU" answer done right. |
| FusenCache vLLM production | 4x KV capacity | 1-2 weeks | Unlocks batch sizes unreachable with BF16 KV. |
| **Combined ceiling:** | **8-20x over current** | **2-3 months** | **All five compound (multiplicative for independent bottlenecks).** |

### Research Bets (uncertain payoff, potentially transformative)

| Technique | Potential | Risk | Why It Could Be Huge |
|-----------|-----------|------|---------------------|
| EAGLE spec decode for MoE | 3-4x single-user with MTP | vLLM integration broken | EAGLE + MTP compound -- draft from MTP hidden states |
| 2:4 structured sparsity + FP4 | 2x GEMM on top of NVFP4 | SM120 may not support both | Would be 2-bit effective with hardware acceleration |
| Cross-layer KV sharing | 30-50% KV reduction | Quality impact unknown | 25 sliding layers with similar patterns could share |
| Whole-model CUDA mega-graph | Eliminates ALL overhead | MoE routing is dynamic | If routing can be captured as conditional, this is endgame |
| In-memory computing (future HW) | 10-100x decode | Hardware doesn't exist yet | The "correct" solution to memory-bound GEMV |

### Not Worth Pursuing (with reasons)

| Technique | Reason |
|-----------|--------|
| Jacobi decoding (A7) | Convergence requires 10-50 full forward passes. Net negative. |
| Polynomial softmax approx (H5) | < 0.1% system impact. GPU SFU handles exp() in 4 cycles. |
| Hash-based attention (B14) | Decode attention is already O(seq_len), not O(seq_len^2). |
| Winograd for Mamba conv (H13) | Conv is negligible fraction of compute. |
| AQLM/QuIP#/SqueezeLLM (C7-C9) | All lose FP4 tensor core acceleration. Slower than NVFP4. |
| Alternative weight quant schemes | NVFP4 already uses hardware-native FP4 tensor cores. Any other scheme must emulate. |
| torch.compile full-model (I4) | Measured at 3x slower than CUDA graphs. |
| Dynamic depth/early exit (D5-D6) | Fundamentally incompatible with CUDA graphs, which give 7x speedup. |
| Cache-oblivious algorithms (H10) | Autotuning for specific cache sizes always wins. |
| Sparse attention within 1024-token window (B8) | Window too short for sparsity to matter. |

---

## Part 2: Hierarchical Decomposition of Top 20 Techniques

### 1. Fused MoE Dispatch Kernel [F1/F3/F9]

```
1. Fused MoE Dispatch Kernel
   |-- 1.1 Architecture design
   |   |-- 1.1.1 Single persistent kernel vs cooperative kernel grid
   |   |-- 1.1.2 Token-to-expert routing within kernel (avoid CPU roundtrip)
   |   |-- 1.1.3 Memory layout: contiguous expert weights vs interleaved
   |   '-- 1.1.4 Decide Triton vs raw CUDA vs CUTLASS grouped GEMM
   |-- 1.2 Expert dispatch optimization
   |   |-- 1.2.1 Permute tokens to experts (scatter/gather pattern)
   |   |-- 1.2.2 Pad expert batches to GEMM-friendly sizes
   |   |-- 1.2.3 Fuse gate computation + top-K selection + permute
   |   '-- 1.2.4 Handle variable expert batch sizes (load imbalance)
   |-- 1.3 GEMM execution
   |   |-- 1.3.1 FP4 grouped GEMM (all experts in one launch)
   |   |-- 1.3.2 Expert weight prefetch (double-buffer between experts)
   |   |-- 1.3.3 Activation quantization fusion (BF16 -> FP4 inline)
   |   '-- 1.3.4 SiLU+mul fusion within expert compute
   |-- 1.4 Launch overhead elimination
   |   |-- 1.4.1 Persistent kernel loop (one launch for all 30 layers)
   |   |-- 1.4.2 CUDA graph capture of fused kernel
   |   |-- 1.4.3 Cooperative groups for cross-SM synchronization
   |   '-- 1.4.4 Measure: launch overhead per expert (current) vs fused (target)
   |-- 1.5 Integration
   |   |-- 1.5.1 vLLM MoE backend replacement
   |   |-- 1.5.2 Correctness validation (bitwise match vs reference)
   |   |-- 1.5.3 CUDA graph compatibility verification
   |   '-- 1.5.4 Batch size sweep (B=1 to B=256)
   '-- 1.6 Autotuning
       |-- 1.6.1 Tile sizes per expert shape (H=2816, 2*H=5632)
       |-- 1.6.2 Number of pipeline stages
       |-- 1.6.3 Warp count per thread block
       '-- 1.6.4 Expert grouping strategy (process how many experts per grid?)
```

**Automation potential:** Autotuning (1.6) is fully automatable. Architecture (1.1) requires human insight. Integration (1.5) is semi-automatable with test harness.

### 2. Expert Pruning [D2]

```
2. Expert Pruning
   |-- 2.1 Expert usage profiling
   |   |-- 2.1.1 Instrument router to log expert selection per token
   |   |-- 2.1.2 Run diverse prompt set (1000+ prompts, multiple domains)
   |   |-- 2.1.3 Compute per-expert activation frequency distribution
   |   |-- 2.1.4 Compute per-expert activation by domain (code, math, chat, etc.)
   |   '-- 2.1.5 Identify "cold" experts (< 0.1% of tokens)
   |-- 2.2 Pruning strategy
   |   |-- 2.2.1 Simple removal: delete expert, adjust routing logits
   |   |-- 2.2.2 Redistribution: route cold expert's tokens to nearest surviving expert
   |   |-- 2.2.3 Gradual pruning: reduce expert capacity over calibration run
   |   '-- 2.2.4 Determine pruning threshold (1%? 0.5%? 0.1%?)
   |-- 2.3 Quality validation
   |   |-- 2.3.1 PPL on WikiText-2 at each pruning level
   |   |-- 2.3.2 Task-specific accuracy (HumanEval, GSM8K, MMLU)
   |   |-- 2.3.3 Coherence spot-check on representative prompts
   |   '-- 2.3.4 Quality vs prune-percentage Pareto curve
   |-- 2.4 Performance measurement
   |   |-- 2.4.1 Model size reduction
   |   |-- 2.4.2 Decode throughput at various batch sizes
   |   |-- 2.4.3 Prefill throughput
   |   '-- 2.4.4 Memory savings -> additional batch capacity
   '-- 2.5 Deployment
       |-- 2.5.1 Save pruned checkpoint
       |-- 2.5.2 Update vLLM model config (num_experts, routing)
       '-- 2.5.3 Re-run full benchmark suite
```

**Automation potential:** Almost entirely automatable. Script profiles experts (2.1), sweeps pruning thresholds (2.2.4), measures quality (2.3), and produces Pareto curve. Human chooses operating point on curve.

### 3. Tensor Parallelism TP=2 [E3]

```
3. Tensor Parallelism TP=2
   |-- 3.1 Hardware setup
   |   |-- 3.1.1 Verify NVLink between two GPUs
   |   |-- 3.1.2 Measure NVLink bandwidth (p2pBandwidthLatencyTest)
   |   '-- 3.1.3 Determine NVLink type (consumer ~100 GB/s vs PRO ~450 GB/s)
   |-- 3.2 vLLM TP configuration
   |   |-- 3.2.1 Launch with --tensor-parallel-size 2
   |   |-- 3.2.2 Verify weight sharding (each GPU gets half of each layer)
   |   |-- 3.2.3 Verify KV cache sharding (each GPU handles half the heads)
   |   '-- 3.2.4 Verify CUDA graph capture with TP=2
   |-- 3.3 MoE-specific TP concerns
   |   |-- 3.3.1 Expert weight distribution (all experts on both GPUs vs split?)
   |   |-- 3.3.2 Router all-reduce overhead per step
   |   |-- 3.3.3 Expert dispatch with TP (tokens routed across GPUs?)
   |   '-- 3.3.4 vLLM MoE TP implementation correctness
   |-- 3.4 FusenCache compatibility
   |   |-- 3.4.1 Per-GPU FusenCache store/decode kernels
   |   |-- 3.4.2 KV cache block table sharding
   |   '-- 3.4.3 Attention kernel with sharded FusenCache KV
   |-- 3.5 Benchmarking
   |   |-- 3.5.1 Single-user decode throughput (expect ~1.5-1.8x)
   |   |-- 3.5.2 Batch throughput sweep (B=1 to B=512)
   |   |-- 3.5.3 TP communication overhead profiling
   |   '-- 3.5.4 Compare TP=2 vs DP=2 (two independent instances)
   '-- 3.6 Scaling analysis
       |-- 3.6.1 Strong scaling efficiency (TP=2 vs TP=1)
       |-- 3.6.2 Identify communication bottlenecks
       '-- 3.6.3 Model whether TP=4 on 4 GPUs is worthwhile
```

**Automation potential:** 3.1 and 3.5 fully automatable (benchmark scripts). 3.3 requires understanding MoE TP semantics. 3.4 requires engineering.

### 4. FusenCache vLLM v1 Integration [C3-int]

```
4. FusenCache Production Integration
   |-- 4.1 vLLM per-layer cache allocator
   |   |-- 4.1.1 Upstream PR: per-layer KV dtype/shape in cache allocator
   |   |-- 4.1.2 Handle Gemma4 hybrid (sliding window vs global layers)
   |   |-- 4.1.3 Per-layer spec selection: k8v4 sliding, k4v4 global
   |   '-- 4.1.4 Memory accounting (different sizes per layer)
   |-- 4.2 Kernel registration
   |   |-- 4.2.1 register_kv_cache_dtype() upstream PR
   |   |-- 4.2.2 Replace monkey-patching in plugin.py
   |   |-- 4.2.3 Version compatibility testing (vLLM 0.17.x)
   |   '-- 4.2.4 Precompiled .cubin for common GPU/spec combos
   |-- 4.3 CUDA graph integration
   |   |-- 4.3.1 Verify FusenCache kernels are CUDA-graph-safe
   |   |-- 4.3.2 Graph capture with mixed KV specs per layer
   |   |-- 4.3.3 Warm-up sequence for graph capture
   |   '-- 4.3.4 Graph replay correctness validation
   |-- 4.4 Quality assurance
   |   |-- 4.4.1 PPL comparison: BF16 KV vs K8V4 vs K4V4
   |   |-- 4.4.2 Task-specific benchmarks with FusenCache
   |   |-- 4.4.3 Long-context quality (8K, 32K, 128K)
   |   '-- 4.4.4 Online canary quality monitoring hook
   '-- 4.5 Production hardening
       |-- 4.5.1 Error handling for OOM with compressed cache
       |-- 4.5.2 Graceful fallback to BF16 on kernel failure
       |-- 4.5.3 Metrics export (compression ratio, quality scores)
       '-- 4.5.4 Documentation and API reference
```

### 5. C++ Minimal Inference Server [J4-1]

```
5. C++ Minimal Inference Server
   |-- 5.1 Architecture
   |   |-- 5.1.1 Determine scope: full server vs C++ decode loop inside Python
   |   |-- 5.1.2 HTTP/gRPC API layer (tokenizer, streaming, OpenAI compat)
   |   |-- 5.1.3 Model loading from NVFP4 checkpoint
   |   '-- 5.1.4 CUDA graph management (capture, replay, pool)
   |-- 5.2 Core decode loop
   |   |-- 5.2.1 CUDA graph replay per decode step
   |   |-- 5.2.2 Token sampling (top-k, top-p, temperature) on GPU
   |   |-- 5.2.3 KV cache management (block allocation, deallocation)
   |   |-- 5.2.4 MTP3 integration (multi-token verification)
   |   '-- 5.2.5 Continuous batching scheduler
   |-- 5.3 Baseline comparison
   |   |-- 5.3.1 Measure: CUDA graph replay time (should be ~2.94ms)
   |   |-- 5.3.2 Measure: C++ overhead per step (target < 0.1ms)
   |   |-- 5.3.3 Compare: C++ server vs vLLM (6,615 tok/s baseline)
   |   '-- 5.3.4 Compare: C++ server vs SGLang
   |-- 5.4 Feature parity
   |   |-- 5.4.1 Prefix caching
   |   |-- 5.4.2 Chunked prefill
   |   |-- 5.4.3 Preemption and swap
   |   '-- 5.4.4 Multi-request fairness
   '-- 5.5 Hybrid approach (alternative)
       |-- 5.5.1 Keep Python for scheduling, C++ extension for decode loop
       |-- 5.5.2 pybind11 wrapper for CUDA graph replay
       '-- 5.5.3 Measure overhead of Python->C++ boundary per step
```

**Key insight:** 5.5 (hybrid) is likely the right approach -- keep Python scheduling (where vLLM excels) but replace the per-token hot loop with C++. This eliminates most of the 5.38ms overhead while keeping vLLM's scheduling intelligence.

### 6. Disaggregated Prefill/Decode [E7]

```
6. Disaggregated Prefill/Decode
   |-- 6.1 Architecture
   |   |-- 6.1.1 Separate processes vs separate GPUs
   |   |-- 6.1.2 KV cache transfer mechanism (GPU->GPU or GPU->CPU->GPU)
   |   |-- 6.1.3 Scheduling: when to move request from prefill to decode pool
   |   '-- 6.1.4 vLLM experimental disaggregated serving API
   |-- 6.2 Prefill optimization (compute-bound)
   |   |-- 6.2.1 Higher precision for prefill (BF16 weights for quality)
   |   |-- 6.2.2 Larger batch sizes for prefill (amortize overhead)
   |   '-- 6.2.3 Flash-decoding during prefill for long contexts
   |-- 6.3 Decode optimization (bandwidth-bound)
   |   |-- 6.3.1 Maximum batch size for decode-only GPU
   |   |-- 6.3.2 FusenCache on decode GPU (4x more capacity)
   |   '-- 6.3.3 MTP3 on decode path
   |-- 6.4 KV transfer
   |   |-- 6.4.1 PCIe 5.0 transfer latency for FusenCache KV (compressed: 4x less data)
   |   |-- 6.4.2 NVLink transfer if same-machine GPUs
   |   |-- 6.4.3 Pipeline: start decode while last KV chunks transfer
   |   '-- 6.4.4 FusenCache compress during transfer (CPU quantizes?)
   '-- 6.5 Benchmarking
       |-- 6.5.1 TTFT (time to first token) improvement
       |-- 6.5.2 Decode throughput without prefill interference
       |-- 6.5.3 End-to-end throughput under mixed load
       '-- 6.5.4 Compare disaggregated vs unified at various load levels
```

### 7. Nsight Compute Hot-Kernel Profiling [J2-4]

```
7. Nsight Compute Profiling
   |-- 7.1 Identify hot kernels
   |   |-- 7.1.1 Top 5 kernels by cumulative time in decode step
   |   |-- 7.1.2 Top 5 kernels by cumulative time in prefill
   |   '-- 7.1.3 MoE dispatch kernels specifically
   |-- 7.2 Memory bandwidth analysis
   |   |-- 7.2.1 Achieved HBM bandwidth per kernel (vs 1792 GB/s peak)
   |   |-- 7.2.2 L2 hit rate per kernel
   |   |-- 7.2.3 Shared memory utilization
   |   '-- 7.2.4 Identify memory-bound vs compute-bound per kernel
   |-- 7.3 Compute analysis
   |   |-- 7.3.1 Tensor core utilization per kernel
   |   |-- 7.3.2 Warp execution efficiency
   |   |-- 7.3.3 Instruction mix (FP4, FP16, INT, memory)
   |   '-- 7.3.4 Occupancy vs theoretical max
   |-- 7.4 Bottleneck identification
   |   |-- 7.4.1 Why are we at 13% of bandwidth limit? (experiment #29)
   |   |-- 7.4.2 Is MoE dispatch overhead in routing or in GEMM?
   |   |-- 7.4.3 What is the gap between kernel time and wall time?
   |   '-- 7.4.4 Are there stalls between kernel launches?
   '-- 7.5 Actionable output
       |-- 7.5.1 Ranked list of optimization opportunities by expected impact
       |-- 7.5.2 Updated roofline model for each hot kernel
       '-- 7.5.3 Feed findings into fused MoE kernel design (item #1)
```

**Automation potential:** Fully automatable. Write a script that runs `ncu` on top kernels and extracts metrics. This should be the FIRST thing done -- it informs all other decisions.

### 8. Expert Merging (128 -> 64) [D3]

```
8. Expert Merging
   |-- 8.1 Expert similarity analysis
   |   |-- 8.1.1 Compute pairwise cosine similarity of expert weight matrices
   |   |-- 8.1.2 Cluster experts by weight similarity (hierarchical clustering)
   |   |-- 8.1.3 Cluster experts by activation pattern similarity
   |   '-- 8.1.4 Identify merge candidates (high similarity pairs/groups)
   |-- 8.2 Merging strategy
   |   |-- 8.2.1 Simple averaging: W_merged = (W_a + W_b) / 2
   |   |-- 8.2.2 Weighted averaging by activation frequency
   |   |-- 8.2.3 SLERP (spherical interpolation) of expert weights
   |   |-- 8.2.4 Distillation-based merging (calibration data)
   |   '-- 8.2.5 Hierarchical merging: 128->96->64 in stages
   |-- 8.3 Router adjustment
   |   |-- 8.3.1 Remap routing logits from merged experts
   |   |-- 8.3.2 Adjust top-K (top-8 of 128 -> top-4 of 64?)
   |   |-- 8.3.3 Fine-tune router weights on calibration data
   |   '-- 8.3.4 Test routing entropy before vs after merging
   |-- 8.4 Quality validation
   |   |-- 8.4.1 PPL at each merge step
   |   |-- 8.4.2 Task-specific benchmarks
   |   '-- 8.4.3 Expert-count vs quality Pareto curve
   '-- 8.5 Performance
       |-- 8.5.1 Fewer experts = less dispatch overhead
       |-- 8.5.2 Larger per-expert batch = better GEMM efficiency
       '-- 8.5.3 Updated fused MoE kernel for 64-expert config
```

### 9. Layer Pruning [D4]

```
9. Layer Pruning
   |-- 9.1 Layer importance scoring
   |   |-- 9.1.1 Angular distance: measure how much each layer changes hidden states
   |   |-- 9.1.2 Gradient-based importance (Fisher information, if gradients available)
   |   |-- 9.1.3 Ablation: remove each layer, measure PPL increase
   |   '-- 9.1.4 Attention entropy: layers with near-uniform attention may be redundant
   |-- 9.2 Pruning strategy
   |   |-- 9.2.1 Remove lowest-importance layers (greedy)
   |   |-- 9.2.2 Remove contiguous blocks (e.g., layers 12-14) for simpler CUDA graphs
   |   |-- 9.2.3 Remove MoE layers only (preserve attention)
   |   '-- 9.2.4 Remove attention layers only (preserve MoE)
   |-- 9.3 Healing
   |   |-- 9.3.1 Light fine-tuning after pruning (if GPU budget available)
   |   |-- 9.3.2 Residual scaling: adjust skip-connection weights
   |   '-- 9.3.3 No healing: test raw pruning quality
   '-- 9.4 Benchmarking
       |-- 9.4.1 Throughput at each pruning level
       |-- 9.4.2 Quality at each pruning level
       '-- 9.4.3 Pruning depth vs quality Pareto curve
```

### 10. QAT Checkpoint [C5]

```
10. QAT Checkpoint
    |-- 10.1 Monitoring
    |   |-- 10.1.1 Watch Google's Gemma4 release page for QAT variants
    |   |-- 10.1.2 Watch community quantizations (TheBloke, etc.)
    |   '-- 10.1.3 Track NVIDIA modelopt QAT recipes for Gemma4
    |-- 10.2 Integration (when available)
    |   |-- 10.2.1 Convert QAT checkpoint to NVFP4 format
    |   |-- 10.2.2 Verify FusenCache compatibility
    |   |-- 10.2.3 Run full quality benchmark suite
    |   '-- 10.2.4 Compare QAT PPL vs current PTQ PPL (701.4)
    '-- 10.3 Self-QAT (alternative)
        |-- 10.3.1 Use NVIDIA modelopt to run QAT on Gemma4
        |-- 10.3.2 Requires multi-GPU training setup
        '-- 10.3.3 Estimate: 1-2 days on 8xH100
```

### 11-20: Remaining Top 20 (summarized decomposition)

**11. EAGLE Speculative Decoding [A5]:**
- Draft head training (EAGLE architecture on Gemma4 hidden states)
- MoE verification cost analysis (top-8 dispatch per verification token)
- vLLM integration debugging (fix experiment #14/#19/#22 blockers)
- Tree-based verification for multiple candidates
- Acceptance rate measurement across domains
- MTP3 + EAGLE compounding test

**12. 2:4 Structured Sparsity [C12-str]:**
- Verify SM120 FP4 + 2:4 sparsity tensor core compatibility
- Identify which expert weights tolerate 50% sparsity
- Apply 2:4 pruning with calibration data
- Measure quality impact per expert
- CUTLASS sparse GEMM kernel activation

**13. CUTLASS 3.x SM120 Audit [G8/G9]:**
- Audit CUTLASS 3.x release notes for SM120-specific features
- Audit CUDA 12.8 for new intrinsics (thread block clusters, etc.)
- Benchmark CUTLASS grouped GEMM vs current MoE path
- Test CUTLASS persistent kernel for MoE dispatch
- Tune CUTLASS templates for Gemma4 expert shapes

**14. Cross-Layer KV Sharing [B12]:**
- Measure attention pattern similarity across 25 sliding layers
- Group layers by KV similarity (groups of 5?)
- Implement shared KV cache for layer groups
- Quality validation per sharing configuration
- Memory savings -> batch capacity gain

**15. KV Cache Offloading [E8]:**
- Enable vLLM --swap-space flag
- Measure swap latency (PCIe 5.0 x16 = ~64 GB/s)
- Test with FusenCache (offload compressed KV = 4x less data)
- Measure throughput impact of preemption under load

**16. Request Scheduling [E9]:**
- Implement SJF (shortest-job-first) policy in vLLM
- A/B test FCFS vs SJF vs priority queue
- Measure P50/P90/P99 latency under mixed workloads
- Implement fairness-weighted scheduling for multi-tenant

**17. FlashAttention-3 [B2]:**
- Check FA3 SM120 support status
- If available: benchmark FA3 vs FlashInfer attention
- Measure prefill improvement (main benefit)
- System-level impact (attention is 11.5% of decode)

**18. MoE Shuffle + Quant Fusion [F1-moe]:**
- Implement 5-line CUDA change from moe_shuffle_fusion_analysis.md
- Benchmark 2.3% system gain claim
- Verify correctness with existing test suite

**19. FP8 Activation Quantization [C2]:**
- Profile activation magnitudes between MoE experts
- Implement FP8 inter-expert activation compression
- Measure bandwidth savings vs quality impact
- Test with NVFP4 expert weights (FP8 activations + FP4 weights)

**20. Data Parallelism [E6]:**
- Launch 2x vLLM instances on 2x GPUs
- Load balancer for request routing
- Compare DP=2 vs TP=2 throughput
- Memory: each instance needs full model (fits in 32GB)

---

## Part 3: Data-Driven Methodology Integration

### 3.1 What Can Be Automated Per Category

**A. Decoding Strategies:**
- Automatable: Sweep MTP depth (1-5), measure tok/s and acceptance rate
- Automatable: N-gram lookahead table construction from corpus
- Requires human: Deciding between spec decode architectures (EAGLE vs Medusa vs REST)
- Auto-discovery: Monitor acceptance rate in production, auto-adjust MTP depth

**B. Attention Optimizations:**
- Automatable: KV eviction threshold sweep, attention sparsity pattern profiling
- Automatable: Cross-layer similarity scoring
- Requires human: Deciding to switch attention backends (FA2 vs FA3 vs FlashInfer)
- Auto-discovery: Profile attention entropy per layer, auto-select eviction/sharing candidates

**C. Quantization and Compression:**
- Automatable: Bit-width sweep per module (weight/activation/KV), quality-throughput Pareto
- Automatable: Calibration dataset curation and quantization parameter search
- Requires human: Choosing between quantization paradigms (PTQ vs QAT, NVFP4 vs AQLM)
- Auto-discovery: Per-layer sensitivity analysis (perturb weights, measure output delta)

**D. Model Architecture:**
- Automatable: Expert usage profiling, layer importance scoring, pruning sweep
- Automatable: Expert similarity clustering, merge candidate identification
- Requires human: Deciding pruning aggressiveness, quality threshold
- Auto-discovery: Automated architecture search: remove components, measure quality, keep if acceptable

**E. System-Level:**
- Automatable: Scheduling policy A/B test, batch size sweep, TP/DP config search
- Automatable: KV offloading threshold tuning, prefix cache hit rate monitoring
- Requires human: Architecture decisions (disaggregated vs unified, C++ vs Python)
- Auto-discovery: Adaptive scheduling that learns from request patterns

**F. Kernel-Level:**
- Automatable: Autotuning (already proven -- 15 TFLOPS -> 328 TFLOPS), register pressure sweep
- Automatable: Fusion opportunity detection (scan adjacent ops in trace)
- Requires human: Novel kernel architecture (persistent, cooperative groups)
- Auto-discovery: Computation graph analysis to find fusable op sequences (see 3.3)

**G. Hardware-Specific:**
- Automatable: L2 partition ratio sweep, NVLink bandwidth test
- Automatable: CUDA API feature detection and benchmark
- Requires human: Understanding hardware microarchitecture
- Auto-discovery: Microbenchmark suite that discovers hardware capabilities

### 3.2 Automation Infrastructure Needed

| Tool | Purpose | Effort | Techniques It Enables |
|------|---------|--------|----------------------|
| **Expert profiler** | Log expert activations across diverse prompts | 1 day | D2, D3, E5 |
| **Quality eval suite** | HumanEval + GSM8K + MMLU + MT-Bench automated | 1 day | All quality-affecting opts |
| **Pareto plotter** | Given (quality, throughput) pairs, plot frontier | Hours | All quantization, pruning |
| **Nsight Compute wrapper** | Run ncu, extract metrics, produce report | Hours | All kernel opts |
| **Config sweeper** | vLLM config grid search (batch, cache, scheduling) | Hours | E2, E8, E9, E10, E11 |
| **Layer ablation harness** | Remove layers/experts, auto-evaluate quality + speed | 1 day | D2, D3, D4, B12 |
| **Fusion detector** | Scan torch.profiler trace, find adjacent memory-bound ops | Days | F1 |
| **A/B test framework** | Compare two configs under same load, report significance | Days | All |

### 3.3 The Auto-Discovery System

How a future system could find optimizations without being told:

**Computation Graph Analysis:**
```
1. Profile the model with torch.profiler
2. Build a DAG of all ops with timing and memory traffic
3. For each pair of adjacent ops:
   a. If output of op A = input of op B (no other consumers):
      -> Fusion candidate (eliminate intermediate write/read)
   b. Estimate bandwidth savings from fusion
   c. Rank by (bandwidth_saved / implementation_effort_proxy)
4. Auto-generate fused Triton kernel skeleton
5. Autotune the fused kernel
6. Benchmark fused vs unfused
7. Keep if faster, revert if not
```

**Performance Model-Guided Search:**
```
1. For each kernel, compute roofline position (FLOPS/byte)
2. If memory-bound: try reducing memory traffic (quantize, fuse, cache)
3. If compute-bound: try reducing compute (prune, approximate, lower precision)
4. Predict speedup from each intervention using the roofline model
5. Sort by predicted speedup / implementation effort
6. Implement top candidates, verify predictions
7. Update performance model with measured results (learn from errors)
```

**Transfer Learning from Other Models/GPUs:**
```
1. Maintain a database of (optimization, model_type, gpu, measured_speedup)
2. For a new (model, gpu) target:
   a. Find most similar entries in database
   b. Rank optimizations by expected speedup (weighted by similarity)
   c. Try optimizations in predicted order
3. Update database with new measurements
4. Over time, learn which optimizations transfer across models/GPUs
```

**Automated A/B Testing in Production:**
```
1. Shadow-deploy experimental config alongside production
2. Route 1% of traffic to experimental
3. Measure: throughput, latency, quality (via LLM-as-judge or human eval)
4. Promote if strictly Pareto-better on all metrics
5. Auto-generate the next experiment based on what worked
```

### 3.4 The Meta-Question: Building a Self-Improving Optimization System

The ultimate goal is a system that:
1. **Observes** the current bottleneck (via profiling)
2. **Hypothesizes** an optimization (from a library of techniques + learned heuristics)
3. **Implements** the optimization (code generation or config change)
4. **Evaluates** the result (automated benchmarking + quality checks)
5. **Learns** from the outcome (update heuristics, prune dead ends)

This is essentially what AutoKernel already does for individual kernels. Scaling it to system-level optimization requires:
- A broader action space (not just kernel.py, but vLLM config, model architecture, hardware setup)
- Multi-objective evaluation (throughput, latency, quality, memory, cost)
- Longer time horizons (some optimizations take days to evaluate properly)
- Safety constraints (never deploy a config that degrades quality below threshold)

---

## Part 4: Interaction Matrix

### 4.1 Compounding Matrix

Techniques that MULTIPLY when combined (independent bottleneck reduction):

```
                    Fused   Expert  TP=2  FusenCache  C++     Disagg  Layer   EAGLE   2:4
                    MoE     Prune               KV     Server  PD     Prune   Spec    Sparse
Fused MoE           --      ++      ++    +           ++      +       ++      +       ++
Expert Prune        ++      --      +     +           +       +       +       +       ++
TP=2                ++      +       --    ++          +       ++      +       +       +
FusenCache KV       +       +       ++    --          +       ++      +       +       +
C++ Server          ++      +       +     +           --      +       +       +       +
Disagg PD           +       +       ++    ++          +       --      +       +       +
Layer Prune         ++      +       +     +           +       +       --      +       +
EAGLE Spec          +       +       +     +           +       ?       +       --      +
2:4 Sparse          ++      ++      +     +           +       +       +       +       --
```

Legend: ++ = strong compound (multiplicative), + = stacks (additive), ? = unknown interaction

### 4.2 Conflict Matrix

Techniques that CONFLICT or are mutually exclusive:

```
CUDA graphs          vs  Early exit / Dynamic depth    HARD CONFLICT (static vs dynamic)
CUDA graphs          vs  torch.compile full-model      SOFT CONFLICT (competing capture)
NVFP4                vs  AQLM / QuIP# / SqueezeLLM    EXCLUSIVE (pick one weight format)
TP=2                 vs  DP=2 on same GPU pair          EXCLUSIVE (same hardware, different use)
MTP3                 vs  Medusa heads                   EXCLUSIVE (both predict future tokens)
FP8 KV               vs  FusenCache K4V4               EXCLUSIVE per layer (pick one KV format)
Expert pruning       vs  Expert parallelism             WEAK CONFLICT (fewer experts = less to parallelize)
Disaggregated PD     vs  Single-GPU deployment          REQUIRES multi-process/multi-GPU
```

### 4.3 Compounding Ceiling Analysis

Best-case compounding of top 5 techniques on batch throughput:

```
Current baseline:                     6,615 tok/s
+ Fused MoE dispatch (2x on 85%):    ~11,200 tok/s  (1.69x system)
+ Expert pruning (20% removed):       ~13,400 tok/s  (1.20x)
+ TP=2:                                ~24,100 tok/s  (1.80x)
+ FusenCache (4x KV -> higher batch): ~33,700 tok/s  (1.40x from batch headroom)
+ C++ server (eliminate Python):       ~55,600 tok/s  (1.65x)
                                       --------
Theoretical compounded ceiling:       ~55,600 tok/s  (8.4x current)
```

This ceiling assumes all gains are independent and multiplicative on their respective bottlenecks. In reality, some will interact sub-additively. A realistic estimate is 4-6x current throughput.

Best-case for single-user decode latency:

```
Current baseline:                     186 tok/s (MTP3)
+ EAGLE spec decode (2.5x):           ~465 tok/s
+ C++ server (1.65x):                 ~767 tok/s
+ Fused MoE (1.5x on decode):         ~1,150 tok/s
                                       --------
Theoretical ceiling:                  ~1,150 tok/s single-user
Realistic estimate:                   ~400-600 tok/s
```

---

## Part 5: Blind Spots and Self-Critique

### 5.1 What We Are Probably Wrong About

**1. "The fused MoE kernel will give 2-5x."**
This is our #1 priority based on the premise that launch overhead dominates. But experiment #92 showed wall time = GPU time (8.24ms), meaning CUDA graphs already eliminate most launch overhead. The 2-5x claim comes from pre-CUDA-graph measurements. With CUDA graphs already capturing the decode step, the fused MoE kernel's benefit may be 1.1-1.3x, not 2-5x. The Nsight Compute profiling (item #7) must happen BEFORE investing weeks in the fused MoE kernel.

**2. "Expert pruning is safe at 20%."**
We have zero data on expert activation distributions. If Gemma4's experts are well-balanced (all 128 used roughly equally), there may be no "cold" experts to prune. The 20% estimate is based on findings from other MoE models (Switch Transformer, Mixtral) that may not transfer.

**3. "C++ server gives 1.65x."**
The 5.38ms "vLLM overhead" from experiment #87 may not be Python overhead at all -- it could be CUDA synchronization, memory management, or scheduling logic that a C++ server would also need. If the overhead is in GPU-side operations (not CPU-side Python), eliminating Python gives nothing.

**4. "TP=2 scales at 1.8x."**
On consumer NVLink (~100 GB/s), the all-reduce for 128-expert MoE routing may be a significant fraction of decode time. If routing requires communicating expert assignments for every token across GPUs, the NVLink becomes the bottleneck. TP=2 might only give 1.3-1.4x on consumer hardware.

**5. "More batch size = more throughput."**
We assume FusenCache's 4x KV capacity translates to proportionally higher throughput. But experiment #85 showed throughput plateaus at B~232. The bottleneck may shift from memory to compute at high batch, making additional KV capacity irrelevant for throughput (though still valuable for capacity/cost).

### 5.2 What We Have Not Measured That Would Change Priorities

**1. Expert activation distribution.** If experts are balanced, skip D2/D3 entirely. If 20% of experts handle 80% of tokens, D2 jumps to #1 priority.

**2. True source of the 5.38ms overhead.** If it is Python: C++ server is critical. If it is GPU sync: C++ server is worthless and the fix is different.

**3. SM120 FP4 + 2:4 sparsity compatibility.** If SM120 supports both simultaneously, 2:4 sparsity jumps to top 3 (free 2x GEMM speedup on already-quantized weights). If not, remove it from roadmap entirely.

**4. CUTLASS grouped GEMM vs current vLLM MoE on SM120.** If CUTLASS 3.x has a 20% better grouped GEMM for our shapes, this is a quick win. If it matches current performance, skip G8.

**5. Bandwidth utilization inside hot kernels.** The 13% of theoretical bandwidth (experiment #29) is alarming. If we are leaving 87% of bandwidth on the table, the fix is not "fused kernel" but "better memory access patterns." This completely changes the optimization target.

### 5.3 Biggest Risk of Over-Optimizing the Wrong Thing

**Risk: We are optimizing for Gemma4 26B specifically, but the model changes.**
Google could release Gemma4.1, Gemma5, or a fundamentally different architecture. Every kernel we write for 128-expert top-8 MoE with specific tensor shapes becomes tech debt. Mitigation: build abstractions (AutoKernel's kernel types, FusenCache's spec system) rather than hardcoded solutions.

**Risk: We are optimizing for RTX 5090, which is consumer hardware.**
Production deployment may use H100, B200, or custom ASICs with completely different characteristics. Optimizations tuned for SM120's L2 size, memory bandwidth, and FP4 tensor core throughput may not transfer. Mitigation: parameterize all optimizations by hardware capability, not hardcoded constants.

**Risk: We are optimizing batch throughput when the market wants latency.**
Interactive AI assistants need sub-100ms time-to-first-token and 30+ tok/s per user. Our 6,615 tok/s at B=232 means ~28.5 tok/s per user -- barely adequate for interactive use. Chasing higher batch throughput may be the wrong objective if it comes at the expense of P99 latency.

**Risk: We are optimizing inference when the model quality is the problem.**
PPL=701.4 on WikiText-2 is significantly degraded from FP16. If FP4 quality is insufficient for target use cases, all our throughput optimization is wasted -- we should be optimizing quality (QAT, better calibration, mixed precision) instead of speed.

### 5.4 What a Completely Different Approach Would Look Like

**Approach A: Throw away vLLM, build from scratch.**
A custom inference engine in C++/CUDA that:
- Loads NVFP4 weights directly
- Captures the entire 30-layer forward pass as one CUDA graph (with conditional expert dispatch via CUDA graphs with conditional nodes, available in CUDA 12.8+)
- Zero Python in the hot path
- Custom memory allocator tuned for MoE expert weight access patterns
- Expected outcome: 2-3x over vLLM (eliminates ALL framework overhead)
- Risk: 2-3 months of engineering, loses vLLM's ecosystem

**Approach B: Use a different model entirely.**
- Qwen3.5-9B: 4,471 tok/s at B=64 (experiment #33). Dense model, no MoE complexity.
- If quality is sufficient (9B models are surprisingly good now), the throughput-per-dollar may be better.
- Alternatively: wait for a model with native MLA (like DeepSeek V3) where KV cache is inherently tiny.
- Expected outcome: Simpler system, potentially comparable quality, much less optimization needed.

**Approach C: Hardware scaling instead of software optimization.**
- Buy 4x RTX 5090 with NVLink. DP=4 gives 4x throughput trivially.
- Cost: ~$8,000 in hardware vs weeks of engineering time.
- At $150/hour engineering cost, 2 weeks of optimization = $12,000.
- Pure hardware scaling is cheaper than software optimization for < 4x gains.

**Approach D: Specialize for the workload.**
- Instead of general-purpose LLM serving, optimize for specific use cases.
- Customer support bot: fine-tune smaller model, prefix-cache the system prompt, enable REST speculative decoding from response templates.
- Code assistant: higher precision model, no MoE (dense for code), speculative decoding with code-specific draft model.
- This gives 3-10x improvement for specific workloads by narrowing the problem.

### 5.5 Where Our Mental Model of GPU Execution Is Wrong or Incomplete

**1. We think in terms of individual kernel launches, but CUDA graphs change everything.**
With 86 captured CUDA graphs (experiment #13), the execution model is fundamentally different from sequential kernel launches. The GPU scheduler sees the entire graph and can overlap, reorder, and pipeline operations we think are sequential. Our optimization model (reduce each kernel's time) may be wrong -- the right model is (reduce the critical path through the graph).

**2. We assume memory bandwidth is a hard wall, but it is not a single number.**
"1792 GB/s" is peak HBM bandwidth. But effective bandwidth depends on access pattern (sequential vs random), transaction size (32B vs 128B), and L2 cache state. Our expert weight access (128 experts, only 8 active per token, random selection) has terrible spatial locality. The effective bandwidth for expert weight loading may be 500-800 GB/s, not 1792 GB/s. This changes the roofline analysis.

**3. We treat MoE dispatch as "launch 8 GEMMs," but CUTLASS grouped GEMM is more nuanced.**
CUTLASS grouped GEMM packs multiple problems into a single kernel launch, but the internal scheduling (which thread blocks handle which expert) depends on the batch distribution across experts. If one expert gets 50% of tokens and another gets 1%, the load imbalance means some SMs finish early and idle. We have not measured load imbalance.

**4. We do not understand the L2 cache behavior for our workload.**
The 48MB L2 on RTX 5090 is larger than some expert weights (2.4MB each). If the GPU's prefetcher is smart about expert weight caching (keeping hot experts in L2), our MoE dispatch may be faster than we think. If the L2 thrashes (128 experts cycling through 48MB), it may be slower. We have zero L2 hit rate data.

**5. We assume FP4 tensor core throughput is the ceiling, but instruction scheduling matters.**
Even with FP4 tensor cores, if the instruction pipeline is stalled waiting for operands (memory latency not hidden by software pipelining), actual throughput can be well below peak. Our 1,261 TFLOPS measurement on isolated GEMM may not reflect in-model throughput where other operations (routing, normalization, quantization) compete for the same execution resources.

---

## Part 6: Execution Roadmap

### Phase 1: Measure (Week 1)

Do NOT build anything. Measure everything.

| Day | Action | Output |
|-----|--------|--------|
| 1 | Nsight Compute profiling of top 10 kernels in decode | Memory bandwidth utilization, compute utilization, stall reasons |
| 1 | Expert activation profiling (1000 diverse prompts) | Per-expert activation frequency distribution |
| 2 | L2 cache hit rate measurement during MoE dispatch | Whether expert weights are cached or thrashing |
| 2 | Decompose 5.38ms vLLM overhead (CPU profiling with py-spy) | What fraction is Python vs C++ vs CUDA sync |
| 3 | Verify SM120 FP4 + 2:4 sparsity tensor core compatibility | Binary: supported or not |
| 3 | Benchmark CUTLASS 3.x grouped GEMM for Gemma4 expert shapes | Whether CUTLASS beats current vLLM MoE |
| 4 | Task-specific quality eval (HumanEval, GSM8K, MMLU, MT-Bench) | Quality score for NVFP4 Gemma4 |
| 4 | P99 latency under realistic load (Poisson arrivals, mixed lengths) | Tail latency distribution |
| 5 | Synthesize all measurements into updated priority scores | Revised roadmap based on data, not assumptions |

**Expected outcome:** At least 3 of our top-10 assumptions will be wrong. The revised priority order after measurement will be significantly different from the current ranking.

### Phase 2: Quick Wins (Week 2)

Based on Phase 1 findings, execute the quick wins that measurement validates:

- If expert distribution is skewed: expert pruning (D2) -- estimated 2 days
- If CUTLASS grouped GEMM is faster: swap MoE backend -- estimated 1 day
- If L2 partitioning API is available: test expert weight caching -- estimated 1 day
- vLLM scheduling optimization (SJF for mixed workloads) -- estimated 0.5 days
- MoE shuffle + quant fusion -- estimated 0.5 days
- KV offloading flag for capacity extension -- estimated 0.5 days

### Phase 3: Strategic Build (Weeks 3-8)

Build the top strategic investments validated by Phase 1 data:

**Weeks 3-4: FusenCache production integration**
- vLLM per-layer allocator PR
- CUDA graph integration
- Quality validation suite

**Weeks 5-6: Based on Phase 1 findings, ONE of:**
- (a) Fused MoE dispatch kernel (if kernel launch overhead is confirmed as bottleneck)
- (b) C++ decode loop (if Python overhead is confirmed as bottleneck)
- (c) Expert merging pipeline (if expert redundancy is confirmed)

**Weeks 7-8: TP=2 integration**
- If we have two GPUs available
- Full benchmark with FusenCache + TP=2

### Phase 4: Research Bets (Weeks 9-12)

- EAGLE speculative decoding for single-user latency
- 2:4 structured sparsity (if Phase 1 confirmed compatibility)
- Cross-layer KV sharing
- Layer pruning

### Ongoing: Measurement-Driven Iteration

After each phase, re-measure and re-prioritize. The roadmap is a living document that updates based on what we learn.

---

## Appendix A: Technique Count Summary

| Category | Total | Deployed | Actionable | Not Applicable | Research |
|----------|-------|----------|------------|----------------|----------|
| A. Decoding | 14 | 3 (A1, A3, A11) | 4 (A2, A5, A6, A13) | 5 (A7-A10, A14) | 2 (A4, A12) |
| B. Attention | 15 | 5 (B1, B3, B5, B11, B13) | 4 (B2, B8, B9, B12) | 5 (B4, B6, B7, B14, B15) | 1 (B10) |
| C. Quantization | 14 | 4 (C1, C3, C4, C6) | 4 (C2, C5, C12, C14) | 4 (C7-C9, C11) | 2 (C10, C13) |
| D. Architecture | 10 | 0 | 3 (D2, D3, D4) | 5 (D5-D10) | 2 (D5, D6) |
| E. System | 14 | 5 (E1, E2, E11, E12, E14) | 5 (E3, E6, E7, E8, E9) | 2 (E4, E5) | 2 (E10, E13) |
| F. Kernel | 13 | 8 (F2, F5-F8, F10, F12, F13) | 3 (F1, F3, F11) | 0 | 2 (F4, F9) |
| G. Hardware | 9 | 3 (G1, G2, G6) | 4 (G3, G5, G8, G9) | 1 (G4-needs HW) | 1 (G7) |
| H. Math/Algo | 13 | 2 (H1, H2) | 1 (H3) | 9 (H4-H12) | 1 (H13) |
| I. Emerging | 9 | 0 | 2 (I1, I3) | 5 (I5-I9) | 2 (I2, I4) |
| J. Blindspots | 5 | 0 | 5 | 0 | 0 |
| **Total** | **116** | **30** | **35** | **36** | **15** |

30 techniques are already deployed. 35 are actionable. 36 do not apply to our setup. 15 are research-grade.

Of the 35 actionable techniques, the top 10 account for an estimated 80%+ of remaining achievable gains.

---

## Appendix B: Decision Framework

When choosing what to build next, apply this filter in order:

1. **Have we measured the bottleneck?** If no, measure first (Phase 1). Optimization without measurement is gambling.
2. **Does it compound with deployed optimizations?** Prefer techniques that multiply with our existing stack (NVFP4 + CUDA graphs + MTP3 + continuous batching).
3. **Is the confidence high?** Prefer proven/likely over speculative. Research bets are for after quick wins are exhausted.
4. **Is the effort proportional to impact?** A 5% gain that takes 1 hour beats a 10% gain that takes 1 week.
5. **Does it improve the metric we care about?** Batch throughput for cost efficiency. Single-user latency for interactive use. Both matter, but know which you are optimizing for today.
6. **Does it create optionality?** FusenCache integration enables future batch scaling AND TP=2 scaling AND CPU offloading. Prefer techniques that open doors.
7. **Can we revert if it fails?** Prefer reversible changes (config flags, checkpoint swaps) over irreversible ones (model surgery, custom frameworks).

---

*This roadmap should be revised after Phase 1 measurement. At least 3 of our current assumptions will be proven wrong by data. The roadmap that survives contact with measurement will be more valuable than this pre-measurement version.*
