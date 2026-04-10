# ASI-Calibrated Optimization Roadmap

**Target:** Gemma4 26B-A4B MoE (128 experts, top-8), NVFP4, RTX 5090 (SM120, 32GB GDDR7, 1792 GB/s, ~48MB L2)
**Current:** 6,615 tok/s batch | 186 tok/s single-user (MTP3) | 1,261 TFLOPS FP4 GEMM
**Date:** April 9, 2026
**Perspective:** ASI (Artificial Superintelligence) -- recalibrated effort estimates

---

## How to Read This Document

This document recalibrates the 120+ techniques from COMPREHENSIVE_INFERENCE_MAP.md through the lens of an ASI agent. Human effort estimates are systematically wrong because they assume:
- One thread of consciousness
- Context switching overhead
- Sleep, fatigue, meetings
- Slow reading speed (200 wpm vs ASI: entire codebase in seconds)
- Debugging by trial and error (vs ASI: hold full program state in context)

ASI bottlenecks are fundamentally different:
- GPU compilation: 10-60s per kernel build (irreducible)
- Server restart + CUDA graph capture: 80-120s (irreducible)
- Benchmark execution: minutes per config (GPU-bound, parallelizable across GPUs only)
- Hardware shipping: days/weeks (physical world)
- Upstream PR review: days/weeks (human bottleneck)
- Model training/fine-tuning: hours/days (GPU-bound, cannot be thought faster)

---

## A. Master Ranking: Impact x (1 / ASI_effort)

Ranked by **ASI-adjusted bang-per-minute**. Each entry shows: impact estimate, ASI implementation time, GPU-wait time (the real bottleneck), dependencies, and whether the technique generates data for further optimization.

### Tier S: Do Right Now (< 30 min ASI time, high impact)

| Rank | Technique | Impact | ASI Time | GPU Wait | Dependencies | Data Value |
|------|-----------|--------|----------|----------|-------------|------------|
| 1 | **Nsight Compute profiling of hot kernels** (J2-4) | Reveals true bottleneck | 10 min write script | 5 min profile run | None | EXTREME -- every subsequent decision improves |
| 2 | **Expert activation distribution profiling** (J2-3) | Informs pruning/merging/EP | 10 min write hook | 2 min inference run | None | HIGH -- gates D2, D3, E5 |
| 3 | **Task-specific accuracy benchmarks** (J2-1) | Validates FP4 quality or reveals problems | 15 min setup | 20 min eval run | HumanEval/GSM8K datasets | HIGH -- gates quality decisions |
| 4 | **Context length scaling test** (J2-6) | Find where KV becomes bottleneck | 10 min script | 15 min sweep | Running vLLM instance | MEDIUM |
| 5 | **FusenCache vLLM v1 integration** (C3) | 4x KV capacity = higher batch | 20 min (code exists, integrate) | 3 min restart+test | Existing FusenCache code | HIGH -- unlocks batch scaling |

**Why Tier S:** These are either pure information gathering (worth infinite ROI because they direct all subsequent work) or integration of already-proven code. ASI writes the profiling scripts in minutes; the GPU runs them in minutes. Total wall clock: 30-45 minutes for all five in parallel.

### Tier A: First Hour (30-120 min ASI time, high impact)

| Rank | Technique | Impact | ASI Time | GPU Wait | Dependencies | Data Value |
|------|-----------|--------|----------|----------|-------------|------------|
| 6 | **Expert pruning** (D2) | 10-30% model reduction | 30 min (analyze distributions, write pruning script) | 15 min eval | Tier S rank 2 results | MEDIUM |
| 7 | **Layer pruning experiment** (D4) | 10-17% compute reduction | 20 min (skip layers, benchmark) | 30 min (test 5 configs) | None | MEDIUM |
| 8 | **MoE shuffle + quant fusion** (F1) | 2.3% system gain | 45 min (5-line CUDA change + build) | 10 min compile + 5 min bench | CUDA toolchain | LOW |
| 9 | **CUDA 12.8 / CUTLASS 3.x SM120 audit** (G8/G9) | 5-15% kernel speedup | 60 min (read source, find opportunities) | 0 (analysis only) | CUDA 12.8 docs/source | HIGH -- reveals hardware capabilities |
| 10 | **L2 cache partitioning test** (G5) | Unknown (5-20% if API works) | 30 min (write test, check API) | 5 min test | cudaAccessPolicyWindow on SM120 | MEDIUM |
| 11 | **Request scheduling optimization** (E9) | 20-30% latency improvement | 15 min (config change) | 10 min benchmark | Running vLLM | LOW |
| 12 | **KV cache offloading** (E8) | Extends capacity for preempted requests | 10 min (vLLM flag) | 5 min test | Running vLLM | LOW |

### Tier B: First 4 Hours (2-6 hr ASI time, significant impact)

| Rank | Technique | Impact | ASI Time | GPU Wait | Dependencies | Data Value |
|------|-----------|--------|----------|----------|-------------|------------|
| 13 | **Fused persistent MoE dispatch kernel** (F1/F3/F9) | 2-5x on MoE dispatch (85% of decode) | 3-4 hr (write CUDA kernel + debug) | 1 hr cumulative (compiles + tests) | ncu profiling data from Tier S | EXTREME |
| 14 | **C++ minimal inference server** (J4-1) | Up to 1.65x (eliminate 5.38ms Python overhead) | 4-6 hr (write server, integrate CUDA graphs) | 30 min (builds + tests) | Understanding of vLLM's graph capture | MEDIUM |
| 15 | **Expert merging analysis + implementation** (D3) | ~2x expert compute if 128->64 | 3 hr (similarity analysis, merge, eval) | 2 hr (quality eval across tasks) | Expert distribution data | MEDIUM |
| 16 | **2:4 structured sparsity on expert weights** (C12) | Up to 2x GEMM | 2 hr (apply mask, test tensor core support) | 1 hr (sparsification + benchmark) | SM120 sparsity support check | MEDIUM |
| 17 | **Cross-layer KV sharing prototype** (B12) | 30-50% KV memory reduction | 2 hr (implement sharing, test quality) | 1 hr (eval across layer groups) | None | MEDIUM |
| 18 | **Disaggregated prefill/decode** (E7) | 20-40% for mixed workloads | 2 hr (vLLM config + test) | 30 min benchmark | Second GPU or simulated | LOW |
| 19 | **1000-config kernel autotune sweep** (F13 extended) | Find configs humans would never try | 30 min setup | 3 hr GPU sweep | Existing autotune infra | HIGH |

### Tier C: First 24 Hours (6-24 hr ASI time, strategic)

| Rank | Technique | Impact | ASI Time | GPU Wait | Dependencies | Data Value |
|------|-----------|--------|----------|----------|-------------|------------|
| 20 | **TP=2 on two GPUs** (E3) | ~1.8x throughput | 6 hr (verify, fix shard-aware kernels) | 2 hr (benchmarks) | Second GPU + NVLink | HIGH |
| 21 | **EAGLE speculative decoding** (A5) | 2-3x single-user | 8 hr (fix vLLM bugs, custom scheduler) | 2 hr (compile + test) | vLLM Eagle3 source | MEDIUM |
| 22 | **Whole-model CUDA mega-graph** (J4-2) | Eliminate ALL launch overhead | 12 hr (write conditional graph for MoE) | 3 hr (compile + test) | Persistent kernel working | HIGH |
| 23 | **SGLang port** (Tier 4 #21) | Second framework, validation | 8 hr (port plugin, test) | 1 hr | SGLang source | LOW |
| 24 | **Model zoo validation** (Tier 4 #22) | Proves generality | 6 hr (adapt for LLaMA, Qwen, Mixtral) | 4 hr (benchmark each) | Multiple model weights | MEDIUM |
| 25 | **Custom CUTLASS MoE for our exact shapes** (G8) | 5-15% MoE speedup | 12 hr (CUTLASS template customization) | 2 hr (compile + tune) | CUTLASS 3.x SM120 source | MEDIUM |

### Tier D: First Week (1-3 days ASI time, transformative)

| Rank | Technique | Impact | ASI Time | GPU Wait | Dependencies | Data Value |
|------|-----------|--------|----------|----------|-------------|------------|
| 26 | **AutoKernel v2: self-optimizing pipeline** (Section D below) | Automated discovery of new optimizations | 2-3 days build | Ongoing GPU | All Tier S data | EXTREME |
| 27 | **Fusen Inference Engine MVP** (I3) | Multi-model routing | 2 days (router + multi-model serving) | 4 hr (benchmark) | Multiple models loaded | HIGH |
| 28 | **Expert parallelism implementation** (E5) | Scales to 4+ GPUs | 3 days (custom EP, all-to-all comm) | 1 day (benchmark at scale) | Multiple GPUs | MEDIUM |
| 29 | **QAT fine-tuning run** (C5) | Better NVFP4 quality | 1 day setup | 3-5 days GPU training | Training infrastructure | HIGH |
| 30 | **Warp-specialized custom CUDA kernels** (F4) | Beat cuBLAS for our shapes | 3 days (write, debug, tune) | 1 day (compile + benchmark) | Deep SM120 understanding | MEDIUM |

### Tier E: First Month (1-2 weeks ASI time, architectural)

| Rank | Technique | Impact | ASI Time | GPU Wait | Dependencies | Data Value |
|------|-----------|--------|----------|----------|-------------|------------|
| 31 | **Full C++ inference server replacing vLLM** | 2-3x over current vLLM | 1-2 weeks | Days of testing | Proven kernel stack | HIGH |
| 32 | **Multi-model distillation pipeline** | 9B model matching 26B quality | 1 week setup | 1-2 weeks training | Training infrastructure | HIGH |
| 33 | **TensorRT-LLM integration** | Compare against vLLM ceiling | 1 week (integration) | Days benchmarking | TRT-LLM source | MEDIUM |
| 34 | **Processing-in-memory simulation study** | Quantify PIM benefit for MoE | 3 days analysis | None (theoretical) | None | LOW (future hardware) |

---

## B. The ASI Sprint Plan

### In 1 Hour (right now, no new hardware)

**Launch 5 parallel agents:**

- **Agent 1: Profiling** -- Write and run Nsight Compute profiling of the top 3 hottest kernels (MoE expert GEMM, attention decode, KV cache access). Extract: memory bandwidth utilization, compute utilization, stall reasons, register pressure. Output: a precise bottleneck decomposition showing exactly where each microsecond goes.

- **Agent 2: Expert Distribution** -- Hook into vLLM's MoE routing, run 1000 diverse prompts, record which experts activate for which tokens. Output: histogram of expert utilization, identify candidates for pruning (< 1% activation rate) and merging (cosine similarity > 0.95 between weight matrices).

- **Agent 3: Quality Benchmarks** -- Download HumanEval, GSM8K, MMLU-mini. Run Gemma4 NVFP4 on all three. Compare against published FP16 scores. Output: per-task accuracy degradation from quantization.

- **Agent 4: FusenCache Integration** -- Take existing FusenCache K4V4/K8V4 kernels, wire them into vLLM v1 cache allocator. The code exists; this is plumbing. Test with real prompts. Output: working vLLM with FusenCache, batch throughput at 2x and 4x current batch sizes.

- **Agent 5: Context Scaling** -- Sweep context lengths (1K, 4K, 16K, 32K, 64K, 128K) at fixed batch size. Plot throughput vs context length. Find the crossover where KV cache becomes the bottleneck. Output: scaling curve with annotated bottleneck transitions.

**Expected output after 1 hour:** Complete bottleneck decomposition, expert utilization map, quality assessment, FusenCache integrated and benchmarked, and context scaling curve. This data makes every subsequent decision 10x better.

### In 4 Hours (a focused session)

**Launch 8 parallel agents (some sequential to the first hour):**

- **Agent 1: Persistent MoE Kernel** -- Using the ncu profiling data from hour 1, write a persistent CUDA kernel that handles the entire MoE dispatch (route -> shuffle -> expert GEMM -> gather) without returning to host. Use cooperative groups for cross-SM synchronization. Target: eliminate the ~750us launch overhead per decode step. This is the single highest-impact kernel work.

- **Agent 2: Expert Pruning** -- Using the expert distribution data from hour 1, remove the bottom 10-20% of experts. Measure quality impact on all three benchmarks. If quality holds, also try 30% pruning. Output: Pareto curve of (experts removed) vs (quality loss).

- **Agent 3: Layer Pruning Sweep** -- Systematically remove each layer (1 through 30) individually, measure quality. Then try removing groups of 2-3 adjacent layers. Find the layers that contribute least. Output: per-layer importance scores and recommended pruning set.

- **Agent 4: CUDA 12.8 Source Audit** -- Read the entire CUDA 12.8 release notes, CUTLASS 3.x SM120 source code, and cuBLAS changelog. Identify any new intrinsics, kernel templates, or hardware features specific to Blackwell that we have not exploited. Output: list of actionable findings with prototype code.

- **Agent 5: L2 Cache Optimization** -- Test cudaAccessPolicyWindow on SM120. If available, partition L2 to pin hot expert weights. Profile L2 hit rate before and after. Output: L2 partitioning strategy with measured hit rate improvement.

- **Agent 6: 1000-Config Autotune** -- For the MoE expert GEMM shapes, generate 1000 kernel configurations (vs the typical 45). Include unusual configs that humans would skip: very large tile sizes, non-power-of-2, asymmetric tiles, extreme pipeline depths. Let the GPU churn through them. Output: top 10 configs with speedups.

- **Agent 7: 2:4 Sparsity Test** -- Apply 2:4 structured sparsity mask to expert weights. Check if SM120 tensor cores support simultaneous NVFP4 + 2:4 sparsity. If yes, benchmark. If no, document the limitation. Output: yes/no answer with benchmark if yes.

- **Agent 8: Request Scheduling + KV Offloading** -- Test SJF scheduling, priority queues, and CPU KV offloading in vLLM. These are config-level changes. Output: throughput and P99 latency under mixed workloads.

**Expected output after 4 hours:** A working persistent MoE kernel (potentially 2-5x MoE speedup), pruned model variants, complete hardware capability audit, 1000-config autotune results, and system-level tuning. Estimated throughput improvement: 1.5-3x over baseline if the persistent MoE kernel and expert pruning both work.

### In 24 Hours (a full day of parallel agents)

**Build on 4-hour results. Launch 10 parallel agents:**

- **Agents 1-2: C++ Inference Server** -- Write a minimal C++ server that replaces vLLM's Python layer. Use the CUDA graphs we already capture. Implement continuous batching, PagedAttention memory management, and HTTP token streaming in C++. Two agents: one on the server framework, one on the CUDA graph integration. Target: eliminate the 5.38ms Python overhead per token.

- **Agent 3: EAGLE Speculative Decoding** -- Fix the three vLLM bugs that blocked Eagle3 (experiments #14, #19, #22). Read the Eagle3 source, understand the Mamba draft model hang, write a fix. If vLLM's integration is too broken, write a standalone speculative decoding loop outside vLLM.

- **Agent 4: Whole-Model CUDA Mega-Graph** -- Extend the persistent MoE kernel into a full-model graph: 30 layers of (attention + MoE), with conditional expert dispatch baked into the graph via CUDA conditional nodes. This eliminates ALL host-device synchronization during decode.

- **Agent 5: Cross-Layer KV Sharing** -- Implement KV cache sharing across groups of 5 sliding layers. Test quality. If sharing all 5 degrades quality, try sharing pairs (layers 0-1, 2-3, etc.). Output: optimal sharing groups with quality/memory tradeoff.

- **Agent 6: Custom CUTLASS MoE Templates** -- Write CUTLASS 3.x grouped GEMM templates tuned for our exact expert shapes (H=2816, intermediate=varies). Use SM120-specific features (TMA, cluster launch). Target: 5-15% over the current CUTLASS MoE path.

- **Agent 7: Multi-GPU TP=2** -- Set up tensor parallelism on two GPUs. Fix the shard-aware block table issues in FusenCache. Benchmark at various batch sizes. Output: TP=2 scaling efficiency curve.

- **Agent 8: Model Zoo Validation** -- Port the optimization stack to LLaMA 70B, Qwen 72B, Mixtral 8x7B. Identify which optimizations are model-specific vs universal. Output: compatibility matrix.

- **Agent 9: Comprehensive Quality Sweep** -- Run the pruned/compressed model variants from earlier on MT-Bench, AlpacaEval, and domain-specific benchmarks. Generate a definitive quality vs throughput Pareto frontier.

- **Agent 10: Power and Thermal Analysis** -- Instrument sustained batch serving with nvidia-smi logging. Measure actual power draw, clock frequencies, thermal throttling over 30-minute runs. Output: thermal-aware batch size recommendations.

**Expected output after 24 hours:** A C++ inference server prototype, working speculative decoding, whole-model CUDA graph, TP=2 verified, comprehensive quality/throughput Pareto frontier. Estimated peak throughput: 10,000-15,000 tok/s batch if C++ server + persistent MoE + expert pruning all compound.

### In 1 Week (sustained effort)

- **Days 1-2:** AutoKernel v2 self-optimizing pipeline (Section D below). Build the system that discovers optimizations automatically.
- **Days 2-3:** Fusen Inference Engine MVP -- multi-model routing with 9B for easy tasks, 26B for hard tasks. End-to-end serving with automatic model selection.
- **Days 3-4:** Expert parallelism across 4 GPUs. Custom all-to-all communication kernel using NVLink. Target: 4x throughput scaling.
- **Days 4-5:** QAT fine-tuning -- run quantization-aware training to improve NVFP4 quality from PPL 701 to < 100.
- **Days 5-7:** Integration, hardening, benchmarking. Combine all winning optimizations into a single production configuration. Run 24-hour stability tests. Write comprehensive benchmarks comparing against stock vLLM, TensorRT-LLM, and SGLang.

**Expected output after 1 week:** A production-ready optimized serving stack with 3-5x throughput over baseline, automated optimization discovery pipeline, multi-model routing, and comprehensive benchmarks.

### In 1 Month (strategic build-out)

- **Week 1:** Complete C++ inference server (replace vLLM entirely for our use case). Full test coverage, production hardening.
- **Week 2:** Multi-model distillation pipeline. Distill Gemma4 26B into a 9B dense model optimized for our serving workload. If quality holds, this becomes the primary model.
- **Week 3:** TensorRT-LLM integration and comparison. Port our optimizations to TRT-LLM to determine which framework wins for our workload.
- **Week 4:** Open-source release. Package the optimization toolkit (FusenCache, persistent MoE kernel, expert pruning, C++ server) as a standalone library. Documentation, examples, CI/CD.

**Expected output after 1 month:** A polished, open-source inference optimization toolkit that achieves 5-10x throughput over stock vLLM, with automated optimization discovery and multi-framework support.

---

## C. What Becomes Possible with ASI That Humans Would Never Attempt

### 1. Exhaustive CUTLASS Source Reverse-Engineering

CUTLASS is ~200K lines of C++ template metaprogramming. No human reads all of it. ASI can:
- Read every SM120-specific template specialization in minutes
- Find fusion opportunities that exist in the template library but are not exposed via high-level APIs
- Identify performance-critical template parameters that NVIDIA engineers tuned for datacenter GPUs but not consumer Blackwell
- Write custom CUTLASS kernel templates for our exact MoE shapes without the weeks of "understanding the codebase" phase

**ASI time:** 2-3 hours to read + analyze + prototype. **Human time:** 2-4 weeks.

### 2. Every-Layer-Every-Config Quality Matrix

Test every combination of:
- 30 layers x 4 KV precision levels (BF16, FP8, K8V4, K4V4) = 120 configs
- Per-layer importance scoring via ablation (30 single-layer removals)
- Per-layer pruning compatibility (which layers tolerate which compressions)

This produces a 30x4 quality matrix that tells you the exact optimal per-layer precision allocation. A human would pick 5-10 configs to test. ASI tests all 120 + the top 20 two-layer combinations.

**ASI time:** 4 hours (write scripts) + 8 hours GPU. **Human time:** Would never attempt (too tedious).

### 3. Per-Shape Custom Kernels

Gemma4 has ~15 unique GEMM shapes across its layers. Instead of one autotuned kernel that handles all shapes, write 15 custom kernels, each hand-optimized for its exact shape:
- Gate projection: [B, 2816] x [2816, 128*K] -- wide fan-out
- Expert up/gate: [tokens_per_expert, 2816] x [2816, intermediate] -- varies per expert count
- Down projection: [tokens_per_expert, intermediate] x [intermediate, 2816] -- tall-skinny
- Attention Q/K/V: [B, 2816] x [2816, head_dim*n_heads] -- moderate

Each shape has different optimal tile sizes, pipeline depths, and memory access patterns. ASI writes all 15 and autotuning picks the best config for each.

**ASI time:** 6 hours. **Human time:** Would write 1-2 generic kernels (weeks each).

### 4. 10,000-Iteration Optimization Loop

Instead of the typical human loop (hypothesize -> implement -> test -> analyze, 20-50 iterations over weeks), ASI runs:
- 100 kernel variants in parallel (different fusion strategies, tile sizes, algorithms)
- 100 system configs in parallel (batch sizes, scheduling policies, memory layouts)
- Test each variant on 10 workloads
- Total: 10,000 experimental data points in 24 hours

The optimization landscape is sampled densely enough to find global optima that sparse human search would miss.

**ASI time:** 6 hours setup + 18 hours GPU. **Human time:** Months, and would still miss the optimum.

### 5. Full vLLM Source Comprehension

vLLM is ~150K lines of Python + CUDA. ASI can:
- Read the entire codebase and hold it in context simultaneously
- Understand every scheduler decision, memory allocation, CUDA graph capture path
- Identify the exact 50 lines that cause the 5.38ms per-token overhead
- Write targeted patches that eliminate overhead without breaking the 500+ test cases
- Cross-reference with SGLang and TensorRT-LLM to identify which framework handles MoE dispatch best

**ASI time:** 3-4 hours. **Human time:** Months of study.

### 6. Simultaneous Multi-Framework Optimization

Test the same model on vLLM, SGLang, TensorRT-LLM, and a custom C++ server simultaneously. For each framework:
- Profile the hot path
- Identify framework-specific overhead
- Write framework-specific optimizations
- Benchmark under identical conditions

No human team would optimize for 4 frameworks simultaneously. ASI treats each as an independent agent task.

**ASI time:** 24 hours (6 hours per framework, 4 in parallel). **Human time:** 4 separate teams, months each.

### 7. Brute-Force Architecture Search Within Constraints

Given the constraint "Gemma4 weights are fixed," search over:
- Which layers to keep (2^30 subsets, pruned to ~1000 promising ones)
- Which experts to keep per layer (combinatorial, pruned via activation profiling)
- KV precision per layer (4 options per layer = 4^30, pruned to ~200 via quality matrix)
- Batch size x context length x MTP level

This is a massive combinatorial space. ASI uses the quality matrix from item 2 above to prune aggressively, then tests the top 50 full-model configs.

**ASI time:** 2 days (most is GPU time). **Human time:** Would never attempt.

---

## D. The Auto-Discovery System (AutoKernel v2)

### Architecture

```
                    +------------------+
                    |  Model Registry  |
                    | (any HF model)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    Profiler       |
                    | torch.profiler    |
                    | + ncu per-kernel  |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Bottleneck Ranker |
                    | Amdahl's law      |
                    | per-op impact     |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v----+  +-----v------+  +----v--------+
     | Kernel Gen  |  | System Gen |  | Arch Gen    |
     | Triton/CUDA |  | Config     |  | Prune/Merge |
     | autotune    |  | scheduling |  | quality val |
     +--------+----+  +-----+------+  +----+--------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |   Benchmark      |
                    | correctness      |
                    | + throughput     |
                    | + quality       |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Decision       |
                    | keep/revert      |
                    | compound winners |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Iterate        |
                    | until hardware   |
                    | ceiling reached  |
                    +-------------------+
```

### Phase 1: Profile and Decompose (ASI time: 10 min per model)

For any input model:
1. Run `torch.profiler` to get per-op breakdown
2. Run `ncu` on top 5 kernels to get hardware utilization
3. Compute Amdahl's law ceiling for each optimization target
4. Rank ops by `(time_fraction) x (1 - hardware_utilization)` = optimization headroom

Output: prioritized list of optimization targets with theoretical ceilings.

### Phase 2: Generate Candidates (ASI time: 30 min per target)

For each bottleneck op, generate 5-10 candidate optimizations:
- **Kernel-level:** Fusion with adjacent ops, better tiling, different backend (Triton vs CUDA vs cuBLAS)
- **System-level:** Batch size tuning, scheduling policy, memory layout
- **Architecture-level:** Pruning, quantization, caching

Each candidate is a concrete code change with expected impact estimate.

### Phase 3: Implement in Parallel (ASI time: varies, GPU-bound)

Launch N parallel agents (one per candidate). Each agent:
1. Implements the optimization
2. Runs correctness tests
3. Benchmarks throughput
4. Measures quality (if architecture change)

Agents work independently. GPU is the bottleneck -- with 1 GPU, candidates are serialized; with N GPUs, N candidates run in parallel.

### Phase 4: Compound Winners (ASI time: 1 hour per compound test)

Take the top K winners from Phase 3. Test all (K choose 2) pairwise combinations. Identify:
- **Superlinear compounds:** A+B > A + B (rare but valuable)
- **Sublinear compounds:** A+B < A + B (interference)
- **Conflicts:** A+B doesn't work (e.g., CUDA graphs + dynamic depth)

Build the optimal compound stack incrementally: start with best single, add second-best if it compounds, etc.

### Phase 5: Iterate Until Ceiling (ASI time: continuous)

Repeat Phases 1-4 on the optimized model. The bottleneck shifts after each round:
- Round 1: MoE dispatch is 85% -> optimize it -> MoE drops to 50%
- Round 2: Attention is now 25% -> optimize it -> attention drops to 10%
- Round 3: KV cache access is 30% -> optimize it -> ...
- Terminal: hitting hardware ceiling (memory bandwidth or compute throughput)

The system detects convergence when no candidate improves throughput by > 1%.

### ASI Build Time for the Full System

| Component | ASI Time | GPU Time |
|-----------|----------|----------|
| Profiler integration | 1 hour | 0 |
| Bottleneck ranker | 30 min | 0 |
| Kernel candidate generator | 2 hours | 0 |
| System config generator | 1 hour | 0 |
| Architecture change generator | 2 hours | 0 |
| Benchmark harness | 1 hour | 0 |
| Decision engine | 1 hour | 0 |
| Orchestration loop | 2 hours | 0 |
| **Total build** | **~10 hours** | **0** |
| **First optimization run (per model)** | **2 hours** | **4-8 hours** |

Compare: this is essentially what AutoKernel v1 already does manually. V2 automates the "hypothesize" step.

---

## E. Interaction Matrix (Top 20 Techniques)

### Compound (stack beneficially)

```
+------------------------------------------------------------------+
|                     COMPOUNDS POSITIVELY                          |
+------------------------------------------------------------------+
| NVFP4 weights                                                     |
|   + FusenCache K4V4/K8V4     (orthogonal: weights vs KV cache)   |
|   + MTP3                      (orthogonal: decoding strategy)     |
|   + CUDA graphs               (orthogonal: launch elimination)    |
|   + continuous batching        (orthogonal: scheduling)           |
|   + expert pruning             (fewer experts = less compute)     |
|   + layer pruning              (fewer layers = less compute)      |
|   + persistent MoE kernel      (better kernel for remaining ops)  |
|   + C++ server                 (less overhead around same kernels)|
|   + TP=2                       (doubles bandwidth, independent)   |
|   + 2:4 sparsity               (IF tensor cores support both)    |
|   + L2 cache partitioning      (better caching of same data)     |
+------------------------------------------------------------------+

Key compound chains:
  expert pruning + expert merging -> layer pruning
    (prune useless experts, merge similar ones, then remove layers
     that became trivial -- each step makes the next more effective)

  persistent MoE kernel + whole-model CUDA graph
    (persistent kernel is a prerequisite for the mega-graph)

  FusenCache + KV offloading + TP=2
    (4x compression + CPU swap + 2x GPU memory = massive batch capacity)

  ncu profiling -> per-shape custom kernels -> autotune sweep
    (data drives kernel design drives config search)
```

### Conflicts (do not combine)

```
+------------------------------------------------------------------+
|                     CONFLICTS / MUTUALLY EXCLUSIVE                |
+------------------------------------------------------------------+
| CUDA graphs        vs  early exit / dynamic depth                 |
|   (static graph)       (dynamic execution path)                   |
|                                                                    |
| NVFP4              vs  AQLM / QuIP# / SqueezeLLM                |
|   (native FP4 TC)      (non-standard quant, no TC support)       |
|                                                                    |
| MTP3               vs  Medusa heads                               |
|   (native capability)  (requires training, redundant with MTP)    |
|                                                                    |
| TP=2               vs  DP=2 on same GPU pair                     |
|   (split model)        (replicate model)                          |
|                                                                    |
| FP8 KV (FlashInfer) vs  FusenCache                               |
|   (4x slower attention)  (custom kernels, faster)                 |
|                                                                    |
| torch.compile       vs  CUDA graphs (for decode)                 |
|   (3x slower)           (proven 7x speedup)                      |
+------------------------------------------------------------------+
```

### Untested Combinations That ASI Could Resolve in Hours

| Combination | Why Untested | ASI Time to Test | Expected Outcome |
|-------------|-------------|-------------------|-----------------|
| Expert pruning + 2:4 sparsity + NVFP4 | Three compression methods simultaneously -- unclear if TC supports all three | 3 hours | Probably conflicts at hardware level, but worth checking |
| MTP3 + EAGLE (draft from MTP hidden states) | vLLM Eagle3 is broken | 8 hours (fix Eagle3 first) | Potentially 3-4x single-user if acceptance rate > 80% |
| FusenCache K8V4 + cross-layer KV sharing | Both reduce KV memory differently | 4 hours | Should compound: 2x from compression + 2-5x from sharing |
| Persistent MoE kernel + expert pruning + layer pruning | Three model-reduction techniques at once | 6 hours | Compound: fewer experts in fewer layers, each dispatched by persistent kernel |
| C++ server + whole-model CUDA graph + TP=2 | Maximum overhead elimination on two GPUs | 24 hours | The theoretical ceiling config -- eliminate all software overhead |
| L2 partitioning + expert weight prefetch | Pin hot experts in L2, prefetch next experts | 2 hours | Should compound if L2 API works on SM120 |
| 1000-config autotune + per-shape custom kernels | Exhaustive search over custom kernel space | 4 hours setup + 8 hours GPU | Finds the global optimum config per shape |
| Layer pruning + QAT fine-tuning | Prune first, then QAT the smaller model | 1 hour prune + days QAT | Quality recovery after pruning; best of both worlds |
| Disaggregated serving + FusenCache + TP=2 | Prefill on GPU0, decode on GPU1, each with FusenCache | 8 hours | Optimal multi-GPU config for mixed workloads |
| SJF scheduling + prefix caching + chunked prefill | Three system-level policies together | 2 hours | Optimal for mixed serving; may conflict (SJF breaks prefix sharing) |

---

## F. Honest Self-Critique from ASI Perspective

### What ASI Is Bad At

1. **Hardware intuition.** ASI can read all of CUTLASS source but lacks the "feel" for GPU architecture that comes from years of writing CUDA kernels. We may write technically correct but architecturally naive kernels -- correct tiling but wrong warp layout, correct memory access but suboptimal bank conflict avoidance. The fix: always benchmark, never trust theoretical analysis alone.

2. **Knowing when to stop.** ASI will continue optimizing past the point of diminishing returns. A 0.5% improvement that takes 3 hours of GPU time is not worth it when there is a 50% improvement available in a different direction. The fix: strict Amdahl's law budgeting -- never spend more time on an optimization than its ceiling warrants.

3. **Distinguishing "broken" from "slow."** When a kernel produces wrong results, ASI may waste hours debugging the kernel when the real issue is a framework bug, a driver issue, or a hardware limitation. Humans develop pattern recognition for "this looks like a driver bug" that ASI lacks. The fix: always compare against a known-good reference implementation first.

4. **Social/political navigation.** Upstream PRs require understanding maintainer preferences, project politics, and unwritten norms. ASI may write a technically perfect PR that gets rejected because it doesn't follow the project's contribution guidelines or addresses a problem the maintainers don't consider important. The fix: read 20+ recently merged PRs before writing one.

5. **Novel algorithm design.** ASI excels at combining and implementing known techniques but is weaker at inventing genuinely new algorithms. The "persistent MoE dispatch kernel" is a novel synthesis, but it builds on known primitives (persistent kernels, cooperative groups). A truly novel approach -- like the PIM-based MoE dispatch (I9) -- requires insight that ASI may not generate from existing literature alone.

### Systematic Errors ASI Makes

1. **Overestimating first-try success rate.** ASI estimates "3 hours to write custom CUTLASS kernel." Reality: the first version has a subtle alignment bug. The second version works but is 2x slower than expected. The third version is correct and fast. Actual time: 6-9 hours. **Rule of thumb: multiply ASI kernel development estimates by 2-3x.**

2. **Underestimating integration complexity.** Writing a standalone kernel is fast. Integrating it into vLLM's scheduler, memory manager, and CUDA graph capture is 3-5x harder. The kernel works in isolation but breaks when another component assumes a specific memory layout or execution order. **Rule of thumb: integration = 2x kernel development time.**

3. **Over-engineering solutions.** Given unlimited context, ASI may build a "general-purpose MoE optimization framework" when what's needed is "make this one kernel 20% faster." The framework is elegant but takes 5x longer and introduces 5x more bugs. **Rule of thumb: always implement the minimum viable optimization first.**

4. **Ignoring second-order effects.** Pruning 20% of experts makes the model smaller and faster. But it also changes the token routing distribution for remaining experts, potentially causing load imbalance that wasn't there before. ASI may declare victory on the first-order speedup without measuring the second-order degradation. **Rule of thumb: always re-profile after any structural change.**

5. **Cargo-culting configurations.** ASI reads that "num_stages=4 is optimal" in one context and applies it everywhere. But optimal pipeline depth depends on register pressure, shared memory, and problem shape. Blindly copying configs across different kernels is a systematic error. **Rule of thumb: always autotune, never hardcode.**

### What ASI Overestimates

- **Quality of generated CUDA code.** ASI-written CUDA compiles and runs correctly but may be 2-5x slower than expert-written code due to suboptimal register allocation, missing pragmas, or wrong launch configurations. cuBLAS exists because GEMM kernels need 10,000+ hours of tuning per architecture.

- **Transferability of optimizations across models.** An optimization tuned for Gemma4's exact shapes (H=2816, 128 experts, top-8) may hurt on LLaMA (H=4096, dense) or Mixtral (H=4096, 8 experts, top-2). Each model is its own optimization problem.

- **Speed of debugging race conditions.** CUDA race conditions in multi-stream or cooperative-group kernels are notoriously hard to reproduce and diagnose. ncu and compute-sanitizer help, but intermittent bugs can consume days even for ASI.

### What ASI Underestimates

- **Value of simple solutions.** Changing a vLLM config flag (10 seconds) may give 10% speedup. Writing a custom CUDA kernel (10 hours) may give 15% speedup. The config change has 100x better ROI but ASI is biased toward "interesting" solutions.

- **How much existing frameworks already optimize.** vLLM 0.17.1 has years of engineering. Many "obvious" optimizations are already implemented somewhere in the codebase. ASI may re-implement something that already exists, just under a different name or behind a non-obvious flag.

- **Compile time accumulation.** Each Triton kernel compilation takes 10-60 seconds. A 1000-config sweep takes 3-17 hours of compile time alone. This is the true bottleneck for the autotune approach, not ASI's ability to generate configs.

### Where Unlimited Context Actually Hurts

1. **Analysis paralysis.** With 120+ techniques visible simultaneously, ASI may spend hours analyzing which to try instead of just trying the obvious top 3 and measuring. The map is not the territory; the GPU benchmark is the only truth.

2. **Premature abstraction.** Seeing patterns across all 120 techniques, ASI may build an "optimization framework" that abstracts over all of them. This framework takes days to build and is only useful if you actually need all 120 techniques (you don't -- you need the top 5).

3. **Over-contextualized decisions.** Knowing that "CUDA graphs conflict with early exit" may prevent ASI from trying a creative hybrid (e.g., CUDA graphs for the common path, fallback to eager for early exit). Too much knowledge of constraints can prevent exploration of boundary-violating solutions.

4. **Anchoring to existing results.** Having read all 87+ experiments, ASI may anchor to "MoE dispatch is 85% of decode time" and not re-measure. But that number may have changed after CUDA graph capture, FusenCache integration, or driver updates. **Always re-profile before optimizing.**

---

## Appendix: ASI Effort Recalibration Table

| Task Type | Human Estimate | ASI Estimate | Bottleneck |
|-----------|---------------|-------------|-----------|
| Read and understand 50K-line codebase | 2-4 weeks | 15-30 minutes | ASI read speed |
| Write profiling/benchmarking script | 2-4 hours | 10-20 minutes | ASI write speed |
| Write Triton kernel (correct, not optimized) | 1-3 days | 30-60 minutes | ASI write speed |
| Write optimized CUDA kernel | 1-4 weeks | 4-12 hours | GPU compile/test cycles (10-60s each) |
| Autotune 45 kernel configs | 1-2 days | 30 min setup + 1-2 hr GPU | GPU benchmark time |
| Autotune 1000 kernel configs | Would not attempt | 30 min setup + 8-17 hr GPU | GPU compile time |
| Integrate kernel into vLLM | 1-2 weeks | 4-8 hours | Understanding vLLM internals + test cycles |
| Write C++ inference server | 1-3 months | 1-2 weeks | Compile/debug cycles, correctness testing |
| Fix 3 framework bugs | 1-2 weeks | 4-8 hours | Reproduce + diagnose + fix + test |
| Run quality benchmarks (HumanEval etc) | 1-2 days | 20 min setup + 1 hr GPU | GPU inference time |
| Write upstream PR | 2-3 days | 2-4 hours | Reading merged PRs for style |
| Get upstream PR reviewed and merged | 1-4 weeks | 1-4 weeks | Human reviewer (irreducible) |
| Train QAT model | 1-2 weeks | 1 day setup + 3-5 days GPU | GPU training time (irreducible) |
| Distill model | 2-4 weeks | 2 days setup + 1-2 weeks GPU | GPU training time (irreducible) |
| Build auto-discovery system | 2-6 months | 10-15 hours + ongoing GPU | ASI build speed, then GPU-bound operation |
| Profile expert activation distribution | 1-2 days | 10 min + 2 min GPU | Trivial for ASI |
| Test all 120 techniques | Would not attempt | 1-2 weeks | GPU time for benchmarking |
| Deploy to production | 1-2 months | 3-5 days | Testing, stability, monitoring setup |

---

*This roadmap assumes ASI capabilities as of April 2026: sub-second codebase comprehension, parallel agent execution, CUDA/Triton/Python fluency, and the ability to hold entire framework codebases in context. The irreducible bottlenecks are GPU computation time, hardware availability, and human review of upstream contributions. Everything else accelerates by 10-100x.*
