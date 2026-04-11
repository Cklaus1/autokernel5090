# Next 20 High-Impact Research Areas

**Date:** 2026-04-09
**Context:** Gemma4 26B-A4B NVFP4 on RTX 5090 (32GB), PRO 6000 2x96GB arriving
**Current peak:** 6,685 tok/s batch (FusenCache eager), 186 tok/s single-user (MTP3)
**Bottleneck:** Attention (FA2) = 63%, MoE grouped GEMM = 30%, everything else = 7%

---

## Scoring Formula

Each area is scored by: `(expected_impact x probability_of_success) / ASI_effort`

- **Expected impact:** 1-10 scale (10 = 2x+ system-level improvement)
- **Probability of success:** 0.0-1.0
- **ASI effort:** hours of wall-clock time (including GPU waits)

---

## What's Ruled Out (not repeated here)

CUDA graphs (done), disable inductor (done), FusenCache (done), fused RMSNorm+FP4 (done, <1% of decode), expert pruning (all essential), expert caching (working set > L2), cross-layer KV sharing (zero similarity), stream parallelism (20x slower), FP8 KV (4x slower), XQA SM120 (slower than FA2), L2 persistence (no benefit), MoE shuffle+quant fusion (done, 2.3%), two-tier brain (built), parallel solver (built), AutoKernel v2 (built).

---

## Rank 1: Prompt Caching / Prefix Sharing (Application-Level)

**What:** Cache KV states for common prefixes (system prompts, few-shot examples, tool descriptions) across requests. Most LLM applications send the same 500-2000 token system prompt with every request. vLLM has automatic prefix caching (`--enable-prefix-caching`) but it interacts with FusenCache and CUDA graphs in untested ways.

**Why high-impact:** System prompts are 500-2000 tokens. At 240KB/token (BF16 KV), that's 120-480 MB of redundant KV computation and storage per new request. With 100 concurrent users sharing the same system prompt, this saves 12-48 GB of KV memory and eliminates prefill entirely for the shared portion. This directly multiplies the effective KV capacity that FusenCache already optimizes. For real deployments, 80-95% of requests share a system prompt prefix.

**Different from what we tried:** We optimized KV compression (how much each token costs). This optimizes KV reuse (how many tokens need to be computed at all). Orthogonal and multiplicative.

**ASI effort:** 2-4 hours (enable flag, test interactions with FusenCache, benchmark)
**Dependencies:** Working vLLM serving instance
**Risk:** Low. Feature exists in vLLM, just needs validation with our stack.

**Score:** (8 x 0.85) / 3 = **2.27**

---

## Rank 2: Continuous Batching Tuning / Chunked Prefill (System-Level)

**What:** Optimize vLLM's scheduler parameters: `max_num_batched_tokens`, `max_num_seqs`, chunked prefill size, scheduling policy (FCFS vs SJF). These knobs directly control how decode and prefill workloads share GPU resources, but we have never systematically tuned them for our MoE model.

**Why high-impact:** Our profiling shows 63% attention + 30% MoE, both bandwidth-bound. But the scheduler determines WHEN and HOW batches are formed. A poorly tuned scheduler can leave the GPU underutilized between batch transitions or stall decode batches with long prefills. Chunked prefill (breaking long prefills into 512-token chunks interleaved with decode) can reduce P99 TTFT by 3-5x under mixed workloads. This is purely config, zero code.

**Different from what we tried:** We optimized what happens inside a decode step. This optimizes when and how decode steps are scheduled. We never swept scheduler parameters.

**ASI effort:** 2-3 hours (parameter sweep, measure throughput + latency at each config)
**Dependencies:** Running vLLM instance
**Risk:** Very low. Config changes only.

**Score:** (7 x 0.90) / 2.5 = **2.52**

---

## Rank 3: Semantic KV Cache Eviction (Algorithm-Level)

**What:** Instead of FIFO/LRU eviction when KV cache fills up, evict tokens whose attention weights are consistently near-zero. Recent research (H2O, Scissorhands, SnapKV, PyramidKV) shows 80-95% of attention mass concentrates on 10-20% of tokens ("heavy hitters"). Evicting the rest preserves quality while dramatically extending effective context.

**Why high-impact:** Our data (Discovery #8) shows throughput drops 3.2x from ctx=128 to ctx=3840 due to KV pressure. FusenCache compresses each token 4x, but semantic eviction could reduce the NUMBER of tokens by 5-10x. Combined with FusenCache, that's 20-40x effective KV compression. At 128K context, this is the difference between 4 concurrent users and 40+.

**Different from what we tried:** FusenCache reduces bytes-per-token. Eviction reduces tokens-kept. They compose multiplicatively. We never explored eviction because we focused on compression.

**ASI effort:** 8-16 hours (implement attention-score-based eviction policy in FusenCache plugin, test quality on real prompts)
**Dependencies:** FusenCache vLLM plugin (done), quality evaluation harness
**Risk:** Medium. Quality degradation if eviction is too aggressive; needs careful threshold tuning per model.

**Score:** (9 x 0.60) / 12 = **0.45**

---

## Rank 4: Model Distillation to Dense 9B (Model-Level)

**What:** Distill Gemma4 26B MoE into a dense 9B model that matches 80-90% of quality. MoE models are inherently inefficient for single-user inference because they load 128 expert weight matrices but only use 8. A dense 9B model reads ALL weights every forward pass but has 3x fewer total weights, making it bandwidth-optimal.

**Why high-impact:** The 26B MoE activates ~2.1B parameters per token but loads expert routing tables for 128 experts. A 9B dense model reads 9B parameters per token — more compute but less wasted bandwidth. At single-user: bandwidth-bound decode would be ~3x faster. Combined with the Fusen Inference Engine (route easy queries to 9B, hard to 26B), this handles 70-80% of traffic at 3x speed.

**Different from what we tried:** We tried making the existing model smaller (pruning). Distillation creates a fundamentally different, purpose-built model. Pruning failed because all experts are essential; distillation can capture the full model's knowledge in a smaller architecture.

**ASI effort:** 3-5 days (dataset generation, training loop, quality eval)
**Dependencies:** GPU hours for training (PRO 6000 can do this), calibration dataset
**Risk:** Medium-high. Quality gap may be unacceptable for some tasks. Training cost is real.

**Score:** (9 x 0.50) / 80 = **0.056**

---

## Rank 5: Request-Level Routing / Multi-Model Serving (Application-Level)

**What:** Route incoming requests to different model configurations based on predicted difficulty. Easy requests (short, simple) go to a fast config (no MTP, smaller batch priority). Hard requests (long context, complex reasoning) go to MTP3 config or even the full 26B with more compute budget. The Fusen Inference Engine plan exists but is not implemented.

**Why high-impact:** Our data shows MTP3 adds 54% single-user speed but reduces batch throughput (4,967 vs 6,615 tok/s). The optimal config depends on the request. A router that classifies requests and sends them to the right config can serve 2x more total requests by not wasting MTP compute on easy queries.

**Different from what we tried:** We optimized a single config for peak throughput. This serves different configs simultaneously and routes intelligently. The two-tier brain architecture was about expert-level fallback; this is about request-level routing across model configs.

**ASI effort:** 8-16 hours (classifier, multi-instance vLLM, load balancer)
**Dependencies:** Multiple vLLM instances, request classifier
**Risk:** Medium. Routing accuracy determines value; misrouting wastes resources.

**Score:** (7 x 0.55) / 12 = **0.32**

---

## Rank 6: Quantized Attention with Custom Dequant (Hardware-Level)

**What:** FA2 is 63% of decode and reads BF16 KV. FP8 KV is 4x slower because FlashInfer's FP8 attention kernel is poorly optimized for Gemma4's head dims (256/512). Write a custom attention wrapper that dequantizes FP8 KV to BF16 in shared memory BEFORE the attention computation, then calls the fast BF16 FA2 kernel. This separates "store compressed" from "compute in BF16."

**Why high-impact:** This would give 2x KV compression (FP8 storage) with the speed of BF16 FA2 (our fastest attention path). The dequant cost in shared memory is ~0.1ms per layer vs the 4x slowdown of native FP8 attention. If successful, it cuts attention's KV bandwidth in half — that's reducing 63% of decode time by up to 50% = ~30% system speedup.

**Different from what we tried:** We tried FP8 KV through FlashInfer's native path (Discovery #4: 4x slower). This bypasses FlashInfer's FP8 kernel entirely by dequanting to BF16 before calling the fast BF16 path. Different execution strategy for the same data format.

**ASI effort:** 8-16 hours (custom kernel, integration with vLLM attention backend)
**Dependencies:** Understanding of FlashInfer's attention dispatch, CUDA kernel dev
**Risk:** Medium. Shared memory bandwidth limits may prevent full benefit. The dequant kernel adds latency that may not be offset by bandwidth savings.

**Score:** (8 x 0.45) / 12 = **0.30**

---

## Rank 7: Output Token Streaming / Early Termination (Application-Level)

**What:** Stream tokens to the user as they're generated (SSE), and implement early termination when the user navigates away or the output exceeds what's needed. In batch serving benchmarks we measure throughput, but real applications often need only the first 50-200 tokens (a summary, a code snippet, a yes/no answer). Requests that complete early free their KV slots immediately.

**Why high-impact:** In real-world serving, 40-60% of generated tokens are never read by the user (they got their answer and moved on). Early termination reclaims KV slots and compute for other requests. At C=256, freeing 40% of slots means the effective capacity is 40% higher — equivalent to a free memory upgrade. This is one of the few optimizations that improves user experience AND throughput simultaneously.

**Different from what we tried:** All our optimizations assumed every request runs to max_tokens. This optimizes the request lifecycle itself. No kernel work needed.

**ASI effort:** 4-8 hours (implement token streaming with SSE in vLLM OpenAI endpoint, add client-disconnect detection, measure KV reclamation rate)
**Dependencies:** Application-level client that supports streaming
**Risk:** Low. vLLM already supports streaming; this is about measuring and optimizing the KV reclamation behavior.

**Score:** (6 x 0.80) / 6 = **0.80**

---

## Rank 8: Adaptive Batch Size Controller (System-Level)

**What:** Dynamically adjust batch size based on real-time GPU utilization, KV pressure, and latency SLOs. Instead of static max_num_seqs, use a PID controller that increases batch when GPU is underutilized and decreases when latency exceeds targets. Our profiling showed throughput peaks at C=256 then drops — an adaptive controller would find and maintain the optimal operating point automatically.

**Why high-impact:** Our sweep data shows throughput varies 2x between suboptimal and optimal batch sizes (C=192: 4,915 vs C=256: 6,615 tok/s). Under real workloads, the optimal batch size shifts constantly as requests arrive and complete. A static config either underutilizes the GPU or overshoots. An adaptive controller maintains peak throughput at all times.

**Different from what we tried:** We found the optimal static batch size. This makes it dynamic. vLLM has continuous batching but no adaptive batch sizing based on real-time metrics.

**ASI effort:** 6-10 hours (implement controller, integrate with vLLM scheduler hooks, tune PID parameters)
**Dependencies:** Metrics collection from vLLM internals
**Risk:** Low-medium. PID tuning can be tricky; oscillation possible.

**Score:** (6 x 0.65) / 8 = **0.49**

---

## Rank 9: 2:4 Structured Sparsity on Expert Weights (Model-Level)

**What:** Apply 2:4 fine-grained structured sparsity to MoE expert weight matrices. SM120 Blackwell tensor cores have hardware support for 2:4 sparsity, which skips half the multiply-accumulate operations. For NVFP4 weights, this would double the effective tensor core throughput on the 30% of decode time spent in MoE GEMMs.

**Why high-impact:** MoE GEMMs are 30% of decode, bandwidth-bound at small batch sizes. At higher batch (where we serve), they become more compute-bound. 2:4 sparsity on compute-bound GEMMs gives up to 2x speedup on that 30% = 15% system speedup. The weight reduction (50% zeros) also halves bandwidth for the bandwidth-bound regime.

**Different from what we tried:** We tried reducing the NUMBER of experts (pruning — failed, all essential). This reduces the DENSITY of each expert's weights while keeping all 128 experts. Completely orthogonal approach.

**ASI effort:** 8-16 hours (apply sparsity masks, verify SM120 supports FP4+sparsity combo, benchmark, check quality)
**Dependencies:** CUTLASS 3.x SM120 sparse tensor core support (needs verification)
**Risk:** Medium. FP4 + 2:4 sparsity combo may not be supported in hardware. Quality degradation from 50% weight removal per expert needs validation.

**Score:** (7 x 0.40) / 12 = **0.23**

---

## Rank 10: Paged Attention Block Size Optimization (System-Level)

**What:** vLLM uses 16-token page blocks for KV cache management. With FusenCache's 4x compression, smaller blocks (4 or 8 tokens) would reduce internal fragmentation — each partially-filled block wastes less memory. Conversely, larger blocks (32 or 64) reduce page table overhead. The optimal block size for compressed KV may be very different from BF16 KV.

**Why high-impact:** Internal fragmentation in paged attention typically wastes 5-15% of KV memory. With FusenCache, the absolute waste per block is 4x smaller, but the relative waste depends on sequence length distribution. For short sequences (< 64 tokens), fragmentation can waste 25%+ of allocated KV. Optimizing block size is free throughput.

**Different from what we tried:** We compressed the KV data itself. This optimizes the container (page structure) that holds the compressed data. Never explored.

**ASI effort:** 4-8 hours (modify vLLM block size, benchmark fragmentation, test different sizes)
**Dependencies:** Understanding of vLLM's block allocator
**Risk:** Low. Block size is a configuration parameter in vLLM.

**Score:** (5 x 0.70) / 6 = **0.58**

---

## Rank 11: Disaggregated Prefill/Decode on PRO 6000 (System-Level)

**What:** Run prefill on GPU 0 and decode on GPU 1. Prefill is compute-bound; decode is bandwidth-bound. Separating them prevents interference: decode batches are never stalled by long prefills, and prefill can run at maximum throughput without contending for KV bandwidth.

**Why high-impact:** Under mixed workloads (some users starting, some mid-conversation), collocated serving sees 3-5x P99 TTFT spikes when a long prefill interrupts the decode batch. Disaggregation eliminates this entirely. The design doc is written; vLLM has native support via `kv_role` and `P2pNcclConnector`.

**Different from what we tried:** We used TP=2 (both GPUs do everything together). This is PP-style (each GPU specializes). TP=2 adds NCCL overhead to every step; disaggregation adds NCCL cost only at the prefill-to-decode handoff.

**ASI effort:** 4-8 hours (configure two vLLM instances per the existing design doc, benchmark)
**Dependencies:** PRO 6000 hardware (arriving next week)
**Risk:** Low. Design doc written, vLLM supports it natively.

**Score:** (7 x 0.75) / 6 = **0.88**

---

## Rank 12: Embedding/Vocabulary Compression (Model-Level)

**What:** Gemma4 has a 262,144-token vocabulary — one of the largest. The embedding and LM head matrices are 262144 x 2816 = ~1.4 GB each in BF16. Vocabulary pruning (remove unused tokens for the target language/domain) or shared embedding factorization can reduce this. For serving in English-only, 60-70% of the vocabulary (CJK, rare scripts) is never used.

**Why high-impact:** The LM head (logit projection) runs every decode step: it's a matrix multiply of [B, 2816] x [2816, 262144]. For B=256, that's 184 GFLOPs — not trivial. Reducing vocab to 100K would save 60% of that computation and ~1.7 GB of VRAM for more KV cache. The softmax over 262K logits is also expensive.

**Different from what we tried:** We focused on MoE experts and attention. The vocabulary layer has been completely ignored. It's a constant overhead per step that scales with batch.

**ASI effort:** 8-16 hours (analyze token distribution, prune vocab, re-index, test)
**Dependencies:** Token frequency analysis on target workload
**Risk:** Medium. Pruned vocab cannot generate pruned tokens — domain restriction is permanent.

**Score:** (5 x 0.55) / 12 = **0.23**

---

## Rank 13: Compiled FusenCache Kernels (.cubin / AOT) (Ecosystem-Level)

**What:** Pre-compile FusenCache Triton kernels to .cubin files for SM120, eliminating JIT compilation overhead. This is the blocker for FusenCache + CUDA graphs: the Triton JIT recompiles each step because the kernel lives outside the CUDA graph capture scope. A pre-compiled .cubin registered as a CUDA function can be captured in the graph.

**Why high-impact:** FusenCache eager = 6,685 tok/s. BF16+CUDA graphs = 6,615 tok/s. FusenCache+CUDA graphs (if it worked) should be ~11,600 tok/s (1.74x from graphs applied to the FusenCache path). This is the single largest known throughput gain remaining. The projections in EXPERIMENT_DISCOVERIES.md show this as the first step to 15,000+ tok/s.

**Different from what we tried:** We tried FusenCache + CUDA graphs and hit JIT recompilation (Discovery #16). The fix is AOT compilation. The approach is different: compile once, register as static kernel, capture in graph.

**ASI effort:** 16-40 hours (Triton AOT compilation pipeline, .cubin extraction, CUDA driver API registration, graph capture testing)
**Dependencies:** Triton AOT compilation support for SM120
**Risk:** Medium-high. Triton's AOT path is not well-documented for custom kernels. May need to write raw CUDA equivalents.

**Score:** (10 x 0.40) / 28 = **0.14**

---

## Rank 14: Workload-Aware Power/Clock Management (Hardware-Level)

**What:** The RTX 5090 runs at 575W TDP, but bandwidth-bound workloads (63% attention + 30% MoE during decode) don't need full clock speed. Underclocking the GPU core while maintaining memory clock could reduce power by 30-40% with <5% throughput loss. Conversely, during prefill (compute-bound), run at full clocks.

**Why high-impact:** For a 2-GPU workstation running 24/7, power cost is significant. More importantly, reduced thermal output means sustained higher memory clocks — GDDR7 is thermally sensitive. Running decode at 80% core clock but 100% memory clock could actually INCREASE sustained bandwidth-bound throughput by avoiding thermal throttling during long runs.

**Different from what we tried:** We never profiled power or thermal behavior. All benchmarks were short bursts. Real serving is 24/7 sustained load.

**ASI effort:** 4-8 hours (nvidia-smi clock management, sustained benchmark, thermal monitoring)
**Dependencies:** nvidia-smi clock controls, sustained workload generator
**Risk:** Low. Clock changes are reversible.

**Score:** (4 x 0.70) / 6 = **0.47**

---

## Rank 15: Token Compression / Prompt Optimization (Data-Level)

**What:** Compress input prompts before they reach the model. Techniques include: LLMLingua-style prompt compression (remove redundant tokens while preserving meaning), context distillation (summarize long contexts into shorter ones), and structured prompt templates that minimize token count. A 2000-token prompt compressed to 500 tokens is 4x faster to prefill and uses 4x less KV.

**Why high-impact:** Prefill time scales linearly with prompt length. KV cache usage scales linearly with context. For applications with long contexts (RAG with 10+ documents, multi-turn conversations), prompt compression of 2-4x directly translates to 2-4x more concurrent users and faster TTFT. Combined with FusenCache's 4x KV compression, that's 8-16x effective compression.

**Different from what we tried:** We compressed KV representations. This compresses the input before it becomes KV. Operates at a completely different layer (application vs kernel).

**ASI effort:** 8-16 hours (implement prompt compression pipeline, quality eval, integration)
**Dependencies:** Prompt compression model or algorithm, quality benchmarks
**Risk:** Medium. Aggressive compression loses information; quality trade-off is workload-dependent.

**Score:** (6 x 0.55) / 12 = **0.28**

---

## Rank 16: Continuous KV Cache Defragmentation (System-Level)

**What:** As requests complete and new ones start, KV cache pages become fragmented — free pages are scattered across physical memory. This causes poor memory access patterns for attention, which reads KV sequentially within a request but jumps across physical pages. Periodic defragmentation (compacting live pages) improves memory access locality.

**Why high-impact:** With 480K+ tokens of KV on PRO 6000, fragmentation is real. Paged attention already handles non-contiguous pages, but the page table indirection adds overhead to every attention operation. Under high churn (requests constantly starting/ending), fragmentation can degrade attention bandwidth by 10-20% vs fresh allocation.

**Different from what we tried:** We optimized the KV data format. This optimizes the physical memory layout of KV pages. Never explored.

**ASI effort:** 12-24 hours (implement background defrag, measure fragmentation rate, benchmark attention throughput before/after)
**Dependencies:** vLLM block allocator internals
**Risk:** Medium. Defrag requires copying KV data, which competes with serving. Needs careful scheduling.

**Score:** (5 x 0.40) / 18 = **0.11**

---

## Rank 17: GGUF/GGML-Style CPU Offloading for Cold Layers (Hardware-Level)

**What:** Gemma4 has 30 layers. During decode, each layer is used once per step. With 192 GB total VRAM on PRO 6000, we have headroom. But for even larger models or more aggressive KV allocation, offloading "cold" layers (those with lowest residual scaling — Discovery #12: layers 2, 4, 8 have scalars < 0.15) to CPU and pipelining them with GPU execution could free 2-3 GB of VRAM per offloaded layer for more KV cache.

**Why high-impact:** Each MoE layer is ~570 MB. Offloading 3 layers = 1.7 GB freed = ~7,000 more KV tokens. The pipelining overhead is small if the CPU layer runs while GPU processes the next layer. Unlike layer pruning (which we estimated at 10% speedup but with quality risk), offloading preserves ALL quality while trading latency for capacity.

**Different from what we tried:** Layer pruning removes layers permanently (quality risk). This keeps layers but runs some on CPU (latency trade-off, no quality loss). Fundamentally different: lossless vs lossy.

**ASI effort:** 16-24 hours (implement CPU offload with async transfer, pipeline scheduling)
**Dependencies:** Sufficient CPU memory bandwidth, async CUDA memcpy
**Risk:** Medium. CPU execution is ~100x slower than GPU; pipelining must perfectly overlap to avoid bubbles.

**Score:** (4 x 0.35) / 20 = **0.07**

---

## Rank 18: Async Output Detokenization (System-Level)

**What:** vLLM detokenizes output tokens synchronously on the CPU, blocking the next decode step. For large batches (B=256), detokenizing 256 tokens through SentencePiece with a 262K vocabulary takes measurable time. Moving detokenization to a separate thread/process and using a ring buffer to decouple it from the GPU pipeline would reduce end-to-end latency.

**Why high-impact:** At 6,685 tok/s (B=256), the GPU generates a token every 38us. SentencePiece detokenization for 262K vocab takes ~2-5us per token. For 256 tokens: 500-1280us per step. That's 3-8% of the step time. Async detokenization eliminates this from the critical path.

**Different from what we tried:** We optimized GPU execution. This optimizes CPU-side post-processing that gates GPU progress. We found that Python overhead is zero (Discovery: wall=gpu=8.24ms), but detokenization was not measured separately.

**ASI effort:** 4-8 hours (profile detokenization cost, implement async pipeline)
**Dependencies:** vLLM detokenizer internals
**Risk:** Low. If detokenization is already negligible, no harm done.

**Score:** (4 x 0.50) / 6 = **0.33**

---

## Rank 19: Custom CUTLASS Grouped GEMM for Exact MoE Shapes (Hardware-Level)

**What:** vLLM uses CUTLASS 3.x grouped GEMM for MoE, but with generic tile sizes tuned for datacenter GPUs (H100/B100). Gemma4's expert shapes (M=small/variable, N=704, K=2816 in NVFP4) are unusual — very tall-and-skinny. Custom CUTLASS templates with tile sizes optimized for these exact shapes on SM120 (different SM count, different L2, different memory hierarchy than datacenter Blackwell) could improve MoE throughput.

**Why high-impact:** MoE GEMMs are 30% of decode. Even 10-15% improvement on that 30% = 3-5% system improvement. At 6,685 tok/s baseline, that's 200-330 extra tok/s. The real value is understanding the hardware: this investigation reveals whether we're near the bandwidth ceiling or leaving compute on the table.

**Different from what we tried:** We profiled MoE and found it's 6 kernels (Discovery #2). But we never tuned the CUTLASS template parameters for our specific shapes. vLLM uses NVIDIA's defaults.

**ASI effort:** 16-40 hours (read CUTLASS SM120 source, customize templates, benchmark)
**Dependencies:** CUTLASS 3.x source with SM120 support
**Risk:** Medium. May find we're already at the hardware bandwidth ceiling and CUTLASS can't do better.

**Score:** (5 x 0.35) / 28 = **0.063**

---

## Rank 20: Inference-Time Compute Scaling / Best-of-N (Algorithm-Level)

**What:** Instead of generating one response, generate N responses in parallel and select the best one using a lightweight reward model or self-evaluation. This trades throughput for quality — using our massive batch capacity (6,685 tok/s) to generate 8 candidates costs the same wall-clock time as 1 candidate at 836 tok/s per candidate, but produces measurably better output.

**Why high-impact:** Recent research (DeepSeek-R1, STaR, self-play) shows inference-time compute scaling can match training-time scaling. Our batch throughput is 50x our single-user throughput — we have massive parallelism to burn. For high-value queries (code generation, reasoning, planning), generating 8 candidates and picking the best can improve pass@1 by 20-40% on HumanEval-style benchmarks, without any model changes.

**Different from what we tried:** All our work optimized tokens-per-second. This uses our high throughput to improve quality-per-query. It's the first area that uses our speed advantage as a quality lever rather than a cost lever.

**ASI effort:** 8-16 hours (implement best-of-N pipeline, reward model selection, quality benchmarks)
**Dependencies:** Reward model or self-evaluation prompt, quality benchmarks
**Risk:** Low-medium. Well-established technique, but reward model quality determines value.

**Score:** (7 x 0.65) / 12 = **0.38**

---

## Master Ranking (sorted by score)

| Rank | Area | Category | Score | Impact | P(success) | ASI Hours |
|------|------|----------|-------|--------|------------|-----------|
| 1 | Continuous batching tuning / chunked prefill | System | **2.52** | 7 | 0.90 | 2.5 |
| 2 | Prompt caching / prefix sharing | Application | **2.27** | 8 | 0.85 | 3 |
| 3 | Disaggregated prefill/decode (PRO 6000) | System | **0.88** | 7 | 0.75 | 6 |
| 4 | Output streaming / early termination | Application | **0.80** | 6 | 0.80 | 6 |
| 5 | Paged attention block size optimization | System | **0.58** | 5 | 0.70 | 6 |
| 6 | Adaptive batch size controller | System | **0.49** | 6 | 0.65 | 8 |
| 7 | Workload-aware power/clock management | Hardware | **0.47** | 4 | 0.70 | 6 |
| 8 | Semantic KV cache eviction | Algorithm | **0.45** | 9 | 0.60 | 12 |
| 9 | Inference-time compute scaling (best-of-N) | Algorithm | **0.38** | 7 | 0.65 | 12 |
| 10 | Async output detokenization | System | **0.33** | 4 | 0.50 | 6 |
| 11 | Request-level routing / multi-model | Application | **0.32** | 7 | 0.55 | 12 |
| 12 | Quantized attention with custom dequant | Hardware | **0.30** | 8 | 0.45 | 12 |
| 13 | Token compression / prompt optimization | Data | **0.28** | 6 | 0.55 | 12 |
| 14 | 2:4 structured sparsity on experts | Model | **0.23** | 7 | 0.40 | 12 |
| 15 | Embedding/vocabulary compression | Model | **0.23** | 5 | 0.55 | 12 |
| 16 | Compiled FusenCache (.cubin AOT) | Ecosystem | **0.14** | 10 | 0.40 | 28 |
| 17 | Continuous KV cache defragmentation | System | **0.11** | 5 | 0.40 | 18 |
| 18 | CPU offloading for cold layers | Hardware | **0.07** | 4 | 0.35 | 20 |
| 19 | Custom CUTLASS grouped GEMM | Hardware | **0.063** | 5 | 0.35 | 28 |
| 20 | Model distillation to dense 9B | Model | **0.056** | 9 | 0.50 | 80 |

---

## The Story These Rankings Tell

The top 5 areas are ALL system/application-level optimizations, not kernel work. This is consistent with our key insight: **the GPU kernels are already near-optimal** (FA2 for attention, CUTLASS 3.x for MoE). The remaining gains come from:

1. **How we USE the GPU** (scheduler tuning, batching, prefix caching) — ranks 1-2
2. **How we ORCHESTRATE across GPUs** (disaggregation) — rank 3
3. **How we SERVE users** (streaming, early termination) — rank 4
4. **How we MANAGE memory** (block size, eviction) — ranks 5, 8

The kernel-level work (custom CUTLASS, compiled FusenCache, structured sparsity) ranks lower because the probability of success is lower and the effort is higher. The one exception is compiled FusenCache (.cubin AOT) — it has the highest raw impact (10/10) but low probability and high effort, making it a strategic bet rather than a quick win.

### Recommended Execution Order

**Week 1 (PRO 6000 arrives):**
1. Scheduler tuning + chunked prefill (2-3 hours)
2. Prefix caching validation (2-4 hours)
3. Disaggregated prefill/decode setup (4-8 hours)
4. Output streaming + early termination (4-8 hours)
5. Block size optimization (4-8 hours)

**Week 2:**
6. Adaptive batch controller (6-10 hours)
7. Power/thermal profiling (4-8 hours)
8. Semantic KV eviction prototype (8-16 hours)
9. Best-of-N pipeline (8-16 hours)

**Week 3+:**
10. Compiled FusenCache AOT (strategic, multi-day)
11. Quantized attention custom dequant (8-16 hours)
12. 2:4 structured sparsity investigation (8-16 hours)
13. Multi-model routing (8-16 hours)

### Blind Spots We May Still Have

- **Network I/O**: All measurements assume local inference. Real deployments have HTTP overhead, TLS, JSON serialization. How much does the network stack cost per request?
- **Tokenization cost**: SentencePiece with 262K vocab — is encoding (not just decoding) a bottleneck for long prompts?
- **Multi-turn state management**: How efficiently does vLLM reuse KV across conversation turns? Is there a "conversation-aware" scheduling mode?
- **Failure modes under load**: What happens at 100% KV utilization? Does vLLM degrade gracefully or cliff?
- **Mixed-precision attention within a single head**: High-variance dims in BF16, low-variance in FP8, within the same head. Never explored at this granularity.

---

## Addendum: Novel Paths Beyond Known Techniques (Session 2)

*Added after exhausting 75 of 116 known techniques and discovering 49 findings.*

### Tier S: Quick Experiments That Could Change Everything

| # | Idea | Gate Test | ASI Effort | If Works |
|---|---|---|---|---|
| N1 | **Expert output caching** — cache gate/up/down outputs for repeated code patterns, skip expert compute on cache hit | Profile expert output similarity on 1000 code prompts | 4 hours | Skip 30-60% of MoE compute |
| N2 | **Mixed quantization per-expert** — hot experts in FP8, cold in NVFP4 (or vice versa) | Profile per-expert activation freq, test quality at mixed precision | 4 hours | Better quality on hot paths, same speed |
| N3 | **Cross-request KV prefix deduplication** — not just prefix caching, but detect shared SUFFIXES and intermediate states | Measure KV similarity across 100 coding requests | 2 hours | 30-50% KV reduction for related requests |

### Tier A: Medium Experiments with Novel Approaches

| # | Idea | Gate Test | ASI Effort | If Works |
|---|---|---|---|---|
| N4 | **Structural code generation** — generate AST skeleton first, fill tokens second | Prototype: LLM generates JSON AST → template fills code | 1 day | More correct code, fewer syntax errors |
| N5 | **Speculative EXECUTION** — run generated code in sandbox DURING generation, feed test results back to guide next tokens | Build sandbox + feedback loop prototype | 1 day | Self-correcting generation |
| N6 | **CPU expert offload with async overlap** — cold experts run on CPU cores while hot experts run on GPU | Profile 32 CPU cores × expert matmul throughput | 4 hours | Free up GPU memory, maintain throughput |
| N7 | **Router prediction across layers** — predict all 30 layers' expert routing from layer 0's router output | Measure cross-layer routing correlation | 2 hours | Skip 29 router computations |

### Tier B: Research Bets

| # | Idea | What it is | If Works |
|---|---|---|---|
| N8 | **Bidirectional function generation** | Generate function signature + return type first, body from both ends | 2x speed for function-body code |
| N9 | **Attention pattern transfer** | Similar coding questions share attention patterns — reuse across requests | Major KV savings |
| N10 | **Semantic token compression** | Common code patterns ("def __init__(self" = 1 token instead of 5) via custom tokenizer | 3-5x context efficiency |
| N11 | **Continuous self-improvement** | Model fine-tunes on its own successful generations during idle GPU time | Quality improves over time |
| N12 | **Memory-mapped expert weights** | mmap experts from NVMe, let OS page in on demand | Unlimited expert count |

### Integration with Main Roadmap

```
THIS WEEK (before PRO 6000):
  - Fix FlashInfer concurrency crash (Opus agents working)
  - Gate test N1 (expert output caching similarity)
  - Gate test N7 (router prediction correlation)

NEXT WEEK (PRO 6000):
  - DP=2 benchmark
  - FusenDiffusion gate test (4 hours)
  - Gate test N2 (mixed quantization)
  - Gate test N6 (CPU expert offload)

THIS MONTH:
  - Build top 3 novel approaches that pass gate tests
  - BCode integration Phase 1 (extract pipeline.ts)
  - Train EAGLE3/DFlash draft for Gemma4 26B

RESEARCH BACKLOG:
  - Structural code generation (N4)
  - Speculative execution (N5)
  - Semantic token compression (N10)
  - Continuous self-improvement (N11)
```
