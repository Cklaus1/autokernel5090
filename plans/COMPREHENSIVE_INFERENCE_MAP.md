# Comprehensive Inference Optimization Map

**Target system:** Gemma4 26B-A4B MoE (128 experts, top-8), NVFP4 weights, RTX 5090 (Blackwell SM120, 32GB GDDR7, 1792 GB/s BW, ~48MB L2)
**Current peak:** 6,615 tok/s throughput (batch serving), 186 tok/s single-request decode (MTP3)
**Date:** April 9, 2026

---

## How to Read This Document

Each technique has:
- **What:** 1-2 sentence description
- **Impact:** Expected effect on our specific setup
- **Status:** Tried / Not tried / Partially explored
- **Feasibility:** Hours / Days / Weeks / Months / Impossible
- **Confidence:** Proven (we measured it) / Likely (strong evidence) / Speculative (plausible theory) / Theoretical (no evidence for our setup)

At the end of each category: compounding and mutual-exclusion notes.

---

## A. Decoding Strategies

### A1. Autoregressive Decoding (Standard)

**What:** Generate one token at a time, each conditioned on all previous tokens. The baseline approach used by all transformer LLMs.

**Impact:** This is our current method. Single-request decode = 120.8 tok/s baseline, 186 tok/s with MTP3. Batch throughput = 6,615 tok/s at B=232.

**Status:** PROVEN BASELINE. Experiment #87b: 120.8 tok/s. Experiment #80: 6,329 tok/s peak batch. Experiment #83: 7,075 tok/s with FP8 KV.

**Feasibility:** N/A (current state)
**Confidence:** Proven

### A2. Speculative Decoding (Draft Model)

**What:** A small "draft" model generates K candidate tokens quickly, then the large model verifies them in one forward pass. Accepted tokens are free; rejected tokens cost one wasted position.

**Impact:** Potentially 1.5-2.5x single-user throughput if acceptance rate > 70%. For MoE models, the draft model must match the vocabulary and tokenizer exactly. The verification pass runs the full 26B model on K+1 tokens, which for MoE means K+1 expert dispatches -- expensive.

**Status:** TRIED, FAILED. Experiment #14 (Eagle3): 3 vLLM bugs blocked it (unpack, model_runner, KV page size). Experiment #19: Fixed bugs but MoE autotuner took 55+ min then hung. Experiment #22: Eagle3 loads but hangs on first generate (Mamba inference broken). Speculative decoding is impractical for MoE hybrid models on vLLM 0.17.1.

**Feasibility:** Days to weeks (requires vLLM fixes upstream or custom scheduler)
**Confidence:** Likely beneficial once infrastructure works, but MoE verification cost is high

### A3. Multi-Token Prediction (MTP)

**What:** The model itself predicts multiple future tokens using additional prediction heads trained alongside the main model. Unlike speculative decoding, no draft model is needed -- the model has native MTP capability.

**Impact:** MEASURED: +36-54% single-request decode. MTP1 = 136.3 tok/s (+13%), MTP2 = 154.9 (+28%), MTP3 = 186.1 (+54%). Batch throughput with MTP3: peak 4,967 tok/s at B=224 (lower than baseline 6,615 because MTP adds compute per step).

**Status:** PROVEN, DEPLOYED. Experiments #93, #97. MTP3 is our best single-request decode config. Required fixing FA2 `get_supported_kernel_block_sizes` to return `MultipleOf(16)` for hybrid Mamba models. MTP2+ re-runs same layer (implementation quirk) but still net positive.

**Feasibility:** Done
**Confidence:** Proven

### A4. Medusa-Style Parallel Heads

**What:** Multiple lightweight heads attached to the final hidden state, each predicting a different future position. Tree-structured verification checks all candidates in one pass.

**Impact:** Similar to MTP but requires training the Medusa heads. Gemma4 was not trained with Medusa heads, so this requires fine-tuning (days of GPU time). Expected 1.5-2x single-user if acceptance rate is good. Less practical than native MTP since Gemma4 already has MTP support.

**Status:** NOT TRIED. MTP3 already provides the benefit Medusa would give, without needing to train extra heads.

**Feasibility:** Weeks (fine-tuning required)
**Confidence:** Likely, but dominated by native MTP

### A5. EAGLE Speculative Decoding

**What:** Trains a lightweight autoregressive "draft head" on top of the target model's hidden states. The draft head sees the same features as the main model, giving high acceptance rates (70-90%).

**Impact:** Could compound with MTP -- EAGLE drafts from MTP hidden states. Potentially 2-3x single-user. But same MoE verification cost problem as A2.

**Status:** TRIED, FAILED. Experiments #14, #19, #22. Eagle3 specifically designed for MoE but vLLM integration is broken. Mamba-based Eagle3 inference hangs.

**Feasibility:** Weeks (needs vLLM Eagle3 fix or custom implementation)
**Confidence:** Likely beneficial for single-user, uncertain for batch

### A6. Lookahead Decoding

**What:** Generates multiple candidate continuations by running N-gram lookups on previously generated tokens, then verifies in parallel. No draft model needed.

**Impact:** Low for our use case. N-gram hit rates are poor for general text. Only helps for repetitive content (code boilerplate, structured output). Maybe 1.1-1.3x for specific workloads.

**Status:** NOT TRIED.

**Feasibility:** Days (algorithmic, no model changes)
**Confidence:** Speculative (workload-dependent)

### A7. Jacobi Decoding / Parallel Token Generation

**What:** Initializes all output positions with guesses, then iteratively refines them in parallel until convergence. Each iteration runs the full model on all positions simultaneously.

**Impact:** Theoretical speedup = sequence_length / iterations_to_converge. In practice, convergence is slow (10-50 iterations) and each iteration costs a full forward pass. Net effect is usually negative for autoregressive models.

**Status:** NOT TRIED.

**Feasibility:** Days to implement, but unlikely to be productive
**Confidence:** Theoretical (negative expected value for our setup)

### A8. Diffusion-Based Text Generation

**What:** Treat text generation as a denoising diffusion process -- start from noise and iteratively refine toward coherent text. All positions generated in parallel.

**Impact:** Requires a completely different model architecture. Cannot be applied to Gemma4. Would need retraining from scratch. Quality is currently below autoregressive models for general text.

**Status:** NOT TRIED.

**Feasibility:** Months (requires new model)
**Confidence:** Theoretical

### A9. Consistency Models for Text

**What:** Distill a diffusion model into a single-step generator. Promises one-shot parallel generation.

**Impact:** Same problem as A8 -- requires a fundamentally different model. Not applicable to Gemma4.

**Status:** NOT TRIED.

**Feasibility:** Months to impossible
**Confidence:** Theoretical

### A10. Non-Autoregressive Generation (NAR)

**What:** Generate all tokens in one forward pass using masked prediction (like BERT) or CTC-style decoding. Quality gap vs autoregressive is significant for open-ended generation.

**Impact:** Not applicable to Gemma4 (autoregressive architecture). Would need a different model. Current NAR models are worse at open-ended generation.

**Status:** NOT TRIED.

**Feasibility:** Impossible for Gemma4
**Confidence:** Theoretical

### A11. Blockwise Parallel Decoding

**What:** Generate tokens in blocks of K, where each block is generated autoregressively but blocks can be partially parallelized via speculative execution.

**Impact:** This is essentially what MTP does. Our MTP3 already generates 3+1 tokens per verification step.

**Status:** COVERED BY MTP (A3).

**Feasibility:** N/A
**Confidence:** Proven (via MTP)

### A12. Staged Speculative Decoding (Multiple Draft Models)

**What:** Chain multiple draft models of increasing size. Tiny model drafts, medium model verifies and extends, large model does final verification. Each stage filters bad candidates cheaply.

**Impact:** Could improve acceptance rate vs single draft, but adds latency for each stage. For MoE models, the verification pass is expensive regardless. The benefit shrinks as the draft quality improves (diminishing returns from more stages).

**Status:** NOT TRIED.

**Feasibility:** Weeks (needs custom scheduler)
**Confidence:** Speculative

### A13. REST (Retrieval-Based Speculative Decoding)

**What:** Use a retrieval datastore (e.g., previous generations, document corpus) to find likely continuations. Retrieved text is verified by the main model in one pass.

**Impact:** Domain-specific. High acceptance rates for repetitive tasks (customer support, code completion from similar files). Low rates for creative/novel text. Requires building and maintaining a retrieval index.

**Status:** NOT TRIED.

**Feasibility:** Days to weeks
**Confidence:** Speculative (domain-dependent)

### A14. Cascade Speculative Decoding

**What:** Multiple speculative candidates generated in parallel by different methods (draft model, n-gram, retrieval), merged into a tree, verified together.

**Impact:** Combines benefits of A2, A6, A13. Higher overall acceptance rate. But tree verification is more complex and the MoE verification cost multiplies with tree width.

**Status:** NOT TRIED.

**Feasibility:** Weeks
**Confidence:** Speculative

### Compounding and Exclusions (Section A)

- **Compound:** MTP + EAGLE (EAGLE drafts from MTP hidden states) -- potentially 3-4x single-user
- **Compound:** Any speculative method + continuous batching -- speculative tokens fill GPU utilization gaps
- **Mutually exclusive:** Diffusion (A8) / Consistency (A9) / NAR (A10) vs autoregressive (A1) -- different architectures
- **Mutually exclusive:** Medusa (A4) vs native MTP (A3) -- both predict future tokens, pick one
- **Key insight:** For BATCH throughput, speculative methods generally HURT because they add compute per step. They help single-user latency only. Our 6,615 tok/s batch throughput is already compute-bound.

---

## B. Attention Optimizations

### B1. FlashAttention

**What:** Fused attention kernel that tiles Q/K/V in SRAM, avoiding materialization of the full attention matrix. O(N) memory instead of O(N^2).

**Impact:** This is what we use. vLLM uses FlashAttention v2 for both prefill and decode on Gemma4.

**Status:** PROVEN, DEPLOYED.

**Feasibility:** N/A (current state)
**Confidence:** Proven

### B2. FlashAttention-3

**What:** Next-gen FlashAttention with warp specialization, FP8 accumulation, and hardware-aware tiling for Hopper/Blackwell. Uses asynchronous memory operations (TMA) and ping-pong scheduling.

**Impact:** FA3 on Blackwell could give 10-30% attention speedup. For decode (memory-bound, short sequence), the gain is modest. For prefill (compute-bound, long sequence), potentially 1.3-1.5x. Attention is ~11.5% of our decode time, so max system-level gain is ~3%.

**Status:** NOT FULLY TRIED. vLLM uses FlashInfer which has its own Blackwell-optimized path. We tested flashinfer-cutlass backend (exp #59, +7%). FA3 proper may not have SM120 support yet.

**Feasibility:** Days (if FA3 releases SM120 support)
**Confidence:** Likely for prefill, marginal for decode

### B3. PagedAttention

**What:** Store KV cache in non-contiguous pages (like virtual memory), eliminating fragmentation waste. Enables sharing KV pages across requests with common prefixes.

**Impact:** vLLM already uses PagedAttention. This is why we can serve B=232 without running out of KV cache memory.

**Status:** PROVEN, DEPLOYED (via vLLM).

**Feasibility:** N/A
**Confidence:** Proven

### B4. Ring Attention / Sequence Parallelism

**What:** Distribute long sequences across multiple GPUs, each holding a chunk of KV cache. Attention computed in a ring pattern with overlapping communication and computation.

**Impact:** Only relevant for very long sequences (>64K tokens) on multi-GPU. We're on a single RTX 5090 with max 128K context. For our batch-serving use case (short sequences, many users), this doesn't apply.

**Status:** NOT TRIED.

**Feasibility:** Days to weeks (multi-GPU required)
**Confidence:** Likely for long-context multi-GPU, irrelevant for our setup

### B5. Multi-Query / Grouped-Query Attention

**What:** MQA uses 1 KV head per layer; GQA uses a small number (Gemma4 uses 8 KV heads for 16 query heads on sliding layers, 2 KV heads for global layers). Reduces KV cache size and memory bandwidth for decode.

**Impact:** Already built into Gemma4's architecture. GQA ratio of 2:1 (sliding) and 8:1 (global) already saves significant KV memory.

**Status:** INHERENT TO MODEL.

**Feasibility:** N/A
**Confidence:** Proven

### B6. Multi-Head Latent Attention (MLA)

**What:** DeepSeek's innovation -- compress KV into a low-rank latent representation, then expand on-the-fly during attention. Dramatically reduces KV cache size (e.g., 4-8x) with minimal quality loss.

**Impact:** Would require model retraining or architectural changes. Cannot retrofit MLA into Gemma4. A future Gemma model might use MLA natively. If they did, KV cache would shrink by 4-8x, compounding with NVFP4 for massive batch sizes.

**Status:** NOT APPLICABLE (model architecture change).

**Feasibility:** Impossible without retraining
**Confidence:** Proven (works for DeepSeek), not applicable

### B7. Linear Attention Approximations (Mamba, RWKV, RetNet)

**What:** Replace softmax attention with linear-complexity alternatives. Mamba uses selective state spaces; RWKV uses linear recurrence; RetNet uses retention mechanism. All achieve O(N) inference instead of O(N^2).

**Impact:** Gemma4 already has Mamba layers in its hybrid architecture (Eagle3 uses Mamba for the draft model). Pure linear attention models are 2-3x faster for very long sequences but slightly worse quality. Cannot retrofit into Gemma4's existing attention layers.

**Status:** NOT APPLICABLE for existing layers. Eagle3's Mamba draft model was tested (exp #22) but hung.

**Feasibility:** Impossible for Gemma4
**Confidence:** Proven (in new models), not applicable

### B8. Sparse Attention (BigBird, Longformer Patterns)

**What:** Only attend to a subset of positions -- local window + global tokens + random connections. Reduces attention from O(N^2) to O(N*sqrt(N)) or O(N*W).

**Impact:** Gemma4 already uses sliding window + global hybrid (B11). Further sparsification within the window could skip low-attention tokens during decode. But the sliding window is already only 1024 tokens -- sparse attention within such a short window has minimal benefit.

**Status:** PARTIALLY INHERENT (via sliding window). Further sparsification not tried.

**Feasibility:** Days
**Confidence:** Speculative (window too short to benefit much)

### B9. KV Cache Eviction Policies (H2O, ScissorHands, SnapKV)

**What:** Dynamically evict low-importance KV entries from cache during generation. H2O keeps "Heavy Hitter" tokens; ScissorHands keeps recent + pivotal; SnapKV selects based on attention patterns during prefill.

**Impact:** For sliding window layers (1024 window), eviction is already handled by the window itself. For global attention layers (5 layers, unlimited context), eviction could help for very long sequences. At our typical serving lengths (< 2K output), there's little to evict.

**Status:** NOT TRIED.

**Feasibility:** Days
**Confidence:** Speculative (minimal impact for short-to-medium sequences)

### B10. Dynamic Sparse Attention (Learned Sparsity Masks)

**What:** Learn which attention connections to compute dynamically at inference time. A lightweight predictor determines which KV entries each query should attend to.

**Impact:** The predictor adds overhead. For short sequences (our batch-serving sweet spot), the overhead exceeds savings. Could help for long-context scenarios only.

**Status:** NOT TRIED.

**Feasibility:** Weeks (requires training predictor)
**Confidence:** Theoretical

### B11. Sliding Window + Global Hybrid

**What:** Most layers use bounded sliding-window attention (recent 1024 tokens), a few layers use full global attention (all tokens). Combines efficient local attention with long-range capability.

**Impact:** This is Gemma4's native architecture. 25 sliding layers + 5 global layers. Optimized our KV cache strategy around this: per-layer specs (k8v4 for sliding, k4v4 for global) based on quality sensitivity analysis.

**Status:** PROVEN, EXPLOITED. Per-layer KV spec selection tested in experiment #5 (+9% throughput from mixed spec).

**Feasibility:** N/A (current state)
**Confidence:** Proven

### B12. Cross-Layer KV Sharing

**What:** Reuse KV cache from one layer in subsequent layers. If adjacent layers compute similar attention patterns, sharing saves memory and bandwidth.

**Impact:** Gemma4's 25 sliding layers have similar attention patterns. Sharing KV across groups of 5 could reduce KV memory by 30-50%. But quality impact is unknown -- the model wasn't trained with shared KV. Identified as Tier 5 in future_work.md.

**Status:** NOT TRIED. Listed as research-grade in our roadmap.

**Feasibility:** Weeks (need quality validation per layer group)
**Confidence:** Speculative

### B13. Attention Sink / StreamingLLM

**What:** Keep a small number of "attention sink" tokens (first few tokens of the sequence) permanently in KV cache alongside the sliding window. Prevents quality degradation when the window moves past the initial context.

**Impact:** vLLM may already handle this for Gemma4's sliding window layers. For infinite-length generation, this prevents quality collapse. For our typical serving (bounded output length), minimal impact.

**Status:** PARTIALLY INHERENT (vLLM handles window management).

**Feasibility:** Hours
**Confidence:** Likely (for long generation), irrelevant for typical use

### B14. Hash-Based Attention (Reformer)

**What:** Use locality-sensitive hashing to find similar Q-K pairs, avoiding computing attention for dissimilar pairs. O(N log N) instead of O(N^2).

**Impact:** Only useful for very long sequences where attention computation dominates. For decode (single query against KV cache), there's no N^2 computation to avoid -- it's already linear. For prefill of very long prompts, could help, but FA2 already handles this well.

**Status:** NOT TRIED.

**Feasibility:** Days
**Confidence:** Theoretical (not useful for decode)

### B15. Differential Attention

**What:** Split attention heads into two groups computing attention in parallel, then take the difference. Suppresses noise and improves signal in attention patterns. Recently proposed by Microsoft.

**Impact:** Requires model retraining. Cannot be applied to Gemma4 without architectural changes. Future models might adopt this.

**Status:** NOT APPLICABLE.

**Feasibility:** Impossible for Gemma4
**Confidence:** Theoretical (promising for future models)

### Compounding and Exclusions (Section B)

- **Compound:** FlashAttention + PagedAttention + GQA + Sliding Window -- all deployed, all stack
- **Compound:** KV cache eviction + FusenCache compression -- evict and compress
- **Compound:** FA3 + any quantized KV format -- faster attention on smaller cache
- **Mutually exclusive:** Linear attention (B7) vs softmax attention (B1) -- architecture choice
- **Mutually exclusive:** MLA (B6) vs GQA (B5) -- model architecture choice
- **Key insight:** For batch decode, attention is only ~11.5% of compute. The 88.5% is MoE GEMM/GEMV. Attention optimizations have ceiling of ~1.3x even if attention were free.

---

## C. Quantization and Compression

### C1. Weight Quantization (NVFP4, INT4, INT8, FP8, AWQ, GPTQ, GGUF)

**What:** Reduce weight precision to lower memory footprint and increase throughput via smaller memory transfers and specialized hardware instructions.

**Impact:** We use NVFP4 (4-bit floating point with block scales). This is the most aggressive format supported by Blackwell's FP4 tensor cores.

- NVFP4: DEPLOYED. Experiment #10: 1,261 TFLOPS pure GEMM (5.71x cuBLAS FP16). Model-level: 186 tok/s MTP3 decode, 6,615 tok/s batch.
- W4A16 (INT4 weights, FP16 activations): DEPLOYED for dequant_fused_gemm. 328 TFLOPS (experiment #86).
- FP8 weight quant: Tested (experiment #49), 2x slower than FP4 block-scaled.
- AWQ/GPTQ: Not tested on Gemma4. These are INT4 schemes with calibration. Potentially better quality than naive round-to-nearest but NVFP4 uses modelopt calibration already.

**Status:** NVFP4 PROVEN, DEPLOYED. Other formats tested and inferior for our hardware.

**Feasibility:** N/A (current state)
**Confidence:** Proven

### C2. Activation Quantization (FP8, INT8, Dynamic Per-Token)

**What:** Quantize activations between layers to reduce memory bandwidth during inference. FP8 activations can use Blackwell's FP8 tensor cores.

**Impact:** NVFP4 already quantizes activations to FP4 for the GEMM operation (online quantization per forward pass). Further activation quantization (e.g., FP8 for inter-layer communication) could reduce memory traffic between MoE expert dispatch and expert compute.

- Experiment #27: max_abs quantization strategy wins over clip99.9 for all activation patterns.
- Experiment #26: Block 16 optimal for activation quantization (cos=0.9952, 12.5% overhead).

**Status:** PARTIALLY DEPLOYED (NVFP4 includes activation quant). Standalone FP8 activation quant not tested.

**Feasibility:** Days
**Confidence:** Likely (could reduce inter-expert memory traffic)

### C3. KV Cache Quantization

**What:** Quantize the key and value tensors stored in the KV cache. Reduces memory footprint and increases batch capacity.

**Impact:** EXTENSIVELY TESTED:
- FP8 KV: 2x capacity (87K tokens), but 4x slower attention on Gemma4 head dims due to FlashInfer FP8 overhead. Experiment #15, #94.
- FusenCache K4V4: 4x capacity. Built custom Triton store/decode kernels (15 experiments). 3.8x compression, 13x store speedup, 2.16x split-K improvement. Correctness validated on real model output.
- FusenCache K8V4: Mixed precision, better quality for K (cosine ~0.97 vs ~0.91 for K4).
- Per-layer KV spec: k8v4 for sliding, k4v4 for global = +9% throughput.

**Status:** PROVEN. Multiple formats tested. FusenCache is our custom solution.

**Feasibility:** Done (FusenCache), integration with vLLM v1 API pending (days)
**Confidence:** Proven

### C4. Mixed Precision (Per-Layer, Per-Module)

**What:** Use different precision for different parts of the model. Sensitive layers get higher precision; insensitive layers get lower.

**Impact:** Already doing this implicitly:
- Weights: NVFP4 everywhere (calibrated per-block)
- KV cache: Per-layer spec selection (k8v4 sliding, k4v4 global)
- Activations: BF16 between layers, FP4 for GEMM input

Could go further: use FP8 for MoE gate computation, INT8 for RMSNorm, etc. But these ops are tiny relative to expert GEMM.

**Status:** PARTIALLY DEPLOYED.

**Feasibility:** Days for further per-module optimization
**Confidence:** Likely (diminishing returns -- non-GEMM ops are <5% of runtime)

### C5. Quantization-Aware Training (QAT)

**What:** Train the model with quantization in the forward pass so weights adapt to quantization noise. Produces higher-quality quantized models than post-training quantization.

**Impact:** Would improve NVFP4 quality (currently PPL=701.4 on WikiText-2, which is degraded vs FP16). Google may release QAT checkpoints for Gemma4 in the future. Running our own QAT would need significant GPU resources.

**Status:** NOT TRIED (we use PTQ via modelopt).

**Feasibility:** Weeks (requires training infrastructure)
**Confidence:** Likely (QAT consistently beats PTQ by 1-5% on benchmarks)

### C6. Post-Training Quantization with Calibration

**What:** Use a representative calibration dataset to determine optimal quantization parameters (scales, zero points, clipping ranges).

**Impact:** This is what NVIDIA modelopt does for our NVFP4 weights. Already deployed.

**Status:** PROVEN, DEPLOYED.

**Feasibility:** N/A
**Confidence:** Proven

### C7. AQLM (Additive Quantization)

**What:** Quantize weight matrices using additive codebooks -- each weight is the sum of multiple codebook entries. Achieves extreme compression (2-3 bits effective) with better quality than scalar quantization.

**Impact:** AQLM at 2-bit would be 2x more compressed than NVFP4, but would lose hardware tensor core acceleration. The dequantization cost (codebook lookups + additions) likely exceeds the memory bandwidth savings on Blackwell.

**Status:** NOT TRIED.

**Feasibility:** Days to test, likely inferior
**Confidence:** Speculative (hardware mismatch)

### C8. QuIP# (Incoherence Processing)

**What:** Apply random rotations to make weight matrices "incoherent" (uniformly distributed), then quantize. The incoherence ensures quantization error is spread evenly. State-of-the-art quality at 2-4 bits.

**Impact:** Better quality than naive round-to-nearest at the same bit width. But requires random rotation during inference (extra compute). At 4-bit, may not meaningfully improve over NVFP4 with calibration. At 2-bit, could enable further compression.

**Status:** NOT TRIED.

**Feasibility:** Days to weeks
**Confidence:** Speculative (unclear if quality gain justifies rotation overhead)

### C9. SqueezeLLM (Non-Uniform Quantization)

**What:** Use non-uniform quantization levels (learned from weight distribution) plus sparse outlier storage. Outlier weights stored in FP16, rest in low-bit.

**Impact:** Better quality at very low bit widths (3-4 bit). But non-uniform quantization cannot use tensor cores -- requires custom dequant kernel + FP16 matmul. We proved that split dequant + cuBLAS is fast (experiment #60-61) but still slower than native FP4 tensor cores.

**Status:** NOT TRIED.

**Feasibility:** Days
**Confidence:** Speculative (likely slower than NVFP4 on Blackwell)

### C10. Microscaling (MXFP4, MXFP8)

**What:** OCP (Open Compute Platform) standard for block-scaled floating point. MXFP4 uses 4-bit floats with shared exponents per block. Blackwell SM120 may have native hardware support.

**Impact:** NVFP4 is essentially NVIDIA's version of microscaling. The question is whether MX-format-specific tensor core instructions exist on SM120 that are faster than the current `_scaled_mm` path. Worth investigating CUDA 12.8 docs.

**Status:** NOT TRIED (NVFP4 is similar in concept).

**Feasibility:** Days to investigate hardware support
**Confidence:** Speculative

### C11. 2-Bit Quantization (BitNet, 1.58-bit)

**What:** Extreme quantization where weights are {-1, 0, 1} (1.58-bit) or {-1, 1} (1-bit). Replaces multiplications with additions/subtractions.

**Impact:** Would require a model trained from scratch with ternary weights. Cannot be applied to Gemma4 post-hoc without catastrophic quality loss. Hardware support for ternary matmul on Blackwell is unknown.

**Status:** NOT TRIED.

**Feasibility:** Impossible for Gemma4 (requires retraining)
**Confidence:** Theoretical

### C12. Pruning (Structured and Unstructured)

**What:** Remove unnecessary weights. Structured pruning removes entire columns, heads, or experts. Unstructured pruning zeros individual weights (requires sparse matmul support).

**Impact:**
- **Expert pruning:** Gemma4 has 128 experts, top-8 active. If some experts are rarely selected, they can be removed. Could reduce model size by 10-30% with minimal quality impact. Requires profiling expert usage distributions.
- **Head pruning:** Remove redundant attention heads. Gemma4 has 16 query heads and 8/2 KV heads. Removing heads saves both compute and KV cache.
- **Unstructured:** Blackwell has structured sparsity support (2:4 pattern). Could get 2x GEMM speedup for 50% sparsity. But NVFP4 already uses 4-bit -- combining with 2:4 sparsity would be 2-bit effective, likely too aggressive.

**Status:** NOT TRIED.

**Feasibility:** Days (expert usage profiling), weeks (pruning + fine-tuning)
**Confidence:** Likely for expert pruning, speculative for weight pruning on already-quantized model

### C13. Knowledge Distillation

**What:** Train a smaller model to mimic the larger model's outputs. The student model is faster but retains most of the teacher's quality.

**Impact:** If we could distill Gemma4 26B into a dense 9B model with similar quality, we'd get 4,471 tok/s batch throughput (experiment #33) with better per-user latency. But distillation rarely preserves >95% of MoE model quality in a dense student.

**Status:** NOT TRIED (orthogonal to kernel optimization).

**Feasibility:** Weeks to months
**Confidence:** Likely (well-established technique), but quality preservation uncertain

### C14. Low-Rank Approximation for Inference

**What:** Decompose weight matrices W = U * V where U is M x R and V is R x N with R << min(M,N). Reduces parameters and compute.

**Impact:** Applying SVD to Gemma4's expert weights could reduce each expert from [H, 2N] to [H, R] + [R, 2N]. But for NVFP4 at 4-bit, the memory savings from rank reduction are modest. Compute savings depend on whether the two smaller matmuls are faster than one larger one on tensor cores.

**Status:** NOT TRIED.

**Feasibility:** Days to prototype, weeks to validate quality
**Confidence:** Speculative

### Compounding and Exclusions (Section C)

- **Compound:** Weight quant (NVFP4) + KV cache quant (FusenCache) + activation quant -- all orthogonal, all stack
- **Compound:** Pruning + quantization -- prune first, then quantize the smaller model
- **Compound:** QAT + any quantization format -- improves quality at same bit width
- **Mutually exclusive:** NVFP4 vs AQLM vs QuIP# vs SqueezeLLM -- pick one weight quant scheme
- **Mutually exclusive:** 2:4 structured sparsity + NVFP4 -- unclear if tensor cores support both simultaneously
- **Key insight:** We're at 4-bit weights + 4-bit KV cache. Going lower requires model retraining or quality sacrifice. The next frontier is pruning (removing computation) rather than quantizing it further.

---

## D. Model Architecture Optimizations

### D1. MoE Routing Optimization

**What:** Improve the expert selection process -- load balancing across experts, routing efficiency, adaptive top-K.

**Impact:** Gemma4 uses top-8 out of 128 experts. If some experts are overloaded, load imbalance causes some GEMM calls to be larger than others (hurting parallelism). Optimizing routing could improve batch throughput by reducing tail latency.

**Status:** NOT TRIED (routing is baked into model weights).

**Feasibility:** Days (profiling), weeks (fine-tuning routing)
**Confidence:** Speculative

### D2. Expert Pruning

**What:** Remove experts that are rarely selected during inference. If expert #73 handles < 0.1% of tokens, removing it barely affects quality but reduces model size.

**Impact:** 128 experts at ~2.4MB each (NVFP4) = 307MB total expert weights. If 20% of experts handle < 1% of tokens, removing them saves ~60MB and reduces the effective working set. More importantly, fewer active experts means less dispatch overhead per token.

**Status:** NOT TRIED. Future_work.md lists "AutoKernel MoE search" as Tier 5.

**Feasibility:** Days (profiling), days (pruning if no fine-tuning)
**Confidence:** Likely (MoE models empirically have underutilized experts)

### D3. Expert Merging

**What:** Combine similar experts into one, reducing total expert count while preserving diversity. Experts with similar weight matrices are averaged or distilled into a single expert.

**Impact:** If we merge 128 experts down to 64, each forward pass dispatches to top-4 instead of top-8 (adjusting routing), halving expert compute. Quality impact depends on expert diversity -- highly similar experts merge well, diverse ones don't.

**Status:** NOT TRIED.

**Feasibility:** Weeks (need to analyze expert similarity, test quality)
**Confidence:** Speculative

### D4. Layer Pruning

**What:** Remove entire transformer layers that contribute minimally to output quality. Often middle layers are most redundant.

**Impact:** Gemma4 has 30 layers. Removing 5 layers (17% reduction) could give ~1.2x speedup if those layers truly add little. But each layer includes MoE dispatch (which is most of the compute), so any removed layer gives proportional speedup. Quality impact is the concern.

**Status:** NOT TRIED.

**Feasibility:** Days (test quality with layers removed)
**Confidence:** Speculative (model-dependent)

### D5. Early Exit

**What:** If the model is "confident" about the next token after layer L < 30, skip the remaining layers. Reduces compute for "easy" tokens.

**Impact:** Could speed up 30-50% of tokens (common words, punctuation). But requires training an exit classifier per layer, and the savings compound poorly with CUDA graphs (which capture a fixed execution graph). Would need to break CUDA graph capture into per-layer segments.

**Status:** NOT TRIED.

**Feasibility:** Weeks (exit classifier training + serving integration)
**Confidence:** Speculative (CUDA graph incompatibility is a hard problem)

### D6. Dynamic Depth

**What:** Different tokens route through different numbers of layers, based on complexity. A router decides at each layer whether to continue or skip.

**Impact:** Similar to D5 but more flexible. Same CUDA graph incompatibility problem. For batch serving, different tokens in the same batch wanting different depths creates scheduling complexity.

**Status:** NOT TRIED.

**Feasibility:** Weeks
**Confidence:** Speculative

### D7. Weight Sharing Across Layers (ALBERT-style)

**What:** Use the same weight matrices for multiple layers. Reduces model size dramatically but requires the model to be trained this way.

**Impact:** Cannot be applied to Gemma4 post-hoc. Would need retraining.

**Status:** NOT APPLICABLE.

**Feasibility:** Impossible for Gemma4
**Confidence:** Theoretical

### D8. Activation Checkpointing

**What:** Don't store intermediate activations; recompute them during backward pass. Trades compute for memory.

**Impact:** Only relevant for training, not inference. During inference, we only do the forward pass and don't need activations for backward. However, for very long prefill sequences, activation memory could be a concern -- but we're not hitting that limit.

**Status:** NOT APPLICABLE (inference only).

**Feasibility:** N/A
**Confidence:** N/A

### D9. Mixture of Depths

**What:** Each token "chooses" how many transformer blocks to pass through, via a learned routing mechanism. Tokens with low routing weight skip certain blocks entirely.

**Impact:** Requires model retraining with the MoD routing mechanism. Cannot retrofit into Gemma4. Would compound nicely with MoE (mixture of depths + mixture of experts = adaptive compute in both dimensions).

**Status:** NOT APPLICABLE (requires retraining).

**Feasibility:** Impossible for Gemma4
**Confidence:** Theoretical

### D10. Matryoshka Representations

**What:** Train embeddings so that any prefix of the embedding dimensions is a valid (lower-quality) representation. Enables dynamic width at inference time.

**Impact:** Not applicable to Gemma4 (not trained this way). Could enable dynamic precision per token if combined with truncated embeddings for "easy" tokens.

**Status:** NOT APPLICABLE.

**Feasibility:** Impossible for Gemma4
**Confidence:** Theoretical

### Compounding and Exclusions (Section D)

- **Compound:** Expert pruning (D2) + expert merging (D3) -- prune first, then merge survivors
- **Compound:** Layer pruning (D4) + quantization -- fewer layers, each quantized
- **Compound:** Early exit (D5) + speculative decoding -- fast exit for draft model
- **Mutually exclusive:** Dynamic depth (D6) + CUDA graphs -- CUDA graphs need static execution paths
- **Mutually exclusive:** Early exit (D5) + CUDA graphs -- same problem
- **Key insight:** Most architecture optimizations require retraining. The practical ones for our deployed model are expert pruning (D2) and layer pruning (D4) -- both can be tested empirically without retraining.

---

## E. System-Level Optimizations

### E1. Continuous Batching

**What:** Dynamically add and remove requests from a running batch as they start and finish, rather than waiting for an entire batch to complete.

**Impact:** Already deployed via vLLM. This is why we can serve 232 concurrent requests efficiently.

**Status:** PROVEN, DEPLOYED.

**Feasibility:** N/A
**Confidence:** Proven

### E2. Prefix Caching

**What:** Cache KV states for common prompt prefixes (system prompts, few-shot examples). New requests sharing the prefix skip prefill for the shared portion.

**Impact:** Tested. Experiment #40: removing prefix caching caused -7% decode throughput. Experiment #77: removing it gave +16% batch32 throughput (fewer cache lookups). Net: prefix caching helps decode-heavy workloads, slightly hurts batch throughput.

**Status:** PROVEN. Currently disabled for batch-serving config, enabled for decode-heavy config.

**Feasibility:** N/A (deployed)
**Confidence:** Proven (workload-dependent tradeoff)

### E3. Tensor Parallelism (TP)

**What:** Split model weights across multiple GPUs, each computing part of every layer. Reduces per-GPU memory and increases aggregate bandwidth.

**Impact:** TP=2 on two GPUs would double KV cache capacity and bandwidth, enabling ~2x batch size. With NVLink between two RTX PRO 6000s (96GB each), could serve much larger batches. Identified as Tier 2 in future_work.md.

**Status:** NOT TRIED on our specific setup. vLLM supports TP natively.

**Feasibility:** Hours to verify, days to optimize
**Confidence:** Likely (well-established for vLLM)

### E4. Pipeline Parallelism (PP)

**What:** Split model layers across GPUs -- GPU0 runs layers 0-14, GPU1 runs layers 15-29. Enables running larger models than fit on one GPU.

**Impact:** For Gemma4 26B on RTX 5090 (32GB), the model fits on one GPU, so PP isn't needed for capacity. PP adds pipeline bubble overhead (GPU idle time between stages). Only useful if we need multi-GPU for larger models.

**Status:** NOT TRIED (not needed).

**Feasibility:** Hours
**Confidence:** Proven (but not needed for our model/GPU combo)

### E5. Expert Parallelism (EP)

**What:** Distribute different MoE experts across different GPUs. Each GPU holds a subset of experts and receives tokens routed to those experts.

**Impact:** With 128 experts and 2 GPUs, each GPU holds 64 experts. Reduces per-GPU expert weight memory by 2x. But requires all-to-all communication for token routing, which is expensive. Best with high-bandwidth interconnects (NVLink).

**Status:** NOT TRIED. vLLM does not natively support EP for Gemma4.

**Feasibility:** Weeks (requires custom EP implementation or vLLM upstream support)
**Confidence:** Likely for multi-GPU, but communication overhead is a concern on consumer NVLink

### E6. Data Parallelism (DP)

**What:** Run independent model replicas on different GPUs, each handling different requests. No communication between replicas during inference.

**Impact:** Linear scaling with number of GPUs. 2 GPUs = 2x throughput. No communication overhead. But each GPU needs the full model, so no memory savings. With NVFP4, Gemma4 26B fits comfortably in 32GB.

**Status:** NOT TRIED (single GPU currently).

**Feasibility:** Hours (just launch multiple vLLM instances)
**Confidence:** Proven (trivially parallel)

### E7. Disaggregated Prefill/Decode

**What:** Use separate GPU pools for prefill (compute-bound) and decode (memory-bandwidth-bound). Prefill GPUs run at high arithmetic intensity; decode GPUs run at high batch sizes.

**Impact:** Eliminates the interference between prefill and decode in mixed batching. Prefill GPUs can use higher precision for quality; decode GPUs optimize for throughput. For our single-GPU setup, not applicable. For multi-GPU deployment, could improve both prefill latency and decode throughput by 20-40%.

**Status:** NOT TRIED.

**Feasibility:** Days (vLLM supports disaggregated serving experimentally)
**Confidence:** Likely (proven at scale by companies like Anthropic, Google)

### E8. KV Cache Offloading (CPU/SSD)

**What:** Move idle KV cache entries to CPU memory or SSD when GPU memory is full. Swap back when the request resumes.

**Impact:** Enables serving more concurrent requests than GPU memory allows, at the cost of swap latency. RTX 5090 has PCIe 5.0 x16 (~64 GB/s), so swapping 1MB of KV cache takes ~15us -- acceptable for preempted requests. With FusenCache 4x compression, we already have 4x more capacity before needing offloading.

**Status:** NOT TRIED. vLLM supports CPU offloading.

**Feasibility:** Hours (vLLM flag)
**Confidence:** Likely (extends capacity at cost of latency for preempted requests)

### E9. Request Scheduling Algorithms

**What:** Order requests to maximize throughput or minimize latency. Shortest-job-first, priority queues, fairness-weighted scheduling.

**Impact:** vLLM uses FCFS by default. For mixed workloads (short chat + long generation), SJF could improve average latency by 20-30% without affecting throughput. For uniform workloads, scheduling policy doesn't matter much.

**Status:** NOT TRIED.

**Feasibility:** Hours (vLLM scheduling is configurable)
**Confidence:** Likely (depends on workload mix)

### E10. Preemption Policies (Swap vs Recompute)

**What:** When GPU memory is exhausted, decide whether to swap KV cache to CPU (preserving state) or evict and recompute from the prompt later. Swap is faster for short evictions; recompute is better for long evictions.

**Impact:** With FusenCache 4x compression, preemption happens less often. When it does, swap is typically better for our setup (PCIe 5.0 is fast, recompute requires full prefill).

**Status:** NOT TRIED (default vLLM preemption is swap).

**Feasibility:** Hours
**Confidence:** Likely (marginal improvement)

### E11. Chunked Prefill

**What:** Break long prefill sequences into chunks, interleaving prefill chunks with decode steps. Prevents long prefills from starving decode requests.

**Impact:** TESTED. Experiment #36: disabling chunked prefill gave marginal improvement for isolated benchmarks (4,513 vs 4,471 tok/s). Experiment #39: disabling for 9B model gave +31% decode. For mixed serving workloads, chunked prefill is important for latency fairness.

**Status:** PROVEN. Disabled for batch-only benchmarks, keep enabled for mixed serving.

**Feasibility:** N/A (deployed)
**Confidence:** Proven

### E12. Memory Pool Management

**What:** Pre-allocate GPU memory in pools and manage allocation/deallocation without cudaMalloc/cudaFree calls. Eliminates allocation overhead and fragmentation.

**Impact:** vLLM already does this for KV cache blocks. PyTorch's caching allocator handles tensor memory. Further optimization would require modifying vLLM's block allocator. Minimal expected gain since allocation is not a bottleneck (< 0.1% of decode time).

**Status:** INHERENT (vLLM + PyTorch allocators).

**Feasibility:** N/A
**Confidence:** Proven (already deployed)

### E13. Multi-GPU Communication Optimization

**What:** Optimize NCCL collective operations, overlap communication with computation, use NVLink vs PCIe strategically.

**Impact:** Only relevant for multi-GPU setups (TP, EP, PP). For single RTX 5090, no inter-GPU communication. For future TP=2, NVLink bandwidth and latency will determine scaling efficiency.

**Status:** NOT TRIED (single GPU).

**Feasibility:** Days (when we go multi-GPU)
**Confidence:** Likely

### E14. CPU Offloading for Cold Experts

**What:** Keep rarely-used MoE experts in CPU memory, loading them to GPU on demand. Hot experts stay on GPU permanently.

**Impact:** With 128 experts at ~2.4MB each (NVFP4) = 307MB total, they all fit in GPU memory easily. Cold expert offloading doesn't help memory. However, for larger models or when combining with increased batch sizes that exhaust GPU memory, this could be useful.

**Status:** ANALYZED AND RULED OUT. Expert working set (top-8 of 128, varying per token) exceeds L2 cache. All experts must reside in HBM regardless. The model fits in GPU memory, so CPU offloading adds latency without saving meaningful memory.

**Feasibility:** Days
**Confidence:** Proven (not beneficial for our setup)

### Compounding and Exclusions (Section E)

- **Compound:** Continuous batching + prefix caching + chunked prefill -- all stack in vLLM
- **Compound:** TP + FusenCache -- TP doubles bandwidth, FusenCache quadruples KV capacity
- **Compound:** Disaggregated serving + DP -- independent replicas per phase
- **Compound:** KV offloading + FusenCache -- offload compressed KV (4x less to swap)
- **Mutually exclusive:** TP vs DP on same GPU pair -- each GPU either has full model (DP) or partial model (TP)
- **Mutually exclusive:** EP vs TP for same model dimension -- usually pick one parallelism strategy per dimension
- **Key insight:** For single GPU, the biggest remaining system-level gain is disaggregated serving (if we had 2 GPUs). For single-GPU max throughput, we've already hit the compute ceiling (experiment #85: batch throughput plateaus at ~7,100 tok/s regardless of KV format).

---

## F. Kernel-Level Optimizations

### F1. Operator Fusion

**What:** Combine multiple sequential GPU operations into a single kernel launch, eliminating intermediate memory reads/writes and launch overhead.

**Impact:** EXTENSIVELY EXPLORED:
- RMSNorm + FP4 quantization fusion: 2.95x speedup on that operation (kernels/fused_norm_fp4.py)
- MoE shuffle + quantization fusion: 2.3% system gain, viable with ~5-line CUDA change (moe_shuffle_fusion_analysis.md)
- SiLU + mul + FP4 quant: Already fused in vLLM's `silu_and_mul_scaled_fp4_experts_quant`
- Norm + shuffle + quant: NOT VIABLE (data dependency -- norm must happen before routing)

**Status:** PROVEN. Multiple fusions deployed. Remaining fusion opportunities have diminishing returns.

**Feasibility:** Hours per fusion
**Confidence:** Proven

### F2. Custom CUDA vs Triton vs cuBLAS

**What:** Choose the right kernel technology for each operation. cuBLAS for standard GEMM, Triton for fusion-friendly ops, custom CUDA for hardware-specific operations.

**Impact:** EXTENSIVELY TESTED:
- cuBLAS FP4 GEMM (via `_scaled_mm`): Best for standard matmul. 1,261 TFLOPS.
- Triton dequant + cuBLAS matmul: Best for W4A16 (328 TFLOPS). Split beats fused.
- Custom CUDA quant kernel: 23us vs 358us Python (experiment #6). 15x faster quantization.
- Triton FP16 matmul: 4x slower than FP4 cuBLASLt (experiment #52).
- CUTLASS grouped GEMM: Used by vLLM for MoE dispatch. Competitive with cuBLAS.

**Status:** PROVEN. Each op uses the optimal backend.

**Feasibility:** N/A (current state)
**Confidence:** Proven

### F3. Persistent Kernels

**What:** A single kernel stays resident on the GPU across multiple iterations/layers, avoiding repeated launch overhead. The kernel loops internally.

**Impact:** Tested for W4A16 matmul (experiment #5): persistent kernel 680 programs + two-level K = 153.4 TFLOPS. Later reverted for the BK=128 configs (experiment #77: -12 TFLOPS). Persistent kernels help when launch overhead is significant relative to kernel runtime. For our MoE dispatch (many small GEMMs), persistent kernels could save ~750us of launch overhead per decode step.

**Status:** TESTED, MIXED RESULTS. Helps for some shapes, hurts for others.

**Feasibility:** Days
**Confidence:** Speculative (shape-dependent)

### F4. Warp Specialization

**What:** Different warps within a thread block perform different roles -- some load data, some compute, some store results. Overlaps data movement with computation.

**Impact:** FA3 uses this. For our GEMM kernels, cuBLAS and CUTLASS already use warp specialization internally. For custom Triton kernels, Triton doesn't expose warp specialization directly (it's a compiler optimization). Would need raw CUDA for explicit warp specialization.

**Status:** IMPLICIT (via cuBLAS/CUTLASS).

**Feasibility:** Weeks (raw CUDA required)
**Confidence:** Likely (proven technique, but hard to beat cuBLAS)

### F5. Tensor Core Utilization

**What:** Ensure all GEMM operations use the appropriate tensor core instructions (FP4, FP8, INT8, TF32) for maximum throughput.

**Impact:** VERIFIED. Experiment #30 in future_work.md: "FLASHINFER_CUTLASS (linear) + VLLM_CUTLASS (MoE) both use CUTLASS 3.x SM120 FP4 MMA -- real tensor cores, not emulation." We're using FP4 tensor cores for all GEMM operations.

**Status:** PROVEN, DEPLOYED.

**Feasibility:** N/A
**Confidence:** Proven

### F6. Memory Access Pattern Optimization

**What:** Ensure memory accesses are coalesced (adjacent threads access adjacent memory), use vectorized loads (128-bit), and minimize bank conflicts in shared memory.

**Impact:** TESTED in our CUDA quantization kernel:
- Experiment #9: Vectorized half2 loads + additive thresholds = 24us (from 26us v2)
- L2 swizzle for W4A16 = 136.8 TFLOPS (experiment #21, from 15 TFLOPS baseline)

**Status:** PROVEN, DEPLOYED.

**Feasibility:** Hours per kernel
**Confidence:** Proven

### F7. Register Pressure Tuning

**What:** Ensure kernels don't use too many registers (causing spill to local memory) or too few (under-utilizing compute). Tune BLOCK_SIZE_K, num_warps, etc.

**Impact:** CRITICAL FINDING: BLOCK_SIZE_K=128 causes register spill on some shapes. num_stages > 4 causes shared memory overflow. Experiment #35: BK=128 reverted (-27 TFLOPS due to spill). Later fixed with Triton 3.6.0: BK=128 = 327 TFLOPS (experiment #82).

**Status:** PROVEN, OPTIMIZED.

**Feasibility:** N/A (done)
**Confidence:** Proven

### F8. Shared Memory Optimization

**What:** Use shared memory effectively for data reuse within thread blocks. Tile data into shared memory, synchronize, compute, repeat.

**Impact:** Implicit in our Triton kernels via `num_stages` parameter (controls shared memory pipeline depth). Experiment #6: stages=4 was optimal. Stages > 4 caused shared memory overflow.

**Status:** PROVEN, OPTIMIZED.

**Feasibility:** N/A
**Confidence:** Proven

### F9. Cooperative Groups for Cross-SM Synchronization

**What:** CUDA cooperative groups allow thread blocks across different SMs to synchronize, enabling algorithms that span the entire GPU.

**Impact:** Required for true persistent kernels that loop across all tiles of a GEMM. Currently unused in our Triton kernels (Triton doesn't support cooperative launches natively). Would need raw CUDA. Potential benefit: one mega-kernel for the entire MoE dispatch, eliminating all launch overhead.

**Status:** NOT TRIED.

**Feasibility:** Weeks (raw CUDA)
**Confidence:** Speculative

### F10. CUTLASS vs cuBLAS vs Custom for GEMM

**What:** CUTLASS is NVIDIA's open-source GEMM library (template-based, highly tunable). cuBLAS is the closed-source optimized library. Custom means writing from scratch.

**Impact:** TESTED:
- cuBLAS (via `_scaled_mm`): Best for individual GEMM calls. 1,261 TFLOPS for FP4.
- CUTLASS grouped GEMM: Best for MoE dispatch (batch multiple expert GEMMs). Used by vLLM.
- Custom Triton GEMM: 4x slower than cuBLAS for FP16 (experiment #52). Cannot compete for standard GEMM.
- CUTLASS 3.x SM120: Identified as Tier 2 investigation target for advanced MoE patterns.

**Status:** PROVEN. cuBLAS for individual GEMM, CUTLASS for grouped MoE.

**Feasibility:** N/A
**Confidence:** Proven

### F11. Async Memory Copies (cp.async, TMA on Blackwell)

**What:** Blackwell's Tensor Memory Accelerator (TMA) enables hardware-accelerated async memory copies from global memory to shared memory, bypassing registers. cp.async was the Hopper precursor.

**Impact:** CUTLASS 3.x and FlashAttention 3 use TMA. Our cuBLAS/CUTLASS paths benefit from TMA implicitly. For custom Triton kernels, Triton's compiler generates TMA instructions automatically when targeting SM90+. The main opportunity is in custom CUDA kernels where TMA isn't being used.

**Status:** IMPLICIT (via CUTLASS/cuBLAS).

**Feasibility:** Days (for custom kernels)
**Confidence:** Likely

### F12. Software Pipelining

**What:** Overlap data loading for the next tile with computation of the current tile. Multiple pipeline stages keep the GPU busy while waiting for memory.

**Impact:** Controlled via Triton's `num_stages` parameter. We found stages=4 optimal for most shapes, stages=3 for some BK=128 configs. cuBLAS and CUTLASS do this internally.

**Status:** PROVEN, OPTIMIZED.

**Feasibility:** N/A
**Confidence:** Proven

### F13. Kernel Autotuning

**What:** Automatically search over kernel configurations (tile sizes, warp counts, pipeline stages) to find the fastest for each problem shape.

**Impact:** CRITICAL. Going from fixed config (15 TFLOPS) to autotuned (328 TFLOPS) = 22x improvement for W4A16. @triton.autotune is our primary tool. FlashInfer autotuner caches 104 configs (experiment #32: 55min -> 63s load).

**Status:** PROVEN, DEPLOYED.

**Feasibility:** N/A
**Confidence:** Proven

### Compounding and Exclusions (Section F)

- **Compound:** Fusion (F1) + autotuning (F13) -- fuse first, then autotune the fused kernel
- **Compound:** TMA (F11) + software pipelining (F12) + persistent kernels (F3) -- maximum memory/compute overlap
- **Compound:** All kernel optimizations compound with system-level optimizations
- **Mutually exclusive:** Triton vs raw CUDA for same kernel -- pick one implementation
- **Key insight:** We've extracted most of the kernel-level performance. The remaining opportunity is a single fused MoE dispatch kernel (persistent, cooperative groups) that eliminates all launch overhead for the 128-expert dispatch. This is the highest-impact kernel work remaining.

---

## G. Hardware-Specific (Blackwell SM120)

### G1. FP4 Tensor Cores

**What:** Blackwell introduces native FP4 (e2m1) tensor core operations, computing A_fp4 * B_fp4 with FP16/FP32 accumulation. Peak throughput is 4x FP8 or 8x FP16.

**Impact:** DEPLOYED. This is the foundation of our NVFP4 approach. 1,261 TFLOPS measured vs ~200 TFLOPS theoretical FP16 peak. The >100% of FP16 peak reflects the FP4 tensor core advantage.

**Status:** PROVEN, DEPLOYED.

**Feasibility:** N/A
**Confidence:** Proven

### G2. TMA (Tensor Memory Accelerator)

**What:** Hardware unit that handles complex memory access patterns (strided, tiled, im2col) asynchronously. Frees up SMs from memory management work.

**Impact:** Used implicitly by CUTLASS 3.x and our cuBLAS calls. For custom kernels, explicit TMA usage could improve memory-bound operations (KV cache access, expert weight loading). The main benefit is reducing register pressure since TMA handles address computation in hardware.

**Status:** IMPLICIT (via libraries).

**Feasibility:** Days (for custom TMA usage)
**Confidence:** Likely

### G3. Transformer Engine

**What:** NVIDIA's library for automatic FP8 casting with per-tensor amax tracking. Handles the complexity of mixed-precision training/inference with FP8.

**Impact:** We use a related approach via modelopt for FP4. Transformer Engine's automatic scaling could potentially handle activation quantization more efficiently than our manual approach. However, TE is designed for FP8, not FP4.

**Status:** NOT DIRECTLY USED (modelopt handles FP4 equivalent).

**Feasibility:** Days to evaluate
**Confidence:** Speculative (FP8 TE may be slower than FP4 native)

### G4. NVLink for Multi-GPU

**What:** High-bandwidth GPU-to-GPU interconnect. Consumer NVLink on RTX 5090 provides ~100 GB/s. Professional NVLink on PRO 6000 provides ~450 GB/s.

**Impact:** Required for efficient TP=2 or EP. With consumer NVLink (100 GB/s), TP=2 scaling is ~1.5-1.7x (communication overhead eats 15-30%). With PRO 6000 NVLink, ~1.8-1.9x. Listed as Tier 2 investigation.

**Status:** NOT TRIED (single GPU currently).

**Feasibility:** Hours to verify
**Confidence:** Likely

### G5. L2 Cache Partitioning

**What:** Some NVIDIA GPUs support partitioning L2 cache between streaming (evict quickly) and persisting (keep in cache). Can prioritize hot data.

**Impact:** RTX 5090 has ~48MB L2. Our expert weights are ~307MB total (128 x 2.4MB). Only ~15% of experts fit in L2 simultaneously. L2 partitioning could reserve a portion for expert weights vs KV cache vs activations. But the `cudaAccessPolicyWindow` API availability on SM120 consumer GPUs is uncertain.

**Status:** NOT TRIED.

**Feasibility:** Days to test
**Confidence:** Speculative (API availability uncertain)

### G6. Hardware Thread Block Scheduling

**What:** Blackwell may have improved thread block scheduling that enables better occupancy and load balancing across SMs. CUDA 12.8 might expose new scheduling hints.

**Impact:** Implicit -- the hardware scheduler handles this automatically. We can influence it via occupancy (register/shared memory usage), grid dimensions, and launch configuration. Our autotuning already finds good configurations.

**Status:** IMPLICIT.

**Feasibility:** N/A
**Confidence:** Proven (via autotuning)

### G7. Dynamic Parallelism (Kernels Launching Kernels)

**What:** A running kernel can launch child kernels without returning to the CPU. Useful for algorithms with data-dependent branching.

**Impact:** Could enable a single MoE dispatch kernel that launches per-expert GEMM kernels internally, based on routing results. Eliminates CPU round-trip for expert dispatch. However, dynamic parallelism has high overhead on consumer GPUs and is generally slower than pre-planned launches.

**Status:** NOT TRIED.

**Feasibility:** Days
**Confidence:** Speculative (overhead may be too high)

### G8. Blackwell-Specific CUTLASS Optimizations

**What:** CUTLASS 3.x has SM120-specific kernels with TMA, warp specialization, and cluster-level parallelism tuned for Blackwell's architecture.

**Impact:** Already used via vLLM's CUTLASS MoE and FlashInfer's CUTLASS attention. Further hand-tuning CUTLASS templates for our specific shapes (H=2816, expert size) could give 5-15% improvement.

**Status:** PARTIALLY DEPLOYED (via vLLM/FlashInfer).

**Feasibility:** Weeks (CUTLASS template customization is complex)
**Confidence:** Likely

### G9. CUDA 12.8 New Features

**What:** CUDA 12.8 brings new APIs, potentially including improved cooperative launch, async barriers, and SM120-specific intrinsics.

**Impact:** Unknown until we audit CUDA 12.8 release notes. Potentially enables features like thread block clusters (SM90+), which group thread blocks for efficient shared memory communication.

**Status:** NOT INVESTIGATED.

**Feasibility:** Days (audit + prototype)
**Confidence:** Speculative

### Compounding and Exclusions (Section G)

- **Compound:** FP4 tensor cores (G1) + TMA (G2) + CUTLASS 3.x (G8) -- the full Blackwell stack
- **Compound:** NVLink (G4) + TP (E3) -- multi-GPU scaling
- **Compound:** L2 partitioning (G5) + expert weight caching -- keep hot experts in L2
- **Key insight:** We're already using the most important Blackwell feature (FP4 tensor cores). The remaining hardware-specific gains are incremental (5-15%) and require deep CUDA expertise.

---

## H. Mathematical / Algorithmic

### H1. Flash-Decoding (Split-K for Long Sequences)

**What:** Split the KV cache across thread blocks along the sequence dimension, each block computes partial attention, then reduce across blocks. Enables parallelism for single-query decode against long KV cache.

**Impact:** TESTED in our FusenCache kernel: 2.16x improvement from split-K (from future_work.md). For standard attention, FlashInfer already uses split-K for decode. Most impactful for long sequences where a single thread block can't process the entire KV cache efficiently.

**Status:** PROVEN, DEPLOYED (both in FusenCache and FlashInfer).

**Feasibility:** N/A
**Confidence:** Proven

### H2. Online Softmax

**What:** Compute softmax in a single pass over the data, maintaining running max and sum. Avoids the traditional 3-pass algorithm (compute max, compute exp, normalize).

**Impact:** Already used by FlashAttention and FlashInfer. Essential for tiled attention -- without online softmax, tiles would need to communicate global max.

**Status:** PROVEN, DEPLOYED.

**Feasibility:** N/A
**Confidence:** Proven

### H3. Approximate Nearest Neighbor for Attention (LSH, PQ)

**What:** Use locality-sensitive hashing or product quantization to find the most relevant KV entries without computing full attention. O(N log N) or O(N) instead of O(N).

**Impact:** Only useful for very long sequences where attention is the bottleneck. For our decode setup (single query against sliding window of 1024), attention is already fast. For 128K context global layers (5 layers), could help for prefill but not decode.

**Status:** NOT TRIED.

**Feasibility:** Weeks
**Confidence:** Speculative (decode is not attention-bottlenecked)

### H4. Matrix Sketching for Approximate Attention

**What:** Use random projections to approximate the attention matrix. Reduces the dimensionality of K and V before computing attention.

**Impact:** Quality degradation is hard to control. For inference (not training), any approximation error propagates through subsequent layers. Not suitable for production serving where quality must be deterministic.

**Status:** NOT TRIED.

**Feasibility:** Weeks
**Confidence:** Theoretical

### H5. Polynomial Approximations for Softmax/GELU

**What:** Replace exp() or GELU with polynomial approximations that are faster on GPU. Taylor expansion or minimax polynomials.

**Impact:** Modern GPUs have fast transcendental units (SFU) that compute exp() in ~4 cycles. Polynomial approximation might save 1-2 cycles per element. For our setup, softmax/GELU is < 1% of compute. Maximum system impact: ~0.1%.

**Status:** NOT TRIED.

**Feasibility:** Hours
**Confidence:** Theoretical (negligible impact)

### H6. FFT-Based Convolutions as Attention Replacement

**What:** Replace attention with FFT-based convolutions that mix tokens in frequency domain. O(N log N) and highly parallelizable.

**Impact:** Requires model retraining. Not applicable to Gemma4.

**Status:** NOT APPLICABLE.

**Feasibility:** Impossible for Gemma4
**Confidence:** Theoretical

### H7. Structured Matrices (Monarch, Butterfly)

**What:** Replace dense weight matrices with structured (sparse) matrices that have O(N^1.5) or O(N log N) parameters and compute, vs O(N^2) for dense.

**Impact:** Requires model retraining. For inference of an existing model, would need to factorize existing weight matrices into structured form, which is lossy. NVFP4 already reduces each weight to 4 bits, so the memory savings from structured matrices are modest.

**Status:** NOT TRIED.

**Feasibility:** Weeks to months
**Confidence:** Theoretical

### H8. Tensor Decomposition (Tucker, CP)

**What:** Decompose a weight tensor into smaller factors. For a 3D tensor, CP decomposition gives W = sum_r(a_r x b_r x c_r). Reduces parameters and compute.

**Impact:** Similar to low-rank approximation (C14) but for higher-order tensors. For standard linear layers (2D), this reduces to SVD. MoE expert weights could potentially be decomposed with shared factors across experts (discovering common "expert basis vectors").

**Status:** NOT TRIED.

**Feasibility:** Weeks
**Confidence:** Speculative

### H9. Randomized Numerical Linear Algebra

**What:** Use random projections, random sampling, and sketching to approximate matrix operations. Can give approximate GEMM in less time.

**Impact:** For inference, we need exact (or quantized-exact) results. Randomized GEMM would introduce uncontrolled error that accumulates across 30 layers. Not suitable for production.

**Status:** NOT APPLICABLE.

**Feasibility:** N/A
**Confidence:** Theoretical (not suitable for inference)

### H10. Cache-Oblivious Algorithms

**What:** Algorithms that perform well across all cache levels without explicit tuning for cache sizes. Recursively divide problems to naturally fit in cache.

**Impact:** For GPU kernels, we DO tune for specific cache sizes (L2 = 48MB, shared memory = 228KB per SM). Cache-oblivious approaches would sacrifice some performance for generality. Since we already autotune, cache-oblivious is dominated.

**Status:** NOT APPLICABLE (autotuning is superior).

**Feasibility:** N/A
**Confidence:** Proven (autotuning dominates)

### H11. Toeplitz/Circulant Matrix Structure

**What:** Exploit special matrix structure where each row is a shifted version of the previous. Convolutions have this structure. Enables FFT-based multiplication.

**Impact:** Attention matrices are NOT Toeplitz (they depend on content, not just position). Some positional bias components might be Toeplitz, but they're tiny relative to content-based attention. No meaningful application.

**Status:** NOT APPLICABLE.

**Feasibility:** N/A
**Confidence:** Theoretical

### H12. Newton-Schulz Iteration for Matrix Operations

**What:** Iterative method for computing matrix inverse square roots, useful for normalization (LayerNorm, RMSNorm).

**Impact:** RMSNorm already uses a simple rsqrt() operation which is well-optimized on GPU. Newton-Schulz would be slower for this simple case. Only useful for more complex matrix operations (like in optimizers during training).

**Status:** NOT APPLICABLE.

**Feasibility:** N/A
**Confidence:** Theoretical (slower for our use case)

### H13. Winograd Transforms for Small Convolutions

**What:** Reduce multiplication count for small convolutions by transforming to Winograd domain. Classic technique for 3x3 convolutions.

**Impact:** Gemma4 is transformer-based with no convolutions (except potentially in Mamba layers which use 1D convolution). For 1D conv in Mamba, Winograd might help but the conv is tiny relative to the attention and FFN compute.

**Status:** NOT TRIED.

**Feasibility:** Days
**Confidence:** Theoretical (negligible impact)

### Compounding and Exclusions (Section H)

- **Compound:** Flash-decoding (H1) + online softmax (H2) -- both used together in FlashAttention
- **Compound:** ANN attention (H3) + KV eviction (B9) -- approximate who to attend to, evict the rest
- **Key insight:** The most impactful mathematical optimization (flash-decoding) is already deployed. Remaining opportunities are either negligible (H5, H13) or require model changes (H6, H7).

---

## I. Emerging / Speculative

### I1. Test-Time Compute Scaling (Thinking Tokens, o1-Style)

**What:** Generate internal "thinking" tokens that improve output quality at the cost of more compute per response. The model reasons step-by-step before answering.

**Impact:** Orthogonal to throughput optimization -- this is a quality/compute tradeoff. For our serving setup, thinking tokens mean more tokens generated per request, increasing total compute. Our optimizations (NVFP4, CUDA graphs, MTP) make thinking tokens cheaper per token, enabling more thinking within a latency budget.

**Status:** NOT TRIED (model capability, not kernel optimization).

**Feasibility:** Hours (if model supports it)
**Confidence:** Proven (o1, Claude, etc. demonstrate this)

### I2. Dynamic Compute Allocation Per Token

**What:** Spend more compute on "hard" tokens and less on "easy" tokens. A lightweight classifier predicts token difficulty and routes to different compute paths.

**Impact:** Related to early exit (D5) and dynamic depth (D6). The fundamental challenge remains CUDA graph incompatibility -- static execution graphs can't accommodate dynamic per-token compute.

**Status:** NOT TRIED.

**Feasibility:** Weeks
**Confidence:** Speculative

### I3. Mixture of Agents

**What:** Route different subtasks to specialized models (code model for code, math model for equations, base model for prose).

**Impact:** This is the Fusen Inference Engine vision (fusen_inference_engine.md). The router adds < 1ms overhead. For our single-model optimization, this means having multiple models loaded and switching between them. GPU memory limits how many models can coexist.

**Status:** DESIGNED (fusen_inference_engine.md), NOT DEPLOYED.

**Feasibility:** Weeks
**Confidence:** Likely

### I4. Compiler-Driven Whole-Model Optimization (XLA, TorchInductor, Triton)

**What:** Let the compiler optimize the entire model as a single computation graph, discovering fusion and scheduling opportunities that humans miss.

**Impact:** TESTED:
- torch.compile (experiment #17): 17x slower for single-layer M=1 decode. Compilation overhead dominates for small shapes.
- torch.compile (experiment #35): 3x slower than CUDA graphs (1,651 vs 4,471 batch64).
- Verdict: CUDA graphs are better than torch.compile for our decode workload. torch.compile may help for large-batch prefill.

**Status:** TESTED, INFERIOR to CUDA graphs for our workload.

**Feasibility:** N/A (tested)
**Confidence:** Proven (CUDA graphs win for decode)

### I5. FPGA/ASIC Inference Accelerators

**What:** Custom hardware designed specifically for transformer inference. Groq's LPU, Google's TPU, Cerebras WSE, etc.

**Impact:** Not applicable to our RTX 5090 setup. For production deployment, custom ASICs could be 2-10x more efficient per watt. But we're optimizing for NVIDIA consumer/professional GPUs.

**Status:** NOT APPLICABLE (different hardware platform).

**Feasibility:** N/A
**Confidence:** Proven (for those platforms)

### I6. Photonic Computing for Matrix Multiplication

**What:** Use light (photons) instead of electrons for matrix multiplication. Theoretically O(1) latency for any matrix size using optical interference.

**Impact:** Currently lab-stage technology. Not commercially available. Cannot integrate with GPU pipeline. 5-10 years from practical use.

**Status:** NOT APPLICABLE.

**Feasibility:** Impossible (not available)
**Confidence:** Theoretical

### I7. Analog Computing for Inference

**What:** Use analog circuits (e.g., crossbar arrays of memristors) for in-memory matrix multiplication. Each weight is stored as a resistance value.

**Impact:** Lab-stage. Precision is limited (4-6 bits effective), which happens to match our NVFP4 approach. Could be ideal for inference if commercialized. 3-7 years from practical use.

**Status:** NOT APPLICABLE.

**Feasibility:** Impossible (not available)
**Confidence:** Theoretical

### I8. Neuromorphic Approaches

**What:** Spiking neural networks on neuromorphic chips (Intel Loihi, IBM TrueNorth). Event-driven computation that only activates neurons when they spike.

**Impact:** Not compatible with transformer architecture. Transformers are inherently non-spiking. Would need completely different model architecture.

**Status:** NOT APPLICABLE.

**Feasibility:** Impossible
**Confidence:** Theoretical

### I9. In-Memory Computing (Processing-in-Memory)

**What:** Perform computation directly in memory chips (HBM-PIM, CXL-PIM). Eliminates the memory-compute bandwidth bottleneck by computing where data lives.

**Impact:** Samsung HBM-PIM exists but is not in RTX 5090. For memory-bound decode (where GEMV is the bottleneck), PIM could give 10-100x bandwidth improvement by computing the GEMV inside HBM. This is the theoretical "correct" solution to decode bottleneck.

**Status:** NOT APPLICABLE (hardware not available).

**Feasibility:** Impossible (hardware required)
**Confidence:** Likely (if hardware existed)

### Compounding and Exclusions (Section I)

- **Compound:** Test-time compute (I1) + all our throughput optimizations -- thinking tokens benefit from fast generation
- **Compound:** MoA (I3) + per-model optimization -- each model in the mixture gets its own optimized config
- **Key insight:** The most impactful near-term emerging technique is mixture of agents (I3), which we've already designed. The most impactful long-term technique is in-memory computing (I9), which would eliminate the decode bottleneck entirely.

---

## J. Blindspots / Self-Introspection

### J1. What Assumptions Are We Making That Might Be Wrong?

1. **"FP4 quality is good enough."** Our PPL on WikiText-2 is 701.4 -- significantly degraded from FP16. We've assumed this is acceptable because output reads coherently. But for tasks requiring precise reasoning (math, code), FP4 may be silently wrong 5-10% of the time. We haven't measured task-specific accuracy (HumanEval, GSM8K, MMLU).

2. **"Single GPU is the right deployment unit."** We optimize everything for one RTX 5090. But two RTX 5090s with NVLink might change the bottleneck entirely -- TP=2 doubles bandwidth, makes us compute-bound instead of memory-bound for decode, and changes which optimizations matter.

3. **"vLLM is the right serving framework."** We spend significant effort working around vLLM bugs and limitations. SGLang, TensorRT-LLM, or a custom serving loop might be faster. Experiment #92 showed vLLM is 100% GPU-bound, but the overhead analysis (experiment #87) showed 65% is "vLLM overhead" (5.38ms/8.32ms per token).

4. **"Batch throughput is the right metric."** We optimize for tok/s at high batch. But real serving has mixed workloads -- latency for interactive users matters more than peak throughput. Our MTP3 work (186 tok/s single-user) addresses this, but we haven't optimized for P99 latency under load.

5. **"The model architecture is fixed."** Gemma4 26B with 128 experts and top-8 routing is what it is. But if Google releases Gemma4.1 with MLA, different expert count, or architectural improvements, all our kernel work needs updating. We should build abstractions, not model-specific code.

6. **"Memory bandwidth is the decode bottleneck."** Experiment #29 showed we're at 13% of bandwidth limit (124 tok/s vs 952 tok/s theoretical). The gap isn't memory bandwidth -- it's kernel launch overhead, Python overhead, or something else we haven't measured. Experiment #92 clarified: wall time = GPU time (8.24ms), so it's GPU compute (GEMV 49.3%, GEMM 39.2%) not bandwidth. But 13% of BW limit suggests we're not fully utilizing the memory system even during GEMV.

7. **"CUDA graphs captured the right thing."** CUDA graphs gave 7x speedup by eliminating launch overhead. But CUDA graphs capture a fixed execution path. MoE routing varies per token (different experts activate), which means the graph either captures a "generic" path or multiple specialized paths. If routing variation is high, graph replay may include unnecessary work.

### J2. What Have We Not Measured That We Should?

1. **Task-specific accuracy:** HumanEval (code), GSM8K (math), MMLU (knowledge), MT-Bench (chat quality). PPL alone doesn't capture real-world quality.

2. **P99 latency under load:** We measure throughput at fixed batch sizes. Real serving has variable arrival rates. What's the tail latency at 50%, 80%, 95% utilization?

3. **Expert activation distribution:** Which of the 128 experts are actually used? How skewed is the distribution? This directly informs expert pruning (D2) and expert parallelism (E5).

4. **Memory bandwidth utilization during decode:** We know wall time = GPU time, but not how efficiently we use HBM bandwidth within each kernel. `ncu` profiling of the hot kernels would reveal this.

5. **Prefill-decode interference:** In mixed serving, how much does a prefill request slow down concurrent decode requests? The chunked prefill analysis (experiment #36) only tested isolated workloads.

6. **Context length scaling:** We've tested up to 4K tokens. How does throughput degrade at 32K, 64K, 128K? At what point does KV cache become the bottleneck rather than compute?

7. **Quality of FusenCache under real workloads:** We validated coherence on a few prompts. We haven't run systematic quality benchmarks comparing BF16 KV vs FusenCache K4V4 vs K8V4 on diverse tasks.

8. **Power consumption and thermal throttling:** RTX 5090 may thermal throttle during sustained batch serving. We should measure actual power draw and clock frequencies under load.

### J3. Where Are We Optimizing the Wrong Thing?

1. **Kernel-level when the bottleneck is system-level.** Our NVFP4 GEMM is 1,261 TFLOPS, but system throughput is 6,615 tok/s. The gap between kernel peak and system throughput suggests the bottleneck is somewhere other than individual kernel performance. MoE dispatch overhead, KV cache management, and scheduling are likely bigger factors than making any single kernel faster.

2. **Throughput when users want latency.** Single-user decode at 186 tok/s (MTP3) is fast, but time-to-first-token (TTFT) for long prompts matters more for interactive use. Our prefill throughput (14,977 tok/s at batch, per experiment #75) is good, but TTFT under load is unmeasured.

3. **The GPU when the bottleneck might be the network.** For production serving, network latency (token streaming over HTTP) may dominate perceived latency. A 50ms network roundtrip matters more than shaving 2ms off decode time.

4. **One model when we should serve two.** A small model (9B) for easy tasks + large model (26B) for hard tasks might give better overall throughput-quality tradeoff than optimizing one model to the limit. The Fusen Engine vision (fusen_inference_engine.md) addresses this.

### J4. What Would a Fundamentally Different Approach Look Like?

1. **Skip the serving framework entirely.** Instead of running inside vLLM, write a minimal C++ inference server that launches CUDA graphs directly. Eliminate all Python overhead. vLLM's overhead is measured at 5.38ms/token (experiment #87) -- a C++ server could eliminate most of this, getting close to the 2.94ms/token GPU-only time.

2. **Compile the entire model into one CUDA graph.** Not individual layers, but the entire 30-layer forward pass including MoE routing and expert dispatch. This is what TensorRT does for dense models, but MoE routing makes it harder (dynamic expert selection). A "mega-graph" with all 128 experts pre-compiled and routing resolved by conditional execution within the graph.

3. **Move to a different model entirely.** DeepSeek V3 with MLA and Multi-Latent Attention would have a fundamentally different optimization landscape (tiny KV cache, heavy compute per token). Qwen3.5-9B is already 4,471 tok/s at batch64 (experiment #33) -- if quality is sufficient, 9B may be the better production model.

4. **Hardware-software co-design.** Instead of optimizing for existing SM120, design a custom kernel for a custom accelerator. The "correct" hardware for MoE inference is processing-in-memory (I9) -- each expert stored in a PIM module, routing done in a central controller, zero data movement.

5. **Stateless inference with KV precomputation.** For applications with fixed system prompts (90%+ of production use), precompute KV caches for all common prefixes and store on SSD. Inference starts from the precomputed state, skipping prefill entirely. Combined with FusenCache compression, SSD-resident KV caches are practical.

### J5. What Don't We Know That We Don't Know?

This is by definition unanswerable, but here are our best guesses at the unknown unknowns:

1. **How other teams are solving the same problem.** Google's internal Gemma4 serving infrastructure likely uses optimizations we haven't thought of. NVIDIA's TensorRT-LLM team may have MoE-specific optimizations not in the public codebase.

2. **Whether there's a bug in our measurement.** We trust vLLM's throughput reporting and our bench.py harness. But an off-by-one in token counting, an incorrect clock normalization, or a CUDA synchronization error could make our numbers wrong. Cross-validating with an independent measurement tool would catch this.

3. **Future hardware capabilities.** NVIDIA's next-gen (Rubin) GPU may have features that make our current optimizations obsolete -- e.g., native MoE dispatch hardware, larger L2 cache that fits all experts, or 2-bit tensor cores.

4. **Interactions between optimizations at scale.** We test each optimization in isolation. The interaction between MTP3 + FusenCache + high batch + long context has not been tested as a combination. Non-linear interactions could produce surprising results (positive or negative).

5. **What the optimal operating point actually is.** We've found local optima (6,615 tok/s batch, 186 tok/s single-user). But the global optimum might require a combination we haven't tried -- e.g., TP=2 + MTP2 + FusenCache K8V4 + expert pruning + disaggregated serving.

---

## Summary: Highest-Impact Unexplored Opportunities

Ranked by expected impact on our specific setup (Gemma4 26B MoE NVFP4, RTX 5090), filtered to things we have NOT yet fully exploited:

| Rank | Technique | Section | Expected Impact | Feasibility | Confidence |
|------|-----------|---------|----------------|-------------|------------|
| 1 | Fused MoE dispatch kernel (persistent, all-expert) | F1/F3/F9 | 2-5x on MoE dispatch (85% of decode) | Days-Weeks | Likely |
| 2 | Expert pruning (remove underused experts) | D2 | 10-30% model size reduction, proportional speedup | Days | Likely |
| 3 | TP=2 on two GPUs (NVLink) | E3 | ~1.8x throughput, 2x KV capacity | Hours-Days | Likely |
| 4 | C++ minimal server (no Python overhead) | J4-1 | Up to 1.65x (eliminate 5.38ms overhead) | Weeks | Likely |
| 5 | FusenCache vLLM v1 integration | C3 | 4x KV capacity, higher batch sizes | Days | Proven |
| 6 | Disaggregated prefill/decode | E7 | 20-40% for mixed workloads | Days | Likely |
| 7 | Expert merging (128 -> 64 experts) | D3 | ~2x expert compute reduction | Weeks | Speculative |
| 8 | Layer pruning (remove 3-5 layers) | D4 | 10-17% compute reduction | Days | Speculative |
| 9 | QAT checkpoint (if Google releases one) | C5 | Better quality at same speed | Hours (to swap) | Likely |
| 10 | EAGLE speculative decoding (when vLLM fixes it) | A5 | 2-3x single-user | Weeks | Likely |
| 11 | 2:4 structured sparsity on expert weights | C12 | Up to 2x GEMM if tensor cores support it | Weeks | Speculative |
| 12 | MTP3 + batch serving fix (already close) | A3 | MTP3 batch = 4,967 tok/s (vs 6,615 baseline) | Days | Proven |
| 13 | Cross-layer KV sharing (groups of 5 sliding layers) | B12 | 30-50% KV memory reduction | Weeks | Speculative |
| 14 | CUDA 12.8 / CUTLASS 3.x SM120 audit | G8/G9 | 5-15% kernel improvement | Days | Speculative |
| 15 | Nsight Compute profiling of hot kernels | J2-4 | Reveals true bottleneck, guides next optimization | Hours | Proven |

---

## Appendix: Optimization Interaction Matrix

Techniques that compound (stack) with each other:

```
NVFP4 + FusenCache + CUDA graphs + MTP3 + continuous batching  = current stack
  + fused MoE dispatch  (saves launch overhead within the captured graph)
  + expert pruning       (fewer experts = smaller graph, less compute)
  + TP=2                 (doubles bandwidth, independent of all above)
  + C++ server           (eliminates Python, independent of kernel opts)
  + disaggregated        (separates prefill/decode, orthogonal to all above)
```

Techniques that are mutually exclusive or conflicting:

```
CUDA graphs  vs  early exit / dynamic depth  (static vs dynamic execution)
CUDA graphs  vs  torch.compile full-model    (competing graph capture methods)
NVFP4        vs  AQLM / QuIP# / SqueezeLLM  (pick one weight quant scheme)
TP           vs  DP on same GPU pair          (each GPU: full model or partial)
MTP          vs  Medusa heads                 (both predict future tokens, pick one)
FP8 KV       vs  FusenCache K4V4             (pick one KV format per layer)
```

---

*This document covers 120+ techniques across 10 categories. For our specific setup, the top opportunities are: (1) fused MoE dispatch kernel, (2) expert pruning, (3) multi-GPU scaling, (4) C++ server, (5) FusenCache production integration. Everything else is either already deployed, requires model retraining, or has marginal expected impact.*
