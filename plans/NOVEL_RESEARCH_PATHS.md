# Novel Research Paths: Beyond the Known

**Date:** 2026-04-09
**Context:** 75+ techniques explored, 116 mapped, 49 discoveries documented
**Model:** Gemma4 26B-A4B MoE (128 experts, top-8, 30 layers, NVFP4, head_dim 256/512)
**Hardware:** RTX 5090 (SM120, 32GB GDDR7, 1792 GB/s, 48MB L2)
**Current peak:** 6,685 tok/s batch | 128 tok/s single-user | 186 tok/s w/ MTP3
**Bottleneck breakdown:** Attention (FA2) = 63%, MoE grouped GEMM = 30%, other = 7%

**Principle:** These are NOT techniques from papers. These are ideas that emerge from the intersection of THIS specific model, THIS specific hardware, and THIS specific use case (coding workstation). Ranked by `(potential_impact x novelty) / effort`.

---

## Tier 1: The Five Ideas Worth Trying This Week

---

### 1. Expert Output Memoization for Code Patterns

**Score: 9.1** `(impact=9 x novelty=0.95) / effort=0.94`

**What it is:** Code has massive token-level repetition that natural language doesn't: `self.`, `return`, `def __init__`, `for i in range(`, etc. When the same token hits the same expert with similar hidden state input, the expert output is deterministic (weights are frozen). Build a hash table mapping `(expert_id, quantized_input_hash) -> output_tensor`. On cache hit, skip the GEMM entirely.

**Why it might work:** MoE GEMMs are 30% of decode. Code has ~15-25% token repetition within a session (imports, variable names, control flow patterns). If 20% of expert invocations are cache hits, that's 6% system-level speedup. But the real win is at batch level: across 200+ concurrent requests writing Python, the same patterns appear constantly. The expert activation table for `self.` followed by a common method name might have 80%+ hit rate across the batch.

The physics: each expert is a 704x2816 matrix. The input is a 2816-dim hidden state. If we quantize the input to 8-bit and hash it, collisions map to outputs that are within FP4 quantization noise of each other. The error is bounded by `max(|W|) * quantization_error(input)`, which for FP4 weights is already huge -- so approximate matching is fine.

**Why it might NOT work:** (1) Hidden states might be too variable even for similar surface tokens -- the context changes the hidden state significantly. (2) Hash table lookup + comparison might cost more than the small expert GEMM (704x2816 is only ~4M FLOPs). (3) Cache pollution: the table grows unbounded.

**How to test cheaply:** Hook into `MoELayer.forward()`, log `(expert_id, input_tensor[:8])` -- the first 8 elements of each expert input -- for 1000 tokens of code generation. Compute pairwise cosine similarity. If clusters exist with >0.99 cosine sim, memoization works. Cost: 1 hour.

**Expected impact if it works:** 5-15% batch throughput improvement (proportional to hit rate).

**ASI effort to prototype:** 4-6 hours (hook + hash table + bypass logic + benchmark).

---

### 2. Router Prediction Cascade: Predict All 30 Layers from Layer 0

**Score: 8.5** `(impact=8 x novelty=1.0) / effort=0.94`

**What it is:** The model runs 30 MoE layers, each with its own router that selects top-8 from 128 experts. But routing decisions across layers are highly correlated -- if layer 0 routes token X to experts {3,7,12,...}, layers 1-29 likely route to overlapping subsets. Train a tiny MLP (2816 -> 30*128 logits, ~11M params, negligible compute) that takes layer 0's router input and predicts ALL 30 layers' routing decisions simultaneously. Preload only the predicted experts' weights into L2/registers before they're needed.

**Why it might work:** Expert routing is determined by the hidden state, which changes smoothly across layers (residual connections mean each layer adds a small delta). If cross-layer routing correlation is >0.7, we can prefetch the right expert weights 29 layers in advance. The real win isn't skipping compute -- it's eliminating the routing latency and enabling expert weight prefetching. On SM120 with 48MB L2, we could preload 2-3 upcoming experts while computing current ones.

Deeper: the router is just a linear projection `hidden_state @ router_weight`. If hidden states are correlated across layers (they are, by the residual stream), then routing decisions are predictable. This is a provable mathematical relationship, not a heuristic.

**Why it might NOT work:** (1) Routing correlation might be lower than expected -- MoE models are designed to specialize layers. (2) The predictor itself adds latency. (3) Mispredictions mean loading wrong experts, wasting bandwidth. (4) Training the predictor requires a calibration dataset.

**How to test cheaply:** Log all 30 layers' routing decisions for 500 tokens. Compute mutual information between layer 0's routing and each subsequent layer. If MI > 0.5 for most layers, the approach is viable. Cost: 30 minutes.

**Expected impact if it works:** 10-20% decode speedup from expert prefetching + potential to reduce routing overhead.

**ASI effort to prototype:** 6-8 hours (logging + MI analysis + predictor training + integration).

---

### 3. Temporal Expert Caching: MoE Weight Residency Optimization

**Score: 7.8** `(impact=7 x novelty=0.9) / effort=0.81`

**What it is:** Currently all 128 expert weight matrices are in VRAM and loaded on-demand by the MoE dispatch kernel. But expert activation follows a power law -- some experts fire 10x more often than others. Rearrange expert weights in memory so the hottest experts are contiguous and aligned to L2 cache lines. Furthermore, pin the top-K hottest experts in L2 cache permanently using CUDA's `cudaAccessPolicyWindow` or manual prefetch with `cp.async`.

**Why it might work:** RTX 5090 has 48MB L2. Each NVFP4 expert is 704*2816*0.5 = ~1MB. We can fit ~40 experts in L2. If the top-40 experts handle 80%+ of activations (power law), those GEMMs hit L2 instead of GDDR7 -- 4-6x lower latency. The 48MB L2 is the single most underexploited resource on this GPU. We tested L2 persistence for KV cache (no benefit because KV access is sequential), but expert weights have random access patterns where L2 residency matters enormously.

The key insight: we tested L2 persistence for the WRONG data structure. KV cache is streamed sequentially (good GDDR7 access pattern). Expert weights are randomly accessed based on routing (terrible GDDR7 access pattern, perfect L2 use case).

**Why it might NOT work:** (1) `cudaAccessPolicyWindow` might not work on SM120/GDDR7 (it was designed for HBM). (2) L2 associativity might cause eviction regardless of hints. (3) The grouped GEMM kernel might already achieve good L2 hit rates through its tiling.

**How to test cheaply:** Profile expert activation frequency with a counter hook (10 min). Check if the distribution follows a power law (Zipf). Run `ncu --metrics l2__t_sectors_*` on the MoE kernel to see current L2 hit rate. If hit rate < 50%, there's headroom. Cost: 1 hour.

**Expected impact if it works:** 5-15% MoE GEMM speedup (30% of decode -> 1.5-4.5% system).

**ASI effort to prototype:** 4-6 hours.

---

### 4. Speculative Routing: Run Cheap Experts First, Expensive Ones Only If Needed

**Score: 7.5** `(impact=8 x novelty=0.85) / effort=0.91`

**What it is:** Top-8 routing means 8 experts fire per token. But router scores vary wildly -- the top expert might have 10x the score of expert #8. Run the top-4 experts first, compute a partial output, and check if adding experts 5-8 changes the output by more than a threshold. If not, skip them. This is "early exit" applied to expert aggregation rather than layer depth.

**Why it might work:** Router softmax concentrates probability. If the top-4 experts capture 95%+ of the routing weight, experts 5-8 contribute <5% of the output magnitude. For NVFP4 with its inherent quantization noise (~1%), signals below 1% are already lost in the noise floor. So skipping low-weight experts doesn't degrade output quality beyond what quantization already does.

The math: output = sum(router_weight_i * expert_i(x)). If weights 5-8 sum to <0.05, max contribution is 0.05 * max(|expert_output|). For NVFP4 where quantization error is already ~1% of signal, this is within noise.

**Why it might NOT work:** (1) Expert weights might be more uniform than expected (128-expert MoE with top-8 suggests the model needs all 8). (2) Small contributions might accumulate across 30 layers to produce visible quality degradation. (3) The conditional logic adds branching overhead that's hard on GPUs.

**How to test cheaply:** Log router softmax scores for 1000 tokens. Compute the cumulative weight of top-4 vs top-8. If top-4 captures >90% weight on >50% of tokens, proceed. Also: run inference with top-4 forced (zero out experts 5-8), check output quality on 20 prompts. Cost: 1 hour.

**Expected impact if it works:** 25-50% MoE compute reduction on qualifying tokens (up to 15% system-level for batch).

**ASI effort to prototype:** 4-6 hours.

---

### 5. Attention-MoE Pipelining Within a Single Decode Step

**Score: 7.2** `(impact=7 x novelty=0.9) / effort=0.88`

**What it is:** Currently each layer runs attention then MoE sequentially. But attention output for token T at layer L only depends on KV from layer L, while MoE at layer L only depends on attention output from layer L. The key insight: we can START layer L+1's attention while layer L's MoE is still running, IF we forward the attention output (pre-MoE) as a "draft" KV entry. The MoE output then corrects the KV entry when it's ready.

More precisely: split the GPU into two persistent warps. Warp group A runs attention for layer L+1 using the pre-residual hidden state from layer L. Warp group B runs MoE for layer L. When MoE finishes, a small correction kernel updates the KV cache entry. This doubles hardware utilization for memory-bound decode where neither attention nor MoE alone saturates the GPU.

**Why it might work:** Attention is 63% of decode, MoE is 30%. If they can overlap even 50%, that's 15% latency reduction. Both are memory-bandwidth-bound but access different memory regions (KV cache vs expert weights). GDDR7 has multiple channels -- concurrent access to different address ranges achieves near-linear bandwidth scaling. SM120 has 170 SMs -- plenty to partition.

**Why it might NOT work:** (1) The residual connection means layer L+1's input IS the MoE output, not the pre-MoE hidden state. Using the pre-MoE state as a draft introduces error. (2) KV cache correction after MoE completes changes future layers' attention. (3) CUDA stream management overhead might exceed the overlap benefit. (4) We tested stream parallelism before and found 20x slower -- but that was parallelizing within MoE, not across attention/MoE.

**How to test cheaply:** Measure attention-MoE overlap potential by running them on separate CUDA streams for one layer. If wall time < sum of individual times, there's bandwidth headroom for pipelining. Don't implement the correction logic yet -- just measure overlap. Cost: 2 hours.

**Expected impact if it works:** 10-20% single-request decode latency improvement.

**ASI effort to prototype:** 8-12 hours (stream partitioning + correction kernel + integration).

---

## Tier 2: Deep Novel Ideas (Medium Effort, High Novelty)

---

### 6. Codebook Expert Compression: Shared Sub-Expert Basis

**Score: 6.8** `(impact=7 x novelty=0.95) / effort=0.98`

**What it is:** 128 experts with 704x2816 weights each = 128 * 2M = 256M parameters in MoE per layer. But experts aren't independent -- they share the same training distribution. Decompose all 128 expert matrices into a shared codebook of K basis matrices plus per-expert lightweight coefficients: `Expert_i = sum(alpha_ij * Basis_j)` where K << 128. Store only the K basis matrices (in NVFP4) plus 128*K scalar coefficients.

**Why it might work:** Expert weight matrices in MoE models show high cosine similarity (0.85-0.95 between experts in the same layer, per MoE literature). SVD of the stacked expert tensor would reveal the effective rank. If K=16 bases capture 95% of variance, we compress 128 expert loads to 16 basis loads + 128*16 scalar multiplies. Memory bandwidth drops 8x for MoE weights.

**Why it might NOT work:** (1) Expert differentiation might be in the low-variance components that basis decomposition discards. (2) The reconstruction `sum(alpha * basis)` adds compute. (3) NVFP4 quantization might have already destroyed the shared structure.

**How to test cheaply:** Load all 128 expert weights for one layer. Compute SVD of the 128*704 x 2816 stacked matrix. Plot singular value decay. If top-32 singular values capture >90% of Frobenius norm, proceed. Cost: 30 minutes.

**Expected impact:** 4-8x MoE bandwidth reduction -> 20-60% MoE speedup -> 6-18% system.

**ASI effort:** 8-12 hours.

---

### 7. Token-Adaptive Quantization: Different Precision Per Token Position

**Score: 6.5** `(impact=6 x novelty=0.9) / effort=0.83`

**What it is:** Currently all tokens use NVFP4 for all expert GEMMs uniformly. But token importance varies: the first token of a function definition matters more than the 47th token of a string literal. Use the router's confidence score (entropy of softmax) as a proxy for importance. High-entropy tokens (router is uncertain) get BF16 expert computation. Low-entropy tokens (router is confident) stay in NVFP4. This is "mixture of precisions" at the token level, not the expert level.

**Why it might work:** Router entropy correlates with token difficulty. Easy tokens (common patterns) have peaked router distributions and are well-served by FP4. Hard tokens (ambiguous completions) have flat distributions and need precision. In batch serving with 200+ tokens, only ~10-20% are "hard," so the BF16 overhead is small.

**Why it might NOT work:** (1) BF16 experts aren't available -- all weights are in NVFP4. Converting back is lossy. (2) The router entropy might not correlate with actual precision sensitivity. (3) Mixed precision in a batched GEMM requires splitting the batch, which fragments efficiency.

**How to test cheaply:** Log router entropy per token for 1000 tokens. Correlate with generation quality (compare top-1 token match between FP4 and FP16 reference). If high-entropy tokens show higher FP4 vs FP16 divergence, the approach is sound. Cost: 2 hours.

**Expected impact:** 3-8% quality improvement at same throughput, or 10-15% throughput at same quality.

**ASI effort:** 6-8 hours.

---

### 8. Semantic Token Grouping: Batch Tokens by Expert Affinity

**Score: 6.3** `(impact=6 x novelty=0.85) / effort=0.81`

**What it is:** In batch decode, 200+ tokens each independently route to 8 experts. The grouped GEMM kernel must handle arbitrary expert-to-token mappings. But if we SORT tokens by their routing pattern before MoE dispatch, tokens going to the same expert set become contiguous in memory. This transforms scattered expert GEMMs into fewer, larger, contiguous GEMMs with better GPU utilization.

**Why it might work:** Larger GEMMs are more efficient on GPUs (better occupancy, amortize launch overhead). If 200 tokens route to 128 experts with top-8, the average expert handles 200*8/128 = 12.5 tokens. After sorting, each expert's batch is contiguous, enabling coalesced memory access. The sort itself is O(N log N) on GPU -- trivial for N=200.

**Why it might NOT work:** (1) vLLM's CUTLASS MoE kernel might already do internal sorting/padding. (2) Sorting disrupts the token ordering that attention expects (but MoE doesn't care about order). (3) The permute/unpermute adds memory copies.

**How to test cheaply:** Profile the MoE kernel with ncu. Check if the per-expert batch sizes are uniform or highly variable. If variable (some experts get 1 token, others get 30), sorting helps. Cost: 30 minutes.

**Expected impact:** 5-10% MoE throughput improvement in batch mode.

**ASI effort:** 4-6 hours.

---

### 9. KV Cache as Structured Memory: Content-Addressable Attention

**Score: 6.0** `(impact=8 x novelty=1.0) / effort=1.33`

**What it is:** Standard attention computes `softmax(Q @ K^T) @ V` over ALL cached tokens. But for code generation, most of the context is irrelevant to the current token -- when generating `return x + y`, you don't need attention over the import statements 500 tokens ago. Build a content-addressable index over the KV cache using locality-sensitive hashing (LSH). For each query, first retrieve the top-M most relevant KV entries via LSH (O(1) lookup), then run full attention only on those M entries.

**Why it might work:** Attention weight distributions are extremely sparse -- typically 90%+ of attention mass falls on <10% of tokens (this is well-documented for code). If we can identify those 10% before computing attention, we skip 90% of the Q@K^T computation. For sliding window layers (1024 tokens), M=128 would be sufficient. For global layers (unlimited), M=256 could replace attending over thousands.

**Why it might NOT work:** (1) LSH adds overhead that might exceed attention savings for short contexts. (2) Approximate nearest neighbor in high dimensions (256/512) is hard. (3) Attention patterns shift across layers -- the LSH index needs per-layer maintenance. (4) This is essentially learned sparse attention (B10), which we flagged as theoretical.

**How to test cheaply:** Dump attention weights for 100 decode steps (hook into FA2 output). Compute the number of entries capturing 95% of attention mass per head per layer. If the median is <15% of context length, the approach has potential. Cost: 2 hours.

**Expected impact:** 30-60% attention speedup at long contexts -> 18-36% system-level at long context.

**ASI effort:** 16-24 hours (LSH index + custom attention kernel + quality validation).

---

### 10. Cross-Request Expert State Sharing

**Score: 5.8** `(impact=7 x novelty=0.85) / effort=1.02`

**What it is:** In batch serving, multiple requests often need the same expert at the same layer in the same decode step. Currently the grouped GEMM handles this by batching tokens per expert. But go further: if request A and request B both route to expert 7 at layer 12, and their input hidden states are similar (both writing Python class methods), the expert OUTPUTS will be similar. Compute the expert once for the centroid of similar inputs, then distribute a corrected version to each request.

**Why it might work:** Expert computation is `y = W2 * gelu(W1 * x)`. If x_A and x_B have cosine similarity > 0.99, then |y_A - y_B| < epsilon * ||W||. For FP4 weights, epsilon is already ~1% of signal. Computing the expert once instead of twice saves 50% compute for that pair.

At batch size 200 with 128 experts and top-8, each expert handles ~12.5 tokens. Clustering those 12.5 inputs by similarity and computing one representative + correction could reduce effective expert compute by 30-50%.

**Why it might NOT work:** (1) Hidden states might not cluster well (different context histories). (2) Centroid computation and correction add overhead. (3) The grouped GEMM is already efficient -- the overhead of clustering might exceed the GEMM savings.

**How to test cheaply:** During batch decode, log hidden state inputs to each expert. Compute within-expert cosine similarity matrix. If >20% of pairs have similarity >0.95, clustering is viable. Cost: 1 hour.

**Expected impact:** 10-20% MoE compute reduction in high-batch scenarios.

**ASI effort:** 8-12 hours.

---

### 11. Persistent Expert Kernel: Never Evict Hot Experts from Registers

**Score: 5.6** `(impact=6 x novelty=0.85) / effort=0.91`

**What it is:** Write a persistent CUDA kernel that pins the top-N most frequently activated experts' weights in shared memory/registers permanently. Instead of loading expert weights from GDDR7 on every forward pass, the kernel stays resident and accepts work items (input tensors) via a work queue in global memory. This eliminates all weight-loading latency for hot experts.

**Why it might work:** SM120 has 228KB shared memory per SM. One NVFP4 expert (704*2816*0.5 bytes = ~1MB) doesn't fit in shared memory. BUT: we can tile the expert GEMM so that hot tiles stay resident. If the GEMM is broken into 32 tiles, each tile is ~32KB, fitting in shared memory with room for the input/output buffers. A persistent kernel that processes the same tile across many tokens avoids reloading.

**Why it might NOT work:** (1) Persistent kernels that hold shared memory prevent other kernels from using those SMs. (2) 170 SMs * 228KB = 38MB total shared memory, but each expert needs ~1MB, so we can only pin ~38 experts across all SMs. (3) Work queue management has latency.

**How to test cheaply:** Benchmark a simple persistent GEMM kernel vs a launch-per-call GEMM for the exact expert shape (704x2816). If the persistent version is faster, the concept works. Cost: 4 hours.

**Expected impact:** 10-30% MoE latency reduction for hot experts.

**ASI effort:** 8-12 hours.

---

### 12. AST-Guided Token Prediction: Structural Code Generation

**Score: 5.5** `(impact=7 x novelty=0.95) / effort=1.21`

**What it is:** Code has rigid syntactic structure. After generating `def foo(`, the next tokens MUST be parameter definitions ending with `):`. After `if `, the next tokens MUST form a boolean expression ending with `:`. Build an AST constraint engine that, given the current partial parse state, restricts the vocabulary to syntactically valid continuations. This doesn't change the model -- it constrains the sampling to skip impossible tokens.

But go further: use the AST state to predict likely token SEQUENCES, not just individual tokens. After `for `, the most likely patterns are `i in range(`, `item in `, `key, value in `. These multi-token patterns can be proposed as speculative sequences and verified in one forward pass -- giving us speculative decoding WITHOUT a draft model.

**Why it might work:** Code is ~10x more predictable than natural language at the token level. A tree-sitter parser running on CPU can identify valid next-token sets in microseconds. If the AST-based sequence predictor has >60% acceptance rate, it beats n-gram speculative decoding (which we measured at <40% for code).

**Why it might NOT work:** (1) The model might need to "think" through invalid tokens before reaching valid ones (chain-of-thought). (2) Vocabulary restriction might hurt diversity/quality. (3) Building a robust incremental parser for all languages is complex.

**How to test cheaply:** Use tree-sitter to parse the generated tokens in real-time. Count how often the model's top-1 token is in the syntactically valid set. If >95%, the constraint engine is free. Count how often the top multi-token AST pattern matches: if >50%, we get speculative decode for free. Cost: 3 hours.

**Expected impact:** 1.5-2.5x single-user decode speed for structured code generation.

**ASI effort:** 12-16 hours (tree-sitter integration + pattern database + speculative verification).

---

### 13. GDDR7 Channel-Aware Memory Allocation

**Score: 5.3** `(impact=5 x novelty=0.9) / effort=0.85`

**What it is:** RTX 5090 has 28 Gbps GDDR7 across multiple memory channels (likely 8-12 channels based on 448-bit bus). Each channel serves a specific address range independently. Currently expert weights are allocated by PyTorch's caching allocator with no awareness of channel mapping. If expert A and expert B's weights both reside on the same memory channel, concurrent access serializes. Allocate expert weights so frequently co-active experts land on DIFFERENT channels.

**Why it might work:** Top-8 routing means 8 experts are loaded simultaneously per token. If those 8 experts' weights are spread across 8 different memory channels, each channel serves one expert independently at full bandwidth. If they collide on 2-3 channels, effective bandwidth drops by 2-4x for those experts.

GDDR7 channels operate at 28 Gbps each. With 8 channels, theoretical peak = 224 GB/s per channel. If 8 experts hit 8 channels vs 4 channels, peak bandwidth is 224 GB/s vs 112 GB/s per expert.

**Why it might NOT work:** (1) GDDR7 channel mapping is not exposed by CUDA APIs -- we'd need to reverse-engineer it. (2) The memory controller likely interleaves at fine granularity (256-byte lines), making channel placement unpredictable. (3) The L2 cache already hides most channel-level contention.

**How to test cheaply:** Allocate pairs of tensors at controlled address offsets and benchmark concurrent access throughput. If throughput varies with offset (suggesting channel boundaries), the approach is viable. Cost: 2 hours.

**Expected impact:** 5-15% MoE bandwidth improvement.

**ASI effort:** 6-8 hours.

---

### 14. Generation-Time Code Execution Feedback Loop

**Score: 5.0** `(impact=8 x novelty=1.0) / effort=1.60`

**What it is:** When generating code, execute partially complete functions in a sandboxed Python interpreter on CPU (in parallel with GPU inference). Use execution results to modify sampling: if `x = compute_something()` was executed and returned `42`, bias the next tokens toward patterns consistent with `x=42` (e.g., `if x > 0:` becomes more likely than `if x < 0:`). This is "speculative execution" -- using runtime information to guide generation.

**Why it might work:** Code generation quality is often limited by the model's inability to "run" the code mentally. Humans write code by alternating between writing and testing. If we provide execution feedback, the model can generate more correct code on the first try, reducing the need for regeneration (which is the ultimate throughput killer at the application level).

For a coding workstation, the CPU is idle during GPU inference. Running partial code on 32 CPU cores has zero GPU cost.

**Why it might NOT work:** (1) Partially complete code usually can't be executed without stubs/mocks. (2) Execution latency might exceed token generation latency (the model generates tokens faster than code runs). (3) Feeding execution results back into the model requires careful prompt engineering.

**How to test cheaply:** Generate 100 Python functions. For each, attempt to execute the first N lines after each line is generated. Measure: (a) what fraction of partial generations are executable, (b) whether execution results could have prevented later errors. Cost: 3 hours.

**Expected impact:** 2-5x effective throughput (fewer regeneration cycles) at the APPLICATION level, not the model level.

**ASI effort:** 16-24 hours.

---

### 15. Bidirectional Expert Weight Streaming

**Score: 4.8** `(impact=5 x novelty=0.85) / effort=0.89`

**What it is:** Currently expert weights are loaded from GDDR7 into L2/registers, used for one GEMM, then evicted. For decode with long sequences, the same experts are needed every decode step (every ~8ms). The weights are ~256MB total per layer. Instead of pulling weights in and evicting them, keep streaming weights in a circular buffer: as layer L finishes with expert weights, don't evict -- keep them warm for the next decode step's layer L. This requires double-buffering: while step N uses buffer A for layer L, preload buffer B with layer L's weights for step N+1.

**Why it might work:** The decode loop is highly periodic. The same 30 layers process the same routing patterns repeatedly. If we can keep expert weights warm across decode steps (not just across tokens within a step), we avoid reloading 256MB per layer per step. With 48MB L2, we can keep ~3 layers' hot experts warm.

**Why it might NOT work:** (1) Expert routing changes between decode steps, so different experts are needed. (2) 48MB L2 for 3 layers out of 30 is only 10% coverage. (3) The L2 replacement policy may not cooperate.

**How to test cheaply:** Measure expert routing stability across consecutive decode steps. If >70% of experts at each layer are the same between step N and step N+1, weight reuse is high. Cost: 1 hour.

**Expected impact:** 3-8% decode latency reduction.

**ASI effort:** 6-8 hours.

---

## Tier 3: Moonshot Ideas (High Novelty, Less Certain)

---

### 16. Learned Expert Bypass: Train a Tiny "Should I Skip This Layer" Head

**Score: 4.5** `(impact=8 x novelty=0.9) / effort=1.60`

**What it is:** Not all 30 layers contribute equally to every token. For "easy" tokens (common code patterns), some layers apply near-identity transformations (output = input + epsilon). Train a 1-layer MLP per layer (2816->1 scalar) that predicts `||MoE_output|| / ||input||`. If the predicted ratio is < threshold (layer would barely change the hidden state), skip the entire layer. This is early-exit applied per-token-per-layer, without modifying the model.

**Why it might work:** Residual networks are known to have variable effective depth -- some inputs need all layers, others converge early. For a 30-layer MoE model, if the average token needs only 20 layers, that's 33% compute reduction. The bypass predictor is ~2816 FLOPs per layer -- negligible vs the 47.6M FLOP MoE.

**Why it might NOT work:** (1) Skipping any layer changes all subsequent layers' inputs (error accumulates). (2) The predictor needs training data (run full model, log layer contributions). (3) For MoE models, even "small" layer contributions might encode critical routing information.

**How to test cheaply:** Run full inference for 1000 tokens, logging `||layer_output - layer_input|| / ||layer_input||` for each layer. If >30% of (token, layer) pairs have ratio < 0.01, layer skipping is viable. Cost: 1 hour.

**Expected impact:** 20-40% compute reduction for easy tokens.

**ASI effort:** 12-16 hours (data collection + predictor training + integration + quality validation).

---

### 17. SM120 Tensor Core Misuse: Non-GEMM Tensor Operations

**Score: 4.3** `(impact=5 x novelty=1.0) / effort=1.16`

**What it is:** SM120 tensor cores run at 838+ TFLOPS for FP4. We use them exclusively for matrix multiply. But tensor cores can compute ANY operation expressible as `D = A @ B + C` with appropriate encoding. Specifically: (1) Use tensor cores for softmax normalization by encoding the reduction as a matrix multiply against a ones vector. (2) Use tensor cores for RMSNorm by encoding the sum-of-squares as a dot product. (3) Use tensor cores for the router's argmax by encoding comparisons as saturating matrix ops.

**Why it might work:** These operations are currently running on CUDA cores at ~50 TFLOPS. If we can express them as tensor core operations, we get 16x throughput (838/50 TFLOPS). Even with the encoding overhead, 4-8x speedup on these operations is plausible. These ops are in the "7% other" category but every bit counts when the big components are optimized.

**Why it might NOT work:** (1) The encoding overhead might exceed the compute savings. (2) FP4 tensor cores only support specific data types and accumulator modes. (3) These operations involve reductions that don't map naturally to matrix multiply.

**How to test cheaply:** Write a 50-line kernel that computes vector norm using tensor core MMA instructions. Benchmark against a standard CUDA core reduction. Cost: 3 hours.

**Expected impact:** 2-4% system-level from accelerating non-GEMM operations.

**ASI effort:** 8-12 hours.

---

### 18. Request-Aware Expert Batching: Scheduling for MoE Efficiency

**Score: 4.2** `(impact=6 x novelty=0.8) / effort=1.14`

**What it is:** vLLM's scheduler batches requests by arrival order. But MoE efficiency depends on expert utilization -- if all tokens in a batch route to the same 8 experts, the grouped GEMM is maximally efficient (one large GEMM per expert). If they route to all 128, it's 128 tiny GEMMs. Modify the scheduler to GROUP requests by predicted expert affinity: queue requests writing similar code types together.

**Why it might work:** Users on a coding workstation tend to work in streaks: 10 minutes of Python, 20 minutes of Rust, etc. Requests from the same user in the same file share vocabulary, style, and expert routing patterns. Batching similar requests together increases per-expert batch size from ~12 to ~25-50, dramatically improving GEMM efficiency.

**Why it might NOT work:** (1) Predicting expert routing from request content without running the model is chicken-and-egg. (2) Grouping by similarity might increase latency for "outlier" requests that don't match any group. (3) The effect might be small if expert utilization is already uniform.

**How to test cheaply:** In batch serving, log per-request expert routing patterns. Compute a clustering metric: what's the effective number of distinct experts used per batch? If it varies widely (some batches use 30 experts, others use 100), routing-aware scheduling helps. Cost: 1 hour.

**Expected impact:** 5-15% batch throughput improvement.

**ASI effort:** 8-12 hours.

---

### 19. FP4 Residual Correction Network

**Score: 4.0** `(impact=6 x novelty=0.95) / effort=1.43`

**What it is:** NVFP4 quantization introduces systematic errors that correlate with input magnitude and distribution. Train a tiny correction network (one linear layer, 2816->2816, ~8M params in FP16) that predicts and corrects the quantization error AFTER each MoE layer. The correction network runs on tensor cores in FP16 while the next layer's routing runs on CUDA cores -- zero added latency if pipelined.

**Why it might work:** Quantization error is not random -- it's a deterministic function of the input. A linear correction can capture the first-order error pattern (the difference between FP4 GEMM output and FP16 GEMM output). If 80% of quantization error is first-order, a single linear layer recovers most quality.

This is different from QAT (which requires retraining the full model) or calibration (which adjusts scales). This is a POST-HOC correction that can be trained in minutes on a calibration dataset by minimizing `||FP16_output - FP4_output - Correction(FP4_output)||`.

**Why it might NOT work:** (1) The correction might overfit to the calibration distribution. (2) An 8M param FP16 layer adds 16MB per layer x 30 layers = 480MB VRAM. (3) The error might be high-order (nonlinear) and not capturable by a linear layer.

**How to test cheaply:** Run 100 tokens through both FP4 and FP16 model paths. Compute the per-layer error `FP16_out - FP4_out`. Fit a linear regression `error = W @ FP4_out`. Measure R^2. If R^2 > 0.5, the correction works. Cost: 2 hours.

**Expected impact:** 5-15% quality improvement at same compute, or equivalent to running a larger model.

**ASI effort:** 10-14 hours.

---

### 20. Compile-Time Expert Fusion: Static Routing for Common Prefixes

**Score: 3.8** `(impact=6 x novelty=0.9) / effort=1.42`

**What it is:** For common code prefixes (imports, class definitions, function signatures), the routing decisions are nearly deterministic. Pre-compute the routing for the top-1000 most common token sequences (based on a code corpus analysis). For these sequences, generate FUSED expert kernels where the routing is baked in at compile time -- no router computation, no permute/scatter, just a straight GEMM with the pre-selected 8 expert weights concatenated.

**Why it might work:** Eliminating routing overhead + permute/unpermute for known sequences removes the entire MoE dispatch machinery. For prefill of common code patterns (which might be 40-60% of all prefill tokens), this is a dedicated fast path. The compiled kernels can be heavily optimized for the specific expert combination.

**Why it might NOT work:** (1) 1000 sequences x 30 layers x many expert combos = enormous number of fused kernels to compile. (2) Routing may depend on context, not just the token, making pre-computation invalid. (3) Maintaining the kernel cache adds complexity.

**How to test cheaply:** Profile routing decisions for the top-100 most common 4-grams in code. Measure routing entropy: if entropy < 1 bit for >50% of common patterns, static routing is viable. Cost: 2 hours.

**Expected impact:** 10-20% prefill speedup for code workloads.

**ASI effort:** 12-16 hours.

---

### 21. Approximate Attention via Learned Projection

**Score: 3.7** `(impact=7 x novelty=0.85) / effort=1.61`

**What it is:** Replace the full `softmax(QK^T)V` attention with a learned low-rank approximation: `phi(Q) @ phi(K)^T @ V` where phi is a random feature map that approximates the softmax kernel. This converts O(N^2) attention to O(N*d) where d is the feature dimension. Unlike generic linear attention (which requires retraining), use a calibrated random feature map that approximates THIS model's attention patterns.

**Why it might work:** For decode with seq_len=1, attention is already O(N) so this helps only for prefill. But for sliding window attention with window=1024, even decode requires 1024 dot products. A feature map with d=64 reduces this to 64 dot products -- 16x reduction. Attention is 63% of decode, so 15x attention speedup = 10x system speedup (theoretical ceiling).

**Why it might NOT work:** (1) Approximation error degrades generation quality. (2) The feature map must be calibrated per model. (3) For head_dim=256/512, the random features need to be in the same dimension, limiting compression. (4) Previous work (Performer, etc.) showed quality gaps.

**How to test cheaply:** Implement random Fourier features approximation to softmax attention. Measure attention output error (Frobenius norm of difference) for varying feature dimensions d. Find the d where error < FP4 quantization noise. Cost: 4 hours.

**Expected impact:** 2-8x attention speedup (theoretical), likely 1.3-2x practical.

**ASI effort:** 12-16 hours.

---

### 22. GPU-CPU Expert Offloading with Predictive Prefetch

**Score: 3.5** `(impact=6 x novelty=0.7) / effort=1.20`

**What it is:** Move cold experts (bottom 50% by activation frequency) to CPU memory (DDR5, ~50 GB/s). Use the router prediction cascade (idea #2) to prefetch needed cold experts 2-3 layers ahead via PCIe 5.0 (64 GB/s). Hot experts (top 50%) stay in VRAM permanently. This frees ~7GB of VRAM for larger batch sizes or longer contexts.

**Why it might work:** Expert activation follows a power law. If top-64 experts handle 90% of activations, the bottom-64 are loaded <10% of the time. Offloading them to CPU frees 128MB/layer * 30 layers * 50% = ~1.9GB (NVFP4) or more if we store CPU-side experts in higher precision for quality. PCIe 5.0 can transfer a 1MB expert in 16us -- within the MoE layer latency budget if prefetched.

**Why it might NOT work:** (1) PCIe 5.0 latency (~2us per transfer) adds up for multiple experts. (2) Prefetch misprediction means stalling on PCIe transfer. (3) CPU memory bandwidth (50 GB/s) is 36x slower than GDDR7, so any fallback is expensive.

**How to test cheaply:** Profile expert activation frequency (idea #3's test). Measure PCIe transfer time for 1MB tensor. If P99 expert miss rate * PCIe latency < 5% of decode time, proceed. Cost: 2 hours.

**Expected impact:** 5-10% more batch capacity (from freed VRAM) at <2% latency cost.

**ASI effort:** 10-14 hours.

---

### 23. Inter-Step KV Delta Compression

**Score: 3.3** `(impact=4 x novelty=0.9) / effort=1.09`

**What it is:** Between decode steps, only ONE new KV entry is added per layer. The rest of the cache is unchanged. But FusenCache re-reads the entire KV cache for attention every step. Instead, maintain a delta buffer: store only the changes since the last step. For sliding window attention, this means the new K/V entry plus the evicted entry. The attention kernel can process "cached attention scores + new entry's contribution" instead of recomputing everything.

**Why it might work:** For window=1024, we recompute 1024 dot products to add 1 new entry. If we cached the partial softmax state (running sum and max), we could update with O(1) work instead of O(1024). This is the online softmax trick applied ACROSS decode steps rather than within a single attention pass.

**Why it might NOT work:** (1) The softmax denominator changes when a new entry is added, requiring rescaling ALL previous attention weights. (2) Numerical stability of incremental softmax across many steps degrades. (3) The O(1024) attention is already near memory-bandwidth optimal in FA2.

**How to test cheaply:** Implement incremental softmax update for a single head/layer. Compare numerical output vs full recomputation. If error stays below 1e-3 over 1000 steps, proceed. Cost: 3 hours.

**Expected impact:** 10-30% attention latency reduction for sliding window layers.

**ASI effort:** 8-12 hours.

---

### 24. Warp-Specialized Attention-MoE Coprocessor

**Score: 3.2** `(impact=7 x novelty=0.95) / effort=2.08`

**What it is:** Write a single mega-kernel that processes an ENTIRE decoder layer (attention + MoE + residual + norm) without returning to the CPU/Python scheduler between operations. Warps 0-63 run attention, warps 64-127 run MoE, warps 128-170 run normalization. Communication via shared memory barriers, not global memory. This eliminates ALL kernel launch overhead and intermediate tensor materialization within a layer.

**Why it might work:** Current inference launches 8-15 kernels per layer (norm, QKV proj, attention, output proj, router, permute, expert GEMM, unpermute, residual). Each launch costs ~5us on SM120 = 75-112us per layer = 2.25-3.4ms for 30 layers. At 128 tok/s decode (7.8ms per token), launch overhead is 30-43% of decode time. A mega-kernel eliminates this entirely.

**Why it might NOT work:** (1) The kernel would be enormously complex -- managing 170 SMs with different roles via shared memory. (2) Different operations have different optimal tile sizes, making unified scheduling difficult. (3) Debug/maintenance nightmare. (4) CUDA compiler might not handle a kernel this complex.

**How to test cheaply:** Measure actual kernel launch overhead with nsys for one decode step. If launch overhead is >15% of decode time, the mega-kernel approach has clear ROI. Also: implement a simpler fusion (attention + RMSNorm) as a proof of concept. Cost: 4 hours.

**Expected impact:** 15-40% decode latency reduction.

**ASI effort:** 24-40 hours.

---

### 25. Entropy-Adaptive Decode: Dynamic Token Batch Size

**Score: 3.0** `(impact=5 x novelty=0.8) / effort=1.33`

**What it is:** When the model's output distribution has low entropy (highly confident prediction), the next token is almost deterministic. Skip computing the full vocabulary logits -- just compute the dot product of the hidden state with the top-K most likely token embeddings (based on a running frequency table). For high-entropy outputs (uncertain), compute full logits as usual.

**Why it might work:** The final LM head projects 2816 -> 262,144 (vocab size). This is a 2816 x 262,144 matrix multiply -- 738M FLOPs per token. If 60% of tokens are "easy" (top-1 confidence > 0.9), we could compute 2816 x 1000 = 2.8M FLOPs instead -- 260x reduction for those tokens. Even at 40% easy tokens, average LM head cost drops by 100x * 0.4 = 40x.

**Why it might NOT work:** (1) The LM head is not the bottleneck (it's in the "7% other" category). (2) Determining which tokens are "easy" requires computing the output distribution -- chicken-and-egg. (3) The embedding matrix is shared with the model input, so it's already in memory.

**How to test cheaply:** Profile LM head time as fraction of decode. Log output entropy for 1000 tokens. If LM head is >3% AND >50% of tokens have entropy < 1 bit, proceed. Cost: 1 hour.

**Expected impact:** 2-5% decode latency reduction.

**ASI effort:** 6-8 hours.

---

### 26. Hardware Clock Exploitation: Boost Clock Pinning + Thermal Management

**Score: 2.8** `(impact=3 x novelty=0.7) / effort=0.32`

**What it is:** RTX 5090 boost clock varies with thermal state and power draw. During batch serving, clock speed fluctuates 5-15% based on which kernels are running (tensor core heavy = power limited, memory heavy = thermal limited). Pin the GPU to a stable high clock by: (1) Underclock slightly to prevent thermal throttling. (2) Use nvidia-smi to lock clocks. (3) Improve airflow/cooling for the workstation. (4) Profile exact clock behavior under our workload.

**Why it might work:** Consistent 2.5 GHz is better than oscillating 2.3-2.7 GHz because branch predictors, cache prefetchers, and pipeline scheduling all work better with stable timing. vLLM's CUDA graph timings are calibrated at one clock speed; if the clock drops during replay, the overlapping is wrong.

**Why it might NOT work:** (1) 3% improvement ceiling. (2) Locking clocks low to avoid throttling might reduce peak performance. (3) Modern GPU power management is already quite good.

**How to test cheaply:** `nvidia-smi -lgc 2400,2400` to lock clocks. Benchmark. Compare against default. Cost: 10 minutes.

**Expected impact:** 1-3% throughput consistency improvement.

**ASI effort:** 30 minutes.

---

### 27. Multi-Query Generation: Parallel Function Completion

**Score: 2.7** `(impact=6 x novelty=0.85) / effort=1.89`

**What it is:** For a coding workstation, the user often needs multiple related completions: all methods of a class, all test cases for a function, all variants of an API handler. Instead of generating them sequentially, generate all N completions in parallel as a single batch, sharing the common prefix KV cache. The shared prefix (class definition, function signature) is computed once; each completion branch diverges from the same KV state.

**Why it might work:** Shared prefix = computed once instead of N times. With N=8 parallel completions sharing a 500-token prefix, we save 7 * 500 tokens of prefill compute. At batch level, the 8 branches are just 8 entries in the decode batch -- efficient on our already-optimized batch pipeline.

**Why it might NOT work:** (1) This is just prefix caching + batching, which vLLM already supports. The "novelty" is in the application-layer orchestration, not the inference engine. (2) Users rarely need 8+ parallel completions simultaneously.

**How to test cheaply:** Use vLLM's batch API to send 8 requests with the same prefix. Measure throughput vs 8 sequential requests. Cost: 30 minutes.

**Expected impact:** 3-8x throughput for multi-completion workloads.

**ASI effort:** 4-6 hours (application layer, minimal inference changes).

---

### 28. Reverse Token Verification: Generate Then Verify Backwards

**Score: 2.5** `(impact=5 x novelty=1.0) / effort=2.50`

**What it is:** Generate a block of K tokens forward as usual, then run the model backward (feed the generated tokens as input) and check if the model's forward pass on the complete sequence assigns high probability to each generated token. Tokens with low backward probability are likely errors -- regenerate just those positions. This is self-consistency checking without a verifier model.

**Why it might work:** Self-consistency is a strong quality signal. If token 47 seems wrong in retrospect (low probability given the full context including tokens 48-50), it probably IS wrong. Catching and fixing one error early prevents cascading errors that waste hundreds of tokens. At the application level, this improves effective throughput by reducing regeneration requests.

**Why it might NOT work:** (1) The backward pass costs one full forward pass per K-token block -- 2x compute. (2) Token probability depends on position; backward verification might flag correct but surprising tokens. (3) Regenerating individual positions in an autoregressive model is not straightforward.

**How to test cheaply:** Generate 100 code blocks (50 tokens each). For each, run a verification forward pass on the complete output. Compute per-token log probability. Flag tokens below the 5th percentile. Have a human check: are flagged tokens actually errors? Cost: 2 hours.

**Expected impact:** 20-40% fewer regeneration requests (application-level throughput).

**ASI effort:** 8-12 hours.

---

## Summary Ranking

| Rank | Idea | Score | Impact | Novelty | Effort (hrs) | Quick Test |
|------|------|-------|--------|---------|-------------|------------|
| 1 | Expert Output Memoization | 9.1 | 9 | 0.95 | 4-6 | Log expert input similarity |
| 2 | Router Prediction Cascade | 8.5 | 8 | 1.0 | 6-8 | Measure cross-layer MI |
| 3 | Temporal Expert Caching (L2) | 7.8 | 7 | 0.9 | 4-6 | Profile activation frequency |
| 4 | Speculative Routing (top-4 early exit) | 7.5 | 8 | 0.85 | 4-6 | Log router softmax scores |
| 5 | Attention-MoE Pipelining | 7.2 | 7 | 0.9 | 8-12 | Measure stream overlap |
| 6 | Codebook Expert Compression | 6.8 | 7 | 0.95 | 8-12 | SVD on stacked experts |
| 7 | Token-Adaptive Quantization | 6.5 | 6 | 0.9 | 6-8 | Router entropy analysis |
| 8 | Semantic Token Grouping | 6.3 | 6 | 0.85 | 4-6 | Profile per-expert batch sizes |
| 9 | Content-Addressable KV Cache | 6.0 | 8 | 1.0 | 16-24 | Dump attention weight sparsity |
| 10 | Cross-Request Expert Sharing | 5.8 | 7 | 0.85 | 8-12 | Hidden state clustering |
| 11 | Persistent Expert Kernel | 5.6 | 6 | 0.85 | 8-12 | Persistent vs launch GEMM bench |
| 12 | AST-Guided Token Prediction | 5.5 | 7 | 0.95 | 12-16 | Tree-sitter parse match rate |
| 13 | GDDR7 Channel-Aware Allocation | 5.3 | 5 | 0.9 | 6-8 | Address-offset BW test |
| 14 | Speculative Code Execution | 5.0 | 8 | 1.0 | 16-24 | Partial execution feasibility |
| 15 | Bidirectional Expert Streaming | 4.8 | 5 | 0.85 | 6-8 | Expert routing stability |
| 16 | Learned Expert Bypass | 4.5 | 8 | 0.9 | 12-16 | Log layer contribution norms |
| 17 | Tensor Core Non-GEMM Ops | 4.3 | 5 | 1.0 | 8-12 | TC vector norm benchmark |
| 18 | Request-Aware Expert Batching | 4.2 | 6 | 0.8 | 8-12 | Expert diversity per batch |
| 19 | FP4 Residual Correction Network | 4.0 | 6 | 0.95 | 10-14 | Linear regression on FP4 error |
| 20 | Compile-Time Expert Fusion | 3.8 | 6 | 0.9 | 12-16 | Routing entropy for common n-grams |
| 21 | Approximate Attention via Projection | 3.7 | 7 | 0.85 | 12-16 | Random Fourier feature error |
| 22 | GPU-CPU Expert Offloading | 3.5 | 6 | 0.7 | 10-14 | PCIe transfer benchmark |
| 23 | Inter-Step KV Delta Compression | 3.3 | 4 | 0.9 | 8-12 | Incremental softmax stability |
| 24 | Warp-Specialized Mega-Kernel | 3.2 | 7 | 0.95 | 24-40 | Measure launch overhead |
| 25 | Entropy-Adaptive Decode | 3.0 | 5 | 0.8 | 6-8 | Profile LM head % + entropy |
| 26 | Boost Clock Pinning | 2.8 | 3 | 0.7 | 0.5 | nvidia-smi clock lock test |
| 27 | Multi-Query Generation | 2.7 | 6 | 0.85 | 4-6 | vLLM batch prefix sharing |
| 28 | Reverse Token Verification | 2.5 | 5 | 1.0 | 8-12 | Backward pass probability audit |

---

## Immediate Action Plan: 48 Hours of Testing

### Hour 0-2: Four Diagnostic Measurements (In Parallel)

These four tests are cheap and gate ALL subsequent decisions:

1. **Expert activation frequency distribution** (gates #3, #11, #15, #22)
   - Hook MoE layers, count per-expert activations for 1000 tokens
   - Expected: power law with top-40 experts handling 80%+ of traffic

2. **Router softmax concentration** (gates #4, #7)
   - Log router softmax scores per token for 1000 tokens
   - Expected: top-4 experts capture >90% weight on >50% of tokens

3. **Cross-layer routing correlation** (gates #2)
   - Log all 30 layers' top-8 expert IDs for 500 tokens
   - Compute pairwise Jaccard similarity between consecutive layers

4. **Expert input similarity** (gates #1, #10)
   - For top-5 most active experts, log input hidden states
   - Compute cosine similarity matrix within each expert
   - Expected: clusters of similar inputs (especially for code)

### Hour 2-4: Act on Measurements

- If power law holds -> implement temporal expert caching (#3)
- If router concentration high -> implement speculative routing (#4)
- If cross-layer correlation high -> implement router cascade (#2)
- If input similarity high -> implement expert memoization (#1)

### Hour 4-8: Validate Winners, Start Tier 2

- Benchmark winning approaches from hour 2-4
- Start SVD analysis for codebook compression (#6) -- runs on CPU, parallel with GPU benchmarks
- Start AST-guided prediction feasibility study (#12) -- pure CPU work

### Hour 8-48: Compound and Integrate

- Combine compatible winning approaches
- Start medium-effort prototypes for top-scoring ideas that passed measurement gates
- Document failures with specific measurements for future reference

---

## What Would a 2030 System Do Differently?

### Hardware That Changes Everything
- **HBM4 (2027):** 2 TB/s bandwidth eliminates MoE weight loading as bottleneck. Expert memoization (#1) becomes less valuable, but expert parallelism across a larger memory pool becomes critical.
- **CXL 3.0 (2027):** Pooled memory across GPUs. Expert offloading (#22) gets 300 GB/s instead of 64 GB/s. Cold experts on CXL are nearly as fast as hot experts on GDDR.
- **Processing-in-Memory (2028):** GEMM computed at the memory die. Expert computation happens where the weights live -- zero data movement. Eliminates bandwidth bottleneck entirely.
- **Photonic interconnects (2029):** Inter-GPU communication at speed of light. Expert parallelism across 8+ GPUs with negligible latency.

### Architecture That Changes Everything
- **Native MoE-aware hardware:** Future GPUs might have dedicated MoE dispatch units (like tensor cores for GEMM), with hardware routing, permute, and gather-scatter.
- **Sparse tensor cores:** 2:4 sparsity is SM80+. Future GPUs might support arbitrary sparsity patterns, making learned expert bypass (#16) a hardware operation.
- **In-SRAM transformers:** If SRAM scales to 512MB (10x current), entire small models fit in on-chip memory. The 9B distilled model could run entirely from SRAM at 10x bandwidth.

### Software That Changes Everything
- **Compiler-generated megakernels (#24):** Future Triton or MLIR compilers might automatically fuse entire decoder layers. What takes 40 hours to hand-write becomes a compiler flag.
- **Learned compilation:** ML models that optimize ML model inference -- the compiler uses gradient descent to find optimal schedules. AutoKernel v3.
- **Neural architecture search for inference:** Design MoE models that are optimized for inference on specific hardware from the start, not adapted post-training.

### The Fundamental Insight

Every optimization we've done treats the model as a fixed black box. The 2030 system co-designs the model and the inference engine: the model IS the kernel. Experts are hardware-native computational units. Attention IS a memory lookup. The distinction between "model" and "hardware" dissolves.

The first step toward that future is idea #6 (codebook expert compression): decomposing experts into hardware-friendly basis elements is the beginning of co-design.
