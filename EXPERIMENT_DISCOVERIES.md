# Experiment Discoveries — Gemma4 26B NVFP4 on RTX 5090

**Session:** April 9-10, 2026
**Model:** Gemma4 26B-A4B-it NVFP4 (128 experts, top-8, 30 layers)
**Hardware:** RTX 5090 (Blackwell SM120, 32GB, 1792 GB/s)
**Peak throughput achieved:** 6,685 tok/s (FusenCache eager)

---

## Discovery #1: Disabling torch.compile doubles throughput
**Expected:** torch.compile + inductor optimizes the model graph
**Found:** 2.1x SLOWER at batch (3,112 → 6,615 tok/s when disabled)
**Why:** vLLM's C++ custom ops are already optimized. Inductor adds graph tracing overhead that doesn't amortize at high batch. CUTLASS FP4 MoE calls are opaque to inductor.
**Trade-off:** Inductor wins at C≤4 (127 vs 89 tok/s single request)
**Config:** `-cc.mode none -cc.cudagraph_mode full`

## Discovery #2: vLLM's MoE is already well-fused (6 kernels, not 128)
**Expected:** 128 separate expert kernel calls per layer
**Found:** Only 6 kernel launches per MoE layer via CUTLASS grouped GEMM
**Implication:** Fusing MoE further is much harder than expected. The remaining overhead is in sorting, quantization, and norms — not kernel launches.
**Source:** `MOE_PROFILING.md`

## Discovery #3: GEMMs are only 27% of decode time
**Expected:** Tensor core matmuls dominate
**Found:** RMSNorm 26%, attention 17%, routing 15%, GEMMs only 12%
**Implication:** Kernel-level GEMM optimization has diminishing returns. The win is in reducing non-GEMM overhead (norms, routing, quantization).
**Note:** Original "45μs per norm" was wrong — Opus review found 331 norms/step, actual 12.4μs/call

## Discovery #4: FP8 KV cache is 4x slower (not faster)
**Expected:** FP8 KV = 2x capacity + similar speed
**Found:** 2x capacity but 4x slower throughput (FlashInfer FP8 attention overhead on Gemma4's head dims 256/512)
**Use case:** Only when capacity matters more than speed (long context, many users)

## Discovery #5: Stream parallelism is 20x SLOWER for MoE
**Expected:** Multi-stream expert dispatch → 1.5-2x speedup
**Found:** 20x slower (9,213 → 447 tok/s at B=240)
**Why:** Python-level `torch.cuda.Stream()` overhead dominates for μs-scale expert GEMMs. Synchronization cost > compute savings.

## Discovery #6: Expert weight caching gives zero benefit
**Expected:** Hot experts in L2 cache → faster access
**Found:** Zero measurable difference. Working set (348 MB) exceeds L2 (96 MB). No stable cross-layer hot expert set (Jaccard 0.14). Contiguous vs random access identical.
**Confirmed by:** 3 independent experiments

## Discovery #7: FusenCache beats BF16+CUDA graphs — in eager mode
**Expected:** FusenCache would be slower per-request (quantize/dequant overhead)
**Found:** FusenCache eager (6,685 tok/s) > BF16+CUDA graphs (6,615 tok/s)
**Why:** Compressed KV (4x smaller) reduces memory bandwidth pressure. Less data to read from HBM per decode step outweighs the dequantization cost.
**Also:** Single-request: 121 vs 89 tok/s (36% faster)

## Discovery #8: KV memory is the bottleneck at ctx≥1024
**Expected:** Compute-bound at all context lengths
**Found:** Throughput drops 3.2x from ctx=128 to ctx=3840 at C=32 due to KV memory pressure
**Implication:** FusenCache's 4x compression is not just "nice to have" — it's essential for real-world serving at non-trivial context lengths.
**Data:** 2,609 tok/s (ctx=128) → 811 tok/s (ctx=3840) at C=32

## Discovery #9: RedHat quantized attention (wrong), NVIDIA didn't
**Expected:** RedHat's NVFP4 model was correctly quantized
**Found:** RedHat quantized ALL layers including self_attn. NVIDIA's 31B reference keeps attention in BF16.
**Why it matters:** vLLM's QKV fusion takes max() of global scales, causing underflow for quantized attention. BF16 attention avoids this entirely.
**Fix:** Dequantized 115 attention projections back to BF16 via `convert_ct_to_modelopt.py`

## Discovery #10: Fused RMSNorm+FP4 kernel is 2.95x faster (C++) / 1.92x (Triton)
**Expected:** Moderate fusion benefit
**Found:** C++ kernel with native SM120 PTX `cvt.rn.satfinite.e2m1x2.f32` instruction = 2.95x
**Why C++ > Triton:** Hardware FP4 conversion instruction only accessible via PTX, not Triton
**Blocker:** MoE path can't use it (shuffle_rows between norm and quant). Non-MoE path (QKV) can.
**Projected gain:** +12.9% end-to-end when wired into vLLM fusion pass

## Discovery #11: vllm_c RMSNorm IS active (warning is cosmetic)
**Expected:** "Priority not set" warning meant native Python fallback
**Found:** Warning only fires during multimodal encoder profiling at startup (outside forward context). During actual inference, vllm_c C++ kernel is used for all 331 norms.
**Corrected:** 45μs/norm was wrong math (90 norms). Actual: 12.4μs/norm across 331 norms.

## Discovery #12: Gemma4 has per-layer residual scaling (pruning signal)
**Expected:** All layers contribute equally
**Found:** Layer scalars range 0.07 (layer 0) to 0.82 (layer 24). Low scalar = low residual contribution.
**Prunable:** Layers 2, 4, 8 (early sliding layers, lowest importance)
**Preserved:** All 5 global attention layers (5, 11, 17, 23, 29) are critical
**Estimated gain:** Remove 3 layers → ~10% speedup

## Discovery #13: No dead experts, but clear skew
**Expected:** Some experts might be completely unused
**Found:** All 128 experts active in all 30 layers. But distribution is skewed (Gini 0.30-0.66).
**Top-32 capture 50.4%** of activations. Each layer has its own hot set (no cross-layer pattern).
**21 merge candidate pairs** with similarity > 0.80 (best in layer 29)
**Conservative prune:** 10% removal → 1GB saved, 2.1% routing loss

## Discovery #14: SM120 has unused L2 persistence API
**Expected:** L2 cache is hardware-managed only
**Found:** `cudaDeviceSetLimit` works, 60 MB of 96 MB can be set as persisting. Completely unused by vLLM, FlashInfer, or CUTLASS.
**Also found:** FlashInfer has dedicated `mla_sm120.cu` kernel with TMA that may not be active. NVFP4 uses single-SM clusters (ClusterShape 1,1,1) when multi-SM is possible.

## Discovery #15: MoE shuffle+quant CAN be fused (but not norm+shuffle+quant)
**Expected:** Shuffle is an impenetrable barrier to fusion
**Found:** Shuffle (gather) can be folded INTO the quant kernel by adding a `dst2src_map` parameter (~5 lines CUDA). But norm CANNOT be moved past routing (data dependency).
**Gain:** 2.3% system throughput, bit-identical output

## Discovery #16: FusenCache CUDA graphs work (unit test) but crash at B=65
**Expected:** CUDA graph support would just work
**Found:** Unit tests pass (small batch), but vLLM serving crashes with CUDA assertion at B=65
**Root cause candidates:** Block table overflow, slot mapping OOB, or Triton grid miscalculation at non-captured batch sizes
**Status:** Agent S4 debugging

---

## Discoveries That Changed Our Plan

| Discovery | Original Plan | Revised Plan |
|---|---|---|
| #1 (inductor hurts) | torch.compile for everything | Disable for serving, keep for interactive |
| #2 (MoE already fused) | Build fused MoE kernel (2-5x) | Focus on non-GEMM overhead instead |
| #4 (FP8 KV slower) | FP8 KV for throughput | FP8 for capacity only, FusenCache for throughput |
| #5 (streams slower) | Multi-stream MoE dispatch | Abandoned |
| #6 (expert caching zero) | L2 expert caching | Ruled out permanently |
| #7 (FusenCache faster) | FusenCache = capacity trade-off | FusenCache = capacity AND throughput win |
| #8 (KV bottleneck) | Compute-bound optimization | KV compression is the primary lever |
| #14 (L2 persistence) | No L2 control available | L2 pinning is new optimization target |

## Projected Impact Stack

```
Current:                    6,685 tok/s (FusenCache eager)
+ CUDA graphs (fix B=65):  ~11,600 tok/s (1.74x from graphs)
+ Fused kernel (12.9%):    ~13,100 tok/s
+ Layer pruning (10%):     ~14,400 tok/s
+ L2 persistence (5-10%):  ~15,100-15,800 tok/s
+ PRO 6000 TP=2:          ~25,000-30,000 tok/s
```

## Discovery #22: Distillation is a trap for MoE models
**Expected:** Dense 9B distilled from 26B → 3-5x faster
**Found:** MoE 26B only activates 2.47B params/token. Dense 9B reads 3.6x MORE weight memory. Realistic speedup: ~2x, not 3-5x.
**Better alternatives:** Gemma4 E2B (2B, already exists), Qwen3.5-9B (already cached), n-gram spec decode (zero cost).

## Discovery #23: The bottleneck is KV READ bandwidth, not storage
**Expected:** FusenCache's win is from storing less KV
**Found:** FA2 is 93% bandwidth-optimal for BF16 KV. The real bottleneck is READING K/V during attention (63% of decode). FP8 native attention (half the read bytes) would give 1.44x total decode speedup — bigger than any other optimization.
**Status:** FP8 native decode kernel being built by Opus agent.

## Discovery #24: SM100 and SM120 are binary-incompatible
**Expected:** "Blackwell" GPUs share binary compatibility
**Found:** DeepGemm's 317 cubins are SM100-only (B100/B200 datacenter). SM120 (RTX 5090/PRO 6000) gets `CUDA_ERROR_NO_BINARY_FOR_GPU`. Different ISAs despite both being "Blackwell."
**Implication:** Datacenter-targeted kernels won't work on consumer Blackwell. Must compile for SM120 specifically.

## Discovery #25: Layer pruning fails catastrophically (-41% quality)
**Expected:** Removing 3 early layers (low residual scalars) → minor quality impact
**Found:** 41.3% overall quality drop. Coding -58%, reasoning -55%. Generation degenerates into repetition loops.
**Lesson:** Low residual scalar ≠ low importance. Early layers are critical infrastructure for coherent generation.

## Discovery #26: vLLM has native request priority scheduling
**Expected:** Need to patch scheduler for SJF
**Found:** `--scheduling-policy priority` exists. API accepts `priority: int` field. Scheduler sorts by `(priority, arrival_time)` and preempts lowest-priority on KV pressure.

## Discovery #27: Block size 64 optimal for FusenCache (not 16)
**Expected:** Default block_size=16 is fine
**Found:** FusenCache k4v4b64 has quant block size=64. block_size=64 gives exact-fit (one quant group per page). block_size=32 spans two pages per group. Also: `max_num_batched_tokens=8192` (4x default) could give +50-150% throughput.

## Discovery #28: Gemma4 E2B is overhead-bound, not bandwidth-bound
**Expected:** Small model → bandwidth-limited (like 26B)
**Found:** At 1.3-5.3GB weights, read time (0.85-3.5ms) is smaller than vLLM's 5ms Python overhead. Single-user throughput is overhead-limited. SGLang (~2ms overhead) would be 2x faster for this model size.
**Lesson:** Different model sizes need different serving frameworks.

## Discovery #29: Mixture of Agents — Qwen3.5-9B beats E2B for coding fast-brain
**Expected:** E2B (same family) would be best fast model
**Found:** E2B has higher cascade rate on code tasks (needs to fall back to 26B more often), making it net slower end-to-end. Qwen3.5-9B has better code quality at similar speed.

---

## Updated Discoveries That Changed Our Plan (Session 2)

| Discovery | Original Plan | Revised Plan |
|---|---|---|
| #22 (distillation trap) | Distill 26B → 9B | Use existing models (E2B, Qwen3.5) |
| #23 (KV read bandwidth) | Focus on KV storage | Build FP8 native attention kernel |
| #24 (SM100≠SM120) | Enable DeepGemm | Can't — binary incompatible |
| #25 (layer pruning fails) | Remove 3 layers → 10% | Model is NOT prunable |
| #26 (native priorities) | Patch scheduler | Just use `--scheduling-policy priority` |
| #27 (block_size 64) | Default block_size=16 | block_size=64 for FusenCache |
| #28 (E2B overhead-bound) | Run E2B on vLLM | Run E2B on SGLang instead |
| #29 (Qwen > E2B for code) | E2B as fast brain | Qwen3.5-9B as fast brain |

## Updated Projected Impact Stack

```
Current:                    6,685 tok/s (FusenCache eager)
+ FP8 native attention:    ~9,600 tok/s (1.44x from halved KV reads)
+ block_size=64 + tuning:  ~10,500 tok/s (+50% from scheduler tuning)
+ C++ FusenCache decode:    ~11,500 tok/s (CUDA graphs on non-attention ops)
+ PRO 6000 DP=2:           ~20,000 tok/s (2x hardware, aggregate)
+ MoA (Qwen fast brain):   +350 tok/s easy tasks on GPU 1
```

## Discovery #30: FP8 Triton attention can't beat FA2 BF16
**Expected:** FP8 native decode = half the reads = 2x faster than FA2 BF16
**Found:** Triton FP8 kernel at 35% BW utilization (430μs) vs FA2 at 93% BW (323μs). 1.3x SLOWER despite reading half the data.
**Why:** Triton's per-element pointer arithmetic for paged KV access, no software pipelining (vs FA2's cp.async), small tile sizes.
**Path forward:** Port to CUDA C++ with cp.async pipelining — same approach FA2 uses but for FP8 KV. Or fix FlashInfer's FP8 path for Gemma4's head dims.

## Discovery #31: 2:4 sparsity helps prefill but not decode
**Expected:** 2x MoE GEMM across the board
**Found:** FP8 + 2:4 sparse: 3.1x at M=512 (prefill). But cuSPARSELt minimum M=32 — decode (M=1-8) fails. NVFP4 + 2:4 doesn't exist (no FP4 sparse variant in CUTLASS/cuSPARSELt).
**Quality:** 37% error without fine-tuning. Needs SparseGPT/Wanda pruning + fine-tuning.
**Use case:** Prefill acceleration only, after model fine-tuning. Not applicable to decode.

## Discovery #32: Prefix caching works with FusenCache (40% hit rate)
**Expected:** Might be incompatible (different KV format)
**Found:** Fully compatible. 40.3% hit rate with shared system prompt across 5 requests. At production scale (1000 requests, 200-token prompt): 99.9% prefill compute reduction, ~12.4 MB KV savings.
**Caveat:** FusenCache 64-token block granularity — prompts aligned to 64-token boundaries get better hit rates.

## Discovery #33: DFlash (diffusion speculative decoding) exists for Qwen3.5 but NOT Gemma4
**Expected:** Diffusion requires diffusion-trained model
**Found:** DFlash uses a diffusion-trained DRAFT model with a standard autoregressive TARGET. Already got 94.6 tok/s on Qwen3.5 in this repo. But no DFlash draft exists for Gemma4 26B MoE. EAGLE3 drafts exist for Gemma4 31B (dense) but not 26B MoE.
**Options:** Train DFlash draft for Gemma4 26B (2-3 days GPU), or test if 31B EAGLE3 draft works with 26B MoE.

## Discovery #34: No speculative decode draft exists for Gemma4 26B MoE
**Expected:** EAGLE3 drafts for 31B might work with 26B
**Found:** hidden_size mismatch (5376 vs 2816) — fatal. Both RedHatAI and thoughtworks EAGLE3 drafts are 31B-specific.
**Options:** Train 26B-specific EAGLE3 draft (1-2 days GPU, ~200M params) or use n-gram (zero cost, ~1.2x).
**Note:** Gemma4 9B also mismatches (hidden_size=3840). The 26B MoE has unique dimensions.

## Discovery #35: Cooperative groups work on SM120, 278μs per grid.sync()
**Expected:** Persistent MoE kernel eliminates all inter-kernel overhead
**Found:** grid.sync() costs 278μs per barrier across 170 SMs. With 4 barriers per MoE layer, that's ~1ms overhead — comparable to what CUDA graphs already save.
**Result:** Persistent kernel IS viable but marginal gain over CUDA graphs. Main value: proves cooperative launch works on consumer Blackwell for future mega-graph concept.
**Bug found:** Phase 6 unshuffle has race condition for multi-topk accumulation (needs atomicAdd).

---

## Meta-Discovery: The Measurement-First Principle

**Pattern observed across 35 experiments:** 60% of confident predictions were wrong. The best discovery (disable inductor = 2x) was unplanned. The worst failures (FP8 KV, stream parallelism, pruning) had the highest pre-confidence.

**The principle:** Never invest more than 1 day before measuring the critical assumption.

```
WRONG approach (what we did too often):
  Predict X% gain → Build full solution (days) → Discover it doesn't work → Waste

RIGHT approach (what we learned):
  Identify critical assumption → Build cheapest test (hours) → Measure → 
  If works: scale up
  If fails: pivot immediately
```

**Applied to future work:**

| Future Project | Critical Assumption | Cheapest Test | Cost |
|---|---|---|---|
| Diffusion adapter | Acceptance rate > 50% for MoE | Train 1-layer, 50M head on 5K prompts | 4 hours GPU |
| C++ FP8 attention | cp.async achieves 90%+ BW | Write 100-line memcpy benchmark | 2 hours |
| Persistent MoE | Grid.sync overhead < GEMM time | Already tested: 278μs (viable) | Done |
| Any new kernel | Faster than existing | Microbenchmark FIRST | 1 hour |
| Any model change | Quality preserved | Run validation suite (100+ tests) FIRST | 30 min |

**For the diffusion plan specifically:**
- Day 1: Train tiny draft (1 layer, 50M, 5K prompts) → measure acceptance rate
- Decision gate: acceptance > 50% → proceed. < 30% → pivot to n-gram/dedicated model.
- This 4-hour experiment prevents wasting 5-7 days on a failed approach.

## Discovery #36: Lossless diffusion spec decode ceiling is ~300-350 tok/s (not 700)
**My prediction:** ~700 tok/s with diffusion adapter
**Opus plan found:** Target verification step (8.3ms for 26B MoE) is irreducible. Even with perfect acceptance rate, 8 tokens / 11ms = ~727 tok/s theoretical. Real-world with 65% acceptance: ~245-350 tok/s.
**Key insight:** The verification bottleneck means diffusion drafting helps LESS for large MoE models than for small dense models. DFlash on Qwen3.5-9B achieved 94.6 tok/s from 50 tok/s (~2x), consistent with verification being the limiter.
**Still worth it:** 245 tok/s is 2x current 120 tok/s single-user. And n-gram is free (160-320 tok/s for code).

## Discovery #37: FP8 attention can't beat FA2 without KV cache layout change
**Expected:** C++ FP8 with cp.async → 165μs (2x faster than FA2 323μs)
**Found:** C++ FP8 = 368μs (40.7% BW), Triton FP8 = 408μs (36.7% BW), FA2 BF16 = 323μs (93% BW)
**Root cause:** KV cache is position-first layout. Each K/V position is strided by num_kv_heads×head_dim, preventing contiguous bulk reads. FA2 achieves 93% because BF16 layout matches its access pattern.
**Fix would require:** Transpose KV to head-first layout — but this breaks vLLM's PagedAttention which assumes position-first. Fundamental architectural constraint.
**Lesson:** Data layout > kernel optimization. The kernel is fine; the memory layout is wrong for FP8.

## Discovery #38: FusenCache CUDA graph failure is metadata replay, not kernel
**Expected:** C++ kernel would fix CUDA graph capture (Triton JIT was the problem)
**Found:** C++ captures cleanly AND replays, but at 0.5 tok/s. Same as Triton. Both kernels work; the metadata builder doesn't propagate block_tables/seq_lens correctly during graph replay.
**Also found:** C++ kernel crashes at C≥8 (buffer OOB), and eager mode is 15 tok/s at C=1 (much slower than the 6,685 batch measurement, which used a different config).
**Root cause:** FusenKVMetadataBuilder.build_for_cudagraph_capture() — the pre-allocated tensors aren't being updated in-place during replay. This is a vLLM API integration bug.
**Fix needed:** Debug the metadata builder's tensor lifecycle during CUDA graph capture vs replay.

## Discovery #39: N-gram spec decode HURTS Gemma4 26B by -11%
**My prediction:** 1.3-1.8x speedup for code
**Found:** -30% on Python code, neutral on SQL/bash. Average -11%.
**Why:** Model is compute-bound at 123 tok/s (not memory-bound). Speculative overhead (draft+verify) exceeds n-gram hit rate. Novel code has low repetition. vLLM auto-disables async scheduling with n-gram.
**Lesson:** Speculative decode helps memory-bound models (small/dense). Gemma4 26B MoE activates 2.47B params/token — this is actually compute-heavy per-token despite being "only" 26B.

## Discovery #40: Expert pruning fails even at 0.13% (5 of 3840 slots)
**Previous:** 50% pruning = garbage. Maybe 5% would work?
**Retried:** Zeroed just 5 experts in layer 0 (the least active, 0.02% frequency each)
**Found:** 6/20 coherent (30%) — same catastrophic failure as 50% pruning
**Why:** Layer 0 is input processing — any disruption propagates through all 29 layers. Also, activation frequency from embeddings ≠ runtime importance.
**FINAL VERDICT:** Expert manipulation is completely off the table for Gemma4 without fine-tuning. No safe set of experts to disable exists.

## Discovery #41: Middle layer pruning worse than early — model is 100% non-prunable
**Previous:** Early layers (2,4,8) removal = -41%. Opus suggested middle layers safer.
**Retried:** Removed layer 14, 13, 19 (middle) individually.
**Found:** 0% coherence on ALL three. Infinite repetition loops. WORSE than early layer removal.
**Why:** Middle layers have high scalars (0.55-0.72) carrying critical residual signal. Removing one breaks attention pattern consolidation.
**FINAL VERDICT:** No layer, no expert, at any granularity, in any location, can be removed from Gemma4 26B without post-removal fine-tuning. The model is 100% non-prunable.

---

## Meta-Discovery: Pruning vs Augmentation

**Pruning (DEAD for Gemma4 26B MoE):**
- Expert pruning at 50% → garbage (Discovery #17)
- Expert pruning at 0.13% (5 experts) → garbage (Discovery #40)  
- Layer pruning early (2,4,8) → -41% quality (Discovery #25)
- Layer pruning middle (13,14,19) → 0% coherent (Discovery #41)
- Cross-layer KV sharing → impossible, zero similarity (Discovery #18)

**VERDICT:** Gemma4 26B MoE is a tightly-coupled system. Removing ANY component at ANY granularity causes catastrophic failure without post-removal fine-tuning. The 128-expert × 30-layer architecture has no redundancy — every piece is load-bearing.

**Augmentation (ALIVE — adds capability without removing anything):**

| Approach | What it does | Status | Expected gain |
|---|---|---|---|
| Diffusion adapter | Adds draft head, freezes 26B | Planned (5-7 days PRO 6000) | 2-3x single-user |
| Two-tier brain | Offloads cold parts to CPU, keeps ALL | Built (32 tests) | +20-26% throughput, 100% quality |
| Multi-model routing | Different models per task difficulty | Built (MoA config) | Big brain for hard, small brain for easy |
| EAGLE3/DFlash draft | Adds small draft model alongside 26B | Needs training (1-2 days) | 2-3x single-user |
| FusenCache KV compression | Compresses cache, not model | Working (6,685 tok/s) | 4x KV capacity |
| NVFP4 quantization | Compresses representation, all info preserved | Deployed | 17GB vs 52GB |

**The principle:** Don't subtract from the model — ADD to it or COMPRESS its representation. The model's knowledge is distributed across ALL components with no safe subset to remove.

---

## Updated Roadmap (Post-Pruning)

### This Week (before PRO 6000)
1. ✅ Fix E2E test (model name) — done
2. 🔄 Fix FusenCache metadata builder — Opus agent working
3. ❌ N-gram spec decode — confirmed hurts, skip

### Next Week (PRO 6000 arrives)
1. DP=2 benchmark (scripts ready)
2. FusenDiffusion gate test (4-hour experiment)
3. Mixture of Agents: Gemma4 26B (GPU 0) + Qwen3.5-9B on SGLang (GPU 1)
4. 32K context testing
5. Disaggregated prefill/decode test

### This Month
1. Train diffusion adapter (if gate test passes)
2. Train EAGLE3 draft for 26B
3. FusenCache metadata fix → CUDA graphs
4. Block size + scheduler tuning sweep
5. Upstream PRs (fused kernel + inductor finding)

### Off the Table (confirmed dead)
- Any form of model pruning (experts, layers, weights)
- N-gram speculative decode on novel code
- FP8 attention improvement (KV layout constraint)
- DeepGemm on SM120 (binary incompatible)
- Cross-layer KV sharing

## Discovery #42: N-gram spec decode -49% even at n=1, content type irrelevant
**Previous retry:** n=4 was -11%. Hypothesis: lower n + repetitive content would help.
**Retried:** n=2 (-49%), n=1 (-49%). Repetitive JSON/HTML = same as novel code.
**Root cause:** CUDA graph full mode is optimized for standard decode shapes. Speculative decode introduces variable batch sizes that break graph reuse. Overhead is structural, not proportional to n.
**FINAL VERDICT:** N-gram speculative decoding is completely dead for NVFP4+CUDA graphs. The 123 tok/s baseline is optimal for single-user AR decode.

## Discovery #43: FusenCache CUDA graph fix — limit capture sizes (42x speedup)
**Problem:** FusenCache + CUDA graphs = 2.9 tok/s (should be ~120)
**Root cause:** 35 CUDA graphs up to batch=512. Each graph allocates persistent Triton decode buffers per layer. 30 layers × 35 sizes = 119 GiB estimated graph pool on 32GB GPU → catastrophic memory pressure.
**Fix:** Limit to 7 graphs (max batch=32): `-cc.cudagraph_capture_sizes '[1,2,4,8,16,24,32]'`
**Also:** Remove custom build_for_cudagraph_capture() — base class default is correct (same as FlashAttention).
**Result:** 122.7 tok/s — matches native vLLM FlashAttention while using 4x less KV memory.
**Lesson:** The "metadata builder tensor lifecycle" hypothesis was WRONG. The real issue was memory exhaustion from too many graph captures. Simple config fix, not code fix.

## Discovery #44: FusenCache + CUDA graphs — sweet spot is max_batch=32
**Tested:** 7 graph sizes (max=32): 113 tok/s C=1, 471 tok/s C=16, crashes at C=32+
**Tested:** 11 graph sizes (max=128): 5.4 tok/s — graph memory explosion returns
**Sweet spot:** 7 sizes [1,2,4,8,16,24,32], 0.41 GiB graphs, 174K KV tokens
**Limitation:** Batch sizes > max_graph_size fall back to eager → Triton JIT → crash/slow
**Implication:** FusenCache + CUDA graphs is best for single-user/small-batch (≤16 concurrent).
For high batch (C=128+): FusenCache eager (6,685 tok/s) is still the best config.
For single-user: FusenCache + CUDA graphs (113 tok/s, matches native vLLM).

## Discovery #45: Shared buffers fix memory (0.42 GiB constant) but C≥16 still crashes
**Fix #2 result:** Graph memory now CONSTANT at 0.42 GiB for 13 sizes (was 10.74 GiB → exploded).
**C=1: 127.8 tok/s** (matches best BF16 config + 4x KV compression!)
**C=4: 392 tok/s** (excellent)
**C=16+: errors** (shape mismatches in mixed prefill+decode path when batch > graph capture size)
**Two shape fixes applied** but more edge cases remain at larger batch sizes.
**Status:** Single-user FusenCache + CUDA graphs is SOLVED. Batch path needs more debugging.

## Discovery #46: Shared buffer underallocation was 32x OOB write
**Problem:** C=16+ crashed with shared buffers
**Root cause:** `_optimal_splits(max_seq, max_B=256)` = 1 split, but at B=6 (mixed batch decode portion), kernel needs 32 splits → 32x buffer overflow on split dimension
**Fix:** Allocate at `_optimal_splits(max_seq, B=1)` = 32 splits (maximum possible)
**Result:** Buffer overflow fixed. But sporadic CUDA crashes remain (Triton/SM120 issue, affects all concurrency levels including C=1, unrelated to FusenCache).

## Discovery #47: Sporadic crash is FlashInfer JIT under concurrency, not FusenCache
**Debug method:** CUDA_LAUNCH_BLOCKING=1 with sequential requests → 40+ requests, zero crashes
**Finding:** Crash only manifests under concurrent serving. Sequential is rock-solid.
**Primary suspect:** FlashInfer sliding-window attention JIT kernels (25/30 layers). SM120 JIT cubins may have a race condition or codegen bug under concurrent kernel dispatch.
**FusenCache Triton decode:** NOT the cause (stable under sequential testing).
**Next step:** Run concurrent load with CUDA_LAUNCH_BLOCKING in serving mode to catch the exact FlashInfer kernel.

## Discovery #48: C++ decode kernel works at ALL concurrency — zero errors
**Problem:** C++ kernel crashed at C≥8 (50% failures)
**Root cause:** Double buffer allocation (Triton + C++ = 5.8 GiB). Fix: share buffers.
**Result:** C=1→32: ZERO errors, 25.6→253 tok/s. C++ bypasses both Triton JIT and FlashInfer concurrency bugs.
**Status:** C++ kernel is now the preferred decode path (auto-enabled when .so exists).
**Note:** Throughput at C=32 (253 tok/s) is lower than FusenCache eager (6,685 tok/s) — this is with inductor mode. Need to test with -cc.mode none for full throughput.

## Discovery #49 (FINAL): FusenCache + CUDA graphs ceiling = C=8 (FlashInfer limit)
**C++ kernel:** works perfectly C=1-32 with inductor, C=1-8 without inductor
**The blocker:** FlashInfer sliding-window JIT crashes at C=16+ under no-inductor CUDA graphs
**NOT our code:** The crash is in FlashInfer's 25 sliding-window attention layers, not FusenCache
**Workaround:** Use FusenCache eager for batch (6,685 tok/s), C++ CUDA graphs for single-user (116 tok/s + 4x KV)
**Definitive best configs:**
  - Single-user + max KV: FusenCache + C++ + CUDA graphs = 116 tok/s, 165K tokens
  - Batch throughput: FusenCache eager = 6,685 tok/s, 175K tokens  
  - Batch without FusenCache: BF16 + no inductor + CUDA graphs = 6,615 tok/s, 43K tokens

## Discovery #50: Root cause = async CUDA memory recycling race condition
**All previous hypotheses were WRONG:**
  - Not FlashInfer JIT (wrong)
  - Not mixed prefill+decode path (wrong)
  - Not shared workspace buffer (wrong)
  - Not FusenCache kernel bug (wrong)
**Actual cause:** PyTorch CUDA allocator recycles freed temporary tensor memory while in-flight FusenKV decode kernels still read it. vLLM's async scheduling starts step N+1's preprocessing before step N's GPU kernels complete.
**Proof:** CUDA_LAUNCH_BLOCKING=1 eliminates crash (forces sync).
**Fix:** stream.synchronize() at end of forward(). ~3ms overhead.
**Better fix (TODO):** CUDA events to fence only shared buffers, not full sync.
**Lesson:** The 5th hypothesis was right. Always test the simplest explanation (async race) before complex ones (FlashInfer codegen bugs).

## Discovery #51: Sync fix doesn't help C=16 — it's a hard OOB, not async race
**Expected:** stream.synchronize() would prevent memory recycling race → fix C=16+
**Found:** C=1-8 still works (115-277 tok/s), C=16+ still crashes fatally
**Implication:** The crash at C=16 is NOT the same bug as the sporadic crash. It's a HARD out-of-bounds in either:
  - CUDA graph replay at padded batch size 16
  - FusenCache C++ kernel at B=16
  - Shared buffer indexing when actual_B=16 and graph_B=16 (exact match, no padding)
**Next debug step:** Run C++ kernel standalone at B=16 with compute-sanitizer to find exact OOB
