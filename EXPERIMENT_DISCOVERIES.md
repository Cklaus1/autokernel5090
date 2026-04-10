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
