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
