# Session Final Status — April 9-10, 2026

## Performance Results (Verified)

```
     18 tok/s  →  enforce_eager baseline
    127 tok/s  →  CUDA graphs + inductor                      7x
  3,112 tok/s  →  high concurrency (C=256)                  173x
  6,615 tok/s  →  disable inductor + CUDA graphs             368x
  6,685 tok/s  →  FusenCache k4v4b64 eager (BEST)           371x
```

### Best Configs

| Use Case | Config | tok/s | KV Tokens |
|---|---|---|---|
| **Max throughput** | FusenCache eager | **6,685** | 166K |
| **Best without FusenCache** | No inductor + CUDA graphs | 6,615 | 43K |
| **Single-user latency** | FusenCache eager | 121 | 166K |
| **Single-user (BF16)** | Inductor + CUDA graphs | 127 | 43K |

### What DOESN'T Work

| Config | Result | Why |
|---|---|---|
| FusenCache + CUDA graphs | ~1 tok/s | Triton kernel runs outside CUDA graph (splitting point), JIT recompiles each step |
| FP8 KV + CUDA graphs | 479 tok/s | FlashInfer FP8 attention 4x slower on Gemma4 head dims |
| Expert pruning | Garbage output | All 128 experts essential (importance range 0.374-0.482) |
| Cross-layer KV sharing | N/A | Zero similarity between layers |
| Stream parallelism | 20x slower | Python stream overhead > GPU compute |
| XQA SM120 | 2-19x slower | Not designed for GQA pattern |
| L2 cache persistence | No benefit | Hardware LRU already optimal |

---

## 20 Experiment Discoveries

1. Disabling torch.compile doubles throughput (2.1x)
2. vLLM MoE already fused (6 kernels, not 128)
3. GEMMs are only 27% of decode time (from microbenchmarks — see #19)
4. FP8 KV is 4x slower
5. Stream parallelism 20x slower
6. Expert weight caching zero benefit
7. FusenCache beats BF16+CUDA graphs in eager mode
8. KV is bottleneck at ctx≥1024
9. RedHat quantized attention wrong
10. Fused RMSNorm+FP4 kernel is 2.95x faster (C++)
11. vllm_c RMSNorm IS active (warning cosmetic)
12. Gemma4 has per-layer residual scaling (pruning signal)
13. No dead experts, but clear skew
14. SM120 has unused L2 persistence API (but it doesn't help)
15. MoE shuffle+quant CAN be fused (+2.3%)
16. FusenCache CUDA graphs crash at B=65 (fixed, but still slow due to JIT)
17. Expert pruning doesn't work (all essential)
18. Cross-layer KV sharing impossible (zero similarity)
19. **REAL profiling: attention=63%, norms<1% (not 26%!)** — microbenchmarks were wrong
20. SM120 XQA slower than FA2, L2 persistence no benefit

---

## What We Built (~70K+ lines across session)

### Kernels
- Fused RMSNorm+FP4 Triton kernel (1.92x, autotuned)
- Fused RMSNorm+FP4 CUDA C++ kernel (2.95x, native SM120 PTX)
- FusenCache KV compression kernels (store + decode + logits_soft_cap)

### Systems
- FusenCache vLLM plugin (4x KV compression, 7 integration bugs fixed)
- AutoKernel v2 auto-discovery system (9 files, 3193 lines)
- Two-tier brain architecture (fast GPU + slow CPU fallback)
- Parallel problem-solving architecture (in progress)
- Auto-config profiler (8 configs, 36 tests)
- Adaptive server monitor (dashboard, metrics, CI)
- Model spec DSL (YAML configs, CLI, 4 presets)

### Analysis
- MoE decode step decomposition (Nsight Compute profiled)
- Expert activation profiling (128 experts × 30 layers)
- Expert topic mapping (12 categories, specialization scores)
- Layer importance scoring (5 metrics, pruning candidates)
- Context length scaling curves
- Cross-layer KV similarity analysis
- SM120 feature audit (CUTLASS, FlashInfer, L2)
- Speculative decoding feasibility report

### Infrastructure
- NVFP4 model conversion pipeline (CT → modelopt)
- Docker build with 3 community PRs
- Perplexity eval harness (3 modes, 3 review passes)
- Pruning quality validation suite (100+ tests, 6 domains)
- Component analysis harness (4 approaches)
- Expert/layer pruning pipelines
- serve_gemma4.sh adaptive launcher
- serve_gemma4_tp2.sh TP=2 for PRO 6000
- 2 upstream PR descriptions ready
- HuggingFace model: cklaus/gemma-4-26B-A4B-it-NVFP4

### Documentation
- EXPERIMENT_DISCOVERIES.md (20 discoveries)
- GEMMA4_NVFP4_BENCHMARKS.md (full analysis)
- MOE_PROFILING.md (decode decomposition)
- 3 optimization roadmaps (comprehensive, V2, ASI-calibrated)
- 6 feasibility/design docs (spec decode, disaggregated, two-tier, etc.)

---

## What's Actually Left to Optimize

After exhaustive investigation, the picture is clear:

### The hardware ceiling
```
Attention decode (FA2):  63% of step  — FA2 is optimal for SM120, can't improve
MoE grouped GEMM:       ~30%         — bandwidth-bound, needs bigger batch
Everything else:          ~7%         — norms, quant, routing — already fast
```

### Remaining levers (ranked by impact)

| # | Optimization | Expected Impact | Effort |
|---|---|---|---|
| 1 | **PRO 6000 TP=2** | 1.7x throughput + 6x KV capacity | Hardware (arriving) |
| 2 | **N-gram speculative decode** | 1.5-2.5x single-user for code | 2-4 hours |
| 3 | **Higher max_model_len** | More KV → more batch → more throughput at long ctx | Config change |
| 4 | **Parallel problem-solving** | 8x quality OR 8x speed (decomposed problems) | Building now |
| 5 | **Disaggregated prefill/decode** | 3-5x better P99 TTFT under load | Day-one on PRO 6000 |
| 6 | **Upstream contributions** | Helps community, gets reviewed | 2 PRs ready |
| 7 | **Compile FusenCache Triton to .cubin** | Pre-compiled kernels avoid JIT, enable CUDA graph capture | Days |
| 8 | **Layer pruning (27L)** | ~10% if quality holds | Need quality validation |

### What's NOT worth pursuing further
- Fused norm+quant integration: norms are <1% of decode (Discovery #19)
- Expert pruning/merging: all experts essential
- L2/cache optimizations: hardware already optimal
- Alternative attention: FA2 is best for SM120
- Additional kernel fusion: MoE is already 6 kernels, can't fuse further

---

## The Key Insight

**FusenCache is the optimization.** Not because of kernel fusion or clever math — because **smaller KV = less memory traffic = faster decode.** The 4x KV compression reduces the bytes read from HBM per decode step, which is the actual bottleneck (63% attention + 30% MoE, both memory-bandwidth-bound).

Everything else we tried (fused kernels, CUDA graphs, stream parallelism, caching, pruning) either doesn't help or helps less than simply reading less data from memory.

The next big jump comes from **more hardware** (PRO 6000 = 2x bandwidth) and **more batch** (1.28M KV tokens = 10x concurrent requests).
