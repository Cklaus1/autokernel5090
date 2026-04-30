# Session Final Status V2 — April 9-11, 2026

## Performance

| Config | C=1 | C=32 | C=256 | KV Tokens |
|---|---|---|---|---|
| enforce_eager baseline | 18 | — | — | 43K |
| Inductor + CUDA graphs | **127** | 2,071 | 3,112 | 43K |
| No inductor + CUDA graphs | 89 | 1,738 | **6,615** | 43K |
| **FusenCache eager** | 121 | — | **6,685** | **175K** |
| FusenCache + CUDA graphs (no-async) | 67 | **1,496** | 1,345 | 165K |
| FusenCache + CUDA graphs (with async) | 116 | — | crashes | 165K |

**Best configs:**
- Max throughput: FusenCache eager = 6,685 tok/s + 4x KV
- Single-user + 4x KV: FusenCache + CUDA graphs = 116 tok/s
- Single-user BF16: Inductor + CUDA graphs = 127 tok/s

## Session Stats

| Metric | Count |
|---|---|
| Commits | 104 |
| Discoveries | 52 |
| Novel research ideas | 528 (scored, top 20 ranked) |
| Tests passing | 330+ |
| Lines of code | ~170K+ |
| Agents launched | 90+ |
| CUDA kernels built | 6 |
| Plans/documents | 30+ |

## 52 Discoveries (Key Highlights)

1. Disabling torch.compile doubles throughput (2.1x)
2. vLLM MoE already fused (6 kernels not 128)
7. FusenCache beats BF16+CUDA graphs in eager mode
19. Real profiling: attention=63%, norms<1% (microbenchmarks were wrong)
23. FP8 attention can't beat FA2 without KV layout change
37. Data layout > kernel optimization for FP8
40. Expert pruning fails at 0.13% (5 of 3840 slots)
43. FusenCache CUDA graph fix: limit capture sizes (42x speedup)
48. C++ decode kernel works at all concurrency — zero errors
50. Root cause of sporadic crash: async CUDA memory recycling
52. C=16+ fixed: async scheduling race + buffer OOB

## What's Built

### Inference
- FusenCache KV plugin (4x compression, 175K tokens)
- 6 CUDA kernels (fused norm+FP4, FusenCache decode C++/Triton, FP8 attention, persistent MoE)
- Fused RMSNorm+FP4 C++ kernel (2.95x, native SM120 PTX)
- AutoKernel v2 auto-discovery system (9 files, plugin registry)
- NVFP4 model on HuggingFace (cklaus/gemma-4-26B-A4B-it-NVFP4)

### Coding Platform
- fusen_solver: 5 modes (isolated, collaborative, racing, decomposed, auto)
- 194 tests, all passing
- Multi-backend routing, session affinity, priority injection
- Agent memory across sessions, Bayesian strategy learning
- Codebase indexing/RAG (4-signal relevance scoring)
- File-level decomposition with dependency ordering
- Sandbox test execution (Docker isolation)
- Shadow mode BCode integration (data-driven promotion)

### Infrastructure
- Docker builds with 3 community PRs
- serve_gemma4.sh (adaptive launcher)
- serve_gemma4_dp2.sh (PRO 6000 DP=2)
- bench_dp2.py, bench_gemma4_nvfp4.py
- E2E integration tests (10 CI checks)
- 2 upstream PR descriptions ready

### Research
- 528 novel research ideas across 5 documents
- Top 20 ranked by Impact×Feasibility with execution roadmap
- 30+ planning documents
- BCode pipeline analysis + integration plan

## What Doesn't Work (Definitively Ruled Out)

- Expert pruning (any %, any layer, any granularity)
- Layer pruning (any layer — early or middle)
- Cross-layer KV sharing (zero similarity)
- N-gram speculative decode (-49% on this model)
- FP8 attention improvement (KV layout constraint)
- DeepGemm on SM120 (binary incompatible with SM100)
- Stream parallelism (20x slower)
- L2 cache persistence (no benefit)
- Distillation for MoE (activation params already small)

## Next Steps

### This Week
1. Expert Output Memoization gate test (4 hours)
2. Router Prediction Cascade gate test (2 hours)
3. Fix async scheduling race properly (Opus agent working)

### Next Week (PRO 6000)
- DP=2 benchmark (scripts ready)
- FusenDiffusion gate test (4 hours)
- Mixture of Agents (Gemma4 26B + Qwen3.5-9B)
- Speculative Editing VS Code extension prototype

### This Month
- Top 5 ranked ideas prototyped
- BCode integration Phase 1 (extract pipeline.ts)
- Train EAGLE3/DFlash draft for Gemma4 26B
