# Experiment Log — 2026-04-17

## Hardware
- 2× RTX PRO 6000 Blackwell (SM120a), 96 GB GDDR7 each
- PCIe-only (no NVLink), AMD 9950X3D host
- WSL2 (pin_memory=False penalty active)

## Baseline Established
- **9,100 tok/s** peak at C=384, Gemma4 26B NVFP4, plain vLLM, FULL_DECODE_ONLY graphs
- 1.38× over 5090's 6,615 tok/s
- KV capacity: 278K tokens (63.66 GiB)

## Completed Experiments

### Deliverables (ready to deploy)
| ID | What | Status | Files |
|---|---|---|---|
| T1-B | Piecewise CUDA graph fix | Patch applied | fusen_kv/backend.py, plugin.py |
| T2-I | FP8 decode attention | 14/14 tests + monkey-patch | autokernel_v2/plugins/fp8_decode_backend.py, patches/fp8_decode_monkey_patch.py |
| ASI-1 | Disaggregated 1P1D | Script fixed, Docker failed (NCCL init) | serve_disaggregated.sh |
| ASI-2 L1 | Per-task max_tokens | Deployed | fusen_solver/strategies/presets.py, core/solver.py |
| T2-N | Fused shuffle+quant MoE | Kernel works, blocked on swizzled scales | patches/fused_shuffle_quant_wrapper.py |

### Gate Tests (killed)
| ID | What | Result | Key Number |
|---|---|---|---|
| T3-M | Router prediction cascade | KILL | Jaccard=0.034, binary MI=0.003 bits |
| T3-O | L2 expert pinning | KILL | Gini=0.117, near-uniform by design |
| T3-L | Semantic KV eviction | MARGINAL | 2.5× at 90% retention |
| I-DLM v2 | Correct per-request masking | KILL | Acceptance 42%→8.6% |

### In Progress
| ID | What | Status |
|---|---|---|
| T2-H | FP8 KV on Qwen3-MoE | Servers launching |
| T3-I | DDTree gate test | Analysis running |
| T1-F | TriAttention calibration | Research running |
| T2-N fix | Swizzled scale format | Opus investigating |
| ASI-1 live | 1P1D benchmark | Docker NCCL failure, debugging |
| T1-B Docker | FusenCache image | IndentationError in container copy |

## Key Insights
1. MASK-to-MASK attention in I-DLM is beneficial (implicit AR chain), not contamination
2. MoE routing is independent across layers (aux loss enforces uniformity)
3. Expert activation is near-uniform (no hot set for L2 pinning)
4. KV eviction at 1.5K context gives only 2.5× — need longer context for better results
5. PRO 6000 scales 1.38× over 5090 at matched concurrency (more bandwidth + FP4 compute)
6. The biggest remaining throughput lever is T1-B (piecewise graphs for FusenCache)
