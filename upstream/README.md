# Upstream Contributions

Two findings from optimizing Gemma4 26B-A4B-it NVFP4 on RTX 5090 (Blackwell
SM120) that benefit all vLLM users running NVFP4 or MoE models.

## Contents

| File | Type | Impact |
|------|------|--------|
| `PR_fused_norm_fp4.md` | Code PR | 2.95x faster RMSNorm+FP4 quant, ~12% end-to-end gain |
| `PR_moe_inductor_default.md` | Feature request / bug | 2.1x peak throughput with two flags |

## PR 1 — Fused RMSNorm + Dynamic FP4 Block Quantization

**File:** `PR_fused_norm_fp4.md`

**The kernel:** `csrc/quantization/fused_kernels/rms_norm_dynamic_fp4_quant.cu`

Profiling Gemma4 26B decode shows RMSNorm (called ~60× per step across 30
layers) accounts for 26% of decode time (4.1 ms of 15.5 ms at B=32). The
current vLLM path runs two separate ops sequentially:

```
vllm_c::rms_norm(input) -> BF16 intermediate  (HBM write)
scaled_fp4_quant(BF16 intermediate)           (HBM read + write)
```

The BF16 intermediate tensor (M × N × 2 bytes) is written to and read from
HBM for no reason — it is only consumed by the immediately following
quantization step.

The fused kernel eliminates that intermediate: it reads input once, computes
norm and FP4 quantization in registers, and writes packed FP4 + scale factors
directly. On SM120 it uses the native `cvt.rn.satfinite.e2m1x2.f32` PTX
instruction for hardware FP4 conversion. On older GPUs a software fallback
handles the conversion.

**Results:**
- 2.95x faster per call on SM120 (Blackwell, CUDA 12.8)
- ~2.8x faster on typical shapes (M=1-128, N=3584-8192)
- Projected +12-13% end-to-end throughput for Gemma4 26B decode
- A `fused_add_rms_norm_dynamic_fp4_quant` variant handles the residual path

**Files that need to change in vLLM:**
1. Add `csrc/quantization/fused_kernels/rms_norm_dynamic_fp4_quant.cu`
2. `csrc/torch_bindings.cpp` — declare + register both ops
3. `CMakeLists.txt` — add to `FP4_ARCHS` source list
4. `vllm/_custom_ops.py` — add `@register_fake` stubs
5. `vllm/model_executor/layers/layernorm.py` — call fused op for NVFP4
6. `vllm/compilation/fusion_pass.py` — add FP4 pattern to `fuse_norm_quant`
   pass (currently only FP8 patterns are registered)

---

## PR 2 — Default to No-Inductor for NVFP4 MoE Serving

**File:** `PR_moe_inductor_default.md`

vLLM defaults to `torch.compile` with the inductor backend. For NVFP4 MoE
models (Gemma4, and others using CUTLASS grouped GEMM), this cuts peak serving
throughput by 2.1x.

| Config | B=1 tok/s | Peak serving tok/s |
|--------|-----------|-------------------|
| Default (inductor + piecewise CUDA graphs) | **127** | 3,112 |
| `-cc.mode none -cc.cudagraph_mode full` | 89 | **6,615** |

**Why inductor hurts:**
1. MoE hot path is already fully fused in C++ (6 kernel launches/layer via
   CUTLASS grouped GEMM, not 128 separate expert calls)
2. The custom ops (`cutlass_fp4_moe_mm`, `scaled_fp4_experts_quant`) are opaque
   to inductor — it cannot optimize them
3. Inductor requires piecewise CUDA graph capture (broken at custom op
   boundaries). Full CUDA graphs (possible without inductor) are more efficient
4. The ops inductor CAN fuse (small norms, activations) are 3-6% of step time.
   The tracing overhead exceeds what it saves at any meaningful concurrency.

**Trade-off:** Inductor wins for single-request interactive use (+43% B=1
latency). The crossover is C≈8-16. For any serving deployment, disable it.

**Proposed fix:** Auto-detect NVFP4/MoE models and set
`-cc.mode none -cc.cudagraph_mode full` by default, or warn users on startup.

---

## Source Data

All benchmarks run on:
- RTX 5090 (SM120 Blackwell), 32 GB VRAM, 1792 GB/s HBM, 96 MB L2
- vLLM 0.19.1rc1 (Docker)
- CUDA 12.8
- Model: `cklaus/gemma-4-26B-A4B-it-NVFP4` (128 experts, top-8, 30 layers)

Full benchmark tables and decode step decomposition are in
`/root/projects/autokernel/GEMMA4_NVFP4_BENCHMARKS.md`.

Full discovery log (all 16 findings from this session) is in
`/root/projects/autokernel/EXPERIMENT_DISCOVERIES.md`.

The kernel source is at:
`/root/projects/autokernel/kernels/csrc/rms_norm_dynamic_fp4_quant.cu`

The torch bindings patch is at:
`/root/projects/autokernel/kernels/csrc/torch_bindings_patch.cpp`
