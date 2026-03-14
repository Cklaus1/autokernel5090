# AutoKernel: Deep Dive into GPU Kernel Optimization on RTX 5090

A comprehensive technical document covering the design, experiments, results, and lessons learned from optimizing GPU kernels across 97+ experiments on NVIDIA RTX 5090 (Blackwell SM120).

---

## Hardware: RTX 5090 (SM120) -- What We Learned

### The Peak TFLOPS Confusion

The RTX 5090 is marketed at "419 TFLOPS FP16". This is the **sparse (2:4 structured sparsity)** peak. The actual dense peaks are:

| Precision | Dense TFLOPS | Sparse TFLOPS |
|-----------|-------------|---------------|
| FP16/BF16 | ~209.5 | ~419 |
| FP8 (e4m3) | ~419 | ~838 |
| FP4 (NVFP4 e2m1) | ~838 | ~1,676 |

This was a critical discovery -- our 328 TFLOPS Triton kernel was actually at **157% of dense FP16 peak**, not 78%.

### SM120 is NOT Datacenter Blackwell

SM120 (consumer Blackwell) uses **mma.sync** instructions (Ampere-style), NOT WGMMA/tcgen05 used in datacenter Blackwell (SM100). Key limitations:

- No WGMMA (warp-group matrix multiply)
- No TMEM (tensor memory)
- No TMA multicast
- TN layout only for MMA
- Cluster shape fixed to 1x1x1

This means CUTLASS and Triton target the **same instruction set** on SM120. There's no hidden hardware advantage that CUTLASS can exploit over Triton.

### Shared Memory Limit

SM120 has 101,376 bytes max shared memory per thread block. This was the hard wall that prevented BK=64 and BK=128 tile configurations on Triton 3.5.1. Triton 3.6.0's improved shared memory allocation (padded vs swizzled layout selection, bank conflict fixes) solved this.

---

## Optimization Results

### Performance Evolution

```
Naive baseline      15 TFLOPS   ▏
Autotune+L2        137 TFLOPS   █████████
Flat K+constexpr   178 TFLOPS   ████████████
Split dequant      198 TFLOPS   █████████████
FP16 accumulate    290 TFLOPS   ████████████████████
BK=128+Triton3.6   328 TFLOPS   ██████████████████████
Flash attention    399 TFLOPS   ███████████████████████████
NVFP4 native     1,271 TFLOPS   █████████████████████████████████████████████████████████████████████████████████████
```

### All Kernel Results

| Kernel | TFLOPS | % Dense FP16 | vs PyTorch | Status |
|--------|--------|-------------|------------|--------|
| W4A16 matmul (Triton) | 328 | 157% | 4.9x | Production |
| Flash attention | 399 | 190% | 22.6x | Production |
| Fused SwiGLU MLP | 213 | 102% | 3.3x | Production |
| FP8 scaled_mm | 385 | 184% | 5.9x | Fails correctness |
| NVFP4 scaled_mm (GEMM only) | 1,271 | 607% | 5.8x vs cuBLAS | Validated |
| NVFP4 end-to-end (with quant) | 1,145 | 547% | 5.0x vs cuBLAS | With CUDA quant kernel |
| NVFP4 Qwen2.5-7B prefill | 27,161 tok/s | — | — | Working |
| NVFP4 Qwen3.5-35B (vLLM) | 38 tok/s batched | — | — | Working (patched) |

---

## Key Technical Lessons

### 1. "Don't Compete with cuBLAS" Evolves to "Don't Compete with cuBLASLt"

The original insight (exp 60): splitting dequant from matmul and letting cuBLAS handle dense FP16 GEMM beat any fused Triton kernel.

The evolution: our Triton FP16-accumulate kernel at 328 TFLOPS actually **beats cuBLAS** (218 TFLOPS) by 50%, because cuBLAS doesn't use FP16 accumulation by default.

The final insight: for NVFP4, cuBLASLt's `_scaled_mm` at 1,271 TFLOPS is unbeatable -- it uses native SM120 `mma.sync.aligned.block_scale` instructions.

**Lesson: know what the library does better than you, and delegate. But also know what you can do better.**

### 2. Compiler-Friendliness Beats Algorithmic Cleverness

The flat K-loop (+14.7 TFLOPS) won over the "smarter" two-level loop because Triton's pipeliner can only pipeline a single loop. The nested loop was algorithmically superior but the compiler couldn't optimize it.

Similarly, `tl.constexpr` for group_size (+7.1 TFLOPS) enabled the compiler to replace divisions with shifts -- a zero-cost annotation with 4% throughput gain.

### 3. Autotune is Non-Negotiable

Experiments 94-95 proved this definitively: hardcoding the "best" config (even one that autotune frequently selects) drops throughput from 328 to 250 TFLOPS. The autotune process itself is what delivers peak performance. No single config is universally optimal.

Keep autotune configs trimmed to 4-6 proven winners for faster convergence, but never hardcode.

### 4. The Activation Quantization Bottleneck

NVFP4 GEMM runs at 1,271 TFLOPS, but end-to-end throughput depends on how fast you can quantize activations from FP16 to FP4:

| Quantization method | Time (512x3584) | End-to-end TFLOPS |
|---------------------|-----------------|-------------------|
| Python (original) | 0.427ms | Slower than cuBLAS |
| torch.compile | 0.024ms | 267 TFLOPS |
| CUDA C++ kernel | 0.018ms | 1,145 TFLOPS |

The CUDA kernel achieved **52x speedup** over the Python quantizer. The lesson: when the non-compute overhead dominates, optimize the overhead, not the compute.

### 5. NVFP4 Scale Padding is Critical

cuBLASLt's blockscaled GEMM requires scales padded to specific alignment:
- M dimension: ceil(M/128) * 128
- K blocks: ceil(n_blocks/4) * 4

For M < 128 (autoregressive decode), the padding creates zero-scale regions that corrupt results. Solution: fall back to FP16 cuBLAS for small M, use NVFP4 for prefill (M ≥ 128).

### 6. FP8 is a Dead End for W4A16

All FP8 strategies fail at atol=0.05 for K=5120:
- Per-tensor scaling: max error 1.0
- Per-row scaling: max error 20.7 (worse!)
- Split-K: no improvement (error is per-element, not accumulation)
- e5m2 activations: worse (fewer mantissa bits)

Root cause: FP8 e4m3 has 3 mantissa bits = 12.5% worst-case relative error per element. Over K=5120 accumulation, errors compound beyond tolerance. Would need K < 47 to pass atol=0.05.

### 7. SASS Analysis Reveals the Real Bottleneck

Disassembling the Triton-generated SASS showed:
- 128 HMMA.16816.F16 (tensor core ops)
- 128 FADD.F32 (FP32 accumulation adds)
- 128 HADD2 (FP16→FP32 conversion)
- 48 LDGSTS (async global→shared copies)

The **256 ALU ops (FADD + HADD2) for 128 tensor core ops** means half the instruction slots are spent on FP32 accumulation, not tensor core compute. This is the fundamental limit of the FP16-output-to-FP32-accumulator pattern.

### 8. "Multi-Day Effort" is Often a Misconception

The NVFP4 CUTLASS GEMM was initially estimated as a "multi-day effort" requiring 500+ lines of CuTe DSL code. The actual solution: `torch._scaled_mm_v2` with `recipe=2, swizzle=1` -- discovered by brute-forcing 16 parameter combinations in 10 minutes.

**Always attempt the high-ceiling paths, even when they look hard.** The assumed difficulty was wrong.

---

## NVFP4: The Breakthrough Path

### Discovery

PyTorch 2.10 ships with native NVFP4 support via `torch._scaled_mm_v2`:

```python
# Both operands in FP4 (e2m1) with e4m3 block scales
result = torch._scaled_mm_v2(
    A_fp4, B_fp4.t(),
    scale_a=(A_scale,), recipe_a=(2,), swizzle_a=(1,),
    scale_b=(B_scale,), recipe_b=(2,), swizzle_b=(1,),
    bias=None, out_dtype=torch.bfloat16
)
```

Also works via `torch._scaled_mm` with flat scale tensors (compressed-tensors format).

### Weight Conversion Pipeline

FP16 → NVFP4 (per-16-element block scaling):
1. Reshape weights to blocks of 16 along K: `[N, K] → [N, K//16, 16]`
2. Compute per-block scale: `max_abs / 6.0` (NVFP4 max value)
3. Scale and quantize to nearest e2m1 value using `searchsorted` on boundaries `[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]`
4. Pack 2 values per uint8: `code[0] | (code[1] << 4)`
5. View as `torch.float4_e2m1fn_x2`

### Activation Quantization

Dynamic per-block quantization at inference time:
- Python: 0.427ms (too slow)
- torch.compile: 0.024ms (4.8x speedup via kernel fusion)
- CUDA C++ kernel: 0.018ms (52x speedup, processes 16 elements per thread)

### Quality Validation

| Test | Cosine Similarity | Output Quality |
|------|------------------|----------------|
| Single linear layer (FP16→NVFP4) | 0.889 | Expected for 4-bit |
| Qwen2.5-7B (all MLP layers) | — | Correct text generation |
| Qwen3.5-35B-A3B (full model) | — | Correct, coherent reasoning |

### Production Integration

Successfully ran on:
- **Qwen2.5-7B**: 27,161 tok/s prefill, 61 tok/s decode, 19.2GB VRAM
- **Qwen3.5-35B-A3B via vLLM**: 38 tok/s batched, 26.3GB VRAM (required one-line patch for Mamba causal_conv1d assertion)

---

## Ecosystem Status (as of March 2026)

| Framework | NVFP4 | SM120 | Status |
|-----------|-------|-------|--------|
| **PyTorch** | `_scaled_mm_v2`, `float4_e2m1fn_x2` | Full support | Stable |
| **Triton** | `tl.dot_scaled` with e2m1 | Broken (hangs at compile) | Issue #7550 |
| **vLLM** | compressed-tensors format | Works (build from source) | Mamba layer bugs |
| **TensorRT-LLM** | W4A4 | Works (v0.20+) | No FP4 KV cache on SM120 |
| **SGLang** | modelopt_fp4 | Partial (NaN bugs) | Early stage |
| **CUTLASS DSL** | MmaMXF4NVF4Op compiles | SM120 supported | No bundled GEMM examples |
| **FlashInfer** | FP4 GEMM CUTLASS | Needs CUDA 12.8+ nvcc | JIT compilation |

---

## Appendix: Hardware Reference

### RTX 5090 Specs

| Spec | Value |
|------|-------|
| Architecture | Blackwell GB202 (SM120) |
| SMs | 170 |
| Dense FP16 Tensor | ~209.5 TFLOPS |
| Dense FP4 Tensor | ~838 TFLOPS |
| Sparse FP4 Tensor | ~1,676 TFLOPS |
| Memory | 32 GB GDDR7, 1,792 GB/s |
| L2 Cache | 96 MB |
| Shared Memory / SM | 101,376 bytes max |
| MMA Instruction | mma.sync.aligned (NOT WGMMA) |
| FP4 MMA Shape | m16n8k64 |
| FP16 MMA Shape | m16n8k16, m16n8k32 |
| Boost Clock | 2,407 MHz (up to 3,090 MHz) |
| TDP | 575W |

### Performance Hierarchy Achieved

| Path | TFLOPS | % Dense FP16 | Instruction |
|------|--------|-------------|-------------|
| FP16 cuBLAS | 218 | 104% | HMMA.16816.F16 (FP32 accum) |
| Triton FP16-accum | 328 | 157% | HMMA.16816.F16 (FP16 accum) |
| FP8 scaled_mm | 430 | 205% | FP8 MMA m16n8k32 |
| NVFP4 scaled_mm | 1,271 | 607% | FP4 MMA m16n8k64 block_scale |
