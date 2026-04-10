# Fused RMSNorm + FP4-E2M1 Quantization CUDA Kernel

## Overview

A single CUDA kernel that fuses RMSNorm normalization with FP4-E2M1 block quantization,
eliminating the BF16 intermediate that would otherwise be written to and read from global
memory between the two operations.

**Measured performance: 2.6-2.95x faster** than separate RMSNorm + scaled_fp4_quant on
RTX 5090 (SM120 Blackwell), across all tested shapes.

## Files

| File | Description |
|------|-------------|
| `csrc/rms_norm_dynamic_fp4_quant.cu` | CUDA kernel source (both norm-only and fused-add variants) |
| `csrc/torch_bindings_patch.cpp` | Torch op registration code for vLLM integration |
| `csrc/build_and_install.py` | Build script: compiles, links, loads, and verifies |
| `csrc/benchmark.py` | Performance benchmark vs separate ops |
| `fused_norm_fp4.py` | Triton reference implementation (for comparison) |

## Kernel Variants

### `rms_norm_dynamic_fp4_quant`

```
Input:  x [M, N] bf16/fp16, weight [N] bf16/fp16, global_scale [1] float32
Output: out_fp4 [M, N/2] uint8 (packed), block_scale [M, N/16] fp8_e4m3fn
```

Per row:
1. Compute variance = sum(x[i]^2) / N using CUB block reduction
2. rrms = rsqrt(variance + epsilon)
3. For each block of 16 elements:
   - Normalize: x_norm[j] = x[j] * rrms * weight[j]
   - Block max: abs_max = max(|x_norm[j]|)
   - Scale factor: sf = global_scale * abs_max / 6.0 (FP4 max)
   - Store sf as FP8-E4M3, read back for exact consistency
   - Quantize using `cvt.rn.satfinite.e2m1x2.f32` PTX (SM120a native)
   - Pack pairs into uint8

### `fused_add_rms_norm_dynamic_fp4_quant`

Same as above, but also fuses the residual connection:
```
hidden = input + residual  (stored back to residual for skip connection)
output = FP4_quant(RMSNorm(hidden, weight, epsilon))
```

## Build and Test

```bash
# Inside Docker container with vLLM + CUDA 12.8+
docker cp kernels/csrc/ vllm-gemma4:/tmp/csrc/
docker exec vllm-gemma4 python3 /tmp/csrc/build_and_install.py

# Run benchmark
docker exec vllm-gemma4 python3 /tmp/csrc/benchmark.py
```

### Build Requirements
- CUDA 12.8+ with nvcc
- SM120a architecture (Blackwell, e.g., RTX 5090)
- PyTorch 2.7+ with CUDA support

## Benchmark Results (RTX 5090)

```
          Shape   Separate (us)    Fused (us)   Speedup
  (   1,  3584)          24.6us         8.8us    2.79x
  (   1, 14336)          24.2us         9.1us    2.65x
  (   4,  3584)          23.8us         8.4us    2.84x
  (   8,  3584)          28.2us         9.5us    2.95x
  (  32,  3584)          22.7us         8.9us    2.56x
  ( 128,  3584)          23.6us         8.8us    2.67x
```

### Gemma4 26B Decode Impact
- Per norm+quant pair: 23.9us -> 8.3us (2.89x)
- 60 layers/step: 1434us -> 495us saved
- **Estimated throughput: +12.9% (121 -> 137 tok/s)**

## Integration with vLLM

### Runtime Loading (current approach)
The kernel compiles into a standalone `.so` and registers in the `_C` namespace
using `TORCH_LIBRARY_FRAGMENT`. After loading:

```python
torch.ops.load_library("/tmp/build_fused_rms_norm_fp4/fused_rms_norm_fp4.so")
# Now available as:
# torch.ops._C.rms_norm_dynamic_fp4_quant(...)
# torch.ops._C.fused_add_rms_norm_dynamic_fp4_quant(...)
```

### Upstream PR (for permanent integration)
1. Add `rms_norm_dynamic_fp4_quant.cu` to `csrc/quantization/fused_kernels/`
2. Patch `csrc/torch_bindings.cpp` (see `torch_bindings_patch.cpp`)
3. Add to `CMakeLists.txt` FP4_ARCHS section
4. Register fake tensors in `vllm/_custom_ops.py`
5. Apply fusion pass patch (`patches/vllm_fp4_norm_quant_fusion.py`)

### Fusion Pass Integration
The pattern matching infrastructure at
`patches/vllm_fp4_norm_quant_fusion.py` automatically activates when
`torch.ops._C.rms_norm_dynamic_fp4_quant` exists. It detects:
```
rms_norm(x, w, eps) -> scaled_fp4_quant(normed, global_scale)
```
and replaces with the single fused kernel call.

## Technical Notes

### FP4-E2M1 Hardware Conversion
On SM120a (Blackwell), the kernel uses the native PTX instruction
`cvt.rn.satfinite.e2m1x2.f32` for FP4 conversion. This converts two float32
values to a packed byte of two 4-bit E2M1 values in a single instruction.

The software fallback (for non-120a targets) uses a lookup table approach
with threshold comparisons.

### Scale Factor Layout
Supports both:
- **Row-major**: `[M, N/16]` - simple row-major storage
- **Swizzled (CUTLASS 128x4)**: For compatibility with CUTLASS FP4 GEMM kernels,
  using the `[numMTiles, numKTiles, 32, 4, 4]` tile layout

### Correctness
Verified 98-99% byte-exact match with separate RMSNorm + scaled_fp4_quant.
The ~1-2% difference comes from:
- Different order of FP8 scale factor rounding
- Fused kernel does norm+quant in FP32 registers (higher precision intermediate)
