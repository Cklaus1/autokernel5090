# AutoKernel Day 1 Status

## Task
Optimize a W4A16 quantized matrix multiplication Triton kernel for Qwen3.5:35B model shapes on RTX 5090 (Blackwell).

## Results
| Metric | Baseline | Current Best | Improvement |
|--------|----------|-------------|-------------|
| Throughput | 15.1 TFLOPS | 196.1 TFLOPS | **13.0x** |
| vs PyTorch | 0.23x | 3.04x | **13.2x** |
| % Peak (419 TFLOPS) | 3.6% | 46.8% | |
| Latency (large shape) | 7095 µs | 548 µs | **12.9x** |

## Key Discoveries (chronological)

1. **Autotune + L2 swizzle** (exp 21): 15→137 TFLOPS. Tile reordering for L2 cache locality + broad config sweep.
2. **Two-level K tiling** (exp 29): 137→144. Hoist scale/zero loads outside inner K loop.
3. **Persistent kernel** (exp 32): 144→153. Cap grid to 680 programs (4x SMs), outer tile loop.
4. **Flat K loop** (exp 36): 155→170. Remove nested group loop — Triton's software pipelining works better with a single flat loop.
5. **Constexpr group_size** (exp 39): 170→177. Make QUANT_GROUP_SIZE a tl.constexpr for compile-time division optimization.
6. **Split dequant + cuBLAS** (exp 60): 177→188. **Paradigm shift**: separate the INT4→FP16 dequantization (Triton, 53µs) from the matmul (cuBLAS, 500µs). cuBLAS FP16 matmul at 215 TFLOPS beats fused Triton kernel (175 TFLOPS) even with dequant overhead.
7. **Transposed dequant + F.linear** (exp 61): 188→196. Store dequanted weights as [N,K] and use F.linear (cuBLAS NT gemm) instead of torch.mm (NN gemm). NT layout is ~4% faster.

## Lessons Learned

### What worked
- **Autotune broadly, then narrow**: Start with many configs, let the autotuner find winners, then focus configs around them.
- **Persistent kernels**: For shapes with many tiles, the persistent pattern with L2 swizzle is consistently beneficial.
- **Flat loops over nested loops**: Triton's compiler pipelines single loops much better than nested ones. Even if a two-level loop is "smarter" (hoisting invariant loads), the flat loop wins because the compiler can overlap loads across iterations.
- **Constexpr parameters**: Making runtime values into tl.constexpr lets the compiler turn divides into shifts and eliminate dead code.
- **Separation of concerns**: The biggest leap came from NOT trying to fuse everything. cuBLAS's FP16 matmul is highly optimized (SASS-level, TMA loads, warp specialization) — beating it with Triton for pure matmul is extremely hard. By splitting dequant and matmul, each can use its best implementation.
- **NT matmul layout**: cuBLAS strongly prefers NT (A @ B^T) over NN (A @ B) for these shapes.

### What didn't work
- **Larger BLOCK_SIZE_K** (128): Causes register spill, always regressed.
- **Wider BLOCK_SIZE_N** (256): Too much register pressure at 8 warps.
- **Fewer programs**: 170 (1x SM) or 340 (2x SM) are too few — 680 (4x SM) is the sweet spot.
- **Eviction policy hints**: `evict_first`/`evict_last` on loads had no measurable effect.
- **`max_num_imprecise_acc`**: No performance difference on Blackwell.
- **Non-persistent 2D grid**: Removing the persistent loop always regressed.
- **Reordering loads**: Compiler handles scheduling; source order doesn't matter much.
- **CUDA graphs**: The copy_ overhead for input buffers negates any launch overhead savings.
- **FP8 matmul**: 2x faster matmul BUT correctness fails — FP8 E4M3 (3 mantissa bits) can't represent dequanted weights accurately enough. Even row-wise scaling only gets 5% within tolerance.
- **Higher num_stages** (5-6): More pipeline stages cause register spill at these tile sizes.

### Architecture insights
- **RTX 5090 Blackwell**: 170 SMs, 419 TFLOPS FP16 peak, 1792 GB/s bandwidth. Ridge point at 234 FLOP/byte.
- **W4A16 is compute-bound**: Arithmetic intensity ~1920 FLOP/byte, well above ridge point. The bottleneck is purely the tensor core utilization and dequant ALU overhead.
- **Dequant overhead is inherent**: The INT4 unpacking (shift, mask, cast, subtract, multiply) adds ~5 ALU ops per weight element that must complete before the tensor core can run, creating pipeline bubbles.
- **cuBLAS ceiling**: Even cuBLAS only achieves ~215 TFLOPS (51% of peak) for M=2048 FP16 matmul. This suggests hardware-level constraints (warp scheduling, memory subsystem) limit utilization.

## Current State
- kernel.py: Triton dequant kernel + F.linear (cuBLAS NT gemm)
- 196 TFLOPS, 46.8% of peak, 3.04x vs PyTorch
- Decode (M=1): 82µs, 14.5x vs PyTorch

## What to try next
- Optimize dequant kernel further (currently 50µs, theoretical minimum 37µs at bandwidth limit)
- Profile with Nsight Compute to find actual bottlenecks in cuBLAS matmul
- Try `torch.compile` or custom cuBLAS workspace settings
- Explore streaming overlap: pipeline dequant of chunk N+1 with matmul of chunk N
- Consider INT8 tensor cores with dynamic activation quantization (risky for correctness)
