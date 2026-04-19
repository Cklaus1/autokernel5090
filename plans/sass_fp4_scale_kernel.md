# SASS-level native FP4 scale kernel — design doc

## Premise audit (important finding)

The task brief assumed CUTLASS's `scaled_fp4_experts_quant` uses **software FP8 E4M3
conversion** for scale factors. Reading the actual source in
`/build/vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh`:

1. **The FP4 data conversion already uses native `cvt.rn.satfinite.e2m1x2.f32`**
   (lines 72-89 and 118-147 — identical to the PTX we use in
   `rms_norm_dynamic_fp4_quant.cu`). So Discovery #10's speedup does *not*
   apply to this op — it's already there.

2. **The FP8 scale factor uses the CUDA runtime cast** `__nv_fp8_e4m3(SFValue)`
   (line 256). On SM89+/SM100+, this *should* lower to native PTX
   `cvt.rn.satfinite.e4m3x2.f32`, but when the readback `float(tmp)` is
   also needed, the compiler sometimes emits redundant round-trips.

Therefore the actual design question is:
- **Can we collapse the `fp32 → fp8 → uint8 + fp32-readback` round-trip
  into fewer PTX instructions via explicit inline asm?**

## What CUTLASS currently does (scale path)

In `cvt_warp_fp16_to_fp4` (nvfp4_utils.cuh):

```cpp
float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
__nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);      // fp32 → fp8 (software OR PTX)
reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp; // store 8 bits
SFValue = float(tmp);                             // fp8 → fp32 readback
```

Disassembly on SM120a (compiled with CUDA 13.0) shows this lowers to
`F2FP.E4M3.PACK_AB.RP` on SM100+ (it's a SASS instruction), BUT the readback
`float(tmp)` calls `__nv_cvt_fp8_to_halfraw` which goes through FP16 then to FP32.

## Our replacement

We use **explicit inline PTX** for both directions:

```asm
cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;  // two fp32 → two fp8 → one .b16
```

And for the readback, we use `cvt.rn.f16x2.e4m3x2` + `cvt.f32.f16` which lowers
to a single `F2FP` instruction — avoiding the double-round in the compiler path.

Additionally: the FP4 conversion path is already optimal, so our kernel keeps
the exact same PTX (`cvt.rn.satfinite.e2m1x2.f32`) — we just isolate the
scale computation to a standalone op so we can benchmark the scale-path
improvement separately.

## Correctness criterion

Byte-identical output vs CUTLASS baseline at M∈{128,512}, K=7168, single
expert (trivial experts layout: offsets=[0, M]). This is achievable because:
- FP4 path uses the exact same PTX.
- FP8 path uses the same `cvt.rn.satfinite.e4m3x2.f32` which is also what
  `__nv_fp8_e4m3(x)` lowers to on SM100+.

If outputs differ by >1 FP4 step OR >1 FP8 ULP on the scales, we KILL.

## Integration path

Own `torch.ops._C.native_fp4_scale_quant(input, global_scale) -> (fp4, sf)`
registered via `TORCH_LIBRARY_FRAGMENT`. Hot-swappable behind
`cutlass_fp4_moe_mm` — replaces the scaled_fp4_experts_quant call in the
single-expert (N=1) degenerate case. For the multi-expert case we keep
the CUTLASS kernel (offsets logic is expert-aware, not relevant to the
scale instruction optimization).

## Shapes targeted

Gemma4-26B MoE decode step: m_topk ∈ {128, 512}, K=7168.
Scale output layout: swizzled `[numMTiles, numKTiles, 32, 4, 4]` FP8 E4M3
(identical to CUTLASS).

## Results (measured on GPU 1, PRO 6000 Blackwell SM120a, CUDA 13.0)

**Correctness:** byte-identical at M=128 and M=512, K=7168.
`fp4_byte_match=1.0000`, `sf_byte_match=1.0000`, max FP4 nibble diff = 0.

**Kernel-only microbench (preallocated outputs, no Python wrapper overhead):**

| M    | K    | Native (us) | CUTLASS (us) | Speedup |
|------|------|-------------|--------------|---------|
|  128 | 7168 |  8.23       |  8.88        | 1.08x   |
|  512 | 7168 |  9.50       |  9.56        | 1.01x   |
| 2048 | 7168 | 11.26       | 12.31        | 1.09x   |
| 8192 | 7168 | 54.78       | 57.80        | 1.06x   |

**Average speedup: 1.06x.**

**Key finding:** The original Discovery #10 2.95× speedup hypothesis does *not*
apply here. Inspection of the actual CUTLASS source shows it already uses
`cvt.rn.satfinite.e2m1x2.f32` for FP4 and `cvt.rn.satfinite.e4m3x2.f32` for
FP8 scale (via `__nv_fp8_e4m3(x)` which lowers to the same PTX on SM89+).
We confirmed byte-identical output with our explicit PTX kernel. The small
1.06× win comes from lighter-weight launch path (no expert-search code,
simpler grid shape), not from the conversion instruction itself.

**e2e projection:** The scale path is a small fraction of an MoE forward
pass (~1-2% of total decode latency on Gemma4-26B). A 1.06× speedup on
that fraction yields **~0.1% e2e** — not worth productionizing.

## Recommendation: SKIP e2e integration

CUTLASS's scaled_fp4_experts_quant is already using the native PTX. The
"software FP8 conversion" premise was incorrect — the bottleneck we hoped
to eliminate doesn't exist. Do not invest further time here; instead
pursue T2-N (fused shuffle+quant) which is orthogonal and has a real
latency floor to remove (HBM round-trip elimination, not instruction count).

## Artifacts

- `/home/cklaus/projects/autokernel/kernels/csrc/native_fp4_scale_kernel.cu` (kernel, 200 LoC)
- `/home/cklaus/projects/autokernel/kernels/csrc/build_native_fp4_scale.py` (build script)
- `/home/cklaus/projects/autokernel/kernels/csrc/test_native_fp4_scale.py` (correctness + bench)
