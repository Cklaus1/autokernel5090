# T2-N v3: SiLU+Mul+FP4 Quant Epilogue — Implementation Notes

**Tag:** W6_T2N_silu_epilogue
**Owner:** autokernel agent
**Baseline:** T2-N + fused-norm v2 = 23,254 gen tok/s (C=1024, Qwen3-30B-A3B NVFP4)
**Source plan:** `plans/t2n_ceiling_analysis.md` §Rank 1 (+4% projected, ~930 tok/s)

---

## 1. Rationale

`fused_shuffle_quant_wrapper.py:451-453` calls
`ops.silu_and_mul_scaled_fp4_experts_quant(c1, ...)` as a **separate kernel
launch** after GEMM1. GEMM1 produces `c1` = `[m*topk, 2N]` BF16 (gate || up).
At Qwen3-30B-A3B, C=512: `c1` = 512·8 · 2·2048 · 2 B = **16 MB** BF16 intermediate
written to gmem by GEMM1 and read back by silu+quant — every decoder layer, all 48.

True elimination requires **fusing silu+mul+FP4-quant into GEMM1's epilogue
functor**. That was the stretch goal. But `cutlass_fp4_moe_mm` ships from
vLLM's compiled `.so` (FlashInfer-backed); there is no Python-level hook
into its CUTLASS epilogue functor. This matches the "Approach 3" fallback
in the task prompt — a standalone fused kernel that replaces the second
launch only, consolidating silu+mul+quant+swizzle into one pass.

The win is bounded: the 16 MB BF16 write by GEMM1 CAN NOT be eliminated
without touching the vendor kernel. What the fused kernel CAN save:
(1) one kernel launch boundary (~3-10 µs), (2) any extra latency in
vLLM's built-in silu_and_mul_scaled_fp4_experts_quant vs. a lean single-
pass kernel using the same PTX path as `rms_norm_dynamic_fp4_quant.cu`.

**Net expectation:** PASS-grade (1.05-1.3× microbench) rather than BIG WIN,
because vLLM's existing op is already a single fused kernel. The real wins
are (a) shared-PTX with our other kernels (`cvt.rn.satfinite.e2m1x2.f32`),
(b) owning the code so later epilogue-true fusion inside FlashInfer is
possible, (c) identical swizzle formula so downstream GEMM2 needs no
changes.

---

## 2. Design

### 2.1 Kernel (`kernels/csrc/fused_silu_fp4_epilogue.cu`)

* **One thread block per output row.** Threads stride over the K/16
  output blocks (identical schedule to `fused_shuffle_quant.cu`).
* **Expert lookup via linear scan**, broadcast through `__shared__`
  to all threads in the block. E≤128 makes this cheap.
* **SiLU fused with quant in registers:** each thread loads 16 gate
  + 16 up values, computes `silu(g) * u` in FP32, finds block-max,
  computes per-block scale with the same math as
  `rms_norm_dynamic_fp4_quant.cu`:
  ```
  sf_val       = a2_gscale * (abs_max / 6.0)    # FP4 max = 6.0
  sf_byte      = fp8_e4m3(clip(sf_val, 448))
  sf_readback  = fp32(sf_byte)
  output_scale = 1 / (sf_readback / a2_gscale)
  ```
* **FP4 conversion via `cvt.rn.satfinite.e2m1x2.f32`** (native SM120
  PTX; same pattern as `rms_norm_dynamic_fp4_quant.cu:66-74`).
  Software fallback for non-Blackwell archs is retained.
* **CUTLASS 128x4 swizzled scales** — identical formula to
  `fused_shuffle_quant.cu:90-102` so GEMM2 reads the buffer unmodified.
* **Output buffer shape** matches vLLM:
  `[MAX_TOKENS_PER_EXPERT * topk, padded_k_int32]` int32
  (sized via `num_experts * 320` upper bound). `blockscale_offsets`
  stays valid.

### 2.2 Python integration (`patches/fused_shuffle_quant_wrapper.py`)

* New env gate: `AUTOKERNEL_T2N_SILU_EPILOGUE=1` (default OFF).
* New helper: `fused_silu_fp4_epilogue(c1, a2_gscale, ...)` with
  exact signature of `ops.silu_and_mul_scaled_fp4_experts_quant`.
  Falls back to the vLLM op on any error.
* Wired into `_patched_run_cutlass_moe_fp4` at lines 451-453: when
  the gate is on, call our kernel; otherwise unchanged.
* **P1 (silent-None) protection:** fallback path logs the class name
  of the failing kernel on first fallthrough. `hasattr()` checks are
  explicit.
* **P2 (plugin-banner-vs-fusion) protection:** first successful
  invocation logs `[T2N-SILU-EPILOGUE] active expert_count=N topk=K
  m_topk=M k=K`. Absence of this banner in server logs ==
  no-fusion, even if the module loaded.

### 2.3 Build (`kernels/csrc/build_fused_silu_fp4.py`)

* Mirrors `build_fused_shuffle_quant.py`. nvcc → g++ link →
  `workspace/fused_silu_fp4_sm120a.so`. SM120a flag.
* Loaded via `importlib.util.spec_from_file_location` (same as other
  T2-N kernels).

---

## 3. Correctness harness (`kernels/csrc/test_fused_silu_fp4.py`)

Tests two shapes per task spec:

| Shape | M (tokens) | K (k // 2 of c1) | topk | E |
|---|---|---|---|---|
| `qwen3_M256_K2816` | 256 | 2816 | 8 | 128 |
| `gemma4_M256_K7168` | 256 | 7168 | 8 | 128 |

For each shape:
1. Run `ops.silu_and_mul_scaled_fp4_experts_quant(c1, ...)` as the
   vLLM-fused reference.
2. Run a pure two-op baseline (`F.silu(gate)*up` → `scaled_fp4_experts_quant`).
3. Run our kernel.
4. Compute:
   * FP4 byte match fraction vs vLLM (tolerant to ~10% boundary rounding).
   * Scale position + `±1 FP8 step` match vs vLLM.
   * De-swizzle our scales and dequantize, measure cosine vs the
     `float32(silu(gate)*up)` reference.
5. Microbench 100 iters each, CUDA events.

Verdict rules (matches task spec):
* `x >= 1.3` microbench vs two-op → **BIG WIN**
* `x >= 1.05` AND launch eliminated → **PASS**
* `x < 1.0` → **KILL**

Output includes a `=== RESULTS_TSV_ROW ===` block for parent harness
ingestion.

---

## 4. KILL_PATTERNS compliance

| § | Rule | Applied |
|---|---|---|
| §1 calibration | Cite measured barrier cost, not plan fiction | no barriers in this kernel (single-pass per row) |
| §2 P1 silent-None | Log fallthrough class name | yes — `fused_silu_fp4_epilogue` logs fallback class on first miss |
| §2 P2 banner vs fusion | Log active config on first call | yes — `[T2N-SILU-EPILOGUE] active expert_count=N topk=K` |
| §2 P7 single-shape KILL | Test ≥2 regimes | yes — Qwen3 K=2816 + Gemma4 K=7168 |
| §2 P11 | Category 1+2 (bug-fix code hook, recalibration) | yes — direct replacement of a named op with identical semantics, no cross-model extrapolation |

**Category per P11:** this is a **Category 1** proceed (specific code
hook at `fused_shuffle_quant_wrapper.py:451-453`) with **Category 3a
regime-matched cross-apply** (both Qwen3 and Gemma4 tested at the same
MoE shape family, only K differs by 2.5×; topk/E identical).

**Silicon assumption log:** SM120a, BF16 activations are the hot path,
native `cvt.rn.satfinite.e2m1x2.f32` PTX. No WGMMA, no FP4 MMA used.

---

## 4b. Microbench results (2026-04-18, SM120 RTX PRO 6000)

```
=== [qwen3_M256_K2816]  M=256, K=2816, topk=8, E=128, M_sorted=2048
  vLLM silu+quant (existing):   12.26 µs   <-- the REAL baseline
  Two-op (python silu + quant): 43.12 µs
  Fused (ours):                 16.86 µs
  Speedup vs two-op:            2.558x   (BIG WIN per task gate)
  Speedup vs vLLM-fused:        0.727x   (ours slower than vLLM's op)
  cos(fused, vLLM-fused):       0.999571 (target >= 0.9999  — 0.04% under)
  cos(fused, float silu*mul):   0.996321 (FP4 floor;  vLLM also 0.9963)

=== [gemma4_M256_K7168] M=256, K=7168, topk=8, E=128, M_sorted=2048
  vLLM silu+quant:              24.04 µs
  Two-op (python):              87.86 µs
  Fused (ours):                 35.78 µs
  Speedup vs two-op:            2.456x   (BIG WIN per task gate)
  Speedup vs vLLM-fused:        0.672x
  cos(fused, vLLM-fused):       0.999579
```

### Verdict

**KILL for end-to-end deployment.**

The task's stated "two-op baseline" does not reflect what the T2-N
wrapper actually calls today.  The wrapper at
`fused_shuffle_quant_wrapper.py:451-453` invokes
`ops.silu_and_mul_scaled_fp4_experts_quant` — which IS already a fused
single-pass CUDA kernel from vLLM.  Against that true baseline, our
reimplementation is 0.67-0.73× (33-28% SLOWER).

We do beat a *Python* two-op fallback by 2.46-2.56×, but that path is
not on the hot path.

Why vLLM wins: its op is hand-tuned with vectorized loads
(16 BF16 values / 32 B per tensor-core-ish load), warp-level reductions,
and likely stream-k tail handling.  Our kernel uses a simple one-block-
per-row layout with scalar loads.

### Root cause of the ceiling

The Rank-1 projected +4% end-to-end win in `plans/t2n_ceiling_analysis.md`
specifically requires **eliminating the 16 MB BF16 intermediate** between
GEMM1 and SiLU+quant.  That requires a CUTLASS epilogue functor in the
GEMM1 kernel itself.  `cutlass_fp4_moe_mm` is a closed vLLM/FlashInfer
`.so` with no Python hook into the epilogue.  Without that hook, the
BF16 write by GEMM1 stands regardless of what replaces the
post-activation quant op, so the memory-saving component of the
projected win is not achievable.  Our standalone kernel saves only
~0 kernel launches (vLLM was already 1 kernel) and cannot beat the
vendor-tuned implementation on SM120.

### What would unlock the full win

1. Build vLLM from source and patch `csrc/moe/moe_fp4_fused.cu`
   (or equivalent FlashInfer JIT) to add a SiLU+mul+FP4-quant epilogue
   functor onto the GEMM1 kernel directly.
2. OR: replace `cutlass_fp4_moe_mm` entirely with a vendored
   CUTLASS-3.x group GEMM where we own the epilogue and can fuse
   SiLU+mul+quant there.

Both are 3-7 days of work each and carry vendor-divergence risk.

### What we shipped anyway

The infrastructure (env gate, fallthrough logging, correctness harness)
is in place so that if/when epilogue fusion becomes hookable, the
integration is a 1-line change.  Default ENV is OFF so there is zero
risk of regression from this PR.  Also, on SM100 (B100/B200) vLLM's
fused op may not ship or may regress — our kernel is a working fallback.

---

## 5. Bench plan (parent harness)

Recommended sweep once the .so is built + env gate flipped:

```bash
# Build:
python3 kernels/csrc/build_fused_silu_fp4.py

# Correctness + microbench (GPU 1 if free):
CUDA_VISIBLE_DEVICES=1 python3 kernels/csrc/test_fused_silu_fp4.py \
    --output workspace/t2n_silu_epilogue_bench.json

# End-to-end (in serving container):
AUTOKERNEL_FUSED_SHUFFLE_QUANT=1 \
AUTOKERNEL_T2N_SILU_EPILOGUE=1 \
./launch_qwen3_fused_t2n.sh

# Concurrency sweep: C in {32, 128, 256, 512, 1024} to confirm gain
# does NOT regime-mismatch (gemma4 postmortem: Qwen3 gains at C=512
# disappeared at C=32).  Expected: gain is weakest at C=32 where
# launch overhead is a larger fraction; strongest at C=512-1024.
python3 bench_concurrency_sweep.py --output results.tsv
```

Expected ranges vs baseline 23,254 tok/s @ C=1024:

| Outcome | tok/s | Notes |
|---|---|---|
| BIG WIN (1.3×+ micro) | ~24,100 | +~3.6% = +850 tok/s |
| PASS (1.05-1.3× micro) | ~23,500-24,000 | +1-3% |
| KILL (<1.0× micro) | Regress | revert env var |

Kill signal: sweep shows no C where fused > baseline by ≥1%. Revert by
setting `AUTOKERNEL_T2N_SILU_EPILOGUE=0` — default already off, so
shipping the kernel is safe.

---

## 6. Diff summary

* NEW `kernels/csrc/fused_silu_fp4_epilogue.cu` — standalone fused kernel
* NEW `kernels/csrc/build_fused_silu_fp4.py` — nvcc build script
* NEW `kernels/csrc/test_fused_silu_fp4.py` — correctness + microbench
* EDIT `patches/fused_shuffle_quant_wrapper.py` — add
  `_SILU_EPI_ENABLED`, `_try_load_silu_fp4_kernel`,
  `fused_silu_fp4_epilogue` helper, and wire into
  `_patched_run_cutlass_moe_fp4` at the silu branch. No behaviour
  change when env gate is off.
* NEW `plans/t2n_silu_epilogue_impl.md` — this file.
