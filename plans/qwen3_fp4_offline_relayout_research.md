# Qwen3 FP4 Offline Re-Layout Research

**Tag:** `W7_qwen3_fp4_relayout_research`  
**Date:** 2026-04-18  
**Status:** Research complete — verdict DEFER  
**Scope:** CPU-only feasibility study. No checkpoint re-layout attempted. No vLLM modified.

---

## Background

v7 Phase 0 KILL root cause: Qwen3-30B-A3B-NVFP4 stores FP4 weights as `[N_out, K_in/2]`
row-major. WMMA 16×16×16 tile needs 16 unique N-rows per load → 16 cache lines each 6%
utilized → ceiling ~112 GB/s. Measured 83.6 GB/s (4.7% of 1,792 GB/s peak). Layout is
the bottleneck.

Source docs: `plans/mega_graph_v7_phase0_results.md §3-4`,
`kernels/csrc/mega_graph_v7_phase0_microbench.cu:178-263`.

---

## Section 1: Current Layout vs Proposed Layouts

### 1.1 Current On-Disk Layout

| Tensor | Shape | Stride | Cache-line utilization with 16×16 WMMA tile |
|---|---|---|---|
| `gate/up.weight` | `[768, 1024]` uint8 | `[1024, 1]` row-major | 8B / 128B = **6.25%** |
| `gate/up.weight_scale` | `[768, 128]` fp8 | `[128, 1]` row-major | not the bottleneck |
| `down.weight` | `[2048, 384]` uint8 | `[384, 1]` row-major | 8B / 128B = **6.25%** |

**Why 6.25%:** A WMMA 16×16 B-tile requires 16 K-elements from each of 16 N-rows. With
`[N, K/2]` layout, the 16 N-rows are 1,024 bytes apart (gate/up stride) or 384 bytes apart
(down stride). Each 32-wide warp load touches 16 different cache lines and consumes only
8 bytes per line (16 FP4 nibbles = 8 uint8 bytes), yielding 8/128 = 6.25% utilization.

### 1.2 Candidate Re-Layouts

#### Layout A: `[K, N/2]` Transposed Row-Major

Swap axes: store as `[K_in, N_out/2]`. A WMMA B-tile dequant at `(k0, n0)` now reads from
rows `k0..k0+15` — **K is the major stride**, so all 16 K-values for a given N-column are
contiguous. Each warp reads 16 consecutive uint8 bytes from one cache line per K-step.

- **Cache-line utilization:** 128B loaded, 128B used (all values needed across the
  16-lane warp for 16 K-rows × 1 N-column). **~100%.**
- **Scale tensor:** must also transpose to `[K/16, N]` to maintain the scale-per-block-of-16-K-elements pairing. The CUTLASS 128×4 swizzle formula (indexed as `[row, block_idx]` where `block_idx` counts blocks along K) applies with `row` now indexing N, `block_idx` indexing K/16. Shape changes from `[N, K/16]` to `[K/16, N]` — the swizzle formula in `fused_shuffle_quant.cu:90-101` remains valid as long as `row_in_buf` and `block_idx` are correctly remapped.
- **Access pattern in `dequant_fp4_tile_qwen3`:** `lane_row_idx` (N-column within tile)
  now maps to `k` in the transposed layout, and `k_block` maps to the N-axis. All 32 lanes
  hit the **same** 16 rows (K-rows) → 1 cache-line request per K-step per warp.

**Projected HBM BW: 100% of ceiling = ~50-80% of 1,792 GB/s peak**, depending on
L1/L2 hit rate for scale tensor and compute overhead.

#### Layout B: CUTLASS Tile-Interleaved (`[N_tiles, K_tiles, 16, 16/2]`)

Store each 16×16 sub-tile contiguously: outer dims tile by (16 N, 16 K), inner dims are the
elements of that tile.

```
offset = (n_tile * K_tiles + k_tile) * (16 * 8) + (k_in_tile * 8 + n_in_tile//2)
```

- **Cache-line utilization:** one cache line per tile per warp = 100%, same as Layout A.
- **Additional benefit:** eliminates the `dequant_fp4_tile_qwen3` scatter-writes to
  `b_dq_smem`; each warp can load a contiguous 128-byte tile directly into registers.
- **Complexity:** Non-trivial offline conversion; the inner tile packing must match the
  WMMA B-fragment memory layout (row-major 16K × 16N for `wmma::matrix_b row_major`).
  Small error in inner indexing → silent numerical corruption.
- **Projected BW:** same theoretical ceiling as Layout A; implementation risk higher.

#### Layout C: Column-Major `[K/2, N]`

Standard "B matrix column-major" as in cuBLAS NT convention.

- Equivalent to Layout A conceptually (K is innermost, N is outer), just with different
  byte-level packing. For uint8 packed FP4 with 2 nibbles/byte, Layout A `[K, N/2]` and
  Layout C `[K/2, N]` require different interpretations of which nibble holds which element.
- **Cache-line utilization:** same as Layout A (~100% for a K-contiguous warp access).
- **vLLM compat:** CUTLASS `cutlass_fp4_moe_mm` expects weight in NT layout (weight is
  "B" matrix transposed) which corresponds to `[N, K/2]` on-disk — i.e., the CURRENT
  layout. Layout C would break `cutlass_fp4_moe_mm` without a separate on-load transpose.

**Best candidate: Layout A `[K_in, N_out/2]`** — highest cache-line utilization, clearest
correctness path, directly addresses the v7 Phase 0 root cause.

### 1.3 Cache-Line Utilization Math Summary

| Layout | Bytes/line loaded | Bytes/line used | Utilization |
|---|---|---|---|
| Current `[N, K/2]` | 128 | 8 | **6.25%** |
| Layout A `[K, N/2]` | 128 | 128 | **100%** |
| Layout B tile-interleaved | 128 | 128 | **100%** |

---

## Section 2: vLLM Compatibility Analysis

### 2.1 Code Paths That Touch Weight Layout

**Primary path (what vLLM does with the weights):**

`cutlass_fp4_moe_mm` in `vllm/model_executor/layers/fused_moe/cutlass_moe.py` (traced via
`plans/moe_shuffle_fusion_analysis.md §2` and `MOE_PROFILING.md §CUTLASS FP4 MoE Execution
Path`) calls into CUTLASS grouped GEMM. The CUTLASS kernel expects:

- Weight tensor B in **NT layout** (non-transposed logical, i.e., weights stored as
  `[N, K/2]` so that the GEMM computes `C = A × B^T` in standard cuBLAS NT convention).
- Scale tensor in **CUTLASS 128×4 swizzled layout** (`SF_LAYOUT_SWIZZLED`), indexed
  `[numMTiles, numKTiles, 32, 4, 4]` where M corresponds to N_out and K corresponds to K_in.

`ModelOptNvFp4LinearMethod` in `vllm/model_executor/layers/quantization/` (confirmed via
`patches/wire_fused_norm_fp4_qwen3_v2.py:107-154` which accesses `.backend`,
`.weight`, `.weight_scale`, `.alpha`, `.input_global_scale_inv`) dispatches to one of:

- `cutlass_scaled_fp4_mm` (standard CUTLASS path)
- `flashinfer_scaled_fp4_mm` (FlashInfer backend)
- `fbgemm.f4f4bf16_rowwise` (FBGEMM backend)

All three backends have the same hardcoded assumption: weight is `[N, K/2]` row-major.

### 2.2 Blocker 1: `cutlass_fp4_moe_mm` / `cutlass_scaled_fp4_mm` NT-layout contract

**Severity: CRITICAL.**

CUTLASS grouped GEMM for FP4 assumes `weight.shape == [N, K/2]` (NT convention: B is
non-transposed in memory, transposed in the math). If the offline-re-laid tensor has shape
`[K, N/2]`, passing it directly to `cutlass_fp4_moe_mm` without patching the call site will
silently compute the wrong GEMM (dimensions mismatch or wrong stride interpretation).

**Evidence:** `plans/mega_graph_v7_phase0_results.md §6.2` explicitly states "Breaks vLLM's
`cutlass_fp4_moe_mm` fallback (scale tensor would also need transpose/re-swizzle)." The
scale tensor shape `[N, K/16]` must also flip to `[K/16, N]` (or equivalently the swizzle
formula's `row` and `block_idx` axes swap roles).

**Fix required:** Patch `cutlass_moe.py::run_cutlass_moe_fp4()` to pass a `layout` flag
or `transpose_b=True` to the CUTLASS kernel call. Current CUTLASS FP4 grouped GEMM C++
interface (vLLM PR #38891 / SM120 path) does not expose a runtime layout toggle — it is
compiled with a fixed layout template parameter. Changing this requires a C++ recompile of
the CUTLASS MoE GEMM extension.

### 2.3 Blocker 2: `flashinfer_scaled_fp4_mm` and FBGEMM backends

**Severity: HIGH.**

FlashInfer's SM120 FP4 JIT kernel (`vllm/utils/flashinfer.py`, confirmed active at ~6,615
tok/s in `GEMMA4_NVFP4_BENCHMARKS.md`) dispatches to a separate CUTLASS or cuDNN kernel
with the same NT-layout assumption. FBGEMM `f4f4bf16_rowwise` likewise hardcodes `[N, K/2]`
weight layout. Re-laid `[K, N/2]` weights would produce incorrect output from both backends
without backend-specific patches.

**Fix required:** Patch each backend's dispatch or add a `weight = weight.T`
(with nibble-repack) before dispatch — but uint8-packed FP4 cannot be transposed with a
plain `.T`; requires a custom repack kernel.

### 2.4 Blocker 3: Scale tensor swizzle formula mismatch

**Severity: HIGH.**

The CUTLASS 128×4 swizzle (documented in `fused_shuffle_quant.cu:72-101` and
`mega_graph_v4b_checkpoint_format.md §6`) maps `(row_in_buf, block_idx)` where `row_in_buf`
is the M-axis (token / N-row) and `block_idx` is the K/64 tile index. In the current
`[N, K/2]` weight layout, N is the row axis and K is the block axis — matching the swizzle
convention directly.

After transposing to `[K, N/2]`, the roles swap: K becomes the row axis and N becomes the
block axis. The `fused_shuffle_quant_kernel` and `rms_norm_dynamic_fp4_quant_kernel` both
call `compute_cutlass_sf_offset` with the assumption that the M-dimension (128-row tile) is
the N-out dimension of the weight matrix. Transposing the weight but NOT re-implementing
the swizzle for the new axis assignment will produce a mis-indexed scale buffer and silent
numerical corruption.

**Fix required:** New swizzle formula with axes swapped, or a separate scale-reswizzle
pass applied offline alongside the weight transpose. Since scales are `fp8_e4m3fn` (1
byte/element) and have shape `[N, K/16]` → transposed `[K/16, N]`, the scale IO is ~16×
smaller than the weight IO — this is manageable, but must be verified end-to-end.

**Key finding on `fused_shuffle_quant.cu:72`:** This swizzle operates on the
ACTIVATION scale (token dimension M, hidden K), NOT on the weight scale. The weight scale
swizzle is done inside CUTLASS's `cutlass_fp4_moe_mm` at kernel-load time. Offline
re-layout of the weight scale would need to produce pre-swizzled data that matches whatever
the re-patched CUTLASS kernel expects — creating a tight coupling between the re-layout
script and the CUTLASS kernel version.

### 2.5 Secondary compat concern: vLLM fallback for non-fused paths

`patches/wire_fused_norm_fp4_qwen3_v2.py` and `patches/fused_norm_fp4_integration.py`
both capture `qkv_proj.weight` and `qkv_proj.weight_scale` at plugin-build time
(`process_weights_after_loading`). If weights are re-laid before loading, the captured
tensors would have the new shape — but the downstream CUTLASS call in `_do_mm` would still
pass them as if they were `[N, K/2]`. This creates a hard bug in the fused-norm path.
A BC flag (`RELAYOUT_V2=1`) would be needed to gate the re-laid weight path through
a different dispatch branch.

---

## Section 3: Re-Layout Script Skeleton

CPU-only pseudocode — DO NOT execute until vLLM compat patches are complete.

```python
# relayout_qwen3_nvfp4.py  — SKELETON ONLY, not for execution
# Transforms [N, K/2] weight + [N, K/16] scale -> [K, N/2] + [K/16, N]

import safetensors.torch as st
import torch
from pathlib import Path

SRC = Path("/root/models/Qwen3-30B-A3B-NVFP4")
DST = Path("/root/models/Qwen3-30B-A3B-NVFP4-RELAY")

def transpose_fp4_packed(w_NK2: torch.Tensor) -> torch.Tensor:
    """
    w_NK2: uint8 [N, K/2], packed as low-nibble=even-K, high-nibble=odd-K.
    Returns uint8 [K, N/2] with same nibble packing convention.

    NOTE: a plain w_NK2.T gives [K/2, N] but with wrong nibble semantics —
    column K=2i and K=2i+1 are packed in the same byte in [N,K/2], but after
    transpose they need to be in the same byte in [K, N/2].
    Requires unpacking all nibbles, transposing the [N, K] logical matrix,
    and repacking as [K, N/2].
    """
    N, K_half = w_NK2.shape
    K = K_half * 2
    # Unpack all nibbles to a logical [N, K] byte tensor (values 0..15)
    lo = w_NK2 & 0x0F                        # even K columns: [N, K/2]
    hi = (w_NK2 >> 4) & 0x0F                 # odd  K columns: [N, K/2]
    logical = torch.empty(N, K, dtype=torch.uint8)
    logical[:, 0::2] = lo
    logical[:, 1::2] = hi
    # Transpose to [K, N]
    logical_T = logical.T.contiguous()       # [K, N]
    # Repack: new even columns = 0,2,4,...  = logical_T[:, 0::2]
    #         new odd  columns = 1,3,5,...  = logical_T[:, 1::2]
    result = (logical_T[:, 0::2] & 0x0F) | ((logical_T[:, 1::2] & 0x0F) << 4)
    return result                             # [K, N/2]

def transpose_scale(s_NK16: torch.Tensor) -> torch.Tensor:
    """
    s_NK16: fp8 [N, K/16] row-major.
    Returns fp8 [K/16, N] row-major (plain .T + contiguous).
    No swizzle needed here — swizzle is applied at runtime by CUTLASS.
    """
    return s_NK16.T.contiguous()

def process_shard(src_path: Path, dst_path: Path):
    tensors = st.load_file(src_path)
    out = {}
    for name, tensor in tensors.items():
        if name.endswith(".weight") and tensor.dtype == torch.uint8:
            # FP4 weight: [N, K/2] -> [K, N/2]
            out[name] = transpose_fp4_packed(tensor)
        elif name.endswith(".weight_scale") and tensor.dtype == torch.float8_e4m3fn:
            # Scale: [N, K/16] -> [K/16, N]
            out[name] = transpose_scale(tensor)
        else:
            out[name] = tensor   # BF16 attention, scalars — unchanged
    st.save_file(out, dst_path)

# Main: iterate over all safetensors shards
for src_shard in sorted(SRC.glob("model*.safetensors")):
    dst_shard = DST / src_shard.name
    process_shard(src_shard, dst_shard)
# Also copy config.json, tokenizer files, etc.
```

**Critical correctness checks before any run:**

1. `transpose_fp4_packed`: verify on a 4×4 toy example that nibble indices match the
   original packing convention (low-nibble = even K-column).
2. Scale transpose: verify that the CUTLASS swizzle applied at runtime is
   `(row=N_col_index, block_idx=K_block_index)` for the new `[K/16, N]` layout.
3. End-to-end numerical: run the v7 Phase 0 microbench against re-laid weights with
   a reference dequant that cross-checks 100 random elements.

---

## Section 4: Effort Estimate

| Task | Lower bound | Realistic | Pessimistic |
|---|---|---|---|
| Research (this task) | 1 day | 2 days | 2 days |
| Re-layout script + unit tests | 0.5 day | 1.5 days | 3 days |
| CUTLASS FP4 MoE C++ patch (layout flag or transposed kernel template) | 2 days | 5 days | 10 days |
| FlashInfer backend patch | 1 day | 2 days | 4 days |
| Scale swizzle verification + re-swizzle for transposed layout | 0.5 day | 1.5 days | 3 days |
| BC flag wiring in vLLM model code + fused-norm plugin | 0.5 day | 1.5 days | 3 days |
| v7 kernel re-tune (WMMA tile config for new layout) | 1 day | 2 days | 3 days |
| Integration test (end-to-end correctness at M=1, M=32, M=512) | 0.5 day | 1 day | 2 days |
| **Total** | **7 days** | **15 days** | **28 days** |

**Lower bound assumes:** CUTLASS accepts a "transposed B" template specialization already
compiled in vLLM's SM120 build; FlashInfer has a layout flag; no BC-incompatible model
format issues discovered.

**Pessimistic assumes:** CUTLASS requires a new kernel template (recompile from source),
FlashInfer FP4 path does not support transposed B, and HuggingFace Hub rejects the
non-standard checkpoint causing additional serialization/format work.

---

## Section 5: Expected HBM BW by Layout Choice

All figures are for the Qwen3-30B-A3B shapes (HIDDEN=2048, INTER=768, TOP_K=8) on
RTX PRO 6000 Blackwell (SM120a, 1,792 GB/s HBM).

| Approach | Cache-line util | Theoretical ceiling | Projected actual | Confidence |
|---|---|---|---|---|
| Current `[N, K/2]` (v7 Phase 0) | 6.25% | ~112 GB/s | 83.6 GB/s (measured) | Confirmed |
| v7.1 K-gang (8× amortization) | ~50% | ~896 GB/s | ~28% (~500 GB/s) | Low (P~0.15) |
| Layout A `[K, N/2]` — pure bandwidth | ~100% | ~1,792 GB/s | 50-70% (~900-1,250 GB/s) | Low (P~0.1-0.2) |
| Layout A + scale fetch overhead | ~100% weight | ~1,200 GB/s effective | 40-60% (~720-1,080 GB/s) | Low |

**Why the actual BW gap to theoretical is large:** At M=1 decode, L2 occupancy for 8×21 MB
weight traffic (≫ 128 MB L2) means cold-line misses dominate. The layout change removes
the cache-line waste penalty but cannot exceed L1-fill-rate limits at 188 SMs × 16 warps ×
16-byte load per warp per cycle. Practical ceiling for sustained HBM on SM120a for
strided-access GEMV is ~50-60% of peak (1.7× the current 112 GB/s layout ceiling).

The 50% gate threshold is theoretically reachable (~50-60% from above) but not guaranteed:
the projection sits right at the gate threshold ± uncertainty, which per P11 §Cat-3/4
means this is a coin-flip outcome, not a confident PASS.

---

## Section 6: Verdict — P11 Category Assessment

**Category 3b (regime-mismatched cross-apply) + Category 4 (literature):**

The re-layout hypothesis draws on well-established memory access theory (cache-line
utilization) AND requires all three vLLM backends to accept the new layout. The theory is
sound; the implementation chain has three independent blockers, each carrying P_fail ~0.3.

- P(Blocker 1 resolved cleanly without CUTLASS recompile): ~0.35
- P(Blocker 2 resolved for FlashInfer): ~0.50
- P(Blocker 3 scale swizzle correct on first attempt): ~0.60
- P(all three clear AND projected BW ≥ 50%): ~0.35 × 0.50 × 0.60 × 0.60 ≈ **0.06**

This is below the P11 Cat-3/4 base rate of P~0.1-0.2 stated in the prompt context. Even
optimistically (assuming Blocker 1 is easier than estimated), ceiling is ~0.15.

**Per §P11 sub-case 3b:** The v7 Phase 0 result was measured at M=1 decode. Re-layout
benefit is also measured at M=1. No regime mismatch on M-axis — this is a favorable
condition. However, the vLLM compat chain introduces a new dimension of cross-apply risk
that P11 Cat-3b doesn't directly penalize.

### P11 Categories:

| Category | Applies? | Rationale |
|---|---|---|
| Cat 1 (Bug-fix PROCEED) | NO | Not fixing a code bug; changing weight layout |
| Cat 2 (Recalibration) | NO | No wrong constant used |
| Cat 3b (Regime-mismatched cross-apply) | PARTIAL | Layout theory applies; vLLM compat chain is the risk |
| Cat 4 (Literature) | YES | Cache-line utilization math is textbook correct but requires compat chain |

**Verdict: DEFER** — not KILL because the theory is sound; not PROCEED because the compat
chain P is below threshold without a verified CUTLASS transposed-B path confirmed in
vLLM's SM120 build.

---

## Section 7: Decision Tree — Re-Layout vs K-Gang vs Ceiling Accept

```
START: v7 Phase 0 KILL — 4.7% HBM BW
│
├─ Is v7.1 K-gang Phase 0 running? (v7.1 dispatch in queue)
│   ├─ YES → Wait for result first. If K-gang ≥ 30% MARGINAL:
│   │     └─ Accept MARGINAL and move to higher-leverage work (attention, M>1 batching)
│   └─ NO → Can dispatch v7.1 K-gang in 3 days (lower cost than re-layout)
│
├─ K-gang Phase 0 result = PASS (≥ 50%)?
│   └─ YES → Re-layout NOT needed. K-gang solved it. KILL re-layout work.
│
├─ K-gang Phase 0 result = MARGINAL (30-50%)?
│   └─ Consider re-layout ONLY IF: CUTLASS transposed-B template exists in vLLM
│       SM120 build (verify in one afternoon: grep CUTLASS headers for
│       `cutlass_fp4_grouped_gemm_transposed` or `KernelScheduleAuto` with
│       `layout_b = cutlass::layout::RowMajor`).
│       └─ If confirmed: Promote re-layout to PROCEED with 15-day budget.
│       └─ If not: DEFER until CUTLASS 3.x adds support or v7.1 MARGINAL accepted.
│
├─ K-gang Phase 0 result = KILL (< 30%)?
│   └─ Layout ceiling is harder than expected. Do NOT invest in re-layout:
│       the 50% gate is still speculative, and vLLM compat adds 15+ days.
│       Redirect to: (a) M>1 batching (M=32 shifts to compute-bound regime where
│       layout matters less), or (b) accept the FP4 decode ceiling and optimize
│       attention instead.
│
└─ Current-ceiling-accept pivot:
    At M=1 decode, the 83.6 GB/s vs 112 GB/s ceiling means we're at 75% of the
    LAYOUT-limited ceiling — not 4.7% of theoretical. The gap from 112 GB/s to
    the 50%-of-peak gate (896 GB/s) cannot be closed by any layout change alone;
    it requires either (a) better cache reuse (M>1), (b) native FP4 MMA (not
    available on SM120a), or (c) a fundamentally different decode strategy.

    CEILING ACCEPT is the correct choice if:
    - v7.1 K-gang Phase 0 lands MARGINAL, AND
    - Overall MoE decode step time is already within 1.5× of eager vLLM, AND
    - Attention optimization (v5a/v5b path) still has headroom.
```

---

## Summary (under 500 words)

**Best-candidate layout:** Layout A `[K_in, N_out/2]` is the only candidate with sound
cache-line math. It changes the access pattern from 16 scattered N-row cache lines per
warp to 16 contiguous K-row cache lines — upgrading utilization from 6.25% to ~100%.
**Projected BW: 40-60% of 1,792 GB/s peak** (720–1,080 GB/s), sitting right at or above
the 50% gate. Confidence is LOW because (a) scale overhead is unaccounted in the
projection, and (b) the vLLM compat chain must first be verified.

**Top 3 vLLM compat blockers:**

1. **CUTLASS `cutlass_fp4_moe_mm` NT-layout contract** (CRITICAL): The C++ grouped GEMM
   kernel is compiled with a fixed `[N, K/2]` weight layout template; accepting `[K, N/2]`
   requires a new kernel template or a `transpose_b=True` flag. No such flag exists in
   vLLM's current SM120 CUTLASS build. Fix: recompile CUTLASS extension — 2–10 days.
2. **Scale tensor swizzle mismatch** (HIGH): CUTLASS 128×4 swizzle indexes `(row=N_dim,
   block=K_dim)`; transposing the weight flips these axes. Offline-re-laid scale in
   `[K/16, N]` fed to the original swizzle formula → silent mis-indexed scale buffer →
   numerically wrong output. Fix: new swizzle formula or scale pre-swizzle pass — 1–3 days.
3. **Multi-backend dispatch breakage** (HIGH): FlashInfer SM120 FP4 JIT, FBGEMM
   `f4f4bf16_rowwise`, and the fused-norm plugin all capture `weight.shape == [N, K/2]`
   hardcoded. Each needs a separate patch or a BC flag gating the re-laid path — 1–4 days
   each.

**Effort:**
- Lower bound: 7 days (if CUTLASS has transposed-B template, FlashInfer is patchable)
- Realistic: 15 days
- Pessimistic: 28 days (CUTLASS recompile from source + format rejection)

**Verdict vs current-ceiling-accept pivot:** The M=1 decode regime is fundamentally at a
layout-imposed ceiling of ~112 GB/s regardless of kernel design. Re-layout moves the
ceiling to ~1,000–1,200 GB/s theoretically but the 50% gate is not confidently cleared
(P ~0.06–0.15). The 15-day realistic effort is a large investment for a coin-flip outcome.
Current-ceiling-accept is viable IF the MoE step is not the dominant bottleneck after
attention optimization.

**Recommendation: DEFER pending v7.1 K-gang Phase 0 result.** If K-gang achieves ≥ 30%
(MARGINAL), the ceiling-accept pivot becomes rational and re-layout should be shelved. If
K-gang KILLs below 30%, re-layout is theoretically attractive but the vLLM compat chain
risk still makes it a 15+ day gamble at P ~0.1. In either case, higher-leverage immediate
actions (attention optimization, M>1 batching to escape M=1 decode regime) should take
priority. Re-layout is a Category 3+4 hypothesis that requires a one-afternoon CUTLASS
header audit to determine whether the compat chain even has a fast path before any
substantial coding begins.

---

*Tag: `W7_qwen3_fp4_relayout_research`. CPU-only research. No checkpoint modified. No vLLM patched.*
