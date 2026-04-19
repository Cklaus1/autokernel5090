# Mega-Graph v4b: FP4 Inline-Weight WMMA Spec

**Date:** 2026-04-18  
**Context:** v4a (cp.async B-load prefetch) running on parallel peer agent.
If v4a PASSES the B-load bottleneck hypothesis, v4b adds FP4 inline weights
as the next performance layer. This document covers the design choice between
Options A/B/C and the smem budget.

**Predecessor state (v3 = mega_graph_gemma4_30layer_v3.cu):**
- BF16 weights, WMMA `m16n16k16` tensor cores, 188 SMs
- H=2048, INTER_DIM=8192, 30 layers, M=1 decode, seq=256
- Measured: 9.84–9.87 ms / 30 layers (v3, after v2's 5-barrier-per-layer)
- Smem used: 20,608 B peak per SM (union-overlaid: 4 KB A-frags + 16 KB C-scratch + 128 B reduce)
- HBM BW: 9.6% of 1792 GB/s at H=2048 — compute-bound, not bandwidth-bound

**v4a focus:** Replace BF16 B-weight loads with cp.async double-buffered
prefetch of BF16 tiles. Hypothesis: latency hiding shortens the B-load stall.

**v4b focus (this spec):** Load FP4 (0.5 B/elem) instead of BF16 (2 B/elem)
weights from HBM. This is a 4× weight-data reduction — at decode M=1 where
weights dominate memory traffic, it directly targets the remaining latency floor.

---

## 1. Background: Why FP4 Weights Now?

At the real Gemma4 scale (H=4096, INTER_DIM=∼7168 per expert, K=4096):

| Weight source | Size per layer (30-layer model, 128 experts) |
|---|---|
| BF16 | ~256 MB / layer |
| FP4 (1/4 size) | ~64 MB / layer |

At M=1 decode, the kernel is memory-bandwidth bound on weight reads (even
though M=1 creates small GEMV problems). Switching to FP4 halves the bytes
read per GEMV (FP4 = 0.5 B/elem vs BF16 = 2 B/elem) → the **bottleneck
shifts from weight-load latency to dequant compute**, which is expected to
be much cheaper.

The v3 baseline at 9.87 ms is 9.6% BW utilization = ~170 GB/s (of 1792 GB/s
peak). At H=2048 the problem is too small. At real H=4096 with 128 MoE
experts, HBM traffic will be ~4× larger → the BW gap grows and FP4
compression becomes critical.

---

## 2. Three Design Options

### Option A — Dequantize-then-WMMA (recommended v4b path)

**Concept:** Load FP4 tiles and FP8 scales via cp.async into smem. Dequantize
to BF16 in smem using a producer warp (or inline in the load warp). Feed BF16
tiles to standard `wmma::load_matrix_sync` with BF16 A/B fragments.

**Per tile (16×K WMMA fragment, K=16 step):**
1. Load FP4 B-tile: `16 × 16 × 0.5 B = 128 B` from HBM
2. Load FP8 scale: `16 × 1 × 1 B = 16 B` (one scale per 16-element block)
3. Dequant in smem: unpack nibbles → FP4 E2M1 lookup → multiply by FP8 scale × fp32_global
4. Write BF16 B-tile to smem: `16 × 16 × 2 B = 512 B`
5. `wmma::load_matrix_sync` from BF16 smem → standard WMMA path

**Smem cost per tile:**
- FP4 staging area: `16 × K/2 B = 128 B` (for one 16-row tile)
- BF16 result area: `16 × K × 2 B = 512 B` (or reuse A-frag smem)
- FP8 scale staging: `16 × 1 B = 16 B`
- Total staging overhead: ~144 B extra per WMMA tile

**Smem budget impact:**

| Region | v3 (BF16) | v4b (FP4, Option A) |
|---|---|---|
| A-frags (8 warps × 16×16 BF16) | 4,096 B | 4,096 B (unchanged) |
| C-scratch 0 (8 warps × 16×16 FP32) | 8,192 B | 8,192 B (unchanged) |
| C-scratch 1 (MLP gate+up) | 8,192 B | 8,192 B (unchanged) |
| Block-reduce | 128 B | 128 B |
| Attention (scores + Q) | 1,280 B | 1,280 B |
| FP4 staging (new) | 0 B | 128 B per active tile |
| FP8 scale staging (new) | 0 B | 16 B per active tile |
| BF16 dequant buffer | 0 B | 512 B (union with C-scratch, usable disjointly) |
| **Peak total** | **20,608 B** | **~21,072 B (~21 KB)** |

With cp.async double-buffering (v4a) two FP4 tile pairs in flight:
`21,072 + 2 × (128 + 16) = ~21,360 B ≈ 21 KB`

This is **well under the 228 KB SM cap** and even under the 48 KB cooperative
default (we use dynamic-smem with `cudaFuncSetAttribute` to raise cap to
∼100 KB as needed).

**Smem budget sanity check:**
- v3 used: **20,608 B** (~20 KB) of 228 KB
- v4a (cp.async ping-pong adds double-buffer tiles, est. +12 KB): **~33 KB**
- v4b (adds FP4 staging, est. +2 KB): **~35 KB** — well under 50 KB practical
  limit. Budget PASSES.

**Advantages:**
- WMMA layout and fragment types are unchanged — lowest implementation risk.
- FP8 scale decode is a single `__nv_fp8_e4m3::__x_to_float()` call per block.
- Hardware PTX `cvt.rn.satfinite.e2m1x2.f32` on SM120a handles 2 FP4 values
  in one instruction (or use the 16-entry lookup table — same result).
- Dequant ops are register-cheap: 1 nibble unpack + 1 lookup + 1 FP32 multiply
  per element = ~3 instructions for 2 elements packed in one byte.

**Disadvantages:**
- Dequant adds a small prologue to the inner loop (estimate: 4–8 cycles per
  16-element block vs ~0 for native FP4 MMA if it existed).
- FP4 → BF16 conversion widens smem footprint by 4× vs storing the FP4 form.
- The effective "FP4 gain" is: 4× less HBM read for B, partially offset by
  the dequant compute. At decode M=1 (memory-bound), net win should be ~2–3×.

**Verdict: Option A is the correct v4b path.** Confirmed by the existing
codebase pattern: `convert_ct_to_modelopt.py` and `rms_norm_dynamic_fp4_quant.cu`
both demonstrate this exact pattern working on SM120a.

---

### Option B — FP8 Intermediate WMMA

**Concept:** Dequant FP4 → FP8 E4M3 (not BF16). Use FP8 tensor-core WMMA
path (`mma.sync.m16n8k32` with E4M3 inputs) if it exists on SM120a.

**SM120a FP8 MMA investigation:**

From `plans/fa3_sm120a_port.md`:
> `cutlass/arch/config.h:48` gates WGMMA on exactly `__CUDA_ARCH__ == 900`.
> SM120 has TMA + STSM + mbarrier but NOT WGMMA.

The `nvcuda::wmma` API on SM120a supports:
- `wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>` — confirmed working
- `mma.sync.aligned.m16n8k32` with FP8 inputs (`e4m3`) — defined in PTX ISA for
  SM89 (Ada Lovelace) and SM90 (Hopper), but **not officially documented for SM120a consumer Blackwell**.

CUTLASS source investigation:
- `cutlass/arch/mma_sm89.h` defines `SM89_16x8x32_F32E4M3E4M3F32_TN` for Ada.
- SM120 (`cute/arch/mma_sm120.h` or similar) — **no equivalent found in CUDA 12.8 headers**.
- The `cute::SM120_16x8x32_F32E4M3E4M3F32_TN` selector does NOT exist in CUTLASS 3.x headers shipped with CUDA 12.8.

The FA3 port experience (Idea #6 in the session log) is confirmatory:
> All FP8 MMA paths rely on WGMMA or GMMA which are SM90-only. Synchronous
> `mma.sync` FP8 paths (m16n8k32) nominally exist for SM89 but the Blackwell
> cooperative group + warp spec paths assume SM90.

**Verdict for Option B:** The SM120a FP8 tensor-core WMMA path via
`mma.sync.m16n8k32 E4M3` is **not confirmed available** in CUDA 12.8 for SM120.
PTX ISA 8.7 lists `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
as available on `sm_89a` and `sm_90a`, but SM120a is not in that list.
Do NOT pursue Option B for v4b without first verifying SM120a FP8 MMA support
in a standalone PTX test (see blocker §5).

---

### Option C — FP8 Inputs Directly to WMMA (skip dequant)

**Concept:** Load FP4, dequant to FP8 (not BF16), and use FP8 WMMA directly,
skipping BF16 dequant entirely. This requires SM120a FP8 MMA support AND a
clean FP4→FP8 conversion path (which exists: the quantization pipeline uses
FP8 scales natively, and `rms_norm_dynamic_fp4_quant.cu` converts FP4 values
back through their FP8 scale as an intermediate).

**Blockage:** Option C has the same dependency as Option B — SM120a FP8 MMA
must be confirmed. Additionally, `cute::SM120_16x8x32_F32E4M3E4M3F32_TN` or
equivalent does not appear in the CUDA 12.8 headers reviewed.

**Verdict for Option C:** Not viable for v4b without resolving the SM120a FP8
MMA availability question. Defer to v4c if Option B investigation succeeds.

---

## 3. Selected Path: Option A

**Decision:** Option A (dequantize FP4 → BF16 in smem, then standard WMMA).

**Rationale:**
1. Lowest risk: WMMA layout, fragment types, and the BF16 → FP32-accumulator
   path are proven on SM120a across v2, v3, and the tc prototype.
2. FP4 dequant is cheap: 3–4 instructions per byte (unpack nibbles + lookup +
   scale multiply). At 16 elements per scale block, the dequant kernel is
   simple and can be inlined in the load warp.
3. The checkpoint format audit (`mega_graph_v4b_checkpoint_format.md`) confirms
   the on-disk format is exactly what Option A needs: `uint8` packed FP4 +
   `fp8_e4m3fn` scales (row-major, not swizzled) + `fp32` global scale.
4. 4× weight data reduction directly targets the M=1 decode memory bottleneck
   at real Gemma4 H=4096 scale.
5. Options B/C are blocked on SM120a FP8 MMA confirmation (see §5).

---

## 4. v4b Implementation Plan

### Phase 1: Weight Loading
Replace the BF16 B-fragment load in the mega-graph inner loop with:
1. Load FP4 bytes: `cp.async.ca.shared.global [dst], [src], 16` (16 B chunk = 32 FP4 values)
2. Load FP8 scales: `cp.async.ca.shared.global [dst_s], [src_s], 2` (2 B = 2 FP8 scales)
3. `cp.async.wait_group 1` (leave one group outstanding for overlap with compute)

### Phase 2: Dequant Kernel (inline in producer warp)
Per 32-element block (2 bytes FP4 + 2 bytes FP8 scale):
```c
// Using hardware PTX (SM120a supports cvt.rn.satfinite.e2m1x2.f32):
uint8_t  packed = fp4_smem[i];          // 2 FP4 nibbles
float    scale  = fp8_to_float(sf_smem[i >> 4]);  // one scale per 16 elements
float    gs     = weight_scale_2;       // fp32 global scale

// Hardware decode: 2 FP4 nibbles -> 2 BF16 values
uint8_t  bf16_pair;
asm("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;"
    : "=h"(bf16_pair) : "f"(lo_f32), "f"(hi_f32));  // hardware only on SM120a+

// Or software path (lookup table, CPU-reference-bit-comparable):
float lo = fp4_table[packed & 0xF] * scale * gs;
float hi = fp4_table[(packed >> 4) & 0xF] * scale * gs;
bf16_smem[2*i]   = __float2bfloat16(lo);
bf16_smem[2*i+1] = __float2bfloat16(hi);
```

### Phase 3: WMMA Path
Unchanged from v3: `wmma::load_matrix_sync` from BF16 smem, same fragment
layout, same FP32 accumulator, same grid.sync barriers.

### Phase 4: Scale Access Pattern
For the mega-graph cooperative kernel, each SM owns rows
`[sm_idx * rows_per_sm, (sm_idx+1) * rows_per_sm)`. Scale tensor is accessed
as `weight_scale[row, block_idx]` — row-major, sequential within each SM's
stripe. **No swizzle needed** for the mega-graph path (swizzle is only
required for the CUTLASS `cutlass_fp4_moe_mm` path, not cooperative WMMA).

---

## 5. Blockers for v4b Integration

### Blocker 1 (high priority): SM120a FP8 MMA confirmation
To close the door on Options B/C (and to know if hardware `cvt.e2m1x2`
requires `__CUDA_ARCH__ >= 1000` specifically vs just `>= 890`):
- Write a minimal 20-line PTX test: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
- Compile with `-arch=sm_120a -ptx` and check for compile error vs runtime result
- If it works: Option B becomes viable for v4c
- If it fails: officially close Options B/C, proceed with Option A only

### Blocker 2 (medium priority): cp.async + dequant overlap
The v4a prefetch infra (cp.async ping-pong) must be verified to work with
FP4 byte loads before wiring in Option A dequant. The FP4 tile sizes are:
- FP4 tile: 16 × K × 0.5 B = 8 B per element row per K step
- BF16 tile: 16 × K × 2 B = 32 B per element row per K step (4× larger)
The smaller FP4 tile may require adjusting the cp.async group size and the
double-buffer staging dimensions.

### Blocker 3 (low priority): Real Gemma4 weight shapes
The current v3 prototype uses H=2048, INTER_DIM=8192. Real Gemma4 26B uses:
- H = 4096 (main hidden dim)
- MoE expert intermediate dim varies by layer (exact value needed from config)
- 128 experts, top-8 routing, 30 layers

At H=4096 the WMMA tile count per GEMV = 4096/16 = 256 tiles (vs 128 in v3).
FP4 compression may shift the bottleneck from BW to compute (dequant + MMA),
which would require increasing WARPS_PER_BLOCK or widening N_PER_WARP to
hide the dequant latency.

---

## 6. Smem Budget — Full Accounting

### SM120a memory resources (RTX PRO 6000 Blackwell)
- Shared memory per SM: **228 KB** (configurable)
- Default cooperative launch limit: ~100 KB with dynamic-smem attribute set
- L2 cache: 128 MB (shared across all SMs)
- Registers per SM: 64 K (256 per thread at 256 threads/SM)

### Per-SM smem allocation table

| Region | Size | Notes |
|---|---|---|
| A-frags (WMMA input activation) | 4,096 B | 8 warps × 16×16 BF16 |
| C-scratch 0 (output accumulator) | 8,192 B | 8 warps × 16×16 FP32 |
| C-scratch 1 (MLP gate+up second output) | 8,192 B | union-overlaid with C-scratch 0 for other stages |
| Block-reduce (RMSNorm) | 128 B | 32 FP32 values |
| Attention scratch (softmax scores + q_h) | 1,280 B | MAX_SEQ FP32 + HEAD_DIM BF16 |
| **v3 peak total** | **20,608 B** | **(~20 KB)** |

After v4a cp.async ping-pong (estimated +12 KB for double-buffer BF16 tiles):
| BF16 double-buffer A | ~4,096 B | ping-pong BF16 A tiles |
| BF16 double-buffer B | ~8,192 B | ping-pong BF16 B tiles |
| **v4a est. total** | **~33 KB** | |

After v4b FP4 additions (Option A, dequant in place):
| FP4 B-tile staging | 128 B | 16-row FP4 tile = 16×8 uint8 |
| FP8 scale staging | 16 B | 16 rows × 1 scale each |
| BF16 dequant output | 512 B | union with existing smem (disjoint use) |
| **v4b est. total** | **~35 KB** | **well under 50 KB practical limit** |

**Conclusion:** Option A fits in ~35 KB of smem per SM. The 228 KB SM cap
allows >6 concurrent SM-blocks if needed (we use 1 block/SM per cooperative
design, so occupancy is 1 — cap is irrelevant). There is ample room for
further staging expansions.

---

## 7. Comparison to v3 and Decision Summary

| Axis | v3 (current) | v4a (peer agent) | v4b (this spec) |
|---|---|---|---|
| Weight dtype | BF16 | BF16 | **FP4** |
| HBM weight traffic | 1× | 1× | **0.25×** |
| Dequant cost | 0 | 0 | ~4 cycles/byte |
| WMMA path | BF16 m16n16k16 | BF16 + cp.async | BF16 m16n16k16 + FP4 inline dequant |
| Smem (peak) | ~20 KB | ~33 KB | **~35 KB** |
| Expected speedup (M=1, H=4096) | baseline | +10-20% (prefetch) | **+2-3×** (BW reduction) |
| Option selected | — | — | **A** |
| Risk | proven | medium | **low** (Option A is a known-good pattern) |

---

## 8. Files to Create for v4b (kernel work — not this task)

```
kernels/csrc/mega_graph_gemma4_30layer_v4b.cu   # FP4 inline dequant
kernels/csrc/build_mega_graph_gemma4_30layer_v4b.py
kernels/csrc/test_mega_graph_gemma4_30layer_v4b.py
```

The `test_fp4_dequant.py` and `test_swizzle_unpacker.py` in this task
provide the CPU reference implementations for bit-compare validation.

---

## 9. v4b Timeline Estimate

With this prep:
- Checkpoint format: fully documented (no surprises)
- FP4 dequant reference: implemented in `test_fp4_dequant.py`
- Swizzle unpacker: implemented in `test_swizzle_unpacker.py`
- Option A design: fully specified

**Estimated implementation time:** 5–7 days (shortened from 10–14 days).
Key remaining work: (1) FP4 B-tile load integration with cp.async, (2) inline
dequant warp logic, (3) correctness harness at H=4096. The barrier
infrastructure, cooperative launch, and WMMA framework are proven and carry
over unchanged from v3.
