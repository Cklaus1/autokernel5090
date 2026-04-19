# Fused Attention + O-Projection Feasibility — §5c

**Tag:** W2_5c_fused_attn_oproj_feasibility
**Date:** 2026-04-18
**Kernel baseline:** `mega_graph_gemma4_30layer_v5a.cu` (8,809 µs/step, BF16, 5 barriers/layer)

---

## 1. Register Pressure

**SM120 cap:** 65,536 regs/SM ÷ 256 threads = **255 regs/thread hard limit.**
**v5a reported:** 154 regs/thread.

For the fused stage, the warp must hold concurrently:

| Fragment | WMMA type | fp32 regs/lane |
|---|---|---|
| V·P accumulator (attention output) | `accumulator<16,16,16,float>` | 8 |
| O-proj A-fragment (same data, reused as A) | `matrix_a<16,16,16,bf16,row_major>` | 4 (bf16 packed) |
| O-proj accumulator (output) | `accumulator<16,16,16,float>` | 8 |

However, the V·P accumulator cannot directly serve as the O-proj A-fragment. WMMA accumulator fragments are `float` typed; O-proj matrix_a expects BF16. A conversion step is mandatory (see §3 below). If conversion is done register-to-register (no smem store), the peak simultaneous live set is:

- 154 (v5a baseline)
- +8 (V·P accumulator held during conversion loop)
- +8 (O-proj accumulator live while consuming converted A values)
- The BF16 A-fragment replaces the V·P accumulator after conversion (~4 regs)

Conservative peak: **154 + 8 + 8 = 170 regs/thread.** Well under 255.

However, the conversion from `accumulator` to `matrix_a` cannot be performed directly via WMMA API — there is no `wmma::load_matrix_sync` from a register-resident accumulator fragment to a `matrix_a` fragment. The only path is `store_matrix_sync` to smem → `load_matrix_sync` back as `matrix_a`. This is precisely the smem roundtrip we are trying to eliminate.

**Register fit verdict: FIT (170 << 255), but see §3 for the layout blocker.**

---

## 2. Barrier Savings Analysis

v5a barrier schedule per layer (5 total):
- `[1]` post-QKV
- `[2]` post-attention core → **this is the barrier between attn and O-proj**
- `[3]` post-O-proj residual
- `[4]` post-MLP gate/up
- `[5]` post-MLP down

Barrier `[2]` exists because `attention_core_stage` writes `attn_out` to **global memory** (`__nv_bfloat16* attn_out`), and `attn_oproj_residual_stage` reads it back from global. Cross-SM visibility requires `grid.sync()`.

If fusion succeeds, barrier `[2]` is eliminated — but only for SMs that do both stages. In v5a, attention runs only on `sm < 16` (one SM per head); all 188 SMs run O-proj. Barrier `[2]` is still needed to synchronize the 16 attention SMs writing `attn_out` with the 172 non-attention SMs that need to read it before O-proj. The barrier is not between the same SMs doing sequential work — it is a **cross-SM dependency** and cannot be fused away unless the attention output is restructured so every SM already has its own slice of `attn_out` before O-proj begins.

v5a.1's distributed approach (all 188 SMs on attention) was tried and KILL'd due to +1 barrier overhead outweighing the SM utilization gain. The same structural asymmetry applies here.

**Barrier elimination possible only if:** attention and O-proj are co-partitioned such that each SM owns exactly the `attn_out` slice it will feed into O-proj — requiring a head-stripe → hidden-stripe reshape that is non-trivial.

**Projected savings (best case, assuming v5b validates barrier bottleneck):**
- Barrier `[2]` cost: ~15 µs × 30 layers = **450 µs** (v5a.1 empirical: ~15 µs/barrier at this scale)
- Best case (all 30 barriers eliminated): −450 µs → step time ~8,359 µs
- Worst case (structural dependency prevents elimination): **0 µs savings**

---

## 3. WMMA Layout Compatibility

**Attention output:** `V·P` WMMA produces `accumulator<M=16, N=16, K=16, float>` fragments. Each warp owns one 16×16 tile of `o16_smem [16, HEAD_DIM=128]` — 8 warps covering 8 N-tiles of 16 columns each. Row 0 of each tile is the real per-head attention output. The full `attn_out` is `[HIDDEN=2048]` flattened.

**O-proj A-fragment:** requires `matrix_a<M=16, N=16, K=16, bf16, row_major>`. The A-matrix is `attn_out [1, HIDDEN=2048]` staged as a `[16, HIDDEN]` padded tile (row 0 real, rows 1-15 zero), iterated in 128 K-steps of 16.

**Incompatibility:** The V·P accumulator fragment for one N-tile covers 16 columns of `[16, HEAD_DIM]`. The O-proj A-fragment for the same K-step covers 16 elements of the same flattened row — but the fragment register layout for `accumulator<float>` vs `matrix_a<bf16>` is different (float accumulator: 8 fp32 regs in a specific WMMA-internal scattered pattern; bf16 matrix_a: 4 regs with different lane-to-element mapping). No WMMA intrinsic performs this conversion in-register.

**The only standard path:** `wmma::store_matrix_sync(smem, acc_frag)` → fp32-to-bf16 convert in smem → `wmma::load_matrix_sync(a_frag, smem)`. This is the existing smem roundtrip.

**WMMA layout verdict: NEEDS FRAGMENT RESHAPE — no in-register path exists via standard WMMA API. PTX-level manipulation of fragment registers is theoretically possible but non-portable and extremely fragile on SM120a.**

---

## 4. Dependency on v5b

v5b's goal is to confirm that barriers are the dominant bottleneck (5→3/layer). The fused attn+O-proj concept is predicated on barrier `[2]` being measurably expensive.

- If v5b PASSES (barriers confirmed as bottleneck): the fusion is still blocked by the layout incompatibility (§3) and the cross-SM dependency structure (§2). Even if v5b gets to 3 barriers, the attn→O-proj barrier requires a deeper architectural change.
- If v5b KILLS (barriers are not the bottleneck): the entire motivation for this fusion evaporates.

**Dependency verdict: REQUIRED — but insufficient. v5b must PASS AND the architectural cross-SM issue must be separately resolved before §5c can deliver savings.**

---

## 5. Summary

| Dimension | Finding |
|---|---|
| Register budget | FIT — 170/255 regs at peak |
| WMMA layout | NEEDS FRAGMENT RESHAPE — no in-register path; smem roundtrip unavoidable via standard API |
| Barrier savings | 0–450 µs/step — zero unless co-partition restructuring done; cross-SM dependency is structural |
| v5b dependency | REQUIRED (and insufficient alone) |

---

## 6. Verdict: DEFER

The register budget is comfortable and is not the gating issue. The blocker is a two-part structural problem:

1. **No in-register fp32-accumulator → bf16-matrix_a conversion path in WMMA.** Eliminating the smem roundtrip requires either PTX-level register manipulation (fragile, SM120a-specific, non-portable) or switching to a different accumulation strategy.

2. **The barrier between attention and O-proj is a cross-SM dependency** (16 attention SMs writing, 188 O-proj SMs reading), not a same-SM sequential dependency. Register-resident passing cannot eliminate a cross-SM barrier; it would require a co-partition design where every SM owns both its attention head slice and the corresponding O-proj output columns.

**DEFER** pending: (a) v5b confirming barrier budget, and (b) a co-partition design that aligns attention head stripes with O-proj output stripes on the same SM. That is a v5c architectural redesign, not an incremental fusion. If v5b kills, this task is moot.
