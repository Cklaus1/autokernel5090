# WMMA as RMSNorm Reduction Primitive — Feasibility Analysis

**Tag:** W2_5h_wmma_reduction_feasibility  
**Date:** 2026-04-18  
**Status:** KILL

---

## 1. What Was Investigated

Can `mma.sync.m16n8k16` (WMMA) replace the current shuffle + smem two-stage
block reduction for RMSNorm's Σ(x²) sum in the Mega-graph v5a kernel?

---

## 2. Current Implementation (v5a Baseline)

File: `kernels/csrc/mega_graph_gemma4_30layer_v5a.cu`, lines 153–229.

`compute_inv_rms` with `BLOCK_SIZE=256` (8 warps):

1. **Scatter accumulate** — each thread strides over H=2048 elements, squaring and
   accumulating into one fp32 register. Cost per warp: 2048/32 = 64 loads + 64
   multiply-adds = ~64 cycles (memory-bound path hidden by pipelining).
2. **Intra-warp shuffle** — 5 rounds of `__shfl_xor_sync`, each 1 cycle on
   SM120. Total: 5 cycles.
3. **Smem staging + barrier** — warp-0 writes 8 partial sums to `red_smem`,
   `__syncthreads`, then one more 5-cycle shuffle pass over 8 values.
4. **Broadcast** — `smem[0]` read back by all threads.

Total *reduction-only* overhead: ~5 (intra-warp) + 1 (`__syncthreads`) + 5
(inter-warp) = **~11–16 cycles** of reduction logic.  
Full function including the strided-load loop: ~80 cycles (dominating cost is
the 2048-element squared-load loop, not the tree reduction).

---

## 3. WMMA Path Cost

Hypothesis: treat the BF16 activation vector as matrix A (1×2048), multiply
by a ones matrix B (2048×1) → scalar sum in the fp32 accumulator.

Using `m16n8k16`: each MMA tile consumes 16 K-elements per warp per issue.  
For H=2048: **2048 / 16 = 128 K-steps**.

Published latency of one `mma.sync.m16n8k16` on SM89/SM120: ~16 cycles
(throughput-bound; pipeline depth ~4 cycles but dependent chain forces serial
execution for a pure dot-product reduction).

128 K-steps × 16 cycles = **2,048 cycles per warp**.

Compare to current: ~80 cycles end-to-end (load + reduce).

**WMMA is ~25× slower** for this scalar reduction.

---

## 4. Why WMMA Cannot Help Here

| Factor | Shuffle+smem | WMMA |
|---|---|---|
| Hardware primitive for scalar reduce | Yes (`__shfl_xor`) | No — WMMA produces a 16×8 tile, not a scalar |
| Input register state | BF16 in smem; already loaded for the squaring loop | Would require reformatting into WMMA fragment layout (additional cost) |
| K-step serial chain | N/A | 128 dependent MMA issues — no ILP for scalar dot product |
| Reduction tree depth | log₂(32) = 5 shuffles | 128 serial MMA issues |
| `Σ(x²)` vs `Σ(x)` | Trivial in scalar loop | WMMA gives `Σ(x)`; squaring requires a separate element-wise pass |

The last point is decisive: RMSNorm needs `Σ(x²)`, not `Σ(x)`. Using WMMA
for the sum would require first squaring into a scratch buffer (extra
bandwidth) then multiplying by the ones vector — adding a full extra memory
round-trip on top of the 25× compute penalty.

---

## 5. Is There Any Case Where WMMA Pays for Reduction?

Only if both conditions hold simultaneously:

1. The data is **already materialized in WMMA fragment registers** (avoiding
   the reformatting overhead), AND
2. The reduction dimensionality is large enough that the 16-wide SIMD width
   amortises the serial K-step chain.

For RMSNorm inputs: activations arrive from global/smem as plain BF16 rows —
condition 1 is never met.

The one scenario where WMMA reduction could break even (not win) is if the
output of a prior WMMA GEMM feeds directly into a norm and can stay in
registers. In Mega-graph v5a the QKV + O-proj outputs are written back to
smem/global before RMSNorm — condition 1 is not met there either.

Softmax LSE reduction (online max+sum over seq_len=256) is entirely scalar and
trivially handled by `block_reduce_max` + `block_reduce_sum`; same conclusion
applies with even greater force.

---

## 6. System-Level Impact Check

- 60 RMSNorm ops/step × 80 cycles = 4,800 cycles ≈ **2.4 µs** at 2 GHz.
- Barrier cost: ~2,250 µs/step.
- Norm reduction is 0.11% of total step time.

Even a hypothetical 2× improvement in reduction would save <1.2 µs. Not worth
any complexity.

---

## 7. Verdict

**KILL.** WMMA is a matrix-multiply engine, not a reduction engine.
For Σ(x²) over H=2048:

- WMMA: ~2,048 cycles (serial K-step chain, plus reformatting overhead,
  plus inability to compute x² inline).
- Current shuffle+smem: ~80 cycles end-to-end, hardware-native, zero
  extra passes.

The idea does not return: unless inputs are already in WMMA fragment registers
(never the case for RMSNorm inputs), WMMA reduction is always dominated by
`__shfl_xor_sync`. No action required in the kernel.
