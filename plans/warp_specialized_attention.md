# Warp-Specialized Attention Decode on SM120a (Consumer Blackwell)

**Status:** design + prototype + microbench. Verdict at the bottom.

**Hardware:** RTX PRO 6000 Blackwell Max-Q (SM120a). 170 SMs, HBM3e
~1.79 TB/s peak, 100 KB SMEM/SM (99 KB opt-in per block). This is
**consumer Blackwell** — NOT Hopper (228 KB) or datacenter Blackwell
(B100/B200 SM100).

**Baseline:** FA2 split-KV at head_dim=256 achieves ~93% of HBM BW on
Gemma-4-26B decode (measured 323 us vs theoretical 300 us for
B=32, seq=2048, KV=512 MB / layer — see `plans/sm120_attention_kernel.md`).

**Premise being tested:** Does explicit warp-specialization
(producer-consumer decoupling via mbarriers) give a meaningful gap
over FA2's implicit cp.async pipelining on SM120 for small-M decode?

---

## 1. Warp Breakdown

One CTA per (batch, q_head, kv_split). 4 warps / 128 threads total.

| Warp | Role | Duty |
|------|------|------|
| 0    | Producer A | cp.async K tile stage i; mbarrier.arrive(full[i]) |
| 1    | Producer B | cp.async V tile stage i; mbarrier.arrive(full[i]) |
| 2    | Consumer   | mbarrier.wait(full[i]); Q·Kᵀ dot; online softmax update |
| 3    | Consumer   | mbarrier.wait(full[i]); P·V accumulate; mbarrier.arrive(empty[i]) |

SM120a provides async cp.async (SM80-class) and mbarrier/bar.sync
primitives. It does NOT provide Hopper's TMA (`cp.async.bulk.tensor`)
nor WGMMA — those are SM90+ datacenter only. This pipeline therefore
uses **cp.async with 16B vector lanes**, not TMA bulk copies.

## 2. Shared Memory Queue

Circular buffer, 3 stages (STAGES=3 keeps us under the 99 KB opt-in):

```
stage_size = 2 * (BLOCK_N * HEAD_DIM * sizeof(bf16))
           = 2 * 16 * 256 * 2 = 16 KB     (K tile + V tile)
queue_size = 3 * 16 KB = 48 KB
+ Q buffer in regs (512 B) + softmax scratch (~2 KB) + mbarriers (64 B)
≈ 50 KB / CTA                    fits in 99 KB opt-in
```

This is **half** of FA2's 96 KB and leaves room for 2 CTAs/SM
(vs FA2's 1 CTA/SM) — the one architectural win we can target on
SM120.

## 3. Barrier Primitives

Per stage: one "full" mbarrier (producers → consumers) and one
"empty" mbarrier (consumers → producers).

```cuda
__shared__ alignas(8) uint64_t full[STAGES];
__shared__ alignas(8) uint64_t empty[STAGES];
if (threadIdx.x == 0) {
    for (int s=0; s<STAGES; ++s) {
        mbarrier_init(&full[s],  PRODUCER_WARPS * 32);   // K + V arrive
        mbarrier_init(&empty[s], CONSUMER_WARPS * 32);
    }
}
__syncthreads();
```

SM120a supports `mbarrier.init.shared::cta.b64`,
`mbarrier.arrive.shared::cta.b64`, `mbarrier.try_wait.parity.shared`.
Same PTX level as SM90, just without the `cp.async.bulk.*` cluster
variants.

## 4. Why This *Might* Beat FA2 (And Why It Probably Won't)

**FA2's implicit pipeline.** FA2 issues `cp.async_commit_group()` +
`cp.async_wait_group<N>()` between tiles. With split-KV and 6
blocks-per-SM, the SM scheduler rotates and hides most load latency.
Measured: 93% HBM BW.

**The 7% residual.** Page-table lookups, softmax reductions, split-KV
combine. None of those are load-latency hides that warp-spec can
attack.

**The hope.** Warp-spec wins on Hopper by overlapping TMA + WGMMA
across different functional units. On SM120a we have **neither**
TMA nor WGMMA — we have the same `cp.async` and same SM80 MMA as
FA2. The *only* delta is that instead of one warp-group doing
load-then-compute serially, we dedicate 2 warps to the LSU and 2 to
the TC/SFU. That's structural, not functional.

**Expected upside:** measurable only if the 7% residual on FA2 comes
from stalled compute waiting on cp.async. At decode (M=1) almost all
compute is a degenerate dot product — tiny, already overlappable by
cp.async_wait_group. So the residual is mostly reduction / launch /
page-table overhead, **none of which warp-spec addresses**.

**Prediction: 92-94% HBM BW at best (noise around FA2), with real
risk of regression from mbarrier polling + lower SMEM-per-block
concurrency not translating (L2 pressure at 2 CTAs/SM vs 1).**

## 5. Correctness Strategy

Swap only the load pipeline. Consumer warps reuse the same online
softmax recurrence as FlashInfer decode:

```
m_new = max(m, max_i(s_i))
alpha = exp(m - m_new)
l_new = alpha * l + sum_i(exp(s_i - m_new))
o_new = alpha * o + sum_i(exp(s_i - m_new) * V_i)
```

Final normalize: `out = o_new / l_new`. Single-split-K first, then
add split-K reduction as a second kernel (same pattern as FA2).

Numerical check: allclose vs `flashinfer.single_decode_with_kv_cache`
at atol=1e-2 / rtol=1e-2 for BF16.

## 6. What Would Justify Shipping This

Two non-overlapping conditions:
1. Kernel-only microbench shows **>= 96% HBM BW** at B=1, seq=4096,
   d=256, BF16 (ie, beats FA2's 93% by enough margin to matter).
2. End-to-end Gemma-4 decode shows **>= 1.03x** step-time speedup.

If (1) fails → KILL for SM120a. The architectural win
(warp-spec decouples LSU/TC) doesn't exist when you only have
cp.async + SM80 MMA. Ship only on SM90 (Hopper) or SM100
(B100/B200) where TMA + WGMMA live on separate schedulers.

## 7. Microbench Plan (GPU 1, <60s)

- B=1, seq_len=4096, head_dim=256, num_heads=1 (single-head decode).
- Compare vs `flashinfer.single_decode_with_kv_cache` (BF16 KV).
- Each kernel: 10 warmup + 100 timed iterations, cudaEventElapsed.
- Compute achieved BW = (KV bytes moved) / latency. Report percent
  of 1792 GB/s peak and speedup vs FlashInfer.

## 8. Prototype File

`kernels/csrc/warp_spec_decode_attention.cu` — ~350 lines, compiles
with `nvcc -arch=sm_120a -std=c++20`.

## 9. Measured Result (GPU 1, B=1 / H=128 / seq=4096 / d=256 / BF16)

| Kernel | Latency | HBM BW | % of 1792 GB/s | Speedup |
|--------|---------|--------|----------------|---------|
| warp-spec (this) | 472.1 us | 1137 GB/s | **63.5%** | 0.71x |
| FlashInfer single_decode | 335.0 us | 1602 GB/s | **89.4%** | 1.00x |

FlashInfer at 89.4% is within noise of the 93% FA2 target (different
workload shape — FA2's 93% was measured on paged-KV B=32). Warp-spec
kernel runs correctly (max rel err 1.6e-4 vs torch float) but is
**29% slower**.

Root cause: my prototype is 1 CTA per head (grid = num_heads). It
does NOT do split-K across the seq dimension — each CTA serially walks
its 4096-token KV. FlashInfer's decode does split-K (multiple CTAs
per head, each covering a seq chunk, combined in a reduction), which
lights up all 170 SMs. The warp-spec structure saves nothing here:
even with perfect producer/consumer overlap, the KV read is serialized
within one CTA.

### First-principles takeaway

The "gain" from warp-specialization on Hopper comes from **decoupling
TMA issue (async engine) from WGMMA (tensor core scheduler)** so the
two can run on different functional units simultaneously. SM120a has
**neither** instruction: loads go through the same LSU as regular
memory, MMA goes through the legacy SM80 tensor-core path that
cp.async_wait_group already overlaps implicitly.

Even if we add split-K and claw back to 90%+ BW, we would only be
matching FlashInfer, not beating it — and FlashInfer has battle-tested
paged-KV, GQA, and soft-cap support that would have to be re-implemented.

## 10. Verdict — KILL for SM120a

Warp-specialized BF16 decode does not beat FA2/FlashInfer on SM120a.
The micro-architectural premise (separate load/compute schedulers)
requires Hopper's TMA + WGMMA, which consumer Blackwell does not
expose.

**Where this pattern IS still promising on this hardware**:
- FP8 KV decode (FA2 doesn't support it on SM120) — 2x data reduction
  dominates any load-pipeline optimization. See
  `plans/sm120_attention_kernel.md` §"Real Opportunity".
- Prefill / chunked prefill (M > 64) where MMA utilization matters.

Recommend not pursuing warp-spec BF16 decode further on SM120a;
redirect to FP8 KV decode (separate workstream) for the Gemma-4 win.

