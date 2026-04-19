# FusenCache TMA v3 Feasibility — SM120a Intermediate Staging
<!-- Tag: W2_5b_tma_feasibility -->

Date: 2026-04-18  
Target: RTX PRO 6000 Blackwell Max-Q (SM120a, cc 12.0).  
Predecessor: warp-spec v2 KILL @ 16.8% BW (302 GB/s, 236 µs, 1.18× baseline).  
Question: can TMA on *contiguous intermediate staging buffers* break the remaining
compute+smem-staging overhead that warp-spec v2 could not move?

---

## 1. TMA Availability on SM120a — CONFIRMED (LIMITED)

### PTX ISA evidence

`cp.async.bulk.tensor` (TMA) was introduced on SM90 (Hopper). The PTX opcode:

```
cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem], [desc, {coord...}], [mbar];
```

is guarded by `__CUDA_ARCH__ >= 900` in NVIDIA's own FMHA warpspec headers
(`flashinfer-0.6.4-patched-backup/data/csrc/fmha_v2/fmha/hopper/utils_tma.h`,
lines 44, 65, 87, 115, 129, 139, 147). SM120a reports `__CUDA_ARCH__ = 1200`,
which satisfies `>= 900`. The nvcc 12.8 toolchain (`-arch=sm_120a`) emits this
class of instruction for Blackwell.

### What SM120a has

| Feature | SM90 (Hopper) | SM120a (Blackwell) |
|---------|:---:|:---:|
| `cp.async.bulk.tensor` (TMA) | YES | YES (`>= 900` guard passes) |
| `mbarrier.arrive.expect_tx` | YES | YES |
| `mbarrier.try_wait.parity` | YES | YES |
| TMA swizzle modes (32/64/128B) | YES | YES |
| WGMMA (async tensor core) | YES | NO (sm_120 only has `mma.sync`) |
| Multicast TMA (shared::cluster) | YES (cluster) | YES (cluster viable) |

### What TMA requires

1. A `CUtensorMap` descriptor (`cuTensorMapEncodeTiled`) set up on the **host**
   pointing at a **rectangular, contiguous, 16-B-aligned tile** in global memory.
2. A single-thread TMA issue instruction (one thread per CTA).
3. An mbarrier to signal completion (replaces cp.async.commit/wait_group).
4. Minimum tile size constraints: box dimension must be a multiple of 16 bytes
   in the innermost dimension; outer dimensions can be 1. Alignment: base address
   must be 16-B aligned (L2-sector aligned for L2::128B promotion).

### Load granularities

TMA bulk loads are not 128-B constrained per-issue; the **descriptor's box size**
determines how many bytes move. Valid innermost box widths: 16, 32, 64, 128, 256 bytes.
A single TMA issue can move up to `box_d0 × box_d1 × ... × box_dN-1` bytes where
the product is bounded only by the SMEM destination size. For our use case a 128-B
or 256-B innermost width with multi-token outer dimension would be typical.

---

## 2. Alignment Constraints vs FusenCache Buffers

Current `fusencache_decode_warpspec_v2.cu` SMEM layout (14.8 KiB total):

| Buffer | Size | Contiguous in gmem? | 16-B aligned? | TMA-eligible? |
|--------|-----:|:---:|:---:|:---:|
| Query (s_q, FP32) | 2,048 B | YES — one `[B,Hq,D]` tensor | YES (torch alloc) | YES |
| K tile ring (s_k_ring) | 6,144 B | **NO** — paged scatter | NO (indirected) | NO |
| V tile ring (s_v_ring) | 6,144 B | **NO** — paged scatter | NO (indirected) | NO |
| Scale ring (s_sc_ring) | 384 B | **NO** — paged scatter | NO | NO |
| QK warp scratch (s_qk_warp) | 24 B | N/A (register→smem spill) | N/A | NO (too small) |
| mid_out [B,Hq,splits,D+1] | 16 B per split per dim | YES — linear tensor | YES | MARGINAL |

The paged KV buffers (K, V, scales) are **definitionally ineligible** — each KV
token may occupy a different physical page, so no contiguous descriptor can cover
a full tile. This was the original rejection reason and remains unchanged.

### Candidate staging buffers for TMA

**(a) Query load into SMEM (s_q)**  
The query tensor `[B, Hq, D]` is dense BF16, allocated by PyTorch at 256-B
alignment. One `[1, D]` row = 512 B (D=256, BF16). Descriptor: 2D box
`(D=256, dtype=BF16) = 512 B` contiguous. Currently loaded by all 128 threads
cooperatively with `__bfloat162float` scalar converts (128 loads × 2 B).

With TMA: one warp-0 thread issues a single 512-B bulk load into s_q; mbarrier
signals completion. Consumers skip the cooperative load + __syncthreads.

**(b) mid_out write-back (stage-2 input)**  
`mid_out [B, Hq, num_splits, D+1]` is a linear FP32 tensor. Stage 1 writes
`D+1 = 257 floats = 1028 B` per (batch, head, split). This is a **store**, not
a load — TMA bulk store (`cp.async.bulk.shared::cta.global.bulk_group`) exists
but is less commonly beneficial in decode (store path is rarely the bottleneck).

**(c) QK score → smem broadcast**  
Size: 2 floats (8 B per token per head group). This is a register-to-smem
write plus warp-shuffle reduction — not a gmem→smem transfer. TMA does not help.

**(d) block_table lookups (page indirection)**  
`block_table [B, max_blocks_per_seq]` is int32, linear per batch. Each tile
reads `BLOCK_KV / page_size` entries (typically 1 entry for page_size=16,
BLOCK_KV=16). That is 4 bytes per lookup. TMA overhead (descriptor setup,
mbarrier, issue latency ~20–40 cycles) vastly exceeds the 4-B load. Not viable.

---

## 3. Per-Stage Benefit Analysis

### Stage (a): Query bulk load — MARGINAL POSITIVE

**Current cost (warp-spec v2):**  
All 128 threads load query into smem before the producer/consumer split.
128 threads × (2 `__bfloat162float` converts + 1 smem store) × `HALF_D/128` iters
= 2 × 128 iters of scalar convert. Estimated: ~8–12 cycles × 2 = ~16–24 cycles
at 100% IPC, but measured in practice as ~3–5 µs due to `__syncthreads` stall
(all 128 threads must finish before producer warp begins issuing cp.async).

**With TMA:**  
Producer warp issues one TMA for Q (512 B), mbarrier wait. All consumers wait on
mbarrier instead of __syncthreads. The BF16→FP32 convert still happens on the
consumer side from smem (TMA moves raw BF16; consumers convert on read). Saves:
the __syncthreads stall for Q load, roughly **1–2 µs** per block.

**Alignment:** torch `[B,Hq,D]` BF16 tensor is `stride(1) = D = 256 elements = 512 B`;
this satisfies 16-B innermost alignment. Descriptor is simple 1D (512 B, BF16).

**Caveat:** The BF16→FP32 conversion moves from load-time to consumer-read-time.
Net saves is only the __syncthreads overhead, not the convert latency.

### Stage (b): QK score staging into smem — NO BENEFIT

QK scores are computed by consumer warps in registers, then written to `s_qk_warp`
(24 B). This is register→smem, not gmem→smem. TMA cannot help.

### Stage (c): Softmax output broadcast — NO BENEFIT

`s_qk_final` (8 B) is written by lane 0 of consumer warp 1. Not a gmem transfer.

### Stage (d): V partial accumulator staging — NO BENEFIT

V accumulator (`acc_even`, `acc_odd`) lives entirely in registers across the tile
loop. It is only written to `mid_out` at the end of the split, as a global store.
TMA bulk store to global *could* replace the strided `mid_out` stores at split end,
but the store pattern is already coalesced (stride-2 interleave of even/odd dims)
and accounts for < 0.5% of measured kernel time.

### Stage (e): Paged KV loads — CONFIRMED INELIGIBLE

Scatter pattern via `block_table` indirection eliminates TMA as established.
No change from cpasync_tma.md analysis.

### Summary table

| Stage | Transfer type | Size | TMA eligible? | Estimated savings |
|-------|:---:|---:|:---:|---:|
| (a) Query → smem | gmem load | 512 B | YES | **1–2 µs / block** |
| (b) QK scores → smem | reg→smem | 8 B | NO | 0 |
| (c) Softmax broadcast | reg→smem | 8 B | NO | 0 |
| (d) V accumulator | reg→gmem (store) | ~2 KiB | MARGINAL | < 0.1 µs |
| (e) K tile ring | paged scatter | 2 KiB/tile | NO | 0 |
| (f) V tile ring | paged scatter | 2 KiB/tile | NO | 0 |
| (g) Scale ring | paged scatter | 128 B/tile | NO | 0 |

---

## 4. Projected Latency Post-TMA

**Warp-spec v2 baseline:** 236 µs (B=16, seq=2048, splits=16).

**Where the 236 µs goes (estimated from profiling of analogous kernels):**

| Component | Estimated fraction | µs |
|-----------|:-:|---:|
| K+V cp.async issue + wait (producer) | 35% | ~83 µs |
| 4-bit nibble unpack + FMA (consumer) | 30% | ~71 µs |
| Cross-warp QK reduce + softmax broadcast (consumer) | 20% | ~47 µs |
| V accumulate (consumer) | 10% | ~24 µs |
| Query load + __syncthreads | 3% | **~7 µs** |
| mid_out store-back | 2% | ~5 µs |

**TMA impact on query load:**  
Replacing the cooperative 128-thread scalar load + __syncthreads with TMA bulk
load + mbarrier saves the __syncthreads stall (not the compute itself). Estimated
1–2 µs of the ~7 µs query-load phase. Applied across 16 splits:
`236 µs × (1 - 1.5/7 × 0.03) ≈ 235 µs`. Essentially noise.

**Why TMA cannot touch the dominant bottlenecks:**  
The top three cost centers — paged KV loads (cp.async), nibble unpack+FMA, and
per-token softmax barrier — are all either ineligible for TMA (paged scatter) or
not gmem→smem transfers (ALU compute, register-level operations). TMA exclusively
accelerates gmem→smem transfers of contiguous rectangular tiles.

**Projected post-TMA latency:** 233–235 µs (< 1% improvement over 236 µs v2).

---

## 5. Warp-Spec v2 Compatibility

TMA + mbarrier is architecturally compatible with the 1P/3C topology:

- Producer warp (warp 0) issues TMA, monitors mbarrier `mbarrier.try_wait.parity`.
  This replaces the cooperative Q-load __syncthreads but not the per-tile
  `bar.sync BAR_FULL/BAR_EMPTY` rendezvous (which gates K/V readiness, not TMA).
- The Q-load TMA would be issued once per block (not per tile), with a separate
  mbarrier from the per-tile K/V mbarriers. This is a valid usage pattern.
- Integration complexity: ~50 LoC diff: add Q mbarrier init, TMA Q descriptor
  (host-side), producer TMA issue, consumer mbarrier wait replacing __syncthreads
  in Q-load section. ~1 day for a careful implementation.

---

## 6. Integration Complexity

| Task | Estimate |
|------|---:|
| Host-side `cuTensorMapEncodeTiled` for Q tensor | 2 hr |
| Producer warp TMA issue + mbarrier (replace Q __syncthreads) | 3 hr |
| Consumer mbarrier wait + BF16→FP32 convert from SMEM | 2 hr |
| Build + correctness verify | 2 hr |
| Bench + comparison | 1 hr |
| **Total** | **~1 day** |

---

## 7. Why the Remaining Ceiling Is Not TMA-Addressable

The warp-spec v2 plan (§6 of fusencache_warp_spec_v2.md) already identified the
correct next moves:

**(a) Warp-per-head split** — eliminates the `bar.sync BAR_CONS` cross-warp
reduce for QK, which accounts for ~20% of kernel time (47 µs). Two consumer warps
each own one head entirely; the intra-warp `__shfl_xor_sync` reduce is O(log32)
= 5 instructions with no smem rendezvous.

**(b) Tensor-core QK** — `mma.sync.m16n8k16` for the QK dot product across 16
KV tokens in one instruction. This collapses ~30% of kernel time (71 µs nibble
unpack + FMA into ~5 µs MMA). Requires bf16/fp16 K tiles (currently uint8 nibble
packed); a tile-conversion step in the producer warp would allow this.

**(c) Offline softmax** — accumulate all 16-token tile scores before broadcasting,
eliminating 16 per-token `bar.sync BAR_CONS` barriers down to 1 per tile.

Combined, (a)+(b)+(c) could reduce kernel time by 50–60%: 236 µs → 95–118 µs.
That would hit the 30% BW gate. TMA contributes < 1 µs.

---

## 8. Verdict

**KILL (TMA v3 for FusenCache decode intermediate staging)**

**Evidence:**
- TMA is **CONFIRMED** available on SM120a (`__CUDA_ARCH__ >= 900` guard passes
  for 1200; evidence in flashinfer FMHA headers).
- Only one stage qualifies for TMA: the Q bulk-load into SMEM. All K/V/scale
  staging is paged-scatter and definitionally ineligible.
- Maximum projected savings from TMA-Q: **1–2 µs** on a 236 µs kernel = **< 1%**.
- The remaining ceiling (83 µs paged-KV latency, 71 µs nibble unpack, 47 µs
  cross-warp QK reduce) is **not addressable by TMA** — it requires warp-per-head
  restructuring, tensor-core MMA, and per-tile offline softmax (v3 path).
- Integration cost (~1 day) is not justified for < 1% gain.

**Proceed path:** DEFER TMA until/unless the KV layout changes to contiguous pages
(page_size ≥ 64, sequence-level linear allocation). Then TMA bulk-loads of K+V
tiles become viable and could recover 30–40% of the paged-KV latency.

**Recommended next experiment:** fusencache_decode_v3 with (a) warp-per-head split
+ (b) MMA-based QK + (c) per-tile offline softmax. Estimated yield: 95–118 µs
(2.0–2.5× over v2's 236 µs), targeting ≥ 30% BW gate.
