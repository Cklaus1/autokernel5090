# FusenCache Decode — Warp-Specialized v2 (SM120a)

Date: 2026-04-17  Target: RTX PRO 6000 Blackwell Max-Q (SM120a, cc 12.0).
Predecessor: `fusencache_cpasync_tma.md` (KILL @ 14.2% BW, 1.24x baseline).

## 1. Why cp.async v1 stalled at 14.2% BW

v1 prototype confirmed the HBM path is **not** the gate. The consumer path
is fully serialised:

```
cp.async.wait  →  per-thread nibble unpack (scalar &0xF, >>4)
              →  per-thread FMA into QK partial
              →  2-level __syncthreads block reduction for score
              →  __syncthreads broadcast of softmax max/p
              →  per-thread V nibble unpack + PV FMA
              →  __syncthreads (next KV token)
```

Every one of the 32 KV tokens per split chain-syncs with `__syncthreads` at
least twice. cp.async already has data in SMEM before compute starts, but
compute cannot run ahead of the barrier. The LSU sits idle during the
compute tail; the ALU sits idle during the next cp.async issue.

## 2. Warp topology

**4 warps total (128 threads), 1 producer + 3 consumers.**

Option comparison:

| topology | producer threads | consumer threads | comment |
|---------:|-----------------:|-----------------:|---------|
| 2P / 2C  | 64               | 64               | producer over-provisioned — one warp handles 2 KiB cp.async in 16 instructions, second warp idle |
| **1P / 3C** | 32            | 96               | producer warp issues K+V+scales (136 instr / tile, ~1-2 µs latency-bounded), consumers get 3× the compute bandwidth for nibble unpack + FMA |
| 1P / 1C / 2R | 32         | 32+64            | splits reduction from PV; adds another barrier — skipped, not worth it here |

Rationale for 1P/3C:
- Producer work per tile: 128 cp.async.cg (K) + 128 (V) + 8 (scales) = 264
  16-B loads = 4.2 KiB / stage. One warp of 32 threads can issue these at
  ~1 instr/cycle (LSU-bound, not issue-bound). Second warp on the
  producer side would only duplicate work.
- Consumer work per KV token: 128 bytes of K nibble unpack (16 vectorized
  ops), QK FMA (128 muladd), warp reduce (log2(32)=5 shfl), softmax math,
  V unpack (same), PV FMA. 3 warps = 96 threads — each owns
  HALF_D/96 ≈ 1.3 dims. We round to 2 dims per thread for 3 warps, with
  the last warp's half-tail masked.
- Result: the producer's LSU work overlaps entirely with the consumers'
  ALU work. Barrier frequency drops from per-token `__syncthreads` to
  per-tile producer/consumer rendezvous (mbarrier arrive/wait).

## 3. Producer/consumer rendezvous

SM120a supports **async mbarrier** (`mbarrier.init.shared::cta.b64`,
`mbarrier.arrive.expect_tx`, `mbarrier.try_wait.parity`). Probe at build
time with an inline asm block; if it fails to assemble under `-arch=sm_120a`
we fall back to `cp.async.wait_group` + `bar.sync 0` (aligned barrier).

**Primary path (mbarrier)**:
```
__shared__ uint64_t bar_full[STAGES], bar_empty[STAGES];
// init: each bar_full expects 4224 bytes (K+V+scales tx count)
// Producer:
//   mbarrier.arrive.expect_tx bar_full[s], 4224
//   cp.async.cg ...  cp.async.commit_group
//   (cp.async completions auto-tick bar_full)
// Consumer:
//   mbarrier.try_wait.parity bar_full[s], phase
//   ... compute from s-ring ...
//   mbarrier.arrive bar_empty[s]
// Producer waits on bar_empty[s] before overwriting
```

**Fallback path (cp.async.wait_group + bar.sync.aligned)**:
Producer maintains `cp.async.commit_group` per stage. Before consumers
touch stage s, producer emits `cp.async.wait_group<STAGES-1>` and then
`bar.sync.aligned 0, BLOCK_THREADS`. Consumers use the same barrier to
gate. This works on SM80+; no mbarrier needed. **For v2 implementation
we use this fallback** for compile reliability; mbarrier is a v3 follow-up.

## 4. Shared-memory budget

| region                     | bytes     |
|----------------------------|-----------|
| Query (2 heads, FP32)      |   2,048   |
| K tile (2 KiB × 3 stages)  |   6,144   |
| V tile (2 KiB × 3 stages)  |   6,144   |
| Scales (128 B × 3 stages)  |     384   |
| Cross-warp QK scratch      |      32   |
| Softmax-score broadcast    |      16   |
| **Total**                  | **~14.8 KiB** |

Well under the 101 KiB opt-in cap. Occupancy will be thread-count bound
(128 threads per block, SMs host ~6 blocks each).

## 5. Vectorized nibble dequant (`prmt.b32` + `lop3.b32`)

The v1 kernel reads 1 byte at a time and unpacks with `&0xF` / `>>4`. v2
reads 4 bytes (one `uint32_t`) at a time per thread and unpacks 8 nibbles
in **3 PTX instructions** total, then applies the symmetric offset using
`lop3.b32`.

```ptx
// Given b32 input = {b3,b2,b1,b0}, each byte = (hi_nibble << 4) | lo_nibble.
// Want: out_lo = {0,b0_lo, 0,b1_lo, 0,b2_lo, 0,b3_lo} (as 4x u16 slots)
//       out_hi = {0,b0_hi, 0,b1_hi, 0,b2_hi, 0,b3_hi}
// prmt.b32 interleaves bytes; AND with 0x0F0F0F0F masks low nibbles.
// Finer: we want 8 individual nibble values in two 32-bit words (4 nibbles each,
// stored in u8 lanes ready for FP32 convert).

// Fast path: convert 4 lo-nibbles to 4 fp32 via integer -> float, combined with
// subtract of k_offset. We use the bitfield-extract pattern:

and.b32 lo4, packed, 0x0F0F0F0F;         // {b3&F, b2&F, b1&F, b0&F}
prmt.b32 hi4, packed, 0, 0x7531;          // extract bytes {0,b3,0,b2,0,b1,0,b0}
// wait — we want high nibbles, let's redo:
// Better pattern using shift-then-mask:
shr.b32 hi4_packed, packed, 4;            // each byte now has hi in low 4 bits, old hi gone
and.b32 hi4, hi4_packed, 0x0F0F0F0F;      // clean hi nibbles packed 4-per-u32
// (lo4 as before)
// Now lo4 and hi4 are each {0..15}×4-per-32.
// Broadcast to FP32: cvt.rn.f32.u8 after unpacking byte-by-byte
// OR convert 4 packed u8 -> 4 packed FP16 via cvt.rn.f16x2.s16 sequence:
//   movmsk → two u16 lanes, cvt to FP16.
// We use a pragmatic path: unpack the two u32s into 8 u16 lanes via prmt,
// then cvt each pair to fp16x2 with sub-offset fused:
prmt.b32 lo_b2, lo4, 0x0, 0x1504;         // lanes {b0,0,b1,0} as u16x2 lo word
prmt.b32 lo_b3, lo4, 0x0, 0x3726;         // lanes {b2,0,b3,0} as u16x2 hi word
// each u16x2 now represents {(int)b_i, 0} — interpret as s16x2, cvt to f16x2,
// subtract 7.5:
cvt.rn.f16x2.s16x2 lo_f2, lo_b2;
cvt.rn.f16x2.s16x2 lo_f3, lo_b3;
sub.f16x2 lo_f2, lo_f2, OFFSET_F16X2;
sub.f16x2 lo_f3, lo_f3, OFFSET_F16X2;
```

Two `prmt.b32` + two `cvt.rn.f16x2.s16x2` + two `sub.f16x2` per 4 bytes of
K (= 8 nibbles = 8 K elements). Compared to v1's 8 `cvt.rn.f32.u32` +
8 `sub.f32` + 8 `cvt.rn.f32.f16` per 4 bytes (= ~24 ops), v2 halves the
unpack op count and lets the FMA path issue as `fma.rn.f16x2` if Q is
pre-converted to FP16x2 (or keep FP32 path with one extra `cvt.f32.f16x2`).

**Pragmatic v2 impl**: keep the FP32 accumulator path (same as v1 for
correctness), but switch the K/V inner loop to read `uint32_t` from SMEM
and unpack 4 bytes per load via:
```cpp
uint32_t packed = *reinterpret_cast<const uint32_t*>(k_row + i*4);
uint32_t lo4 = packed & 0x0F0F0F0F;
uint32_t hi4 = (packed >> 4) & 0x0F0F0F0F;
// Extract 4 lo-nibbles & 4 hi-nibbles as 8 fp32 values via bit-ops,
// subtract offset, multiply by scale (shared across 4 dims since
// scale_block=64 > 4), add to QK partial.
```

## 6. Why this CAN beat cp.async alone

- **Distinct pipes**: cp.async drives LSU; nibble unpack+FMA drives ALU
  (and with `fma.rn.f16x2`, the Tensor Core adjacent FMA pipe). These
  issue in parallel on SM120a.
- **Barrier count drops**: v1 issues `__syncthreads` per KV token (32/split).
  v2 issues `bar.sync.aligned` per **tile** (16 KV tokens/split → 2
  barriers/split). 16× barrier reduction.
- **Producer never blocks on consumer**: producer holds 2 cp.async groups
  outstanding; `wait_group<1>` allows the current stage to drain while
  the next is in-flight.
- **Consumer never blocks on producer** (steady state): after prologue of
  STAGES-1 tiles, each iteration finds its tile ready.
- Proof-of-feasibility: the earlier cooperative-grid experiment showed
  `grid.sync()` at 0.82 µs/barrier on SM120a. Our tile cadence is ~8 µs
  (256 KV tokens × 32 ns/tok = 8 µs), so barriers are < 10% overhead.

## 7. Go/no-go gates

- Compile must succeed in 2 attempts.
- Non-NaN, < 1e-3 max-abs error vs baseline C++ decode → correctness PASS.
- BW < 30% of 1792 GB/s → **KILL** (warp-spec didn't move the needle).
- 30-50% → PARTIAL (needs mega-graph / further fusion).
- ≥ 50% → **PASS** (production candidate).

## 8. Implementation scope (v2 prototype)

- 4-bit K + 4-bit V only, scale_block=64, HEAD_DIM=256, BLOCK_KV=16,
  STAGES=3, BLOCK_THREADS=128 (1 producer warp 0; consumers warps 1-3).
- Reuse v1 stage-2 reduction (`decode_stage2<256, 256>`).
- Same TORCH_LIBRARY signature as cp.async v1, exposed as
  `torch.ops.fusencache_warpspec.decode_attention`.
- Single-kernel path; no cooperative grid launch needed.
