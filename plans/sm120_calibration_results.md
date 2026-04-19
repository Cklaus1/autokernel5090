# SM120a Extended Calibration Results

**Tag:** W6_sm120_calibration_extended
**Date run:** _parent fills in_
**Hardware:** RTX PRO 6000 Blackwell, SM120a, 188 SMs, 1792 GB/s GDDR7
**CUDA toolkit:** _parent fills in (expected 12.8+)_
**SM clock (GHz):** _parent fills in via `nvidia-smi -q | grep "Graphics Clock"`_

Run command:
```bash
cd kernels/csrc
python3 build_sm120_calibration.py           # first time only
python3 run_sm120_calibration.py --test all --clock-ghz <actual_ghz> --table
```

---

## BENCH 1 — grid.sync cost (3 conditions)

Reference: 3 µs measured today under WMMA-active cooperative kernel (v5b audit).

| Condition | cycles/barrier | µs/barrier | Notes |
|---|---|---|---|
| 1a: WMMA-idle | _TBD_ | _TBD_ | No tensor-core activity between syncs |
| 1b: WMMA-active | _TBD_ | **~3 µs expected** | Matches today's measurement |
| 1c: cp.async-active | _TBD_ | _TBD_ | cp.async.ca.shared between syncs |
| **spread (max - min)** | — | _TBD_ | <1 µs = stable; ≥1 µs = state-dependent |

**Implication for mega_graph design:**
- If all 3 ≈ 3 µs: barrier budget in mega_graph_cooperative_kernel.md §2.2 is validated.
- If spread > 1 µs: recompute §2.2 tables with worst-case (highest) value.

---

## BENCH 2 — wmma::load vs cp.async crossover

Reference: v4a KILL'd cp.async at N=16 tile shapes because wmma::load already overlaps via LSU.

| B-tile | K-steps | wmma_us | cpasync_us | Winner | Ratio |
|---|---|---|---|---|---|
| 16 | 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 16 | 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 16 | 16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 16 | 32 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 32 | 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 32 | 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 32 | 16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 32 | 32 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 64 | 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 64 | 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 64 | 16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 64 | 32 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 128 | 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 128 | 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 128 | 16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 128 | 32 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 256 | 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 256 | 8 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 256 | 16 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 256 | 32 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Crossover point:** _TBD_ (B-tile ≥ ___ AND K-steps ≥ ___)

**Implication for KILL verdicts:**
- If no crossover: v4a cp.async KILL is valid at all decode-relevant shapes. Keep KILL.
- If crossover at B-tile ≥ 64: cp.async prefetch worth revisiting for v5c+ with larger tiles.

---

## BENCH 3 — mma.sync BF16 throughput

Reference: Hopper H100 ~1000 TOPS BF16/SM at tensor-core peak.
Prediction: SM120a consumer Blackwell 500-800 TOPS/SM (lower clock, same or slightly smaller TC array).

| Level | TOPS BF16 | cycles | ops | Notes |
|---|---|---|---|---|
| per-warp (1 warp, block 0) | _TBD_ | _TBD_ | _TBD_ | Minimal occupancy |
| per-SM (8 warps, block 0) | _TBD_ | _TBD_ | _TBD_ | Full block, 1 SM |
| full-SM occupancy (all SMs) | _TBD_ | _TBD_ | _TBD_ | All 188 SMs, cooperative |

**Interpretation guide:**
- `full_sm / per_sm < 0.9`: register spill or occupancy bottleneck hurts SM utilization.
- `full_sm / per_warp ≈ 8`: scales linearly with warps — ideal.
- Compare to 1000 TOPS (H100): ratio gives SM120a tensor-core efficiency vs Hopper.

---

## BENCH 4 — cvt.rn.satfinite.e2m1x2.f32 throughput

Reference: KILL_PATTERNS.md — "encode only (fp32→fp4×2)", no decode direction.
Prediction: 1–4 cycles/conversion (single-cycle integer pipe instruction, possibly 2 due to dependency chain).

| Metric | Value |
|---|---|
| Total cycles (8192 conversions) | _TBD_ |
| Cycles per fp4x2 pair | _TBD_ |
| ns per fp4x2 pair | _TBD_ |
| Effective throughput (pairs/cycle) | _TBD_ |

**Implication:**
- If > 4 cycles/pair: v6 FP4 dequant loop was serial-instruction-bound, not just smem-round-trip-bound.
  The KILL verdict (0.50× at H=2048) is partly from this instruction, not fixable by layout alone.
- If ≤ 2 cycles/pair: smem serialization (not the instruction) was the primary bottleneck in v6.

---

## BENCH 5 — __shfl_sync vs atomicAdd on smem

Reference: persistent_moe_dispatch.cu W5_D1 bug — atomicAdd race in phase1_route_and_shuffle.
Prediction: shfl_sync ~30-50% faster; atomic serialization at 8-warp contention.

| Method | ns/reduction | total cycles | Notes |
|---|---|---|---|
| `__shfl_sync` tree (8 warps) | _TBD_ | _TBD_ | 3-level tree + smem write + warp-0 reduce |
| `atomicAdd` to smem (8 warps) | _TBD_ | _TBD_ | All 8 leaders contend on one smem address |
| **Winner** | _TBD_ | — | by _TBD_× |

**Implication:**
- Fixes the W5_D1 race by replacing atomicAdd with shfl_sync.
- If shfl_sync is ALSO faster: double win (correctness + performance).
- If atomicAdd is faster despite the race: fix for correctness only; accept minor perf regression.

---

## BENCH 6 — HBM sustained BW at M=1 decode

Reference: mega-graph dense-proxy hits 26% HBM utilization (1792 GB/s × 26% ≈ 466 GB/s).
Prediction: strided decode pattern ≈ 20-35% of nameplate; sequential ceiling ≈ 70-85%.

| Pattern | GB/s | % of 1792 GB/s | Notes |
|---|---|---|---|
| Sequential coalesced (ceiling) | _TBD_ | _TBD_ | Pure 128-bit stride read/write |
| Strided M=1 decode (col stride=2048) | _TBD_ | _TBD_ | Mimics weight column access per token |

**Critical interpretation:**
- If strided BW ≈ 26% of nameplate: the 26% figure in KILL_PATTERNS.md is the
  **hardware ceiling** at M=1 decode, not a kernel efficiency shortfall.
  This means FP4 compression cannot buy more BW headroom than already exists.
  All "FP4 reduces weight bytes, gets more BW" projections need revisiting.
- If strided BW >> 26%: the mega-graph kernel IS leaving BW on the table;
  optimization headroom exists.

---

## BENCH 7 — CUDA graph node-count crash point

Reference: Discoveries #56-58 — crashes at ~450 nodes (30 layers × ~15 nodes/layer).
mega_graph_cooperative_kernel.md §1.1 cites this as motivation for the persistent-kernel approach.

| N nodes | Capture | Replay (avg) | Status |
|---|---|---|---|
| 50 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 100 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 150 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 200 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 250 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 300 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 350 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 400 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 450 | _TBD_ ms | _TBD_ ms | _TBD_ |
| 500 | _TBD_ ms | _TBD_ ms | _TBD_ |

**Crash node count:** _TBD_

**Implication:**
- If crash at N ≈ 450: confirms Discovery #56-58. Persistent-kernel approach justified.
- If no crash at 500: Discovery #56-58 may be batch-size or kernel-content specific.
  Extend sweep or reproduce with real Gemma4 graph structure.
- Capture-time scaling (ms vs N) may reveal O(N) or O(N²) internal bookkeeping.

---

## BENCH 8 — TMA cp.async.bulk.tensor availability

Reference: conflicting evidence — W5_5b says guarded by `__CUDA_ARCH__ >= 900` (passes for sm_1200),
but NVIDIA docs say consumer Blackwell lacks TMA.

| Test | Result |
|---|---|
| ptxas compile acceptance (sm_120a) | _TBD_ (check build log) |
| Runtime: instruction executes without fault | _TBD_ |
| **TMA verdict** | _TBD_: AVAILABLE / ABSENT / INCONCLUSIVE |

**Expected result:** TMA ABSENT.
- If TMA ABSENT: any code path behind `__CUDA_ARCH__ >= 900` that uses cp.async.bulk.tensor
  will fault at runtime on SM120a. All such guards must be changed to `__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1200` (i.e., explicitly exclude consumer Blackwell).
- If TMA AVAILABLE: the upstream guard is correct and TMA can be used on SM120a.
  This would be a significant positive finding — TMA enables software pipelining for the attention path.

---

## Priority ranking (run this order)

1. **BENCH 8 (TMA)** — compile-time answer first; if compile fails, that's already the result.
2. **BENCH 6 (HBM BW)** — most directly impactful on open KILL verdicts (FP4 BW headroom claims).
3. **BENCH 1 (grid.sync spread)** — validates or invalidates the mega-graph barrier budget table.
4. **BENCH 3 (mma TOPS)** — cite-worthy; affects all TFLOPS projections.
5. **BENCH 4 (cvt_fp4)** — confirms or refutes "dequant-in-WMMA is serial-bound" KILL reasoning.
6. **BENCH 5 (shfl vs atomic)** — correctness fix for W5_D1; perf data is a bonus.
7. **BENCH 2 (crossover)** — only matters if planning larger-tile versions of fused kernels.
8. **BENCH 7 (graph crash)** — useful confirmation but persistent-kernel path is already designed.
