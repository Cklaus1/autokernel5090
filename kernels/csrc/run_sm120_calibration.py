#!/usr/bin/env python3
"""
SM120a Calibration Microbenchmark Runner — W6_sm120_calibration_extended

Usage:
    python3 run_sm120_calibration.py --test all
    python3 run_sm120_calibration.py --test grid_sync
    python3 run_sm120_calibration.py --test wmma_cpasync
    python3 run_sm120_calibration.py --test mma_throughput
    python3 run_sm120_calibration.py --test cvt_fp4
    python3 run_sm120_calibration.py --test shfl_vs_atomic
    python3 run_sm120_calibration.py --test hbm_bw
    python3 run_sm120_calibration.py --test graph_nodes
    python3 run_sm120_calibration.py --test tma_probe

    python3 run_sm120_calibration.py --test all --clock-ghz 2.61 --num-sms 188

Options:
    --clock-ghz FLOAT   SM clock in GHz (default: 2.61 for PRO 6000 boost clock).
                        Obtain actual value via: nvidia-smi -q | grep "Graphics Clock"
    --num-sms INT       Number of SMs for cooperative kernels (default: 188 for PRO 6000)
    --iters INT         Override iteration count for timing loops (default: per-bench)
    --table             Print markdown table suitable for pasting into KILL_PATTERNS.md
    --csv               Print CSV output
    --out FILE          Write results to FILE (markdown)

Prerequisites:
    python3 build_sm120_calibration.py  (builds sm120_calib_bench)

The runner drives the compiled binary via Python subprocess for non-cooperative
benches, and uses the CUDA Python / ctypes interface for cooperative kernel
launches (which require cudaLaunchCooperativeKernel).

For bench7 (graph node-count stress test), the runner uses PyTorch CUDA graph
APIs directly — no compiled binary needed for that bench.

Output:
    Per-bench: raw cycle count, µs (derived from --clock-ghz), interpretation.
    At end: summary table with measured values vs KILL_PATTERNS.md entries.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Optional imports — fail gracefully if torch not present
# ---------------------------------------------------------------------------
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

SCRIPT_DIR = Path(__file__).parent.resolve()
BENCH_BIN  = SCRIPT_DIR / "sm120_calib_bench"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Result buffer slots (must match sm120_calibration_bench.cu layout)
SLOT_B1_T0     = 0
SLOT_B1_T1     = 1
SLOT_B1_ITERS  = 2
SLOT_B1_TAG    = 3
SLOT_B2_BASE   = 8   # [8 + combo*2] = cycles, [8 + combo*2 + 1] = tag
SLOT_B3_CYCLES = 16
SLOT_B3_OPS    = 17
SLOT_B4_CYCLES = 18
SLOT_B4_N      = 19
SLOT_B5_SHFL   = 20
SLOT_B5_ATOM   = 21
SLOT_B6_SEQ_CY = 22
SLOT_B6_SEQ_B  = 23
SLOT_B6_STR_CY = 24
SLOT_B6_STR_B  = 25
SLOT_B8_RESULT = 44

MAX_RESULTS    = 64

BENCH2_BTILE_SIZES  = [16, 32, 64, 128, 256]
BENCH2_KSTEP_COUNTS = [4, 8, 16, 32]
BENCH2_COMBOS = [(b, k) for b in BENCH2_BTILE_SIZES for k in BENCH2_KSTEP_COUNTS]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def cycles_to_us(cycles: int, clock_ghz: float) -> float:
    """Convert cycle count to microseconds given SM clock in GHz."""
    return cycles / (clock_ghz * 1e3)


def cycles_to_ns(cycles: int, clock_ghz: float) -> float:
    return cycles / clock_ghz


def header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def row(label: str, value: str, note: str = "") -> None:
    note_str = f"  # {note}" if note else ""
    print(f"  {label:<40s} {value}{note_str}")


# ---------------------------------------------------------------------------
# CUDA helper via torch (when available)
# ---------------------------------------------------------------------------

def require_torch(bench_name: str) -> None:
    if not HAS_TORCH:
        print(f"SKIP {bench_name}: torch not available. Install with: pip install torch")
        sys.exit(0)
    if not torch.cuda.is_available():
        print(f"SKIP {bench_name}: no CUDA device available.")
        sys.exit(0)


def alloc_result_buf() -> "torch.Tensor":
    """Allocate device result buffer (MAX_RESULTS × uint64)."""
    buf = torch.zeros(MAX_RESULTS, dtype=torch.int64, device="cuda")
    return buf


def read_result(buf: "torch.Tensor", slot: int) -> int:
    """Read one uint64 slot from result buffer (as Python int)."""
    val = buf[slot].item()
    # torch.int64 is signed; reinterpret as unsigned
    if val < 0:
        val = val + (1 << 64)
    return val


# ---------------------------------------------------------------------------
# Bench 1 — grid.sync cost
# ---------------------------------------------------------------------------

def bench_grid_sync(clock_ghz: float, num_sms: int, iters: int = 200) -> dict:
    """
    Measures grid.sync() overhead in three conditions:
      1a: WMMA-idle   (no tensor-core activity between syncs)
      1b: WMMA-active (mma.sync fires between each sync)
      1c: cp.async-active (cp.async fires between each sync)

    Returns dict: {condition: µs_per_barrier}
    """
    require_torch("bench1_grid_sync")
    header("BENCH 1 — grid.sync cost at OCCUPIED state (3 conditions)")

    if not BENCH_BIN.exists():
        print(f"ERROR: {BENCH_BIN} not found. Run: python3 build_sm120_calibration.py")
        return {}

    results = {}

    # Load the compiled extension as a shared library via torch.ops or ctypes.
    # For cooperative kernels we use torch custom CUDA extensions.
    # Since we built a standalone binary (not a .so), we run it as subprocess
    # with a --bench argument that selects which sub-bench runs.

    for cond_id, cond_name in [(1, "wmma_idle"), (2, "wmma_active"), (3, "cpasync_active")]:
        print(f"\n  Running condition {cond_id}: {cond_name} ...")
        env = os.environ.copy()
        cmd = [str(BENCH_BIN), "--bench", "grid_sync",
               "--cond", str(cond_id),
               "--iters", str(iters),
               "--num-sms", str(num_sms),
               "--clock-ghz", str(clock_ghz)]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print(f"    FAILED (rc={result.returncode})")
            print("    STDERR:", result.stderr[:500])
            results[cond_name] = None
            continue
        # Parse output: expect line "RESULT cycles=<N> iters=<M>"
        cycles = None
        for line in result.stdout.splitlines():
            if line.startswith("RESULT"):
                parts = dict(p.split("=") for p in line.split()[1:] if "=" in p)
                cycles = int(parts.get("cycles", 0))
                break
        if cycles is None:
            print(f"    Could not parse output:\n{result.stdout[:300]}")
            results[cond_name] = None
            continue
        us_per_barrier = cycles_to_us(cycles // iters, clock_ghz)
        results[cond_name] = us_per_barrier
        row(f"  grid.sync [{cond_name}]",
            f"{us_per_barrier:.2f} µs/barrier",
            f"raw {cycles} cycles / {iters} iters")

    print()
    print("  REFERENCE: today's measured value = 3.0 µs (WMMA-active, v5b audit)")
    print("  EXPECTED: all 3 conditions should be within ±0.5 µs if barrier cost")
    print("  is dominated by SM rendezvous latency rather than execution state.")
    if results:
        vals = [v for v in results.values() if v is not None]
        if vals:
            spread = max(vals) - min(vals)
            print(f"  SPREAD: {spread:.2f} µs — {'stable' if spread < 1.0 else 'UNSTABLE: report spread'}")
    return results


# ---------------------------------------------------------------------------
# Bench 2 — wmma::load vs cp.async crossover sweep
# ---------------------------------------------------------------------------

def bench_wmma_cpasync(clock_ghz: float, iters: int = 100) -> dict:
    """
    Sweeps B-tile size × K-step count, measures wmma-load vs cp.async path.
    Returns dict: {(b_tile, k_steps): {'wmma_us': float, 'cpasync_us': float}}
    """
    require_torch("bench2_wmma_cpasync")
    header("BENCH 2 — wmma::load vs cp.async crossover sweep")

    if not BENCH_BIN.exists():
        print(f"ERROR: {BENCH_BIN} not found.")
        return {}

    results = {}
    print(f"  {'B-tile':>8}  {'K-steps':>8}  {'wmma_us':>10}  {'cpasync_us':>12}  {'winner':>8}")
    print("  " + "-" * 52)

    for idx, (b_tile, k_steps) in enumerate(BENCH2_COMBOS):
        # wmma path
        wmma_us = _run_bench2_single(b_tile, k_steps, "wmma", iters, clock_ghz)
        cpas_us = _run_bench2_single(b_tile, k_steps, "cpasync", iters, clock_ghz)
        results[(b_tile, k_steps)] = {"wmma_us": wmma_us, "cpasync_us": cpas_us}
        if wmma_us is not None and cpas_us is not None:
            winner = "wmma" if wmma_us < cpas_us else "cpasync"
            ratio = cpas_us / wmma_us if wmma_us > 0 else float("nan")
            print(f"  {b_tile:>8}  {k_steps:>8}  {wmma_us:>10.3f}  {cpas_us:>12.3f}  "
                  f"{winner:>8}  (ratio={ratio:.2f}×)")
        else:
            print(f"  {b_tile:>8}  {k_steps:>8}  {'ERR':>10}  {'ERR':>12}")

    # Identify crossover
    crossover = None
    for b_tile in BENCH2_BTILE_SIZES:
        for k_steps in BENCH2_KSTEP_COUNTS:
            r = results.get((b_tile, k_steps), {})
            w = r.get("wmma_us")
            c = r.get("cpasync_us")
            if w is not None and c is not None and c < w:
                crossover = (b_tile, k_steps)
                break
        if crossover:
            break

    print()
    if crossover:
        print(f"  CROSSOVER: cp.async wins at B-tile={crossover[0]}, K-steps={crossover[1]}")
        print(f"  IMPLICATION: KILL of cp.async (v4a) is valid below B-tile={crossover[0]}.")
        print(f"  Revisit at larger B-tiles or higher K-step counts.")
    else:
        print("  No crossover found in sweep — wmma::load wins at all tested shapes.")
        print("  CONFIRMS: v4a cp.async KILL verdict valid for N=16 tile shapes.")
    return results


def _run_bench2_single(b_tile: int, k_steps: int, method: str,
                        iters: int, clock_ghz: float) -> Optional[float]:
    cmd = [str(BENCH_BIN), "--bench", "wmma_cpasync",
           "--b-tile", str(b_tile), "--k-steps", str(k_steps),
           "--method", method, "--iters", str(iters),
           "--clock-ghz", str(clock_ghz)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            parts = dict(p.split("=") for p in line.split()[1:] if "=" in p)
            cycles = int(parts.get("cycles", 0))
            return cycles_to_us(cycles // max(iters, 1), clock_ghz)
    return None


# ---------------------------------------------------------------------------
# Bench 3 — mma.sync BF16 throughput
# ---------------------------------------------------------------------------

def bench_mma_throughput(clock_ghz: float, num_sms: int, iters: int = 1000) -> dict:
    """
    Measures mma.sync m16n16k16 BF16 throughput at three occupancy levels.
    Returns dict with per-warp, per-SM, full-SM-occupancy TOPS estimates.
    """
    require_torch("bench3_mma_throughput")
    header("BENCH 3 — mma.sync BF16 throughput on SM120a")

    if not BENCH_BIN.exists():
        print(f"ERROR: {BENCH_BIN} not found.")
        return {}

    results = {}

    for level, coop in [("per_warp", False), ("per_sm", False), ("full_sm_occupancy", True)]:
        cmd = [str(BENCH_BIN), "--bench", "mma_throughput",
               "--level", level,
               "--iters", str(iters),
               "--num-sms", str(num_sms),
               "--clock-ghz", str(clock_ghz)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAILED [{level}]: {r.stderr[:200]}")
            results[level] = None
            continue
        cycles, ops = None, None
        for line in r.stdout.splitlines():
            if line.startswith("RESULT"):
                parts = dict(p.split("=") for p in line.split()[1:] if "=" in p)
                cycles = int(parts.get("cycles", 0))
                ops    = int(parts.get("ops", 0))
                break
        if cycles is None or ops is None:
            results[level] = None
            continue
        # FLOPS = ops × 2 × 16×16×16 (each mma does 2*M*N*K FMAs)
        flops = ops * 2 * 16 * 16 * 16
        secs  = cycles / (clock_ghz * 1e9)
        tops  = flops / secs / 1e12
        results[level] = tops
        row(f"  mma_throughput [{level}]",
            f"{tops:.1f} TOPS BF16",
            f"{cycles} cycles, {ops} ops")

    print()
    print("  REFERENCE: Hopper H100 ~1000 TOPS BF16 per SM (tensor-core peak).")
    print("  SM120a consumer Blackwell expected: 500-800 TOPS/SM (prediction).")
    print("  NOTE: per_warp < per_sm < full_sm_occupancy is the expected order.")
    print("        If full_sm < per_sm, register pressure is limiting occupancy.")
    return results


# ---------------------------------------------------------------------------
# Bench 4 — cvt.rn.satfinite.e2m1x2.f32 throughput
# ---------------------------------------------------------------------------

def bench_cvt_fp4(clock_ghz: float) -> dict:
    """
    Measures fp32 -> fp4x2 conversion throughput via PTX cvt instruction.
    Returns: cycles_per_conversion, ns_per_conversion
    """
    require_torch("bench4_cvt_fp4")
    header("BENCH 4 — cvt.rn.satfinite.e2m1x2.f32 throughput (fp32->fp4x2)")

    if not BENCH_BIN.exists():
        print(f"ERROR: {BENCH_BIN} not found.")
        return {}

    cmd = [str(BENCH_BIN), "--bench", "cvt_fp4", "--clock-ghz", str(clock_ghz)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[:300]}")
        return {}

    cycles, n = None, None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            parts = dict(p.split("=") for p in line.split()[1:] if "=" in p)
            cycles = int(parts.get("cycles", 0))
            n      = int(parts.get("n", 0))
            break

    if cycles is None:
        print(f"  Could not parse output:\n{r.stdout[:300]}")
        return {}

    cycles_per_pair = cycles / (n // 2) if n else 0
    ns_per_pair     = cycles_to_ns(cycles_per_pair, clock_ghz)
    cycles_per_fp32 = cycles_per_pair / 2
    ns_per_fp32     = ns_per_pair / 2

    row("  cycles/fp4x2 pair", f"{cycles_per_pair:.2f} cycles")
    row("  ns/fp4x2 pair",     f"{ns_per_pair:.2f} ns")
    row("  cycles/fp32 input", f"{cycles_per_fp32:.2f} cycles")
    row("  ns/fp32 input",     f"{ns_per_fp32:.2f} ns")

    print()
    print("  CONTEXT: KILL_PATTERNS.md says cvt is 'encode only (fp32->fp4x2)'.")
    print("  This bench confirms throughput. If >2 cycles/op, serial dequant loop")
    print("  in v6 (KILL'd) was dominated by this instruction, not smem.")
    print("  PREDICTION: 1-4 cycles/op (single scalar issue on INT pipe).")

    return {
        "cycles_per_pair": cycles_per_pair,
        "ns_per_pair":     ns_per_pair,
        "total_cycles":    cycles,
        "n":               n,
    }


# ---------------------------------------------------------------------------
# Bench 5 — __shfl_sync vs atomicAdd on smem
# ---------------------------------------------------------------------------

def bench_shfl_vs_atomic(clock_ghz: float, iters: int = 10000) -> dict:
    """
    Compares __shfl_sync tree reduction vs atomicAdd-to-smem for cross-warp
    reduction in a 256-thread block (8 warps).
    """
    require_torch("bench5_shfl_vs_atomic")
    header("BENCH 5 — __shfl_sync vs atomicAdd on smem (cross-warp reduction)")

    if not BENCH_BIN.exists():
        print(f"ERROR: {BENCH_BIN} not found.")
        return {}

    cmd = [str(BENCH_BIN), "--bench", "shfl_vs_atomic",
           "--iters", str(iters), "--clock-ghz", str(clock_ghz)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[:300]}")
        return {}

    shfl_cy, atom_cy = None, None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            parts = dict(p.split("=") for p in line.split()[1:] if "=" in p)
            shfl_cy = int(parts.get("shfl_cycles", 0))
            atom_cy = int(parts.get("atom_cycles", 0))
            break

    if shfl_cy is None:
        print(f"  Could not parse:\n{r.stdout[:300]}")
        return {}

    shfl_ns = cycles_to_ns(shfl_cy // iters, clock_ghz)
    atom_ns = cycles_to_ns(atom_cy // iters, clock_ghz)

    row("  shfl_sync tree (8 warps -> 1)", f"{shfl_ns:.1f} ns/reduction",
        f"{shfl_cy} total cycles")
    row("  atomicAdd to smem (8 warps)",  f"{atom_ns:.1f} ns/reduction",
        f"{atom_cy} total cycles")
    winner = "shfl_sync" if shfl_cy < atom_cy else "atomicAdd"
    ratio  = max(shfl_cy, atom_cy) / max(min(shfl_cy, atom_cy), 1)
    row("  Winner", f"{winner} by {ratio:.2f}×")

    print()
    print("  CONTEXT: persistent_moe_dispatch.cu W5_D1 flag — atomicAdd race.")
    print("  If atomicAdd is also SLOWER, the fix (shfl_sync) is both correct")
    print("  AND faster. If atomicAdd is somehow faster, need to fix race only.")
    print("  PREDICTION: shfl_sync ~30-50% faster; atomicAdd adds serialization")
    print("  at 8-warp contention even with smem proximity advantage.")

    return {
        "shfl_ns": shfl_ns,
        "atom_ns": atom_ns,
        "winner":  winner,
        "ratio":   ratio,
    }


# ---------------------------------------------------------------------------
# Bench 6 — HBM sustained BW at M=1 decode
# ---------------------------------------------------------------------------

def bench_hbm_bw(clock_ghz: float, num_sms: int) -> dict:
    """
    Measures HBM bandwidth via:
      (a) Pure sequential coalesced copy (ceiling BW)
      (b) Strided read at HIDDEN=2048 fp16 columns (M=1 decode pattern)
    """
    require_torch("bench6_hbm_bw")
    header("BENCH 6 — HBM sustained BW at M=1 decode traffic pattern")

    if not BENCH_BIN.exists():
        print(f"ERROR: {BENCH_BIN} not found.")
        return {}

    cmd = [str(BENCH_BIN), "--bench", "hbm_bw",
           "--num-sms", str(num_sms),
           "--clock-ghz", str(clock_ghz)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[:300]}")
        return {}

    seq_cy, seq_b, str_cy, str_b = None, None, None, None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            parts = dict(p.split("=") for p in line.split()[1:] if "=" in p)
            seq_cy = int(parts.get("seq_cycles", 0))
            seq_b  = int(parts.get("seq_bytes", 0))
            str_cy = int(parts.get("str_cycles", 0))
            str_b  = int(parts.get("str_bytes", 0))
            break

    if seq_cy is None:
        print(f"  Could not parse:\n{r.stdout[:300]}")
        return {}

    def bw_gb_s(cycles: int, bytes_xfer: int) -> float:
        secs = cycles / (clock_ghz * 1e9)
        return bytes_xfer / secs / 1e9 if secs > 0 else 0.0

    seq_gb_s = bw_gb_s(seq_cy, seq_b)
    str_gb_s = bw_gb_s(str_cy, str_b)
    nameplate_gb_s = 1792.0  # PRO 6000 GDDR7 nameplate
    seq_pct = seq_gb_s / nameplate_gb_s * 100
    str_pct = str_gb_s / nameplate_gb_s * 100

    row("  Sequential (ceiling BW)", f"{seq_gb_s:.0f} GB/s ({seq_pct:.1f}% of 1792 GB/s)",
        f"{seq_b//1024//1024} MB transferred")
    row("  Strided M=1 decode pattern", f"{str_gb_s:.0f} GB/s ({str_pct:.1f}% of 1792 GB/s)",
        f"{str_b//1024//1024} MB transferred")

    print()
    print("  CONTEXT: mega-graph dense-proxy shows 26% HBM utilization.")
    print("  If strided pattern hits ≤30% of sequential, the 26% figure is NOT")
    print("  kernel-specific — it's the sustained-BW ceiling at M=1 decode.")
    print("  This would INVALIDATE FP4/compression win projections based on BW headroom.")
    print("  PREDICTION: strided ≈20-35% of nameplate; sequential ≈70-85% of nameplate.")

    return {
        "seq_gb_s": seq_gb_s,
        "str_gb_s": str_gb_s,
        "seq_pct":  seq_pct,
        "str_pct":  str_pct,
    }


# ---------------------------------------------------------------------------
# Bench 7 — CUDA graph node-count stress test (host-side)
# ---------------------------------------------------------------------------

def bench_graph_nodes(num_sms: int) -> dict:
    """
    Constructs synthetic CUDA graphs of N nodes (simple memset chains),
    captures, and replays 3× each. Finds crash point.
    N sweep: 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
    """
    require_torch("bench7_graph_nodes")
    header("BENCH 7 — CUDA graph node-count stress test")

    if torch.cuda.get_device_capability()[0] < 8:
        print("  SKIP: requires CUDA arch >= sm_80 for graph support.")
        return {}

    node_counts = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    results = {}
    crash_at = None

    buf = torch.zeros(1024, dtype=torch.int32, device="cuda")

    print(f"  {'N nodes':>10}  {'capture':>10}  {'replay×3':>12}  {'status':>10}")
    print("  " + "-" * 46)

    for n in node_counts:
        ok = True
        cap_ms = replay_ms = 0.0
        try:
            # Capture graph of N sequential fill kernels
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.graph(g):
                for _ in range(n):
                    buf.fill_(0)
            torch.cuda.synchronize()
            cap_ms = (time.perf_counter() - t0) * 1000

            # Replay 3×
            t1 = time.perf_counter()
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            replay_ms = (time.perf_counter() - t1) * 1000 / 3

            status = "OK"
        except RuntimeError as e:
            ok = False
            crash_at = n
            status = f"CRASH: {str(e)[:40]}"
        except Exception as e:
            ok = False
            crash_at = n
            status = f"ERR: {str(e)[:40]}"

        results[n] = {"ok": ok, "cap_ms": cap_ms, "replay_ms": replay_ms}
        print(f"  {n:>10}  {cap_ms:>8.1f}ms  {replay_ms:>10.2f}ms  {status}")

        if not ok:
            print(f"\n  CRASH detected at N={n} nodes. Stopping sweep.")
            break

    print()
    if crash_at:
        print(f"  RESULT: Graph crash at N={crash_at} nodes.")
        print(f"  This confirms Discovery #56-58 'reduced bookkeeping budget' hypothesis.")
        print(f"  Mega-graph design must stay below {crash_at} nodes.")
        prev = node_counts[node_counts.index(crash_at) - 1] if node_counts.index(crash_at) > 0 else 0
        print(f"  Safe limit: ≤{prev} nodes (last successful count).")
    else:
        print(f"  No crash in 50-500 node range. Discovery #56-58 crash may be")
        print(f"  shape/content specific or require >500 nodes to reproduce.")
        print(f"  Consider extending sweep to 1000+ nodes if needed.")

    return {"crash_at": crash_at, "results": results}


# ---------------------------------------------------------------------------
# Bench 8 — TMA probe
# ---------------------------------------------------------------------------

def bench_tma_probe(clock_ghz: float) -> dict:
    """
    Attempts cp.async.bulk.tensor instruction on SM120a.
    Detects: (a) compile-time rejection, (b) runtime illegal instruction,
             (c) success.
    """
    require_torch("bench8_tma_probe")
    header("BENCH 8 — TMA cp.async.bulk.tensor availability on SM120a")

    if not BENCH_BIN.exists():
        print(f"ERROR: {BENCH_BIN} not found. Check build log for TMA compile errors.")
        print("  If build failed on bench8, re-run: python3 build_sm120_calibration.py")
        print("  If you see 'cp.async.bulk' in nvcc error, TMA is NOT accepted by ptxas")
        print("  for sm_120a — that itself is the ground-truth answer: TMA ABSENT.")
        return {"result": "build_failed", "tma_available": False}

    cmd = [str(BENCH_BIN), "--bench", "tma_probe", "--clock-ghz", str(clock_ghz)]
    r = subprocess.run(cmd, capture_output=True, text=True)

    tma_available = None
    detail = ""

    if r.returncode != 0:
        # Check if binary ran at all (vs compile-time skip)
        if "SKIP" in r.stdout or "skipped" in r.stdout.lower():
            tma_available = None
            detail = "SKIP_TMA was defined at compile time — rebuild without --skip-tma"
        else:
            tma_available = False
            detail = f"Runtime error (rc={r.returncode}): {r.stderr[:200]}"
    else:
        for line in r.stdout.splitlines():
            if line.startswith("RESULT"):
                parts = dict(p.split("=") for p in line.split()[1:] if "=" in p)
                code = int(parts.get("code", 1))
                if code == 0:
                    tma_available = True
                    detail = "Instruction executed without illegal-instruction fault"
                elif code == 1:
                    tma_available = False
                    detail = "cudaErrorIllegalInstruction — TMA not present on this SM"
                elif code == 2:
                    tma_available = False
                    detail = "Compile-time arch guard: __CUDA_ARCH__ < 900 (should not happen on sm_120a)"
                elif code == 3:
                    tma_available = None
                    detail = "Bench was compiled with SKIP_TMA — rebuild to test"
                else:
                    tma_available = False
                    detail = f"Unknown code {code}"
                break

    if tma_available is True:
        verdict = "TMA AVAILABLE on SM120a"
        implication = ("The __CUDA_ARCH__ >= 900 guard in upstream code correctly passes. "
                       "W5_5b audit 'guarded correctly' conclusion holds.")
    elif tma_available is False:
        verdict = "TMA ABSENT on SM120a"
        implication = ("NVIDIA consumer-Blackwell docs are correct: no TMA on SM120. "
                       "Any code path behind __CUDA_ARCH__ >= 900 that uses TMA will "
                       "fault at runtime on SM120a hardware.")
    else:
        verdict = "INCONCLUSIVE — rebuild without --skip-tma"
        implication = "Re-run after rebuild."

    row("  TMA verdict", verdict)
    row("  Detail",      detail)
    print()
    print(f"  IMPLICATION: {implication}")

    return {
        "tma_available": tma_available,
        "verdict":       verdict,
        "detail":        detail,
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict, clock_ghz: float) -> None:
    header("SUMMARY — Calibration constants for KILL_PATTERNS.md §1")
    print()
    print("| Constant | Measured | Source |")
    print("|---|---|---|")

    # Bench 1
    gs = all_results.get("grid_sync", {})
    if gs:
        for cond, val in gs.items():
            if val is not None:
                print(f"| `grid.sync` [{cond}] | **{val:.2f} µs/barrier** "
                      f"| SM120a calib bench W6 |")
    else:
        print("| `grid.sync` cost variants | NOT RUN | — |")

    # Bench 2
    wc = all_results.get("wmma_cpasync", {})
    if wc:
        crossover = next(
            ((b, k) for (b, k), r in sorted(wc.items())
             if r.get("cpasync_us") and r.get("wmma_us") and r["cpasync_us"] < r["wmma_us"]),
            None
        )
        if crossover:
            print(f"| `cp.async` crossover point | **B-tile≥{crossover[0]}, K-steps≥{crossover[1]}** "
                  f"| SM120a calib bench W6 |")
        else:
            print("| `cp.async` crossover point | **No crossover (wmma wins all shapes)** "
                  "| SM120a calib bench W6 |")
    else:
        print("| `cp.async` vs `wmma::load` crossover | NOT RUN | — |")

    # Bench 3
    mt = all_results.get("mma_throughput", {})
    for level in ["per_warp", "per_sm", "full_sm_occupancy"]:
        val = mt.get(level)
        if val is not None:
            print(f"| `mma.sync` BF16 [{level}] | **{val:.0f} TOPS** "
                  f"| SM120a calib bench W6 |")

    # Bench 4
    cv = all_results.get("cvt_fp4", {})
    if cv.get("cycles_per_pair") is not None:
        print(f"| `cvt.rn.satfinite.e2m1x2.f32` | **{cv['cycles_per_pair']:.1f} cycles/fp4x2** "
              f"| SM120a calib bench W6 |")

    # Bench 5
    sv = all_results.get("shfl_vs_atomic", {})
    if sv.get("winner"):
        print(f"| `__shfl_sync` vs `atomicAdd` smem | **{sv['winner']} wins by {sv['ratio']:.2f}×** "
              f"| SM120a calib bench W6 |")

    # Bench 6
    hb = all_results.get("hbm_bw", {})
    if hb.get("seq_gb_s"):
        print(f"| HBM ceiling BW (sequential) | **{hb['seq_gb_s']:.0f} GB/s ({hb['seq_pct']:.0f}%)** "
              f"| SM120a calib bench W6 |")
    if hb.get("str_gb_s"):
        print(f"| HBM BW M=1 decode pattern | **{hb['str_gb_s']:.0f} GB/s ({hb['str_pct']:.0f}%)** "
              f"| SM120a calib bench W6 |")

    # Bench 7
    gn = all_results.get("graph_nodes", {})
    crash_at = gn.get("crash_at")
    if crash_at is not None:
        print(f"| CUDA graph node-count crash | **crashes at N={crash_at}** "
              f"| SM120a calib bench W6 |")
    elif "graph_nodes" in all_results:
        print("| CUDA graph node-count crash | **no crash at 50-500 nodes** "
              "| SM120a calib bench W6 |")

    # Bench 8
    tma = all_results.get("tma_probe", {})
    if "tma_available" in tma:
        avail_str = "AVAILABLE" if tma["tma_available"] else ("ABSENT" if tma["tma_available"] is False else "INCONCLUSIVE")
        print(f"| TMA `cp.async.bulk.tensor` | **{avail_str}** | SM120a calib bench W6 |")

    print()
    print(f"  clock_ghz={clock_ghz} used for all µs/ns conversions")
    print("  Paste rows above into plans/KILL_PATTERNS.md §1 after verification.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BENCH_MAP = {
    "grid_sync":       bench_grid_sync,
    "wmma_cpasync":    bench_wmma_cpasync,
    "mma_throughput":  bench_mma_throughput,
    "cvt_fp4":         bench_cvt_fp4,
    "shfl_vs_atomic":  bench_shfl_vs_atomic,
    "hbm_bw":          bench_hbm_bw,
    "graph_nodes":     bench_graph_nodes,
    "tma_probe":       bench_tma_probe,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="SM120a calibration benchmark runner")
    parser.add_argument("--test", default="all",
                        choices=["all"] + list(BENCH_MAP.keys()),
                        help="Which bench to run (default: all)")
    parser.add_argument("--clock-ghz", type=float, default=2.61,
                        help="SM clock in GHz. Get via: nvidia-smi -q | grep 'Graphics Clock'")
    parser.add_argument("--num-sms", type=int, default=188,
                        help="Number of SMs (188 for PRO 6000)")
    parser.add_argument("--iters", type=int, default=None,
                        help="Override iteration count for all timing loops")
    parser.add_argument("--table", action="store_true",
                        help="Print markdown summary table at end")
    parser.add_argument("--out", type=str, default=None,
                        help="Write results markdown to FILE")
    args = parser.parse_args()

    print(f"SM120a Calibration Runner | clock={args.clock_ghz} GHz | SMs={args.num_sms}")
    print(f"Binary: {BENCH_BIN}")
    print(f"PyTorch: {'available' if HAS_TORCH else 'NOT FOUND (some benches will skip)'}")

    if not BENCH_BIN.exists() and args.test not in ("graph_nodes",):
        print(f"\nWARNING: {BENCH_BIN} not found.")
        print("Run first: python3 build_sm120_calibration.py")

    tests_to_run = list(BENCH_MAP.keys()) if args.test == "all" else [args.test]

    all_results = {}

    # Per-bench signatures differ; dispatch with right kwargs
    for test_name in tests_to_run:
        fn = BENCH_MAP[test_name]
        import inspect
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        kwargs = {}
        if "clock_ghz" in param_names:
            kwargs["clock_ghz"] = args.clock_ghz
        if "num_sms" in param_names:
            kwargs["num_sms"] = args.num_sms
        if "iters" in param_names and args.iters is not None:
            kwargs["iters"] = args.iters

        try:
            result = fn(**kwargs)
            all_results[test_name] = result or {}
        except SystemExit:
            raise
        except Exception as e:
            print(f"  ERROR in {test_name}: {e}")
            all_results[test_name] = {}

    if args.table or args.test == "all":
        print_summary_table(all_results, args.clock_ghz)

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w") as f:
            f.write("# SM120a Calibration Results\n\n")
            f.write("_Generated by run_sm120_calibration.py — parent fills in measured values_\n\n")
            f.write(f"clock_ghz={args.clock_ghz}, num_sms={args.num_sms}\n\n")
            # Re-print table to file
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_summary_table(all_results, args.clock_ghz)
            f.write(buf.getvalue())
        print(f"\nResults written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
