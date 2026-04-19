#!/usr/bin/env python3
"""
Build script for SM120a calibration microbenchmark suite.
Tag: W6_sm120_calibration_extended

Usage:
    python3 build_sm120_calibration.py [--skip-tma] [--verbose]

Produces: sm120_calib_bench  (executable in kernels/csrc/)

Requirements:
    nvcc in PATH, GPU architecture sm_120a supported (CUDA 12.8+)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC = SCRIPT_DIR / "sm120_calibration_bench.cu"
OUT = SCRIPT_DIR / "sm120_calib_bench"

NVCC_FLAGS = [
    "--gpu-architecture=sm_120a",
    "-O3",
    "-lineinfo",
    "-std=c++17",
    # Allow cooperative groups
    "--generate-code", "arch=compute_120,code=sm_120a",
    # Include paths
    "-I/usr/local/cuda/include",
    # Link
    "-lcuda", "-lcudart",
]

DEFINES_BASE = []


def run(cmd: list[str], verbose: bool = False) -> int:
    if verbose:
        print("CMD:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=not verbose)
    if not verbose and result.returncode != 0:
        print("STDOUT:", result.stdout.decode(errors="replace"))
        print("STDERR:", result.stderr.decode(errors="replace"))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SM120a calibration bench")
    parser.add_argument("--skip-tma", action="store_true",
                        help="Define SKIP_TMA — omit TMA probe (bench8) from build")
    parser.add_argument("--verbose", action="store_true",
                        help="Print full nvcc command and output")
    args = parser.parse_args()

    # Check nvcc exists
    nvcc = os.environ.get("NVCC", "nvcc")
    check = subprocess.run([nvcc, "--version"], capture_output=True)
    if check.returncode != 0:
        print("ERROR: nvcc not found. Install cuda-toolkit-12-8 or set NVCC=<path>.")
        return 1
    if args.verbose:
        print(check.stdout.decode(errors="replace").strip())

    defines = list(DEFINES_BASE)
    if args.skip_tma:
        defines.append("-DSKIP_TMA")
        print("INFO: SKIP_TMA defined — bench8 (TMA probe) will be stubbed out.")

    cmd = [nvcc] + NVCC_FLAGS + defines + [str(SRC), "-o", str(OUT)]

    print(f"Building {SRC.name} -> {OUT.name} ...")
    rc = run(cmd, verbose=args.verbose)
    if rc != 0:
        print("BUILD FAILED. If error mentions 'cp.async.bulk', re-run with --skip-tma")
        print("to isolate whether the TMA instruction causes the compile failure.")
        return rc

    print(f"BUILD OK: {OUT}")
    print()
    print("Next step:")
    print("  python3 run_sm120_calibration.py --test all")
    print("  python3 run_sm120_calibration.py --test grid_sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
