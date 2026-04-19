#!/usr/bin/env python3
"""Consolidate all probe outputs into /tmp/fa3_build_attempt.log."""
import subprocess, shutil, os

LOG = "/tmp/fa3_build_attempt.log"
SRC = "/root/.claude/projects/-home-cklaus-projects/e4c28e4a-3ab7-482d-b3d6-8cba0c2c5bdd/tool-results/b6q2d8zbz.txt"

if os.path.exists(SRC):
    shutil.copy(SRC, LOG)
with open(LOG, "a") as f:
    f.write("\n\n===== SECOND ATTEMPT: patched enable_sm90 (__CUDA_ARCH__ == 900 || == 1200) =====\n")
    for obj in ["/tmp/fa3_probe_120a.o", "/tmp/fa3_probe_120a_patched.o"]:
        if os.path.exists(obj):
            f.write(f"{obj}: {os.path.getsize(obj)} bytes\n")
    f.write("\n--- disasm ---\n")
    r = subprocess.run(["python3", "/home/cklaus/projects/autokernel/plans/fa3_disasm.py"], capture_output=True, text=True)
    f.write(r.stdout)
    f.write(r.stderr)
    f.write("\n--- opcode counts (patched) ---\n")
    r = subprocess.run(["python3", "/home/cklaus/projects/autokernel/plans/fa3_dump.py"], capture_output=True, text=True)
    f.write(r.stdout)
    f.write(r.stderr)
    f.write("\n===== ANALYSIS =====\n")
    f.write("Unpatched:   sm_120a cubin built but kernel body stripped (140KB, ~86 SASS lines, just LDC/EXIT stub)\n")
    f.write("             Reason: hopper/utils.h::enable_sm90 guards Kernel::operator() with `#if __CUDA_ARCH__ == 900`\n")
    f.write("Patched:     Added `|| __CUDA_ARCH__ == 1200` to enable_sm90; cubin grew to 452KB/12k SASS lines\n")
    f.write("             BUT: 0 HMMA/WGMMA/IMMA/TCGEN/MMA ops in emitted SASS — all MMA inline-asm paths silently\n")
    f.write("             reduced to `assert(0)+__brkpt()` (CUTE_INVALID_CONTROL_PATH) because\n")
    f.write("             CUTLASS_ARCH_MMA_SM90A_ENABLED is gated on __CUDA_ARCH__ == 900 in cutlass/arch/config.h:48.\n")
    f.write("             With -DNDEBUG the assert is a no-op; __brkpt ends up as dead code the optimizer dropped.\n")
    f.write("             Kernel would launch but produce no output (or trap). UTMA/TMA/STSM ops emitted OK.\n")

print(f"Log written: {LOG}")
print(f"Size: {os.path.getsize(LOG)} bytes")
