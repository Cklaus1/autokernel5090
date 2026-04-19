#!/usr/bin/env python3
"""Disassemble probed object files and look for WGMMA / HMMA / TCGEN / invalid-path markers."""
import subprocess, sys, os

CUOBJ = "/usr/local/cuda-12.8/bin/cuobjdump"

for obj in ["/tmp/fa3_probe_120a.o", "/tmp/fa3_probe_120a_patched.o"]:
    if not os.path.exists(obj):
        print(f"[skip] {obj} missing"); continue
    print(f"\n===== {obj} ({os.path.getsize(obj)} bytes) =====")
    r = subprocess.run([CUOBJ, "--dump-sass", obj], capture_output=True, text=True)
    sass = r.stdout or ""
    n_lines = sass.count("\n")
    print(f"SASS lines: {n_lines}")
    for token in ["WGMMA", "HMMA", "IMMA", "BMMA", "TCGEN", "TMEM", "QSPC", "WGMMA.MMA", "UTMALDG", "TMA"]:
        c = sass.upper().count(token)
        if c:
            print(f"  {token}: {c}")
    # Show first 20 non-NOP lines
    real = [l for l in sass.split("\n") if "NOP" not in l and "/*" in l]
    print("Sample real SASS instructions (first 15):")
    for l in real[:15]:
        print(l.strip()[:180])
