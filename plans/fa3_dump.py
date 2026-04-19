#!/usr/bin/env python3
"""Grep SASS for specific tokens."""
import subprocess

r = subprocess.run(["/usr/local/cuda-12.8/bin/cuobjdump", "--dump-sass", "/tmp/fa3_probe_120a_patched.o"], capture_output=True, text=True)
sass = r.stdout

# count many opcodes
for tok in ["WGMMA","HMMA","IMMA","BMMA","MMA","TCGEN","TMEM","UTMA","STS","LDSM","STSM","UBARRIER","BAR","FENCE","SETMAXNREG","WARPSPEC"]:
    c = sass.upper().count(tok)
    print(f"{tok}: {c}")

# Dump lines containing MMA
print("\n--- MMA-ish lines (first 10) ---")
for line in sass.splitlines():
    if 'MMA' in line.upper() and 'XMMA' not in line.upper():
        print(line.strip()[:200])

# Dump tail of SASS
print("\n--- Tail SASS (last 20 opcodes) ---")
lines = [l for l in sass.splitlines() if '/*' in l and '/*0x' not in l[:10]]
for l in lines[-20:]:
    print(l.strip()[:200])
