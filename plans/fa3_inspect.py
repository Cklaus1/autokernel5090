#!/usr/bin/env python3
"""Inspect a compiled object to confirm arch + look for wgmma."""
import subprocess

OBJ = "/tmp/fa3_probe_120a.o"
CUOBJ = "/usr/local/cuda-12.8/bin/cuobjdump"

for args in [["--list-text"], ["--list-elf"], ["--list-ptx"], ["--dump-sass"]]:
    print(f"\n=== cuobjdump {' '.join(args)} ===")
    r = subprocess.run([CUOBJ] + args + [OBJ], capture_output=True, text=True)
    print("RC:", r.returncode)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[:4000])
    if 'wgmma' in out.lower():
        idx = out.lower().index('wgmma')
        print("FOUND wgmma at:", out[max(0,idx-100):idx+400])
