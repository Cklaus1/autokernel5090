#!/usr/bin/env python3
"""Launch the I-DLM v1 SGLang server, capturing output to a file.

Usage:
    run_idlm_v1.py nograph  # --disable-cuda-graph on port 30001
    run_idlm_v1.py graph    # with CUDA graphs on port 30001
"""
import os
import sys
import subprocess

mode = sys.argv[1] if len(sys.argv) > 1 else "nograph"

args = [
    "/tmp/idlm_venv/bin/python",
    "/home/cklaus/projects/aigpu/I-DLM/inference/sglang/sglang/launch_server.py",
    "--model-path", "/root/models/I-DLM-Qwen3-8B",
    "--dtype", "bfloat16",
    "--dllm-algorithm", "IDLMBlockN",
    "--dllm-algorithm-config", "/home/cklaus/projects/aigpu/I-DLM/inference/configs/idlm_blockN4_config.yaml",
    "--port", "30001",
    "--trust-remote-code",
    "--attention-backend", "flashinfer",
    "--mem-fraction-static", "0.85",
]

if mode == "nograph":
    args.append("--disable-cuda-graph")
    logpath = "/tmp/idlm_logs/v1_nograph.log"
elif mode == "graph":
    # Limit graph BS so flashinfer temp allocator has enough room
    args += ["--cuda-graph-max-bs", "16"]
    logpath = "/tmp/idlm_logs/v1_graph.log"
else:
    raise SystemExit(f"unknown mode: {mode}")

os.makedirs("/tmp/idlm_logs", exist_ok=True)
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"

with open(logpath, "w") as log:
    print(f"Launching mode={mode} pid TBD -> {logpath}", flush=True)
    p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"Launched pid={p.pid}", flush=True)
    p.wait()
    print(f"Exited rc={p.returncode}", flush=True)
