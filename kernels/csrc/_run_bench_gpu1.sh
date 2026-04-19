#!/usr/bin/env bash
# Run the cp.async microbench pinned to GPU 1.
export CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"/../..
exec python3 kernels/csrc/bench_fusencache_cpasync.py "$@"
