#!/usr/bin/env bash
# Run the v5a test harness pinned to GPU 1.
export CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"/../..
exec python3 kernels/csrc/test_mega_graph_gemma4_30layer_v5a.py "$@"
