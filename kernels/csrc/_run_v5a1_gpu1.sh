#!/usr/bin/env bash
# Build + run v5a.1 pinned to GPU 1.
set -e
export CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")"/../..

echo "== BUILD =="
.venv/bin/python kernels/csrc/build_mega_graph_gemma4_30layer_v5a1.py

echo
echo "== TEST =="
exec .venv/bin/python kernels/csrc/test_mega_graph_gemma4_30layer_v5a1.py "$@"
