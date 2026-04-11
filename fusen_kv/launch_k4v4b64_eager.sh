#!/bin/bash
set -e

# FusenCache K4V4B64 launch script — EAGER MODE with async scheduling.
#
# This configuration uses enforce-eager (no CUDA graphs) with async scheduling
# enabled. The FusenCache backend's metadata cloning + stream sync in
# backend.py forward() protects against async scheduling races.
#
# Tradeoffs vs CUDA graph mode (launch_k4v4b64.sh):
#   Pro:  Async scheduling enabled (better CPU-GPU overlap for high throughput)
#   Pro:  No CUDA graph capture overhead at startup
#   Con:  ~3-4x higher per-step latency (no graph replay optimization)
#   Con:  Higher GPU kernel launch overhead
#
# Use this for throughput-optimized serving. Use launch_k4v4b64.sh for
# latency-optimized serving.

# Install entry_points for plugin discovery in subprocess
mkdir -p /usr/local/lib/python3.12/dist-packages/fusen_kv_plugin-0.1.dist-info
cat > /usr/local/lib/python3.12/dist-packages/fusen_kv_plugin-0.1.dist-info/METADATA << 'METAEOF'
Metadata-Version: 2.1
Name: fusen-kv-plugin
Version: 0.1
METAEOF

cat > /usr/local/lib/python3.12/dist-packages/fusen_kv_plugin-0.1.dist-info/entry_points.txt << 'EPEOF'
[vllm.general_plugins]
fusen_kv = fusen_kv.plugin:register
EPEOF

cat > /usr/local/lib/python3.12/dist-packages/fusen_kv_plugin-0.1.dist-info/RECORD << 'RECEOF'
fusen_kv_plugin-0.1.dist-info/METADATA,,
fusen_kv_plugin-0.1.dist-info/entry_points.txt,,
fusen_kv_plugin-0.1.dist-info/RECORD,,
RECEOF

export PYTHONPATH=/fusen:${PYTHONPATH:-}

exec python3 /fusen/fusen_kv/launch_vllm.py \
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
  --quantization modelopt \
  --max-model-len 4096 \
  --trust-remote-code \
  --port 8001 \
  --kv-cache-dtype k4v4b64 \
  --enforce-eager \
  --gpu-memory-utilization 0.92 \
  -cc.mode none
