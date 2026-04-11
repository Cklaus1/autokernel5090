#!/bin/bash
set -e

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

# NOTE: --no-async-scheduling is required with CUDA graphs because vLLM's
# async scheduler modifies shared tensors (input_ids, positions, etc.) while
# the GPU replays CUDA graphs that read them. FusenCache's metadata cloning
# (block_table, seq_lens, slot_mapping) protects OUR tensors, but other
# model inputs are outside our control and race with graph replay.
#
# For eager mode (--enforce-eager): async scheduling IS safe with FusenCache
# thanks to metadata cloning + stream sync in backend.py forward().
exec python3 /fusen/fusen_kv/launch_vllm.py \
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
  --quantization modelopt \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --trust-remote-code \
  --port 8001 \
  --kv-cache-dtype k4v4b64 \
  --no-async-scheduling \
  -cc.mode none \
  -cc.cudagraph_mode full \
  -cc.cudagraph_capture_sizes '[1,2,4,8,16,24,32,48,64]' \
  -cc.max_cudagraph_capture_size 64
