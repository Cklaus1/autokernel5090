#!/bin/bash
# Run CUDA_LAUNCH_BLOCKING crash pinpointing test inside vllm-built container.
# Usage: bash /root/projects/autokernel/profiling/clb_run.sh
set -euo pipefail

docker run --rm --gpus all \
  -e CUDA_LAUNCH_BLOCKING=1 \
  -e FUSEN_SYNC=1 \
  -e PYTHONPATH=/fusen \
  -v /root/models:/models:ro \
  -v /root/projects/autokernel:/fusen:ro \
  vllm-built bash -c '
    set -euo pipefail
    # Install dist-info so the spawned EngineCore subprocess can discover
    # the FusenKV plugin via importlib.metadata entry_points.
    DIST_INFO=/usr/local/lib/python3.12/dist-packages/fusen_kv_plugin-0.1.dist-info
    mkdir -p "$DIST_INFO"
    printf "Metadata-Version: 2.1\nName: fusen-kv-plugin\nVersion: 0.1\n" \
        > "$DIST_INFO/METADATA"
    printf "[vllm.general_plugins]\nfusen_kv = fusen_kv.plugin:register\n" \
        > "$DIST_INFO/entry_points.txt"
    printf "fusen_kv_plugin-0.1.dist-info/METADATA,,\nfusen_kv_plugin-0.1.dist-info/entry_points.txt,,\nfusen_kv_plugin-0.1.dist-info/RECORD,,\n" \
        > "$DIST_INFO/RECORD"

    export PYTHONPATH=/fusen:${PYTHONPATH:-}
    export CUDA_LAUNCH_BLOCKING=1
    export FUSEN_SYNC=1

    python3 /fusen/profiling/clb_test.py
  ' 2>&1
