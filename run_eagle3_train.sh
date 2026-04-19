#!/usr/bin/env bash
# Launch EAGLE3 draft training for Qwen3-30B-A3B on GPU 1.
set -u
LOG=/tmp/eagle3_train.log
: > "$LOG"
echo "[$(date +%H:%M:%S)] launching container" | tee -a "$LOG"

# Phase A + B chained. The script internally logs to $LOG (bind-mounted).
# `--max-seconds 3300` caps phase B at 55 min to stay inside the 2 h envelope.

docker run --rm --gpus '"device=1"' \
    -v /root/models:/models \
    -v /home/cklaus/projects/autokernel:/work \
    -v /tmp:/tmp \
    -w /work \
    --name eagle3_train \
    --entrypoint python3 \
    vllm-fusencache-gemma4fix:latest \
    /work/train_eagle3_qwen3.py \
        --phase both \
        --n-examples 600 \
        --batch-size 4 \
        --lr 3e-4 \
        --max-steps 5000 \
        --max-seconds 3300 \
        --log-every 25 \
        --ckpt-every-s 900 \
    >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] container exited $?" | tee -a "$LOG"
