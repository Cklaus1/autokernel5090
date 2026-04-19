#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
exec /tmp/idlm_venv/bin/python /home/cklaus/projects/aigpu/I-DLM/inference/sglang/sglang/launch_server.py \
  --model-path /root/models/I-DLM-Qwen3-8B \
  --dtype bfloat16 \
  --dllm-algorithm idlm_blockN \
  --dllm-algorithm-config /home/cklaus/projects/aigpu/I-DLM/inference/configs/idlm_blockN4_config.yaml \
  --port 30001 \
  --trust-remote-code \
  --disable-cuda-graph \
  --attention-backend flashinfer \
  --mem-fraction-static 0.85
