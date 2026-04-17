#!/bin/bash
# Production vLLM serving for Qwen3.5 NVFP4 on RTX 5090
# 74+ experiments optimized

MODE=${1:-production}
MODEL=${2:-9b}       # 9b or 35b
THINK=${3:-nothink}  # nothink or think

# Common flags
TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser qwen3_xml"
if [ "$THINK" = "think" ]; then
    THINK_FLAGS=""
else
    THINK_FLAGS="--default-chat-template-kwargs {\"enable_thinking\":false}"
fi
COMMON="--dtype bfloat16 --trust-remote-code --language-model-only --host 0.0.0.0 --port 8000 --enforce-eager $TOOL_FLAGS $THINK_FLAGS"

# Model selection
if [ "$MODEL" = "35b" ]; then
    MODEL_PATH="Sehyo/Qwen3.5-35B-A3B-NVFP4"
    GPU_UTIL=0.93   # 25GB model, tight on 32GB
    CTX=65536       # 64K context — 1.2GB KV cache fits in ~5.4GB headroom
else
    MODEL_PATH="Kbenkhaled/Qwen3.5-9B-NVFP4"
    GPU_UTIL=0.90
    CTX=65536
fi

case $MODE in
    production)
        echo "=== PRODUCTION: $MODEL_PATH, ctx=$CTX, $THINK, tool calling ==="
        python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --gpu-memory-utilization $GPU_UTIL \
            --max-model-len $CTX \
            --kv-cache-dtype fp8_e5m2 \
            --mamba-ssm-cache-dtype float16 \
            $COMMON
        ;;
    interactive)
        echo "=== INTERACTIVE: $MODEL_PATH, DFlash, $THINK, tool calling ==="
        DRAFT_MODEL="z-lab/Qwen3.5-9B-DFlash"
        if [ "$MODEL" = "35b" ]; then
            DRAFT_MODEL="z-lab/Qwen3.5-35B-A3B-DFlash"
        fi
        python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --gpu-memory-utilization $GPU_UTIL \
            --max-model-len $CTX \
            --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT_MODEL\",\"num_speculative_tokens\":6}" \
            $COMMON
        ;;
    sustained)
        echo "=== SUSTAINED: $MODEL_PATH, $THINK, no cliff, tool calling ==="
        python3 -m vllm.entrypoints.openai.api_server \
            --model $MODEL_PATH \
            --gpu-memory-utilization $GPU_UTIL \
            --max-model-len $CTX \
            --kv-cache-dtype fp8_e5m2 \
            --mamba-ssm-cache-dtype float16 \
            --mamba-cache-mode align \
            $COMMON
        ;;
    *)
        echo "Usage: $0 {production|interactive|sustained} {9b|35b} {nothink|think}"
        ;;
esac
