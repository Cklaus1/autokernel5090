#!/usr/bin/env python3
"""
Launch vLLM with autokernel's FP8 decode attention kernel.

This script applies the monkey-patch to FlashInferImpl.forward() before
starting vLLM's API server. The patch replaces the FP8 decode path with
our Triton split-K paged decode kernel.

Usage (local):
    python3 patches/launch_fp8_decode_vllm.py \
        --model google/gemma-4-27b-it \
        --quantization modelopt \
        --kv-cache-dtype fp8 \
        --max-model-len 8192 \
        [... other vLLM args ...]

Usage (Docker):
    docker run --gpus all -v /path/to/autokernel:/autokernel \
        -e AUTOKERNEL_FP8_DECODE=1 \
        vllm/vllm-openai:latest \
        python3 /autokernel/patches/launch_fp8_decode_vllm.py \
        --model google/gemma-4-27b-it \
        --kv-cache-dtype fp8

Environment variables:
    AUTOKERNEL_FP8_DECODE=0   Disable the patch (pass-through to FlashInfer)
    AUTOKERNEL_FP8_DECODE=1   Enable the patch (default)

Kill switch: set AUTOKERNEL_FP8_DECODE=0 to disable at any time.
The patch only activates for FP8 KV cache decode; BF16 KV and all
prefill paths are untouched.
"""

import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("autokernel.launch")


def main():
    # Step 1: Ensure autokernel is on the path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Step 2: Apply the monkey-patch
    kill_switch = os.environ.get("AUTOKERNEL_FP8_DECODE", "1")
    if kill_switch == "0":
        logger.info("AUTOKERNEL_FP8_DECODE=0 — running stock vLLM (no FP8 decode patch)")
    else:
        logger.info("Applying autokernel FP8 decode attention patch...")
        from patches.fp8_decode_monkey_patch import apply_patch
        success = apply_patch()
        if success:
            logger.info("Patch applied successfully")
        else:
            logger.warning("Patch failed to apply — running stock vLLM")

    # Step 3: Delegate to vLLM's entrypoint
    # Remove our script name from sys.argv so vLLM sees only its own args
    logger.info("Starting vLLM API server with args: %s", sys.argv[1:])

    # Use vLLM's standard entrypoint
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    parser = make_arg_parser()
    args = parser.parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
