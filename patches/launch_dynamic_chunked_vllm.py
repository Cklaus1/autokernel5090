#!/usr/bin/env python3
"""Launch vLLM API server with dynamic per-request chunked-prefill sizing.

Applies patches/dynamic_chunked_prefill before invoking the stock vLLM API server.

Usage (Docker):
    docker run --gpus '"device=1"' -v /home/cklaus/projects/autokernel:/autokernel \
        -v /root/models:/models:ro -p 8005:8000 --name vllm-dynchunk \
        vllm-fusencache-gemma4fix:latest \
        python3 /autokernel/patches/launch_dynamic_chunked_vllm.py \
            --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
            --quantization modelopt \
            --max-model-len 4096 \
            --served-model-name gemma-4-26B-A4B-it-NVFP4 \
            --scheduling-policy priority \
            --enable-prefix-caching \
            --port 8000
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("autokernel.launch_dynchunk")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logger.info("Applying autokernel dynamic chunked-prefill patch...")
    from patches.dynamic_chunked_prefill import _apply_patch
    ok = _apply_patch()
    logger.info("Patch install: %s", "OK" if ok else "FAIL")

    # Follow vllm/entrypoints/openai/api_server.py __main__ path.
    import uvloop
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server "
                    "(autokernel dynamic chunked prefill)."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    logger.info("Starting vLLM API server")
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
