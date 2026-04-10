"""Launch vLLM with FusenKV plugin pre-registered.

Since FusenKV is not pip-installed (just mounted), the entry_points
mechanism doesn't discover it. This script manually registers the plugin
before vLLM parses CLI arguments, so --kv-cache-dtype k4v4b64 is accepted.

Usage:
    python3 /fusen/fusen_kv/launch_vllm.py [all vllm server args...]
"""
import sys
import os

# Ensure fusen_kv and kv_cache_gen are importable
fusen_path = os.environ.get("PYTHONPATH", "/fusen")
if fusen_path not in sys.path:
    sys.path.insert(0, fusen_path)

# Capture and remove our custom kv-cache-dtype from sys.argv BEFORE
# vLLM's argparser sees it (because the argparser's choices are frozen
# in the EngineArgs dataclass and can't be monkey-patched).
_fusen_kv_dtype = None
for i, arg in enumerate(sys.argv):
    if arg == "--kv-cache-dtype" and i + 1 < len(sys.argv):
        from fusen_kv.plugin import FUSEN_DTYPES
        if sys.argv[i + 1] in FUSEN_DTYPES:
            _fusen_kv_dtype = sys.argv[i + 1]
            # Replace with 'auto' so argparse doesn't reject it
            sys.argv[i + 1] = "auto"
            break

# Register FusenKV plugin BEFORE vLLM parses args
from fusen_kv.plugin import register
register()

# Now launch vLLM's API server with all remaining args
from vllm.entrypoints.openai.api_server import (
    cli_env_setup,
    FlexibleArgumentParser,
    make_arg_parser,
    run_server,
    validate_parsed_serve_args,
)
import uvloop

if __name__ == "__main__":
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server (FusenKV enabled)."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()

    # Restore the FusenKV dtype after argparse validation
    if _fusen_kv_dtype:
        args.kv_cache_dtype = _fusen_kv_dtype
        import logging
        logging.getLogger("fusen_kv").info(
            "FusenKV: restored --kv-cache-dtype to '%s' after argparse",
            _fusen_kv_dtype,
        )

    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
