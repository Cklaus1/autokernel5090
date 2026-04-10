"""Launch vLLM with FusenKV plugin pre-registered.

Since FusenKV is not pip-installed (just mounted), the entry_points
mechanism doesn't discover it. This script manually registers the plugin
before vLLM parses CLI arguments, so --kv-cache-dtype k4v4b64 is accepted.

CRITICAL: vLLM spawns EngineCore in a separate process (especially on WSL
where fork is not used). We must ensure our plugin is registered in ALL
processes, not just the main one. We achieve this by:
1. Installing a .pth file in site-packages that auto-registers our plugin
2. Setting PYTHONPATH so the .pth import works in subprocesses

Usage:
    python3 /fusen/fusen_kv/launch_vllm.py [all vllm server args...]
"""
import sys
import os
import site
import logging

logger = logging.getLogger("fusen_kv.launch")

# Ensure fusen_kv and kv_cache_gen are importable
fusen_path = os.environ.get("FUSEN_PATH", "/fusen")
if fusen_path not in sys.path:
    sys.path.insert(0, fusen_path)

# Also ensure PYTHONPATH is set for subprocesses (spawn method)
existing_pypath = os.environ.get("PYTHONPATH", "")
if fusen_path not in existing_pypath:
    os.environ["PYTHONPATH"] = fusen_path + (":" + existing_pypath if existing_pypath else "")


def _install_auto_register():
    """Ensure FusenKV plugin is registered in ALL vLLM processes.

    When vLLM uses 'spawn' multiprocessing (e.g., on WSL), child processes
    don't inherit parent monkey-patches. We solve this two ways:

    1. Install a fake Python package entry_point so vLLM's load_general_plugins()
       discovers and calls our register() in every process.

    2. As fallback, create a .pth file in site-packages that auto-imports
       our registration module at Python startup.
    """
    # Method 1: Register a fake entry_point for vLLM's plugin discovery.
    # This uses importlib.metadata's distributions API to inject our plugin.
    try:
        _install_entry_point()
        logger.info("FusenKV: installed entry_point for subprocess discovery")
        return True
    except Exception as e:
        logger.warning("FusenKV: entry_point install failed: %s, trying .pth", e)

    # Method 2: .pth file approach (fallback)
    try:
        return _install_pth_file()
    except Exception as e:
        logger.warning("FusenKV: .pth install failed: %s", e)
        return False


def _install_entry_point():
    """Install a fake dist-info so importlib.metadata.entry_points() finds us.

    Creates a minimal .dist-info directory in site-packages with an
    entry_points.txt that declares our plugin for vLLM's plugin system.
    """
    # Find writable site-packages
    site_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for sp_dir in site_dirs:
        if os.path.isdir(sp_dir) and os.access(sp_dir, os.W_OK):
            dist_info = os.path.join(sp_dir, "fusen_kv-0.1.0.dist-info")
            os.makedirs(dist_info, exist_ok=True)

            # METADATA (required)
            with open(os.path.join(dist_info, "METADATA"), "w") as f:
                f.write(
                    "Metadata-Version: 2.1\n"
                    "Name: fusen-kv\n"
                    "Version: 0.1.0\n"
                    "Summary: FusenKV attention backend for vLLM\n"
                )

            # entry_points.txt (the key file for plugin discovery)
            with open(os.path.join(dist_info, "entry_points.txt"), "w") as f:
                f.write(
                    "[vllm.general_plugins]\n"
                    "fusen_kv = fusen_kv.plugin:register\n"
                )

            # RECORD (empty, but needed for valid dist-info)
            with open(os.path.join(dist_info, "RECORD"), "w") as f:
                f.write("")

            # Invalidate importlib.metadata caches so our new dist-info is found
            import importlib
            importlib.invalidate_caches()

            # Verify it's discoverable
            from importlib.metadata import entry_points
            eps = entry_points(group="vllm.general_plugins")
            found = any(ep.name == "fusen_kv" for ep in eps)
            if found:
                logger.info("FusenKV: entry_point verified in %s", dist_info)
            else:
                logger.warning("FusenKV: entry_point written but not found by importlib")

            return


def _install_pth_file():
    """Fallback: install a .pth file for auto-registration."""
    autoload_dir = "/tmp/fusen_kv_autoload"
    os.makedirs(autoload_dir, exist_ok=True)

    with open(os.path.join(autoload_dir, "__init__.py"), "w") as f:
        f.write(
            "import os, sys\n"
            "fp = os.environ.get('FUSEN_PATH', '/fusen')\n"
            "if fp not in sys.path: sys.path.insert(0, fp)\n"
            "try:\n"
            "    from fusen_kv.plugin import register; register()\n"
            "except Exception: pass\n"
        )

    site_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for sp_dir in site_dirs:
        if os.path.isdir(sp_dir) and os.access(sp_dir, os.W_OK):
            pth_path = os.path.join(sp_dir, "fusen_kv_autoload.pth")
            with open(pth_path, "w") as f:
                f.write(f"{autoload_dir}\n")
                f.write("import fusen_kv_autoload\n")
            logger.info("FusenKV: installed .pth at %s", pth_path)
            return True

    return False


# Install auto-registration for subprocesses
_install_auto_register()


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
        logger.info(
            "FusenKV: restored --kv-cache-dtype to '%s' after argparse",
            _fusen_kv_dtype,
        )

    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
