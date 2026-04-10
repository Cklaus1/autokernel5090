"""FusenKV: Data-driven KV cache compression plugin for vLLM.

Usage:
    # Install
    pip install fusen-kv

    # vLLM auto-discovers via entry_points
    vllm serve model --kv-cache-dtype fusen

    # Or explicit
    VLLM_PLUGINS=fusen_kv vllm serve model --kv-cache-dtype k4v4

    # Python API
    from fusen_kv import register
    register()  # registers with vLLM's attention backend registry
"""

__version__ = "0.1.0"

from fusen_kv.plugin import register  # noqa: F401
