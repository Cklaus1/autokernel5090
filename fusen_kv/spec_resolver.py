"""Resolve vLLM kv_cache_dtype strings to KVCacheSpec objects.

Bridges vLLM's --kv-cache-dtype CLI arg to our spec system.
"""

import sys
import os

# Ensure kv_cache_gen is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv_cache_gen.config import parse_spec
from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS

# Default spec when user just says "fusen"
_DEFAULT_SPEC = "k4v4b64"


def resolve_spec(kv_cache_dtype: str) -> KVCacheSpec:
    """Resolve a vLLM kv_cache_dtype string to a KVCacheSpec.

    Args:
        kv_cache_dtype: String from --kv-cache-dtype. Can be:
            "fusen"         → default spec (k4v4b64)
            "k4v4"          → alias for k4v4b64
            "k8v4b32"       → specific spec
            "k4v4kb64vb64"  → fully specified
            "auto"          → adaptive selection (returns k4v4b64 for now)

    Returns:
        KVCacheSpec instance
    """
    if kv_cache_dtype in ("fusen", "auto"):
        return PREDEFINED_SPECS[_DEFAULT_SPEC]

    spec = parse_spec(kv_cache_dtype)
    if spec is None:
        # parse_spec returns None for "auto"/"bf16" — use default
        return PREDEFINED_SPECS[_DEFAULT_SPEC]

    return spec
