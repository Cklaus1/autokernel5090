# Auto-injected by AUTOKERNEL dynamic_chunked_prefill plugin.
# This file is placed on PYTHONPATH and imported automatically by Python's
# site.py at interpreter startup in *every* subprocess (APIServer, EngineCore,
# worker). It applies the dynamic chunked-prefill monkey-patch to vLLM v1
# Scheduler before vLLM's own imports land.
import os
import sys

if os.environ.get("AUTOKERNEL_DYNAMIC_CHUNK", "1") != "0":
    _root = "/autokernel"  # mounted inside the container
    if _root not in sys.path:
        sys.path.insert(0, _root)
    try:
        # Delayed import: vLLM may not be fully loaded yet; the patch lazily
        # imports vllm.v1.core.sched.scheduler. If that import fails here
        # (e.g. in non-vLLM subprocesses), fall through silently.
        from patches.dynamic_chunked_prefill import _apply_patch
        _apply_patch()
    except ImportError:
        pass
    except Exception as e:
        # Don't let plugin errors kill Python startup.
        sys.stderr.write(f"[sitecustomize] dynamic_chunked_prefill failed: {e}\n")
