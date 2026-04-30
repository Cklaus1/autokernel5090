"""T2-N vLLM plugin entry point: install fused shuffle+quant monkey-patch.

This module is loaded by vLLM's plugin system via the entry_points mechanism:

    [vllm.general_plugins]
    fused_shuffle_quant = fused_shuffle_quant_plugin:register

When vLLM starts, ``load_general_plugins()`` discovers and calls
``register()`` before any model is loaded. ``register()`` installs the
monkey-patch for ``vllm.model_executor.layers.fused_moe.cutlass_moe.
run_cutlass_moe_fp4`` so every MoE forward pass uses the fused
shuffle_rows+scaled_fp4_experts_quant CUDA kernel from
``workspace/fused_shuffle_quant_sm120a.so``.

Disable at runtime with ``AUTOKERNEL_FUSED_SHUFFLE_QUANT=0``.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)


def register() -> None:
    """vLLM plugin entry point.

    Imports the wrapper module and applies ``patch_cutlass_moe_fp4``.
    Safe to call multiple times; the wrapper guards against double-patching.

    Logs a clear success/skip line so operators can confirm from the server
    log that the fused path is active.
    """
    # Make sure the wrapper module is importable. We expect the patches
    # directory to be on PYTHONPATH (either via bind-mount + PYTHONPATH env
    # inside a container, or via a local install on the host).
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        import fused_shuffle_quant_wrapper as wrap
    except Exception as e:
        logger.error(
            "[T2-N] plugin failed to import fused_shuffle_quant_wrapper: %s", e
        )
        return

    # W8 (tag W8_rank3_silent_fire_debug): eagerly probe the Rank 3 Triton
    # kernel so the [T2N-UNSHUF-WSUM] load/import-failure banner appears at
    # plugin-register time (startup) instead of first MoE forward. This makes
    # silent fall-throughs visible in `docker logs` before any traffic.
    try:
        if hasattr(wrap, "is_unshuffle_weightedsum_available"):
            _ok = wrap.is_unshuffle_weightedsum_available()
            print(
                f"[T2N-UNSHUF-WSUM] eager-probe at plugin register: "
                f"available={_ok}",
                flush=True,
            )
    except Exception as e:
        logger.warning(
            "[T2N-UNSHUF-WSUM] eager-probe raised class=%s msg=%s",
            type(e).__name__, e,
        )

    try:
        ok = wrap.patch_cutlass_moe_fp4()
        if ok:
            logger.info(
                "[T2-N] Patched run_cutlass_moe_fp4 via plugin: "
                "fused shuffle+quant active"
            )
            # Also print to stdout so it is visible in docker logs even when
            # the root logger is configured for WARNING.
            print(
                "[T2-N] Patched run_cutlass_moe_fp4 via plugin: "
                "fused shuffle+quant active",
                flush=True,
            )
        else:
            logger.warning(
                "[T2-N] patch_cutlass_moe_fp4 returned False -- fused kernel NOT active"
            )
            print(
                "[T2-N] patch_cutlass_moe_fp4 returned False -- fused kernel NOT active",
                flush=True,
            )
    except Exception as e:
        logger.error("[T2-N] plugin register() raised: %s", e, exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    register()
