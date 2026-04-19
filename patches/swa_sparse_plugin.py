"""Wrapper around the Triton sparse sliding-window attention decode kernel.

Exposes ``swa_sparse_decode(query, kv_cache, block_tables, seq_lens,
window_size, ...) -> output`` with a vLLM-friendly signature.

Auto-selects between the Triton sparse path and a caller-provided dense fallback
based on the guard ``max_seq_len > window * 1.25``. Below that threshold the
FlashInfer tuned CUTLASS path beats Triton by ~8x (kernel bench), so we punt
back to the caller's dense path by returning ``None`` and letting the caller
run its own decode wrapper.

First-call diagnostic: logs
    [SWA] sparse path active for layer_id=... at max_seq_len=X, window=Y
exactly once per layer, so operators can confirm the kernel is engaged.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Make the Triton kernel importable whether the plugin is installed as a
# package or is just on PYTHONPATH next to the patches dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_TRITON_DIR = os.path.join(_REPO, "kernels", "triton")
if _TRITON_DIR not in sys.path:
    sys.path.insert(0, _TRITON_DIR)

_swa_decode_attention = None


def _lazy_import_kernel():
    """Import the Triton kernel lazily so module import never fails on a
    non-GPU / no-Triton box."""
    global _swa_decode_attention
    if _swa_decode_attention is None:
        from swa_decode import swa_decode_attention as _fn  # type: ignore

        _swa_decode_attention = _fn
    return _swa_decode_attention


# Guard threshold: kernel only wins when seq_len meaningfully exceeds window.
_GUARD_RATIO = float(os.environ.get("AUTOKERNEL_SWA_GUARD_RATIO", "1.25"))

# Emit the "[SWA] sparse path active" banner at most once per layer.
_SEEN_LAYERS: set[str] = set()


def sparse_path_eligible(max_seq_len: int, window: int) -> bool:
    """Return True iff the sparse kernel is expected to beat FlashInfer
    dense+mask. Based on the microbench (see
    plans/swa_sparse_attention_gemma4.md): at ``seq_len > window * 1.25``
    the sparse kernel wins; at ``seq_len <= window`` FlashInfer is ~8x
    faster so we must fall back."""
    if window is None or window <= 0:
        return False
    return max_seq_len > int(window * _GUARD_RATIO)


# Per-layer cache of FP8 scale tensors. Key: layer_id. Value: (k_scale_t, v_scale_t).
# Scales are constants at model-load; we materialize the 1-element FP32 tensors once
# to avoid a torch.tensor(...) host allocation per decode step.
_FP8_SCALE_CACHE: dict = {}


def swa_sparse_decode(
    query: torch.Tensor,           # [batch, num_q_heads, head_dim] BF16/FP16
    k_cache: torch.Tensor,         # [num_blocks, page_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,         # [num_blocks, page_size, num_kv_heads, head_dim]
    block_tables: torch.Tensor,    # [batch, max_num_blocks] int32
    seq_lens: torch.Tensor,        # [batch] int32
    window_size: int,
    sm_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    layer_id: str = "",
    k_scale: Optional[float] = None,  # Python float; required for FP8 KV
    v_scale: Optional[float] = None,  # Python float; required for FP8 KV
) -> Optional[torch.Tensor]:
    """Decode-step sparse sliding-window attention.

    Returns the attention output ``[batch, num_q_heads, head_dim]``.
    Returns ``None`` when the guard is not satisfied; caller should fall back
    to FlashInfer's dense+mask path.

    FP8 KV is auto-detected from ``k_cache.dtype``. When FP8, both ``k_scale``
    and ``v_scale`` (Python floats) must be provided (they come from
    ``layer._k_scale_float`` / ``layer._v_scale_float`` in vLLM).
    """
    # Guard on max seq_len in the batch (worst-case per-sequence decision is
    # fine because the kernel handles the seq_len<window case correctly; we
    # only gate to avoid the FlashInfer win region). Skip during graph capture
    # (host sync not allowed); the captured graph always routes to the kernel.
    if not torch.cuda.is_current_stream_capturing():
        max_seq = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
        if not sparse_path_eligible(max_seq, window_size):
            return None
    else:
        max_seq = -1  # unknown during capture

    fn = _lazy_import_kernel()

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(query.shape[-1])

    # Detect FP8 KV and lazily build cached scale tensors.
    is_fp8 = k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    k_scale_t = None
    v_scale_t = None
    if is_fp8:
        if k_scale is None or v_scale is None:
            # Caller didn't hand us scales for an FP8 KV cache -- we cannot run
            # the kernel correctly. Return None so caller falls back.
            return None
        cache_key = f"{layer_id}:{query.device}"
        cached = _FP8_SCALE_CACHE.get(cache_key)
        if cached is None or cached[2] != (float(k_scale), float(v_scale)):
            k_scale_t = torch.tensor([float(k_scale)], dtype=torch.float32,
                                     device=query.device)
            v_scale_t = torch.tensor([float(v_scale)], dtype=torch.float32,
                                     device=query.device)
            _FP8_SCALE_CACHE[cache_key] = (k_scale_t, v_scale_t,
                                           (float(k_scale), float(v_scale)))
        else:
            k_scale_t, v_scale_t, _ = cached

    # One-shot diagnostic banner per layer.
    if layer_id and layer_id not in _SEEN_LAYERS:
        _SEEN_LAYERS.add(layer_id)
        tag = "FP8" if is_fp8 else "BF16"
        msg = (
            f"[SWA] sparse path active ({tag} KV) for {layer_id} "
            f"(layers tagged={len(_SEEN_LAYERS)}) "
            f"at max_seq_len={max_seq}, window={window_size}"
        )
        logger.info(msg)
        # Also to stdout so docker logs show it even under WARNING root.
        print(msg, flush=True)

    return fn(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        window=window_size,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        k_scale=k_scale_t,
        v_scale=v_scale_t,
    )
