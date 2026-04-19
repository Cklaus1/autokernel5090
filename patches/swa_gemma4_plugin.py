"""vLLM plugin: route Gemma4 sliding-window decode to the Triton SWA kernel.

Loaded by vLLM's ``load_general_plugins()`` via the entry_points mechanism:

    [vllm.general_plugins]
    swa_gemma4 = swa_gemma4_plugin:register

``register()`` monkey-patches
``vllm.v1.attention.backends.flashinfer.FlashInferImpl.forward`` so that, on
decode batches whose longest sequence exceeds ``window * 1.25``, sliding
layers (``self.window_left >= 0``) route to the Triton sparse kernel at
``kernels/triton/swa_decode.py``. Global layers and prefill stay on
FlashInfer unchanged.

It also patches ``FlashInferMetadataBuilder.build`` to stash the per-batch
``block_table_tensor`` and ``seq_lens`` onto the resulting metadata object
(FlashInferMetadata is a plain ``@dataclass`` without ``__slots__``, so
attribute addition is safe).

Disable at runtime with ``AUTOKERNEL_SWA_SPARSE=0``.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)

# [FIXED-SILENT-NONE] Fix 2: SWA metadata stash DEBUG-level elevated to WARNING.
# Ref: plans/silent_none_sweep_20260419.md §C1, plans/KILL_PATTERNS.md §P1
logger.info("[FIXED-SILENT-NONE] swa_gemma4_plugin: metadata-stash failure and "
            "block_table/seq_lens=None fallthrough now emit WARNING")

_PATCH_APPLIED = False


def _banner(msg: str) -> None:
    logger.info(msg)
    # Ensure docker logs show it regardless of root log level.
    print(msg, flush=True)


def register() -> None:
    """vLLM plugin entry point."""
    global _PATCH_APPLIED

    if os.environ.get("AUTOKERNEL_SWA_SPARSE", "1") == "0":
        _banner("[SWA] plugin disabled via AUTOKERNEL_SWA_SPARSE=0")
        return

    # Make the wrapper importable.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    try:
        import swa_sparse_plugin as swa  # noqa: F401  (used via attribute access)
    except Exception as e:
        logger.error("[SWA] failed to import swa_sparse_plugin: %s", e, exc_info=True)
        return

    try:
        from vllm.v1.attention.backends import flashinfer as fi_backend
    except Exception as e:
        logger.error("[SWA] vLLM flashinfer backend not importable: %s", e)
        return

    if _PATCH_APPLIED:
        _banner("[SWA] monkey-patch already applied")
        return

    _patch_metadata_builder(fi_backend)
    _patch_impl_forward(fi_backend, swa)

    _PATCH_APPLIED = True
    _banner(
        "[SWA] Installed FlashInferImpl.forward + FlashInferMetadataBuilder.build "
        "monkey-patches (AUTOKERNEL_SWA_SPARSE=1); sliding-window decode will "
        "route to Triton SWA kernel when max_seq_len > window*1.25"
    )


def _patch_metadata_builder(fi_backend) -> None:
    """Patch ``FlashInferMetadataBuilder.build`` to stash block_table_tensor +
    seq_lens onto the returned metadata, so the forward path can reach them
    without having to inspect the FlashInfer wrapper internals."""

    if getattr(fi_backend.FlashInferMetadataBuilder.build, "_swa_patched", False):
        return

    _orig_build = fi_backend.FlashInferMetadataBuilder.build

    def _patched_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = _orig_build(self, common_prefix_len, common_attn_metadata, fast_build)
        # These are GPU tensors already on the right device.
        try:
            md._swa_block_table = common_attn_metadata.block_table_tensor
            md._swa_seq_lens = common_attn_metadata.seq_lens
            # num_decode_tokens == num_decodes for uniform-decode batches
            # (FlashInfer backend sets reorder_batch_threshold=1 in the
            # non-spec case). Decode rows are at the front.
        except Exception as e:
            # [FIXED-SILENT-NONE] Fix 2 — Ref: plans/silent_none_sweep_20260419.md §C1, plans/KILL_PATTERNS.md §P1
            logger.warning(
                "[SWA] metadata stash FAILED on %s (%s) — SWA kernel will NOT fire; "
                "fallback is silent at forward time (class=%s)",
                type(common_attn_metadata).__name__, e,
                type(common_attn_metadata).__name__,
            )
        return md

    _patched_build._swa_patched = True
    fi_backend.FlashInferMetadataBuilder.build = _patched_build


def _patch_impl_forward(fi_backend, swa) -> None:
    """Wrap ``FlashInferImpl.forward`` so decode calls on sliding layers take
    the Triton SWA path when eligible."""

    if getattr(fi_backend.FlashInferImpl.forward, "_swa_patched", False):
        return

    import torch

    from vllm.v1.attention.backends.flashinfer import (
        FIDecode,
        is_quantized_kv_cache,
    )

    # FP8 KV eligibility gate. On by default so that when a user flags
    # --kv-cache-dtype fp8 AND AUTOKERNEL_SWA_SPARSE=1, the SWA kernel handles
    # both sliding-layer decode and the FP8 dequant inline.
    _swa_fp8_enabled = os.environ.get("AUTOKERNEL_SWA_SPARSE_FP8", "1") == "1"

    _orig_forward = fi_backend.FlashInferImpl.forward

    def _patched_forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        # Conditions under which we even consider the fast path:
        if attn_metadata is None:
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        # sliding_window layer only. FlashInferImpl sets
        # window_left = sliding_window - 1  (i.e. >= 0) for sliding layers,
        # -1 for full-attention layers.
        if getattr(self, "window_left", -1) < 0:
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        window = int(self.window_left) + 1

        # We only intervene in the pure-decode, non-TRTLLM, non-cascade case.
        if attn_metadata.use_cascade or output_scale is not None:
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        if attn_metadata.num_prefill_tokens > 0 or attn_metadata.num_decode_tokens <= 0:
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        if not isinstance(attn_metadata.decode, FIDecode):
            # TRTLLMDecode path — not our target backend.
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        # FP8 KV: route into the SWA kernel's FP8 dequant path when enabled.
        # Otherwise fall back to FlashInfer (Discovery #37 perf caveat).
        kv_dtype_str = getattr(self, "kv_cache_dtype", "auto") or "auto"
        is_fp8_kv = is_quantized_kv_cache(kv_dtype_str)
        if is_fp8_kv and not _swa_fp8_enabled:
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )
        # Require per-layer FP8 scales to be present on FP8 path.
        k_scale_f = None
        v_scale_f = None
        if is_fp8_kv:
            k_scale_f = getattr(layer, "_k_scale_float", None)
            v_scale_f = getattr(layer, "_v_scale_float", None)
            if k_scale_f is None or v_scale_f is None:
                return _orig_forward(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale, output_block_scale,
                )

        block_table = getattr(attn_metadata, "_swa_block_table", None)
        seq_lens = getattr(attn_metadata, "_swa_seq_lens", None)
        if block_table is None or seq_lens is None:
            # [FIXED-SILENT-NONE] Fix 2 — Ref: plans/silent_none_sweep_20260419.md §C1, plans/KILL_PATTERNS.md §P1
            logger.warning(
                "[SWA] _swa_block_table=%s _swa_seq_lens=%s on %s — "
                "SWA Triton kernel skipped; falling back to FlashInfer (plugin installed but ineffective)",
                "None" if block_table is None else "ok",
                "None" if seq_lens is None else "ok",
                type(attn_metadata).__name__,
            )
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

        # Decode tokens live at the front of the batch.
        num_decode = attn_metadata.num_decode_tokens
        try:
            # Guard early: only call the kernel when ANY decode seq exceeds
            # window * guard_ratio. Skip the GPU-side max() by using seq_lens
            # on the decode slice (which is still on GPU; .max() is one sync
            # but the kernel itself launches immediately after).
            seq_lens_decode = seq_lens[:num_decode]
            if seq_lens_decode.numel() == 0:
                return _orig_forward(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale, output_block_scale,
                )
            # During CUDA graph capture, host-side syncs (.item()) are
            # unsupported. We can't branch on seq_len; bake the SWA kernel
            # unconditionally into the captured graph. At replay time, if
            # the per-sequence seq_len <= window the kernel still returns
            # the correct result (window_start=0, full attention on the
            # available tokens -- tested: cos=1.0 at seq_len=4096,win=4096).
            is_capturing = torch.cuda.is_current_stream_capturing()
            if is_capturing:
                # Skip the guard; always go through SWA kernel while capturing.
                max_seq = window + 1  # sentinel: "assume eligible"
            else:
                max_seq = int(seq_lens_decode.max().item())
                if not swa.sparse_path_eligible(max_seq, window):
                    return _orig_forward(
                        self, layer, query, key, value, kv_cache, attn_metadata,
                        output, output_scale, output_block_scale,
                    )

            # Unpack kv_cache [num_blocks, 2, block_size, num_kv_heads, head_dim].
            # Our Triton kernel uses explicit strides, so the non-contiguous
            # K/V slices are fine — do NOT call .contiguous() (that would copy
            # the entire KV cache on every decode step).
            if kv_cache.ndim != 5 or kv_cache.shape[1] != 2:
                # Not the expected layout; fall back.
                return _orig_forward(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale, output_block_scale,
                )
            # vLLM stores FP8 KV with the container viewed as uint8 in some
            # paths; mirror FlashInfer's view trick to reinterpret as FP8.
            if is_fp8_kv and kv_cache.dtype not in (torch.float8_e4m3fn,
                                                    torch.float8_e5m2):
                fp8_target = (torch.float8_e5m2 if kv_dtype_str == "fp8_e5m2"
                              else torch.float8_e4m3fn)
                try:
                    kv_cache = kv_cache.view(fp8_target)
                except Exception:
                    return _orig_forward(
                        self, layer, query, key, value, kv_cache, attn_metadata,
                        output, output_scale, output_block_scale,
                    )
            k_cache = kv_cache[:, 0]
            v_cache = kv_cache[:, 1]

            # Decode query: [num_decode_tokens, num_heads, head_size].
            decode_q = query[:num_decode]
            block_table_decode = block_table[:num_decode]

            layer_id = getattr(layer, "layer_name", None) or getattr(layer, "prefix", "") or f"sm{window}"

            out = swa.swa_sparse_decode(
                decode_q,
                k_cache,
                v_cache,
                block_table_decode,
                seq_lens_decode,
                window_size=window,
                sm_scale=self.scale,
                logits_soft_cap=(self.logits_soft_cap or 0.0),
                layer_id=layer_id,
                k_scale=k_scale_f,
                v_scale=v_scale_f,
            )
            if out is None:
                return _orig_forward(
                    self, layer, query, key, value, kv_cache, attn_metadata,
                    output, output_scale, output_block_scale,
                )

            # Write decode output into the output tensor (keeps output layout
            # identical to the dense path). output shape = [num_tokens, H, D]
            # but caller may later reshape to [num_tokens, H*D]; copy in-place.
            num_actual_tokens = attn_metadata.num_actual_tokens
            out_view = output[:num_actual_tokens]
            out_view[:num_decode].copy_(out, non_blocking=True)

            # If there is a prefill tail (should not happen in pure decode
            # bench, but safe), defer to original forward for the prefill
            # portion only. Easiest correct thing: only intervene when
            # num_prefill_tokens == 0, which we already gated on.
            return output

        except Exception as e:
            logger.warning(
                "[SWA] fast path raised %s: %s -- falling back to FlashInfer", type(e).__name__, e
            )
            return _orig_forward(
                self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale, output_block_scale,
            )

    _patched_forward._swa_patched = True
    fi_backend.FlashInferImpl.forward = _patched_forward


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    register()
