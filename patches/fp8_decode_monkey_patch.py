"""
Monkey-patch FlashInfer's FP8 decode attention in vLLM with autokernel's
Triton split-K paged decode kernel.

Intercepts FlashInferImpl.forward() and reroutes the FP8 decode path
(both native FI and TRTLLM) to our kernel, which handles GQA, paged KV,
split-K, logits_soft_cap, and heterogeneous head dims (Gemma4).

Kill switch: AUTOKERNEL_FP8_DECODE=0
Force enable: AUTOKERNEL_FP8_DECODE=1 (default when this module is imported)

vLLM KV cache layout (NHD): [num_blocks, 2, block_size, num_kv_heads, head_dim]
vLLM KV cache layout (HND): [num_blocks, 2, num_kv_heads, block_size, head_dim]
Our kernel expects separate k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]

Query:
  vLLM: [num_tokens, num_heads, head_size]  (flat, decode tokens at front)
  Ours: [batch, num_q_heads, head_dim]      (same shape for decode: 1 token/req)

The patch slices kv_cache[:,0] and kv_cache[:,1] to get separate K/V,
reshapes query for our kernel, and writes output back into vLLM's buffer.
"""

import os
import sys
import math
import logging
import functools
import atexit

logger = logging.getLogger("autokernel.fp8_decode_patch")

_ENABLED = os.environ.get("AUTOKERNEL_FP8_DECODE", "1") != "0"
_PATCHED = False

# --- Observability counters ---
# Incremented each time a decode request falls back to FlashInfer
# instead of routing through our custom FP8 Triton kernel.
_fallback_fire_count = 0

# When set, any fallback raises RuntimeError instead of silently routing.
_ASSERT_NO_FALLBACK = os.environ.get("AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK", "0") == "1"


def _log_fallback_summary():
    """Emit a one-line banner at process exit with total fallback count."""
    if _fallback_fire_count > 0:
        logger.warning(
            "[FP8-decode] FALLBACK SUMMARY: FP8 decode patch fell back to FlashInfer "
            "%d time(s) during this process lifetime. The custom Triton split-K path "
            "did NOT handle these requests. If banked throughput claims assume the "
            "custom kernel was active, those numbers are suspect.",
            _fallback_fire_count,
        )
    else:
        logger.info(
            "[FP8-decode] FALLBACK SUMMARY: fallback_fire_count=0 — "
            "all FP8 decode requests were handled by the custom Triton split-K kernel."
        )


atexit.register(_log_fallback_summary)


def _get_kernel_root():
    """Find the autokernel project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _import_fp8_kernel():
    """Lazy-import the FP8 decode attention kernel."""
    root = _get_kernel_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    from kernels.fp8_decode_attention import fp8_decode_attention
    return fp8_decode_attention


def _run_autokernel_fp8_decode(
    decode_query,       # [num_decode_tokens, num_q_heads, head_dim]
    kv_cache,           # [num_blocks, 2, block_size, num_kv_heads, head_dim] (NHD, pre-permute)
    block_tables,       # [num_decodes, max_num_blocks]
    seq_lens,           # [num_decodes]
    k_scale_float,      # float
    v_scale_float,      # float
    sm_scale,           # float
    logits_soft_cap,    # float or None
    output_slice,       # [num_decode_tokens, num_q_heads, head_dim] — write here
):
    """Bridge between vLLM's tensor conventions and our kernel.

    The kv_cache tensor arrives BEFORE the stride_order permute, in the
    raw get_kv_cache_shape() layout:
      NHD: [num_blocks, 2, block_size, num_kv_heads, head_dim]
      HND: [num_blocks, 2, num_kv_heads, block_size, head_dim]

    Our kernel expects NHD-style separate tensors:
      k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
      v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    """
    import torch
    fp8_decode_attention = _import_fp8_kernel()

    # Split K and V from the interleaved cache
    # kv_cache shape: [num_blocks, 2, ...]
    # After the FP8 view cast, the raw shape is the same, just typed as fp8.
    k_cache_raw = kv_cache[:, 0]  # [num_blocks, block_size, num_kv_heads, head_dim] (NHD)
    v_cache_raw = kv_cache[:, 1]  # or [num_blocks, num_kv_heads, block_size, head_dim] (HND)

    # Detect HND layout: if dim1 == num_kv_heads and dim2 == block_size
    # In NHD: shape is [num_blocks, block_size, num_kv_heads, head_dim]
    # In HND: shape is [num_blocks, num_kv_heads, block_size, head_dim]
    # We can detect by checking VLLM_KV_CACHE_LAYOUT env var
    layout = os.environ.get("VLLM_KV_CACHE_LAYOUT", "NHD")
    if layout == "HND":
        # Transpose from [num_blocks, num_kv_heads, block_size, head_dim]
        #             to  [num_blocks, block_size, num_kv_heads, head_dim]
        k_cache = k_cache_raw.permute(0, 2, 1, 3).contiguous()
        v_cache = v_cache_raw.permute(0, 2, 1, 3).contiguous()
    else:
        k_cache = k_cache_raw
        v_cache = v_cache_raw

    # Ensure FP8 E4M3 dtype
    assert k_cache.dtype == torch.float8_e4m3fn, (
        f"Expected FP8 E4M3 KV cache, got {k_cache.dtype}"
    )

    num_decode_tokens = decode_query.shape[0]
    num_q_heads = decode_query.shape[1]
    head_dim = decode_query.shape[2]

    # For standard decode: 1 token per request, so batch = num_decode_tokens
    # The query is already [num_decode_tokens, num_q_heads, head_dim] which
    # matches our kernel's [batch, num_q_heads, head_dim]
    q = decode_query
    if q.dtype != torch.bfloat16:
        q = q.to(torch.bfloat16)

    # Build scale tensors from float scalars
    k_scale_t = torch.tensor([k_scale_float], dtype=torch.float32,
                             device=q.device)
    v_scale_t = torch.tensor([v_scale_float], dtype=torch.float32,
                             device=q.device)

    cap = logits_soft_cap if logits_soft_cap is not None else 0.0

    # Run our kernel
    result = fp8_decode_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_tables,
        seq_lens=seq_lens,
        k_scale=k_scale_t,
        v_scale=v_scale_t,
        sm_scale=sm_scale,
        logits_soft_cap=cap,
    )

    # Write result into output buffer — result is [batch, num_q_heads, head_dim] BF16
    # output_slice is [num_decode_tokens, num_q_heads, head_dim]
    output_slice.copy_(result.to(output_slice.dtype))


def apply_patch():
    """Monkey-patch FlashInferImpl.forward to intercept FP8 decode."""
    global _PATCHED

    if _PATCHED:
        logger.debug("FP8 decode patch already applied")
        return True

    if not _ENABLED:
        logger.info("FP8 decode patch disabled via AUTOKERNEL_FP8_DECODE=0")
        return False

    try:
        from vllm.v1.attention.backends.flashinfer import (
            FlashInferImpl,
            FIDecode,
            TRTLLMDecode,
        )
    except ImportError:
        logger.warning("Cannot import FlashInferImpl — vLLM not installed or wrong version")
        return False

    original_forward = FlashInferImpl.forward

    # One-time banner: note which metadata type will cause the fallback path.
    # This fires at patch-apply time (server startup) so it appears in logs
    # even if no requests are ever served.
    logger.warning(
        "[FP8-decode] INIT: patch applied to FlashInferImpl.forward. "
        "Fallback path X (FlashInfer native decode_wrapper.run) is ACTIVE for "
        "metadata type Y (FIDecode). Requests carrying FIDecode decode metadata "
        "will NOT use the custom Triton split-K kernel — they silently fall back. "
        "Set AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1 to raise instead of routing. "
        "Monitor '[FP8-decode] FALLBACK' log lines at WARNING level during serving."
    )

    @functools.wraps(original_forward)
    def patched_forward(self, layer, query, key, value, kv_cache,
                        attn_metadata, output=None, output_scale=None,
                        output_block_scale=None):
        """Patched forward: intercept FP8 decode, delegate everything else."""
        import torch

        # Only intercept when:
        # 1. KV cache dtype is FP8
        # 2. There are decode tokens
        # 3. Not DCP (our kernel doesn't handle distributed context parallel)
        # 4. No output quantization fusion (FP4/FP8 output — TRTLLM-specific)
        is_fp8_kv = self.kv_cache_dtype.startswith("fp8")
        has_decodes = (attn_metadata is not None
                       and attn_metadata.num_decode_tokens > 0)
        use_dcp = getattr(self, 'dcp_world_size', 1) > 1
        has_output_quant = output_scale is not None

        if not (is_fp8_kv and has_decodes and not use_dcp and not has_output_quant):
            # Fall through to original for non-FP8, prefill-only, DCP, or
            # fused output quant paths
            return original_forward(
                self, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale,
            )

        # --- Begin patched decode path ---
        # We handle the full forward (prefill + decode) but only replace
        # the decode kernel call. For prefill we delegate to the original.

        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.q_data_type == query.dtype

        # Replicate bmm1_scale / bmm2_scale init from original
        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_fp8_kv:
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float
        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_fp8_kv:
                self.bmm2_scale *= layer._v_scale_float

        num_actual_tokens = attn_metadata.num_actual_tokens

        # FP8 dtype cast for KV cache
        if (self.kv_sharing_target_layer_name is None
                and self.kv_cache_dtype.startswith("fp8")):
            from vllm.v1.attention.backends.flashinfer import FlashInferBackend
            torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.kv_cache_dtype
            )
            kv_cache = kv_cache.view(torch_dtype)

        # Trim padding
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        if attn_metadata.use_cascade:
            # Delegate cascade to original (rare case)
            return original_forward(
                self, layer, query, key, value, kv_cache,
                attn_metadata, output_padded, output_scale, output_block_scale,
            )

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # ---- Handle prefill portion (delegate to original FlashInfer) ----
        if num_prefill_tokens > 0:
            # We need to run the original prefill path. Rather than
            # duplicating all that code, we call the original forward
            # but with a metadata that has decode disabled.
            # However, this is complex. Instead, we replicate the prefill
            # section inline.
            from vllm.v1.attention.backends.flashinfer import (
                FlashInferBackend, FIPrefill, TRTLLMPrefill,
                BatchPrefillWithPagedKVCacheWrapper, BatchDCPPrefillWrapper,
                trtllm_batch_context_with_kv_cache,
                trtllm_prefill_attn_kvfp8_dequant,
                _get_trtllm_gen_workspace_buffer,
                FP8_DTYPE, FP4_DTYPE,
            )
            from vllm.v1.attention.backends.utils import get_kv_cache_layout
            from vllm.utils.torch_utils import is_strictly_contiguous
            from flashinfer.utils import FP4Tensor

            stride_order = FlashInferBackend.get_kv_cache_stride_order()
            kv_cache_permute = kv_cache.permute(*stride_order)

            prefill_query = query[num_decode_tokens:]
            prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)

            if not prefill_use_trtllm:
                assert isinstance(attn_metadata.prefill, FIPrefill)
                prefill_wrapper = attn_metadata.prefill.wrapper
                assert prefill_wrapper is not None
                prefill_wrapper.run(
                    prefill_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[num_decode_tokens:],
                )
            else:
                # TRTLLM prefill — replicate original logic
                prefill_query = prefill_query.contiguous().reshape(prefill_query.shape)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                assert get_kv_cache_layout() == "HND"

                if (attn_metadata.q_data_type != FP8_DTYPE
                        and self.kv_cache_dtype.startswith("fp8")):
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_permute, block_tables_prefill,
                        layer._k_scale, layer._v_scale,
                        attn_metadata.q_data_type,
                    )
                else:
                    mock_kv_cache = kv_cache_permute
                    mock_block_table = block_tables_prefill

                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=output[num_decode_tokens:],
                )

        # ---- Handle decode portion: OUR KERNEL ----
        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]

            # Get block tables and seq lens from either TRTLLMDecode or FIDecode metadata
            if isinstance(attn_metadata.decode, TRTLLMDecode):
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens
            elif isinstance(attn_metadata.decode, FIDecode):
                # FIDecode doesn't carry block_tables/seq_lens directly.
                # They're baked into the wrapper's plan. We need them from
                # the common metadata, which isn't stored in FIDecode.
                # Fall back to original FlashInfer decode for this case.
                #
                # WARNING: this fallback means the custom Triton split-K kernel
                # is INERT for this request.  If vLLM has selected the FIDecode
                # path (common on SM120 because is_device_capability(100)==False),
                # the "+71% throughput" claim for T2-I is NOT realised.
                # See: plans/tier2_3_audit_20260419.md §5, KILL_PATTERNS.md §P1.
                global _fallback_fire_count
                _fallback_fire_count += 1
                meta_type = type(attn_metadata.decode).__name__
                logger.warning(
                    "[FP8-decode] FALLBACK #%d: decode metadata is %s (FIDecode) — "
                    "routing to FlashInfer native decode_wrapper.run(). "
                    "Custom Triton split-K kernel did NOT execute for this request. "
                    "FIDecode does not expose block_tables/seq_lens outside the wrapper plan; "
                    "fix requires extracting those fields or switching to TRTLLMDecode path. "
                    "Set AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1 to raise instead.",
                    _fallback_fire_count,
                    meta_type,
                )
                if _ASSERT_NO_FALLBACK:
                    raise RuntimeError(
                        f"[FP8-decode] ASSERT_NO_FALLBACK: decode metadata is {meta_type} "
                        f"(FIDecode), expected TRTLLMDecode. "
                        f"Custom kernel cannot run. "
                        f"Unset AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK to allow fallback."
                    )
                from vllm.v1.attention.backends.flashinfer import FlashInferBackend
                stride_order = FlashInferBackend.get_kv_cache_stride_order()
                kv_cache_permute = kv_cache.permute(*stride_order)
                decode_wrapper = attn_metadata.decode.wrapper
                decode_wrapper.run(
                    decode_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[:num_decode_tokens],
                )
                return output_padded
            else:
                # Unknown decode metadata type — fall back to original forward.
                # Count and warn so this class of silent bypass is visible.
                global _fallback_fire_count
                _fallback_fire_count += 1
                meta_type = type(attn_metadata.decode).__name__
                logger.warning(
                    "[FP8-decode] FALLBACK #%d: unrecognised decode metadata type %s "
                    "(expected TRTLLMDecode or FIDecode) — routing to original FlashInferImpl.forward. "
                    "Custom Triton split-K kernel did NOT execute for this request.",
                    _fallback_fire_count,
                    meta_type,
                )
                if _ASSERT_NO_FALLBACK:
                    raise RuntimeError(
                        f"[FP8-decode] ASSERT_NO_FALLBACK: unrecognised decode metadata type "
                        f"{meta_type}. Unset AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK to allow fallback."
                    )
                return original_forward(
                    self, layer, query, key, value, kv_cache,
                    attn_metadata, output_padded, output_scale, output_block_scale,
                )

            # Run our FP8 decode kernel
            _run_autokernel_fp8_decode(
                decode_query=decode_query,
                kv_cache=kv_cache,         # raw, before permute
                block_tables=block_tables_decode,
                seq_lens=seq_lens_decode,
                k_scale_float=layer._k_scale_float,
                v_scale_float=layer._v_scale_float,
                sm_scale=self.scale,       # use the unpacked scale, not bmm1_scale
                logits_soft_cap=self.logits_soft_cap,
                output_slice=output[:num_decode_tokens],
            )

        return output_padded

    FlashInferImpl.forward = patched_forward
    _PATCHED = True
    logger.info(
        "[FP8-decode] AutoKernel FP8 decode patch applied to FlashInferImpl.forward — "
        "TRTLLMDecode path uses Triton split-K kernel; FIDecode path falls back to "
        "FlashInfer native (WARNING logged per fallback). "
        "Fallback count exposed via module _fallback_fire_count and atexit banner."
    )
    return True


def remove_patch():
    """Remove the monkey-patch and restore original FlashInfer decode."""
    global _PATCHED
    if not _PATCHED:
        return

    try:
        from vllm.v1.attention.backends.flashinfer import FlashInferImpl
        # The original is stored in the closure; we can't easily get it back.
        # Instead, reload the module.
        import importlib
        import vllm.v1.attention.backends.flashinfer as fi_mod
        importlib.reload(fi_mod)
        _PATCHED = False
        logger.info("AutoKernel FP8 decode patch removed")
    except Exception as e:
        logger.warning("Failed to remove patch: %s", e)


# Auto-apply when imported (unless disabled)
if _ENABLED:
    apply_patch()
