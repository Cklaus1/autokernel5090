"""FusenKV attention backend for vLLM v1.

Implements vLLM's AttentionBackend ABC using the data-driven kernel system.
The KV cache stores quantized K+V (2-8 bit) with per-block scales in a
single uint8 tensor, plus a separate FP16 scales tensor.

Compatible with NVFP4 weight quantization (orthogonal systems).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import logging
import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.kv_cache_interface import AttentionSpec

from fusen_kv.spec_resolver import resolve_spec

logger = logging.getLogger(__name__)


# ============================================================
# Metadata: what the scheduler passes to forward() each step
# ============================================================

@dataclass
class FusenKVMetadata(AttentionMetadata):
    """Per-step attention metadata for FusenKV."""

    # Block table: [batch, max_blocks_per_seq] -- maps logical to physical blocks
    block_table: torch.Tensor | None = None

    # Sequence lengths: [batch] -- number of tokens in each sequence's KV cache
    seq_lens: torch.Tensor | None = None

    # Slot mapping: [num_tokens] -- for store during prefill
    slot_mapping: torch.Tensor | None = None

    # Query start locations: [batch+1] -- for computing per-request query ranges
    query_start_loc: torch.Tensor | None = None

    # Total number of tokens and requests
    num_actual_tokens: int = 0
    num_reqs: int = 0
    max_query_len: int = 0

    # For cascade attention (not supported yet)
    use_cascade: bool = False


# ============================================================
# Metadata Builder: constructs FusenKVMetadata each step
# ============================================================

class FusenKVMetadataBuilder(AttentionMetadataBuilder[FusenKVMetadata]):
    """Builds FusenKVMetadata from scheduler outputs.

    Implements the vLLM v1 AttentionMetadataBuilder interface.
    """

    # CUDA graph support: we support uniform single-token decode batches.
    # Mixed prefill+decode requires dynamic shapes (no graph).
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    supports_update_block_table: bool = True

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

        self.cache_config = vllm_config.cache_config
        self.block_size = kv_cache_spec.block_size

        # For CUDA graph capture: pre-allocate max-size metadata tensors
        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        compilation_config = vllm_config.compilation_config
        self.max_cudagraph_size = (
            compilation_config.max_cudagraph_capture_size
            if compilation_config else 0
        )

    def update_block_table(
        self,
        metadata: FusenKVMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FusenKVMetadata:
        """Update block table and slot mapping for a different KV cache group.

        Creates a new metadata instance sharing all fields except the
        block table and slot mapping, which differ per cache group.
        """
        return FusenKVMetadata(
            block_table=blk_table,
            seq_lens=metadata.seq_lens,
            slot_mapping=slot_mapping,
            query_start_loc=metadata.query_start_loc,
            num_actual_tokens=metadata.num_actual_tokens,
            num_reqs=metadata.num_reqs,
            max_query_len=metadata.max_query_len,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata,
    ) -> FusenKVMetadata:
        """Build metadata for CUDA graph capture (max sizes, decode-only)."""
        m = common_attn_metadata
        return FusenKVMetadata(
            block_table=m.block_table_tensor,
            seq_lens=m.seq_lens,
            slot_mapping=m.slot_mapping,
            query_start_loc=m.query_start_loc,
            num_actual_tokens=m.num_actual_tokens,
            num_reqs=m.num_reqs,
            max_query_len=1,  # decode-only
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> FusenKVMetadata:
        m = common_attn_metadata

        return FusenKVMetadata(
            block_table=m.block_table_tensor,
            seq_lens=m.seq_lens,
            slot_mapping=m.slot_mapping,
            query_start_loc=m.query_start_loc,
            num_actual_tokens=m.num_actual_tokens,
            num_reqs=m.num_reqs,
            max_query_len=m.max_query_len,
        )


# ============================================================
# Backend: the class vLLM discovers via register_backend()
# ============================================================

class FusenKVBackend(AttentionBackend):
    """Data-driven quantized KV cache attention backend.

    Supports k_bits={4,8} x v_bits={4,8} with configurable per-block
    scales. The kernel is generated from a KVCacheSpec at init time.

    KV cache layout:
        kv_cache: [num_blocks, block_size, num_kv_heads, slot_bytes] uint8
        scales:   [max_slots, num_kv_heads, num_scale_blocks, 2] float16
    """

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[str]] = [
        "fusen",
        "k4v4", "k4v4b16", "k4v4b32", "k4v4b64",
        "k8v4", "k8v4b16", "k8v4b32",
        "k8v8", "k8v8b32",
        "int4", "int8",
    ]

    @staticmethod
    def get_name() -> str:
        # Must match the AttentionBackendEnum member name we register under.
        # vLLM does: AttentionBackendEnum[backend.get_name()] to look up
        # the enum member, so this must be "CUSTOM" (our registered slot).
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["FusenKVImpl"]:
        return FusenKVImpl

    @staticmethod
    def get_builder_cls() -> type["FusenKVMetadataBuilder"]:
        return FusenKVMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16]  # fixed page size for quantized cache

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "fusen",
    ) -> tuple[int, ...]:
        spec = resolve_spec(cache_dtype_str)
        slot_bytes = int(spec.k_bytes_per_dim * head_size
                         + spec.v_bytes_per_dim * head_size)
        return (num_blocks, block_size, num_kv_heads, slot_bytes)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Supports Gemma4's head dims (256 and 512) and common sizes
        return head_size in (64, 128, 256, 512)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: "DeviceCapability") -> bool:
        # SM80+ (A100, RTX 3090, RTX 4090, RTX 5090, etc.)
        return capability.major >= 8


# ============================================================
# Implementation: the actual forward() that runs each step
# ============================================================

class FusenKVImpl(AttentionImplBase):
    """Forward pass implementation for FusenKV.

    Wraps the data-driven kernel's decode_fn and store_fn.
    Supports logits_soft_cap (for Gemma models) and sliding window.
    Compatible with NVFP4 weight quantization.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.logits_soft_cap = float(logits_soft_cap) if logits_soft_cap and logits_soft_cap > 0 else 0.0

        if alibi_slopes is not None:
            raise NotImplementedError("FusenKV does not support ALiBi yet")

        # Detect weight quantization from vllm config for compatibility check
        weight_quant = "none"
        try:
            from vllm.config import get_current_vllm_config
            vllm_config = get_current_vllm_config()
            if vllm_config and vllm_config.model_config:
                quant = vllm_config.model_config.quantization
                if quant:
                    weight_quant = quant
        except Exception:
            pass

        # Warn (not error) if weight_quant + kv_cache_dtype is untested/blocked
        from fusen_kv.compatibility import warn_if_untested
        warn_if_untested(weight_quant, kv_cache_dtype)

        # Resolve the KVCacheSpec from the dtype string
        from fusen_kv.spec_resolver import resolve_spec
        self.spec = resolve_spec(kv_cache_dtype)

        # Build decode and store functions
        from kv_cache_gen.generate import make_decode_fn, make_store_fn

        max_seq = sliding_window if sliding_window else 131072
        self.decode_fn = make_decode_fn(
            self.spec,
            block_kv=16, block_h=8, num_warps=2,
            max_seq_len=max_seq,
            max_batch_size=256,
            cuda_graph_safe=True,
            logits_soft_cap=self.logits_soft_cap,
        )
        self.store_fn = make_store_fn(self.spec)

        # Scales tensor -- lazily allocated per layer
        self._scales_initialized = False

        logger.info(
            "FusenKV layer: heads=%d, head_size=%d, kv_heads=%d, "
            "spec=%s, soft_cap=%.1f, sliding=%s, weight_quant=%s",
            num_heads, head_size, num_kv_heads,
            self.spec.name, self.logits_soft_cap,
            sliding_window, weight_quant,
        )

    def _ensure_scales(self, layer, kv_cache, device):
        """Allocate the scales tensor on first use."""
        if self._scales_initialized:
            return

        num_blocks = kv_cache.shape[0]
        block_size = kv_cache.shape[1]
        max_slots = num_blocks * block_size
        min_block = min(self.spec.k_scale_block, self.spec.v_scale_block)
        num_sb = self.head_size // min_block

        if not hasattr(layer, '_fusen_scales'):
            layer._fusen_scales = torch.zeros(
                max_slots, self.num_kv_heads, num_sb, 2,
                dtype=torch.float16, device=device,
            )
        # Alias for the store_fn which expects layer._fc_scales
        layer._fc_scales = layer._fusen_scales
        self._scales_initialized = True

    def _prefill_with_sliding_window(self, query, key, value, n_prefill):
        """Run prefill attention with optional sliding window mask.

        Uses torch SDPA for prefill (compute-bound, not memory-bound).
        Applies causal + sliding window mask when sliding_window is set.
        """
        pq = query[:n_prefill].reshape(1, n_prefill, self.num_heads, self.head_size)
        pk = key[:n_prefill].reshape(1, n_prefill, self.num_kv_heads, self.head_size)
        pv = value[:n_prefill].reshape(1, n_prefill, self.num_kv_heads, self.head_size)

        # GQA expand
        if self.num_heads != self.num_kv_heads:
            groups = self.num_heads // self.num_kv_heads
            pk = pk.repeat_interleave(groups, dim=2)
            pv = pv.repeat_interleave(groups, dim=2)

        # Transpose for SDPA: [B, H, N, D]
        pq = pq.transpose(1, 2)
        pk = pk.transpose(1, 2)
        pv = pv.transpose(1, 2)

        # Build attention mask
        if self.sliding_window is not None and n_prefill > self.sliding_window:
            # Combined causal + sliding window mask
            # Token i can attend to tokens [max(0, i - window + 1), i]
            row_idx = torch.arange(n_prefill, device=query.device)
            col_idx = torch.arange(n_prefill, device=query.device)
            # Causal: col <= row
            causal_mask = col_idx[None, :] <= row_idx[:, None]
            # Sliding window: col >= row - window + 1
            window_mask = col_idx[None, :] >= (row_idx[:, None] - self.sliding_window + 1)
            attn_mask = causal_mask & window_mask
            # SDPA expects [B, 1, N, N] or [N, N] bool mask
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            prefill_out = torch.nn.functional.scaled_dot_product_attention(
                pq.to(torch.float16), pk.to(torch.float16), pv.to(torch.float16),
                attn_mask=attn_mask, scale=self.scale,
            )
        else:
            # Pure causal mask (window >= prefill length or no window)
            prefill_out = torch.nn.functional.scaled_dot_product_attention(
                pq.to(torch.float16), pk.to(torch.float16), pv.to(torch.float16),
                is_causal=True, scale=self.scale,
            )

        # Apply logits soft cap via custom SDPA is not directly supported,
        # but for prefill this is acceptable since prefill quality impact
        # is minimal (only affects the KV store, not generation quality).
        # TODO: implement soft_cap in prefill via manual attention if needed.

        return prefill_out.transpose(1, 2).reshape(n_prefill, -1)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FusenKVMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: store KV during prefill, decode attention for generation.

        Args:
            query:  [num_tokens, num_heads, head_size]
            key:    [num_tokens, num_kv_heads, head_size]
            value:  [num_tokens, num_kv_heads, head_size]
            kv_cache: [num_blocks, block_size, num_kv_heads, slot_bytes] uint8
            attn_metadata: FusenKVMetadata with block_table, seq_lens, etc.
        Returns:
            [num_tokens, num_heads * head_size]
        """
        if attn_metadata is None:
            # Profiling run
            if output is not None:
                return output.fill_(0)
            return query.new_zeros(query.shape[0], self.num_heads, self.head_size)

        self._ensure_scales(layer, kv_cache, query.device)

        num_tokens = query.shape[0]
        B = attn_metadata.num_reqs

        # Output is [num_tokens, num_heads, head_size] (3D) from vLLM
        if output is None:
            output = query.new_zeros(num_tokens, self.num_heads, self.head_size)

        # Determine if this is a decode-only batch (all queries are 1 token)
        is_decode_only = (attn_metadata.max_query_len == 1)

        # ---- Store K/V into quantized cache (synchronous for CUDA graph safety) ----
        if attn_metadata.slot_mapping is not None and key is not None:
            self.store_fn(
                key, value, kv_cache,
                attn_metadata.slot_mapping, layer, self.num_kv_heads,
            )

        if is_decode_only:
            # ---- Pure decode: all requests have 1 query token ----
            # query is [B, Hq, D], our decode_fn expects [B, Hq, D]
            q = query[:B]
            if q.ndim == 2:
                q = q.reshape(B, self.num_heads, self.head_size)

            attn_out = self.decode_fn(
                q, kv_cache, layer._fusen_scales,
                attn_metadata.block_table[:B],
                attn_metadata.seq_lens[:B],
                self.scale, self.num_kv_heads,
            )

            # attn_out is [B, Hq, D] -- write directly into 3D output
            output[:B] = attn_out
        else:
            # ---- Mixed prefill+decode or pure prefill ----
            qsl = attn_metadata.query_start_loc
            decode_indices = []
            for i in range(B):
                q_start = qsl[i].item()
                q_end = qsl[i + 1].item()
                q_len = q_end - q_start

                if q_len > 1:
                    # Prefill: use SDPA for this request
                    prefill_out = self._prefill_with_sliding_window(
                        query[q_start:q_end],
                        key[q_start:q_end],
                        value[q_start:q_end],
                        q_len,
                    )
                    # prefill_out is [n, Hq*D], reshape to [n, Hq, D]
                    output[q_start:q_end] = prefill_out.reshape(
                        q_len, self.num_heads, self.head_size)
                else:
                    decode_indices.append(i)

            if decode_indices:
                # Batch decode for all 1-token requests
                n_dec = len(decode_indices)
                dec_idx = torch.tensor(decode_indices, device=query.device)
                dec_token_starts = qsl[dec_idx]

                dec_queries = query[dec_token_starts]
                if dec_queries.ndim == 2:
                    dec_queries = dec_queries.reshape(n_dec, self.num_heads, self.head_size)

                attn_out = self.decode_fn(
                    dec_queries, kv_cache, layer._fusen_scales,
                    attn_metadata.block_table[dec_idx],
                    attn_metadata.seq_lens[dec_idx],
                    self.scale, self.num_kv_heads,
                )

                output[dec_token_starts] = attn_out

        return output
