"""FusenKV attention backend for vLLM v1.

Implements vLLM's AttentionBackend ABC using the data-driven kernel system.
The KV cache stores quantized K+V (2-8 bit) with per-block scales in a
single uint8 tensor, plus a separate FP16 scales tensor.

Compatible with NVFP4 weight quantization (orthogonal systems).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import logging
import os
import torch

# Enable bounds checking via FUSEN_DEBUG=1 for diagnosing OOB crashes.
# Disabled by default (zero overhead in production).
_DEBUG = os.environ.get("FUSEN_DEBUG", "0") == "1"

# ---- Try to load C++ FusenCache decode attention kernel ----
# The C++ kernel is CUDA-graph compatible and avoids Triton JIT overhead.
# Falls back to Triton if the .so is not available.
_FUSENCACHE_CPP_SO = "/tmp/build_fusencache/fusencache_decode.so"
_HAS_CPP_DECODE = False
try:
    if os.path.exists(_FUSENCACHE_CPP_SO):
        torch.ops.load_library(_FUSENCACHE_CPP_SO)
        # Verify the op is registered
        _ = torch.ops.fusencache.decode_attention
        _HAS_CPP_DECODE = True
        logging.getLogger(__name__).info(
            "FusenCache C++ decode kernel loaded from %s", _FUSENCACHE_CPP_SO)
except Exception as e:
    logging.getLogger(__name__).warning(
        "FusenCache C++ decode kernel not available, falling back to Triton: %s", e)
    _HAS_CPP_DECODE = False

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


def _auto_num_kv_splits(max_seq_len: int, max_batch_size: int) -> int:
    """Choose num_kv_splits for split-K decode (mirrors Triton auto-select)."""
    # Heuristic: enough splits to keep SMs busy but not so many that
    # the reduce stage dominates. Typical: 8 for long seqs, 4 for short.
    blocks_per_seq = (max_seq_len + 15) // 16  # page_size=16
    if blocks_per_seq >= 512:
        return 16
    elif blocks_per_seq >= 128:
        return 8
    elif blocks_per_seq >= 32:
        return 4
    else:
        return 2


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

    # NOTE: build_for_cudagraph_capture() is NOT overridden.
    # The base class default calls self.build(common_prefix_len=0, ...).
    # This is correct for CUDA graph compatibility:
    # 1. build() references persistent tensors from CommonAttentionMetadata
    #    (seq_lens, block_table, slot_mapping) -- same addresses as captured
    # 2. No seq_lens.fill_(1) needed -- vLLM's dummy_run already sets
    #    appropriate values during capture
    # 3. Avoids the risk of fill_(1) corrupting seq_lens state that
    #    other code paths depend on

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

        # Try C++ decode kernel first (CUDA graph compatible, no Triton JIT)
        self._use_cpp_decode = False
        if _HAS_CPP_DECODE and self.spec.k_bits == 4 and self.spec.v_bits == 4:
            self._use_cpp_decode = True
            self._cpp_num_kv_splits = _auto_num_kv_splits(max_seq, 256)
            self._cpp_buffers = {}  # persistent buffers keyed by (B, Hq, D, device)
            logger.info(
                "FusenKV: using C++ decode kernel (num_kv_splits=%d)",
                self._cpp_num_kv_splits,
            )
            # Still build Triton as fallback (e.g. for non-k4v4 layers if mixed)
            self.decode_fn = make_decode_fn(
                self.spec,
                block_kv=16, block_h=8, num_warps=2,
                max_seq_len=max_seq,
                max_batch_size=256,
                cuda_graph_safe=True,
                logits_soft_cap=self.logits_soft_cap,
            )
        else:
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

    def _cpp_decode(self, query, kv_cache, scales, block_table, seq_lens,
                    scale, num_kv_heads):
        """Decode using the C++ CUDA kernel (CUDA graph compatible).

        Same signature as Triton decode_fn for drop-in replacement.
        Manages persistent output/mid_out buffers for CUDA graph safety.
        """
        B, Hq, D = query.shape
        Hk = num_kv_heads
        kv_group_size = Hq // Hk
        page_size = kv_cache.shape[1]

        # Persistent buffers (same pattern as Triton's cuda_graph_safe mode)
        key = (B, Hq, D, query.device, query.dtype)
        if key not in self._cpp_buffers:
            self._cpp_buffers[key] = {
                'mid_out': torch.empty(
                    B, Hq, self._cpp_num_kv_splits, D + 1,
                    dtype=torch.float32, device=query.device),
                'output': torch.empty(
                    B, Hq, D, dtype=query.dtype, device=query.device),
            }
        bufs = self._cpp_buffers[key]
        mid_out = bufs['mid_out']
        output = bufs['output']

        torch.ops.fusencache.decode_attention(
            output,
            query,
            kv_cache,
            scales,
            block_table,
            seq_lens,
            mid_out,
            float(scale),
            float(self.logits_soft_cap),
            self._cpp_num_kv_splits,
            D,                          # head_dim
            Hk,                         # num_kv_heads
            kv_group_size,
            page_size,
            self.spec.k_bits,           # k_bits
            self.spec.v_bits,           # v_bits
            self.spec.k_scale_block,    # scale_block_k
            self.spec.v_scale_block,    # scale_block_v
            float(self.spec.k_sym_offset),  # k_offset
            float(self.spec.v_sym_offset),  # v_offset
        )
        return output

    def _pytorch_decode(self, query, kv_cache, scales, block_table, seq_lens):
        """PyTorch reference decode: dequantize from paged cache + attention.

        Used for debugging to isolate Triton kernel issues.
        """
        B = query.shape[0]
        Hq = self.num_heads
        Hk = self.num_kv_heads
        D = self.head_size
        groups = Hq // Hk
        block_size = kv_cache.shape[1]
        k_bytes_per_dim = self.spec.k_bytes_per_dim
        v_bytes_per_dim = self.spec.v_bytes_per_dim
        k_region_bytes = int(k_bytes_per_dim * D)
        v_region_start = k_region_bytes
        min_sb = min(self.spec.k_scale_block, self.spec.v_scale_block)
        num_sb = D // min_sb

        q = query.reshape(B, Hq, D).float()
        outputs = torch.zeros(B, Hq, D, device=query.device, dtype=torch.float32)

        for b in range(B):
            sl = seq_lens[b].item()
            # Gather dequantized K/V for all positions: [sl, Hk, D]
            k_all = torch.zeros(sl, Hk, D, device=query.device, dtype=torch.float32)
            v_all = torch.zeros(sl, Hk, D, device=query.device, dtype=torch.float32)

            for pos in range(sl):
                blk_idx = pos // block_size
                blk_off = pos % block_size
                phys_block = block_table[b, blk_idx].item()
                slot = phys_block * block_size + blk_off

                for h in range(Hk):
                    raw = kv_cache[phys_block, blk_off, h].to(torch.int32)

                    # Dequantize K (4-bit: 2 values per byte)
                    k_packed = raw[:k_region_bytes]
                    k_lo = (k_packed & 0xF).float() - self.spec.k_sym_offset
                    k_hi = ((k_packed >> 4) & 0xF).float() - self.spec.k_sym_offset
                    # Apply scales
                    for sb_idx in range(num_sb):
                        sc = scales[slot, h, sb_idx, 0].float()
                        s = sb_idx * min_sb // 2
                        e = s + min_sb // 2
                        k_lo[s:e] *= sc
                        k_hi[s:e] *= sc
                    # Interleave
                    k_all[pos, h, 0::2] = k_lo
                    k_all[pos, h, 1::2] = k_hi

                    # Dequantize V (4-bit)
                    v_packed = raw[v_region_start:v_region_start + k_region_bytes]
                    v_lo = (v_packed & 0xF).float() - self.spec.v_sym_offset
                    v_hi = ((v_packed >> 4) & 0xF).float() - self.spec.v_sym_offset
                    for sb_idx in range(num_sb):
                        sc = scales[slot, h, sb_idx, 1].float()
                        s = sb_idx * min_sb // 2
                        e = s + min_sb // 2
                        v_lo[s:e] *= sc
                        v_hi[s:e] *= sc
                    v_all[pos, h, 0::2] = v_lo
                    v_all[pos, h, 1::2] = v_hi

            # GQA expand: [sl, Hk, D] -> [sl, Hq, D]
            if groups > 1:
                k_all = k_all.repeat_interleave(groups, dim=1)
                v_all = v_all.repeat_interleave(groups, dim=1)

            # Attention for this request
            q_b = q[b]  # [Hq, D]
            # k_all: [sl, Hq, D] -> transpose to [Hq, D, sl]
            k_t = k_all.permute(1, 2, 0)  # [Hq, D, sl]
            scores = torch.bmm(q_b.unsqueeze(1), k_t).squeeze(1)  # [Hq, sl]
            scores = scores * self.scale
            if self.logits_soft_cap > 0:
                scores = self.logits_soft_cap * torch.tanh(scores / self.logits_soft_cap)
            probs = torch.nn.functional.softmax(scores, dim=-1)  # [Hq, sl]
            # v_all: [sl, Hq, D] -> [Hq, sl, D]
            v_t = v_all.permute(1, 0, 2)
            out_b = torch.bmm(probs.unsqueeze(1), v_t).squeeze(1)  # [Hq, D]
            outputs[b] = out_b

        return outputs.to(query.dtype)  # [B, Hq, D]

    def _prefill_with_sliding_window(self, query, key, value, n_prefill):
        """Run prefill attention with optional sliding window mask.

        Uses manual attention for prefill to support logits_soft_cap (Gemma4).
        Falls back to SDPA when no soft cap is needed.
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

        # Use float32 for attention computation to avoid overflow
        pq_f = pq.float()
        pk_f = pk.float()
        pv_f = pv.float()

        if self.logits_soft_cap > 0:
            # Manual attention with logits_soft_cap (required for Gemma4).
            # SDPA does not support soft_cap, so we compute attention manually.
            # attn_weights: [B, H, N, N]
            attn_weights = torch.matmul(pq_f, pk_f.transpose(-2, -1)) * self.scale

            # Apply logits soft cap: cap * tanh(logits / cap)
            attn_weights = self.logits_soft_cap * torch.tanh(
                attn_weights / self.logits_soft_cap
            )

            # Build causal mask
            causal_mask = torch.ones(
                n_prefill, n_prefill, dtype=torch.bool, device=query.device
            ).tril()

            # Sliding window mask
            if self.sliding_window is not None and n_prefill > self.sliding_window:
                row_idx = torch.arange(n_prefill, device=query.device)
                col_idx = torch.arange(n_prefill, device=query.device)
                window_mask = col_idx[None, :] >= (
                    row_idx[:, None] - self.sliding_window + 1
                )
                causal_mask = causal_mask & window_mask

            attn_weights = attn_weights.masked_fill(
                ~causal_mask[None, None, :, :], float("-inf")
            )
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            prefill_out = torch.matmul(attn_weights, pv_f).to(query.dtype)
        else:
            # No soft cap needed -- use SDPA (faster, fused kernel)
            if self.sliding_window is not None and n_prefill > self.sliding_window:
                row_idx = torch.arange(n_prefill, device=query.device)
                col_idx = torch.arange(n_prefill, device=query.device)
                causal_mask = col_idx[None, :] <= row_idx[:, None]
                window_mask = col_idx[None, :] >= (
                    row_idx[:, None] - self.sliding_window + 1
                )
                attn_mask = (causal_mask & window_mask).unsqueeze(0).unsqueeze(0)
                prefill_out = torch.nn.functional.scaled_dot_product_attention(
                    pq_f, pk_f, pv_f, attn_mask=attn_mask, scale=self.scale,
                ).to(query.dtype)
            else:
                prefill_out = torch.nn.functional.scaled_dot_product_attention(
                    pq_f, pk_f, pv_f, is_causal=True, scale=self.scale,
                ).to(query.dtype)

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
            output tensor [num_tokens, num_heads, head_size] (3D)
        """
        if attn_metadata is None:
            # Profiling run
            if output is not None:
                return output.fill_(0)
            return query.new_zeros(query.shape[0], self.num_heads, self.head_size)

        self._ensure_scales(layer, kv_cache, query.device)

        num_tokens = attn_metadata.num_actual_tokens
        B = attn_metadata.num_reqs

        # Output is [num_tokens, num_heads, head_size] (3D) from vLLM
        if output is None:
            output = query.new_zeros(query.shape[0], self.num_heads, self.head_size)

        # Determine if this is a decode-only batch (all queries are 1 token)
        is_decode_only = (attn_metadata.max_query_len == 1)

        # ---- Store K/V into quantized cache ----
        # For decode-only under CUDA graphs: use full tensors (no dynamic
        # slicing). Padded slot_mapping entries are -1 and the store kernel
        # skips them (slot < 0 guard). For mixed prefill+decode (eager): use
        # num_actual_tokens to avoid storing garbage from padding positions.
        if attn_metadata.slot_mapping is not None and key is not None:
            if is_decode_only:
                # Full tensor — safe because padded slots are -1
                self.store_fn(
                    key, value, kv_cache,
                    attn_metadata.slot_mapping, layer, self.num_kv_heads,
                )
            else:
                self.store_fn(
                    key[:num_tokens], value[:num_tokens], kv_cache,
                    attn_metadata.slot_mapping[:num_tokens], layer, self.num_kv_heads,
                )

        if is_decode_only:
            # ---- Pure decode: all requests have 1 query token ----
            # CUDA graph compatibility: use the full tensor dimension from
            # query.shape[0] instead of slicing to B=num_reqs. Under CUDA
            # graphs, tensors are pre-allocated at the padded batch size and
            # seq_lens for padded entries are 0, causing the decode kernel to
            # skip them (split_start >= split_end early return + safe_sum
            # guard outputs zeros). Dynamic slicing (query[:B]) would create
            # tensors with shapes that differ from the captured graph,
            # causing CUDA assertion failures at batch sizes where the actual
            # B doesn't match a captured graph size (e.g. B=65 padded to 72).
            padded_B = query.shape[0]
            q = query
            if q.ndim == 2:
                q = q.reshape(padded_B, self.num_heads, self.head_size)
            q = q.contiguous()

            # Bounds checking (FUSEN_DEBUG=1 to enable)
            if _DEBUG:
                num_blocks = kv_cache.shape[0]
                block_size = kv_cache.shape[1]
                max_slots = num_blocks * block_size
                bt = attn_metadata.block_table[:B]
                sl = attn_metadata.seq_lens[:B]
                # Only check entries with seq_len > 0
                active = sl > 0
                if active.any():
                    active_bt = bt[active]
                    max_blk = active_bt.max().item()
                    if max_blk >= num_blocks:
                        logger.error(
                            "FUSEN_DEBUG: block_table OOB! max_block=%d >= "
                            "num_blocks=%d, B=%d, padded_B=%d",
                            max_blk, num_blocks, B, padded_B,
                        )
                    max_slot = max_blk * block_size + (block_size - 1)
                    scales_cap = layer._fusen_scales.shape[0]
                    if max_slot >= scales_cap:
                        logger.error(
                            "FUSEN_DEBUG: slot OOB! max_slot=%d >= "
                            "scales_capacity=%d, B=%d",
                            max_slot, scales_cap, B,
                        )

            _decode = self._cpp_decode if self._use_cpp_decode else self.decode_fn
            attn_out = _decode(
                q, kv_cache, layer._fusen_scales,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                self.scale, self.num_kv_heads,
            )

            output[:padded_B] = attn_out.to(output.dtype)
        else:
            # ---- Mixed prefill+decode or pure prefill ----
            qsl = attn_metadata.query_start_loc
            decode_indices = []
            for i in range(B):
                q_start = qsl[i].item()
                q_end = qsl[i + 1].item() if (i + 1) < qsl.shape[0] else num_tokens
                q_len = q_end - q_start

                if q_len > 1:
                    # Prefill: use manual attention (supports soft_cap)
                    prefill_out = self._prefill_with_sliding_window(
                        query[q_start:q_end],
                        key[q_start:q_end],
                        value[q_start:q_end],
                        q_len,
                    )
                    # prefill_out is [n, Hq*D], reshape to [n, Hq, D]
                    output[q_start:q_end] = prefill_out.reshape(
                        q_len, self.num_heads, self.head_size).to(output.dtype)
                else:
                    decode_indices.append(i)

            if decode_indices:
                # Batch decode for all 1-token requests
                n_dec = len(decode_indices)
                dec_idx = torch.tensor(decode_indices, device=query.device)
                dec_token_starts = qsl[dec_idx]

                dec_queries = query[dec_token_starts].contiguous()
                if dec_queries.ndim == 2:
                    dec_queries = dec_queries.reshape(n_dec, self.num_heads, self.head_size)

                _decode = self._cpp_decode if self._use_cpp_decode else self.decode_fn
                attn_out = _decode(
                    dec_queries, kv_cache, layer._fusen_scales,
                    attn_metadata.block_table[dec_idx],
                    attn_metadata.seq_lens[dec_idx],
                    self.scale, self.num_kv_heads,
                )

                output[dec_token_starts] = attn_out.to(output.dtype)

        return output
