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
# Enable sync-after-each-kernel for crash pinpointing via FUSEN_SYNC=1.
# Only activates in the mixed (eager) path -- never during CUDA graph capture/replay.
_SYNC = os.environ.get("FUSEN_SYNC", "0") == "1"

def _can_sync():
    """Return True only when we are NOT inside a CUDA graph capture."""
    if not _SYNC:
        return False
    return not torch.cuda.is_current_stream_capturing()

# ---- Try to load C++ FusenCache decode attention kernel ----
# The C++ kernel is CUDA-graph compatible and avoids Triton JIT overhead.
# Falls back to Triton if the .so is not available.
_FUSENCACHE_CPP_SO = "/tmp/build_fusencache/fusencache_decode.so"
_HAS_CPP_DECODE = False
_HAS_CPP_STORE = False
try:
    if os.path.exists(_FUSENCACHE_CPP_SO):
        torch.ops.load_library(_FUSENCACHE_CPP_SO)
        # Verify the ops are registered
        _ = torch.ops.fusencache.decode_attention
        _HAS_CPP_DECODE = True
        logging.getLogger(__name__).info(
            "FusenCache C++ decode kernel loaded from %s", _FUSENCACHE_CPP_SO)
    try:
        _ = torch.ops.fusencache.store_kv
        _HAS_CPP_STORE = True
        logging.getLogger(__name__).info(
            "FusenCache C++ store kernel loaded (CUDA graph safe)")
    except AttributeError:
        pass
except Exception as e:
    logging.getLogger(__name__).warning(
        "FusenCache C++ kernels not available, falling back to Triton: %s", e)
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

    # CUDA graph support: dynamic based on C++ kernel availability.
    #
    # On SM120/Blackwell + Triton 3.6.0, CUDA graph replay of Triton-generated
    # store/decode kernels crashes with "illegal instruction". However, our C++
    # CUDA kernels (fusencache_decode.so) don't have this issue.
    #
    # When BOTH C++ decode AND C++ store kernels are available:
    #   -> UNIFORM_SINGLE_TOKEN_DECODE (full CUDA graphs for decode-only)
    #
    #   NOTE: ALWAYS was tested and crashes during mixed prefill+decode graph
    #   replay (Discovery #57). The universal decode path logic is graph-safe
    #   but something in the C++ kernel or vLLM's graph replay corrupts state.
    #
    #   CRITICAL: new vLLM (0.1.dev100+) auto-downgrades FULL → FULL_DECODE_ONLY
    #   when backend declares < ALWAYS. This changes scheduler behavior and
    #   causes 3.4x throughput regression at C=128. The fix is in plugin.py:
    #   _patch_cudagraph_mode_override() restores FULL mode after resolution.
    #
    # Otherwise (Triton fallback):
    #   -> NEVER (piecewise CUDA graphs only, attention runs in eager)
    _cudagraph_support = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        if (_HAS_CPP_DECODE and _HAS_CPP_STORE)
        else AttentionCGSupport.NEVER
    )
    # CRITICAL: must be False. When True, vLLM caches metadata and calls
    # update_block_table() which updates block_table/slot_mapping but reuses
    # the STALE query_start_loc from a prior step. When batch size changes,
    # the universal decode path reads past the end of the smaller allocation
    # → illegal memory access. Setting False forces build() every step,
    # ensuring query_start_loc always reflects the current batch.
    supports_update_block_table: bool = False

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
        self._max_seq = max_seq

        # ---- Shared decode buffers (CUDA graph memory fix) ----
        # Pre-allocate ONE set of mid_out/output buffers at max_batch_size.
        # All CUDA graph captures use the SAME tensor addresses (just
        # different seq_lens padding), eliminating per-graph-size allocation
        # that caused 10-120 GiB memory explosion with many capture sizes.
        #
        # Detect max batch size from vllm config, fall back to 512.
        # CRITICAL: _max_B must be >= max(max_num_seqs, max_cudagraph_capture_size)
        # to avoid OOB during CUDA graph capture (which pads batches up to
        # max_cudagraph_capture_size) AND during eager fallback (which can
        # send B up to max_num_seqs for decode, or max_num_batched_tokens
        # for mixed prefill+decode -- the latter is handled by temp buffer
        # allocation in make_decode_fn when B > _max_B).
        _max_B = 512
        try:
            from vllm.config import get_current_vllm_config
            vllm_cfg = get_current_vllm_config()
            if vllm_cfg and vllm_cfg.scheduler_config:
                _max_B = vllm_cfg.scheduler_config.max_num_seqs
            if vllm_cfg and vllm_cfg.compilation_config:
                cg_max = vllm_cfg.compilation_config.max_cudagraph_capture_size
                if cg_max and cg_max > 0:
                    # Use the LARGER of max_num_seqs and max_cudagraph_capture_size.
                    # CUDA graphs pad batches up to cg_max, so shared buffers
                    # must accommodate that. Previously this REPLACED _max_B
                    # which was correct when cg_max >= max_num_seqs, but could
                    # silently under-allocate if the config was read before
                    # _set_cudagraph_sizes() finalized cg_max.
                    _max_B = max(_max_B, cg_max)
        except Exception:
            pass

        # Determine num_kv_splits for shared buffer allocation.
        # The shared buffer dimension caps the maximum splits used at runtime.
        # The decode kernel's adaptive split-K will never exceed this value.
        #
        # MEMORY BUDGET: With 30 layers, each buffer is:
        #   [_max_B, num_heads, _num_kv_splits, head_size+1] * 4 bytes
        # At _max_B=512, num_heads=16, head_size=256, 32 splits:
        #   512*16*32*257*4 = 271 MB/layer * 30 = 8.1 GB (too much!)
        # Capping at 8 splits: 67 MB/layer * 30 = 2.0 GB (fits)
        #
        # Split-K with 8 splits still provides good SM utilization for
        # small batches (B=1..8 -> 8 splits -> 8 blocks per head group).
        # For large batches (B=128+), adaptive splits returns 2-4 anyway.
        from kv_cache_gen.generate import _optimal_splits
        _MAX_SHARED_SPLITS = 8  # Cap for memory -- saves ~7 GB on Gemma4
        _num_kv_splits = min(_MAX_SHARED_SPLITS, _optimal_splits(max_seq, 1))
        self._max_shared_splits = _num_kv_splits

        # Determine output dtype: match the model's compute dtype
        _out_dtype = torch.float16
        try:
            vllm_cfg  # may not exist if the try block above failed
            if vllm_cfg and vllm_cfg.model_config:
                _out_dtype = vllm_cfg.model_config.dtype
        except (NameError, Exception):
            pass

        self._shared_mid_out = torch.empty(
            _max_B, num_heads, _num_kv_splits, head_size + 1,
            dtype=torch.float32, device='cuda',
        )
        self._shared_output = torch.empty(
            _max_B, num_heads, head_size,
            dtype=_out_dtype, device='cuda',
        )
        logger.info(
            "FusenKV: shared decode buffers allocated at max_B=%d "
            "(mid_out=%.1f MiB, output=%.1f MiB)",
            _max_B,
            self._shared_mid_out.nbytes / (1024 * 1024),
            self._shared_output.nbytes / (1024 * 1024),
        )

        # Try C++ decode kernel first (CUDA graph compatible, no Triton JIT)
        self._use_cpp_decode = False
        if _HAS_CPP_DECODE and self.spec.k_bits == 4 and self.spec.v_bits == 4:
            self._use_cpp_decode = True
            # CRITICAL: Use the SAME splits count and shared buffers as
            # Triton (_num_kv_splits from _optimal_splits(max_seq, 1)).
            # Previously, a separate _cpp_num_kv_splits from
            # _auto_num_kv_splits(max_seq, _max_B) was used, which:
            #   1. Allocated SEPARATE buffers (~1.2 GB wasted across layers)
            #   2. Used too few splits (e.g. 8 vs 32), causing poor SM
            #      utilization at high concurrency and OOM from memory
            #      pressure of double-buffering.
            # Now the C++ kernel reuses the already-allocated shared
            # buffers and gets the same max split count.
            self._cpp_num_kv_splits = _num_kv_splits
            self._cpp_shared_mid_out = self._shared_mid_out
            self._cpp_shared_output = self._shared_output
            logger.info(
                "FusenKV: using C++ decode kernel (num_kv_splits=%d, "
                "shared buffers with Triton fallback)",
                self._cpp_num_kv_splits,
            )
            # Still build Triton as fallback (e.g. for non-k4v4 layers if mixed)
            self.decode_fn = make_decode_fn(
                self.spec,
                block_kv=16, block_h=8, num_warps=2,
                num_kv_splits=_num_kv_splits,
                max_seq_len=max_seq,
                max_batch_size=_max_B,
                cuda_graph_safe=True,
                logits_soft_cap=self.logits_soft_cap,
                shared_mid_out=self._shared_mid_out,
                shared_output=self._shared_output,
            )
        else:
            self.decode_fn = make_decode_fn(
                self.spec,
                block_kv=16, block_h=8, num_warps=2,
                num_kv_splits=_num_kv_splits,
                max_seq_len=max_seq,
                max_batch_size=_max_B,
                cuda_graph_safe=True,
                logits_soft_cap=self.logits_soft_cap,
                shared_mid_out=self._shared_mid_out,
                shared_output=self._shared_output,
            )

        self._triton_store_fn = make_store_fn(self.spec)
        self._use_cpp_store = False
        if _HAS_CPP_STORE and self.spec.k_bits == 4 and self.spec.v_bits == 4:
            self._use_cpp_store = True
            logger.info("FusenKV: using C++ store kernel (CUDA graph safe)")

        _page_size = 16  # fixed page size for quantized cache

        def _store_fn(key, value, kv_cache, slot_mapping, layer, num_kv_heads):
            """Dispatch to C++ or Triton store kernel."""
            if self._use_cpp_store:
                D = key.shape[-1]
                torch.ops.fusencache.store_kv(
                    key,
                    value,
                    kv_cache,
                    layer._fusen_scales,
                    slot_mapping.int(),
                    D,                          # head_dim
                    _page_size,                 # page_size
                    self.spec.k_bits,
                    self.spec.v_bits,
                    self.spec.k_scale_block,
                    self.spec.v_scale_block,
                    float(self.spec.k_sym_offset),
                    float(self.spec.v_sym_offset),
                )
            else:
                self._triton_store_fn(key, value, kv_cache, slot_mapping,
                                      layer, num_kv_heads)

        self.store_fn = _store_fn

        # Scales tensor -- lazily allocated per layer
        self._scales_initialized = False

        # CUDA event for lightweight stream fencing (replaces full stream sync).
        # At the end of forward(), we record an event after all kernels launch.
        # At the start of the NEXT forward(), we make the current stream wait
        # for that event before launching new kernels. This ensures:
        #   1. No in-flight kernel reads stale/recycled memory from freed temps
        #   2. CPU is NOT blocked (unlike synchronize()), preserving async scheduling
        #   3. GPU-side ordering is enforced without serializing the CPU pipeline
        # Cost: ~5us per layer per step (vs ~100us for full synchronize).
        self._prev_step_event = torch.cuda.Event()
        # Record an initial event so the first forward() doesn't wait on garbage
        self._prev_step_event.record()

        # Retain references to cloned metadata tensors from the previous step.
        # Without this, Python GC frees the clones when the next step starts,
        # and the CUDA allocator may recycle the memory while the GPU is still
        # reading them (the GPU hasn't finished the previous step's kernels yet).
        # By storing them as instance state, they stay alive until the NEXT
        # step clones them (at which point the event fence guarantees the GPU
        # has finished reading them).
        self._prev_block_table = None
        self._prev_seq_lens = None
        self._prev_slot_mapping = None
        self._prev_query_start_loc = None

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
        Uses shared pre-allocated buffers so all CUDA graph capture sizes
        share the same tensor addresses (no per-graph memory explosion).

        Uses adaptive num_kv_splits like the Triton path: more splits at
        small batch (better SM utilization) capped at the pre-allocated
        buffer size (_cpp_num_kv_splits).
        """
        B, Hq, D = query.shape
        Hk = num_kv_heads
        kv_group_size = Hq // Hk
        page_size = kv_cache.shape[1]

        # Adaptive split-K: match Triton behavior for SM utilization.
        # At small B, use more splits; at large B, fewer splits.
        # Capped at the pre-allocated buffer dimension.
        from kv_cache_gen.generate import _optimal_splits
        num_kv_splits = min(
            self._cpp_num_kv_splits,
            _optimal_splits(self._max_seq, B),
        )
        # Ensure at least 1 split
        num_kv_splits = max(1, num_kv_splits)

        # Use shared buffers when B fits, otherwise allocate temporary buffers
        # (eager mode with B > max_cudagraph_capture_size).
        if B <= self._cpp_shared_mid_out.shape[0]:
            mid_out = self._cpp_shared_mid_out
            output = self._cpp_shared_output
        else:
            mid_out = torch.empty(
                B, Hq, self._cpp_num_kv_splits, D + 1,
                dtype=torch.float32, device=query.device,
            )
            output = torch.empty(
                B, Hq, D,
                dtype=self._cpp_shared_output.dtype, device=query.device,
            )

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
            num_kv_splits,
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

        # ASYNC SCHEDULING FIX: Clone metadata tensors to prevent CPU-GPU race.
        #
        # Under vLLM's async scheduling, the CPU prepares the NEXT step's
        # metadata (block_table, seq_lens, slot_mapping) while the GPU is
        # still running the CURRENT step's kernels. These metadata tensors
        # are persistent and updated IN-PLACE, so without protection, the
        # GPU reads corrupted values mid-kernel, causing "illegal memory
        # access" crashes at C=16+.
        #
        # The fix: clone metadata tensors into private copies before passing
        # them to our GPU kernels. The CPU can freely modify the originals
        # for the next step while our kernels read the clones.
        #
        # Cost: ~32KB of GPU memcpy per layer per step (trivial at RTX 5090's
        # 1.8 TB/s bandwidth). Combined with CUDA events for shared buffer
        # protection, this eliminates both the CPU-GPU metadata race and the
        # GPU-GPU shared buffer race WITHOUT blocking the CPU thread.
        #
        # Skip during CUDA graph capture: tensors must have fixed addresses,
        # and CUDA graphs handle ordering natively.
        _capturing = torch.cuda.is_current_stream_capturing()

        if not _capturing:
            # GPU-side fence: ensure previous step's kernels finish before
            # we start reusing shared decode buffers (mid_out, output).
            # After this wait completes on the GPU, the previous step's clones
            # (stored in self._prev_*) are safe to release.
            torch.cuda.current_stream().wait_event(self._prev_step_event)

            # Clone metadata to isolate from async scheduler writes.
            # This is the key fix: our kernels read clones, CPU modifies originals.
            _block_table = attn_metadata.block_table.clone() if attn_metadata.block_table is not None else None
            _seq_lens = attn_metadata.seq_lens.clone() if attn_metadata.seq_lens is not None else None
            _slot_mapping = attn_metadata.slot_mapping.clone() if attn_metadata.slot_mapping is not None else None
            _query_start_loc = attn_metadata.query_start_loc.clone() if attn_metadata.query_start_loc is not None else None

            # Retain references to clones so Python GC doesn't free them while
            # the GPU is still reading them. The previous step's clones are now
            # safe to release (the event fence above guarantees completion).
            self._prev_block_table = _block_table
            self._prev_seq_lens = _seq_lens
            self._prev_slot_mapping = _slot_mapping
            self._prev_query_start_loc = _query_start_loc
        else:
            # During CUDA graph capture: use originals (fixed addresses required)
            _block_table = attn_metadata.block_table
            _seq_lens = attn_metadata.seq_lens
            _slot_mapping = attn_metadata.slot_mapping
            _query_start_loc = attn_metadata.query_start_loc

        self._ensure_scales(layer, kv_cache, query.device)

        num_tokens = attn_metadata.num_actual_tokens
        B = attn_metadata.num_reqs

        # Output is [num_tokens, num_heads, head_size] (3D) from vLLM
        if output is None:
            output = query.new_zeros(query.shape[0], self.num_heads, self.head_size)

        # Determine if this is a decode-only batch (all queries are 1 token)
        is_decode_only = (attn_metadata.max_query_len == 1)

        # ---- Store K/V into quantized cache ----
        # CUDA graph safe: ALWAYS use full tensors (no dynamic slicing).
        # Padded slot_mapping entries are -1 and the store kernel skips
        # them (slot < 0 guard). This ensures identical tensor shapes
        # during graph capture and replay for ALL batch types.
        if _slot_mapping is not None and key is not None:
            self.store_fn(
                key, value, kv_cache,
                _slot_mapping, layer, self.num_kv_heads,
            )
            if _can_sync():
                torch.cuda.synchronize()
                logger.info("FUSEN_SYNC: store OK, is_decode=%s, B=%d, num_tokens=%d",
                            is_decode_only, B, num_tokens)

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
            # Skip during CUDA graph capture (.item() calls would crash capture)
            if _DEBUG and not _capturing:
                num_blocks = kv_cache.shape[0]
                block_size = kv_cache.shape[1]
                max_slots = num_blocks * block_size
                bt = _block_table[:B]
                sl = _seq_lens[:B]
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
                _block_table,
                _seq_lens,
                self.scale, self.num_kv_heads,
            )

            output[:padded_B] = attn_out[:padded_B].to(output.dtype)
            if _can_sync():
                torch.cuda.synchronize()
                logger.info("FUSEN_SYNC: decode OK, padded_B=%d, B=%d", padded_B, B)
        else:
            # ---- Mixed prefill+decode or pure prefill (CUDA graph safe) ----
            #
            # "Universal decode" approach: treat EVERY query token as a
            # separate decode request against the KV cache. After storing
            # K/V (above), the cache has all tokens. Each query token at
            # position p within request i should causally attend to KV
            # positions 0..(context_before_prefill + p). We compute
            # per-token pseudo-seq_lens and per-token block_table using
            # pure tensor ops (no Python loops, no .item() calls).
            #
            # This makes the mixed path fully CUDA-graph-safe, enabling
            # AttentionCGSupport.ALWAYS.
            qsl = _query_start_loc
            padded_T = query.shape[0]  # padded num_tokens (graph-fixed shape)

            # Compute per-token request assignment: which request owns each token
            # query_start_loc is [num_reqs+1] (or [padded_num_reqs+1])
            # Use searchsorted: for token t, find i such that qsl[i] <= t < qsl[i+1]
            token_positions = torch.arange(
                padded_T, device=query.device, dtype=qsl.dtype)
            # searchsorted with right=True: returns index of first element
            # strictly greater than the value. For qsl_right = qsl[1:],
            # this gives the request index that owns each token.
            qsl_right = qsl[1:]  # [num_reqs] or [padded_reqs]
            token_request_ids = torch.searchsorted(
                qsl_right, token_positions, right=True)
            # Clamp to valid request range (padded tokens get last request id
            # but their seq_lens will be 0, producing zero output)
            max_req_idx = qsl_right.shape[0] - 1
            token_request_ids = token_request_ids.clamp(max=max_req_idx)

            # Per-request query lengths: qsl[i+1] - qsl[i]
            query_lens = qsl_right - qsl[:qsl_right.shape[0]]  # [num_reqs]

            # Position of each token within its request (0-indexed)
            position_in_request = token_positions - qsl[token_request_ids]

            # Per-token pseudo seq_lens for causal attention:
            # token at position p in request i should see:
            #   (seq_lens[i] - query_lens[i]) + (p + 1)
            # = context_len_before_this_batch + position_in_prefill + 1
            per_req_seq = _seq_lens[token_request_ids]
            per_req_qlen = query_lens[token_request_ids]
            pseudo_seq_lens = per_req_seq - per_req_qlen + position_in_request + 1
            # Clamp: must be >= 0 and must not exceed the request's total
            # seq_len. The min(pseudo, per_req_seq) ensures padded tokens
            # (where seq_lens=0 and query_lens=0 but position > 0) get
            # pseudo_seq_lens=0, preventing reads from invalid block_table
            # entries. The decode kernel handles seq_lens=0 safely
            # (split_start >= split_end early return + zero output).
            pseudo_seq_lens = pseudo_seq_lens.clamp(min=0)
            pseudo_seq_lens = torch.min(pseudo_seq_lens, per_req_seq)

            # Per-token block table: expand from [num_reqs, max_blocks] to
            # [padded_T, max_blocks] by indexing with token_request_ids
            token_block_table = _block_table[token_request_ids]

            # Reshape query for decode kernel: [padded_T, H, D]
            q = query
            if q.ndim == 2:
                q = q.reshape(padded_T, self.num_heads, self.head_size)
            q = q.contiguous()

            if _can_sync():
                logger.info(
                    "FUSEN_SYNC: universal decode path padded_T=%d, B=%d, "
                    "num_tokens=%d, max_query_len=%d",
                    padded_T, B, num_tokens, attn_metadata.max_query_len,
                )

            # Run decode kernel on ALL tokens (both prefill and decode)
            # The kernel reads from the quantized KV cache for each token,
            # using pseudo_seq_lens for causal masking.
            _decode = self._cpp_decode if self._use_cpp_decode else self.decode_fn
            attn_out = _decode(
                q, kv_cache, layer._fusen_scales,
                token_block_table,
                pseudo_seq_lens,
                self.scale, self.num_kv_heads,
            )

            output[:padded_T] = attn_out[:padded_T].to(output.dtype)
            if _can_sync():
                torch.cuda.synchronize()
                logger.info("FUSEN_SYNC: universal decode OK, padded_T=%d", padded_T)

        # STREAM FENCE: Record a CUDA event after all kernels (store + decode)
        # so the NEXT forward() call can wait for them via wait_event().
        # The event-based approach is GPU-side only (~5us) and preserves
        # async scheduling for shared buffer protection.
        #
        # Additionally, synchronize the stream to prevent use-after-free of
        # temporary tensors created during this forward() (e.g., .to() dtype
        # conversions, .contiguous() copies). PyTorch's CUDA caching allocator
        # is stream-aware, but vLLM's async scheduling can cause the CPU to
        # allocate new tensors (for the next step) that reuse memory from
        # temps freed in this step. The sync ensures all GPU work from this
        # layer completes before any memory can be recycled.
        #
        # Combined with metadata cloning (above), this provides complete
        # protection against async scheduling races:
        #   - Cloning: prevents CPU from modifying block_table/seq_lens/slot_mapping
        #     while our kernels read them (CPU-GPU metadata race)
        #   - Sync: prevents memory recycling while kernels are in-flight
        #     (allocator use-after-free race)
        #
        # Skip during CUDA graph capture: sync would break capture, and graphs
        # handle their own ordering.
        if not _capturing:
            self._prev_step_event.record()

        return output
