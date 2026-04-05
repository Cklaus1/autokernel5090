# SPDX-License-Identifier: Apache-2.0
"""FusenCache v4 attention backend for vLLM.

FP8 K + FP8 V (same compression as native FP8) plus selective
landmark-based attention for long context.

Dense path: native-speed FP8 attention (no custom decode kernel).
Selective path: landmark scoring + sparse attention for seq_len > threshold.
"""

import math
import os
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

logger = init_logger(__name__)

# v4 selective config
_V4_ENABLED = os.environ.get("FUSENCACHE_V4", "1") == "1"
_V4_HOT_WINDOW = int(os.environ.get("FUSENCACHE_HOT_WINDOW", "1024"))
_V4_CHUNK_SIZE = int(os.environ.get("FUSENCACHE_CHUNK_SIZE", "32"))
_V4_TOP_M = int(os.environ.get("FUSENCACHE_TOP_M", "16"))
_V4_MIN_SEQ = int(os.environ.get("FUSENCACHE_MIN_SEQ", "2048"))


class FusenCacheAttentionBackend(AttentionBackend):
    """FusenCache v4: FP8 KV + selective long-context attention."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["fusen"]

    @staticmethod
    def get_name() -> str:
        return "FUSENCACHE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["FusenCacheAttentionImpl"]:
        return FusenCacheAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FusenCacheMetadataBuilder"]:
        return FusenCacheMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int,
        head_size: int, cache_dtype_str: str = "fusen",
    ) -> tuple[int, ...]:
        # v4: FP8+FP8 = 2D per slot. head_size = D.
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype) -> bool:
        return kv_cache_dtype == "fusen"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0


@dataclass
class FusenCacheMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False


class FusenCacheMetadataBuilder(
        AttentionMetadataBuilder[FusenCacheMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH)

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build_for_cudagraph_capture(self, common_attn_metadata):
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        return FusenCacheMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
        )


class FusenCacheAttentionImpl(AttentionImpl["FusenCacheMetadata"]):
    """FusenCache v4: FP8 cache + selective landmarks.

    Dense: FP8 cast + standard attention (near-native speed).
    Selective: landmark scoring + sparse gather for long context.
    """

    supports_quant_query_input: bool = False

    def __init__(self, num_heads, head_size, scale,
                 num_kv_heads=None, alibi_slopes=None,
                 sliding_window=None, kv_cache_dtype="fusen",
                 logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        self.num_heads = num_heads
        self.head_size = head_size  # = actual head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self._sel_logged = False

    # ------------------------------------------------------------------ #
    #  Store: FP8 cast + landmark tracking                                #
    # ------------------------------------------------------------------ #
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        D = key.shape[-1]
        Hk = self.num_kv_heads
        block_size = kv_cache.shape[1]

        k = key[:N]
        v = value[:N]
        if k.ndim == 2:
            k = k.reshape(N, Hk, D)
            v = v.reshape(N, Hk, D)

        # FP8 cast
        k_fp8 = k.to(torch.float8_e4m3fn)
        v_fp8 = v.to(torch.float8_e4m3fn)

        # Pack: [k_fp8(D) | v_fp8(D)]
        k_bytes = k_fp8.view(torch.uint8)
        v_bytes = v_fp8.view(torch.uint8)
        packed = torch.cat([k_bytes, v_bytes], dim=2)

        safe_slot = slot_mapping.clamp(min=0)
        blk_idx = (safe_slot // block_size).long()
        blk_off = (safe_slot % block_size).long()
        kv_cache[blk_idx, blk_off] = packed

        # v4: Update landmarks (skip during CUDA graph capture)
        if _V4_ENABLED and not torch.cuda.is_current_stream_capturing():
            from vllm.fusencache.v4_selective import LandmarkTracker
            if not hasattr(layer, '_fc_v4_tracker'):
                layer._fc_v4_tracker = LandmarkTracker(
                    config=type('C', (), {
                        'chunk_size': _V4_CHUNK_SIZE})(),
                    device=key.device)
            layer._fc_v4_tracker.update(k, slot_mapping, block_size)

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        D = query.shape[-1]
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(num_tokens, self.num_heads * D,
                                 dtype=query.dtype, device=query.device)
        if attn_metadata is None:
            return output.fill_(0)

        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, D)

        if not attn_metadata.is_prefill:
            # Decode: check if selective attention should be used
            if (_V4_ENABLED
                    and attn_metadata.max_seq_len > _V4_MIN_SEQ
                    and hasattr(layer, '_fc_v4_tracker')
                    and layer._fc_v4_tracker._initialized):
                attn_out = self._decode_selective(
                    q, kv_cache, attn_metadata, layer)
            else:
                attn_out = self._decode_dense(
                    q, kv_cache, attn_metadata)
        else:
            # Prefill: standard attention on uncompressed K/V
            k = key[:N].view(N, self.num_kv_heads, D)
            v = value[:N].view(N, self.num_kv_heads, D)
            attn_out = self._prefill_attention(q, k, v, attn_metadata)

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Prefill: uncompressed attention (graph-safe)                       #
    # ------------------------------------------------------------------ #
    def _prefill_attention(self, query, key, value, attn_metadata):
        N, Hq, D = query.shape
        Hk = key.shape[1]
        use_gqa = (Hk < Hq)

        if (_HAS_FLASH_ATTN and D <= 256
                and attn_metadata.max_query_len == attn_metadata.max_seq_len):
            output = torch.empty(N, Hq, D, device=query.device,
                                 dtype=query.dtype)
            flash_attn_varlen_func(
                q=query, k=key, v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale, causal=True, out=output)
            return output

        # Graph-safe SDPA fallback (no .item() calls)
        output = torch.empty(N, Hq, D, device=query.device,
                             dtype=query.dtype)
        q_t = query.transpose(0, 1)
        k_t = key.transpose(0, 1)
        v_t = value.transpose(0, 1)
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=True, scale=self.scale,
            enable_gqa=use_gqa)
        output[:] = out.transpose(0, 1).to(query.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Dense decode: Triton kernel (graph-safe, no .item())               #
    # ------------------------------------------------------------------ #
    _use_triton = True

    def _decode_dense(self, query, kv_cache, attn_metadata):
        """Dense FP8+FP8 decode via v4 Triton kernel."""
        if self._use_triton:
            try:
                from vllm.v1.attention.ops.triton_fusencache_v4 import (
                    fusencache_v4_decode,
                )
                return fusencache_v4_decode(
                    query=query,
                    kv_cache=kv_cache,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    scale=self.scale,
                    num_kv_heads=self.num_kv_heads,
                )
            except Exception as e:
                logger.warning("v4 Triton decode failed: %s", e)
                self.__class__._use_triton = False

        # Python fallback (not graph-safe due to .item())
        B, Hq, D = query.shape
        Hk = self.num_kv_heads
        block_size = kv_cache.shape[1]
        outputs = []
        for i in range(B):
            seq_len = attn_metadata.seq_lens[i].item()
            if seq_len <= 0:
                outputs.append(torch.zeros(Hq, D, device=query.device,
                                           dtype=query.dtype))
                continue
            pos = torch.arange(seq_len, device=query.device)
            blk_idx = attn_metadata.block_table[i, pos // block_size].long()
            blk_off = (pos % block_size).long()
            slots = kv_cache[blk_idx, blk_off]
            k_fp = slots[:, :, :D].contiguous().view(torch.float8_e4m3fn).float()
            v_fp = slots[:, :, D:2*D].contiguous().view(torch.float8_e4m3fn).float()
            if self.num_kv_groups > 1:
                k_fp = k_fp.repeat_interleave(self.num_kv_groups, dim=1)
                v_fp = v_fp.repeat_interleave(self.num_kv_groups, dim=1)
            qi = query[i].float()
            scores = torch.einsum('hd,shd->hs', qi, k_fp) * self.scale
            attn_w = torch.softmax(scores, dim=-1)
            out_i = torch.einsum('hs,shd->hd', attn_w, v_fp)
            outputs.append(out_i.to(query.dtype))
        return torch.stack(outputs, dim=0)

    # ------------------------------------------------------------------ #
    #  Selective decode: landmark-based sparse attention                   #
    # ------------------------------------------------------------------ #
    def _decode_selective(self, query, kv_cache, attn_metadata, layer):
        """v4 selective: hot window + top-M cold chunks via landmarks."""
        B, Hq, D = query.shape
        Hk = self.num_kv_heads
        block_size = kv_cache.shape[1]
        tracker = layer._fc_v4_tracker
        cs = _V4_CHUNK_SIZE

        total_attended = 0
        total_seq = 0
        outputs = []

        for i in range(B):
            seq_len = attn_metadata.seq_lens[i].item()
            total_seq += seq_len

            if seq_len <= 0:
                outputs.append(torch.zeros(Hq, D, device=query.device,
                                           dtype=query.dtype))
                continue

            # Fallback for short sequences
            if seq_len <= _V4_HOT_WINDOW:
                pos = torch.arange(seq_len, device=query.device)
                all_pos = pos
            else:
                hot_start = seq_len - _V4_HOT_WINDOW
                num_chunks = hot_start // cs
                remainder = hot_start % cs

                # Score landmarks
                selected_cold = torch.tensor(
                    [], dtype=torch.long, device=query.device)
                if num_chunks > 0:
                    chunk_starts = torch.arange(
                        0, num_chunks * cs, cs, device=query.device)
                    cblk = attn_metadata.block_table[
                        i, chunk_starts // block_size].long()
                    coff = (chunk_starts % block_size).long()
                    phys_ids = (cblk * block_size + coff) // cs

                    landmarks = tracker.get_landmarks(phys_ids)
                    if landmarks is not None:
                        if self.num_kv_groups > 1:
                            landmarks = landmarks.repeat_interleave(
                                self.num_kv_groups, dim=1)
                        qi = query[i].float()
                        scores = torch.einsum(
                            'hd,chd->hc', qi, landmarks) * self.scale
                        chunk_scores = scores.max(dim=0).values
                        k = min(_V4_TOP_M, num_chunks)
                        _, top_idx = chunk_scores.topk(k)

                        offsets = torch.arange(cs, device=query.device)
                        selected_cold = (
                            top_idx.unsqueeze(1) * cs
                            + offsets.unsqueeze(0)
                        ).reshape(-1)

                if remainder > 0:
                    rem = torch.arange(
                        num_chunks * cs, hot_start, device=query.device)
                    selected_cold = torch.cat([selected_cold, rem])

                hot_pos = torch.arange(
                    hot_start, seq_len, device=query.device)
                all_pos = torch.cat([selected_cold, hot_pos]).long()
                all_pos, _ = all_pos.sort()

            total_attended += all_pos.shape[0]

            # Gather and attend at selected positions
            blk_idx = attn_metadata.block_table[
                i, all_pos // block_size].long()
            blk_off = (all_pos % block_size).long()
            slots = kv_cache[blk_idx, blk_off]

            k_fp8 = slots[:, :, :D].contiguous().view(torch.float8_e4m3fn)
            k_fp = k_fp8.float()
            v_fp8 = slots[:, :, D:2*D].contiguous().view(torch.float8_e4m3fn)
            v_fp = v_fp8.float()

            if self.num_kv_groups > 1:
                k_fp = k_fp.repeat_interleave(self.num_kv_groups, dim=1)
                v_fp = v_fp.repeat_interleave(self.num_kv_groups, dim=1)

            qi = query[i].float()
            scores = torch.einsum('hd,shd->hs', qi, k_fp) * self.scale
            attn_w = torch.softmax(scores, dim=-1)
            out_i = torch.einsum('hs,shd->hd', attn_w, v_fp)
            outputs.append(out_i.to(query.dtype))

        if not self._sel_logged and total_seq > 0:
            logger.info("FusenCache v4: selective decode, attending %d/%d "
                        "tokens (%.1fx reduction)",
                        total_attended, total_seq,
                        total_seq / max(total_attended, 1))
            self._sel_logged = True

        return torch.stack(outputs, dim=0)
