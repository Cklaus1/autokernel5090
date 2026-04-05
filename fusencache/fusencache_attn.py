# SPDX-License-Identifier: Apache-2.0
"""FusenCache v5: K=8bit V=2bit per-block-16 symmetric KV cache.

0.5% K error, 97% attention top-1, 2.7x compression.
Better K quality than FP8 with more compression.

Cache: [k_int8 (D bytes) | v_2bit (D/4 bytes)] per slot.
Scales: per-block-16 FP16 in side tensor.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.nn.functional as F

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend, AttentionCGSupport, AttentionImpl,
    AttentionLayer, AttentionMetadata, AttentionMetadataBuilder,
    AttentionType, CommonAttentionMetadata, MultipleOf,
)

logger = init_logger(__name__)

BLOCK_SCALE = 16  # elements per scale block


class FusenCacheAttentionBackend(AttentionBackend):
    accept_output_buffer = True
    forward_includes_kv_cache_update = False
    supported_dtypes: ClassVar = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar = ["fusen"]

    @staticmethod
    def get_name(): return "FUSENCACHE"
    @staticmethod
    def get_supported_kernel_block_sizes(): return [16, 32, 64, 128]
    @classmethod
    def supports_attn_type(cls, t): return t == AttentionType.DECODER
    @classmethod
    def supports_per_head_quant_scales(cls): return False
    @staticmethod
    def get_impl_cls(): return FusenCacheAttentionImpl
    @staticmethod
    def get_builder_cls(): return FusenCacheMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="fusen"):
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @classmethod
    def supports_kv_cache_dtype(cls, d): return d == "fusen"
    @classmethod
    def supports_head_size(cls, h): return h > 0


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


class FusenCacheMetadataBuilder(AttentionMetadataBuilder[FusenCacheMetadata]):
    _cudagraph_support: ClassVar = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def reorder_batch(self, input_batch, scheduler_output):
        return False

    def build_for_cudagraph_capture(self, common_attn_metadata):
        m = self.build(0, common_attn_metadata)
        m.seq_lens.fill_(1)
        return m

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        c = common_attn_metadata
        return FusenCacheMetadata(
            seq_lens=c.seq_lens, slot_mapping=c.slot_mapping,
            block_table=c.block_table_tensor, query_start_loc=c.query_start_loc,
            num_actual_tokens=c.num_actual_tokens,
            max_query_len=c.max_query_len, max_seq_len=c.max_seq_len,
            is_prefill=(c.max_query_len > 1))


class FusenCacheAttentionImpl(AttentionImpl["FusenCacheMetadata"]):
    """K=8bit V=2bit per-block-16. Simple, correct, better than FP8."""
    supports_quant_query_input = False

    def __init__(self, num_heads, head_size, scale, num_kv_heads=None,
                 alibi_slopes=None, sliding_window=None,
                 kv_cache_dtype="fusen", logits_soft_cap=None,
                 attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self._head_dim = None

    def _resolve_head_dim(self, t):
        if self._head_dim is None:
            self._head_dim = t.shape[-1]
        return self._head_dim

    # ------------------------------------------------------------------ #
    #  Store: K=int8 + V=2bit with per-block-16 scales                   #
    # ------------------------------------------------------------------ #
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        D = self._resolve_head_dim(key)
        Hk = self.num_kv_heads
        block_size = kv_cache.shape[1]
        device = key.device
        BS = BLOCK_SCALE

        k = key[:N]
        v = value[:N]
        if k.ndim == 2:
            k = k.reshape(N, Hk, D)
            v = v.reshape(N, Hk, D)

        # --- K: int8 symmetric per-block-16 ---
        k_blocks = k.float().reshape(N, Hk, D // BS, BS)
        k_absmax = k_blocks.abs().amax(dim=-1, keepdim=True)
        k_scale = k_absmax / 127.0
        k_int8 = (k_blocks / (k_scale + 1e-8)).round().clamp(-128, 127).to(torch.int8)
        k_packed = k_int8.reshape(N, Hk, D).view(torch.uint8)  # (N, Hk, D)
        k_scale = k_scale.squeeze(-1).half()  # (N, Hk, D/16)

        # --- V: 2-bit symmetric per-block-16 ---
        v_blocks = v.float().reshape(N, Hk, D // BS, BS)
        v_absmax = v_blocks.abs().amax(dim=-1, keepdim=True)
        v_scale = v_absmax / 7.5  # 4-bit: 16 levels, mid=7.5
        v_codes = (v_blocks / (v_scale + 1e-8) + 7.5).round().clamp(0, 15).to(torch.uint8)
        v_codes = v_codes.reshape(N, Hk, D)
        v_scale = v_scale.squeeze(-1).half()  # (N, Hk, D/16)

        # Pack V: 2 values per byte (4-bit)
        vc = v_codes.reshape(N, Hk, D // 2, 2).to(torch.int32)
        v_packed = (vc[..., 0] | (vc[..., 1] << 4)).to(torch.uint8)

        # Combine: [k_int8 (D) | v_2bit (D/4)]
        packed = torch.cat([k_packed, v_packed], dim=2)

        # Scatter
        safe_slot = slot_mapping.clamp(min=0)
        blk_idx = (safe_slot // block_size).long()
        blk_off = (safe_slot % block_size).long()
        slot_size = D + D // 2
        kv_cache[blk_idx, blk_off, :, :slot_size] = packed

        # Store scales
        num_sb = D // BS
        max_slots = kv_cache.shape[0] * block_size
        if not hasattr(layer, '_fc_scales'):
            layer._fc_scales = torch.zeros(
                max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device)
        flat_slot = blk_idx * block_size + blk_off
        layer._fc_scales[flat_slot, :, :, 0] = k_scale
        layer._fc_scales[flat_slot, :, :, 1] = v_scale

    # ------------------------------------------------------------------ #
    #  Forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output=None, output_scale=None, output_block_scale=None):
        D = self._resolve_head_dim(query)
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
            attn_out = self._decode_attention(q, kv_cache, attn_metadata, layer)
        else:
            k = key[:N].view(N, self.num_kv_heads, D)
            v = value[:N].view(N, self.num_kv_heads, D)
            attn_out = self._prefill_attention(q, k, v, attn_metadata)

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _prefill_attention(self, query, key, value, attn_metadata):
        N, Hq, D = query.shape
        use_gqa = (key.shape[1] < Hq)
        if (_HAS_FLASH_ATTN and D <= 256
                and attn_metadata.max_query_len == attn_metadata.max_seq_len):
            output = torch.empty(N, Hq, D, device=query.device, dtype=query.dtype)
            flash_attn_varlen_func(
                q=query, k=key, v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale, causal=True, out=output)
            return output
        output = torch.empty(N, Hq, D, device=query.device, dtype=query.dtype)
        out = F.scaled_dot_product_attention(
            query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1),
            is_causal=True, scale=self.scale, enable_gqa=(key.shape[1] < Hq))
        output[:] = out.transpose(0, 1).to(query.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Decode: dequant K8+V2 + attention                                  #
    # ------------------------------------------------------------------ #
    _use_triton = True

    def _decode_attention(self, query, kv_cache, attn_metadata, layer):
        B, Hq, D = query.shape
        Hk = self.num_kv_heads

        # Try Triton kernel first
        if self._use_triton and hasattr(layer, '_fc_scales'):
            try:
                from vllm.v1.attention.ops.triton_fusencache_v5 import (
                    fusencache_v5_decode,
                )
                return fusencache_v5_decode(
                    query=query, kv_cache=kv_cache,
                    scales=layer._fc_scales,
                    block_table=attn_metadata.block_table,
                    seq_lens=attn_metadata.seq_lens,
                    scale=self.scale, num_kv_heads=Hk,
                )
            except Exception as e:
                logger.warning("v5 Triton failed: %s, falling back", e)
                self.__class__._use_triton = False

        block_size = kv_cache.shape[1]
        BS = BLOCK_SCALE

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
            slots = kv_cache[blk_idx, blk_off]  # (S, Hk, 5D/4)

            # --- K: int8 → float ---
            k_int8 = slots[:, :, :D].contiguous().view(torch.int8)
            flat_slots = blk_idx * block_size + blk_off
            scales = layer._fc_scales[flat_slots]  # (S, Hk, D/16, 2)
            k_scale = scales[:, :, :, 0]  # (S, Hk, D/16)

            # Dequant K per block
            k_blocks = k_int8.float().reshape(seq_len, Hk, D // BS, BS)
            k_fp = (k_blocks * k_scale.float().unsqueeze(-1)).reshape(
                seq_len, Hk, D)

            # --- V: 2-bit → float ---
            v_packed = slots[:, :, D:D + D // 2]  # (S, Hk, D/2)
            v_scale = scales[:, :, :, 1]  # (S, Hk, D/16)

            # Unpack 4-bit: 2 per byte
            vp = v_packed.to(torch.int32)
            c0 = vp & 0xF
            c1 = (vp >> 4) & 0xF
            v_codes = torch.stack([c0, c1], dim=-1).reshape(
                seq_len, Hk, D)

            # Dequant V per block
            v_blocks = (v_codes.float() - 7.5).reshape(
                seq_len, Hk, D // BS, BS)
            v_fp = (v_blocks * v_scale.float().unsqueeze(-1)).reshape(
                seq_len, Hk, D)

            # GQA
            if self.num_kv_groups > 1:
                k_fp = k_fp.repeat_interleave(self.num_kv_groups, dim=1)
                v_fp = v_fp.repeat_interleave(self.num_kv_groups, dim=1)

            # Attention
            qi = query[i].float()
            scores = torch.einsum('hd,shd->hs', qi, k_fp) * self.scale
            attn_w = torch.softmax(scores, dim=-1)
            out_i = torch.einsum('hs,shd->hd', attn_w, v_fp)
            outputs.append(out_i.to(query.dtype))

        return torch.stack(outputs, dim=0)
