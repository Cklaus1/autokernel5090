# SPDX-License-Identifier: Apache-2.0
"""FusenCache v4c: Monkey-patch for subprocess injection.

Patches TritonAttentionImpl to add landmark tracking + selective attention.
Activated by FUSENCACHE_V4C=1 environment variable.

Import this module BEFORE the engine starts to apply the patch.
Or add to vllm/__init__.py for automatic activation.
"""

import os
import math
import functools
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED = os.environ.get("FUSENCACHE_V4C", "0") == "1"
_HOT_WINDOW = int(os.environ.get("FUSENCACHE_HOT_WINDOW", "1024"))
_CHUNK_SIZE = int(os.environ.get("FUSENCACHE_CHUNK_SIZE", "32"))
_TOP_M = int(os.environ.get("FUSENCACHE_TOP_M", "16"))
_MIN_SEQ = int(os.environ.get("FUSENCACHE_MIN_SEQ", "2048"))


class _Tracker:
    """Lightweight per-layer chunk landmark tracker."""

    def __init__(self, cs, device):
        self.cs = cs
        self.device = device
        self.ok = False

    def init(self, Hk, D):
        if self.ok:
            return
        n = 4096
        self.lm = torch.zeros(n, Hk, D, dtype=torch.float16, device=self.device)
        self.sums = torch.zeros(n, Hk, D, dtype=torch.float32, device=self.device)
        self.counts = torch.zeros(n, Hk, dtype=torch.int32, device=self.device)
        self.Hk, self.D = Hk, D
        self.ok = True

    def update(self, keys, slot_mapping):
        N = slot_mapping.shape[0]
        if N <= 0 or keys.ndim != 3:
            return
        Hk, D = keys.shape[1], keys.shape[2]
        self.init(Hk, D)
        cs = self.cs
        cids = (slot_mapping.clamp(min=0) // cs).long()
        mx = cids.max().item() + 1
        if mx > self.lm.shape[0]:
            self._grow(mx + 256)
        self.sums.reshape(-1, Hk * D).index_add_(
            0, cids, keys[:N].float().reshape(N, -1))
        self.counts.index_add_(
            0, cids, torch.ones(N, Hk, dtype=torch.int32, device=self.device))
        done = self.counts[:, 0] >= cs
        if done.any():
            idx = done.nonzero(as_tuple=True)[0]
            c = self.counts[idx, :1].float().unsqueeze(-1)
            self.lm[idx] = (self.sums[idx] / c).half()

    def _grow(self, sz):
        old = self.lm.shape[0]
        Hk, D = self.Hk, self.D
        for a, dt in [('lm', torch.float16), ('sums', torch.float32)]:
            o = getattr(self, a)
            n = torch.zeros(sz, Hk, D, dtype=dt, device=self.device)
            n[:old] = o
            setattr(self, a, n)
        o = self.counts
        n = torch.zeros(sz, Hk, dtype=torch.int32, device=self.device)
        n[:old] = o
        self.counts = n


def _get_tracker(layer, device):
    """Get or create tracker on the layer."""
    if not hasattr(layer, '_fc_tracker'):
        layer._fc_tracker = _Tracker(_CHUNK_SIZE, device)
    return layer._fc_tracker


def _patched_do_kv_cache_update(original_fn, self, layer, key, value,
                                 kv_cache, slot_mapping):
    """Wrapped do_kv_cache_update: native store + landmark tracking."""
    # Native store (untouched)
    result = original_fn(self, layer, key, value, kv_cache, slot_mapping)

    # Track landmarks (outside graph capture only)
    if not torch.cuda.is_current_stream_capturing():
        N = slot_mapping.shape[0]
        if N > 0 and key.ndim == 3:
            tracker = _get_tracker(layer, key.device)
            tracker.update(key[:N], slot_mapping[:N])

    return result


def _patched_forward(original_fn, self, layer, query, key, value,
                     kv_cache, attn_metadata, output=None,
                     output_scale=None, output_block_scale=None):
    """Wrapped forward: native for short, selective for long."""

    # Always use native for: prefill, None metadata, short context,
    # graph capture, or when tracker not ready
    if (attn_metadata is None
            or attn_metadata.is_prefill
            or attn_metadata.max_seq_len <= _MIN_SEQ
            or not hasattr(layer, '_fc_tracker')
            or not layer._fc_tracker.ok):
        return original_fn(self, layer, query, key, value, kv_cache,
                           attn_metadata, output, output_scale,
                           output_block_scale)

    # Check if selective would help
    num_chunks = ((attn_metadata.max_seq_len - _HOT_WINDOW) // _CHUNK_SIZE)
    if num_chunks <= _TOP_M:
        return original_fn(self, layer, query, key, value, kv_cache,
                           attn_metadata, output, output_scale,
                           output_block_scale)

    # === Selective path ===
    assert output is not None
    N = attn_metadata.num_actual_tokens
    B = attn_metadata.seq_lens.shape[0]
    Hq, D = query.shape[1], query.shape[2]

    # Get native cache layout: (num_blocks, 2, block_size, Hk, D)
    key_cache, value_cache = kv_cache.unbind(1)
    fp8_dtype = torch.float8_e4m3fn
    if key_cache.dtype == torch.uint8:
        key_cache = key_cache.view(fp8_dtype)
        value_cache = value_cache.view(fp8_dtype)

    Hk = key_cache.shape[2]
    block_size = key_cache.shape[1]
    num_kv_groups = Hq // Hk
    tracker = layer._fc_tracker
    device = query.device

    total_attended = 0
    total_seq = 0
    _logged = not hasattr(_patched_forward, '_sel_logged')

    for i in range(B):
        seq_len = attn_metadata.seq_lens[i].item()
        total_seq += seq_len

        if seq_len <= _HOT_WINDOW:
            # Shouldn't reach here (checked above), but safety
            continue

        hot_start = seq_len - _HOT_WINDOW
        nc = hot_start // _CHUNK_SIZE
        rem = hot_start % _CHUNK_SIZE

        # Score landmarks
        selected = torch.tensor([], dtype=torch.long, device=device)
        if nc > 0:
            cs_starts = torch.arange(0, nc * _CHUNK_SIZE, _CHUNK_SIZE,
                                     device=device)
            bt = attn_metadata.block_table[i]
            cblk = bt[cs_starts // block_size].long()
            coff = (cs_starts % block_size).long()
            pids = (cblk * block_size + coff) // _CHUNK_SIZE

            lm = tracker.lm[pids].float()
            if num_kv_groups > 1:
                lm = lm.repeat_interleave(num_kv_groups, dim=1)

            qi = query[i].float()
            sc = torch.einsum('hd,chd->hc', qi, lm) * self.scale
            cs_sc = sc.max(dim=0).values
            k = min(_TOP_M, nc)
            _, top = cs_sc.topk(k)

            offs = torch.arange(_CHUNK_SIZE, device=device)
            selected = (top.unsqueeze(1) * _CHUNK_SIZE
                        + offs.unsqueeze(0)).reshape(-1)

        if rem > 0:
            r = torch.arange(nc * _CHUNK_SIZE, hot_start, device=device)
            selected = torch.cat([selected, r])

        hot = torch.arange(hot_start, seq_len, device=device)
        all_pos = torch.cat([selected, hot]).long()
        all_pos, _ = all_pos.sort()
        total_attended += all_pos.shape[0]

        # Gather from native FP8 cache
        bt = attn_metadata.block_table[i]
        bi = bt[all_pos // block_size].long()
        bo = (all_pos % block_size).long()

        k_sel = key_cache[bi, bo].float()
        v_sel = value_cache[bi, bo].float()
        if num_kv_groups > 1:
            k_sel = k_sel.repeat_interleave(num_kv_groups, dim=1)
            v_sel = v_sel.repeat_interleave(num_kv_groups, dim=1)

        scores = torch.einsum('hd,shd->hs', qi, k_sel) * self.scale
        attn_w = torch.softmax(scores, dim=-1)
        out_i = torch.einsum('hs,shd->hd', attn_w, v_sel)

        if output.ndim == 3:
            output[i] = out_i.to(output.dtype)
        else:
            output[i] = out_i.reshape(-1).to(output.dtype)

    if _logged and total_seq > 0:
        logger.info("FusenCache v4c: SELECTIVE attending %d/%d "
                    "tokens (%.1fx reduction)",
                    total_attended, total_seq,
                    total_seq / max(total_attended, 1))
        _patched_forward._sel_logged = True

    return output


def apply_patch():
    """Apply the monkey-patch to TritonAttentionImpl."""
    if not _ENABLED:
        return

    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

    orig_store = TritonAttentionImpl.do_kv_cache_update
    orig_forward = TritonAttentionImpl.forward

    # Only patch forward — leave store untouched for graph safety.
    # Landmarks computed lazily from KV cache on first selective decode.
    def new_forward(self, layer, query, key, value, kv_cache,
                    attn_metadata, output=None, output_scale=None,
                    output_block_scale=None):
        return _patched_forward(
            orig_forward, self, layer, query, key, value, kv_cache,
            attn_metadata, output, output_scale, output_block_scale)

    TritonAttentionImpl.forward = new_forward

    logger.info("FusenCache v4c: Patched TritonAttentionImpl "
                "(hot=%d, chunk=%d, top_m=%d, min_seq=%d)",
                _HOT_WINDOW, _CHUNK_SIZE, _TOP_M, _MIN_SEQ)


# Auto-apply on import if enabled
# DISABLED: monkey-patching breaks CUDA graph capture
# apply_patch()
