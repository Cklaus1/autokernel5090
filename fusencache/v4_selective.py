# SPDX-License-Identifier: Apache-2.0
"""FusenCache v4: Selective attention layer for long-context FP8 KV serving.

NOT a custom attention backend. Instead, hooks into any existing backend
(FP8, FP16, etc.) to add:
1. Write-time chunk landmarks (persistent, no decode-time scan)
2. Selective attention routing for long sequences

Usage:
    # Apply v4 to a running vLLM LLM instance
    from vllm.fusencache.v4_selective import apply_fusencache_v4
    llm = LLM(model=..., kv_cache_dtype='fp8_e4m3')
    apply_fusencache_v4(llm, hot_window=512, chunk_size=32, top_m=16)

Or via environment variable:
    FUSENCACHE_V4=1 FUSENCACHE_HOT_WINDOW=1024 python serve.py
"""

import os
import math
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class FusenCacheV4Config:
    """Configuration for v4 selective attention."""

    def __init__(self,
                 hot_window: int = 1024,
                 chunk_size: int = 32,
                 top_m: int = 16,
                 min_seq_for_selective: int = 2048):
        self.hot_window = hot_window
        self.chunk_size = chunk_size
        self.top_m = top_m
        self.min_seq_for_selective = min_seq_for_selective

    @staticmethod
    def from_env():
        return FusenCacheV4Config(
            hot_window=int(os.environ.get("FUSENCACHE_HOT_WINDOW", "1024")),
            chunk_size=int(os.environ.get("FUSENCACHE_CHUNK_SIZE", "32")),
            top_m=int(os.environ.get("FUSENCACHE_TOP_M", "16")),
            min_seq_for_selective=int(os.environ.get(
                "FUSENCACHE_MIN_SEQ", "2048")),
        )


class LandmarkTracker:
    """Tracks chunk landmarks during KV cache writes.

    Attached to each attention layer as layer._fc_v4_tracker.
    Accumulates FP8 key means per chunk without scanning the cache at decode.
    """

    def __init__(self, config: FusenCacheV4Config, device: torch.device):
        self.config = config
        self.device = device
        self._initialized = False

    def _lazy_init(self, max_slots: int, num_kv_heads: int, head_dim: int):
        """Initialize buffers on first use."""
        if self._initialized:
            return
        cs = self.config.chunk_size
        max_chunks = max_slots // cs + 1

        self.landmarks = torch.zeros(
            max_chunks, num_kv_heads, head_dim,
            dtype=torch.float16, device=self.device)
        self.chunk_sums = torch.zeros(
            max_chunks, num_kv_heads, head_dim,
            dtype=torch.float32, device=self.device)
        self.chunk_counts = torch.zeros(
            max_chunks, num_kv_heads,
            dtype=torch.int32, device=self.device)
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._initialized = True

    def update(self, key: torch.Tensor, slot_mapping: torch.Tensor,
               block_size: int):
        """Update landmarks with new keys.

        Args:
            key: (N, num_kv_heads, head_dim) — fresh keys being stored
            slot_mapping: (N,) — which cache slot each key goes to
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        Hk = key.shape[1] if key.ndim == 3 else key.shape[-2]
        D = key.shape[-1]
        self._lazy_init(
            max_slots=100000,  # will grow if needed
            num_kv_heads=Hk, head_dim=D)

        cs = self.config.chunk_size
        safe_slot = slot_mapping.clamp(min=0).long()

        # Compute chunk IDs from flat slot positions
        # flat_slot = block_idx * block_size + block_offset
        # But we receive slot_mapping which IS the flat slot
        chunk_ids = safe_slot // cs

        # Grow buffers if needed
        max_chunk = chunk_ids.max().item() + 1
        if max_chunk > self.landmarks.shape[0]:
            new_size = max_chunk + 100
            self.landmarks = torch.zeros(
                new_size, Hk, D, dtype=torch.float16, device=self.device)
            old_sums = self.chunk_sums
            self.chunk_sums = torch.zeros(
                new_size, Hk, D, dtype=torch.float32, device=self.device)
            self.chunk_sums[:old_sums.shape[0]] = old_sums
            old_counts = self.chunk_counts
            self.chunk_counts = torch.zeros(
                new_size, Hk, dtype=torch.int32, device=self.device)
            self.chunk_counts[:old_counts.shape[0]] = old_counts

        # Vectorized accumulation
        k_float = key[:N].float().reshape(N, -1)  # (N, Hk*D)
        sums_flat = self.chunk_sums.reshape(
            self.chunk_sums.shape[0], -1)
        sums_flat.index_add_(0, chunk_ids, k_float)

        ones = torch.ones(N, Hk, dtype=torch.int32, device=self.device)
        self.chunk_counts.index_add_(0, chunk_ids, ones)

        # Finalize completed chunks
        complete = self.chunk_counts[:, 0] >= cs
        if complete.any():
            idx = complete.nonzero(as_tuple=True)[0]
            counts = self.chunk_counts[idx, :1].float().unsqueeze(-1)
            self.landmarks[idx] = (
                self.chunk_sums[idx] / counts).half()

    def get_landmarks(self, phys_chunk_ids: torch.Tensor) -> torch.Tensor:
        """Get landmarks for given physical chunk IDs.

        Returns: (num_chunks, num_kv_heads, head_dim) float
        """
        if not self._initialized:
            return None
        return self.landmarks[phys_chunk_ids].float()


def selective_attention(
    query: torch.Tensor,       # (Hq, D)
    kv_cache_k: torch.Tensor,  # KV cache K portion for this request
    kv_cache_v: torch.Tensor,  # KV cache V portion for this request
    seq_len: int,
    tracker: LandmarkTracker,
    config: FusenCacheV4Config,
    block_table: torch.Tensor,  # (max_blocks,) for this request
    block_size: int,
    scale: float,
    num_kv_groups: int,
) -> torch.Tensor:
    """Selective attention: hot window + top-M cold chunks.

    Uses persistent landmarks for chunk scoring — no cold cache scan.
    Gathers selected positions from the NATIVE FP8 cache.

    Returns: (Hq, D)
    """
    Hq, D = query.shape
    device = query.device
    sel = config
    cs = sel.chunk_size

    hot_start = seq_len - sel.hot_window
    num_chunks = hot_start // cs
    remainder = hot_start % cs

    # Score landmarks
    selected_cold = torch.tensor([], dtype=torch.long, device=device)
    if num_chunks > 0 and tracker._initialized:
        # Map sequence chunks to physical chunk IDs
        chunk_starts = torch.arange(
            0, num_chunks * cs, cs, device=device)
        cblk = block_table[chunk_starts // block_size].long()
        coff = (chunk_starts % block_size).long()
        phys_ids = (cblk * block_size + coff) // cs

        landmarks = tracker.get_landmarks(phys_ids)
        if landmarks is not None:
            if num_kv_groups > 1:
                landmarks = landmarks.repeat_interleave(
                    num_kv_groups, dim=1)

            qi = query.float()
            scores = torch.einsum(
                'hd,chd->hc', qi, landmarks) * scale
            chunk_scores = scores.max(dim=0).values
            k = min(sel.top_m, num_chunks)
            _, top_idx = chunk_scores.topk(k)

            offsets = torch.arange(cs, device=device)
            selected_cold = (
                top_idx.unsqueeze(1) * cs + offsets.unsqueeze(0)
            ).reshape(-1)

    # Remainder
    if remainder > 0:
        rem = torch.arange(num_chunks * cs, hot_start, device=device)
        selected_cold = torch.cat([selected_cold, rem])

    # Hot window
    hot_pos = torch.arange(hot_start, seq_len, device=device)
    all_pos = torch.cat([selected_cold, hot_pos]).long()
    all_pos, _ = all_pos.sort()

    # Gather K and V from native cache at selected positions
    blk_idx = block_table[all_pos // block_size].long()
    blk_off = (all_pos % block_size).long()

    # K cache: (num_blocks, block_size, Hk, D) — native dtype (FP8 or FP16)
    k_selected = kv_cache_k[blk_idx, blk_off]  # (S, Hk, D)
    v_selected = kv_cache_v[blk_idx, blk_off]  # (S, Hk, D)

    # Convert to float if needed
    if k_selected.dtype in (torch.float8_e4m3fn, torch.uint8):
        if k_selected.dtype == torch.uint8:
            k_selected = k_selected.view(torch.float8_e4m3fn)
        k_selected = k_selected.float()
    else:
        k_selected = k_selected.float()

    if v_selected.dtype in (torch.float8_e4m3fn, torch.uint8):
        if v_selected.dtype == torch.uint8:
            v_selected = v_selected.view(torch.float8_e4m3fn)
        v_selected = v_selected.float()
    else:
        v_selected = v_selected.float()

    # GQA expand
    if num_kv_groups > 1:
        k_selected = k_selected.repeat_interleave(num_kv_groups, dim=1)
        v_selected = v_selected.repeat_interleave(num_kv_groups, dim=1)

    # Attention over selected positions
    scores = torch.einsum('hd,shd->hs', query.float(), k_selected) * scale
    attn_w = torch.softmax(scores, dim=-1)
    out = torch.einsum('hs,shd->hd', attn_w, v_selected)

    return out.to(query.dtype)
