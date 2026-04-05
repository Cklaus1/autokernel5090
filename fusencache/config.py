# SPDX-License-Identifier: Apache-2.0
"""FusenCache configuration."""

import os
from dataclasses import dataclass


@dataclass
class FusenCacheConfig:
    """Configuration for FusenCache KV-cache compression.

    v4 layout per token per head:
        [k_fp8 (D bytes) | v_fp8 (D bytes)]

    Total slot size = 2 * head_dim bytes.
    Compression: 2.0x vs FP16.
    Same compression as native FP8 but with selective landmarks for long context.
    """
    head_dim: int

    @property
    def slot_size(self) -> int:
        """Total bytes per head per position (FP8 K + FP8 V)."""
        return 2 * self.head_dim

    @property
    def effective_head_size(self) -> int:
        """Head size for FullAttentionSpec with dtype=uint8.

        v4: slot = 2D (same as FP8 native). effective = D.
        """
        return self.head_dim

    @staticmethod
    def from_cache_dtype(cache_dtype: str,
                         head_dim: int) -> "FusenCacheConfig":
        if cache_dtype != "fusen":
            raise ValueError(f"Unknown FusenCache dtype: {cache_dtype}")
        return FusenCacheConfig(head_dim=head_dim)


@dataclass
class FusenCacheSelectiveConfig:
    """Configuration for v3 selective attention.

    Only attend to hot_window recent tokens + top_m cold chunks
    instead of the full sequence. Reduces decode work from O(T)
    to O(hot_window + top_m * chunk_size).
    """
    enabled: bool = False
    hot_window: int = 512
    chunk_size: int = 32
    top_m: int = 16

    @staticmethod
    def from_env() -> "FusenCacheSelectiveConfig":
        enabled = os.environ.get("FUSEN_SELECTIVE", "0") == "1"
        hot_window = int(os.environ.get("FUSEN_HOT_WINDOW", "512"))
        chunk_size = int(os.environ.get("FUSEN_CHUNK_SIZE", "32"))
        top_m = int(os.environ.get("FUSEN_TOP_M", "16"))
        return FusenCacheSelectiveConfig(
            enabled=enabled,
            hot_window=hot_window,
            chunk_size=chunk_size,
            top_m=top_m,
        )
