# SPDX-License-Identifier: Apache-2.0
"""FusenCache configuration."""

import os
from dataclasses import dataclass


@dataclass
class FusenCacheConfig:
    """FusenCache v5: K=8bit V=4bit per-block-16 symmetric.

    Per token per head:
        [k_int8 (D bytes) | v_4bit (D/2 bytes)]

    Per-block-16 FP16 scales in separate tensor.
    Slot = 3D/2 bytes. Compression: 2.3x vs FP16 (with scales).
    0.5% K error, 97% attention top-1 — better than FP8.
    """
    head_dim: int

    @property
    def k_packed_size(self) -> int:
        return self.head_dim  # int8 = 1 byte per element

    @property
    def v_packed_size(self) -> int:
        return self.head_dim // 2  # 4 bits = 2 per byte = 4 per byte

    @property
    def slot_size(self) -> int:
        return self.k_packed_size + self.v_packed_size  # 5D/4

    @property
    def effective_head_size(self) -> int:
        return self.slot_size // 2  # 5D/8

    @staticmethod
    def from_cache_dtype(cache_dtype, head_dim):
        if cache_dtype != "fusen":
            raise ValueError(f"Unknown dtype: {cache_dtype}")
        return FusenCacheConfig(head_dim=head_dim)


@dataclass
class FusenCacheSelectiveConfig:
    enabled: bool = False
    hot_window: int = 512
    chunk_size: int = 32
    top_m: int = 16

    @staticmethod
    def from_env():
        return FusenCacheSelectiveConfig(
            enabled=os.environ.get("FUSEN_SELECTIVE", "0") == "1",
            hot_window=int(os.environ.get("FUSEN_HOT_WINDOW", "512")),
            chunk_size=int(os.environ.get("FUSEN_CHUNK_SIZE", "32")),
            top_m=int(os.environ.get("FUSEN_TOP_M", "16")),
        )
