"""KV Cache quantization spec — describes a KV cache format declaratively."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheSpec:
    """Fully describes a KV cache quantization format.

    From this spec, the kernel generator produces:
    - A Triton decode kernel (dequant + attention)
    - A store function (quantize + pack + scatter)
    - Memory calculations (bytes per token, compression ratio)
    """

    name: str

    # K config
    k_bits: int             # 2, 4, 8, 16
    k_sym_offset: float     # symmetric offset: 0 for int8, 7.5 for int4, 1.5 for int2
    k_scale_block: int      # elements per scale group (16, 32, 64, 0=none)

    # V config
    v_bits: int
    v_sym_offset: float
    v_scale_block: int

    # Derived properties
    @property
    def k_bytes_per_dim(self) -> float:
        return self.k_bits / 8

    @property
    def v_bytes_per_dim(self) -> float:
        return self.v_bits / 8

    def slot_bytes(self, D: int) -> int:
        """Total bytes per token per head in the cache."""
        return int(self.k_bytes_per_dim * D + self.v_bytes_per_dim * D)

    def scale_bytes(self, D: int) -> int:
        """Scale tensor bytes per token per head (FP16 scales)."""
        b = 0
        if self.k_scale_block > 0:
            b += (D // self.k_scale_block) * 2  # fp16
        if self.v_scale_block > 0:
            b += (D // self.v_scale_block) * 2
        return b

    def compression_vs_bf16(self, D: int) -> float:
        """Compression ratio vs BF16 (2 bytes per element, K+V)."""
        bf16_bytes = 4 * D  # 2 bytes K + 2 bytes V per dim
        our_bytes = self.slot_bytes(D) + self.scale_bytes(D)
        return bf16_bytes / our_bytes

    @property
    def k_is_packed(self) -> bool:
        """Whether K values are sub-byte packed (need nibble extraction)."""
        return self.k_bits < 8

    @property
    def v_is_packed(self) -> bool:
        return self.v_bits < 8

    @property
    def k_quant_levels(self) -> int:
        return 2 ** self.k_bits

    @property
    def v_quant_levels(self) -> int:
        return 2 ** self.v_bits

    @property
    def dot_strategy(self) -> str:
        """How to compute QK^T given packed K."""
        if self.k_bits >= 8:
            return "direct"         # tl.dot(q, k)
        elif self.k_bits == 4:
            return "split_even_odd"  # split Q into even/odd, two half-dots
        elif self.k_bits == 2:
            return "split_4way"     # 4 quarter-dots
        else:
            raise ValueError(f"Unsupported k_bits={self.k_bits}")

    @property
    def is_float_format(self) -> bool:
        """Whether values are stored as float (FP8) vs integer codes.

        FP8 formats (E5M2, E4M3) store raw floating-point bytes that just need
        a cast to FP16/BF16, not integer dequant ((code - offset) * scale).
        Detected by: no scale blocks and no symmetric offset.
        """
        return self.k_scale_block == 0 and self.v_scale_block == 0 and self.k_sym_offset == 0

    @property
    def has_scales(self) -> bool:
        return self.k_scale_block > 0 or self.v_scale_block > 0


# Predefined specs for known-good configurations
PREDEFINED_SPECS = {
    "k8v4b16": KVCacheSpec(
        name="k8v4b16",
        k_bits=8, k_sym_offset=127.5, k_scale_block=16,
        v_bits=4, v_sym_offset=7.5, v_scale_block=16,
    ),
    "k8v4b32": KVCacheSpec(
        name="k8v4b32",
        k_bits=8, k_sym_offset=127.5, k_scale_block=32,
        v_bits=4, v_sym_offset=7.5, v_scale_block=32,
    ),
    "k4v4b16": KVCacheSpec(
        name="k4v4b16",
        k_bits=4, k_sym_offset=7.5, k_scale_block=16,
        v_bits=4, v_sym_offset=7.5, v_scale_block=16,
    ),
    "k4v4b32": KVCacheSpec(
        name="k4v4b32",
        k_bits=4, k_sym_offset=7.5, k_scale_block=32,
        v_bits=4, v_sym_offset=7.5, v_scale_block=32,
    ),
    "k4v4b64": KVCacheSpec(
        name="k4v4b64",
        k_bits=4, k_sym_offset=7.5, k_scale_block=64,
        v_bits=4, v_sym_offset=7.5, v_scale_block=64,
    ),
    "k8v8b32": KVCacheSpec(
        name="k8v8b32",
        k_bits=8, k_sym_offset=127.5, k_scale_block=32,
        v_bits=8, v_sym_offset=127.5, v_scale_block=32,
    ),
    "k4v2b16": KVCacheSpec(
        name="k4v2b16",
        k_bits=4, k_sym_offset=7.5, k_scale_block=16,
        v_bits=2, v_sym_offset=1.5, v_scale_block=16,
    ),
    "k4v2b32": KVCacheSpec(
        name="k4v2b32",
        k_bits=4, k_sym_offset=7.5, k_scale_block=32,
        v_bits=2, v_sym_offset=1.5, v_scale_block=32,
    ),
    "k8v2b16": KVCacheSpec(
        name="k8v2b16",
        k_bits=8, k_sym_offset=127.5, k_scale_block=16,
        v_bits=2, v_sym_offset=1.5, v_scale_block=16,
    ),
    # FP8 formats: raw floating-point bytes, no integer dequant needed.
    # The kernel currently uses integer-code dequant ((code - offset) * scale);
    # FP8 support requires a cast-based dequant path (follow-up kernel change).
    "fp8_e5m2": KVCacheSpec(
        name="fp8_e5m2",
        k_bits=8, k_sym_offset=0, k_scale_block=0,
        v_bits=8, v_sym_offset=0, v_scale_block=0,
    ),
    "fp8_e4m3": KVCacheSpec(
        name="fp8_e4m3",
        k_bits=8, k_sym_offset=0, k_scale_block=0,
        v_bits=8, v_sym_offset=0, v_scale_block=0,
    ),
}
