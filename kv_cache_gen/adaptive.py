"""Adaptive KV cache spec selector — picks optimal specs based on runtime conditions."""

from dataclasses import dataclass, field

from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn, make_store_fn


# ---------------------------------------------------------------------------
# Optimal Triton configs from sweep results
# ---------------------------------------------------------------------------
SLIDING_CONFIG = dict(block_kv=16, block_h=8, num_warps=2, num_kv_splits=32)
GLOBAL_CONFIG = dict(block_kv=16, block_h=8, num_warps=2, num_kv_splits=32)
GLOBAL_CONFIG_B1 = dict(block_kv=16, block_h=8, num_warps=2, num_kv_splits=64)


# ---------------------------------------------------------------------------
# Spec shorthand helpers
# ---------------------------------------------------------------------------
def _spec(name: str) -> KVCacheSpec:
    return PREDEFINED_SPECS[name]


# Priority -> (sliding_spec_name, global_spec_name)
_PRIORITY_DEFAULTS = {
    "throughput": ("k4v4b16", "k4v4b16"),
    "quality":    ("k8v8b32", "k8v4b16"),
    "latency":    ("k8v4b16", "k8v4b16"),
}

# Ordered from least to most compressed — used for VRAM fallback.
_COMPRESSION_ORDER = [
    "k8v8b32",
    "k8v4b16",
    "k8v4b32",
    "k4v4b16",
    "k4v4b32",
    "k4v4b64",
    "k4v2b16",
    "k4v2b32",
    "k8v2b16",
]


@dataclass
class AdaptiveConfig:
    """Selects optimal KV cache spec based on runtime conditions."""

    # Per-layer-type spec assignments
    sliding_spec: KVCacheSpec
    global_spec: KVCacheSpec

    # Triton configs per layer type
    sliding_config: dict = field(default_factory=lambda: dict(SLIDING_CONFIG))
    global_config: dict = field(default_factory=lambda: dict(GLOBAL_CONFIG))


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

def estimate_vram_mb(
    config: AdaptiveConfig,
    batch_size: int,
    num_sliding_layers: int = 25,
    num_global_layers: int = 5,
    sliding_seq_len: int = 1024,
    global_seq_len: int = 8192,
    head_dim_sliding: int = 256,
    head_dim_global: int = 512,
    num_kv_heads_sliding: int = 8,
    num_kv_heads_global: int = 2,
) -> float:
    """Estimate KV cache VRAM in MB for the given configuration.

    Accounts for packed cache bytes + scale tensors for every layer, head,
    token, and batch element.
    """
    sliding_spec = config.sliding_spec
    global_spec = config.global_spec

    # Bytes per token per head for each layer type
    sliding_bytes_per_tok = (
        sliding_spec.slot_bytes(head_dim_sliding)
        + sliding_spec.scale_bytes(head_dim_sliding)
    )
    global_bytes_per_tok = (
        global_spec.slot_bytes(head_dim_global)
        + global_spec.scale_bytes(head_dim_global)
    )

    sliding_total = (
        batch_size
        * num_sliding_layers
        * num_kv_heads_sliding
        * sliding_seq_len
        * sliding_bytes_per_tok
    )
    global_total = (
        batch_size
        * num_global_layers
        * num_kv_heads_global
        * global_seq_len
        * global_bytes_per_tok
    )

    return (sliding_total + global_total) / (1024 * 1024)


# ---------------------------------------------------------------------------
# Spec selector
# ---------------------------------------------------------------------------

def _next_more_compressed(spec_name: str) -> str | None:
    """Return the next more-compressed spec in the ordering, or None."""
    if spec_name not in _COMPRESSION_ORDER:
        return None
    idx = _COMPRESSION_ORDER.index(spec_name)
    if idx + 1 < len(_COMPRESSION_ORDER):
        return _COMPRESSION_ORDER[idx + 1]
    return None


def select_config(
    head_dim_sliding: int = 256,
    head_dim_global: int = 512,
    num_kv_heads_sliding: int = 8,
    num_kv_heads_global: int = 2,
    max_batch_size: int = 128,
    max_seq_len: int = 8192,
    vram_gb: float = 32.0,
    priority: str = "throughput",
) -> AdaptiveConfig:
    """Select optimal KV cache specs based on runtime conditions.

    Args:
        head_dim_sliding: head dimension for sliding-window layers.
        head_dim_global: head dimension for global-attention layers.
        num_kv_heads_sliding: number of KV heads in sliding layers.
        num_kv_heads_global: number of KV heads in global layers.
        max_batch_size: maximum concurrent batch size.
        max_seq_len: maximum sequence length (used for global layers).
        vram_gb: available GPU VRAM in gigabytes.
        priority: one of "throughput", "quality", "latency".

    Returns:
        AdaptiveConfig with spec and Triton config for each layer type.
    """
    if priority not in _PRIORITY_DEFAULTS:
        raise ValueError(
            f"Unknown priority {priority!r}; choose from {list(_PRIORITY_DEFAULTS)}"
        )

    sliding_name, global_name = _PRIORITY_DEFAULTS[priority]

    # Pick Triton config — use higher kv_splits for B=1 on global layers
    global_cfg = dict(GLOBAL_CONFIG_B1 if max_batch_size == 1 else GLOBAL_CONFIG)
    sliding_cfg = dict(SLIDING_CONFIG)

    config = AdaptiveConfig(
        sliding_spec=_spec(sliding_name),
        global_spec=_spec(global_name),
        sliding_config=sliding_cfg,
        global_config=global_cfg,
    )

    # VRAM budget check — sliding windows are capped at 1024 tokens
    vram_budget_mb = vram_gb * 1024
    sliding_seq = min(1024, max_seq_len)

    vram_needed = estimate_vram_mb(
        config,
        batch_size=max_batch_size,
        sliding_seq_len=sliding_seq,
        global_seq_len=max_seq_len,
        head_dim_sliding=head_dim_sliding,
        head_dim_global=head_dim_global,
        num_kv_heads_sliding=num_kv_heads_sliding,
        num_kv_heads_global=num_kv_heads_global,
    )

    # Iteratively downgrade specs until they fit (or we run out of options)
    while vram_needed > vram_budget_mb:
        # Try compressing global first (larger footprint), then sliding
        downgraded = False
        next_global = _next_more_compressed(config.global_spec.name)
        if next_global is not None:
            config.global_spec = _spec(next_global)
            downgraded = True
        else:
            next_sliding = _next_more_compressed(config.sliding_spec.name)
            if next_sliding is not None:
                config.sliding_spec = _spec(next_sliding)
                downgraded = True

        if not downgraded:
            break  # Already at maximum compression

        vram_needed = estimate_vram_mb(
            config,
            batch_size=max_batch_size,
            sliding_seq_len=sliding_seq,
            global_seq_len=max_seq_len,
            head_dim_sliding=head_dim_sliding,
            head_dim_global=head_dim_global,
            num_kv_heads_sliding=num_kv_heads_sliding,
            num_kv_heads_global=num_kv_heads_global,
        )

    return config


# ---------------------------------------------------------------------------
# Convenience: build decode/store functions for each layer type
# ---------------------------------------------------------------------------

def make_layer_functions(config: AdaptiveConfig) -> dict:
    """Return ``{layer_type: (decode_fn, store_fn)}`` for each layer type.

    Keys are ``"sliding"`` and ``"global"``.
    """
    return {
        "sliding": (
            make_decode_fn(config.sliding_spec, **config.sliding_config),
            make_store_fn(config.sliding_spec),
        ),
        "global": (
            make_decode_fn(config.global_spec, **config.global_config),
            make_store_fn(config.global_spec),
        ),
    }
