"""AutoKernel v2 optimization plugins."""

from .disable_inductor import DisableInductorPlugin
from .dp_routing import DPRoutingPlugin
from .fp8_decode_backend import FP8DecodeBackendPlugin
from .fusencache_kv import FusenCacheKVPlugin
from .ngram_spec_decode import NgramSpecDecodePlugin
from .scheduler_tuning import SchedulerTuningPlugin

ALL_PLUGINS = [
    DisableInductorPlugin,
    FP8DecodeBackendPlugin,
    FusenCacheKVPlugin,
    NgramSpecDecodePlugin,
    SchedulerTuningPlugin,
    DPRoutingPlugin,
]

__all__ = [
    "DisableInductorPlugin",
    "FP8DecodeBackendPlugin",
    "FusenCacheKVPlugin",
    "NgramSpecDecodePlugin",
    "SchedulerTuningPlugin",
    "DPRoutingPlugin",
    "ALL_PLUGINS",
]
