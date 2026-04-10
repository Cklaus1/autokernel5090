"""AutoKernel v2 optimization plugins."""

from .disable_inductor import DisableInductorPlugin
from .fusencache_kv import FusenCacheKVPlugin
from .ngram_spec_decode import NgramSpecDecodePlugin
from .scheduler_tuning import SchedulerTuningPlugin
from .dp_routing import DPRoutingPlugin

ALL_PLUGINS = [
    DisableInductorPlugin,
    FusenCacheKVPlugin,
    NgramSpecDecodePlugin,
    SchedulerTuningPlugin,
    DPRoutingPlugin,
]

__all__ = [
    "DisableInductorPlugin",
    "FusenCacheKVPlugin",
    "NgramSpecDecodePlugin",
    "SchedulerTuningPlugin",
    "DPRoutingPlugin",
    "ALL_PLUGINS",
]
