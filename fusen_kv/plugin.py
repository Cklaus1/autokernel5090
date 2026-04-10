"""vLLM plugin entry point. Registers FusenKV as a custom attention backend.

This module is loaded by vLLM's plugin system via the entry_points mechanism:

    [project.entry-points."vllm.general_plugins"]
    fusen_kv = "fusen_kv.plugin:register"

When vLLM starts, it calls load_general_plugins() which discovers and
executes this register() function. After that, users can specify:

    --kv-cache-dtype fusen     # auto-select best spec
    --kv-cache-dtype k4v4      # specific format
    --kv-cache-dtype k8v4b32   # fully specified
"""

import logging

logger = logging.getLogger(__name__)

# All dtype strings FusenKV handles
FUSEN_DTYPES = [
    "fusen",
    "k4v4", "k4v4b16", "k4v4b32", "k4v4b64",
    "k8v4", "k8v4b16", "k8v4b32",
    "k8v8", "k8v8b32",
    "k4v2", "k4v2b16", "k4v2b32",
    "k8v2", "k8v2b16",
    "k4v4kb64vb64", "k8v4kb32vb32", "k8v4kb16vb16",
    "int4", "int8",
]


def _patch_cache_dtype():
    """Monkey-patch vLLM's CacheDType Literal to accept FusenKV dtype strings.

    vLLM uses a Literal type for kv_cache_dtype validation. Without this patch,
    pydantic rejects our custom dtype strings at config parse time (before our
    backend code even runs).

    This patches three things:
    1. The CacheDType type annotation (for pydantic validation)
    2. The CacheConfig field default choices
    3. The field validator to not reject our strings
    """
    try:
        import vllm.config.cache as cache_mod

        # Step 1: Expand the CacheDType Literal
        # Literal types are immutable, so we create a new one that includes
        # both the original types and our custom ones
        from typing import get_args, Literal
        original_types = list(get_args(cache_mod.CacheDType))
        all_types = original_types + [d for d in FUSEN_DTYPES if d not in original_types]

        # Build new Literal dynamically
        new_cache_dtype = Literal[tuple(all_types)]  # type: ignore
        cache_mod.CacheDType = new_cache_dtype

        # Step 2: Patch the CacheConfig class to accept our dtypes
        # The pydantic model uses the Literal for validation. We need to
        # update the field's annotation on the class.
        config_cls = cache_mod.CacheConfig
        if hasattr(config_cls, '__annotations__'):
            config_cls.__annotations__['cache_dtype'] = new_cache_dtype

        # Step 3: Patch CacheConfig.__init__ to bypass pydantic Literal validation
        # for FusenKV dtypes. Pydantic v2 compiles validators at class definition
        # time, so updating annotations alone doesn't work. We wrap __init__ to
        # temporarily use 'auto' during pydantic validation, then restore our dtype.
        original_init = config_cls.__init__

        def _patched_init(self, *args, **kwargs):
            cache_dtype = kwargs.get('cache_dtype', None)
            if cache_dtype in FUSEN_DTYPES:
                kwargs['cache_dtype'] = 'auto'
                original_init(self, *args, **kwargs)
                object.__setattr__(self, 'cache_dtype', cache_dtype)
            else:
                original_init(self, *args, **kwargs)

        config_cls.__init__ = _patched_init

        # Step 4: Patch the validator to pass through our dtypes
        if hasattr(cache_mod.CacheConfig, '_validate_cache_dtype'):
            original_validate = cache_mod.CacheConfig._validate_cache_dtype.__func__

            @classmethod  # type: ignore
            def _patched_validate(cls, cache_dtype):
                if cache_dtype in FUSEN_DTYPES:
                    logger.info("FusenKV: using '%s' KV cache format", cache_dtype)
                    return cache_dtype
                return original_validate(cls, cache_dtype)

            cache_mod.CacheConfig._validate_cache_dtype = _patched_validate

        # Step 5: Patch arg_utils to accept our strings in CLI
        try:
            import vllm.engine.arg_utils as arg_mod
            # The EngineArgs class mirrors CacheConfig's type
            if hasattr(arg_mod, 'EngineArgs') and hasattr(arg_mod.EngineArgs, '__annotations__'):
                arg_mod.EngineArgs.__annotations__['kv_cache_dtype'] = new_cache_dtype
        except Exception:
            pass

        logger.info("FusenKV: patched CacheDType to accept %d custom dtypes", len(FUSEN_DTYPES))

    except Exception as e:
        logger.warning("FusenKV: CacheDType patch failed: %s. "
                       "Users may need to bypass validation manually.", e)


def _patch_backend_selection():
    """Patch vLLM's backend selection to route FusenKV dtypes to CUSTOM backend.

    When vLLM selects an attention backend, it checks each backend's
    supported_kv_cache_dtypes. Our CUSTOM backend declares support for
    our dtypes. But vLLM's selection logic may not check CUSTOM by default.

    This patch ensures FusenKV dtypes are routed to our backend.
    """
    try:
        from vllm.platforms.cuda import CudaPlatform

        original_get_attn = CudaPlatform.get_attn_backend_cls

        @classmethod  # type: ignore
        def _patched_get_attn(cls, selected_backend, attn_selector_config,
                              num_heads=None):
            # If the user specified a FusenKV dtype, force CUSTOM backend
            kv_dtype = getattr(attn_selector_config, 'kv_cache_dtype', None)
            if kv_dtype in FUSEN_DTYPES:
                from vllm.v1.attention.backends.registry import AttentionBackendEnum
                backend = AttentionBackendEnum.CUSTOM
                logger.info("FusenKV: routing kv_cache_dtype='%s' to CUSTOM backend",
                            kv_dtype)
                return backend.get_path()

            return original_get_attn.__func__(cls, selected_backend,
                                               attn_selector_config, num_heads)

        CudaPlatform.get_attn_backend_cls = _patched_get_attn
        logger.info("FusenKV: patched backend selection for CUDA platform")

    except Exception as e:
        logger.warning("FusenKV: backend selection patch failed: %s", e)


def _patch_dtype_mapping():
    """Add FusenKV dtype strings to vLLM's STR_DTYPE_TO_TORCH_DTYPE mapping.

    vLLM uses this mapping in kv_cache_dtype_str_to_dtype() to convert the
    --kv-cache-dtype CLI string to a torch dtype. FusenKV stores cache as
    uint8 packed bytes.
    """
    try:
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        import torch
        for dtype_str in FUSEN_DTYPES:
            if dtype_str not in STR_DTYPE_TO_TORCH_DTYPE:
                STR_DTYPE_TO_TORCH_DTYPE[dtype_str] = torch.uint8
        logger.info("FusenKV: patched STR_DTYPE_TO_TORCH_DTYPE with %d entries",
                     len(FUSEN_DTYPES))
    except Exception as e:
        logger.warning("FusenKV: dtype mapping patch failed: %s", e)


def _patch_kv_cache_spec():
    """Patch vLLM's KV cache spec to use correct page sizes for FusenKV.

    vLLM's real_page_size_bytes formula assumes standard BF16 KV layout:
        2 * block_size * num_kv_heads * head_size * dtype_size
    For FusenKV with sub-byte K+V, the actual bytes per page are much smaller.

    We need two patches:
    1. Override real_page_size_bytes property on AttentionSpec to use correct
       formula when page_size_padded carries our custom marker.
    2. Patch Attention.get_kv_cache_spec to set page_size_padded to the true size.

    The assertion `page_size_padded >= real_page_size` prevents us from just
    setting page_size_padded smaller, so we must fix real_page_size_bytes too.
    """
    try:
        from vllm.v1.kv_cache_interface import (
            AttentionSpec, FullAttentionSpec, SlidingWindowSpec,
        )

        # Patch 1: Override real_page_size_bytes on ALL spec classes.
        # For FusenCache (dtype=uint8, custom kv_cache_dtype), the page formula
        # should use slot_bytes (0.5*D for K + 0.5*D for V for k4v4) instead of
        # the default 2*head_size*dtype_size.
        # We detect FusenCache specs by dtype=uint8 (no standard backend uses uint8).
        import torch as _torch

        # We need slot_bytes(D) for FusenCache. For k4v4: 0.5*D + 0.5*D = D.
        # For k8v4: 1*D + 0.5*D = 1.5*D. For k8v8: 1*D + 1*D = 2*D.
        # We compute this from the kv_cache_dtype in the vllm config.
        _fusen_slot_bytes_cache = {}

        def _get_fusen_slot_bytes(head_size):
            """Get slot bytes per token per head for FusenCache."""
            if head_size not in _fusen_slot_bytes_cache:
                try:
                    from vllm.config import get_current_vllm_config
                    cfg = get_current_vllm_config()
                    cache_dtype = cfg.cache_config.cache_dtype
                    from fusen_kv.spec_resolver import resolve_spec as _resolve
                    spec = _resolve(cache_dtype)
                    _fusen_slot_bytes_cache[head_size] = spec.slot_bytes(head_size)
                except Exception:
                    # Fallback: k4v4 default (0.5 + 0.5) * D = D bytes per slot
                    _fusen_slot_bytes_cache[head_size] = head_size
            return _fusen_slot_bytes_cache[head_size]

        for spec_cls in (AttentionSpec, FullAttentionSpec):
            if 'real_page_size_bytes' not in spec_cls.__dict__:
                continue
            original_fn = spec_cls.__dict__['real_page_size_bytes'].fget

            def _make_patched(orig):
                @property
                def _patched(self):
                    if self.dtype == _torch.uint8:
                        # FusenCache: use slot_bytes instead of standard formula
                        slot_bytes = _get_fusen_slot_bytes(self.head_size)
                        return self.block_size * self.num_kv_heads * slot_bytes
                    return orig(self)
                return _patched

            spec_cls.real_page_size_bytes = _make_patched(original_fn)
            logger.info("FusenKV: patched %s.real_page_size_bytes",
                         spec_cls.__name__)
    except Exception as e:
        logger.warning("FusenKV: kv_cache_spec patch failed: %s", e)


def _register_startup_hook():
    """Register a warmup hook to precompile Triton kernels on first use.

    This avoids the 2-5s JIT compilation latency on first inference.
    We hook into vLLM's engine initialization rather than doing it at
    plugin load time (which happens in every subprocess).
    """
    # Precompilation is deferred to first forward() call via Triton's
    # internal JIT cache. For explicit precompilation, users can call:
    #   from fusen_kv.warmup import precompile_common
    #   precompile_common()
    pass


def register():
    """Register FusenKV backend with vLLM.

    Called by vLLM's plugin system via entry_points. Performs:
    1. Register CUSTOM attention backend → FusenKVBackend
    2. Patch CacheDType to accept our dtype strings (k4v4, k8v8, etc.)
    3. Patch backend selection to route our dtypes to CUSTOM
    """
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        # Step 1: Register attention backend
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "fusen_kv.backend.FusenKVBackend",
        )
        logger.info("FusenKV: attention backend registered")

        # Step 2: Patch dtype validation
        _patch_cache_dtype()

        # Step 3: Patch backend selection
        _patch_backend_selection()

        # Step 4: Patch dtype string → torch.dtype mapping
        _patch_dtype_mapping()

        # Step 5: Patch KV cache spec for correct page sizing
        _patch_kv_cache_spec()

        # Step 6: Startup hooks
        _register_startup_hook()

        logger.info("FusenKV plugin v0.1.0 loaded successfully")

    except ImportError:
        logger.warning(
            "vLLM not found. FusenKV plugin registration skipped. "
            "The kernel can still be used standalone via kv_cache_gen."
        )
    except Exception as e:
        logger.warning("FusenKV plugin registration failed: %s", e)
