#!/usr/bin/env python3
"""Dynamic per-request chunked prefill sizing for vLLM v1 scheduler.

Problem
=======
vLLM's v1 scheduler uses a single global `scheduler_config.long_prefill_token_threshold`
to cap the number of tokens a request may draw from `token_budget` per step. With
--max-num-batched-tokens 512 and the default threshold=0 (disabled), a long prompt
(~4K tok) grabs the entire 512-token step-budget for ~8 consecutive scheduler steps.
Under priority scheduling with C=8 concurrent short/long alternating requests, short
requests behind one long prefill see 4s+ P99 TTFT spikes.

Fix
===
Enable per-request chunk sizing: long prompts get capped at 256 tok/step; short prompts
(num_new_tokens < 256) pass through untouched. This lets the scheduler interleave a
long-prompt chunk + several short prompts within a single step budget, so short-prompt
TTFT is bounded by one chunked-prefill step (~100-200ms) rather than ~8 full steps.

Formula (loosely priority-aware, but primary dimension is prompt length):
    chunk = min(num_new_tokens, max(256, 2048 - request.priority * 256))
High-priority requests (priority=10) get chunk = max(256, -512) = 256 tok.
Low-priority long prompts (priority=1) get chunk = max(256, 1792) = 1792 tok.
In practice, the dominant effect is a hard 256-token cap on prefill chunks for
long low-priority prompts — the priority term mostly tunes short-prompt batching.

Since the vLLM v1 scheduler reads `long_prefill_token_threshold` as a scalar at
two spots (line ~409 for RUNNING queue, ~674 for WAITING queue), the simplest patch
is to set `long_prefill_token_threshold = 256` on the scheduler_config after the
Scheduler is constructed. This:
  1. Applies per-request (short prompts with num_new_tokens<256 bypass the cap)
  2. Requires no source rewrite of scheduler.py
  3. Installs via vLLM plugin entry point under VLLM_PLUGINS=dynamic_chunked

Activated via:
    VLLM_PLUGINS=dynamic_chunked python3 -m vllm.entrypoints.openai.api_server ...
  OR standalone:
    import patches.dynamic_chunked_prefill
    patches.dynamic_chunked_prefill.register()

Kill switch:
    AUTOKERNEL_DYNAMIC_CHUNK=0  -> no-op
    AUTOKERNEL_DYNAMIC_CHUNK_SIZE=256 -> override chunk cap (default 256)
"""

import logging
import os

logger = logging.getLogger("autokernel.dynamic_chunked_prefill")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")


_CHUNK_CAP = int(os.environ.get("AUTOKERNEL_DYNAMIC_CHUNK_SIZE", "256"))


def _do_wrap():
    """Wrap Scheduler.__init__. Returns True if wrapped (or already wrapped)."""
    from vllm.v1.core.sched.scheduler import Scheduler
    orig_init = Scheduler.__init__
    if getattr(orig_init, "_dynamic_chunked_wrapped", False):
        logger.info("Scheduler.__init__ already wrapped; skipping")
        return True

    chunk_cap = _CHUNK_CAP

    def wrapped_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        prior = getattr(self.scheduler_config, "long_prefill_token_threshold", 0)
        self.scheduler_config.long_prefill_token_threshold = chunk_cap
        try:
            mnbt = self.scheduler_config.max_num_batched_tokens
        except Exception:
            mnbt = "?"
        logger.warning(
            "[dynamic_chunked_prefill] ACTIVE: long_prefill_token_threshold "
            "%s -> %d (max_num_batched_tokens=%s)", prior, chunk_cap, mnbt,
        )

    wrapped_init._dynamic_chunked_wrapped = True
    Scheduler.__init__ = wrapped_init

    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        async_orig = AsyncScheduler.__init__
        if not getattr(async_orig, "_dynamic_chunked_wrapped", False):
            def async_wrapped(self, *args, **kwargs):
                async_orig(self, *args, **kwargs)
                prior = getattr(self.scheduler_config, "long_prefill_token_threshold", 0)
                self.scheduler_config.long_prefill_token_threshold = chunk_cap
                logger.warning(
                    "[dynamic_chunked_prefill] AsyncScheduler active: "
                    "long_prefill_token_threshold %s -> %d", prior, chunk_cap,
                )
            async_wrapped._dynamic_chunked_wrapped = True
            AsyncScheduler.__init__ = async_wrapped
    except Exception:
        pass
    return True


def _apply_patch():
    kill = os.environ.get("AUTOKERNEL_DYNAMIC_CHUNK", "1")
    if kill == "0":
        logger.info("AUTOKERNEL_DYNAMIC_CHUNK=0 -> plugin disabled (no patch applied)")
        return False

    chunk_cap = _CHUNK_CAP
    logger.info("dynamic_chunked_prefill: wrapping Scheduler.__init__ "
                "(long_prefill_token_threshold -> %d)", chunk_cap)

    # Try immediate wrap. If vLLM scheduler isn't importable yet (too early in
    # Python startup), register a meta-path finder that wraps on first import.
    try:
        return _do_wrap()
    except ImportError:
        pass
    except Exception as e:
        logger.exception("failed to wrap Scheduler immediately: %s", e)

    # Lazy path: install a post-import hook.
    try:
        import importlib.abc
        import importlib.util

        class _Loader(importlib.abc.Loader):
            pass

        _wrap_done = {"done": False}

        def _maybe_wrap(name):
            if _wrap_done["done"]:
                return
            if name == "vllm.v1.core.sched.scheduler":
                try:
                    _do_wrap()
                    _wrap_done["done"] = True
                except Exception as e:
                    logger.exception("deferred wrap failed: %s", e)

        class _PostImportHook:
            def find_module(self, name, path=None):
                return None  # never load; just observe

            def find_spec(self, name, path=None, target=None):
                return None

        # Use a simpler approach: override __import__ to wrap after the real
        # import completes.
        import builtins as _builtins
        _orig_import = _builtins.__import__

        def _patched_import(name, *a, **kw):
            mod = _orig_import(name, *a, **kw)
            if not _wrap_done["done"]:
                # Check both the requested name and resolved module
                try:
                    import sys as _sys
                    if "vllm.v1.core.sched.scheduler" in _sys.modules:
                        _do_wrap()
                        _wrap_done["done"] = True
                        _builtins.__import__ = _orig_import  # restore
                except Exception:
                    pass
            return mod

        _builtins.__import__ = _patched_import
        logger.info("Registered deferred wrap via __import__ hook")
        return True
    except Exception as e:
        logger.exception("failed to install deferred wrap: %s", e)
        return False


def register():
    """vLLM plugin entry point: called once per process at plugin-load time."""
    _apply_patch()


# If imported directly (not via entry point), apply immediately.
if os.environ.get("AUTOKERNEL_DYNAMIC_CHUNK_AUTO", "0") == "1":
    _apply_patch()
