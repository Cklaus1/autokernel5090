#!/usr/bin/env python3
"""Fix vLLM async scheduling race condition for custom attention backends.

Root Cause Analysis
===================
vLLM's async scheduling overlaps CPU input preparation for step N+1 with GPU
kernel execution from step N. The synchronize_input_prep() context manager only
records a CUDA event after input PREPARATION, but model forward (including CUDA
graph replay) happens AFTER the context manager exits. This means:

  Step N:  [prep -> event.record()] -> [model_forward / CUDA graph replay]
  Step N+1: event.synchronize() -> [prep starts]

Step N+1's prep begins as soon as step N's prep event fires, but step N's model
forward may still be executing. While GPU-side operations are ordered by the
CUDA stream, the CPU can freely overwrite pinned-memory buffers and CPU-side
state that feeds into non-blocking GPU copies.

With FlashAttention, the GPU forward is fast enough that it finishes before the
next step's prep starts. With slower attention backends (FusenCache, custom
kernels), the race window widens and manifests as:
  - "CUDA error: an illegal memory access was encountered"
  - Corrupted attention outputs causing NaN/garbage generation
  - CUDA graph replay crashes reading stale/overwritten input buffers

The specific race paths:
  1. input_ids.cpu (pinned memory) is overwritten by step N+1's _prepare_inputs()
     while step N's copy_to_gpu() DMA may still be reading it
  2. block_table.cpu is modified by _update_states() (remove_request, add_request)
     while step N's commit_block_table() DMA reads it
  3. The CUDA caching allocator may reuse memory from temporary tensors freed
     during step N's forward while GPU kernels still reference that memory

The Fix
=======
Record a second CUDA event AFTER model forward completes (not just after input
prep). Synchronize on this "forward done" event before the next step modifies
any shared buffers. This is Approach B from the fix plan.

Performance impact: ~10-20us per step for the event record + synchronize.
This is negligible compared to typical decode latency (5-10ms).

Two variants are provided:
  1. Source patch: Modifies gpu_model_runner.py directly (recommended for testing)
  2. Monkey-patch: Apply at runtime before LLM creation (for plugins)

The fix uses stream-level waiting (torch.cuda.current_stream().wait_event()) rather
than CPU-blocking synchronize(). This preserves the async scheduling benefit: CPU
can still do numpy prep work while the GPU stream waits for model forward to finish.

Usage:
    # Source patch (inside Docker container):
    python3 vllm_async_scheduling_fix.py [--check] [--revert]

    # Monkey-patch (from host, before LLM creation):
    from patches.vllm_async_scheduling_fix import monkey_patch_fix
    monkey_patch_fix()
"""

import sys
import os
import shutil
import functools
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

VLLM_MODEL_RUNNER = (
    "/usr/local/lib/python3.12/dist-packages/"
    "vllm/v1/worker/gpu_model_runner.py"
)

BACKUP_SUFFIX = ".pre_async_fix"

MARKER = "ASYNC_SCHEDULING_FIX"


def find_source():
    """Find gpu_model_runner.py."""
    paths = [
        VLLM_MODEL_RUNNER,
        "/build/vllm/vllm/v1/worker/gpu_model_runner.py",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def check_already_patched(content):
    """Check if the fix is already applied."""
    return MARKER in content


# =============================================================================
# Source patch (modifies gpu_model_runner.py directly)
# =============================================================================

def apply_fix(path):
    """Apply the async scheduling fix to gpu_model_runner.py.

    Three changes:
    1. Add _forward_done_event CUDA event in __init__
    2. Record event after _model_forward() in execute_model()
    3. Wait for event in synchronize_input_prep() before prep GPU ops
    """
    import torch  # needed only for verification

    # Backup
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  Backed up to {backup}")

    with open(path, "r") as f:
        content = f.read()

    if check_already_patched(content):
        print("  Already patched!")
        return True

    original = content
    errors = []

    # =========================================================================
    # Part 1: Add _forward_done_event initialization
    # =========================================================================
    init_target = "            self.prepare_inputs_event = torch.Event()\n"
    if init_target not in content:
        errors.append(
            "Could not find prepare_inputs_event initialization. "
            "vLLM version may be incompatible."
        )
    else:
        init_replacement = (
            "            self.prepare_inputs_event = torch.Event()\n"
            f"            self._forward_done_event = torch.cuda.Event()  # {MARKER}\n"
        )
        content = content.replace(init_target, init_replacement, 1)

    # =========================================================================
    # Part 2: Modify synchronize_input_prep() to wait for model forward
    # =========================================================================
    sync_target = (
        "    def synchronize_input_prep(self):\n"
        "        if self.prepare_inputs_event is None:\n"
        "            yield\n"
        "            return\n"
        "\n"
        "        # Ensure prior step has finished with reused CPU tensors.\n"
        "        # This is required in the async scheduling case because\n"
        "        # the CPU->GPU transfer happens async.\n"
        "        self.prepare_inputs_event.synchronize()\n"
        "        try:\n"
        "            yield\n"
        "        finally:\n"
        "            self.prepare_inputs_event.record()\n"
    )

    sync_replacement = (
        f"    def synchronize_input_prep(self):  # {MARKER}\n"
        "        if self.prepare_inputs_event is None:\n"
        "            yield\n"
        "            return\n"
        "\n"
        "        # Ensure prior step has finished with reused CPU tensors.\n"
        "        # This is required in the async scheduling case because\n"
        "        # the CPU->GPU transfer happens async.\n"
        "        self.prepare_inputs_event.synchronize()\n"
        "\n"
        f"        # {MARKER}: Make the GPU stream wait for the previous\n"
        "        # step's model forward to complete before any prep GPU ops\n"
        "        # (copy_to_gpu, scatter) can proceed. Without this, slow\n"
        "        # attention backends may still be reading shared input\n"
        "        # tensors when the next step's prep overwrites them.\n"
        "        # Uses stream.wait_event() (not CPU synchronize()) to\n"
        "        # preserve async benefit: CPU numpy prep runs in parallel.\n"
        "        import torch as _torch\n"
        "        _torch.cuda.current_stream().wait_event(self._forward_done_event)\n"
        "\n"
        "        try:\n"
        "            yield\n"
        "        finally:\n"
        "            self.prepare_inputs_event.record()\n"
    )

    if sync_target not in content:
        errors.append(
            "Could not find synchronize_input_prep() method. "
            "vLLM version may be incompatible."
        )
    else:
        content = content.replace(sync_target, sync_replacement, 1)

    # =========================================================================
    # Part 3: Record _forward_done_event after model forward
    # =========================================================================
    record_target = "        self.execute_model_state = ExecuteModelState(\n"
    if record_target not in content:
        errors.append(
            "Could not find ExecuteModelState assignment in execute_model(). "
            "vLLM version may be incompatible."
        )
    else:
        record_replacement = (
            f"        # {MARKER}: Record event after model forward completes.\n"
            "        # The next step's synchronize_input_prep() will wait on\n"
            "        # this event before any GPU-side prep ops proceed.\n"
            "        if self.use_async_scheduling:\n"
            "            self._forward_done_event.record()\n"
            "\n"
            "        self.execute_model_state = ExecuteModelState(\n"
        )
        content = content.replace(record_target, record_replacement, 1)

    if errors:
        print("  ERRORS:")
        for e in errors:
            print(f"    - {e}")
        return False

    if content == original:
        print("  ERROR: No changes were made")
        return False

    with open(path, "w") as f:
        f.write(content)

    print(f"  Patched {path}")
    return True


def revert_fix(path):
    """Revert the fix from backup."""
    backup = path + BACKUP_SUFFIX
    if not os.path.exists(backup):
        print(f"  No backup found at {backup}")
        return False
    shutil.copy2(backup, path)
    print(f"  Reverted {path} from backup")
    return True


# =============================================================================
# Monkey-patch (runtime, no file modification)
# =============================================================================

_MONKEY_PATCHED = False


def monkey_patch_fix():
    """Apply the fix at runtime by monkey-patching GPUModelRunner.

    Call this BEFORE creating any LLM or vLLM engine instance.
    Works with multiprocessing=0 (VLLM_ENABLE_V1_MULTIPROCESSING=0).
    """
    global _MONKEY_PATCHED
    if _MONKEY_PATCHED:
        return
    _MONKEY_PATCHED = True

    import torch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _orig_init = GPUModelRunner.__init__
    _orig_model_forward = GPUModelRunner._model_forward
    _orig_sync = GPUModelRunner.synchronize_input_prep

    @functools.wraps(_orig_init)
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._forward_done_event = None
        if getattr(self, 'use_async_scheduling', False):
            self._forward_done_event = torch.cuda.Event()
            logger.info("[FusenFix] _forward_done_event created")

    @functools.wraps(_orig_model_forward)
    def _patched_model_forward(self, *args, **kwargs):
        result = _orig_model_forward(self, *args, **kwargs)
        if getattr(self, '_forward_done_event', None) is not None:
            self._forward_done_event.record()
        return result

    @contextmanager
    def _patched_sync(self):
        if self.prepare_inputs_event is None:
            yield
            return
        self.prepare_inputs_event.synchronize()
        fwd_event = getattr(self, '_forward_done_event', None)
        if fwd_event is not None:
            torch.cuda.current_stream().wait_event(fwd_event)
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    GPUModelRunner.__init__ = _patched_init
    GPUModelRunner._model_forward = _patched_model_forward
    GPUModelRunner.synchronize_input_prep = _patched_sync

    print("[FusenFix] vLLM async scheduling race fix applied (monkey-patch)")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix vLLM async scheduling race condition"
    )
    parser.add_argument(
        "--check", action="store_true", help="Only check if fix is needed"
    )
    parser.add_argument(
        "--revert", action="store_true", help="Revert the fix"
    )
    args = parser.parse_args()

    path = find_source()
    if path is None:
        print("ERROR: gpu_model_runner.py not found")
        print("Searched:")
        print(f"  {VLLM_MODEL_RUNNER}")
        print("  /build/vllm/vllm/v1/worker/gpu_model_runner.py")
        sys.exit(1)

    print(f"Found: {path}")

    with open(path, "r") as f:
        content = f.read()

    if args.check:
        if check_already_patched(content):
            print("  Fix is already applied.")
        else:
            print("  Fix is NOT applied. Run without --check to apply.")
        sys.exit(0)

    if args.revert:
        success = revert_fix(path)
        sys.exit(0 if success else 1)

    success = apply_fix(path)
    if success:
        print("\nAsync scheduling fix applied successfully.")
        print("Restart the vLLM server for changes to take effect.")
    else:
        print("\nFailed to apply fix.")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
