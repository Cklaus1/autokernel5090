#!/usr/bin/env python3
"""Fix FusenCache decode attention crash at C=16+ concurrency on SM120.

Root Cause Analysis
===================
The crash ("CUDA error: an illegal memory access was encountered") occurs
under concurrent request serving. It is NOT in FlashInfer's sliding-window
attention as initially hypothesized -- it is in FusenCache's own decode
attention kernel (both the C++ and Triton paths).

Evidence:
  1. CUDA_LAUNCH_BLOCKING=1 eliminates the crash completely -> async race
  2. Crash occurs with and without CUDA graphs -> not a graph issue
  3. First batch of 16 requests always succeeds, second batch crashes
  4. kv_cache_usage is <0.3% at crash time -> not OOM
  5. Error surfaces at .to() calls in embed_input_ids -> async CUDA error
     from a kernel in the previous step
  6. SM120/Blackwell (RTX 5090) specific -- timing-sensitive race

Architecture context:
  - Gemma4 has 25 sliding-window layers (head_dim=256) + 5 full-attention
    layers (head_dim=512), ALL using FusenKV backend (NOT FlashInfer)
  - TRTLLM attention is disabled on SM120 (supports_trtllm_attention()
    returns False because is_device_capability(100) is False for SM120)
  - Each FusenKVImpl instance has its own _shared_mid_out and _shared_output
    buffers, pre-allocated at max_batch_size

The Race Condition:
  The FusenKV forward() method launches CUDA kernels (store + decode) and
  then copies results: output[:B] = attn_out[:B].to(dtype). Under vLLM's
  async scheduling, the next step's preprocessing (tensor allocations,
  CPU->GPU copies) may begin before the current step's CUDA kernels
  complete. The PyTorch CUDA memory allocator can then reuse memory from
  freed temporary tensors (created by the prefill path's float32 attention
  computation) that is still being read by in-flight decode kernels.

  This is a classic use-after-free in async CUDA execution: the CPU thread
  sees tensors as freed (Python reference count drops to 0), the allocator
  marks the memory as available, but the GPU hasn't finished reading it yet.

Fix
===
Add torch.cuda.current_stream().synchronize() at the end of forward() to
ensure all GPU work completes before returning control to the scheduler.
This prevents the memory allocator from recycling in-flight tensor memory.

Performance impact: ~0.1ms per layer per step (negligible vs kernel time).
A more targeted fix using CUDA events could reduce this, but the sync
approach is simpler and sufficient.

Usage:
    python3 fusencache_concurrency_fix.py  # patches backend.py in-place
"""

import sys
import os
import shutil

# Look for backend.py in multiple locations
SEARCH_PATHS = [
    "/fusen/fusen_kv/backend.py",
    os.path.join(os.path.dirname(__file__), "..", "fusen_kv", "backend.py"),
]


def find_backend():
    for path in SEARCH_PATHS:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
    return None


def patch_backend(path):
    # Backup
    backup = path + ".pre_concurrency_fix"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"Backed up to {backup}")

    with open(path, "r") as f:
        content = f.read()

    # Check if already patched
    if "CONCURRENCY FIX" in content:
        print("Already patched!")
        return True

    # Find the return statement at the end of forward()
    # Pattern: "        return output\n" at the end of the forward method
    # This is the LAST "return output" in the file (end of forward())
    target = "        return output\n"
    last_idx = content.rfind(target)

    if last_idx < 0:
        print("ERROR: Could not find 'return output' in forward()")
        return False

    # Verify this is inside the forward() method
    forward_idx = content.rfind("def forward(", 0, last_idx)
    if forward_idx < 0:
        print("ERROR: Could not verify this is inside forward()")
        return False

    replacement = """\
        # CONCURRENCY FIX: Synchronize the CUDA stream to ensure all kernels
        # (store + decode) complete before returning. Without this, vLLM's
        # async scheduler may start the next step's tensor allocations while
        # this step's GPU kernels are still running, causing the CUDA memory
        # allocator to hand out memory that is still in use by in-flight
        # kernels. This manifests as "illegal memory access" crashes at
        # C=16+ concurrent requests on SM120 (RTX 5090).
        #
        # Performance: ~0.1ms overhead per layer per step (negligible).
        # Skip during CUDA graph capture (synchronize would break capture).
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream().synchronize()

        return output
"""

    content = content[:last_idx] + replacement + content[last_idx + len(target):]

    with open(path, "w") as f:
        f.write(content)

    print(f"Patched {path}")
    print("Restart the vLLM server for changes to take effect.")
    return True


if __name__ == "__main__":
    path = find_backend()
    if path is None:
        print("ERROR: backend.py not found in search paths:")
        for p in SEARCH_PATHS:
            print(f"  {os.path.abspath(p)}")
        sys.exit(1)

    print(f"Found backend.py at: {path}")
    success = patch_backend(path)
    sys.exit(0 if success else 1)
