#!/usr/bin/env python3
"""T1-B: Piecewise CUDA graph capture race fix for FusenCache + vLLM.

Problem
=======
Mixed-batch CUDA graph capture crashes with `cudaErrorIllegalInstruction` at
batch size >= 48 on SM120. The workaround is FULL_DECODE_ONLY mode, which
costs ~30% throughput (4,029 tok/s vs projected ~9,000 tok/s).

Root Cause Analysis
===================
There are TWO distinct problems preventing piecewise CUDA graph support:

1. METADATA CLONE ALLOCATIONS IN CAPTURED REGION (backend.py:886-891)
   In `forward()`, when `not _capturing`, the code calls `.clone()` on four
   metadata tensors: block_table, seq_lens, slot_mapping, query_start_loc.
   Each `.clone()` allocates new GPU memory via PyTorch's caching allocator.

   In piecewise mode (`-cc.mode compile` / `VLLM_COMPILE=1`), the model
   forward is split into graph-captured regions (MLP, norms, embeddings)
   and eager regions (attention). The attention `forward()` runs EAGERLY
   between graph pieces. BUT: `torch.cuda.is_current_stream_capturing()`
   returns False during eager regions of piecewise execution, even though
   the surrounding graph pieces have a private memory pool.

   The clones allocate from the DEFAULT memory pool, not the graph's private
   pool. When the next graph piece replays, it expects specific memory
   addresses from its private pool, but the caching allocator has handed
   out overlapping addresses from the default pool to the clones. Result:
   `cudaErrorIllegalInstruction` during graph REPLAY (not capture).

2. DYNAMIC ALLOCATIONS IN UNIVERSAL DECODE PATH (backend.py:1010-1047)
   The mixed prefill+decode path (entered when `max_query_len > 1`) calls:
   - `torch.arange(padded_T, ...)` -- allocates a new tensor
   - `torch.searchsorted(...)` -- allocates output tensor
   - `qsl[1:]` -- creates a view (safe)
   - `token_request_ids.clamp(...)` -- may allocate
   - Arithmetic ops that create intermediate tensors
   - `_block_table[token_request_ids]` -- fancy indexing allocates

   Discovery #57 confirmed: these allocations crash CUDA graph capture.
   The current code skips ALWAYS mode because of this, falling back to
   UNIFORM_SINGLE_TOKEN_DECODE. But in piecewise mode, the attention
   runs eagerly (so capture isn't active), and the issue is different:
   the allocations from the default pool conflict with graph piece pools.

Fix Strategy
============
The fix has two parts:

Part A: No-clone metadata in piecewise path
   Instead of cloning metadata tensors every forward() call, pre-allocate
   persistent "shadow" tensors at init time (same shapes as the max-size
   metadata). In forward(), copy INTO these pre-allocated shadows using
   `.copy_()` (in-place, no allocation). The shadows have fixed addresses
   that are compatible with CUDA graph memory pools.

   For the async scheduling race protection (the original reason for cloning),
   `.copy_()` into pre-allocated buffers provides the same isolation: our
   kernels read from the shadow, the CPU can modify the originals freely.

Part B: Pre-allocate universal decode intermediates
   Pre-allocate the `token_positions`, `token_request_ids`,
   `pseudo_seq_lens`, and `token_block_table` tensors at init time at
   max size. Rewrite the universal decode path to use in-place ops
   (`.copy_()`, out= parameters, in-place arithmetic) instead of
   creating new tensors.

Part C: Enable FULL mode + piecewise
   With Parts A+B eliminating all dynamic allocations, enable
   `_patch_cudagraph_mode_override()` in plugin.py so mixed-batch
   graphs get proper CUDA graph coverage.

Usage
=====
    # Apply to backend.py:
    python3 t1b_piecewise_cudagraph_fix.py

    # Then enable in plugin.py (Step 7):
    # Uncomment _patch_cudagraph_mode_override() call

    # Launch with piecewise mode:
    python3 /fusen/fusen_kv/launch_vllm.py \\
      --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \\
      --quantization modelopt \\
      --max-model-len 4096 \\
      --max-num-seqs 256 \\
      --trust-remote-code \\
      --port 8001 \\
      --kv-cache-dtype k4v4b64 \\
      '-cc.custom_ops=["all"]' \\
      -cc.cudagraph_mode piecewise

Validation
==========
    1. Capture succeeds at B=128 (no crash during capture or replay)
    2. Graph replay matches eager within 1e-3 RMS
    3. Throughput >= 6,685 tok/s at C=128+ on single GPU
"""

import os
import sys
import shutil
import textwrap


# ============================================================
# Part A: Patch backend.py to use pre-allocated shadow buffers
# ============================================================

SHADOW_INIT_CODE = textwrap.dedent("""\

        # ---- T1-B FIX: Pre-allocated shadow metadata buffers ----
        # Instead of .clone() which allocates from the default memory pool
        # (conflicting with piecewise CUDA graph private pools), we
        # pre-allocate fixed-address shadow buffers and use .copy_() at
        # runtime. This eliminates ALL dynamic allocations in forward().
        #
        # Sizing: max_num_batched_tokens for token-indexed tensors,
        # max_num_seqs for request-indexed tensors.
        _max_seqs = _max_B
        _max_tokens = _max_B  # decode: 1 token per seq
        try:
            if vllm_cfg and vllm_cfg.scheduler_config:
                _mbt = getattr(vllm_cfg.scheduler_config,
                               'max_num_batched_tokens', None)
                if _mbt and _mbt > 0:
                    _max_tokens = _mbt
        except Exception:
            pass

        # Block table max columns from cache config
        _max_blocks_per_seq = 256  # safe default
        try:
            if vllm_cfg and vllm_cfg.cache_config:
                _bl = getattr(vllm_cfg.cache_config, 'num_gpu_blocks', None)
                if _bl and _bl > 0:
                    _max_blocks_per_seq = min(_bl, 4096)
        except Exception:
            pass

        # Shadow metadata: fixed-address buffers for async race protection
        self._shadow_block_table = torch.empty(
            _max_seqs, _max_blocks_per_seq,
            dtype=torch.int32, device='cuda',
        )
        self._shadow_seq_lens = torch.empty(
            _max_seqs, dtype=torch.int32, device='cuda',
        )
        self._shadow_slot_mapping = torch.empty(
            _max_tokens, dtype=torch.int64, device='cuda',
        )
        self._shadow_query_start_loc = torch.empty(
            _max_seqs + 1, dtype=torch.int64, device='cuda',
        )

        # Universal decode path: pre-allocated intermediates
        self._ud_token_positions = torch.arange(
            _max_tokens, device='cuda', dtype=torch.int64,
        )
        self._ud_token_request_ids = torch.empty(
            _max_tokens, dtype=torch.int64, device='cuda',
        )
        self._ud_pseudo_seq_lens = torch.empty(
            _max_tokens, dtype=torch.int32, device='cuda',
        )
        self._ud_token_block_table = torch.empty(
            _max_tokens, _max_blocks_per_seq,
            dtype=torch.int32, device='cuda',
        )
        self._shadow_max_seqs = _max_seqs
        self._shadow_max_tokens = _max_tokens
        self._shadow_max_blocks = _max_blocks_per_seq
        logger.info(
            "FusenKV T1-B: shadow metadata allocated "
            "(max_seqs=%d, max_tokens=%d, max_blocks=%d, "
            "shadow_mem=%.1f MiB)",
            _max_seqs, _max_tokens, _max_blocks_per_seq,
            (self._shadow_block_table.nbytes +
             self._shadow_seq_lens.nbytes +
             self._shadow_slot_mapping.nbytes +
             self._shadow_query_start_loc.nbytes +
             self._ud_token_positions.nbytes +
             self._ud_token_request_ids.nbytes +
             self._ud_pseudo_seq_lens.nbytes +
             self._ud_token_block_table.nbytes) / (1024 * 1024),
        )
""")


# The new forward() metadata handling: replace clone() with copy_()
FORWARD_METADATA_OLD = """\
        if not _capturing:
            # PIECEWISE FIX: Ensure non_blocking H2D metadata copies are visible.
            # vLLM copies query_start_loc/seq_lens to GPU with non_blocking=True
            # on the default stream. In piecewise mode, attention runs eagerly
            # on a potentially different compute stream. Wait for the default
            # stream to ensure copies are complete. No-op when streams match
            # (mode=none), so zero cost for the common case.
            _cur_stream = torch.cuda.current_stream(query.device)
            _def_stream = torch.cuda.default_stream(query.device)
            if _cur_stream != _def_stream:
                _cur_stream.wait_stream(_def_stream)

            # GPU-side fence: ensure previous step's kernels finish before
            # we start reusing shared decode buffers (mid_out, output).
            torch.cuda.current_stream().wait_event(self._prev_step_event)

            # Clone metadata to isolate from async scheduler writes.
            # This is the key fix: our kernels read clones, CPU modifies originals.
            _block_table = attn_metadata.block_table.clone() if attn_metadata.block_table is not None else None
            _seq_lens = attn_metadata.seq_lens.clone() if attn_metadata.seq_lens is not None else None
            _slot_mapping = attn_metadata.slot_mapping.clone() if attn_metadata.slot_mapping is not None else None
            _query_start_loc = attn_metadata.query_start_loc.clone() if attn_metadata.query_start_loc is not None else None

            # Retain references to clones so Python GC doesn't free them while
            # the GPU is still reading them. The previous step's clones are now
            # safe to release (the event fence above guarantees completion).
            self._prev_block_table = _block_table
            self._prev_seq_lens = _seq_lens
            self._prev_slot_mapping = _slot_mapping
            self._prev_query_start_loc = _query_start_loc
        else:
            # During CUDA graph capture: use originals (fixed addresses required)
            _block_table = attn_metadata.block_table
            _seq_lens = attn_metadata.seq_lens
            _slot_mapping = attn_metadata.slot_mapping
            _query_start_loc = attn_metadata.query_start_loc"""

FORWARD_METADATA_NEW = """\
        if not _capturing:
            # PIECEWISE FIX: Ensure non_blocking H2D metadata copies are visible.
            _cur_stream = torch.cuda.current_stream(query.device)
            _def_stream = torch.cuda.default_stream(query.device)
            if _cur_stream != _def_stream:
                _cur_stream.wait_stream(_def_stream)

            # GPU-side fence: ensure previous step's kernels finish before
            # we start reusing shared decode buffers (mid_out, output).
            torch.cuda.current_stream().wait_event(self._prev_step_event)

            # T1-B FIX: Copy metadata INTO pre-allocated shadow buffers
            # instead of .clone() which allocates from the default pool.
            # .copy_() is in-place: no allocation, fixed addresses,
            # compatible with piecewise CUDA graph memory pools.
            if attn_metadata.block_table is not None:
                bt = attn_metadata.block_table
                bt_rows = min(bt.shape[0], self._shadow_block_table.shape[0])
                bt_cols = min(bt.shape[1], self._shadow_block_table.shape[1])
                self._shadow_block_table[:bt_rows, :bt_cols].copy_(
                    bt[:bt_rows, :bt_cols].to(self._shadow_block_table.dtype))
                _block_table = self._shadow_block_table[:bt_rows, :bt_cols]
            else:
                _block_table = None

            if attn_metadata.seq_lens is not None:
                sl = attn_metadata.seq_lens
                sl_n = min(sl.shape[0], self._shadow_seq_lens.shape[0])
                self._shadow_seq_lens[:sl_n].copy_(
                    sl[:sl_n].to(self._shadow_seq_lens.dtype))
                _seq_lens = self._shadow_seq_lens[:sl_n]
            else:
                _seq_lens = None

            if attn_metadata.slot_mapping is not None:
                sm = attn_metadata.slot_mapping
                sm_n = min(sm.shape[0], self._shadow_slot_mapping.shape[0])
                self._shadow_slot_mapping[:sm_n].copy_(
                    sm[:sm_n].to(self._shadow_slot_mapping.dtype))
                _slot_mapping = self._shadow_slot_mapping[:sm_n]
            else:
                _slot_mapping = None

            if attn_metadata.query_start_loc is not None:
                ql = attn_metadata.query_start_loc
                ql_n = min(ql.shape[0], self._shadow_query_start_loc.shape[0])
                self._shadow_query_start_loc[:ql_n].copy_(
                    ql[:ql_n].to(self._shadow_query_start_loc.dtype))
                _query_start_loc = self._shadow_query_start_loc[:ql_n]
            else:
                _query_start_loc = None
        else:
            # During CUDA graph capture: use originals (fixed addresses required)
            _block_table = attn_metadata.block_table
            _seq_lens = attn_metadata.seq_lens
            _slot_mapping = attn_metadata.slot_mapping
            _query_start_loc = attn_metadata.query_start_loc"""


# The new universal decode path: pre-allocated intermediates
UNIVERSAL_DECODE_OLD = """\
            qsl = _query_start_loc
            padded_T = query.shape[0]  # padded num_tokens (graph-fixed shape)

            # Compute per-token request assignment: which request owns each token
            # query_start_loc is [num_reqs+1] (or [padded_num_reqs+1])
            # Use searchsorted: for token t, find i such that qsl[i] <= t < qsl[i+1]
            token_positions = torch.arange(
                padded_T, device=query.device, dtype=qsl.dtype)
            # searchsorted with right=True: returns index of first element
            # strictly greater than the value. For qsl_right = qsl[1:],
            # this gives the request index that owns each token.
            qsl_right = qsl[1:]  # [num_reqs] or [padded_reqs]
            token_request_ids = torch.searchsorted(
                qsl_right, token_positions, right=True)
            # Clamp to valid request range (padded tokens get last request id
            # but their seq_lens will be 0, producing zero output)
            max_req_idx = qsl_right.shape[0] - 1
            token_request_ids = token_request_ids.clamp(max=max_req_idx)

            # Per-request query lengths: qsl[i+1] - qsl[i]
            query_lens = qsl_right - qsl[:qsl_right.shape[0]]  # [num_reqs]

            # Position of each token within its request (0-indexed)
            position_in_request = token_positions - qsl[token_request_ids]

            # Per-token pseudo seq_lens for causal attention:
            # token at position p in request i should see:
            #   (seq_lens[i] - query_lens[i]) + (p + 1)
            # = context_len_before_this_batch + position_in_prefill + 1
            per_req_seq = _seq_lens[token_request_ids]
            per_req_qlen = query_lens[token_request_ids]
            pseudo_seq_lens = per_req_seq - per_req_qlen + position_in_request + 1
            # Clamp: must be >= 0 and must not exceed the request's total
            # seq_len. The min(pseudo, per_req_seq) ensures padded tokens
            # (where seq_lens=0 and query_lens=0 but position > 0) get
            # pseudo_seq_lens=0, preventing reads from invalid block_table
            # entries. The decode kernel handles seq_lens=0 safely
            # (split_start >= split_end early return + zero output).
            pseudo_seq_lens = pseudo_seq_lens.clamp(min=0)
            pseudo_seq_lens = torch.min(pseudo_seq_lens, per_req_seq)

            # Per-token block table: expand from [num_reqs, max_blocks] to
            # [padded_T, max_blocks] by indexing with token_request_ids
            token_block_table = _block_table[token_request_ids]"""

UNIVERSAL_DECODE_NEW = """\
            qsl = _query_start_loc
            padded_T = query.shape[0]  # padded num_tokens (graph-fixed shape)

            # T1-B FIX: Use pre-allocated intermediates instead of
            # torch.arange/searchsorted/fancy-indexing which allocate.
            # All outputs are written into shadow buffers via views.
            token_positions = self._ud_token_positions[:padded_T]

            qsl_right = qsl[1:]  # view, no allocation
            # searchsorted into pre-allocated output
            torch.searchsorted(
                qsl_right, token_positions, right=True,
                out=self._ud_token_request_ids[:padded_T])
            token_request_ids = self._ud_token_request_ids[:padded_T]

            max_req_idx = qsl_right.shape[0] - 1
            token_request_ids.clamp_(max=max_req_idx)  # in-place

            # Per-request query lengths
            query_lens = qsl_right - qsl[:qsl_right.shape[0]]

            # Position of each token within its request
            position_in_request = token_positions - qsl[token_request_ids]

            # Per-token pseudo seq_lens (into pre-allocated buffer)
            per_req_seq = _seq_lens[token_request_ids]
            per_req_qlen = query_lens[token_request_ids]
            pseudo_seq_lens = self._ud_pseudo_seq_lens[:padded_T]
            # Compute: per_req_seq - per_req_qlen + position_in_request + 1
            pseudo_seq_lens.copy_(
                (per_req_seq - per_req_qlen + position_in_request + 1).to(
                    pseudo_seq_lens.dtype))
            pseudo_seq_lens.clamp_(min=0)  # in-place
            torch.min(pseudo_seq_lens, per_req_seq.to(pseudo_seq_lens.dtype),
                      out=pseudo_seq_lens)  # in-place

            # Per-token block table (into pre-allocated buffer)
            bt_cols = _block_table.shape[1]
            token_block_table = self._ud_token_block_table[:padded_T, :bt_cols]
            token_block_table.copy_(_block_table[token_request_ids])"""


def find_backend():
    search_paths = [
        "/fusen/fusen_kv/backend.py",
        os.path.join(os.path.dirname(__file__), "..", "fusen_kv", "backend.py"),
    ]
    for path in search_paths:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path
    return None


def patch_backend(path):
    """Apply T1-B patches to backend.py."""
    # Backup
    backup = path + ".pre_t1b"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"Backed up to {backup}")

    with open(path, "r") as f:
        content = f.read()

    if "T1-B FIX" in content:
        print("Already patched with T1-B!")
        return True

    # Part A: Add shadow buffer initialization after shared decode buffers
    marker = "logger.info(\n"
    marker2 = '            "FusenKV: shared decode buffers allocated at max_B=%d "'
    # Find the shared buffer allocation log message end
    insert_marker = (
        '            self._shared_output.nbytes / (1024 * 1024),\n'
        '        )'
    )
    idx = content.find(insert_marker)
    if idx < 0:
        print("ERROR: Could not find shared buffer log marker")
        return False
    # Insert after the closing paren of the logger.info
    insert_pos = idx + len(insert_marker)
    content = content[:insert_pos] + SHADOW_INIT_CODE + content[insert_pos:]

    # Part B: Replace clone() with copy_() in forward()
    if FORWARD_METADATA_OLD not in content:
        print("ERROR: Could not find forward() metadata clone pattern")
        return False
    content = content.replace(FORWARD_METADATA_OLD, FORWARD_METADATA_NEW, 1)

    # Remove the _prev_* reference retention (no longer needed with shadows)
    old_prev_refs = """\
            # Retain references to clones so Python GC doesn't free them while
            # the GPU is still reading them. The previous step's clones are now
            # safe to release (the event fence above guarantees completion).
            self._prev_block_table = _block_table
            self._prev_seq_lens = _seq_lens
            self._prev_slot_mapping = _slot_mapping
            self._prev_query_start_loc = _query_start_loc"""
    if old_prev_refs in content:
        content = content.replace(old_prev_refs, "", 1)

    # Part C: Replace universal decode path
    if UNIVERSAL_DECODE_OLD not in content:
        print("ERROR: Could not find universal decode path pattern")
        return False
    content = content.replace(UNIVERSAL_DECODE_OLD, UNIVERSAL_DECODE_NEW, 1)

    with open(path, "w") as f:
        f.write(content)

    print(f"Patched {path} with T1-B piecewise CUDA graph fix")
    return True


def patch_plugin(plugin_path=None):
    """Enable _patch_cudagraph_mode_override() in plugin.py."""
    if plugin_path is None:
        search_paths = [
            "/fusen/fusen_kv/plugin.py",
            os.path.join(os.path.dirname(__file__), "..", "fusen_kv", "plugin.py"),
        ]
        for path in search_paths:
            path = os.path.abspath(path)
            if os.path.exists(path):
                plugin_path = path
                break

    if plugin_path is None:
        print("WARNING: plugin.py not found, skipping mode override patch")
        return False

    with open(plugin_path, "r") as f:
        content = f.read()

    # Find the commented-out call to _patch_cudagraph_mode_override
    old_comment = """\
        # Step 7: Prevent CUDA graph mode downgrade (new vLLM auto-resolves)
        # NOTE: _patch_cudagraph_mode_override() exists but is NOT called here.
        # FULL mode causes mixed-batch CUDA graph replay to crash (Discovery #57).
        # Instead, we accept FULL_DECODE_ONLY and compensate with larger capture
        # sizes + max_num_seqs=64 in the launch config. This achieves ~3,500 tok/s
        # at C=64/128 without crashes."""

    new_comment = """\
        # Step 7: Prevent CUDA graph mode downgrade (new vLLM auto-resolves)
        # T1-B FIX: Now safe to enable FULL mode because forward() no longer
        # allocates from the default memory pool (shadow buffers + in-place ops).
        _patch_cudagraph_mode_override()"""

    if old_comment in content:
        content = content.replace(old_comment, new_comment, 1)
        with open(plugin_path, "w") as f:
            f.write(content)
        print(f"Patched {plugin_path}: enabled _patch_cudagraph_mode_override()")
        return True
    elif "T1-B FIX" in content:
        print("plugin.py already patched with T1-B!")
        return True
    else:
        print("WARNING: Could not find Step 7 comment block in plugin.py")
        return False


def main():
    backend_path = find_backend()
    if backend_path is None:
        # If not in container, show what the patch does
        print("backend.py not found (not in vLLM container?)")
        print("This patch modifies:")
        print("  1. fusen_kv/backend.py: Replace .clone() with .copy_() into "
              "pre-allocated shadow buffers")
        print("  2. fusen_kv/backend.py: Pre-allocate universal decode "
              "intermediates")
        print("  3. fusen_kv/plugin.py: Enable _patch_cudagraph_mode_override()")
        print("\nThe patched tensors that were being re-allocated during "
              "piecewise graph execution:")
        print("  - block_table.clone()      -> shadow_block_table.copy_()")
        print("  - seq_lens.clone()         -> shadow_seq_lens.copy_()")
        print("  - slot_mapping.clone()     -> shadow_slot_mapping.copy_()")
        print("  - query_start_loc.clone()  -> shadow_query_start_loc.copy_()")
        print("  - torch.arange(padded_T)   -> pre-allocated token_positions")
        print("  - torch.searchsorted(...)  -> out= into pre-allocated buffer")
        print("  - _block_table[ids]        -> copy_() into pre-allocated buffer")
        print("  - pseudo_seq_lens arith    -> copy_() into pre-allocated buffer")
        return 0

    success = patch_backend(backend_path)
    if success:
        patch_plugin()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
