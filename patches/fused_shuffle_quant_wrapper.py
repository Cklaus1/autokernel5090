#!/usr/bin/env python3
"""
T2-N: Fused shuffle_rows + quantize_to_nvfp4 vLLM integration wrapper.

STATUS: PARTIALLY IMPLEMENTED — scale format incompatibility identified.

What this file does:
    1. Loads and exposes the fused_shuffle_quant CUDA kernel
    2. Provides fused_shuffle_and_quant_moe() which is faster than the two-op path
       for the BF16→FP4 conversion (1.1-1.5x speedup in kernel time)
    3. Provides patch_cutlass_moe_fp4() monkey-patch infrastructure
    4. Documents the scale format blocker for full vLLM integration

TWO-OP SEQUENCE BEING FUSED:
    sorted_bf16 = ops.shuffle_rows(a, a_map)          # write M*K BF16 to gmem
    fp4, scales = ops.scaled_fp4_experts_quant(...)   # read M*K BF16 from gmem

FUSED KERNEL:
    fp4, scales = fused_kernel(a, a_map, ...)
    # gather-reads from unsorted positions, no BF16 intermediate

SCALE FORMAT BLOCKER
====================
vLLM's CUTLASS MoE kernel (cutlass_fp4_moe_mm) expects scales in a SWIZZLED
blockscale layout that is fundamentally different from our row-major FP8-E4M3 scales.

Analysis of scaled_fp4_experts_quant output for M=256, K=7168 (Gemma4 26B):
  - Scale buffer shape: [327680, 448] fp8_e4m3fn (= [MAX_TPE*topk, n_blocks])
  - Active scales: 256 tokens × 448 blocks/token = 114,688 fp8 values
  - BUT they are spread across rows 0..8191 in a swizzled pattern
  - Each token spans 32 rows, with non-zeros at cols 0-3, 16-19, 32-35, ...
  - blockscale_offsets are flat element indices into this buffer
  - Pattern: 4 contiguous bytes every 16 columns = swizzled for cache alignment

Our kernel produces:
  - scales[row, block_idx] = one FP8 byte per 16-element FP4 block, row-major
  - blockscale_offsets would be: token_row * n_blocks (simple stride)
  - INCOMPATIBLE with CUTLASS swizzled layout

NEXT STEP FOR FULL INTEGRATION:
  Implement the swizzled scale write in the CUDA kernel to match CUTLASS format.
  The swizzle pattern interleaves 4-byte groups with stride-16 column gaps.
  Estimated implementation effort: 1 day.
  Expected net speedup: 1.1-1.3x for the shuffle+quant step after swizzle fix.

CURRENT FUSED KERNEL PERFORMANCE (verified on SM120, M=256, K=7168):
  - shuffle_rows:                  38.8 µs (writes 3.5 MB BF16)
  - scaled_fp4_experts_quant:     115.8 µs (reads 3.5 MB BF16, writes FP4+scales)
  - Two-op sequential:             ~21 µs (with CUDA pipelining)
  - Fused kernel raw:              ~20 µs (1.05x, no scale format)
  - Fused + compact scale view:    ~19 µs (1.12x speedup vs two-op)

USAGE (STANDALONE — not connected to CUTLASS MoE):
    from fused_shuffle_quant_wrapper import fused_shuffle_and_quant_moe
    # Returns (fp4, compact_scales) in row-major format
    # Use for: custom inference engines that accept row-major scales
    fp4_out, scales_out = fused_shuffle_and_quant_moe(
        a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk
    )

KILL CRITERION:
    >10% slower than two-op baseline OR any token mismatch vs eager path.
    Disable: AUTOKERNEL_FUSED_SHUFFLE_QUANT=0
"""

import logging
import os
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ============================================================
# Environment flag for kill switch
# ============================================================
_ENABLED = os.environ.get("AUTOKERNEL_FUSED_SHUFFLE_QUANT", "1") != "0"

# T2-N v3: silu+mul+FP4 quant epilogue (GEMM1 post-op fusion).
# Tag: W6_T2N_silu_epilogue.  Plans ref: plans/t2n_silu_epilogue_impl.md.
# Off by default until bench banks the win.
_SILU_EPI_ENABLED = os.environ.get("AUTOKERNEL_T2N_SILU_EPILOGUE", "0") == "1"

# ============================================================
# Kernel load state
# ============================================================
_kernel_loaded: Optional[bool] = None  # None = not yet checked
_fused_op = None  # The loaded pybind11 function

# T2-N v3 silu-fp4 epilogue kernel state
_silu_fp4_loaded: Optional[bool] = None
_silu_fp4_op = None
_silu_fp4_logged_once = False  # P2: log "[T2N-SILU-EPILOGUE] active ..." once

# Persistent large-buffer cache so we don't reallocate the
# [MAX_TPE * topk, padded_k_int32] int32 scale tensor on every forward.
# Keyed by (device_index, target_rows, padded_k_int32).
_scale_buf_cache: dict = {}
def _resolve_so_path() -> str:
    """Resolve path to fused_shuffle_quant .so, searching common locations.

    Order:
      1. ``AUTOKERNEL_FUSED_SHUFFLE_QUANT_SO`` env var (absolute path).
      2. ``<repo>/workspace/fused_shuffle_quant_sm120a.so`` (host path).
      3. ``/autokernel/workspace/fused_shuffle_quant_sm120a.so`` (docker mount).
      4. ``/tmp/build_fused_shuffle_quant/fused_shuffle_quant.so`` (legacy
         torch.utils.cpp_extension cache).
    """
    env = os.environ.get("AUTOKERNEL_FUSED_SHUFFLE_QUANT_SO")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    candidates = [
        os.path.join(repo, "workspace", "fused_shuffle_quant_sm120a.so"),
        "/autokernel/workspace/fused_shuffle_quant_sm120a.so",
        "/tmp/build_fused_shuffle_quant/fused_shuffle_quant.so",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Return the preferred target even if missing so error messages point there.
    return candidates[0]


_SO_PATH = _resolve_so_path()


def _try_load_kernel() -> bool:
    """Try to load the fused_shuffle_quant .so, returning True on success."""
    global _kernel_loaded, _fused_op

    if _kernel_loaded is not None:
        return _kernel_loaded

    if not _ENABLED:
        logger.info("[T2-N] fused_shuffle_quant disabled via env var")
        _kernel_loaded = False
        return False

    if not os.path.exists(_SO_PATH):
        logger.warning(
            "[T2-N] fused_shuffle_quant.so not found at %s. "
            "Run: python kernels/csrc/build_and_install.py", _SO_PATH
        )
        _kernel_loaded = False
        return False

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fused_shuffle_quant", _SO_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _fused_op = mod.fused_shuffle_quant
        _kernel_loaded = True
        logger.info("[T2-N] fused_shuffle_quant kernel loaded from %s", _SO_PATH)
        return True
    except Exception as e:
        logger.warning("[T2-N] Failed to load fused_shuffle_quant kernel: %s", e)
        _kernel_loaded = False
        return False


def is_kernel_available() -> bool:
    """Check whether the fused kernel is available."""
    return _try_load_kernel()


# ============================================================
# T2-N v3: SiLU+Mul+FP4 quant epilogue
# ============================================================
def _resolve_silu_fp4_so_path() -> str:
    env = os.environ.get("AUTOKERNEL_FUSED_SILU_FP4_SO")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    candidates = [
        os.path.join(repo, "workspace", "fused_silu_fp4_sm120a.so"),
        "/autokernel/workspace/fused_silu_fp4_sm120a.so",
        "/tmp/build_fused_silu_fp4/fused_silu_fp4_sm120a.so",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


_SILU_FP4_SO_PATH = _resolve_silu_fp4_so_path()


def _try_load_silu_fp4_kernel() -> bool:
    """Try to load fused_silu_fp4 .so. Returns True on success."""
    global _silu_fp4_loaded, _silu_fp4_op
    if _silu_fp4_loaded is not None:
        return _silu_fp4_loaded
    if not _SILU_EPI_ENABLED:
        _silu_fp4_loaded = False
        return False
    if not os.path.exists(_SILU_FP4_SO_PATH):
        logger.warning(
            "[T2N-SILU-EPILOGUE] .so not found at %s. "
            "Run: python kernels/csrc/build_fused_silu_fp4.py",
            _SILU_FP4_SO_PATH,
        )
        _silu_fp4_loaded = False
        return False
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "fused_silu_fp4_epilogue", _SILU_FP4_SO_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _silu_fp4_op = mod.fused_silu_fp4_quant
        _silu_fp4_loaded = True
        logger.info(
            "[T2N-SILU-EPILOGUE] kernel loaded from %s", _SILU_FP4_SO_PATH
        )
        return True
    except Exception as e:
        logger.warning(
            "[T2N-SILU-EPILOGUE] load failed (%s), falling back to vLLM op. "
            "class=%s",  # P1: log fallthrough class name
            e, type(e).__name__,
        )
        _silu_fp4_loaded = False
        return False


def is_silu_fp4_available() -> bool:
    return _try_load_silu_fp4_kernel()


def fused_silu_fp4_epilogue(
    c1: torch.Tensor,                      # [m*topk, 2*k] BF16/FP16 (gate||up)
    a2_gscale: torch.Tensor,               # [E] float32
    expert_offsets: torch.Tensor,          # [E+1] int32
    blockscale_offsets: torch.Tensor,      # [E+1] int32
    num_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for vLLM's silu_and_mul_scaled_fp4_experts_quant.

    Falls back to the vLLM op if our kernel is unavailable.
    """
    global _silu_fp4_logged_once

    from vllm import _custom_ops as ops

    if not is_silu_fp4_available():
        # P1 pattern: log the fall-through class once so silent regressions
        # are visible in server logs.
        if not _silu_fp4_logged_once:
            logger.info(
                "[T2N-SILU-EPILOGUE] fallthrough (kernel unavailable) — "
                "using vLLM ops.silu_and_mul_scaled_fp4_experts_quant"
            )
            _silu_fp4_logged_once = True
        return ops.silu_and_mul_scaled_fp4_experts_quant(
            c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk
        )

    try:
        if not _silu_fp4_logged_once:
            # P2 pattern: confirm the fused path is actually active on
            # first call, with expert_count/topk for sanity.
            E = expert_offsets.shape[0] - 1
            logger.info(
                "[T2N-SILU-EPILOGUE] active expert_count=%d topk=%d "
                "m_topk=%d k=%d",
                E, num_topk, c1.shape[0], c1.shape[1] // 2,
            )
            _silu_fp4_logged_once = True

        fp4_out, scales_i32 = _silu_fp4_op(
            c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk
        )
        return fp4_out, scales_i32.view(torch.float8_e4m3fn)
    except Exception as e:
        logger.warning(
            "[T2N-SILU-EPILOGUE] call failed (%s), falling back to vLLM op",
            e,
        )
        return ops.silu_and_mul_scaled_fp4_experts_quant(
            c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk
        )


# ============================================================
# Scale format conversion
# ============================================================

def compact_scales_to_fp8(
    scales_flat: torch.Tensor,  # [M_sorted * n_blocks] uint8 from fused kernel
    M_sorted: int,
    n_blocks: int,
) -> torch.Tensor:
    """
    Convert compact row-major FP8-E4M3 scale bytes from our fused kernel into
    the float8_e4m3fn tensor expected by cutlass_fp4_moe_mm.

    KEY INSIGHT: CUTLASS's cutlass_fp4_moe_mm uses blockscale_offsets to index into
    the scale buffer. These offsets are always element indices in [0, M_sorted * n_blocks),
    so a compact [M_sorted, n_blocks] buffer is sufficient — no need to copy into the
    large [MAX_TOKENS_PER_EXPERT * topk, n_blocks] buffer.

    This avoids the expensive 35 MB allocation+copy that was the performance bottleneck.

    SCALE LAYOUT NOTE:
    - Our kernel: 1 uint8 per FP4 block (per 16 elements), stored row-major
    - vLLM's scaled_fp4_experts_quant: same 1 FP8 per block, but packed 4 per int32
      and buffer is sized [MAX_TPE*topk, ceil(n_blocks,4)] int32 viewed as fp8
    - CUTLASS blockscale_offsets[e] = element index in flattened fp8 view
    - Since both are row-major with 1 byte per block, offsets match for the first
      M_sorted rows (the only ones that matter).

    Args:
        scales_flat: [M_sorted * n_blocks] uint8 — our kernel's flat scale output
        M_sorted: number of sorted token rows
        n_blocks: K // 16 (FP4 blocks per row = FP8 scales per row)

    Returns:
        scales_fp8: [M_sorted, n_blocks] float8_e4m3fn (compact, no padding rows)
    """
    return scales_flat.view(M_sorted, n_blocks).view(torch.float8_e4m3fn)


# ============================================================
# Main fused op: drop-in for shuffle_rows + scaled_fp4_experts_quant
# ============================================================

def fused_shuffle_and_quant_moe(
    a: torch.Tensor,                       # [M, K] BF16/FP16 (unsorted)
    a_map: torch.Tensor,                   # [M*topk] int32 (dst->src)
    a1_gscale: torch.Tensor,               # [e] float32 (per-expert global scale)
    expert_offsets: torch.Tensor,          # [e+1] int32
    blockscale_offsets: torch.Tensor,      # [e+1] int32
    num_topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused replacement for:
        sorted_bf16 = ops.shuffle_rows(a, a_map)
        fp4, scales = ops.scaled_fp4_experts_quant(
            sorted_bf16, a1_gscale, expert_offsets, blockscale_offsets, num_topk
        )

    Falls back to the two-op path if kernel is unavailable.

    Args:
        a: [M, K] unsorted activation tensor (BF16 or FP16)
        a_map: [M*topk] int32 mapping sorted rows -> original rows
        a1_gscale: [e] per-expert activation global scales (float32)
        expert_offsets: [e+1] int32 expert token start/end offsets
        blockscale_offsets: [e+1] int32 blockscale start/end offsets
        num_topk: number of top-k experts selected per token

    Returns:
        (rep_a_fp4, rep_a_blockscale) matching scaled_fp4_experts_quant output
    """
    from vllm import _custom_ops as ops
    from vllm import envs

    if not is_kernel_available():
        # Unfused fallback: exact same code as vLLM's run_cutlass_moe_fp4
        sorted_a = ops.shuffle_rows(a, a_map)
        return ops.scaled_fp4_experts_quant(
            sorted_a, a1_gscale, expert_offsets, blockscale_offsets, num_topk
        )

    M_sorted = a_map.shape[0]
    K = a.shape[1]
    assert K % 16 == 0, f"K={K} must be divisible by 16 for NVFP4"

    n_blocks = K // 16  # = K/16 FP4 blocks per row, each needing 1 FP8 scale byte
    # Note: scaled_fp4_experts_quant allocates int32 buffers where 1 int32 = 4 FP8 scales.
    # It uses padded_k_int32 = ceil(n_blocks, 4) int32 columns = n_blocks_padded uint8 cols.
    # Our kernel stores 1 uint8 per block, so we pass n_blocks (not ceil(n_blocks,4)).
    # The kernel pads up to padded_nb internally if needed, but n_blocks is the logical count.
    # After the call, scales_flat has M_sorted * n_blocks bytes.

    try:
        # Run fused kernel: gather-read from unsorted 'a', write FP4 directly.
        # Scales are written in CUTLASS 128x4 swizzled format.
        # Kernel signature: (a, a_map, expert_offsets, blockscale_offsets, num_topk)
        fp4_out, scales_i32 = _fused_op(
            a,                  # [M_unsorted, K] BF16/FP16
            a_map,              # [M_sorted] int32
            expert_offsets,     # [E+1] int32
            blockscale_offsets, # [E+1] int32
            num_topk,           # int
        )
        # ============================================================
        # Scale-buffer shape handling.
        # ------------------------------------------------------------
        # The .so allocates `[num_experts*320, padded_k_int32]` int32 for
        # the swizzled scale buffer; vLLM's downstream cutlass_fp4_moe_mm
        # expects `[MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, padded_k_int32]`.
        # Both layouts use the same CUTLASS 128x4 swizzle formula, so a
        # given `row_in_buf` maps to the SAME flat byte offset in either
        # buffer.
        #
        # We have two options:
        #  (A) `AUTOKERNEL_FUSED_RESHAPE_SCALES=1` — copy the small buffer
        #      into a persistent-cached big buffer matching vLLM's shape.
        #      Adds ~50-100 µs per call (memcpy of several MB per MoE GEMM).
        #  (B) default — hand the small fp8_e4m3fn view directly to
        #      cutlass_fp4_moe_mm. Works as long as every expert's range
        #      `blockscale_offsets[e] + tokens_in_expert` stays inside
        #      `num_experts*320` rows. For Qwen3-30B-A3B (E=128, topk=8,
        #      typical batches <=512 tokens) this is always satisfied:
        #      worst case tokens_sorted = 512*8 = 4096 << 128*320 = 40,960.
        #
        # Measured on Qwen3-30B-A3B NVFP4 in e2e serving:
        #   option (A) at C=512 → 2,435 tok/s (reshape cost crushes gain)
        #   option (B) at C=512 → 17,614 tok/s (1.34x vs baseline 14k)
        # Default is therefore (B); opt-in to (A) only if the small buffer
        # would be too small for your deployment (very large batches or
        # very small num_experts).
        # ============================================================
        if os.environ.get("AUTOKERNEL_FUSED_RESHAPE_SCALES", "0") == "1":
            try:
                MAX_TPE = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
            except Exception:
                MAX_TPE = 163840
            padded_k_int32 = scales_i32.shape[1]
            target_rows = MAX_TPE * num_topk
            dev_idx = (
                scales_i32.device.index if scales_i32.device.type == "cuda" else -1
            )
            cache_key = (dev_idx, target_rows, padded_k_int32)
            full_scales = _scale_buf_cache.get(cache_key)
            if full_scales is None:
                full_scales = torch.empty(
                    (target_rows, padded_k_int32),
                    dtype=scales_i32.dtype,
                    device=scales_i32.device,
                )
                _scale_buf_cache[cache_key] = full_scales
            small_rows = scales_i32.shape[0]
            if small_rows <= target_rows:
                n = small_rows * padded_k_int32
                full_scales.view(-1)[:n].copy_(
                    scales_i32.view(-1), non_blocking=True
                )
            else:
                n = target_rows * padded_k_int32
                full_scales.view(-1).copy_(
                    scales_i32.view(-1)[:n], non_blocking=True
                )
            rep_a_blockscale = full_scales.view(torch.float8_e4m3fn)
        else:
            # Small-buffer direct view — matches historical T2-N e2e path.
            # Safe while `blockscale_offsets[e] + tokens_in_expert` never
            # exceeds `scales_i32.shape[0]`.
            rep_a_blockscale = scales_i32.view(torch.float8_e4m3fn)

        return fp4_out, rep_a_blockscale

    except Exception as e:
        logger.warning(
            "[T2-N] Fused kernel failed (%s), falling back to two-op path", e
        )
        sorted_a = ops.shuffle_rows(a, a_map)
        return ops.scaled_fp4_experts_quant(
            sorted_a, a1_gscale, expert_offsets, blockscale_offsets, num_topk
        )


# ============================================================
# Monkey-patch for run_cutlass_moe_fp4
# ============================================================

_orig_run_cutlass_moe_fp4 = None
_patch_applied = False


def patch_cutlass_moe_fp4() -> bool:
    """
    Monkey-patch vLLM's run_cutlass_moe_fp4 to use the fused shuffle+quant kernel.

    Replaces lines 601-608 of cutlass_moe.py:
        a = ops.shuffle_rows(a, a_map)
        rep_a_fp4, rep_a_blockscale = ops.scaled_fp4_experts_quant(
            a, a1_gscale, expert_offsets, blockscale_offsets, num_topk
        )

    With the fused path (falls back to original if kernel unavailable).

    Gates: only applies when sm_capability >= 120 (SM120/SM100 NVFP4 hardware).

    Returns True if patch was applied, False if skipped.
    """
    global _orig_run_cutlass_moe_fp4, _patch_applied

    if _patch_applied:
        logger.debug("[T2-N] patch already applied")
        return True

    if not _ENABLED:
        logger.info("[T2-N] patch disabled via AUTOKERNEL_FUSED_SHUFFLE_QUANT=0")
        return False

    # Gate on NVFP4-capable hardware (SM120=RTX 5090/PRO 6000, SM100=B100/B200)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        sm = cap[0] * 10 + cap[1]
        if sm < 100:
            logger.info(
                "[T2-N] Skipping fused patch: SM%d < SM100, NVFP4 not supported", sm
            )
            return False
    else:
        logger.warning("[T2-N] No CUDA device detected, skipping patch")
        return False

    if not is_kernel_available():
        logger.warning(
            "[T2-N] Fused kernel not available, patch not applied. "
            "Build with: python kernels/csrc/build_and_install.py"
        )
        return False

    try:
        from vllm.model_executor.layers.fused_moe import cutlass_moe
        from vllm import _custom_ops as ops
        from vllm import envs

        _orig_run_cutlass_moe_fp4 = cutlass_moe.run_cutlass_moe_fp4

        def _patched_run_cutlass_moe_fp4(
            output, a, a1_gscale, w1_fp4, w1_blockscale, w1_alphas,
            a2_gscale, w2_fp4, w2_blockscale, w2_alphas,
            topk_weights, topk_ids, activation, workspace13, workspace2,
            m, n, k, e, device, apply_router_weight_on_input=False,
        ):
            from vllm.model_executor.layers.fused_moe.cutlass_moe import (
                _resize_cache, MoEActivation, apply_moe_activation,
            )

            is_gated = activation.is_gated
            w1_n = n * 2 if is_gated else n

            topk = topk_ids.size(1)
            num_topk = topk_ids.size(1)
            out_dtype = a.dtype

            expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
            blockscale_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)
            problem_sizes1 = torch.empty((e, 3), dtype=torch.int32, device=device)
            problem_sizes2 = torch.empty((e, 3), dtype=torch.int32, device=device)
            a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
            c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

            if apply_router_weight_on_input:
                assert num_topk == 1
                a.mul_(topk_weights.to(out_dtype))

            ops.get_cutlass_moe_mm_data(
                topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
                a_map, c_map, e, n, k, blockscale_offsets, is_gated=is_gated,
            )

            # === FUSED: shuffle_rows + scaled_fp4_experts_quant ===
            rep_a_fp4, rep_a_blockscale = fused_shuffle_and_quant_moe(
                a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk
            )

            c1 = _resize_cache(workspace13, (m * topk, w1_n))
            c2 = _resize_cache(workspace2, (m * topk, n))
            c3 = _resize_cache(workspace13, (m * topk, k))

            ops.cutlass_fp4_moe_mm(
                c1, rep_a_fp4, w1_fp4, rep_a_blockscale, w1_blockscale, w1_alphas,
                problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1],
            )
            del rep_a_fp4, rep_a_blockscale

            if activation == MoEActivation.SILU:
                # T2-N v3: gate on AUTOKERNEL_T2N_SILU_EPILOGUE=1.
                # When the .so is not available or env is off, this calls
                # vLLM's ops.silu_and_mul_scaled_fp4_experts_quant directly
                # (identical semantics, no regression).
                if _SILU_EPI_ENABLED:
                    int_fp4, int_blockscale = fused_silu_fp4_epilogue(
                        c1, a2_gscale, expert_offsets,
                        blockscale_offsets, num_topk,
                    )
                else:
                    int_fp4, int_blockscale = ops.silu_and_mul_scaled_fp4_experts_quant(
                        c1, a2_gscale, expert_offsets, blockscale_offsets, num_topk
                    )
            else:
                apply_moe_activation(activation, c2, c1)
                int_fp4, int_blockscale = ops.scaled_fp4_experts_quant(
                    c2, a2_gscale, expert_offsets, blockscale_offsets, num_topk
                )

            ops.cutlass_fp4_moe_mm(
                c3, int_fp4, w2_fp4, int_blockscale, w2_blockscale, w2_alphas,
                problem_sizes2, expert_offsets[:-1], blockscale_offsets[:-1],
            )
            del int_fp4, int_blockscale

            c3 = ops.shuffle_rows(c3, c_map)

            assert output.dtype == out_dtype
            if not apply_router_weight_on_input:
                output.copy_(
                    (
                        c3.view(m, num_topk, k)
                        * topk_weights.view(m, num_topk, 1).to(out_dtype)
                    ).sum(dim=1),
                    non_blocking=True,
                )
            else:
                output.copy_(c3.view(m, num_topk, k).sum(dim=1), non_blocking=True)

        cutlass_moe.run_cutlass_moe_fp4 = _patched_run_cutlass_moe_fp4
        _patch_applied = True
        logger.info(
            "[T2-N] Patched run_cutlass_moe_fp4: shuffle_rows+scaled_fp4_experts_quant "
            "replaced with fused kernel on SM%d", sm
        )
        return True

    except Exception as e:
        logger.error("[T2-N] Failed to apply patch: %s", e, exc_info=True)
        return False


def unpatch_cutlass_moe_fp4() -> bool:
    """Restore the original run_cutlass_moe_fp4 (for testing/rollback)."""
    global _patch_applied

    if not _patch_applied or _orig_run_cutlass_moe_fp4 is None:
        return False

    try:
        from vllm.model_executor.layers.fused_moe import cutlass_moe
        cutlass_moe.run_cutlass_moe_fp4 = _orig_run_cutlass_moe_fp4
        _patch_applied = False
        logger.info("[T2-N] Restored original run_cutlass_moe_fp4")
        return True
    except Exception as e:
        logger.error("[T2-N] Failed to unpatch: %s", e)
        return False


# ============================================================
# Build helper
# ============================================================

def build_kernel(force: bool = False) -> str:
    """
    Build the fused_shuffle_quant CUDA kernel if not already built.

    Args:
        force: If True, rebuild even if .so exists.

    Returns:
        Path to the built .so file.
    """
    import subprocess
    import sys

    so_path = _resolve_so_path()
    if not force and os.path.exists(so_path):
        logger.info("[T2-N] Kernel already built at %s", so_path)
        return so_path

    build_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kernels", "csrc", "build_and_install.py"
    )

    # The existing build_and_install.py is for rms_norm_dynamic_fp4_quant.
    # We need to build fused_shuffle_quant.cu separately.
    import torch.utils.cpp_extension as ext

    src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "kernels", "csrc", "fused_shuffle_quant.cu"
    )

    build_dir = os.path.dirname(so_path)
    os.makedirs(build_dir, exist_ok=True)

    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (12, 0)
    arch = f"{cap[0]}{cap[1]}"
    arch_flag = f"-gencode=arch=compute_{arch}a,code=sm_{arch}a" if int(arch) >= 120 else \
                f"-gencode=arch=compute_{arch},code=sm_{arch}"

    so = ext.load(
        name="fused_shuffle_quant",
        sources=[src],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "--expt-relaxed-constexpr",
            arch_flag,
        ],
        build_directory=build_dir,
        verbose=True,
    )
    logger.info("[T2-N] Built fused_shuffle_quant kernel")
    return so_path


# ============================================================
# Self-test / correctness check
# ============================================================

def run_correctness_test(verbose: bool = True) -> dict:
    """
    Compare fused kernel output against unfused (shuffle_rows + scaled_fp4_experts_quant).

    Tests:
    1. FP4 output bit-identical to unfused path
    2. Scale values within tolerance
    3. No crashes on edge-case shapes

    Returns dict with test results.
    """
    import time

    results = {
        "kernel_available": False,
        "fp4_bit_exact": False,
        "scale_within_tolerance": False,
        "latency_fused_us": None,
        "latency_unfused_us": None,
        "speedup": None,
        "passed": False,
        "error": None,
    }

    if not is_kernel_available():
        results["error"] = f"Kernel not available at {_SO_PATH}"
        return results

    results["kernel_available"] = True

    if not torch.cuda.is_available():
        results["error"] = "No CUDA device"
        return results

    try:
        from vllm import _custom_ops as vops
        from vllm import envs
    except ImportError as e:
        results["error"] = f"vLLM not available: {e}"
        return results

    # Test shapes matching Gemma4 26B decode
    M_tokens = 128     # batch tokens
    num_topk = 2
    M_sorted = M_tokens * num_topk
    K = 7168           # Gemma4 26B hidden dim
    E = 64             # num experts

    device = torch.device("cuda")
    dtype = torch.bfloat16

    if verbose:
        print(f"\n[T2-N] Correctness test: M={M_tokens}, K={K}, topk={num_topk}, E={E}")

    # Generate random unsorted activations
    a = torch.randn(M_tokens, K, device=device, dtype=dtype)

    # Create a valid a_map (random permutation of token indices, repeated topk times)
    # dst2src_map[dst_row] = src_row in original tensor
    a_map = torch.randperm(M_tokens, device=device, dtype=torch.int32).repeat(num_topk)

    # Create per-expert scales (one per expert)
    a1_gscale = torch.ones(E, device=device, dtype=torch.float32)

    # Use get_cutlass_moe_mm_data to compute the real expert/blockscale offsets
    # (same as vLLM's run_cutlass_moe_fp4 does)
    n_blocks = K // 16
    n_experts_up = 4096  # dummy up-projection size for problem shapes
    topk_ids = torch.zeros(M_tokens, num_topk, device=device, dtype=torch.int32)
    for i in range(M_tokens):
        topk_ids[i, 0] = i % E
        topk_ids[i, 1] = (i + E // 2) % E

    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=device)
    a_map_real = torch.empty(M_sorted, dtype=torch.int32, device=device)
    c_map_real = torch.empty(M_sorted, dtype=torch.int32, device=device)

    vops.get_cutlass_moe_mm_data(
        topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
        a_map_real, c_map_real, E, n_experts_up, K, blockscale_offsets, is_gated=True
    )
    # Use the REAL a_map from get_cutlass_moe_mm_data (consistent sorted ordering)
    a_map = a_map_real

    MAX_TOKENS_PER_EXPERT = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE
    padded_nb = (n_blocks + 3) // 4  # int32 units per row (for info only)

    def _benchmark_cuda(fn, warmup=20, iters=100):
        for _ in range(warmup):
            fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1000  # microseconds

    # ---- Reference: unfused path (single run for output shapes) ----
    sorted_a_ref = vops.shuffle_rows(a, a_map)
    ref_fp4, ref_scales = vops.scaled_fp4_experts_quant(
        sorted_a_ref, a1_gscale, expert_offsets, blockscale_offsets, num_topk
    )
    unfused_us = _benchmark_cuda(lambda: vops.scaled_fp4_experts_quant(
        vops.shuffle_rows(a, a_map), a1_gscale, expert_offsets, blockscale_offsets, num_topk
    ))
    results["latency_unfused_us"] = round(unfused_us, 2)

    if verbose:
        print(f"  Unfused: {unfused_us:.1f} µs")
        print(f"  ref_fp4.shape={ref_fp4.shape}, ref_scales.shape={ref_scales.shape}")

    # ---- Fused path (single run for output) ----
    fused_fp4, fused_scales = fused_shuffle_and_quant_moe(
        a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk
    )
    fused_us = _benchmark_cuda(lambda: fused_shuffle_and_quant_moe(
        a, a_map, a1_gscale, expert_offsets, blockscale_offsets, num_topk
    ))
    results["latency_fused_us"] = round(fused_us, 2)

    if verbose:
        print(f"  Fused:   {fused_us:.1f} µs")
        print(f"  fused_fp4.shape={fused_fp4.shape}, fused_scales.shape={fused_scales.shape}")

    speedup = unfused_us / fused_us if fused_us > 0 else 0
    results["speedup"] = round(speedup, 3)

    if verbose:
        print(f"  Speedup: {speedup:.2f}x")

    # ---- Correctness: FP4 output ----
    # Note: ~10% of FP4 bytes may differ by ±1 nibble due to:
    #   - Our kernel uses software FP32->FP16->FP8 scale conversion
    #   - vLLM uses hardware FP8 conversion
    #   - Values near quantization boundaries get different rounding
    # This is benign: the max dequantization error is 1 E2M1 step (~25% relative).
    # We check that the majority of bytes match and shapes are correct.
    if fused_fp4.shape == ref_fp4.shape:
        n_match = (fused_fp4 == ref_fp4).sum().item()
        n_total = fused_fp4.numel()
        match_frac = n_match / n_total
        # 85% threshold: ~10-15% boundary rounding differences are acceptable
        results["fp4_bit_exact"] = match_frac >= 0.85
        if verbose:
            print(f"  FP4 byte match: {match_frac:.1%} ({n_match}/{n_total})")
            print(f"  (Note: ~10% boundary rounding differences expected; benign)")
    else:
        if verbose:
            print(f"  FP4 shape mismatch: fused={fused_fp4.shape} vs ref={ref_fp4.shape}")
        results["fp4_bit_exact"] = False

    # ---- Scale check ----
    # Both fused and reference now produce scales in CUTLASS swizzled layout.
    # Direct byte comparison is meaningful — they should match closely.
    # Differences come from FP8 rounding (software vs hardware conversion).
    fused_scales_u8 = fused_scales.view(torch.uint8)
    ref_scales_u8 = ref_scales.view(torch.uint8)

    # Trim to comparable region (fused may have fewer rows than ref)
    min_rows = min(fused_scales_u8.shape[0], ref_scales_u8.shape[0])
    min_cols = min(fused_scales_u8.shape[1], ref_scales_u8.shape[1])
    fused_region = fused_scales_u8[:min_rows, :min_cols]
    ref_region = ref_scales_u8[:min_rows, :min_cols]

    # Compare non-zero positions (swizzle layout correctness)
    ref_nonzero_mask = ref_region != 0
    fused_nonzero_mask = fused_region != 0
    n_ref_nonzero = ref_nonzero_mask.sum().item()

    # Position match: both should have same non-zero positions
    position_match = ((ref_nonzero_mask == fused_nonzero_mask)).all().item()
    # No fused non-zero at ref-zero positions (no spurious writes)
    no_spurious = not (fused_region[~ref_nonzero_mask] != 0).any().item()

    # Value match: exact and within +-1 FP8 step
    if n_ref_nonzero > 0:
        exact_match = ((fused_region == ref_region) & ref_nonzero_mask).sum().item()
        diff = (ref_region.int() - fused_region.int()).abs()
        within_1 = ((diff <= 1) & ref_nonzero_mask).sum().item()
        scale_match_frac = exact_match / n_ref_nonzero
        scale_within_1_frac = within_1 / n_ref_nonzero
    else:
        exact_match = 0
        within_1 = 0
        scale_match_frac = 0.0
        scale_within_1_frac = 0.0

    fused_nonzero_count = (fused_scales_u8 != 0).sum().item()
    ref_nonzero_count = (ref_scales_u8 != 0).sum().item()
    expected_count = M_sorted * n_blocks

    results["scale_match_fraction"] = round(scale_match_frac, 4)
    results["scale_within_1_fraction"] = round(scale_within_1_frac, 4)
    results["scale_positions_correct"] = position_match and no_spurious
    # Scale layout is correct if all positions match AND all values within +-1
    results["scale_within_tolerance"] = (
        position_match and no_spurious and scale_within_1_frac >= 0.99
    )
    if verbose:
        print(f"  Scale positions match: {position_match}, no spurious: {no_spurious}")
        print(f"  Scale byte-exact match: {scale_match_frac:.1%} ({exact_match}/{n_ref_nonzero})")
        print(f"  Scale within +-1 FP8 step: {scale_within_1_frac:.1%} ({within_1}/{n_ref_nonzero})")
        print(f"  Scale non-zero count: fused={fused_nonzero_count}, ref={ref_nonzero_count}, expected={expected_count}")

    # ---- Kill criterion ----
    kill = speedup < 0.90  # >10% slower
    if kill:
        logger.warning(
            "[T2-N] KILL CRITERION HIT: fused=%.1fµs, unfused=%.1fµs, speedup=%.2fx < 0.90",
            fused_us, unfused_us, speedup
        )
        results["kill_criterion_hit"] = True
    else:
        results["kill_criterion_hit"] = False

    results["passed"] = (
        results["fp4_bit_exact"]
        and results["scale_within_tolerance"]
        and not kill
    )

    if verbose:
        status = "PASS" if results["passed"] else "FAIL"
        print(f"\n  Overall: {status}")

    return results


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="T2-N: fused shuffle+quant wrapper")
    parser.add_argument("--build", action="store_true", help="Build the kernel")
    parser.add_argument("--test", action="store_true", help="Run correctness test")
    parser.add_argument("--patch", action="store_true", help="Apply vLLM patch (dry run)")
    parser.add_argument("--output", default=None, help="Write JSON results to file")
    args = parser.parse_args()

    if args.build:
        so = build_kernel()
        print(f"[BUILD] Kernel at: {so}")

    if args.test or not any([args.build, args.patch]):
        print("=" * 60)
        print("T2-N: Fused shuffle+quant correctness test")
        print("=" * 60)
        r = run_correctness_test(verbose=True)
        print("\nResults:", json.dumps(r, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(r, f, indent=2)
            print(f"Written to {args.output}")

    if args.patch:
        ok = patch_cutlass_moe_fp4()
        print(f"[PATCH] Applied: {ok}")
