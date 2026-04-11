"""Generate decode and store functions from a KVCacheSpec."""

import math

import torch
import triton

from kv_cache_gen.spec import KVCacheSpec
from kv_cache_gen.kernel import (
    _universal_decode_stage1, _universal_decode_stage2, _universal_store_kernel,
)


# Split-K lookup table: seq_len → optimal num_kv_splits (for low-batch cases)
# Derived from bench_seqlen_scaling.py results on RTX 5090.
# At short sequences, fewer splits avoids underutilization.
# At long sequences, more splits improves bandwidth saturation.
_SPLITS_LOOKUP = [
    (512,    16),
    (1024,   32),
    (2048,   32),
    (8192,   64),
    (32768,  64),
    (65536,  128),
]


def _optimal_splits_seqlen(max_seq_len: int) -> int:
    """Select base num_kv_splits from sequence length (low-batch path)."""
    for threshold, splits in _SPLITS_LOOKUP:
        if max_seq_len <= threshold:
            return splits
    return 128


def _optimal_splits(max_seq_len: int, batch_size: int, num_sms: int = 170) -> int:
    """Adaptive split-K: fewer splits when batch is large.

    Goal: enough total blocks (B x splits) to keep all SMs busy,
    but not so many that mid_out memory explodes.

    At B=1:   32 splits  (32 blocks per head group)
    At B=4:   32 splits  (seq_len lookup dominates at low batch)
    At B=16:  16 splits  (256 blocks -- good SM utilization)
    At B=32:   8 splits  (256 blocks)
    At B=128:  2 splits  (256 blocks -- batch nearly saturates)
    At B=256:  1 split   (256 blocks -- batch alone saturates)

    The max is capped at 32 to bound mid_out memory. At very long
    sequences with low batch, the seq_len lookup may suggest more,
    but we clamp to 32.
    """
    # Batch-adaptive: target ~2 blocks per SM for good occupancy
    # (with Hq=16 head groups, each batch element launches multiple blocks,
    #  so 2x multiplier keeps SMs busy without exploding mid_out)
    target_blocks = num_sms * 2  # 340 on RTX 5090
    batch_splits = max(1, target_blocks // max(batch_size, 1))
    # Clamp to power of 2, max 32 to bound memory
    batch_splits = min(32, max(1, 2 ** int(math.log2(max(batch_splits, 1)))))

    # For low-batch + long seq, seq_len lookup may push splits higher
    # (but still capped at 32)
    if batch_size <= 4:
        seqlen_splits = _optimal_splits_seqlen(max_seq_len)
        return min(32, max(batch_splits, seqlen_splits))

    return batch_splits


def make_decode_fn(spec: KVCacheSpec, block_kv=16, block_h=None,
                    num_warps=4, num_stages=1, num_kv_splits=None,
                    max_seq_len=None, max_batch_size=None,
                    persistent_buffers=False, cuda_graph_safe=False,
                    logits_soft_cap=0.0,
                    shared_mid_out=None, shared_output=None):
    """Generate a Triton decode function from a KVCacheSpec.

    Args:
        spec: KV cache quantization spec
        block_kv: KV tokens per Triton block (8, 16, 32)
        block_h: heads per block (auto if None)
        num_warps: Triton warps (2, 4, 8)
        num_stages: pipeline stages (1, 2)
        num_kv_splits: split-KV parallelism. If None, auto-selected from
            max_seq_len and max_batch_size (no runtime overhead).
            Explicit value overrides auto-selection.
        max_seq_len: hint for auto split-K selection (avoids GPU sync)
        max_batch_size: hint for auto split-K selection
        persistent_buffers: if True, pre-allocate mid_out and output tensors
            at first call and reuse them (eliminates ~300MB alloc churn at B=240).
            Requires fixed B/Hq/D across calls (typical for decode).
        cuda_graph_safe: if True, enables persistent_buffers and ensures no
            Python-level tensor allocations in the hot path. The returned
            function is safe for torch.cuda.CUDAGraph capture.
        logits_soft_cap: if > 0, applies tanh soft capping to attention logits
            before softmax: score = cap * tanh(score / cap). Used by Gemma models.
        shared_mid_out: if provided, a pre-allocated [max_B, Hq, NUM_KV_SPLITS, D+1]
            float32 tensor shared across all CUDA graph capture sizes. Eliminates
            per-graph buffer allocation (saves 10+ GiB with many graph sizes).
        shared_output: if provided, a pre-allocated [max_B, Hq, D] tensor shared
            across all CUDA graph capture sizes. Must be same dtype as query.

    Returns a callable: (query, kv_cache, scales, block_table, seq_lens,
                          scale, num_kv_heads) → output
    """
    _block_kv = block_kv
    _block_h = block_h
    _num_warps = num_warps
    _num_stages = num_stages
    _persistent = persistent_buffers or cuda_graph_safe
    _cuda_graph_safe = cuda_graph_safe
    _logits_soft_cap = float(logits_soft_cap) if logits_soft_cap else 0.0

    # Split-K configuration:
    # - If explicit num_kv_splits given, use it as a fixed value (no adaptation).
    # - Otherwise, adapt per-call based on actual batch size B.
    # _max_kv_splits is always the upper bound (for buffer pre-allocation).
    _adaptive = (num_kv_splits is None)
    _seq_hint = max_seq_len if max_seq_len is not None else 8192
    if num_kv_splits is not None:
        _fixed_splits = num_kv_splits
        _max_kv_splits = num_kv_splits
    else:
        _fixed_splits = None
        # Max splits = what we'd use at B=1 (highest split count)
        _max_kv_splits = _optimal_splits(_seq_hint, 1)

    # Shared buffers mode: all CUDA graph captures use the same pre-allocated
    # tensors at max_batch_size. The kernel handles padding via seq_lens=0.
    _shared_mid = shared_mid_out
    _shared_out = shared_output
    _use_shared = (_shared_mid is not None and _shared_out is not None)

    # Persistent buffer state (allocated on first call) — only used when
    # shared buffers are NOT provided (legacy per-graph-size allocation).
    _buffers = {}

    def _get_buffers(B, Hq, D, device, out_dtype=torch.float16):
        """Get or allocate persistent mid_out and output buffers.

        Always allocates at _max_kv_splits so the buffer can be reused
        even when adaptive splits chooses a smaller split count.
        """
        key = (B, Hq, D, device, out_dtype)
        if key not in _buffers:
            _buffers[key] = {
                'mid_out': torch.empty(B, Hq, _max_kv_splits, D + 1,
                                       dtype=torch.float32, device=device),
                'output': torch.empty(B, Hq, D, dtype=out_dtype, device=device),
            }
        return _buffers[key]

    def decode(query, kv_cache, scales, block_table, seq_lens,
               scale, num_kv_heads):
        B, Hq, D = query.shape
        Hk = num_kv_heads
        kv_group_size = Hq // Hk
        block_size = kv_cache.shape[1]

        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_KV = _block_kv
        # BLOCK_H: query heads per Triton program block.
        # The kernel handles GQA by reading one KV head for each group
        # of BLOCK_H query heads. VALID_BLOCK_H = min(BLOCK_H, kv_group_size)
        # is the effective number of heads computed per block.
        if _block_h is not None:
            BLOCK_H = _block_h
        else:
            BLOCK_H = min(8, kv_group_size)
        BLOCK_H = max(1, BLOCK_H)
        # The grid parallelizes over head groups using VALID_BLOCK_H
        VALID_BLOCK_H = BLOCK_H if kv_group_size > BLOCK_H else kv_group_size

        # Adaptive split-K: fewer splits at large batch to save memory
        if _adaptive:
            NUM_KV_SPLITS = _optimal_splits(_seq_hint, B)
        else:
            NUM_KV_SPLITS = _fixed_splits

        if _use_shared and B <= _shared_mid.shape[0]:
            # Shared buffers: all CUDA graph captures use the SAME tensor
            # addresses. Pass full max-sized buffers — the kernel skips
            # padded entries where seq_lens=0 (zero output, no OOB).
            mid_out = _shared_mid
            output = _shared_out
        elif _use_shared:
            # Batch size exceeds shared buffer capacity (e.g., eager mode
            # with B > max_cudagraph_capture_size). Allocate temporary
            # buffers. This path is NOT CUDA-graph-safe but that's fine
            # because we're already in eager mode (B > max_capture_size).
            mid_out = torch.empty(B, _shared_mid.shape[1], NUM_KV_SPLITS,
                                  _shared_mid.shape[3],
                                  dtype=torch.float32, device=query.device)
            output = torch.empty(B, _shared_out.shape[1], _shared_out.shape[2],
                                 dtype=_shared_out.dtype, device=query.device)
        elif _persistent:
            bufs = _get_buffers(B, Hq, D, query.device, query.dtype)
            mid_out = bufs['mid_out']
            output = bufs['output']
        else:
            mid_out = torch.empty(B, Hq, NUM_KV_SPLITS, D + 1,
                                  dtype=torch.float32, device=query.device)
            output = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)

        num_head_groups = triton.cdiv(Hq, VALID_BLOCK_H)
        grid1 = (B, num_head_groups, NUM_KV_SPLITS)

        k_region_bytes = int(spec.k_bytes_per_dim * D)
        v_region_start = k_region_bytes

        _universal_decode_stage1[grid1](
            query, kv_cache, scales, block_table, seq_lens, mid_out,
            query.stride(0), query.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0),
            mid_out.stride(0), mid_out.stride(1), mid_out.stride(2),
            scales.stride(0), scales.stride(1), scales.stride(2), scales.stride(3),
            scale,
            HEAD_DIM=D, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
            BLOCK_H=BLOCK_H, NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size, Q_HEAD_NUM=Hq,
            PAGE_SIZE=block_size,
            K_BITS=spec.k_bits, V_BITS=spec.v_bits,
            K_OFFSET=spec.k_sym_offset, V_OFFSET=spec.v_sym_offset,
            SCALE_BLOCK_K=spec.k_scale_block, SCALE_BLOCK_V=spec.v_scale_block,
            SCALE_BLOCK_STORE=min(spec.k_scale_block, spec.v_scale_block),
            K_REGION_BYTES=k_region_bytes,
            V_REGION_START=v_region_start,
            LOGITS_SOFT_CAP=_logits_soft_cap,
            num_warps=_num_warps, num_stages=_num_stages,
        )

        grid2 = (B, Hq)
        _universal_decode_stage2[grid2](
            mid_out, output, seq_lens,
            HEAD_DIM=D, BLOCK_D=BLOCK_D,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            stride_mid_b=mid_out.stride(0), stride_mid_h=mid_out.stride(1),
            stride_mid_s=mid_out.stride(2),
            stride_out_b=output.stride(0), stride_out_h=output.stride(1),
            num_warps=2,
        )
        return output

    decode.__name__ = f"decode_{spec.name}_kv{block_kv}_w{num_warps}"
    decode.spec = spec
    decode.triton_config = {
        "block_kv": block_kv, "block_h": block_h,
        "num_warps": num_warps, "num_stages": num_stages,
    }
    decode.persistent_buffers = _persistent
    decode.cuda_graph_safe = _cuda_graph_safe
    return decode


def make_store_fn(spec: KVCacheSpec):
    """Generate a KV cache store function from a KVCacheSpec.

    Returns a callable: (key, value, kv_cache, slot_mapping, layer,
                          num_kv_heads) → None
    Quantizes K and V, packs into cache, stores scales.

    Uses a Triton kernel for supported configs (k_bits/v_bits in {2,4,8}
    with scale blocks), falling back to PyTorch for unsupported configs.
    """

    # Determine if we can use the Triton store kernel
    _use_triton = (spec.k_bits in (2, 4, 8) and spec.v_bits in (2, 4, 8)
                   and spec.k_scale_block > 0 and spec.v_scale_block > 0)

    def _quantize_symmetric(tensor, bits, scale_block, offset):
        """Quantize tensor to symmetric integer with per-block scales."""
        N, Hk, D = tensor.shape
        levels = 2 ** bits
        mid = (levels - 1) / 2.0  # 7.5 for 4-bit, 127 for 8-bit, 1.5 for 2-bit

        blocks = tensor.float().reshape(N, Hk, D // scale_block, scale_block)
        absmax = blocks.abs().amax(dim=-1, keepdim=True)
        scale = absmax / mid
        codes = (blocks / (scale + 1e-8) + mid).round().clamp(0, levels - 1).to(torch.uint8)
        codes = codes.reshape(N, Hk, D)
        scale = scale.squeeze(-1).half()
        return codes, scale

    def _pack_codes(codes, bits, D):
        """Pack integer codes into bytes."""
        N, Hk = codes.shape[:2]
        if bits == 8:
            return codes  # already 1 byte per value
        elif bits == 4:
            # 2 values per byte
            c = codes.reshape(N, Hk, D // 2, 2).to(torch.int32)
            return (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
        elif bits == 2:
            # 4 values per byte
            c = codes.reshape(N, Hk, D // 4, 4).to(torch.int32)
            return (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).to(torch.uint8)
        else:
            raise ValueError(f"Unsupported bits={bits}")

    def _store_pytorch(key, value, kv_cache, slot_mapping, layer, num_kv_heads):
        """PyTorch fallback store implementation."""
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        D = key.shape[-1]
        Hk = num_kv_heads
        block_size = kv_cache.shape[1]

        k = key[:N]
        v = value[:N]
        if k.ndim == 2:
            k = k.reshape(N, Hk, D)
            v = v.reshape(N, Hk, D)

        # Quantize K
        k_codes, k_scale = _quantize_symmetric(k, spec.k_bits, spec.k_scale_block, spec.k_sym_offset)
        k_packed = _pack_codes(k_codes, spec.k_bits, D)

        # Quantize V
        v_codes, v_scale = _quantize_symmetric(v, spec.v_bits, spec.v_scale_block, spec.v_sym_offset)
        v_packed = _pack_codes(v_codes, spec.v_bits, D)

        # Combine: [K region | V region]
        packed = torch.cat([k_packed, v_packed], dim=2)

        # Scatter into cache
        safe_slot = slot_mapping.clamp(min=0)
        blk_idx = (safe_slot // block_size).long()
        blk_off = (safe_slot % block_size).long()
        slot_size = packed.shape[2]
        kv_cache[blk_idx, blk_off, :, :slot_size] = packed

        # Store scales
        min_block = min(spec.k_scale_block, spec.v_scale_block)
        num_sb = D // min_block
        num_blocks_total = kv_cache.shape[0]
        max_slots = num_blocks_total * block_size
        if not hasattr(layer, '_fc_scales'):
            layer._fc_scales = torch.zeros(
                max_slots, Hk, num_sb, 2, dtype=torch.float16, device=key.device)
        flat_slot = (blk_idx * block_size + blk_off).clamp(0, max_slots - 1)

        k_repeat = spec.k_scale_block // min_block
        k_scale_expanded = k_scale.repeat_interleave(k_repeat, dim=-1)
        layer._fc_scales[flat_slot, :, :, 0] = k_scale_expanded

        v_repeat = spec.v_scale_block // min_block
        v_scale_expanded = v_scale.repeat_interleave(v_repeat, dim=-1)
        layer._fc_scales[flat_slot, :, :, 1] = v_scale_expanded

    def _store_triton(key, value, kv_cache, slot_mapping, layer, num_kv_heads):
        """Triton store: quantize + pack + scatter in a single kernel launch."""
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        D = key.shape[-1]
        Hk = num_kv_heads
        block_size = kv_cache.shape[1]

        k = key[:N]
        v = value[:N]
        if k.ndim == 2:
            k = k.reshape(N, Hk, D)
            v = v.reshape(N, Hk, D)

        # Ensure contiguous (N, Hk, D) layout
        k = k.contiguous()
        v = v.contiguous()

        # Allocate scales tensor if needed
        min_block = min(spec.k_scale_block, spec.v_scale_block)
        num_sb = D // min_block
        num_blocks_total = kv_cache.shape[0]
        max_slots = num_blocks_total * block_size
        if not hasattr(layer, '_fc_scales'):
            layer._fc_scales = torch.zeros(
                max_slots, Hk, num_sb, 2, dtype=torch.float16, device=key.device)

        BLOCK_D = triton.next_power_of_2(D)
        k_region_bytes = int(spec.k_bytes_per_dim * D)
        v_region_start = k_region_bytes

        grid = (N, Hk)

        _universal_store_kernel[grid](
            k, v,
            kv_cache, layer._fc_scales, slot_mapping,
            k.stride(0), k.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            layer._fc_scales.stride(0), layer._fc_scales.stride(1),
            layer._fc_scales.stride(2), layer._fc_scales.stride(3),
            PAGE_SIZE=block_size,
            HEAD_DIM=D, BLOCK_D=BLOCK_D,
            K_BITS=spec.k_bits, V_BITS=spec.v_bits,
            K_OFFSET=spec.k_sym_offset, V_OFFSET=spec.v_sym_offset,
            K_LEVELS_MINUS_1=2**spec.k_bits - 1,
            V_LEVELS_MINUS_1=2**spec.v_bits - 1,
            SCALE_BLOCK_K=spec.k_scale_block,
            SCALE_BLOCK_V=spec.v_scale_block,
            SCALE_BLOCK_STORE=min_block,
            V_REGION_START=v_region_start,
            num_warps=4,
        )

    # Async store: run on a dedicated CUDA stream to overlap with next layer's compute
    _store_stream = None

    def store(key, value, kv_cache, slot_mapping, layer, num_kv_heads):
        if _use_triton:
            _store_triton(key, value, kv_cache, slot_mapping, layer, num_kv_heads)
        else:
            _store_pytorch(key, value, kv_cache, slot_mapping, layer, num_kv_heads)

    def store_async(key, value, kv_cache, slot_mapping, layer, num_kv_heads):
        """Store KV on a separate CUDA stream, overlapping with next layer's compute.

        Call sync_store() before reading from this layer's cache (i.e., before
        the next decode on the same layer, not the next layer in the stack).
        """
        nonlocal _store_stream
        if _store_stream is None:
            _store_stream = torch.cuda.Stream()

        # Record where the default stream is so the store stream waits for
        # the KV tensors to be ready (they were just produced on default stream)
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(_store_stream):
            _store_stream.wait_event(event)
            store(key, value, kv_cache, slot_mapping, layer, num_kv_heads)

    def sync_store():
        """Wait for any in-flight async store to complete."""
        if _store_stream is not None:
            torch.cuda.current_stream().wait_stream(_store_stream)

    store.__name__ = f"store_{spec.name}"
    store.spec = spec
    store.store_async = store_async
    store.sync_store = sync_store
    return store
