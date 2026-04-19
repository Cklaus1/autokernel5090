# SWA Decode num_warps Tune — W5_B2_swa_numwarps_tune

**Status:** CPU edit complete; GPU bench pending (parent will run).
**Baseline:** `swa_decode_attention` @ 2.52× vs FlashInfer (C=8, Gemma4 12K prompt), 30% HBM BW (511 GB/s of 1,792 GB/s peak).

## Rationale

The Stage-1 kernel is purely HBM-bandwidth-bound: each decode step reads `window_pages × page_size × num_kv_heads × head_dim × 2 bytes` from KV cache and does almost no compute (one QK dot per page, trivial). At `num_warps=4` the hardware warp scheduler cannot hide the full L2/HBM latency because only 4 independent warp instruction-streams are in flight per SM. Doubling to `num_warps=8` puts 8 warps in the issue window, doubling memory-level parallelism and allowing the SM to overlap two cache-miss latency chains simultaneously. The `tl.multiple_of(addr, 16)` hints tell the Triton compiler that K and V address tensors are always 16-byte-aligned (true by PyTorch allocator guarantee for BF16: 2 bytes × 8 elements = 16 bytes; for FP8: 1 byte × 16 elements = 16 bytes). With this hint the backend can emit a single `ld.global.v4.b32` (128-bit vector load) instead of four scalar loads, cutting the number of issued memory instructions by 4× and reducing address-generation overhead. Together these changes target moving effective HBM bandwidth from ~30% to 40–50% of peak, projecting a 1.2–1.33× additional latency reduction on top of the banked 2.52× baseline.

## Diff Summary

| File | Line (post-edit) | Before | After |
|---|---|---|---|
| `kernels/triton/swa_decode.py` | 126 | *(absent)* | `k_addrs = tl.multiple_of(k_addrs, 16)` |
| `kernels/triton/swa_decode.py` | 149 | *(absent)* | `v_addrs = tl.multiple_of(v_addrs, 16)` |
| `kernels/triton/swa_decode.py` | 342 | `num_warps=4,` | `num_warps=8,  # W5_B2 ...` |

New file: `kernels/triton/test_swa_decode_correctness.py` — standalone CPU-importable harness; runs BF16 + FP8 paths, 6 cases, cos ≥ 0.999 assertion; skips gracefully on CPU-only machines.

## Expected Behavioural Change

- **BF16 path:** both `multiple_of` hints + `num_warps=8` active. Maximum projected benefit.
- **FP8 path:** `multiple_of` hints equally active (FP8 is 1 byte, 16-alignment still valid); `num_warps=8` shared. The FP8 branch conditionally loads via `k_fp8`/`v_fp8` aliases — hints apply to the same `k_addrs`/`v_addrs` tensors, so both branches benefit.
- **Graph-capture gate** (`torch.cuda.is_current_stream_capturing()` at line 270) is untouched; only the kernel launch config changed.
- **Stage-2 merge kernel** unchanged (it reads FP32 `mid_out`, not KV cache, so BW hints don't apply).

## Risk: Register Pressure at num_warps=8

Doubling warp count halves the register file available per warp on SM80/SM90 (256 KB RF ÷ 8 warps = 32 KB / warp = 32 regs / thread at 32-thread warps). The Stage-1 kernel is register-light: it holds `q` (BLOCK_H × BLOCK_D floats, typically 4 × 128 = 512 floats = 2 KB), `acc`, `e_max`, `e_sum` — well under 32 regs/thread for head_dim=128. For head_dim=256 (Gemma4) the register count roughly doubles but should still fit. If `nvcc`/Triton reports register spill (visible as `spill stores` in PTX/SASS), fall back to `num_warps=4` or try `num_warps=6` (not a power-of-2 but Triton accepts it). The `multiple_of` hints are zero-risk and should be kept regardless of the warp count outcome.
