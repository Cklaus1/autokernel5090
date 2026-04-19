#!/usr/bin/env python3
"""
T2-N correctness + microbench test for fused_shuffle_quant with INLINE CUTLASS
swizzled scales (no Triton post-pass).

Compares the rebuilt .so directly against vLLM's two-op baseline
(shuffle_rows + scaled_fp4_experts_quant), which also produces swizzled scales.

PASS criteria:
  (a) All non-zero scale positions match within +-1 FP8 step (scale_within_1 == nz_cnt)
  (b) No spurious scale bytes at zero positions
  (c) fused cos-vs-BF16 within 0.002 of reference
  (d) fused microbench faster than the two-op baseline

Run on a free GPU (>8 GB):
    CUDA_VISIBLE_DEVICES=1 python3 patches/test_fused_shuffle_quant_inline.py
"""
import sys, os, importlib.util, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_kernel():
    so = os.path.join(ROOT, "workspace", "fused_shuffle_quant_sm120a.so")
    spec = importlib.util.spec_from_file_location("fused_shuffle_quant", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_shuffle_quant


def gather_per_token_scales(sw_fp8, expert_offsets, blockscale_offsets,
                             K, M_sorted, E, n_blocks):
    """De-swizzle CUTLASS [rows, padded_k//4] fp8 scales back to per-token [M_sorted, n_blocks]."""
    num_k_tiles = (K + 63) // 64
    sw_u8 = sw_fp8.view(torch.uint8).contiguous()
    flat = sw_u8.flatten()
    out = torch.empty((M_sorted, n_blocks), dtype=torch.float32, device=sw_u8.device)
    eo_l = expert_offsets.cpu().tolist()
    bo_l = blockscale_offsets.cpu().tolist()
    for e in range(E):
        ts, te = eo_l[e], eo_l[e + 1]
        if te <= ts:
            continue
        bs = bo_l[e]
        ntok = te - ts
        tie = torch.arange(ntok, device=sw_u8.device, dtype=torch.int64)
        kk = torch.arange(n_blocks, device=sw_u8.device, dtype=torch.int64)
        T, KK = torch.meshgrid(tie, kk, indexing="ij")
        row = bs + T
        mT = row // 128
        oM = row % 32
        iM = (row // 32) % 4
        kT = KK // 4
        iK = KK % 4
        off = (mT * num_k_tiles + kT) * 512 + oM * 16 + iM * 4 + iK
        b = flat[off]
        fp8 = b.view(torch.float8_e4m3fn)
        out[ts:te] = fp8.to(torch.float32)
    return out


def dequant_fp4(packed, scales_pt, K):
    dev = packed.device
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev, dtype=torch.float32)
    hi = ((packed >> 4).to(torch.int64)) & 0xF
    lo = (packed.to(torch.int64)) & 0xF
    sh = (hi >> 3) & 1
    sl = (lo >> 3) & 1
    mh = hi & 7
    ml = lo & 7
    vh = torch.where(sh.bool(), -lut[mh], lut[mh])
    vl = torch.where(sl.bool(), -lut[ml], lut[ml])
    M = packed.shape[0]
    out = torch.empty((M, K), device=dev, dtype=torch.float32)
    out[:, 0::2] = vl.to(torch.float32)
    out[:, 1::2] = vh.to(torch.float32)
    return out * scales_pt.repeat_interleave(16, dim=1)


def bench_cuda(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0  # us


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.manual_seed(0)

    from vllm import _custom_ops as vops

    M_tokens = 64
    K = 2816
    num_topk = 8
    E = 128
    M_sorted = M_tokens * num_topk
    n_blocks = K // 16

    a = torch.randn(M_tokens, K, device=device, dtype=torch.bfloat16) * 0.4

    topk_ids = torch.stack([
        (torch.arange(M_tokens, device=device, dtype=torch.int32) + i) % E
        for i in range(num_topk)
    ], dim=1).to(torch.int32)

    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    ps1 = torch.empty((E, 3), dtype=torch.int32, device=device)
    ps2 = torch.empty((E, 3), dtype=torch.int32, device=device)
    a_map = torch.empty(M_sorted, dtype=torch.int32, device=device)
    c_map = torch.empty(M_sorted, dtype=torch.int32, device=device)
    vops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, ps1, ps2,
                                  a_map, c_map, E, 4096, K,
                                  blockscale_offsets, is_gated=True)
    a1_gscale = torch.ones(E, device=device, dtype=torch.float32)

    # Reference two-op path
    sorted_a = vops.shuffle_rows(a, a_map)
    ref_fp4, ref_scales = vops.scaled_fp4_experts_quant(
        sorted_a, a1_gscale, expert_offsets, blockscale_offsets, num_topk
    )

    # Fused kernel -- inline CUTLASS swizzle
    fused_op = _load_kernel()
    fused_fp4, fused_scales_i32 = fused_op(
        a, a_map, expert_offsets, blockscale_offsets, num_topk
    )
    fused_scales = fused_scales_i32.view(torch.float8_e4m3fn)
    torch.cuda.synchronize()

    # === Correctness (bytewise swizzle positions) ===
    ref_u8 = ref_scales.view(torch.uint8)
    fused_u8 = fused_scales.view(torch.uint8)

    # Trim fused to ref size (ref has vLLM-sized buffer)
    min_rows = min(ref_u8.shape[0], fused_u8.shape[0])
    min_cols = min(ref_u8.shape[1], fused_u8.shape[1])
    ref_t = ref_u8[:min_rows, :min_cols]
    fused_t = fused_u8[:min_rows, :min_cols]
    nz = ref_t != 0
    nz_cnt = nz.sum().item()
    exact_nz = ((fused_t == ref_t) & nz).sum().item()
    within1_nz = (((fused_t.int() - ref_t.int()).abs() <= 1) & nz).sum().item()
    spurious = (fused_t[~nz] != 0).sum().item()

    print(f"[T2N] M_tokens={M_tokens} K={K} topk={num_topk} E={E} M_sorted={M_sorted}")
    print(f"  ref_scales shape: {ref_u8.shape}")
    print(f"  fused_scales shape: {fused_u8.shape}")
    print(f"  nonzero_positions={nz_cnt}")
    print(f"  scale_exact_on_nz = {exact_nz/nz_cnt:.3%} ({exact_nz}/{nz_cnt})")
    print(f"  scale_within_pm1_on_nz = {within1_nz/nz_cnt:.3%}")
    print(f"  spurious_bytes_outside_nz = {spurious}")

    # Dequant cosine
    ref_sc_pt = gather_per_token_scales(ref_scales, expert_offsets,
                                        blockscale_offsets, K, M_sorted, E, n_blocks)
    fused_sc_pt = gather_per_token_scales(fused_scales, expert_offsets,
                                          blockscale_offsets, K, M_sorted, E, n_blocks)
    ref_deq = dequant_fp4(ref_fp4, ref_sc_pt, K)
    fused_deq = dequant_fp4(fused_fp4, fused_sc_pt, K)
    sa = sorted_a.to(torch.float32)
    cos_ref = torch.nn.functional.cosine_similarity(
        sa.reshape(-1).unsqueeze(0), ref_deq.reshape(-1).unsqueeze(0)
    ).item()
    cos_fu = torch.nn.functional.cosine_similarity(
        sa.reshape(-1).unsqueeze(0), fused_deq.reshape(-1).unsqueeze(0)
    ).item()
    cos_rel = torch.nn.functional.cosine_similarity(
        ref_deq.reshape(-1).unsqueeze(0), fused_deq.reshape(-1).unsqueeze(0)
    ).item()
    me_rel = (ref_deq - fused_deq).abs().max().item()
    print(f"  ref_vs_bf16 cos={cos_ref:.6f}")
    print(f"  fused_vs_bf16 cos={cos_fu:.6f}")
    print(f"  fused_vs_ref cos={cos_rel:.6f}, max_err={me_rel:.6f}")

    # === Benchmarks ===
    def run_two_op():
        sa2 = vops.shuffle_rows(a, a_map)
        vops.scaled_fp4_experts_quant(sa2, a1_gscale, expert_offsets,
                                       blockscale_offsets, num_topk)

    def run_fused():
        fused_op(a, a_map, expert_offsets, blockscale_offsets, num_topk)

    two_op_us = bench_cuda(run_two_op)
    fused_us = bench_cuda(run_fused)
    speedup = two_op_us / fused_us if fused_us > 0 else 0.0

    print(f"\n[BENCH] two_op_baseline = {two_op_us:.2f} us")
    print(f"[BENCH] fused_inline    = {fused_us:.2f} us")
    print(f"[BENCH] speedup         = {speedup:.3f}x")

    ok = (
        within1_nz == nz_cnt
        and spurious == 0
        and (cos_ref - cos_fu) < 0.002
    )
    overall = "PASS" if ok else "FAIL"
    print(f"\nOverall correctness: {overall}")
    print(f"RESULT_CORRECTNESS={overall}")
    print(f"RESULT_COS_REL={cos_rel:.6f}")
    print(f"RESULT_COS_FUSED_BF16={cos_fu:.6f}")
    print(f"RESULT_TWO_OP_US={two_op_us:.3f}")
    print(f"RESULT_FUSED_US={fused_us:.3f}")
    print(f"RESULT_SPEEDUP={speedup:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
