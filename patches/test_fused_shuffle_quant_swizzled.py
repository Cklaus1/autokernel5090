#!/usr/bin/env python3
"""
T2-N correctness test for fused_shuffle_quant with CUTLASS swizzled scales.

Tests that the fused-kernel-plus-reswizzle pipeline produces a scale-factor
buffer bit-compatible with vLLM\'s scaled_fp4_experts_quant output, within one
FP8-E4M3 step of every non-zero scale position (the inherent difference between
software and hardware FP32->FP8 rounding).

For FP4 (E2M1) activations, the theoretical precision floor is ~cos=0.995 and
max_abs_err=0.25 (one FP4 step) against BF16 input; hence we check that the
fused path\'s cosine similarity vs the canonical reference is within 0.002 of
the reference\'s own roundtrip cosine.

Run on a GPU with >8 GB free:
    CUDA_VISIBLE_DEVICES=<idx> python3 patches/test_fused_shuffle_quant_swizzled.py
"""
import sys, os, importlib.util
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
    """De-swizzle CUTLASS [target_rows, padded_k//4] fp8 scales back to per-token [M_sorted, n_blocks]."""
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


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.manual_seed(0)

    from vllm import _custom_ops as vops
    from vllm import envs
    from patches.fused_shuffle_quant_swizzle import rowmajor_kernel_scales_to_cutlass_swizzled

    M_tokens = 64
    K = 2816
    num_topk = 8
    E = 128
    M_sorted = M_tokens * num_topk
    n_blocks = K // 16
    MAX_TPE = envs.VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE

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

    sorted_a = vops.shuffle_rows(a, a_map)
    ref_fp4, ref_scales = vops.scaled_fp4_experts_quant(
        sorted_a, a1_gscale, expert_offsets, blockscale_offsets, num_topk
    )

    fused_op = _load_kernel()
    padded_M = ((M_sorted + 127) // 128) * 128
    padded_nb = ((n_blocks + 3) // 4) * 4
    fused_fp4, fused_sc_flat = fused_op(a, a_map, padded_M, padded_nb)

    target_rows = MAX_TPE * num_topk
    fused_scales = rowmajor_kernel_scales_to_cutlass_swizzled(
        fused_sc_flat, padded_M, padded_nb, M_sorted, K,
        expert_offsets, blockscale_offsets, target_rows,
    ).view(torch.float8_e4m3fn)
    torch.cuda.synchronize()

    ref_u8 = ref_scales.view(torch.uint8)
    fused_u8 = fused_scales.view(torch.uint8)
    nz = ref_u8 != 0
    nz_cnt = nz.sum().item()
    exact_nz = ((fused_u8 == ref_u8) & nz).sum().item()
    within1_nz = (((fused_u8.int() - ref_u8.int()).abs() <= 1) & nz).sum().item()
    # No spurious writes outside non-zero positions
    spurious = (fused_u8[~nz] != 0).sum().item()

    print(f"M_tokens={M_tokens} K={K} topk={num_topk} E={E}")
    print(f"scale_exact_on_nonzero = {exact_nz/nz_cnt:.3%} ({exact_nz}/{nz_cnt})")
    print(f"scale_within_pm1_on_nonzero = {within1_nz/nz_cnt:.3%}")
    print(f"scale_spurious_bytes_outside_nonzero = {spurious}")

    ref_sc_pt = gather_per_token_scales(ref_scales, expert_offsets,
                                        blockscale_offsets, K, M_sorted, E, n_blocks)
    fused_sc_pt = gather_per_token_scales(fused_scales, expert_offsets,
                                          blockscale_offsets, K, M_sorted, E, n_blocks)

    ref_deq = dequant_fp4(ref_fp4, ref_sc_pt, K)
    fused_deq = dequant_fp4(fused_fp4, fused_sc_pt, K)

    # Floor: reference-vs-BF16 roundtrip
    sa = sorted_a.to(torch.float32)
    cos_ref = torch.nn.functional.cosine_similarity(
        sa.reshape(-1).unsqueeze(0), ref_deq.reshape(-1).unsqueeze(0)
    ).item()
    me_ref = (sa - ref_deq).abs().max().item()
    cos_fu = torch.nn.functional.cosine_similarity(
        sa.reshape(-1).unsqueeze(0), fused_deq.reshape(-1).unsqueeze(0)
    ).item()
    me_fu = (sa - fused_deq).abs().max().item()
    cos_rel = torch.nn.functional.cosine_similarity(
        ref_deq.reshape(-1).unsqueeze(0), fused_deq.reshape(-1).unsqueeze(0)
    ).item()
    me_rel = (ref_deq - fused_deq).abs().max().item()

    print(f"ref_vs_bf16: cos={cos_ref:.6f} max_err={me_ref:.6f} (FP4 precision floor)")
    print(f"fused_vs_bf16: cos={cos_fu:.6f} max_err={me_fu:.6f}")
    print(f"fused_vs_ref: cos={cos_rel:.6f} max_err={me_rel:.6f}")

    # PASS criteria:
    #  (a) all non-zero scale positions match within +-1 FP8 step
    #  (b) no spurious scale bytes at zero positions
    #  (c) fused cos vs BF16 input is within 0.002 of reference cos (same FP4 precision)
    ok = (
        within1_nz == nz_cnt
        and spurious == 0
        and (cos_ref - cos_fu) < 0.002
    )
    print("\nOverall: " + ("PASS" if ok else "FAIL"))
    print(f"COSINE_REL={cos_rel:.6f} MAXERR_REL={me_rel:.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
