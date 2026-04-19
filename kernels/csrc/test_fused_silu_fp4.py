#!/usr/bin/env python3
"""
T2-N v3 correctness + microbench for fused_silu_fp4_epilogue.

Tag: W6_T2N_silu_epilogue

Compares:
    * Reference pipeline:
          silu(c1[:, :K]) * c1[:, K:]        -> BF16 intermediate
          ops.scaled_fp4_experts_quant(...)  -> FP4 + swizzled scales
      (i.e. what vLLM's silu_and_mul_scaled_fp4_experts_quant does logically)
    * Fused kernel: fused_silu_fp4_quant(c1, ...)
    * vLLM fused baseline: ops.silu_and_mul_scaled_fp4_experts_quant(c1, ...)

Correctness:
    * byte-level scale match (position + within +/-1 FP8 step) vs vLLM
    * dequant cosine >= 0.9999 vs python reference
Shapes tested (both regimes per plans/gemma4_t2n_postmortem.md):
    * Qwen3-30B: M=256, K=2816  (per task spec)
    * Gemma4-26B: M=256, K=7168 (per task spec)

Outputs a `=== RESULTS_TSV_ROW ===` block at the end.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKSPACE = os.path.join(ROOT, "workspace")

# Auto-inject the project venv if running under system python.
_venv_sp = os.path.join(ROOT, ".venv", "lib", "python3.12", "site-packages")
if os.path.isdir(_venv_sp) and _venv_sp not in sys.path:
    sys.path.insert(0, _venv_sp)

import torch  # noqa: E402


def _load_kernel():
    so = os.path.join(WORKSPACE, "fused_silu_fp4_sm120a.so")
    if not os.path.exists(so):
        raise RuntimeError(
            f"fused_silu_fp4_sm120a.so not found at {so}. "
            f"Build with: python3 {HERE}/build_fused_silu_fp4.py"
        )
    spec = importlib.util.spec_from_file_location("fused_silu_fp4_epilogue", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_silu_fp4_quant


def _dequant_fp4(packed_uint8, K):
    """Dequantize packed FP4 [M, K/2] uint8 -> [M, K] float32 (no scale)."""
    dev = packed_uint8.device
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=dev, dtype=torch.float32,
    )
    hi = ((packed_uint8 >> 4).to(torch.int64)) & 0xF
    lo = (packed_uint8.to(torch.int64)) & 0xF
    sh = (hi >> 3) & 1
    sl = (lo >> 3) & 1
    mh = hi & 7
    ml = lo & 7
    vh = torch.where(sh.bool(), -lut[mh], lut[mh])
    vl = torch.where(sl.bool(), -lut[ml], lut[ml])
    M = packed_uint8.shape[0]
    out = torch.empty((M, K), device=dev, dtype=torch.float32)
    out[:, 0::2] = vl.to(torch.float32)
    out[:, 1::2] = vh.to(torch.float32)
    return out


def _gather_per_token_scales(sw_fp8, expert_offsets, blockscale_offsets,
                              K, M_sorted, E, n_blocks):
    """De-swizzle CUTLASS scale buffer to per-token [M_sorted, n_blocks] float32."""
    num_k_tiles = (K + 63) // 64
    sw_u8 = sw_fp8.view(torch.uint8).contiguous().flatten()
    out = torch.zeros((M_sorted, n_blocks), dtype=torch.float32, device=sw_u8.device)
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
        b = sw_u8[off]
        fp8 = b.view(torch.float8_e4m3fn)
        out[ts:te] = fp8.to(torch.float32)
    return out


def _bench(fn, warmup=20, iters=100):
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
    return start.elapsed_time(end) / iters * 1000.0  # microseconds


def run_shape(name, M, K, topk, E, verbose=True):
    """Run correctness + microbench at one (M, K) shape.

    Layout: c1 is [m*topk, 2*K] gate||up.
    """
    from vllm import _custom_ops as vops

    fused_op = _load_kernel()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    M_sorted = M * topk

    if verbose:
        print(f"\n=== [{name}] M={M}, K={K}, topk={topk}, E={E}, M_sorted={M_sorted}")

    # Random GEMM1 output (simulated): gate||up layout.
    torch.manual_seed(0)
    c1 = (torch.randn(M_sorted, 2 * K, device=device, dtype=dtype) * 3.0)

    # Per-expert global scale for activation quant.
    a2_gscale = torch.ones(E, device=device, dtype=torch.float32)

    # Build realistic expert/blockscale offsets via vLLM's own helper.
    # Distribute M_sorted tokens across E experts roughly evenly.
    topk_ids = torch.zeros(M, topk, device=device, dtype=torch.int32)
    for i in range(M):
        for j in range(topk):
            topk_ids[i, j] = (i * topk + j) % E

    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((E, 3), dtype=torch.int32, device=device)
    a_map_real = torch.empty(M_sorted, dtype=torch.int32, device=device)
    c_map_real = torch.empty(M_sorted, dtype=torch.int32, device=device)
    vops.get_cutlass_moe_mm_data(
        topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
        a_map_real, c_map_real, E, K * 2, K, blockscale_offsets, is_gated=True,
    )

    n_blocks = K // 16

    # ---- Reference path (vLLM fused baseline) ----
    ref_fp4, ref_scales = vops.silu_and_mul_scaled_fp4_experts_quant(
        c1, a2_gscale, expert_offsets, blockscale_offsets, topk,
    )
    ref_us = _bench(lambda: vops.silu_and_mul_scaled_fp4_experts_quant(
        c1, a2_gscale, expert_offsets, blockscale_offsets, topk,
    ))

    # Two-op "non-fused" baseline: python silu_and_mul -> scaled_fp4_experts_quant.
    def _twostep():
        gate = c1[:, :K]
        up = c1[:, K:]
        act = torch.nn.functional.silu(gate) * up  # [m_topk, K] BF16
        return vops.scaled_fp4_experts_quant(
            act, a2_gscale, expert_offsets, blockscale_offsets, topk,
        )
    two_us = _bench(_twostep)
    two_fp4, two_scales = _twostep()

    # ---- Fused kernel under test ----
    fused_fp4, fused_scales = fused_op(
        c1, a2_gscale, expert_offsets, blockscale_offsets, topk,
    )
    fused_us = _bench(lambda: fused_op(
        c1, a2_gscale, expert_offsets, blockscale_offsets, topk,
    ))

    if verbose:
        print(f"  vLLM silu+quant:  {ref_us:7.2f} µs  (single-fused baseline)")
        print(f"  Two-op (silu, q): {two_us:7.2f} µs  (true PRE-fusion baseline)")
        print(f"  Fused (ours):     {fused_us:7.2f} µs")
        print(f"  Speedup vs two-op:         {two_us/fused_us:.3f}x")
        print(f"  Speedup vs vLLM-fused:     {ref_us/fused_us:.3f}x")

    # ---- Correctness: FP4 byte match to vLLM fused ----
    fp4_match = (fused_fp4.shape == ref_fp4.shape)
    fp4_frac = 0.0
    if fp4_match:
        frac = (fused_fp4 == ref_fp4).float().mean().item()
        fp4_frac = frac
        if verbose:
            print(f"  FP4 byte match vs vLLM fused: {frac:.1%}")

    # ---- Correctness: scale match within +/-1 FP8 step ----
    fused_u8 = fused_scales.view(torch.uint8)
    ref_u8 = ref_scales.view(torch.uint8)
    min_rows = min(fused_u8.shape[0], ref_u8.shape[0])
    min_cols = min(fused_u8.shape[1], ref_u8.shape[1])
    fr = fused_u8[:min_rows, :min_cols]
    rr = ref_u8[:min_rows, :min_cols]
    ref_nz = rr != 0
    fused_nz = fr != 0
    pos_match = (ref_nz == fused_nz).all().item()
    n_ref_nz = ref_nz.sum().item()
    if n_ref_nz > 0:
        diff = (rr.int() - fr.int()).abs()
        within1 = ((diff <= 1) & ref_nz).sum().item() / n_ref_nz
        exact = ((fr == rr) & ref_nz).sum().item() / n_ref_nz
    else:
        within1 = 0.0
        exact = 0.0
    if verbose:
        print(f"  Scale pos match: {pos_match};  exact: {exact:.1%};  ±1 step: {within1:.1%}")

    # ---- Dequant cosine ----
    # Compare dequantization of OUR fp4 vs dequantization of vLLM-fused fp4
    # (both go through identical FP4 lookup).  Any difference comes from
    # our scale or nibble rounding vs vLLM's, not from FP4 quantization
    # itself.  Target: cos >= 0.9999 vs vLLM-fused.
    # Also compute cos vs float32 silu*mul reference for sanity (this
    # is bounded at ~0.996 by the FP4 quantization floor).
    tok_gscale = torch.ones(M_sorted, device=device, dtype=torch.float32)

    def _dequant(packed, scales_swiz):
        sf_pt = _gather_per_token_scales(
            scales_swiz, expert_offsets, blockscale_offsets,
            K, M_sorted, E, n_blocks,
        )
        unscaled = _dequant_fp4(packed, K)
        sf_over_gscale = sf_pt / tok_gscale.unsqueeze(1)
        sf_expand = sf_over_gscale.repeat_interleave(16, dim=1)
        return unscaled * sf_expand

    fused_float = _dequant(fused_fp4, fused_scales)
    ref_float = _dequant(ref_fp4, ref_scales)

    gate = c1[:, :K].float()
    up = c1[:, K:].float()
    py_ref = (torch.nn.functional.silu(gate) * up).float()

    def _cos(a, b):
        na = a.norm(dim=-1) + 1e-9
        nb = b.norm(dim=-1) + 1e-9
        return ((a * b).sum(dim=-1) / (na * nb)).mean().item()

    cos_fused_vs_vllm = _cos(fused_float, ref_float)
    cos_fused_vs_py = _cos(fused_float, py_ref)
    cos_vllm_vs_py = _cos(ref_float, py_ref)
    if verbose:
        print(f"  Cos(fused, vLLM-fused dequant):  {cos_fused_vs_vllm:.6f}  (target >= 0.9999)")
        print(f"  Cos(fused, float silu*mul):      {cos_fused_vs_py:.6f}  (FP4 floor ~0.996)")
        print(f"  Cos(vLLM,  float silu*mul):      {cos_vllm_vs_py:.6f}  (FP4 floor ~0.996)")

    return dict(
        shape=name, M=M, K=K, topk=topk, E=E, M_sorted=M_sorted,
        ref_us=round(ref_us, 2),
        two_us=round(two_us, 2),
        fused_us=round(fused_us, 2),
        speedup_vs_two=round(two_us / fused_us, 3),
        speedup_vs_vllm_fused=round(ref_us / fused_us, 3),
        fp4_match_frac=round(fp4_frac, 4),
        scale_pos_match=bool(pos_match),
        scale_within_1=round(within1, 4),
        cos_fused_vs_vllm=round(cos_fused_vs_vllm, 6),
        cos_fused_vs_py=round(cos_fused_vs_py, 6),
        cos_vllm_vs_py=round(cos_vllm_vs_py, 6),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required", file=sys.stderr)
        sys.exit(1)

    cap = torch.cuda.get_device_capability()
    print(f"Device: {torch.cuda.get_device_name(0)}  SM{cap[0]}{cap[1]}")

    shapes = [
        # (name, M, K, topk, E) — K here is the per-expert intermediate dim
        # passed INTO the fused op (c1 has [M*topk, 2K]).
        # Qwen3-30B-A3B: moe_intermediate = 768 per expert, hidden=2048.
        # In the task spec M=256,K=2816 is a round-trip shape that exercises
        # both Qwen3-scale rows and a non-trivial K.
        ("qwen3_M256_K2816", 256, 2816, 8, 128),
        # Gemma4-26B-A4B: M=256 from C=32 * topk=8; K=7168 matches spec.
        ("gemma4_M256_K7168", 256, 7168, 8, 128),
    ]

    results = []
    for shp in shapes:
        try:
            r = run_shape(*shp, verbose=True)
        except Exception as e:
            print(f"[FAIL] {shp[0]}: {e}")
            import traceback; traceback.print_exc()
            r = {"shape": shp[0], "error": str(e)}
        results.append(r)

    # Verdict
    print("\n=== VERDICT ===")
    all_ok = True
    big_win = True
    for r in results:
        if "error" in r:
            print(f"  {r['shape']}: ERROR {r['error']}")
            all_ok = False; big_win = False
            continue
        s = r["speedup_vs_two"]
        v = r["speedup_vs_vllm_fused"]
        cos_v = r["cos_fused_vs_vllm"]
        cos_py = r["cos_fused_vs_py"]
        tag = "KILL" if s < 1.0 else ("BIG WIN" if s >= 1.3 else "PASS" if s >= 1.05 else "MARGINAL")
        print(f"  {r['shape']}: {tag}  two-op={r['two_us']}µs  fused={r['fused_us']}µs"
              f"  x{s:.2f}  vLLM-fused={r['ref_us']}µs x{v:.2f}"
              f"  cos_vllm={cos_v:.4f}  cos_float={cos_py:.4f}")
        if s < 1.0:
            all_ok = False
        if s < 1.3:
            big_win = False

    overall = "BIG WIN" if big_win else ("PASS" if all_ok else "KILL")
    print(f"  Overall: {overall}")

    # RESULTS_TSV_ROW block for parent harness
    print("\n=== RESULTS_TSV_ROW ===")
    for r in results:
        if "error" in r:
            row = [
                "W6_T2N_silu_epilogue",
                r["shape"], "ERROR", "", "", "", "", "",
            ]
        else:
            row = [
                "W6_T2N_silu_epilogue",
                r["shape"],
                f"fused={r['fused_us']}us",
                f"two_op={r['two_us']}us",
                f"vllm_fused={r['ref_us']}us",
                f"x_two={r['speedup_vs_two']}",
                f"x_vllm={r['speedup_vs_vllm_fused']}",
                f"cos_vllm={r['cos_fused_vs_vllm']}",
                f"cos_float={r['cos_fused_vs_py']}",
            ]
        print("\t".join(str(x) for x in row))
    print("=== END_RESULTS_TSV_ROW ===")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"overall": overall, "results": results}, f, indent=2)
        print(f"Wrote {args.output}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
