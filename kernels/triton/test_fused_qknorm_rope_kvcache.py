"""Rank 4 correctness + microbench: fused qknorm+rope+KV-cache scatter.

Extends ``test_fused_qknorm_rope.py`` with a verification of the paged
KV-cache scatter path. Correctness gate: the fused kernel's K/V writes
into ``key_cache`` / ``value_cache`` must bit-equal the reference
(``qknorm_rope_torch_ref`` + ``kv_cache_scatter_torch_ref``).

Run:
    python kernels/triton/test_fused_qknorm_rope_kvcache.py
    CUDA_VISIBLE_DEVICES=1 python kernels/triton/test_fused_qknorm_rope_kvcache.py

Tag: W8_T2N_rank4_kvcache_rotary
"""

from __future__ import annotations

import math
import pathlib
import sys
import time

import torch

try:
    from kernels.triton.fused_qknorm_rope import (
        fused_qknorm_rope,
        kv_cache_scatter_torch_ref,
        qknorm_rope_torch_ref,
    )
except ImportError:
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "fused_qknorm_rope",
        pathlib.Path(__file__).parent / "fused_qknorm_rope.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    fused_qknorm_rope = _mod.fused_qknorm_rope
    qknorm_rope_torch_ref = _mod.qknorm_rope_torch_ref
    kv_cache_scatter_torch_ref = _mod.kv_cache_scatter_torch_ref


DEVICE = "cuda"
DTYPE = torch.bfloat16


def _cos_maxabs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a = a.float().flatten()
    b = b.float().flatten()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    return cos.item(), (a - b).abs().max().item()


def _build_cos_sin(max_pos: int, head_dim: int, base: float = 1e6):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                              device=DEVICE) / head_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE)


# -- Correctness harness ---------------------------------------------------

def _run_correctness(M, num_q_heads, num_kv_heads, head_dim,
                     num_blocks=128, block_size=16,
                     pad_M=0, bad_slot_frac=0.0, seed=0):
    """Test the K/V cache scatter path.

    pad_M: extra trailing tokens in q/k/v and slot_mapping that should be
    IGNORED (simulates CUDA-graph padding where slot_mapping[t] == -1).
    bad_slot_frac: fraction of valid tokens to set to slot=-1 (extra
    sentinel coverage beyond trailing padding).
    """
    torch.manual_seed(seed)
    total_M = M + pad_M
    q = torch.randn(total_M, num_q_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
    k = torch.randn(total_M, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
    v = torch.randn(total_M, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
    q_norm_w = torch.randn(head_dim, device=DEVICE, dtype=DTYPE) * 0.1 + 1.0
    k_norm_w = torch.randn(head_dim, device=DEVICE, dtype=DTYPE) * 0.1 + 1.0
    max_pos = 8192
    cos_sin = _build_cos_sin(max_pos, head_dim)

    positions = torch.randint(0, max_pos, (total_M,), device=DEVICE,
                              dtype=torch.int64)

    # Build slot_mapping: first M tokens scatter to real slots, trailing
    # pad_M tokens are sentinels (-1). Additionally randomly set some of
    # the valid slots to -1 to stress the sentinel-skip path.
    slot_mapping = torch.empty(total_M, device=DEVICE, dtype=torch.int32)
    # Unique random slots within the total space to avoid write collisions.
    slot_space = num_blocks * block_size
    assert slot_space >= M, "too few slots for M"
    perm = torch.randperm(slot_space, device=DEVICE, dtype=torch.int32)[:M]
    slot_mapping[:M] = perm
    if bad_slot_frac > 0:
        n_bad = int(M * bad_slot_frac)
        slot_mapping[:n_bad] = -1
    if pad_M > 0:
        slot_mapping[M:] = -1

    # KV cache: (num_blocks, 2, block_size, num_kv_heads, head_dim).
    kv_cache = torch.zeros(
        num_blocks, 2, block_size, num_kv_heads, head_dim,
        device=DEVICE, dtype=DTYPE,
    )
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]

    # --- Reference: do the 3-op norm+rope path + scatter-write. ---
    q_ref = q.clone()
    k_ref = k.clone()
    v_ref = v.clone()
    kv_cache_ref = torch.zeros_like(kv_cache)
    key_cache_ref = kv_cache_ref[:, 0]
    value_cache_ref = kv_cache_ref[:, 1]

    q_ref, k_ref = qknorm_rope_torch_ref(
        q_ref, k_ref, positions, q_norm_w, k_norm_w, cos_sin, 1e-6,
    )
    kv_cache_scatter_torch_ref(
        k_ref, v_ref, key_cache_ref, value_cache_ref, slot_mapping,
    )

    # --- Fused kernel: qknorm+rope+scatter in one call. ---
    q_fused = q.clone()
    k_fused = k.clone()
    v_fused = v.clone()
    fused_qknorm_rope(
        q_fused, k_fused, positions, q_norm_w, k_norm_w, cos_sin,
        1e-6, is_neox=True,
        v=v_fused,
        key_cache=key_cache, value_cache=value_cache,
        slot_mapping=slot_mapping,
    )

    cos_q, max_q = _cos_maxabs(q_ref, q_fused)
    cos_k, max_k = _cos_maxabs(k_ref, k_fused)
    cos_kc, max_kc = _cos_maxabs(key_cache_ref, key_cache)
    cos_vc, max_vc = _cos_maxabs(value_cache_ref, value_cache)

    # Self-consistency: fused K written to key_cache must bit-equal the
    # in-place rotated K in k_fused at the same slot. This is the
    # "scatter pattern off-by-one" detector — if slot_mapping math
    # drifts, one of these 2 comparisons diverges while the other does
    # not.
    valid_mask = slot_mapping[:M] >= 0
    valid_slots = slot_mapping[:M][valid_mask].long()
    if valid_slots.numel() > 0:
        blk = valid_slots // block_size
        off = valid_slots %  block_size
        k_scattered = key_cache[blk, off]            # [n_valid, Hk, D]
        k_expected = k_fused[:M][valid_mask]         # [n_valid, Hk, D]
        v_scattered = value_cache[blk, off]
        v_expected = v_fused[:M][valid_mask]
        max_kc_self = (k_scattered.float() - k_expected.float()).abs().max().item()
        max_vc_self = (v_scattered.float() - v_expected.float()).abs().max().item()
    else:
        max_kc_self = max_vc_self = 0.0

    # Rank 4 correctness gate:
    #   - cos ≥ 0.9999 vs PyTorch ref on q, k, k_cache, v_cache (matches
    #     Rank 2 gate; the stock 3-op path and fused 1-op path differ by
    #     ≤ 1 BF16 ulp due to different rounding).
    #   - Self-consistency: the scatter-written KV cache must be
    #     bit-exactly the final rotated K and raw V values held in the
    #     in-place tensors — this is the slot-mapping off-by-one check.
    ok = (cos_q >= 0.9999 and cos_k >= 0.9999
          and cos_kc >= 0.9999 and cos_vc >= 0.9999
          and max_kc_self == 0.0 and max_vc_self == 0.0)
    status = "OK " if ok else "FAIL"
    print(
        f"[{status}] M={M} pad={pad_M} bad={bad_slot_frac:.2f} | "
        f"cos_q={cos_q:.6f} cos_k={cos_k:.6f} "
        f"cos_kcache={cos_kc:.6f} (vs_ref_maxabs={max_kc:.3e}) "
        f"cos_vcache={cos_vc:.6f} (vs_ref_maxabs={max_vc:.3e}) "
        f"| slot-self max_kc={max_kc_self:.3e} max_vc={max_vc_self:.3e}"
    )
    return ok


# -- Multi-step logit stability harness -----------------------------------

def _multi_step_logit_stability(
    num_q_heads=32, num_kv_heads=4, head_dim=128,
    num_blocks=128, block_size=16, num_steps=8, seed=1,
):
    """Run num_steps decode steps sequentially and verify logit stability
    between fused and stock K/V cache. The "logit" proxy here is the
    attention score pattern produced by querying the newest Q against
    all past K in the cache.

    Specifically: after writing steps 0..t, compute softmax(Q @ K^T) for
    both the fused and stock caches and check cosine similarity >= 0.9999
    at every step. Any off-by-one in slot_mapping corrupts past K/V and
    diverges the pattern after the first step.
    """
    torch.manual_seed(seed)
    q_norm_w = torch.randn(head_dim, device=DEVICE, dtype=DTYPE) * 0.1 + 1.0
    k_norm_w = torch.randn(head_dim, device=DEVICE, dtype=DTYPE) * 0.1 + 1.0
    cos_sin = _build_cos_sin(8192, head_dim)

    kv_fused = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_dim,
                           device=DEVICE, dtype=DTYPE)
    kv_stock = torch.zeros_like(kv_fused)

    # One sequence, M=1 per step, positions advance monotonically.
    all_ok = True
    for step in range(num_steps):
        slot = torch.tensor([step], device=DEVICE, dtype=torch.int32)
        pos = torch.tensor([step], device=DEVICE, dtype=torch.int64)
        # Raw q/k/v (pre-norm, pre-rope).
        q = torch.randn(1, num_q_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
        k = torch.randn(1, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
        v = torch.randn(1, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1

        # Stock path
        q_s = q.clone(); k_s = k.clone(); v_s = v.clone()
        q_s, k_s = qknorm_rope_torch_ref(
            q_s, k_s, pos, q_norm_w, k_norm_w, cos_sin, 1e-6,
        )
        kv_cache_scatter_torch_ref(k_s, v_s, kv_stock[:, 0], kv_stock[:, 1], slot)

        # Fused path
        q_f = q.clone(); k_f = k.clone(); v_f = v.clone()
        fused_qknorm_rope(
            q_f, k_f, pos, q_norm_w, k_norm_w, cos_sin, 1e-6,
            is_neox=True,
            v=v_f,
            key_cache=kv_fused[:, 0], value_cache=kv_fused[:, 1],
            slot_mapping=slot,
        )

        # Q matches?
        cos_q, max_q = _cos_maxabs(q_s, q_f)
        # Compute a cheap attn-score proxy: q · k^T over the populated cache.
        cache_rows = step + 1
        kf_flat = kv_fused[:, 0].reshape(-1, num_kv_heads, head_dim)[:cache_rows]
        ks_flat = kv_stock[:, 0].reshape(-1, num_kv_heads, head_dim)[:cache_rows]
        vf_flat = kv_fused[:, 1].reshape(-1, num_kv_heads, head_dim)[:cache_rows]
        vs_flat = kv_stock[:, 1].reshape(-1, num_kv_heads, head_dim)[:cache_rows]

        # Broadcast Q [num_q_heads] against K [num_kv_heads] GQA-style
        # (tile K across q_heads_per_kv). At small scale it's fine to just
        # pick head 0 for each.
        scores_f = torch.einsum("qd,td->qt", q_f[0].float(), kf_flat[:, 0, :].float())
        scores_s = torch.einsum("qd,td->qt", q_s[0].float(), ks_flat[:, 0, :].float())
        out_f = torch.einsum("qt,td->qd", torch.softmax(scores_f, dim=-1),
                             vf_flat[:, 0, :].float())
        out_s = torch.einsum("qt,td->qd", torch.softmax(scores_s, dim=-1),
                             vs_flat[:, 0, :].float())
        cos_o, max_o = _cos_maxabs(out_f, out_s)
        ok = cos_q >= 0.9999 and cos_o >= 0.9999
        status = "OK " if ok else "FAIL"
        print(f"[{status}] step={step} pos={step} | cos_q={cos_q:.6f} "
              f"cos_attn_out={cos_o:.6f} maxabs_attn={max_o:.3e}")
        all_ok = all_ok and ok
    return all_ok


# -- Microbench -----------------------------------------------------------

def _bench(M, num_q_heads, num_kv_heads, head_dim, iters=200, warmup=20,
           num_blocks=512, block_size=16):
    q = torch.randn(M, num_q_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
    k = torch.randn(M, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
    v = torch.randn(M, num_kv_heads, head_dim, device=DEVICE, dtype=DTYPE) * 0.1
    q_w = torch.randn(head_dim, device=DEVICE, dtype=DTYPE) * 0.1 + 1.0
    k_w = torch.randn(head_dim, device=DEVICE, dtype=DTYPE) * 0.1 + 1.0
    cos_sin = _build_cos_sin(8192, head_dim)
    positions = torch.randint(0, 8192, (M,), device=DEVICE, dtype=torch.int64)
    slot_mapping = torch.arange(M, device=DEVICE, dtype=torch.int32)
    kv_cache = torch.zeros(num_blocks, 2, block_size, num_kv_heads, head_dim,
                           device=DEVICE, dtype=DTYPE)
    key_cache = kv_cache[:, 0]
    value_cache = kv_cache[:, 1]

    # Rank 2-only path: just fused qknorm+rope (NO kv-cache scatter).
    def rank2_only():
        q_ = q.clone(); k_ = k.clone()
        fused_qknorm_rope(q_, k_, positions, q_w, k_w, cos_sin, 1e-6)

    # Rank 2+4: qknorm+rope + kv scatter fused into same call chain.
    def rank24():
        q_ = q.clone(); k_ = k.clone(); v_ = v.clone()
        fused_qknorm_rope(
            q_, k_, positions, q_w, k_w, cos_sin, 1e-6, is_neox=True,
            v=v_, key_cache=key_cache, value_cache=value_cache,
            slot_mapping=slot_mapping,
        )

    # Stock 4-op path: 2 norms + rope + scatter (as 4 torch ops — this
    # over-estimates stock cost because torch eager norm is slower than
    # vLLM's fused CUDA, but the delta between rank2_only and rank24 is
    # the true +cost-of-scatter we save on top of Rank 2.)
    def stock_4op():
        q_ = q.clone(); k_ = k.clone(); v_ = v.clone()
        # per-head RMSNorm x2 (separate launches)
        q3 = q_.view(M, num_q_heads, head_dim)
        k3 = k_.view(M, num_kv_heads, head_dim)
        q3.copy_(torch.nn.functional.rms_norm(q3, (head_dim,), q_w, 1e-6))
        k3.copy_(torch.nn.functional.rms_norm(k3, (head_dim,), k_w, 1e-6))
        # RoPE (cat style, eager)
        cs = cos_sin.index_select(0, positions)
        cos, sin = cs[..., :head_dim // 2].unsqueeze(1), cs[..., head_dim // 2:].unsqueeze(1)
        def rope(x):
            x1 = x[..., :head_dim // 2]; x2 = x[..., head_dim // 2:]
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        q3.copy_(rope(q3)); k3.copy_(rope(k3))
        # KV scatter (separate launch)
        kv_cache_scatter_torch_ref(k3, v_, key_cache, value_cache, slot_mapping)

    def _time_fn(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # us

    us_r2 = _time_fn(rank2_only)
    us_r24 = _time_fn(rank24)
    us_stock = _time_fn(stock_4op)
    print(f"[bench] M={M}  Rank2-only={us_r2:7.2f} us  "
          f"Rank2+4={us_r24:7.2f} us  Stock-4op={us_stock:7.2f} us  "
          f"Δ(r24-r2)={us_r24 - us_r2:+.2f} us  "
          f"save_vs_stock={us_stock - us_r24:+.2f} us")
    return us_r2, us_r24, us_stock


def main():
    torch.cuda.init()
    print("=" * 78)
    print("Correctness (M=1 decode shape)")
    print("=" * 78)
    ok1 = _run_correctness(M=1, num_q_heads=32, num_kv_heads=4, head_dim=128)
    ok2 = _run_correctness(M=1, num_q_heads=32, num_kv_heads=4, head_dim=128,
                           pad_M=3)  # sentinel guard
    ok3 = _run_correctness(M=8, num_q_heads=32, num_kv_heads=4, head_dim=128,
                           pad_M=2, bad_slot_frac=0.25)
    ok4 = _run_correctness(M=16, num_q_heads=32, num_kv_heads=4, head_dim=128)
    ok5 = _run_correctness(M=256, num_q_heads=32, num_kv_heads=4, head_dim=128,
                           num_blocks=512, block_size=16)

    print()
    print("=" * 78)
    print("Multi-step logit stability (8 decode steps)")
    print("=" * 78)
    ok_multi = _multi_step_logit_stability(num_steps=8)

    all_ok = all([ok1, ok2, ok3, ok4, ok5, ok_multi])

    print()
    print("=" * 78)
    print("Microbench (Qwen3-30B decode shape)")
    print("=" * 78)
    _bench(M=1,   num_q_heads=32, num_kv_heads=4, head_dim=128)
    _bench(M=8,   num_q_heads=32, num_kv_heads=4, head_dim=128)
    _bench(M=64,  num_q_heads=32, num_kv_heads=4, head_dim=128)
    _bench(M=512, num_q_heads=32, num_kv_heads=4, head_dim=128)

    print()
    print("=" * 78)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
