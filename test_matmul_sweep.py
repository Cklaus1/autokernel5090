"""Wide config sweep for FP16-dot matmul."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

M, K, N_dim = 2048, 5120, 5120
flops = 2 * M * K * N_dim

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N_dim, device='cuda', dtype=torch.float16)


@triton.autotune(
    configs=[
        # BK=32
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
        # BK=64
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=2, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=4, num_warps=8),
        # BK=128
        triton.Config({'BM': 256, 'BN': 128, 'BK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 128}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 128}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 128}, num_stages=3, num_warps=8),
        # Larger tiles
        triton.Config({'BM': 256, 'BN': 256, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 64}, num_stages=2, num_warps=8),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        # Smaller with more warps
        triton.Config({'BM': 64, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64}, num_stages=4, num_warps=4),
        # Different warp counts
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=4),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=16),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=4),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=16),
        # Higher stage counts
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=5, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=5, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=5, num_warps=8),
    ],
    key=['M_key', 'N_key', 'K_key'],
)
@triton.jit
def matmul_sweep(
    A_ptr, W_ptr, O_ptr,
    M_key, N_key, K_key,
    stride_am, stride_ak, stride_wk, stride_wn, stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M_key, BM)
    num_n = tl.cdiv(N_key, BN)
    GRP: tl.constexpr = 8
    group_id = pid // (num_m * GRP)
    first_n = group_id * GRP
    gsn = min(num_n - first_n, GRP)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)

    a_ptr = tl.make_block_ptr(A_ptr, (M_key, K_key), (stride_am, stride_ak),
                               (pid_m * BM, 0), (BM, BK), (1, 0))
    w_ptr = tl.make_block_ptr(W_ptr, (K_key, N_key), (stride_wk, stride_wn),
                               (0, pid_n * BN), (BK, BN), (1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K_key, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        ww = tl.load(w_ptr, boundary_check=(0, 1))
        partial = tl.dot(aa, ww, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        w_ptr = tl.advance(w_ptr, (BK, 0))

    o_ptr = tl.make_block_ptr(O_ptr, (M_key, N_key), (stride_om, stride_on),
                               (pid_m * BM, pid_n * BN), (BM, BN), (1, 0))
    tl.store(o_ptr, acc.to(tl.float16), boundary_check=(0, 1))


o = torch.empty(M, N_dim, device='cuda', dtype=torch.float16)
def grid(META):
    return (triton.cdiv(M, META['BM']) * triton.cdiv(N_dim, META['BN']),)
def fn():
    matmul_sweep[grid](
        a, w, o, M, N_dim, K,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1), o.stride(0), o.stride(1),
    )

fn(); torch.cuda.synchronize()
ms = do_bench(fn) * 1e-3
tflops = flops / ms / 1e12
print(f"Best config: {ms*1e6:.0f} us = {tflops:.1f} TFLOPS")

# Show winning config
best = matmul_sweep.best_config
print(f"Winner: BM={best.kwargs['BM']}, BN={best.kwargs['BN']}, BK={best.kwargs['BK']}, "
      f"stages={best.num_stages}, warps={best.num_warps}")
