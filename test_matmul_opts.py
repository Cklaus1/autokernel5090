"""Test different matmul approaches to find the fastest."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * K * N

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)


# Approach 1: Current kernel (FP16 dot + FP32 accum)
@triton.autotune(
    configs=[
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
    ],
    key=['M_key', 'N_key', 'K_key'],
)
@triton.jit
def matmul_fp16dot_fp32acc(
    A_ptr, W_ptr, O_ptr,
    M_key, N_key, K_key,
    stride_am, stride_ak, stride_wk, stride_wn, stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M_key, BM)
    num_n = tl.cdiv(N_key, BN)
    GRP = 8
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


# Approach 2: Pure FP16 accumulation (no FP32)
@triton.autotune(
    configs=[
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 32}, num_stages=4, num_warps=8),
    ],
    key=['M_key', 'N_key', 'K_key'],
)
@triton.jit
def matmul_pure_fp16(
    A_ptr, W_ptr, O_ptr,
    M_key, N_key, K_key,
    stride_am, stride_ak, stride_wk, stride_wn, stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M_key, BM)
    num_n = tl.cdiv(N_key, BN)
    GRP = 8
    group_id = pid // (num_m * GRP)
    first_n = group_id * GRP
    gsn = min(num_n - first_n, GRP)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)

    a_ptr = tl.make_block_ptr(A_ptr, (M_key, K_key), (stride_am, stride_ak),
                               (pid_m * BM, 0), (BM, BK), (1, 0))
    w_ptr = tl.make_block_ptr(W_ptr, (K_key, N_key), (stride_wk, stride_wn),
                               (0, pid_n * BN), (BK, BN), (1, 0))

    # Pure FP16 accumulation — no FP32 overhead
    acc = tl.zeros((BM, BN), dtype=tl.float16)
    for _ in range(0, tl.cdiv(K_key, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        ww = tl.load(w_ptr, boundary_check=(0, 1))
        acc = tl.dot(aa, ww, acc, out_dtype=tl.float16)  # FP16 MMA + FP16 accum
        a_ptr = tl.advance(a_ptr, (0, BK))
        w_ptr = tl.advance(w_ptr, (BK, 0))

    o_ptr = tl.make_block_ptr(O_ptr, (M_key, N_key), (stride_om, stride_on),
                               (pid_m * BM, pid_n * BN), (BM, BN), (1, 0))
    tl.store(o_ptr, acc, boundary_check=(0, 1))


# Approach 3: FP16 dot with larger BK (more compute per memory load)
@triton.autotune(
    configs=[
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=4, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 256, 'BK': 64}, num_stages=2, num_warps=8),
    ],
    key=['M_key', 'N_key', 'K_key'],
)
@triton.jit
def matmul_fp16dot_bigK(
    A_ptr, W_ptr, O_ptr,
    M_key, N_key, K_key,
    stride_am, stride_ak, stride_wk, stride_wn, stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M_key, BM)
    num_n = tl.cdiv(N_key, BN)
    GRP = 8
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


def run_kernel(kernel_fn, name):
    o = torch.empty(M, N, device='cuda', dtype=torch.float16)
    def grid(META):
        return (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)
    def fn():
        kernel_fn[grid](
            a, w, o,
            M, N, K,
            a.stride(0), a.stride(1), w.stride(0), w.stride(1), o.stride(0), o.stride(1),
        )
    # Warmup (runs autotune)
    fn()
    torch.cuda.synchronize()
    # Benchmark
    ms = do_bench(fn) * 1e-3  # seconds
    tflops = flops / ms / 1e12
    # Check correctness
    fn()
    torch.cuda.synchronize()
    ref = torch.mm(a, w)
    err = (o - ref).abs().max().item()
    print(f"{name:30s}: {ms*1e6:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}")


print(f"M={M}, K={K}, N={N}, FP16 peak=419 TFLOPS, FP16-accum peak=838 TFLOPS\n")
run_kernel(matmul_fp16dot_fp32acc, "FP16-dot + FP32-acc (current)")
run_kernel(matmul_pure_fp16, "Pure FP16 accumulation")
run_kernel(matmul_fp16dot_bigK, "FP16-dot + larger BK")

# Also test cuBLAS for reference
ms = do_bench(lambda: torch.mm(a, w)) * 1e-3
tflops = flops / ms / 1e12
print(f"{'cuBLAS torch.mm':30s}: {ms*1e6:.0f} us = {tflops:.1f} TFLOPS")
