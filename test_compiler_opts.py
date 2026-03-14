"""Test Triton compiler options and num_ctas for potential gains."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

@triton.jit
def matmul_k(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group_id = pid // (num_m * G)
    first_n = group_id * G
    gsn = min(num_n - first_n, G)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)
    a_ptr = tl.make_block_ptr(A, (M, K), (stride_am, stride_ak), (pid_m*BM, 0), (BM, BK), (1, 0))
    b_ptr = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn), (0, pid_n*BN), (BK, BN), (1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a_tile = tl.load(a_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(a_tile, b_tile, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)

# Baseline
for _ in range(5):
    matmul_k[grid](a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3)
torch.cuda.synchronize()
t = do_bench(lambda: matmul_k[grid](a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
    c.stride(0), c.stride(1),
    BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3),
    warmup=50, rep=200)
print(f"Baseline (w=8, s=3):         {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Try num_ctas=2 (cooperative CTA)
for num_ctas in [2, 4]:
    try:
        for _ in range(5):
            matmul_k[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3, num_ctas=num_ctas)
        torch.cuda.synchronize()
        t = do_bench(lambda: matmul_k[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3, num_ctas=num_ctas),
            warmup=50, rep=200)
        print(f"num_ctas={num_ctas}:                 {flops/(t*1e-3)/1e12:.1f} TFLOPS")
    except Exception as e:
        print(f"num_ctas={num_ctas}: FAIL ({str(e)[:80]})")

# Try enable_fp_fusion
for fusion in [True, False]:
    try:
        for _ in range(5):
            matmul_k[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3,
                enable_fp_fusion=fusion)
        torch.cuda.synchronize()
        t = do_bench(lambda: matmul_k[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3,
            enable_fp_fusion=fusion),
            warmup=50, rep=200)
        print(f"fp_fusion={fusion}:            {flops/(t*1e-3)/1e12:.1f} TFLOPS")
    except Exception as e:
        print(f"fp_fusion={fusion}: FAIL ({str(e)[:80]})")

# Try maxnreg to limit register usage
for maxnreg in [64, 128, 256]:
    try:
        for _ in range(5):
            matmul_k[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3,
                maxnreg=maxnreg)
        torch.cuda.synchronize()
        t = do_bench(lambda: matmul_k[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3,
            maxnreg=maxnreg),
            warmup=50, rep=200)
        print(f"maxnreg={maxnreg}:                {flops/(t*1e-3)/1e12:.1f} TFLOPS")
    except Exception as e:
        print(f"maxnreg={maxnreg}: FAIL ({str(e)[:80]})")
