"""Test different Triton FP16-dot implementations to find the fastest."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# Version 1: Current approach - fp16 dot, explicit convert, add to fp32
@triton.jit
def matmul_v1(
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
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr, boundary_check=(0, 1))
        b = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(a, b, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Version 2: fp16 dot, accumulate into fp32 via tl.dot with fp32 acc but out_dtype=fp16
# This should be equivalent but let the compiler optimize
@triton.jit
def matmul_v2(
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

    # Use fp32 acc with fp32 dot, but store as fp16
    # This is standard FP32 accum — baseline comparison
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr, boundary_check=(0, 1))
        b = tl.load(b_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Version 3: Unrolled - process 2 K-iterations before adding to fp32
@triton.jit
def matmul_v3(
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
    num_k = tl.cdiv(K, BK)
    # Process pairs of iterations
    for k in range(0, num_k, 2):
        a1 = tl.load(a_ptr, boundary_check=(0, 1))
        b1 = tl.load(b_ptr, boundary_check=(0, 1))
        p1 = tl.dot(a1, b1, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a2 = tl.load(a_ptr, boundary_check=(0, 1))
        b2 = tl.load(b_ptr, boundary_check=(0, 1))
        p2 = tl.dot(a2, b2, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        acc += p1.to(tl.float32) + p2.to(tl.float32)
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

for name, kernel in [("v1 (fp16_dot+fp32_add)", matmul_v1),
                      ("v2 (fp32_accum)", matmul_v2),
                      ("v3 (unrolled_2x)", matmul_v3)]:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        for _ in range(3):
            kernel[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        t = do_bench(lambda: kernel[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3),
            warmup=50, rep=200)
        print(f"{name:25s}: {flops/(t*1e-3)/1e12:.1f} TFLOPS")
    except Exception as e:
        print(f"{name:25s}: FAIL ({str(e)[:60]})")
