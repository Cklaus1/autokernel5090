"""Test chained FP16 dot: use tl.dot(a, b, prev, out_dtype=fp16) to chain
FP16 accumulations in tensor core, flush to FP32 every N steps.
This should get closer to pure FP16 throughput while controlling precision."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


# Version A: Chain 2 FP16 dots before FP32 flush (manually unrolled)
@triton.jit
def matmul_chain2(
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

    # Process pairs: chain 2 FP16 dots, then flush to FP32
    for k in range(0, num_k, 2):
        a1 = tl.load(a_ptr, boundary_check=(0, 1))
        b1 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a1, b1, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a2 = tl.load(a_ptr, boundary_check=(0, 1))
        b2 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a2, b2, p, out_dtype=tl.float16)  # chain: accumulate in FP16
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        acc += p.to(tl.float32)  # flush to FP32

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Version B: Chain 4 FP16 dots before FP32 flush
@triton.jit
def matmul_chain4(
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

    for k in range(0, num_k, 4):
        a1 = tl.load(a_ptr, boundary_check=(0, 1))
        b1 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a1, b1, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a2 = tl.load(a_ptr, boundary_check=(0, 1))
        b2 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a2, b2, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a3 = tl.load(a_ptr, boundary_check=(0, 1))
        b3 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a3, b3, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a4 = tl.load(a_ptr, boundary_check=(0, 1))
        b4 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a4, b4, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        acc += p.to(tl.float32)

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Version C: Chain 8 FP16 dots (K=5120, BK=32 -> 160 iters, 160/8=20 flushes)
@triton.jit
def matmul_chain8(
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

    for k in range(0, num_k, 8):
        a1 = tl.load(a_ptr, boundary_check=(0, 1))
        b1 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a1, b1, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a2 = tl.load(a_ptr, boundary_check=(0, 1))
        b2 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a2, b2, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a3 = tl.load(a_ptr, boundary_check=(0, 1))
        b3 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a3, b3, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a4 = tl.load(a_ptr, boundary_check=(0, 1))
        b4 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a4, b4, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a5 = tl.load(a_ptr, boundary_check=(0, 1))
        b5 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a5, b5, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a6 = tl.load(a_ptr, boundary_check=(0, 1))
        b6 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a6, b6, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a7 = tl.load(a_ptr, boundary_check=(0, 1))
        b7 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a7, b7, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a8 = tl.load(a_ptr, boundary_check=(0, 1))
        b8 = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a8, b8, p, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        acc += p.to(tl.float32)

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Reference: current approach (chain=1)
@triton.jit
def matmul_chain1(
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
        a = tl.load(a_ptr, boundary_check=(0, 1))
        b = tl.load(b_ptr, boundary_check=(0, 1))
        p = tl.dot(a, b, out_dtype=tl.float16)
        acc += p.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K
BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
ref = torch.mm(a, b)

print(f"{'Variant':15s} {'TFLOPS':>8s} {'max_err':>8s}")
print("-" * 35)

for name, kernel in [
    ("chain1 (curr)", matmul_chain1),
    ("chain2", matmul_chain2),
    ("chain4", matmul_chain4),
    ("chain8", matmul_chain8),
]:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        for _ in range(3):
            kernel[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        kernel[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        max_err = (c - ref).abs().max().item()
        t = do_bench(lambda: kernel[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3),
            warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        print(f"{name:15s} {tflops:8.1f} {max_err:8.4f}")
    except Exception as e:
        print(f"{name:15s} FAIL: {str(e)[:60]}")
