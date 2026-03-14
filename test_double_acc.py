"""Test double-buffered FP16 accumulators to overlap FP32 add with FP16 MMA."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

# Idea: Use two FP16 accumulators alternating, so the FP32 add of acc_a
# can overlap with the FP16 MMA writing to acc_b

@triton.jit
def matmul_double_acc(
    A, B, C, M, N, K: tl.constexpr,
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

    # Unrolled by 2: do two FP16 dots, then two FP32 adds
    for k in range(0, num_k, 2):
        a0 = tl.load(a_ptr, boundary_check=(0, 1))
        b0 = tl.load(b_ptr, boundary_check=(0, 1))
        p0 = tl.dot(a0, b0, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        a1 = tl.load(a_ptr, boundary_check=(0, 1))
        b1 = tl.load(b_ptr, boundary_check=(0, 1))
        p1 = tl.dot(a1, b1, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        # Add both FP16 partials to FP32 accumulator
        acc += p0.to(tl.float32)
        acc += p1.to(tl.float32)

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Unrolled by 4
@triton.jit
def matmul_unroll4(
    A, B, C, M, N, K: tl.constexpr,
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
        a0 = tl.load(a_ptr, boundary_check=(0, 1))
        b0 = tl.load(b_ptr, boundary_check=(0, 1))
        p0 = tl.dot(a0, b0, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

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

        a3 = tl.load(a_ptr, boundary_check=(0, 1))
        b3 = tl.load(b_ptr, boundary_check=(0, 1))
        p3 = tl.dot(a3, b3, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

        # Sum FP16 partials in FP16 first, then single FP32 add
        s01 = p0 + p1  # FP16 + FP16 = FP16
        s23 = p2 + p3
        s0123 = s01 + s23
        acc += s0123.to(tl.float32)  # single FP32 add per 4 iterations!

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Baseline: standard fp16-dot + fp32-add
@triton.jit
def matmul_baseline(
    A, B, C, M, N, K: tl.constexpr,
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
        partial = tl.dot(a, b, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
ref = torch.mm(a, b)

BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)

for name, kern, bk in [
    ("baseline (BK=32)", matmul_baseline, 32),
    ("unroll2 (BK=32)", matmul_double_acc, 32),
    ("unroll4 (BK=32)", matmul_unroll4, 32),
]:
    try:
        g = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        kern[g](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=bk, G=G,
                num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        err = (c - ref).abs().max().item()
        t = do_bench(lambda: kern[g](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=bk, G=G,
                num_warps=8, num_stages=3), warmup=50, rep=200)
        print(f"{name:30s}: {flops/(t*1e-3)/1e12:.1f} TFLOPS, err={err:.4f}")
    except Exception as e:
        print(f"{name:30s}: FAIL - {type(e).__name__}: {str(e)[:100]}")
