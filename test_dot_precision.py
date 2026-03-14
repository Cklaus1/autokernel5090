"""Test tl.dot input_precision and allow_tf32 options."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

@triton.jit
def matmul_prec(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    PREC: tl.constexpr,
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
        if PREC == 0:
            # Current approach
            partial = tl.dot(a_tile, b_tile, out_dtype=tl.float16)
            acc += partial.to(tl.float32)
        elif PREC == 1:
            # Try input_precision="ieee"
            partial = tl.dot(a_tile, b_tile, out_dtype=tl.float16, input_precision="ieee")
            acc += partial.to(tl.float32)
        elif PREC == 2:
            # Try input_precision="tf32"
            partial = tl.dot(a_tile, b_tile, out_dtype=tl.float16, input_precision="tf32")
            acc += partial.to(tl.float32)
        elif PREC == 3:
            # FP32 with allow_tf32
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=True)
        elif PREC == 4:
            # FP32 without allow_tf32
            acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

for prec, label in [(0, "fp16_dot (current)"),
                     (1, "fp16_dot ieee"),
                     (2, "fp16_dot tf32"),
                     (3, "fp32_dot allow_tf32"),
                     (4, "fp32_dot no_tf32")]:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        for _ in range(5):
            matmul_prec[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, PREC=prec, num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        t = do_bench(lambda: matmul_prec[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, PREC=prec, num_warps=8, num_stages=3),
            warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        print(f"{label:25s}: {tflops:.1f} TFLOPS")
    except Exception as e:
        print(f"{label:25s}: FAIL ({str(e)[:60]})")
