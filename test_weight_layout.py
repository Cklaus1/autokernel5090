"""Test different weight memory layouts for better cache performance."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)

# Standard K×N layout (row-major)
b_kn = torch.randn(K, N, device='cuda', dtype=torch.float16)

# N×K layout (for column-major access)
b_nk = b_kn.t().contiguous()

# Blocked layout: reshape to (K//BK, N//BN, BK, BN) for tile-level locality
BK_TILE, BN_TILE = 32, 128

@triton.jit
def matmul_standard(
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
        at = tl.load(a_ptr, boundary_check=(0, 1))
        bt = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(at, bt, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Test B in K×N layout (row-major, current)
BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

c = torch.empty(M, N, device='cuda', dtype=torch.float16)
for _ in range(5):
    matmul_standard[grid](a, b_kn, c, M, N, K,
        a.stride(0), a.stride(1), b_kn.stride(0), b_kn.stride(1),
        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, G=G,
        num_warps=8, num_stages=3)
torch.cuda.synchronize()
t = do_bench(lambda: matmul_standard[grid](a, b_kn, c, M, N, K,
    a.stride(0), a.stride(1), b_kn.stride(0), b_kn.stride(1),
    c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, G=G,
    num_warps=8, num_stages=3), warmup=50, rep=200)
print(f"B as K×N (row-major):  {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Test B transposed: use N×K layout with swapped strides
# B is (K, N) but we pass it as if it's column-major
for _ in range(5):
    matmul_standard[grid](a, b_nk, c, M, N, K,
        a.stride(0), a.stride(1), b_nk.stride(1), b_nk.stride(0),
        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, G=G,
        num_warps=8, num_stages=3)
torch.cuda.synchronize()
t = do_bench(lambda: matmul_standard[grid](a, b_nk, c, M, N, K,
    a.stride(0), a.stride(1), b_nk.stride(1), b_nk.stride(0),
    c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, G=G,
    num_warps=8, num_stages=3), warmup=50, rep=200)
print(f"B as N×K (col-major):  {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Test with order=(0,1) for B (column-major block ptr)
@triton.jit
def matmul_colmajor_b(
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
    # Column-major order for B
    b_ptr = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn), (0, pid_n*BN), (BK, BN), (0, 1))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        at = tl.load(a_ptr, boundary_check=(0, 1))
        bt = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(at, bt, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

# B stored as column-major (N×K contiguous)
b_colmaj = b_kn.t().contiguous().t()  # K×N but column-major stride
for _ in range(5):
    matmul_colmajor_b[grid](a, b_colmaj, c, M, N, K,
        a.stride(0), a.stride(1), b_colmaj.stride(0), b_colmaj.stride(1),
        c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, G=G,
        num_warps=8, num_stages=3)
torch.cuda.synchronize()
t = do_bench(lambda: matmul_colmajor_b[grid](a, b_colmaj, c, M, N, K,
    a.stride(0), a.stride(1), b_colmaj.stride(0), b_colmaj.stride(1),
    c.stride(0), c.stride(1), BM=BM, BN=BN, BK=BK, G=G,
    num_warps=8, num_stages=3), warmup=50, rep=200)
print(f"B col-major order:     {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Test F.linear (cuBLAS with N×K layout, which is what F.linear expects)
t = do_bench(lambda: torch.nn.functional.linear(a, b_nk), warmup=50, rep=200)
print(f"F.linear (cuBLAS):     {flops/(t*1e-3)/1e12:.1f} TFLOPS")
