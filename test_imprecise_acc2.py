"""Test max_num_imprecise_acc parameter in tl.dot for controlled FP16 accumulation."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def matmul_imprecise(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    MAX_IMP: tl.constexpr,
):
    """Use tl.dot with FP32 output but max_num_imprecise_acc to control FP16 partial accumulation."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group_id = pid // (num_m * G)
    first_n = group_id * G
    gsn = min(num_n - first_n, G)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)

    a_block_ptr = tl.make_block_ptr(
        base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(1, 0)
    )

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, max_num_imprecise_acc=MAX_IMP)
        a_block_ptr = tl.advance(a_block_ptr, (0, BK))
        b_block_ptr = tl.advance(b_block_ptr, (BK, 0))

    c_block_ptr = tl.make_block_ptr(
        base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0)
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

BM, BN, BK, G = 256, 128, 32, 8

# Reference result with full FP32 accumulation
ref = torch.nn.functional.linear(a, torch.randn(N, K, device='cuda', dtype=torch.float16))
# Just use torch.mm for reference
ref = torch.mm(a, b)

print(f"{'MAX_IMP':>8s} {'TFLOPS':>8s} {'max_err':>8s} {'mean_err':>10s}")
print("-" * 40)

grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

for max_imp in [0, 1, 2, 4, 8, 16, 32, 64, 128, 160]:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        # Warmup
        for _ in range(3):
            matmul_imprecise[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, MAX_IMP=max_imp,
                num_warps=8, num_stages=3)
        torch.cuda.synchronize()

        # Check accuracy
        matmul_imprecise[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, MAX_IMP=max_imp,
            num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        max_err = (c - ref).abs().max().item()
        mean_err = (c - ref).abs().mean().item()

        t = do_bench(lambda: matmul_imprecise[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, MAX_IMP=max_imp,
            num_warps=8, num_stages=3),
            warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        print(f"{max_imp:8d} {tflops:8.1f} {max_err:8.4f} {mean_err:10.6f}")
    except Exception as e:
        print(f"{max_imp:8d} FAIL: {str(e)[:60]}")
