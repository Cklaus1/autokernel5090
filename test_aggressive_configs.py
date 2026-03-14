"""Test aggressive matmul configs to find higher throughput."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

@triton.jit
def matmul_fp16_dot_fixed(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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
        partial = tl.dot(a, b, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
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

configs = [
    # (BM, BN, BK, G, warps, stages)
    # Current best
    (256, 128, 32, 8, 8, 3),
    (256, 128, 32, 8, 8, 4),
    (128, 256, 32, 8, 8, 3),
    (128, 256, 32, 8, 8, 4),
    # Larger tiles
    (256, 256, 32, 8, 8, 3),
    (256, 256, 32, 8, 8, 4),
    (256, 256, 64, 8, 8, 3),
    # More stages (deeper pipeline)
    (256, 128, 32, 8, 8, 5),
    (256, 128, 32, 8, 8, 6),
    (128, 256, 32, 8, 8, 5),
    # Different BK
    (256, 128, 64, 8, 8, 3),
    (256, 128, 64, 8, 8, 4),
    (128, 256, 64, 8, 8, 3),
    (128, 256, 64, 8, 8, 4),
    (256, 128, 128, 8, 8, 3),
    # Different grouping
    (256, 128, 32, 4, 8, 3),
    (256, 128, 32, 16, 8, 3),
    # More warps
    (256, 128, 32, 8, 16, 3),
    (128, 256, 32, 8, 16, 3),
    # Fewer warps
    (256, 128, 32, 8, 4, 3),
    (128, 128, 32, 8, 4, 3),
]

print(f"{'Config':50s} {'TFLOPS':>8s}")
print("-" * 60)

best_tflops = 0
best_config = None

for BM, BN, BK, G, warps, stages in configs:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    label = f"BM={BM} BN={BN} BK={BK} G={G} w={warps} s={stages}"
    try:
        # Warmup
        for _ in range(3):
            matmul_fp16_dot_fixed[grid](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, num_warps=warps, num_stages=stages)
        torch.cuda.synchronize()
        t = do_bench(lambda: matmul_fp16_dot_fixed[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=warps, num_stages=stages),
            warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        marker = " ***" if tflops > best_tflops else ""
        print(f"{label:50s} {tflops:8.1f}{marker}")
        if tflops > best_tflops:
            best_tflops = tflops
            best_config = (BM, BN, BK, G, warps, stages)
    except Exception as e:
        print(f"{label:50s} FAIL: {str(e)[:50]}")

print(f"\nBest: {best_config} -> {best_tflops:.1f} TFLOPS")
