"""Test matmul configs for qwen35_mlp_gate shape: M=2048, N=13824, K=5120."""
import torch
import triton, triton.language as tl
from triton.testing import do_bench

M, N, K = 2048, 13824, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)

@triton.jit
def matmul_test(A, B, C, M, N, K,
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
    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

configs = [
    (128, 256, 64, 3, 8, 32),
    (128, 256, 64, 3, 8, 16),
    (128, 256, 64, 3, 8, 8),
    (128, 128, 64, 3, 8, 32),
    (128, 128, 64, 3, 8, 8),
    (128, 256, 64, 2, 8, 32),
    (256, 128, 64, 3, 8, 32),
]

print(f"M={M}, N={N}, K={K}")
print(f"N/128={N/128:.1f}, N/256={N/256:.1f}, N%128={N%128}, N%256={N%256}")

for BM, BN, BK, stages, warps, G in configs:
    o = torch.empty(M, N, device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    try:
        def fn():
            matmul_test[grid](a, w, o, M, N, K,
                a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                o.stride(0), o.stride(1),
                BM=BM, BN=BN, BK=BK, G=G,
                num_stages=stages, num_warps=warps)
        fn(); torch.cuda.synchronize()
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = flops / ms / 1e9
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:2d} s={stages} G={G:2d} tiles={tiles:5d}: {ms*1000:.0f} us = {tflops:.1f} TFLOPS")
    except Exception as e:
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:2d} s={stages} G={G:2d}: FAILED")
