"""Test warp specialization: some warps load data, others compute."""
import torch
import triton, triton.language as tl
from triton.testing import do_bench

M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)

# Test: what if we use different BK values with adjusted stages?
# The insight: with shared mem limit 101376 bytes, we can fit:
# BM=128, BN=256: per-stage = (128+256)*64*2 = 49152 -> 2 stages = 98304 < 101376
# BM=128, BN=128: per-stage = (128+128)*64*2 = 32768 -> 3 stages = 98304 < 101376  
# BM=128, BN=128, BK=128: per-stage = (128+128)*128*2 = 65536 -> 1 stage = 65536

@triton.jit
def matmul_test(A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM); num_n = tl.cdiv(N, BN)
    group_id = pid // (num_m * G); first_n = group_id * G
    gsn = min(num_n - first_n, G)
    pid_m = (pid % (num_m * gsn)) // gsn; pid_n = first_n + (pid % gsn)
    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m*BM, 0), block_shape=(BM, BK), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n*BN), block_shape=(BK, BN), order=(1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK)); b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m*BM, pid_n*BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

# Try all viable configs that fit in 101376 shared memory
# Formula: shared = (BM + BN) * BK * 2 * effective_stages
# Where effective_stages = min(requested_stages, floor(101376 / ((BM+BN)*BK*2)))
configs = [
    # Current best
    (128, 256, 64, 3, 8, 32),  # 98304 bytes (2 eff stages)
    # Can we fit more compute per shared mem byte?
    (128, 256, 32, 3, 8, 32),  # 49152 * 3 stages would be too much... 49152*2=98304 
    # What about asymmetric tiles that pack better?
    (64, 256, 64, 3, 8, 32),   # (64+256)*64*2 = 40960 -> 2 stages = 81920
    (96, 256, 64, 3, 8, 32),   # (96+256)*64*2 = 45056 -> 2 stages = 90112
    (128, 192, 64, 3, 8, 32),  # (128+192)*64*2 = 40960 -> 2 stages = 81920
    (160, 256, 64, 3, 8, 32),  # (160+256)*64*2 = 53248 -> 1 stage
    (128, 256, 48, 3, 8, 32),  # (128+256)*48*2 = 36864 -> 2 stages = 73728
    (192, 128, 64, 3, 8, 32),  # same as 128,192 but transposed
    (192, 192, 64, 3, 8, 32),  # (192+192)*64*2 = 49152 -> 2 stages = 98304
]

for BM, BN, BK, stages, warps, G in configs:
    o = torch.empty(M, N, device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    shmem_per_stage = (BM + BN) * BK * 2
    try:
        def fn():
            matmul_test[grid](a, w, o, M, N, K,
                a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                o.stride(0), o.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, num_stages=stages, num_warps=warps)
        fn(); torch.cuda.synchronize()
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = flops / ms / 1e9
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:2d} tiles={tiles:4d} shmem/stg={shmem_per_stage:6d}: {ms*1000:.0f} us = {tflops:.1f} TFLOPS")
    except Exception as e:
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:2d}: FAILED ({str(e)[:60]})")
