"""Test persistent kernel matmul - one program per SM, loops over tiles."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def matmul_persistent(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent matmul: each SM processes multiple output tiles."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    num_tiles = num_m * num_n

    # Each SM processes tiles in round-robin
    for tile_id in range(pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_n
        pid_n = tile_id % num_n

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


@triton.jit
def matmul_persistent_grouped(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    NUM_SMS: tl.constexpr, G: tl.constexpr,
):
    """Persistent matmul with L2-friendly tile grouping."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    num_tiles = num_m * num_n

    for tile_id in range(pid, num_tiles, NUM_SMS):
        # Swizzle for L2 locality
        group_id = tile_id // (num_m * G)
        first_n = group_id * G
        gsn = min(num_n - first_n, G)
        pid_m = (tile_id % (num_m * gsn)) // gsn
        pid_n = first_n + (tile_id % gsn)

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

NUM_SMS = 170  # RTX 5090

print("=== Persistent Kernel (round-robin) ===")
configs_p = [
    (256, 128, 32, 8, 3),
    (128, 256, 32, 8, 3),
    (128, 128, 32, 8, 3),
    (256, 128, 32, 8, 4),
    (128, 128, 64, 8, 3),
]

for BM, BN, BK, warps, stages in configs_p:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    label = f"BM={BM} BN={BN} BK={BK} w={warps} s={stages}"
    try:
        for _ in range(3):
            matmul_persistent[(NUM_SMS,)](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, NUM_SMS=NUM_SMS,
                num_warps=warps, num_stages=stages)
        torch.cuda.synchronize()
        t = do_bench(lambda: matmul_persistent[(NUM_SMS,)](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, NUM_SMS=NUM_SMS,
            num_warps=warps, num_stages=stages),
            warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        print(f"  {label:45s} {tflops:8.1f} TFLOPS")
    except Exception as e:
        print(f"  {label:45s} FAIL: {str(e)[:60]}")

print("\n=== Persistent Kernel (grouped swizzle) ===")
for BM, BN, BK, warps, stages in configs_p:
    for G in [4, 8]:
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)
        label = f"BM={BM} BN={BN} BK={BK} G={G} w={warps} s={stages}"
        try:
            for _ in range(3):
                matmul_persistent_grouped[(NUM_SMS,)](
                    a, b, c, M, N, K,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    BM=BM, BN=BN, BK=BK, NUM_SMS=NUM_SMS, G=G,
                    num_warps=warps, num_stages=stages)
            torch.cuda.synchronize()
            t = do_bench(lambda: matmul_persistent_grouped[(NUM_SMS,)](
                a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, NUM_SMS=NUM_SMS, G=G,
                num_warps=warps, num_stages=stages),
                warmup=50, rep=200)
            tflops = flops / (t * 1e-3) / 1e12
            print(f"  {label:45s} {tflops:8.1f} TFLOPS")
        except Exception as e:
            print(f"  {label:45s} FAIL: {str(e)[:60]}")

# Also test: non-persistent with G=4 to confirm previous finding
print("\n=== Non-persistent G=4 (confirmation) ===")

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
    a_block_ptr = tl.make_block_ptr(A, (M, K), (stride_am, stride_ak), (pid_m*BM, 0), (BM, BK), (1, 0))
    b_block_ptr = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn), (0, pid_n*BN), (BK, BN), (1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        partial = tl.dot(a, b, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BK))
        b_block_ptr = tl.advance(b_block_ptr, (BK, 0))
    c_block_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

for G in [2, 4, 6, 8, 12, 16]:
    BM, BN, BK = 256, 128, 32
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    t = do_bench(lambda: matmul_standard[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, num_warps=8, num_stages=3),
        warmup=50, rep=200)
    tflops = flops / (t * 1e-3) / 1e12
    print(f"  G={G:2d}: {tflops:.1f} TFLOPS")
