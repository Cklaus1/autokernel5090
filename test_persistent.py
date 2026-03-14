"""Test persistent kernel approach - each SM processes multiple tiles."""
import torch
import triton, triton.language as tl
from triton.testing import do_bench

M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)
ref = torch.mm(a, w)

NUM_SMS = 170

@triton.jit
def matmul_persistent(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    total_tiles = num_m * num_n

    for tile_id in range(pid, total_tiles, NUM_SMS):
        # L2 grouping via tile ordering
        GRP: tl.constexpr = 8
        group_id = tile_id // (num_m * GRP)
        first_n = group_id * GRP
        gsn = min(num_n - first_n, GRP)
        pid_m = (tile_id % (num_m * gsn)) // gsn
        pid_n = first_n + (tile_id % gsn)

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


o = torch.empty(M, N, device='cuda', dtype=torch.float16)
def fn():
    matmul_persistent[(NUM_SMS,)](
        a, w, o, M, N, K,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        o.stride(0), o.stride(1),
        BM=128, BN=256, BK=64, NUM_SMS=NUM_SMS,
        num_stages=3, num_warps=8,
    )

fn(); torch.cuda.synchronize()
err = (o - ref).abs().max().item()
ms = do_bench(fn, warmup=25, rep=100)
tflops = flops / ms / 1e9
print(f"Persistent (NUM_SMS={NUM_SMS}): {ms*1000:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}")

# Compare with standard
@triton.jit
def matmul_std(A, B, C, M, N, K,
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

o2 = torch.empty(M, N, device='cuda', dtype=torch.float16)
grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
def fn2():
    matmul_std[grid](a, w, o2, M, N, K,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        o2.stride(0), o2.stride(1),
        BM=128, BN=256, BK=64, G=32, num_stages=3, num_warps=8)
fn2(); torch.cuda.synchronize()
ms2 = do_bench(fn2, warmup=25, rep=100)
tflops2 = flops / ms2 / 1e9
print(f"Standard (G=32):               {ms2*1000:.0f} us = {tflops2:.1f} TFLOPS")
