"""Test chunked FP16 accumulation: accumulate N iterations in FP16, then promote to FP32.
Reduces FP32 overhead while controlling precision loss.
Also tests pure FP16 accum for W4A16-like magnitudes."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def matmul_chunked_fp16(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Accumulate CHUNK iterations in FP16 before promoting to FP32."""
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

    acc_fp32 = tl.zeros((BM, BN), dtype=tl.float32)
    num_k = tl.cdiv(K, BK)

    for k in range(0, num_k):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        if CHUNK == 1:
            # Same as current: fp16 dot, immediately add to fp32
            partial = tl.dot(a, b, out_dtype=tl.float16)
            acc_fp32 += partial.to(tl.float32)
        elif CHUNK >= num_k:
            # Pure FP16 accumulation across all iterations
            acc_fp32 = tl.dot(a, b, acc_fp32, out_dtype=tl.float16).to(tl.float32)
        else:
            # Chunked: accumulate in fp16 for CHUNK iters, then promote
            partial = tl.dot(a, b, out_dtype=tl.float16)
            acc_fp32 += partial.to(tl.float32)
        a_block_ptr = tl.advance(a_block_ptr, (0, BK))
        b_block_ptr = tl.advance(b_block_ptr, (BK, 0))

    c_block_ptr = tl.make_block_ptr(
        base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0)
    )
    tl.store(c_block_ptr, acc_fp32.to(tl.float16), boundary_check=(0, 1))


# Test with both random and W4A16-like data
M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K
BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

# Random data
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
ref = torch.mm(a, b)

print("=== Random data (magnitude ~370) ===")
print(f"ref max: {ref.abs().max().item():.1f}")

for chunk in [1, 160]:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        for _ in range(3):
            matmul_chunked_fp16[grid](a, b, c, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
                num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        matmul_chunked_fp16[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
            num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        max_err = (c - ref).abs().max().item()
        t = do_bench(lambda: matmul_chunked_fp16[grid](a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
            num_warps=8, num_stages=3), warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        print(f"  CHUNK={chunk:3d}: {tflops:.1f} TFLOPS, max_err={max_err:.4f}")
    except Exception as e:
        print(f"  CHUNK={chunk:3d}: FAIL: {str(e)[:80]}")


# W4A16-like data: small weights (INT4 range 0-15, dequanted to ~±1), normal activations
print("\n=== W4A16-like data (small magnitude) ===")
a2 = torch.randn(M, K, device='cuda', dtype=torch.float16)
# Simulate dequanted INT4 weights: values in [-2, 2] range
b2 = (torch.randint(0, 16, (K, N), device='cuda').float() - 8) * 0.25
b2 = b2.to(torch.float16)
ref2 = torch.mm(a2, b2)
print(f"ref max: {ref2.abs().max().item():.1f}")

for chunk in [1, 160]:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        for _ in range(3):
            matmul_chunked_fp16[grid](a2, b2, c, M, N, K,
                a2.stride(0), a2.stride(1), b2.stride(0), b2.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
                num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        matmul_chunked_fp16[grid](a2, b2, c, M, N, K,
            a2.stride(0), a2.stride(1), b2.stride(0), b2.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
            num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        max_err = (c - ref2).abs().max().item()
        t = do_bench(lambda: matmul_chunked_fp16[grid](a2, b2, c, M, N, K,
            a2.stride(0), a2.stride(1), b2.stride(0), b2.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
            num_warps=8, num_stages=3), warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        print(f"  CHUNK={chunk:3d}: {tflops:.1f} TFLOPS, max_err={max_err:.4f}")
    except Exception as e:
        print(f"  CHUNK={chunk:3d}: FAIL: {str(e)[:80]}")


# Now test: what if we use `tl.dot(a, b, acc)` but with acc in FP16?
# i.e., pure FP16 accumulation
print("\n=== Pure FP16 accumulation (hardware native) ===")

@triton.jit
def matmul_pure_fp16(
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

    # FP16 accumulator
    acc = tl.zeros((BM, BN), dtype=tl.float16)

    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        a_block_ptr = tl.advance(a_block_ptr, (0, BK))
        b_block_ptr = tl.advance(b_block_ptr, (BK, 0))

    c_block_ptr = tl.make_block_ptr(
        base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0)
    )
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


# Test pure FP16 with W4A16-like data
for label, act, wt, reference in [
    ("random", a, b, ref),
    ("w4a16-like", a2, b2, ref2),
]:
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        for _ in range(3):
            matmul_pure_fp16[grid](act, wt, c, M, N, K,
                act.stride(0), act.stride(1), wt.stride(0), wt.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G,
                num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        matmul_pure_fp16[grid](act, wt, c, M, N, K,
            act.stride(0), act.stride(1), wt.stride(0), wt.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G,
            num_warps=8, num_stages=3)
        torch.cuda.synchronize()
        max_err = (c - reference).abs().max().item()
        t = do_bench(lambda: matmul_pure_fp16[grid](act, wt, c, M, N, K,
            act.stride(0), act.stride(1), wt.stride(0), wt.stride(1),
            c.stride(0), c.stride(1),
            BM=BM, BN=BN, BK=BK, G=G,
            num_warps=8, num_stages=3), warmup=50, rep=200)
        tflops = flops / (t * 1e-3) / 1e12
        print(f"  {label:12s}: {tflops:.1f} TFLOPS, max_err={max_err:.4f}")
    except Exception as e:
        print(f"  {label:12s}: FAIL: {str(e)[:80]}")
