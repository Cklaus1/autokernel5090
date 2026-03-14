"""Hybrid FP16/FP32 accumulation: accumulate N iterations in FP16, then flush to FP32.
This reduces FP32 add overhead (which costs ~15% throughput) while controlling precision."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def matmul_hybrid(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Every CHUNK iterations of FP16 dot + FP16 accum, flush to FP32."""
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

    for outer in range(0, num_k, CHUNK):
        # FP16 accumulator for this chunk
        acc_fp16 = tl.zeros((BM, BN), dtype=tl.float16)
        for inner in range(0, CHUNK):
            k_idx = outer + inner
            if k_idx < num_k:
                a = tl.load(a_block_ptr, boundary_check=(0, 1))
                b = tl.load(b_block_ptr, boundary_check=(0, 1))
                acc_fp16 = tl.dot(a, b, acc_fp16, out_dtype=tl.float16)
                a_block_ptr = tl.advance(a_block_ptr, (0, BK))
                b_block_ptr = tl.advance(b_block_ptr, (BK, 0))
        # Flush to FP32
        acc_fp32 += acc_fp16.to(tl.float32)

    c_block_ptr = tl.make_block_ptr(
        base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0)
    )
    tl.store(c_block_ptr, acc_fp32.to(tl.float16), boundary_check=(0, 1))


# Simulate W4A16 dequanted weights for accuracy testing
M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K
BM, BN, BK, G = 256, 128, 32, 8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
num_k_iters = K // BK  # 160

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
# W4A16-like: dequanted INT4 weights
b_w4 = ((torch.randint(0, 16, (K, N), device='cuda').float() - 8) * 0.25).to(torch.float16)
# Random weights
b_rand = torch.randn(K, N, device='cuda', dtype=torch.float16)

ref_w4 = torch.mm(a, b_w4)
ref_rand = torch.mm(a, b_rand)

print(f"K iterations: {num_k_iters}")
print(f"ref_w4 max: {ref_w4.abs().max().item():.1f}, ref_rand max: {ref_rand.abs().max().item():.1f}")
print()

print(f"{'CHUNK':>6s} {'TFLOPS_w4':>10s} {'err_w4':>8s} {'TFLOPS_rand':>12s} {'err_rand':>9s} {'PASS?':>6s}")
print("-" * 60)

for chunk in [1, 2, 4, 8, 16, 32, 160]:
    results = {}
    for label, wt, reference in [("w4", b_w4, ref_w4), ("rand", b_rand, ref_rand)]:
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)
        try:
            for _ in range(3):
                matmul_hybrid[grid](a, wt, c, M, N, K,
                    a.stride(0), a.stride(1), wt.stride(0), wt.stride(1),
                    c.stride(0), c.stride(1),
                    BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
                    num_warps=8, num_stages=3)
            torch.cuda.synchronize()
            matmul_hybrid[grid](a, wt, c, M, N, K,
                a.stride(0), a.stride(1), wt.stride(0), wt.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
                num_warps=8, num_stages=3)
            torch.cuda.synchronize()
            max_err = (c - reference).abs().max().item()

            t = do_bench(lambda: matmul_hybrid[grid](a, wt, c, M, N, K,
                a.stride(0), a.stride(1), wt.stride(0), wt.stride(1),
                c.stride(0), c.stride(1),
                BM=BM, BN=BN, BK=BK, G=G, CHUNK=chunk,
                num_warps=8, num_stages=3), warmup=50, rep=200)
            tflops = flops / (t * 1e-3) / 1e12
            results[label] = (tflops, max_err)
        except Exception as e:
            results[label] = (0, str(e)[:30])

    if isinstance(results.get('w4', (0,0))[1], float):
        w4_t, w4_e = results['w4']
        r_t, r_e = results.get('rand', (0, 0))
        passes = "YES" if w4_e < 0.05 else "NO"
        print(f"{chunk:6d} {w4_t:10.1f} {w4_e:8.4f} {r_t:12.1f} {r_e:9.4f} {passes:>6s}")
    else:
        print(f"{chunk:6d} FAIL: {results}")
