"""Test split-K matmul for more parallelism."""
import torch, sys
sys.path.insert(0, '/root/projects/autokernel')
import triton, triton.language as tl
from triton.testing import do_bench

torch.manual_seed(42)
M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K

def _pack_int4_weights(K, N, device):
    w = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    K_packed = K // 8
    packed = torch.zeros(K_packed, N, device=device, dtype=torch.int32)
    for i in range(8):
        packed |= (w[i::8] & 0xF) << (i * 4)
    return packed

activation = torch.randn(M, K, device='cuda', dtype=torch.float16)
packed_weights = _pack_int4_weights(K, N, 'cuda')
scales = torch.randn(K//128, N, device='cuda', dtype=torch.float16).abs() * 0.01 + 0.001
zeros = torch.randint(0, 16, (K//128, N), device='cuda').to(torch.float16)

import kernel
kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
W = kernel._dequant_cache[list(kernel._dequant_cache.keys())[0]]
ref = torch.mm(activation, W)


@triton.jit
def matmul_splitk(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group_id = pid_mn // (num_m * G)
    first_n = group_id * G
    gsn = min(num_n - first_n, G)
    pid_m = (pid_mn % (num_m * gsn)) // gsn
    pid_n = first_n + (pid_mn % gsn)

    # Split K dimension
    k_per_split = tl.cdiv(K, SPLIT_K * BK) * BK
    k_start = pid_k * k_per_split
    k_end = min(k_start + k_per_split, K)

    a_ptr = tl.make_block_ptr(
        base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, k_start), block_shape=(BM, BK), order=(1, 0))
    b_ptr = tl.make_block_ptr(
        base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(k_start, pid_n * BN), block_shape=(BK, BN), order=(1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(k_end - k_start, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

    # Atomic add for split-K reduction
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    m_mask = offs_m < M
    n_mask = offs_n < N
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])
    else:
        tl.atomic_add(c_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])


for SPLIT_K in [1, 2, 4, 8]:
    o = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256), SPLIT_K)
    try:
        def fn(sk=SPLIT_K):
            o.zero_()
            matmul_splitk[grid](
                activation, W, o, M, N, K,
                activation.stride(0), activation.stride(1), W.stride(0), W.stride(1),
                o.stride(0), o.stride(1),
                BM=128, BN=256, BK=64, G=32, SPLIT_K=sk,
                num_stages=3, num_warps=8,
            )
        fn(); torch.cuda.synchronize()
        err = (o - ref).abs().max().item()
        correct = err <= 0.05
        tiles = triton.cdiv(M, 128) * triton.cdiv(N, 256) * SPLIT_K
        waves = tiles / 170
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = flops / ms / 1e9
        print(f"SPLIT_K={SPLIT_K}: {ms*1000:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}  tiles={tiles} waves={waves:.1f}  {'PASS' if correct else 'FAIL'}")
    except Exception as e:
        short = str(e).split('\n')[0][:100]
        print(f"SPLIT_K={SPLIT_K}: FAILED ({short})")
