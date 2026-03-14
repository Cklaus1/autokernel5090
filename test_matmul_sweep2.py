"""Sweep matmul configs with Triton 3.6.0 on real W4A16 data."""
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


@triton.jit
def matmul_test(
    A, B, C,
    M, N, K,
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

    a_ptr = tl.make_block_ptr(
        base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))
    b_ptr = tl.make_block_ptr(
        base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

    c_ptr = tl.make_block_ptr(
        base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


configs = [
    # (BM, BN, BK, stages, warps, G)
    (128, 256, 64, 3, 8, 8),
    (128, 256, 64, 4, 8, 8),
    (128, 256, 64, 5, 8, 8),
    (128, 256, 64, 2, 8, 8),
    (256, 128, 64, 3, 8, 8),
    (256, 128, 64, 4, 8, 8),
    (128, 128, 64, 3, 8, 8),
    (128, 128, 64, 4, 8, 8),
    # BK=128
    (128, 256, 128, 2, 8, 8),
    (128, 256, 128, 3, 8, 8),
    (256, 128, 128, 2, 8, 8),
    (256, 128, 128, 3, 8, 8),
    (128, 128, 128, 2, 8, 8),
    (128, 128, 128, 3, 8, 8),
    # Different group sizes
    (128, 256, 64, 3, 8, 4),
    (128, 256, 64, 3, 8, 16),
    (128, 256, 64, 3, 8, 32),
    # 256x256
    (256, 256, 64, 2, 8, 8),
    (256, 256, 64, 3, 8, 8),
]

print(f"Triton {triton.__version__}, M={M}, K={K}, N={N}")
ref = torch.mm(activation, W)

results = []
for BM, BN, BK, stages, warps, G in configs:
    o = torch.empty(M, N, device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    try:
        def fn():
            matmul_test[grid](
                activation, W, o, M, N, K,
                activation.stride(0), activation.stride(1), W.stride(0), W.stride(1), o.stride(0), o.stride(1),
                BM=BM, BN=BN, BK=BK, G=G,
                num_stages=stages, num_warps=warps,
            )
        fn(); torch.cuda.synchronize()
        err = (o - ref).abs().max().item()
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = flops / ms / 1e9
        results.append((tflops, BM, BN, BK, stages, warps, G, err))
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:3d} s={stages} w={warps} G={G:2d}: {ms*1000:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}")
    except Exception as e:
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:3d} s={stages} w={warps} G={G:2d}: FAILED ({e})")

results.sort(reverse=True)
print(f"\nTop 5:")
for tflops, BM, BN, BK, stages, warps, G, err in results[:5]:
    print(f"  {tflops:.1f} TFLOPS: BM={BM}, BN={BN}, BK={BK}, s={stages}, w={warps}, G={G}, err={err:.4f}")
