"""Test chunked FP16 accumulation: FP16 within chunks, FP32 between chunks.
This reduces FP32 conversion overhead while maintaining accuracy."""
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
def matmul_chunked(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    CHUNK: tl.constexpr,  # accumulate this many BK iters in FP16 before converting to FP32
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

    acc_f32 = tl.zeros((BM, BN), dtype=tl.float32)
    num_iters = tl.cdiv(K, BK)

    for k in range(0, num_iters):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))

        if CHUNK == 1:
            # Every iteration: FP16 dot + FP32 add (original approach)
            partial = tl.dot(aa, bb, out_dtype=tl.float16)
            acc_f32 += partial.to(tl.float32)
        else:
            # Accumulate CHUNK iterations in FP16
            if k % CHUNK == 0:
                acc_f16 = tl.dot(aa, bb, out_dtype=tl.float16)
            else:
                acc_f16 = tl.dot(aa, bb, acc_f16, out_dtype=tl.float16)

            if (k % CHUNK == CHUNK - 1) or (k == num_iters - 1):
                acc_f32 += acc_f16.to(tl.float32)

        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

    c_ptr = tl.make_block_ptr(
        base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc_f32.to(tl.float16), boundary_check=(0, 1))


for CHUNK in [1, 2, 4, 8, 80]:  # 80 = all iters in FP16 (K/BK=80)
    o = torch.empty(M, N, device='cuda', dtype=torch.float16)
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
    try:
        def fn(chunk=CHUNK):
            matmul_chunked[grid](
                activation, W, o, M, N, K,
                activation.stride(0), activation.stride(1), W.stride(0), W.stride(1),
                o.stride(0), o.stride(1),
                BM=128, BN=256, BK=64, G=32, CHUNK=chunk,
                num_stages=3, num_warps=8,
            )
        fn(); torch.cuda.synchronize()
        err = (o - ref).abs().max().item()
        correct = err <= 0.05
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = flops / ms / 1e9
        print(f"CHUNK={CHUNK:2d}: {ms*1000:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}  {'PASS' if correct else 'FAIL'}")
    except Exception as e:
        print(f"CHUNK={CHUNK:2d}: FAILED ({e})")
