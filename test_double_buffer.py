"""Test: what if we manually double-buffer in registers?
Use BK=128 but split into 2 BK=64 dots per iteration, 
overlapping the next load with the current compute."""
import torch
import triton, triton.language as tl
from triton.testing import do_bench

M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)

@triton.jit
def matmul_unrolled2(A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    """Manually unrolled 2x: load next tile while computing current."""
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
    num_iters = tl.cdiv(K, BK)
    
    # Peel first iteration
    aa = tl.load(a_ptr, boundary_check=(0, 1))
    bb = tl.load(b_ptr, boundary_check=(0, 1))
    a_ptr = tl.advance(a_ptr, (0, BK))
    b_ptr = tl.advance(b_ptr, (BK, 0))
    
    for k in range(1, num_iters):
        # Compute current while loading next
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    
    # Last iteration
    partial = tl.dot(aa, bb, out_dtype=tl.float16)
    acc += partial.to(tl.float32)
    
    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m*BM, pid_n*BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

# Standard for comparison
@triton.jit
def matmul_std(A, B, C, M, N, K,
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

grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)

for name, kfn in [("standard", matmul_std), ("unrolled_pipelined", matmul_unrolled2)]:
    o = torch.empty(M, N, device='cuda', dtype=torch.float16)
    def fn(k=kfn):
        k[grid](a, w, o, M, N, K,
            a.stride(0), a.stride(1), w.stride(0), w.stride(1),
            o.stride(0), o.stride(1),
            BM=128, BN=256, BK=64, G=32, num_stages=3, num_warps=8)
    fn(); torch.cuda.synchronize()
    results = []
    for i in range(5):
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = flops / ms / 1e9
        results.append(tflops)
    avg = sum(results) / len(results)
    print(f"{name:25s}: avg={avg:.1f} TFLOPS  [{', '.join(f'{t:.1f}' for t in results)}]")
