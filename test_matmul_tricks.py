"""Test various low-level matmul tricks."""
import torch, sys
sys.path.insert(0, '/root/projects/autokernel')
import triton, triton.language as tl
from triton.testing import do_bench

torch.manual_seed(42)
M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)
wt = w.t().contiguous()  # [N, K]
ref = torch.mm(a, w)

# Baseline: current best
@triton.jit
def matmul_baseline(
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

# Test with eviction policy
@triton.jit
def matmul_evict(
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

    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1), eviction_policy='evict_first')
        bb = tl.load(b_ptr, boundary_check=(0, 1), eviction_policy='evict_first')
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))

    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

# Test with transposed B (A @ B.T)
@triton.jit
def matmul_trans(
    A, BT, C, M, N, K,
    stride_am, stride_ak, stride_btn, stride_btk, stride_cm, stride_cn,
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
    # BT is [N, K], we need [BK, BN] blocks -> transpose
    bt_ptr = tl.make_block_ptr(base=BT, shape=(N, K), strides=(stride_btn, stride_btk),
        offsets=(pid_n * BN, 0), block_shape=(BN, BK), order=(1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bbt = tl.load(bt_ptr, boundary_check=(0, 1))
        bb = tl.trans(bbt)
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        bt_ptr = tl.advance(bt_ptr, (0, BK))

    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)

tests = [
    ("baseline", matmul_baseline, w, lambda: None),
    ("evict_first", matmul_evict, w, lambda: None),
    ("transposed_B", matmul_trans, wt, lambda: None),
]

for name, kernel_fn, weight, setup in tests:
    o = torch.empty(M, N, device='cuda', dtype=torch.float16)
    try:
        def fn(kfn=kernel_fn, ww=weight):
            kfn[grid](a, ww, o, M, N, K,
                a.stride(0), a.stride(1), ww.stride(0), ww.stride(1),
                o.stride(0), o.stride(1),
                BM=128, BN=256, BK=64, G=32,
                num_stages=3, num_warps=8)
        fn(); torch.cuda.synchronize()
        err = (o - ref).abs().max().item()
        ms = do_bench(fn, warmup=25, rep=100)
        tflops = flops / ms / 1e9
        print(f"{name:20s}: {ms*1000:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}")
    except Exception as e:
        print(f"{name:20s}: FAILED ({str(e)[:80]})")
