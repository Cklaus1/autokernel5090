"""Chunked FP16 accumulation: accumulate N FP16 dots, then flush to FP32."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

M, K, N_dim = 2048, 5120, 5120
flops = 2 * M * K * N_dim

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N_dim, device='cuda', dtype=torch.float16)
ref = torch.mm(a, w)


@triton.autotune(
    configs=[
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_stages=4, num_warps=8),
    ],
    key=['M_key', 'N_key', 'K_key'],
)
@triton.jit
def matmul_chunked(
    A_ptr, W_ptr, O_ptr,
    M_key, N_key, K_key,
    stride_am, stride_ak, stride_wk, stride_wn, stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M_key, BM)
    num_n = tl.cdiv(N_key, BN)
    GRP: tl.constexpr = 8
    group_id = pid // (num_m * GRP)
    first_n = group_id * GRP
    gsn = min(num_n - first_n, GRP)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)

    a_ptr = tl.make_block_ptr(A_ptr, (M_key, K_key), (stride_am, stride_ak),
                               (pid_m * BM, 0), (BM, BK), (1, 0))
    w_ptr = tl.make_block_ptr(W_ptr, (K_key, N_key), (stride_wk, stride_wn),
                               (0, pid_n * BN), (BK, BN), (1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    num_k = tl.cdiv(K_key, BK)

    for outer in range(0, tl.cdiv(num_k, CHUNK)):
        # Accumulate CHUNK iterations in FP16
        fp16_acc = tl.zeros((BM, BN), dtype=tl.float16)
        for inner in range(0, CHUNK):
            k_idx = outer * CHUNK + inner
            if k_idx < num_k:
                aa = tl.load(a_ptr, boundary_check=(0, 1))
                ww = tl.load(w_ptr, boundary_check=(0, 1))
                fp16_acc = tl.dot(aa, ww, fp16_acc, out_dtype=tl.float16)
                a_ptr = tl.advance(a_ptr, (0, BK))
                w_ptr = tl.advance(w_ptr, (BK, 0))
        # Flush to FP32
        acc += fp16_acc.to(tl.float32)

    o_ptr = tl.make_block_ptr(O_ptr, (M_key, N_key), (stride_om, stride_on),
                               (pid_m * BM, pid_n * BN), (BM, BN), (1, 0))
    tl.store(o_ptr, acc.to(tl.float16), boundary_check=(0, 1))


for chunk_size in [2, 4, 8, 16, 32]:
    o = torch.empty(M, N_dim, device='cuda', dtype=torch.float16)
    def grid(META):
        return (triton.cdiv(M, META['BM']) * triton.cdiv(N_dim, META['BN']),)
    def fn():
        matmul_chunked[grid](
            a, w, o, M, N_dim, K,
            a.stride(0), a.stride(1), w.stride(0), w.stride(1), o.stride(0), o.stride(1),
            CHUNK=chunk_size,
        )
    try:
        fn(); torch.cuda.synchronize()
        ms = do_bench(fn) * 1e-3
        tflops = flops / ms / 1e12
        fn(); torch.cuda.synchronize()
        err = (o - ref).abs().max().item()
        status = "PASS" if err < 0.05 else ("WARN" if err < 1.0 else "FAIL")
        print(f"chunk={chunk_size:3d}: {ms*1e6:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f} {status}")
    except Exception as e:
        print(f"chunk={chunk_size:3d}: FAILED - {e}")
    # Reset autotune cache for next chunk size
    matmul_chunked.cache.clear()

print(f"\nReference: cuBLAS = {flops / (do_bench(lambda: torch.mm(a, w)) * 1e-3) / 1e12:.1f} TFLOPS")
