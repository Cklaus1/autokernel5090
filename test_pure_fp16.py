"""Test pure FP16 accumulation with real W4A16 data."""
import torch, sys
sys.path.insert(0, '/root/projects/autokernel')
import triton, triton.language as tl
from triton.testing import do_bench

torch.manual_seed(42)
M, N, K = 2048, 5120, 5120

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


@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 256, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=3, num_warps=8),
    ],
    key=['M_key', 'N_key', 'K_key'],
)
@triton.jit
def matmul_pure_fp16(
    A_ptr, W_ptr, O_ptr,
    M_key, N_key, K_key,
    stride_am, stride_ak, stride_wk, stride_wn, stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
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
    acc = tl.zeros((BM, BN), dtype=tl.float16)
    for _ in range(0, tl.cdiv(K_key, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        ww = tl.load(w_ptr, boundary_check=(0, 1))
        acc = tl.dot(aa, ww, acc, out_dtype=tl.float16)
        a_ptr = tl.advance(a_ptr, (0, BK))
        w_ptr = tl.advance(w_ptr, (BK, 0))
    o_ptr = tl.make_block_ptr(O_ptr, (M_key, N_key), (stride_om, stride_on),
                               (pid_m * BM, pid_n * BN), (BM, BN), (1, 0))
    tl.store(o_ptr, acc, boundary_check=(0, 1))


o = torch.empty(M, N, device='cuda', dtype=torch.float16)
def grid(META):
    return (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)
def fn():
    matmul_pure_fp16[grid](
        activation, W, o, M, N, K,
        activation.stride(0), activation.stride(1), W.stride(0), W.stride(1),
        o.stride(0), o.stride(1),
    )

fn(); torch.cuda.synchronize()
ms = do_bench(fn, warmup=25, rep=100) * 1e-3
tflops = 2*M*N*K / ms / 1e12
fn(); torch.cuda.synchronize()
err = (o - ref).abs().max().item()
print(f"Pure FP16 accum: {ms*1e6:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}")

atol, rtol = 0.05, 0.05
within = torch.allclose(o.float(), ref.float(), atol=atol, rtol=rtol)
abs_diff = (o.float() - ref.float()).abs()
pct_within = (abs_diff <= atol + rtol * ref.float().abs()).float().mean().item() * 100
print(f"Correctness: {within}, max_err={abs_diff.max().item():.4f}, within_tol={pct_within:.2f}%")
