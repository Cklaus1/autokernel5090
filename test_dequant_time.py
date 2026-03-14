"""Measure dequant kernel time separately."""
import torch, sys
sys.path.insert(0, '/root/projects/autokernel')
import triton
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

# Just dequant
W = torch.empty((K, N), device='cuda', dtype=torch.float16)
def fn_dequant():
    kernel.dequant_kernel[kernel._dequant_grid](
        packed_weights, scales, zeros, W,
        K, N,
        packed_weights.stride(0), packed_weights.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        W.stride(0), W.stride(1),
        QUANT_GROUP_SIZE=128,
    )

fn_dequant(); torch.cuda.synchronize()
ms_dq = do_bench(fn_dequant, warmup=25, rep=100)
print(f"Dequant only: {ms_dq*1000:.0f} us")

# Full kernel_fn (first call, no cache)
kernel._dequant_cache.clear()
def fn_full_uncached():
    kernel._dequant_cache.clear()
    return kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)

fn_full_uncached(); torch.cuda.synchronize()
ms_full = do_bench(fn_full_uncached, warmup=5, rep=50)
print(f"Full (uncached): {ms_full*1000:.0f} us")

# Full kernel_fn (cached)
kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
def fn_full_cached():
    return kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
ms_cached = do_bench(fn_full_cached, warmup=25, rep=100)
print(f"Full (cached):   {ms_cached*1000:.0f} us")

print(f"\nDequant overhead: {ms_dq*1000:.0f} us = {ms_dq/ms_cached*100:.1f}% of cached time")
