"""Measure Python overhead in kernel_fn vs raw matmul launch."""
import torch, sys, time
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

# Warm up - populate cache
kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
torch.cuda.synchronize()

# Measure kernel_fn with cached W
def fn_kernel():
    kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
ms_kernel = do_bench(fn_kernel, warmup=25, rep=100)
tflops_kernel = flops / ms_kernel / 1e9
print(f"kernel_fn (cached): {ms_kernel*1000:.0f} us = {tflops_kernel:.1f} TFLOPS")

# Measure raw matmul with pre-cached W
W = kernel._dequant_cache[list(kernel._dequant_cache.keys())[0]]
output = kernel._out_buf[(M, N, activation.dtype)]

def fn_raw():
    kernel.matmul_fp16_dot[kernel._matmul_grid](
        activation, W, output,
        M, N, K,
        activation.stride(0), activation.stride(1),
        W.stride(0), W.stride(1),
        output.stride(0), output.stride(1),
        USE_FP16_DOT=True,
    )
ms_raw = do_bench(fn_raw, warmup=25, rep=100)
tflops_raw = flops / ms_raw / 1e9
print(f"raw matmul:         {ms_raw*1000:.0f} us = {tflops_raw:.1f} TFLOPS")

print(f"\nPython overhead: {(ms_kernel - ms_raw)*1000:.0f} us ({(ms_kernel - ms_raw)/ms_kernel*100:.1f}%)")
