"""Test if CUDA graph eliminates launch overhead."""
import torch, sys
sys.path.insert(0, '/root/projects/autokernel')
from triton.testing import do_bench

torch.manual_seed(42)
M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K

def _pack_int4_weights(K, N, device):
    w = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    packed = torch.zeros(K//8, N, device=device, dtype=torch.int32)
    for i in range(8):
        packed |= (w[i::8] & 0xF) << (i * 4)
    return packed

activation = torch.randn(M, K, device='cuda', dtype=torch.float16)
packed_weights = _pack_int4_weights(K, N, 'cuda')
scales = torch.randn(K//128, N, device='cuda', dtype=torch.float16).abs() * 0.01 + 0.001
zeros = torch.randint(0, 16, (K//128, N), device='cuda').to(torch.float16)

import kernel

# Warm up
kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
torch.cuda.synchronize()

# Regular benchmark
def fn():
    kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
ms_reg = do_bench(fn, warmup=25, rep=100)
tflops_reg = flops / ms_reg / 1e9
print(f"Regular:    {ms_reg*1000:.0f} us = {tflops_reg:.1f} TFLOPS")

# With CUDA graph
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
torch.cuda.current_stream().wait_stream(s)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)

def fn_graph():
    g.replay()

ms_graph = do_bench(fn_graph, warmup=25, rep=100)
tflops_graph = flops / ms_graph / 1e9
print(f"CUDA Graph: {ms_graph*1000:.0f} us = {tflops_graph:.1f} TFLOPS")
print(f"Speedup: {ms_reg/ms_graph:.3f}x")
