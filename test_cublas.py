"""Test cuBLAS F.linear vs Triton matmul for the FP16 matmul step."""
import torch
import triton
from triton.testing import do_bench

torch.manual_seed(42)
M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(K, N, device='cuda', dtype=torch.float16)
wt = w.t().contiguous()

# cuBLAS via torch.mm
def fn_mm():
    torch.mm(a, w)
ms_mm = do_bench(fn_mm, warmup=25, rep=100)
print(f"torch.mm:     {ms_mm*1000:.0f} us = {flops/ms_mm/1e9:.1f} TFLOPS")

# cuBLAS via F.linear (weight is [N, K])
def fn_linear():
    torch.nn.functional.linear(a, wt)
ms_linear = do_bench(fn_linear, warmup=25, rep=100)
print(f"F.linear:     {ms_linear*1000:.0f} us = {flops/ms_linear/1e9:.1f} TFLOPS")

# cuBLAS via torch.matmul
def fn_matmul():
    torch.matmul(a, w)
ms_matmul = do_bench(fn_matmul, warmup=25, rep=100)
print(f"torch.matmul: {ms_matmul*1000:.0f} us = {flops/ms_matmul/1e9:.1f} TFLOPS")

# torch._int_mm for comparison (if available)
try:
    a_int = torch.randint(-128, 127, (M, K), device='cuda', dtype=torch.int8)
    w_int = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)
    def fn_int():
        torch._int_mm(a_int, w_int)
    ms_int = do_bench(fn_int, warmup=25, rep=100)
    print(f"torch._int_mm: {ms_int*1000:.0f} us = {flops/ms_int/1e9:.1f} TFLOPS")
except Exception as e:
    print(f"torch._int_mm: {e}")
