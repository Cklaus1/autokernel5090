"""Test torch.compile / inductor for matmul performance."""

import torch
import torch._dynamo
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
w = torch.randn(N, K, device='cuda', dtype=torch.float16)  # for F.linear

# Baseline: raw torch.mm
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
print(f"torch.mm:              {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Baseline: F.linear
t = do_bench(lambda: torch.nn.functional.linear(a, w), warmup=50, rep=200)
print(f"F.linear:              {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# torch.compile with default mode
@torch.compile(mode="default")
def mm_default(x, y):
    return torch.mm(x, y)

# Warmup compile
for _ in range(5):
    mm_default(a, b)
torch.cuda.synchronize()
t = do_bench(lambda: mm_default(a, b), warmup=50, rep=200)
print(f"compile(default) mm:   {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# torch.compile with max-autotune
@torch.compile(mode="max-autotune")
def mm_autotune(x, y):
    return torch.mm(x, y)

for _ in range(5):
    mm_autotune(a, b)
torch.cuda.synchronize()
t = do_bench(lambda: mm_autotune(a, b), warmup=50, rep=200)
print(f"compile(max-autotune): {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# torch.compile with max-autotune on F.linear
@torch.compile(mode="max-autotune")
def linear_autotune(x, w):
    return torch.nn.functional.linear(x, w)

for _ in range(5):
    linear_autotune(a, w)
torch.cuda.synchronize()
t = do_bench(lambda: linear_autotune(a, w), warmup=50, rep=200)
print(f"compile linear:        {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Try with FP16 reduced precision
torch._C._set_cublas_allow_fp16_accumulation(True)
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
print(f"mm + fp16_accum:       {flops/(t*1e-3)/1e12:.1f} TFLOPS")

@torch.compile(mode="max-autotune")
def mm_fp16_accum(x, y):
    return torch.mm(x, y)

for _ in range(5):
    mm_fp16_accum(a, b)
torch.cuda.synchronize()
t = do_bench(lambda: mm_fp16_accum(a, b), warmup=50, rep=200)
print(f"compile + fp16_accum:  {flops/(t*1e-3)/1e12:.1f} TFLOPS")
torch._C._set_cublas_allow_fp16_accumulation(False)

# Try different linalg libraries
for lib in ['default', 'cusolver', 'cublaslt']:
    try:
        torch.backends.cuda.preferred_linalg_library(lib)
        t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
        print(f"linalg={lib:10s}:      {flops/(t*1e-3)/1e12:.1f} TFLOPS")
    except Exception as e:
        print(f"linalg={lib:10s}:      FAIL ({str(e)[:50]})")
