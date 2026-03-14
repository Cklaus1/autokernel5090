"""Test cuBLAS with COMPUTE_16F (FP16 accumulation).
If cuBLAS supports this mode, it should give ~350+ TFLOPS on Blackwell."""

import torch
import ctypes
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
w = torch.randn(N, K, device='cuda', dtype=torch.float16)

# Approach 1: torch.backends setting
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
t = do_bench(lambda: torch.nn.functional.linear(a, w), warmup=50, rep=200)
print(f"allow_fp16_reduced_precision: {flops/(t*1e-3)/1e12:.1f} TFLOPS")

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
t = do_bench(lambda: torch.nn.functional.linear(a, w), warmup=50, rep=200)
print(f"standard:                     {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Approach 2: Try cublasGemmEx directly with COMPUTE_16F
try:
    libcublas = ctypes.cdll.LoadLibrary('libcublas.so')
    print("\nlibcublas loaded")

    # cublasComputeType_t values
    CUBLAS_COMPUTE_16F = 64   # 0x40
    CUBLAS_COMPUTE_32F = 68   # 0x44

    # Try to see if we can access cublasGemmEx
    print("cublasGemmEx available:", hasattr(libcublas, 'cublasGemmEx'))
except Exception as e:
    print(f"libcublas: {e}")

# Approach 3: Use torch._C._cublas directly
try:
    import torch._C
    cublas_fns = [x for x in dir(torch._C) if 'cublas' in x.lower() or 'gemm' in x.lower()]
    print(f"\ntorch._C GEMM functions: {cublas_fns}")
except Exception as e:
    print(f"torch._C: {e}")

# Approach 4: Check if torch has internal FP16 compute mode
try:
    # PyTorch 2.x has torch._linalg_utils or similar
    print(f"\nmatmul backend: {torch.backends.cuda.preferred_linalg_library()}")

    # Check CUDA math mode
    print(f"allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"allow_fp16_reduced: {torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
    print(f"allow_bf16_reduced: {torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}")
except Exception as e:
    print(f"settings: {e}")

# Approach 5: Use torch.matmul with explicit dtype control
# Some versions of PyTorch support compute_dtype argument
print("\n=== Testing different matmul paths ===")
b = w.t().contiguous()

# Standard mm
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
print(f"torch.mm:       {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# F.linear
t = do_bench(lambda: torch.nn.functional.linear(a, w), warmup=50, rep=200)
print(f"F.linear:       {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Check if there's a way to use half-precision compute via environment variables
import os
os.environ['CUBLAS_PEDANTIC_MATH'] = '0'

t = do_bench(lambda: torch.nn.functional.linear(a, w), warmup=50, rep=200)
print(f"PEDANTIC=0:     {flops/(t*1e-3)/1e12:.1f} TFLOPS")
