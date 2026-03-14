"""Test CUDA kernel via NVRTC (runtime compilation) for SM120.
Uses cuBLAS internally but wraps it with custom accumulation logic."""

import torch
import ctypes
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

# Instead of custom CUDA kernels, let's try a different approach:
# Use cuBLAS with custom math mode via ctypes

libcublas = ctypes.cdll.LoadLibrary('libcublas.so')
print(f"libcublas loaded: {libcublas}")

# Check available functions
print(f"cublasSetMathMode: {hasattr(libcublas, 'cublasSetMathMode')}")

# cuBLAS math modes
CUBLAS_DEFAULT_MATH = 0
CUBLAS_TENSOR_OP_MATH = 1
CUBLAS_PEDANTIC_MATH = 2
CUBLAS_TF32_TENSOR_OP_MATH = 3
CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION = 16

# cuBLAS compute types
CUBLAS_COMPUTE_16F = 64
CUBLAS_COMPUTE_16F_PEDANTIC = 65
CUBLAS_COMPUTE_32F = 68
CUBLAS_COMPUTE_32F_PEDANTIC = 69
CUBLAS_COMPUTE_32F_FAST_16F = 74
CUBLAS_COMPUTE_32F_FAST_16BF = 75
CUBLAS_COMPUTE_32F_FAST_TF32 = 77

# Try different cuBLAS math modes via PyTorch internal API
print("\n=== cuBLAS Math Mode Experiments ===")

# Get current handle
try:
    handle = torch.cuda.current_blas_handle()
    print(f"cuBLAS handle: {handle}")

    # Try COMPUTE_32F_FAST_16F - this should use FP16 tensor cores but FP32 accumulation
    # with fast (imprecise) reduction
    for mode_name, mode_val in [
        ("DEFAULT_MATH", CUBLAS_DEFAULT_MATH),
        ("TENSOR_OP_MATH", CUBLAS_TENSOR_OP_MATH),
        ("PEDANTIC_MATH", CUBLAS_PEDANTIC_MATH),
        ("TF32_TENSOR_OP_MATH", CUBLAS_TF32_TENSOR_OP_MATH),
    ]:
        try:
            ret = libcublas.cublasSetMathMode(ctypes.c_void_p(handle), ctypes.c_int(mode_val))
            if ret != 0:
                print(f"  {mode_name}: SetMathMode returned {ret}")
                continue
            t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
            tflops = flops / (t * 1e-3) / 1e12
            ref = torch.mm(a, b)
            # Reset to default for error comparison
            libcublas.cublasSetMathMode(ctypes.c_void_p(handle), ctypes.c_int(CUBLAS_DEFAULT_MATH))
            ref2 = torch.mm(a, b)
            err = (ref - ref2).abs().max().item()
            print(f"  {mode_name:25s}: {tflops:.1f} TFLOPS (diff from default: {err:.6f})")
        except Exception as e:
            print(f"  {mode_name}: FAIL ({e})")

    # Reset
    libcublas.cublasSetMathMode(ctypes.c_void_p(handle), ctypes.c_int(CUBLAS_DEFAULT_MATH))

except Exception as e:
    print(f"Handle error: {e}")

# Try cublasGemmEx directly with different compute types
print("\n=== cublasGemmEx with different compute types ===")
try:
    # CUDA data types
    CUDA_R_16F = 2
    CUDA_R_32F = 0

    # cublasOperation_t
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1

    # GemmAlgo
    CUBLAS_GEMM_DEFAULT = -1
    CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99

    handle = torch.cuda.current_blas_handle()

    for compute_name, compute_type in [
        ("COMPUTE_32F", CUBLAS_COMPUTE_32F),
        ("COMPUTE_32F_FAST_16F", CUBLAS_COMPUTE_32F_FAST_16F),
        ("COMPUTE_16F", CUBLAS_COMPUTE_16F),
        ("COMPUTE_32F_FAST_TF32", CUBLAS_COMPUTE_32F_FAST_TF32),
    ]:
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)

        # cublasGemmEx(handle, transa, transb, m, n, k,
        #              alpha, A, Atype, lda, B, Btype, ldb,
        #              beta, C, Ctype, ldc, computeType, algo)
        try:
            # Note: cuBLAS is column-major, so we compute C = B^T * A^T
            # to get row-major C = A * B
            ret = libcublas.cublasGemmEx(
                ctypes.c_void_p(handle),
                ctypes.c_int(CUBLAS_OP_T),  # transa
                ctypes.c_int(CUBLAS_OP_N),  # transb
                ctypes.c_int(N),   # m
                ctypes.c_int(M),   # n
                ctypes.c_int(K),   # k
                ctypes.byref(alpha),
                ctypes.c_void_p(b.data_ptr()),  # A (= B in row-major)
                ctypes.c_int(CUDA_R_16F),
                ctypes.c_int(K),   # lda
                ctypes.c_void_p(a.data_ptr()),  # B (= A in row-major)
                ctypes.c_int(CUDA_R_16F),
                ctypes.c_int(K),   # ldb
                ctypes.byref(beta),
                ctypes.c_void_p(c.data_ptr()),
                ctypes.c_int(CUDA_R_16F),
                ctypes.c_int(N),   # ldc
                ctypes.c_int(compute_type),
                ctypes.c_int(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            )
            torch.cuda.synchronize()

            if ret != 0:
                print(f"  {compute_name:25s}: cublasGemmEx returned {ret}")
                continue

            ref = torch.mm(a, b)
            max_err = (c - ref).abs().max().item()

            def run_gemm():
                libcublas.cublasGemmEx(
                    ctypes.c_void_p(handle),
                    ctypes.c_int(CUBLAS_OP_T),
                    ctypes.c_int(CUBLAS_OP_N),
                    ctypes.c_int(N), ctypes.c_int(M), ctypes.c_int(K),
                    ctypes.byref(alpha),
                    ctypes.c_void_p(b.data_ptr()), ctypes.c_int(CUDA_R_16F), ctypes.c_int(K),
                    ctypes.c_void_p(a.data_ptr()), ctypes.c_int(CUDA_R_16F), ctypes.c_int(K),
                    ctypes.byref(beta),
                    ctypes.c_void_p(c.data_ptr()), ctypes.c_int(CUDA_R_16F), ctypes.c_int(N),
                    ctypes.c_int(compute_type),
                    ctypes.c_int(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                )

            t = do_bench(run_gemm, warmup=50, rep=200)
            tflops = flops / (t * 1e-3) / 1e12
            print(f"  {compute_name:25s}: {tflops:.1f} TFLOPS, max_err={max_err:.4f}")

        except Exception as e:
            print(f"  {compute_name:25s}: FAIL ({str(e)[:60]})")

except Exception as e:
    print(f"GemmEx error: {e}")
