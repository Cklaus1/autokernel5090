"""Test cuBLAS COMPUTE_32F_FAST_16F via cublasLt."""

import torch
import ctypes
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
w = torch.randn(N, K, device='cuda', dtype=torch.float16)

ref = torch.mm(a, b)

# Approach: use PyTorch's internal API to set compute mode
print("=== PyTorch Internal cuBLAS APIs ===")

# Standard
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
out = torch.mm(a, b)
err = (out - ref).abs().max().item()
print(f"Standard torch.mm:           {flops/(t*1e-3)/1e12:.1f} TFLOPS, err={err:.6f}")

# FP16 accum
torch._C._set_cublas_allow_fp16_accumulation(True)
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
out = torch.mm(a, b)
err = (out - ref).abs().max().item()
print(f"FP16 accum torch.mm:         {flops/(t*1e-3)/1e12:.1f} TFLOPS, err={err:.6f}")
torch._C._set_cublas_allow_fp16_accumulation(False)

# Try allow_tf32
torch.backends.cuda.matmul.allow_tf32 = True
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
out = torch.mm(a, b)
err = (out - ref).abs().max().item()
print(f"allow_tf32 torch.mm:         {flops/(t*1e-3)/1e12:.1f} TFLOPS, err={err:.6f}")
torch.backends.cuda.matmul.allow_tf32 = False

# Try fp16_reduced_precision_reduction
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
out = torch.mm(a, b)
err = (out - ref).abs().max().item()
print(f"fp16_reduced_precision:      {flops/(t*1e-3)/1e12:.1f} TFLOPS, err={err:.6f}")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# Try BOTH fp16_accum + fp16_reduced
torch._C._set_cublas_allow_fp16_accumulation(True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
t = do_bench(lambda: torch.mm(a, b), warmup=50, rep=200)
out = torch.mm(a, b)
err = (out - ref).abs().max().item()
print(f"FP16 accum + reduced:        {flops/(t*1e-3)/1e12:.1f} TFLOPS, err={err:.6f}")
torch._C._set_cublas_allow_fp16_accumulation(False)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

# Try F.linear variants
print("\n=== F.linear ===")
t = do_bench(lambda: torch.nn.functional.linear(a, w), warmup=50, rep=200)
print(f"F.linear standard:           {flops/(t*1e-3)/1e12:.1f} TFLOPS")

torch._C._set_cublas_allow_fp16_accumulation(True)
t = do_bench(lambda: torch.nn.functional.linear(a, w), warmup=50, rep=200)
out_fl = torch.nn.functional.linear(a, w)
err = (out_fl - torch.mm(a, w.t())).abs().max().item()
print(f"F.linear FP16 accum:         {flops/(t*1e-3)/1e12:.1f} TFLOPS, err={err:.6f}")
torch._C._set_cublas_allow_fp16_accumulation(False)

# Check what Triton actually generates
print("\n=== Triton SASS/PTX Analysis ===")
try:
    import triton
    import triton.language as tl

    @triton.jit
    def simple_dot(A, B, C, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        a_ptr = tl.make_block_ptr(A, (M, K), (stride_am, stride_ak), (pid_m*BM, 0), (BM, BK), (1, 0))
        b_ptr = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn), (0, pid_n*BN), (BK, BN), (1, 0))
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):
            at = tl.load(a_ptr, boundary_check=(0, 1))
            bt = tl.load(b_ptr, boundary_check=(0, 1))
            partial = tl.dot(at, bt, out_dtype=tl.float16)
            acc += partial.to(tl.float32)
            a_ptr = tl.advance(a_ptr, (0, BK))
            b_ptr = tl.advance(b_ptr, (BK, 0))
        c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
        tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

    c = torch.empty(M, N, device='cuda', dtype=torch.float16)
    # Compile and get info
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    compiled = simple_dot[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1), BM=128, BN=128, BK=32,
        num_warps=8, num_stages=3)

    # Try to get the compiled kernel's asm
    print(f"Kernel compiled successfully")

    # Check for mma instructions in Triton cache
    import glob
    cache_dirs = glob.glob('/root/.triton/cache/**/*.ptx', recursive=True)
    if cache_dirs:
        latest = max(cache_dirs, key=lambda f: __import__('os').path.getmtime(f))
        with open(latest) as f:
            ptx = f.read()
        # Find mma instructions
        mma_lines = [l.strip() for l in ptx.split('\n') if 'mma' in l.lower() and not l.strip().startswith('//')]
        if mma_lines:
            print(f"MMA instructions found ({len(mma_lines)}):")
            for l in mma_lines[:5]:
                print(f"  {l}")
        else:
            print("No MMA instructions found in PTX")
    else:
        print("No PTX files in cache")

except Exception as e:
    print(f"Triton analysis failed: {e}")
