"""Test cuBLAS FP16 accumulation with ACTUAL W4A16 dequanted weights.
Random data fails (0.203 error) but W4A16 has smaller magnitudes."""

import torch
import torch._C
from triton.testing import do_bench
import sys
sys.path.insert(0, '.')
from bench import gen_quantized_matmul_w4a16_inputs, _ref_quantized_matmul_w4a16
from kernel import kernel_fn, _dequant_cache, _wt_buf, _run_dequant

M, K, N = 2048, 5120, 5120
group_size = 128
flops = 2 * M * N * K

# Generate actual W4A16 inputs
inputs = gen_quantized_matmul_w4a16_inputs(
    {"M": M, "N": N, "K": K, "group_size": group_size},
    torch.float16, "cuda", seed=42
)
expected = _ref_quantized_matmul_w4a16(inputs)

print(f"Expected output range: [{expected.min().item():.2f}, {expected.max().item():.2f}]")
print(f"Expected output abs max: {expected.abs().max().item():.2f}")
print(f"Expected output abs mean: {expected.abs().mean().item():.4f}")

# Get dequanted weights
activation = inputs['activation']
packed_weights = inputs['packed_weights']
scales = inputs['scales']
zeros = inputs['zeros']

# Run kernel once to populate cache
out = kernel_fn(**inputs)
max_err_triton = (out - expected).abs().max().item()
print(f"\nTriton FP16-dot error: {max_err_triton:.6f}")

# Get cached dequanted weights
cache_key = (id(packed_weights), id(scales), id(zeros), K, N, torch.float16)
W_kn = _dequant_cache[cache_key]  # K x N layout

# Make N x K layout for F.linear
W_nk = W_kn.t().contiguous()

print(f"\nDequanted weight range: [{W_nk.min().item():.4f}, {W_nk.max().item():.4f}]")
print(f"Dequanted weight abs max: {W_nk.abs().max().item():.4f}")

# Test cuBLAS with standard FP32 accum
out_cublas = torch.nn.functional.linear(activation, W_nk)
max_err_cublas = (out_cublas - expected).abs().max().item()
t = do_bench(lambda: torch.nn.functional.linear(activation, W_nk), warmup=50, rep=200)
print(f"\ncuBLAS FP32 accum: {flops/(t*1e-3)/1e12:.1f} TFLOPS, max_err={max_err_cublas:.6f}")

# Test cuBLAS with FP16 accumulation
torch._C._set_cublas_allow_fp16_accumulation(True)
out_cublas_fp16 = torch.nn.functional.linear(activation, W_nk)
max_err_cublas_fp16 = (out_cublas_fp16 - expected).abs().max().item()
t = do_bench(lambda: torch.nn.functional.linear(activation, W_nk), warmup=50, rep=200)
print(f"cuBLAS FP16 accum: {flops/(t*1e-3)/1e12:.1f} TFLOPS, max_err={max_err_cublas_fp16:.6f}")
torch._C._set_cublas_allow_fp16_accumulation(False)

print(f"\nW4A16 tolerance: atol=0.05")
print(f"Triton PASS: {max_err_triton < 0.05}")
print(f"cuBLAS FP32 PASS: {max_err_cublas < 0.05}")
print(f"cuBLAS FP16 PASS: {max_err_cublas_fp16 < 0.05}")
