"""Split-K cuBLAS FP16 accumulation: find minimum splits for correctness.
Each chunk does cuBLAS FP16 accum (fast), partial sums accumulated in FP32.
Goal: find the sweet spot where error < 0.05 AND speed > 290 TFLOPS."""

import torch
import torch._C
from triton.testing import do_bench
import sys
sys.path.insert(0, '.')
from bench import gen_quantized_matmul_w4a16_inputs, _ref_quantized_matmul_w4a16

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

# Use actual W4A16 data for accuracy testing
inputs = gen_quantized_matmul_w4a16_inputs(
    {"M": M, "N": N, "K": K, "group_size": 128},
    torch.float16, "cuda", seed=42
)
expected = _ref_quantized_matmul_w4a16(inputs)

# Get dequanted weights
from kernel import kernel_fn, _dequant_cache
out = kernel_fn(**inputs)
cache_key = (id(inputs['packed_weights']), id(inputs['scales']), id(inputs['zeros']), K, N, torch.float16)
W_kn = _dequant_cache[cache_key]  # K x N
W_nk = W_kn.t().contiguous()  # N x K for F.linear
activation = inputs['activation']

print(f"Expected max: {expected.abs().max().item():.1f}")
print(f"Triton error: {(out - expected).abs().max().item():.6f}")
print()

# Test split-K with cuBLAS FP16 accumulation
torch._C._set_cublas_allow_fp16_accumulation(True)

print(f"{'Splits':>6s} {'K_chunk':>8s} {'TFLOPS':>8s} {'max_err':>8s} {'PASS?':>6s}")
print("-" * 45)

for num_splits in [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160]:
    k_chunk = K // num_splits
    if k_chunk * num_splits != K:
        continue

    def splitk_matmul():
        result = torch.zeros(M, N, device='cuda', dtype=torch.float32)
        for i in range(num_splits):
            a_chunk = activation[:, i*k_chunk:(i+1)*k_chunk].contiguous()
            w_chunk = W_nk[:, i*k_chunk:(i+1)*k_chunk].contiguous()
            result += torch.nn.functional.linear(a_chunk, w_chunk).float()
        return result.half()

    # Check accuracy
    result = splitk_matmul()
    max_err = (result - expected).abs().max().item()
    passes = max_err < 0.05

    # Benchmark
    t = do_bench(splitk_matmul, warmup=25, rep=100)
    tflops = flops / (t * 1e-3) / 1e12

    print(f"{num_splits:6d} {k_chunk:8d} {tflops:8.1f} {max_err:8.4f} {'YES' if passes else 'NO':>6s}")

torch._C._set_cublas_allow_fp16_accumulation(False)

# Also test: single cuBLAS call with FP32 accumulation for reference
print(f"\ncuBLAS FP32 reference:")
t = do_bench(lambda: torch.nn.functional.linear(activation, W_nk), warmup=25, rep=100)
max_err = (torch.nn.functional.linear(activation, W_nk) - expected).abs().max().item()
print(f"{'1':>6s} {'5120':>8s} {flops/(t*1e-3)/1e12:8.1f} {max_err:8.4f}")
