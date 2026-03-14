"""Test CUDA graphs to eliminate kernel launch overhead."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import sys
sys.path.insert(0, '.')
from bench import gen_quantized_matmul_w4a16_inputs, _ref_quantized_matmul_w4a16
from kernel import kernel_fn, _dequant_cache, matmul_fp16_dot

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

inputs = gen_quantized_matmul_w4a16_inputs(
    {"M": M, "N": N, "K": K, "group_size": 128},
    torch.float16, "cuda", seed=42
)

# Warmup kernel_fn to populate cache
for _ in range(5):
    kernel_fn(**inputs)
torch.cuda.synchronize()

# Get cached weights
activation = inputs['activation']
cache_key = (id(inputs['packed_weights']), id(inputs['scales']), id(inputs['zeros']), K, N, torch.float16)
W = _dequant_cache[cache_key]
output = torch.empty(M, N, device='cuda', dtype=torch.float16)

# Baseline: direct Triton call
t = do_bench(lambda: matmul_fp16_dot[(triton.cdiv(M, 256) * triton.cdiv(N, 128),)](
    activation, W, output, M, N, K,
    activation.stride(0), activation.stride(1),
    W.stride(0), W.stride(1),
    output.stride(0), output.stride(1),
    USE_FP16_DOT=True,
), warmup=50, rep=200)
print(f"Direct Triton:  {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# CUDA Graph capture
print("\nCapturing CUDA graph...")
# Warmup for graph capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        matmul_fp16_dot[(triton.cdiv(M, 256) * triton.cdiv(N, 128),)](
            activation, W, output, M, N, K,
            activation.stride(0), activation.stride(1),
            W.stride(0), W.stride(1),
            output.stride(0), output.stride(1),
            USE_FP16_DOT=True,
        )
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    matmul_fp16_dot[(triton.cdiv(M, 256) * triton.cdiv(N, 128),)](
        activation, W, output, M, N, K,
        activation.stride(0), activation.stride(1),
        W.stride(0), W.stride(1),
        output.stride(0), output.stride(1),
        USE_FP16_DOT=True,
    )

# Benchmark graph replay
t = do_bench(lambda: g.replay(), warmup=50, rep=200)
print(f"CUDA Graph:     {flops/(t*1e-3)/1e12:.1f} TFLOPS")

# Check correctness of graph output
expected = _ref_quantized_matmul_w4a16(inputs)
g.replay()
torch.cuda.synchronize()
max_err = (output - expected).abs().max().item()
print(f"Max error:      {max_err:.6f} (need < 0.05)")
