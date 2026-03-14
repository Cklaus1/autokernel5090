"""Check which autotune config is selected and if forcing a specific one helps."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import sys
sys.path.insert(0, '.')
from bench import gen_quantized_matmul_w4a16_inputs
from kernel import kernel_fn, matmul_fp16_dot, _dequant_cache

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

inputs = gen_quantized_matmul_w4a16_inputs(
    {"M": M, "N": N, "K": K, "group_size": 128},
    torch.float16, "cuda", seed=42
)

# Warm up to trigger autotune
for _ in range(5):
    kernel_fn(**inputs)
torch.cuda.synchronize()

# Check selected config
activation = inputs['activation']
cache_key = (id(inputs['packed_weights']), id(inputs['scales']), id(inputs['zeros']), K, N, torch.float16)
W = _dequant_cache[cache_key]
output = torch.empty(M, N, device='cuda', dtype=torch.float16)

# Try to get the best config from autotune
try:
    best = matmul_fp16_dot.best_config
    print(f"Autotune selected: {best}")
except:
    print("Cannot read best_config directly")

# Try all configs with controlled warm-up
configs = [
    (256, 128, 32, 8, 8, 3),
    (256, 128, 32, 8, 8, 4),
    (128, 256, 32, 8, 8, 3),
    (128, 256, 32, 8, 8, 4),
    (128, 128, 32, 8, 8, 4),
    (128, 128, 64, 8, 8, 3),
]

# Define a non-autotuned kernel
@triton.jit
def matmul_fixed(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group_id = pid // (num_m * G)
    first_n = group_id * G
    gsn = min(num_n - first_n, G)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)
    a_ptr = tl.make_block_ptr(A, (M, K), (stride_am, stride_ak), (pid_m*BM, 0), (BM, BK), (1, 0))
    b_ptr = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn), (0, pid_n*BN), (BK, BN), (1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr, boundary_check=(0, 1))
        b = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(a, b, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

print(f"\n{'Config':45s} {'TFLOPS':>8s}")
print("-" * 55)

# Run each config 5 times and take the best
for BM, BN, BK, G, warps, stages in configs:
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    # Heavy warmup
    for _ in range(10):
        matmul_fixed[grid](activation, W, output, M, N, K,
            activation.stride(0), activation.stride(1),
            W.stride(0), W.stride(1),
            output.stride(0), output.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=warps, num_stages=stages)
    torch.cuda.synchronize()

    best_t = float('inf')
    for trial in range(5):
        t = do_bench(lambda: matmul_fixed[grid](activation, W, output, M, N, K,
            activation.stride(0), activation.stride(1),
            W.stride(0), W.stride(1),
            output.stride(0), output.stride(1),
            BM=BM, BN=BN, BK=BK, G=G, num_warps=warps, num_stages=stages),
            warmup=50, rep=200)
        if t < best_t:
            best_t = t
    tflops = flops / (best_t * 1e-3) / 1e12
    label = f"BM={BM} BN={BN} BK={BK} G={G} w={warps} s={stages}"
    print(f"{label:45s} {tflops:8.1f}")
