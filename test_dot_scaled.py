"""Test tl.dot_scaled for block-scaled MMA on SM120.
This could unlock NVFP4/FP8 tensor core instructions at 2-4x FP16 throughput."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import numpy as np

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

# Helper: convert FP16 tensor to FP8 E4M3 with E8M0 block scaling
def to_fp8_scaled(tensor, group_size=32):
    """Convert FP16/FP32 tensor to FP8 E4M3 with E8M0 block scaling."""
    rows, cols = tensor.shape
    assert cols % group_size == 0

    # Reshape to groups
    grouped = tensor.reshape(rows, cols // group_size, group_size)

    # Compute per-group scale: max absolute value per group
    group_max = grouped.abs().amax(dim=-1)  # [rows, cols//group_size]

    # E8M0 scale factor: power of 2 encoding
    # E8M0 can represent 2^(e-127) for e in [0, 254], plus special values
    # We compute log2(max) and clamp
    log2_max = torch.log2(group_max.clamp(min=1e-12))
    # E4M3 max value is 448, so scale = max_val / 448
    # But for E8M0, the scale IS the exponent: scale = 2^exponent
    # We want: val_fp8 * 2^exponent ≈ val_orig
    # So exponent = ceil(log2(max_val / fp8_max))
    fp8_max = 448.0  # max representable in E4M3
    exponent = torch.ceil(log2_max - np.log2(fp8_max)).clamp(-127, 127).to(torch.int32)
    scale_float = (2.0 ** exponent.float())  # actual scale

    # Scale down values and convert to FP8
    scaled_vals = grouped / scale_float.unsqueeze(-1)
    fp8_vals = scaled_vals.to(torch.float8_e4m3fn).reshape(rows, cols)

    # E8M0 encoding: biased exponent = exponent + 127
    e8m0_scale = (exponent + 127).clamp(0, 254).to(torch.uint8)

    return fp8_vals, e8m0_scale


# Test 1: Simple dot_scaled with FP8 inputs
print("=== Test 1: FP8 E4M3 block-scaled dot ===")

@triton.jit
def matmul_dot_scaled_fp8(
    A, A_scale, B, B_scale, C,
    M, N, K,
    stride_am, stride_ak,
    stride_ask, stride_asg,
    stride_bk, stride_bn,
    stride_bsk, stride_bsg,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group_id = pid // (num_m * G)
    first_n = group_id * G
    gsn = min(num_n - first_n, G)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)

    # Load block pointers for data
    a_ptr = tl.make_block_ptr(A, (M, K), (stride_am, stride_ak), (pid_m*BM, 0), (BM, BK), (1, 0))
    b_ptr = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn), (0, pid_n*BN), (BK, BN), (1, 0))

    # Scale pointers
    num_k_groups = K // GROUP_SIZE
    a_scale_ptr = tl.make_block_ptr(A_scale, (M, num_k_groups), (stride_ask, stride_asg),
                                     (pid_m*BM, 0), (BM, BK // GROUP_SIZE), (1, 0))
    b_scale_ptr = tl.make_block_ptr(B_scale, (N, num_k_groups), (stride_bsk, stride_bsg),
                                     (pid_n*BN, 0), (BN, BK // GROUP_SIZE), (1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BK)):
        a_data = tl.load(a_ptr, boundary_check=(0, 1))
        b_data = tl.load(b_ptr, boundary_check=(0, 1))
        a_sc = tl.load(a_scale_ptr, boundary_check=(0, 1))
        b_sc = tl.load(b_scale_ptr, boundary_check=(0, 1))

        acc = tl.dot_scaled(a_data, a_sc, "e4m3",
                           b_data, b_sc, "e4m3",
                           acc=acc)

        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
        a_scale_ptr = tl.advance(a_scale_ptr, (0, BK // GROUP_SIZE))
        b_scale_ptr = tl.advance(b_scale_ptr, (0, BK // GROUP_SIZE))

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Test 2: Mixed precision - FP16 activation × FP8 weight
print("\n=== Test 2: FP16 × FP8 mixed dot_scaled ===")

@triton.jit
def matmul_dot_scaled_mixed(
    A, B, B_scale, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bsk, stride_bsg,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
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

    num_k_groups = K // GROUP_SIZE
    b_scale_ptr = tl.make_block_ptr(B_scale, (N, num_k_groups), (stride_bsk, stride_bsg),
                                     (pid_n*BN, 0), (BN, BK // GROUP_SIZE), (1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BK)):
        a_data = tl.load(a_ptr, boundary_check=(0, 1))
        b_data = tl.load(b_ptr, boundary_check=(0, 1))
        b_sc = tl.load(b_scale_ptr, boundary_check=(0, 1))

        # FP16 activation (no scale), FP8 weight (with scale)
        acc = tl.dot_scaled(a_data, None, "fp16",
                           b_data, b_sc, "e4m3",
                           acc=acc)

        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
        b_scale_ptr = tl.advance(b_scale_ptr, (0, BK // GROUP_SIZE))

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Prepare data
a_fp16 = torch.randn(M, K, device='cuda', dtype=torch.float16)
b_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)
ref = torch.mm(a_fp16, b_fp16)

GROUP_SIZE = 32
BM, BN, BK, G = 128, 128, 128, 8

# Convert to FP8 with block scaling
a_fp8, a_scale = to_fp8_scaled(a_fp16.float(), group_size=GROUP_SIZE)
b_fp8, b_scale = to_fp8_scaled(b_fp16.float(), group_size=GROUP_SIZE)

# b_scale needs to be [N, K//GROUP_SIZE] for rhs
b_fp8_t = b_fp8  # K×N
b_scale_t = b_scale.reshape(K // GROUP_SIZE, N).t().contiguous()  # [N, K//GROUP_SIZE]

print(f"a_fp8: {a_fp8.shape} {a_fp8.dtype}")
print(f"a_scale: {a_scale.shape} {a_scale.dtype}")
print(f"b_fp8: {b_fp8_t.shape} {b_fp8_t.dtype}")
print(f"b_scale_t: {b_scale_t.shape} {b_scale_t.dtype}")

# Test FP8 × FP8
grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)

try:
    # Need to cast FP8 to uint8 for Triton
    a_u8 = a_fp8.view(torch.uint8)
    b_u8 = b_fp8_t.view(torch.uint8)

    matmul_dot_scaled_fp8[grid](
        a_u8, a_scale, b_u8, b_scale_t, c,
        M, N, K,
        a_u8.stride(0), a_u8.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_u8.stride(0), b_u8.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    )
    torch.cuda.synchronize()
    max_err = (c - ref).abs().max().item()
    t = do_bench(lambda: matmul_dot_scaled_fp8[grid](
        a_u8, a_scale, b_u8, b_scale_t, c,
        M, N, K,
        a_u8.stride(0), a_u8.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_u8.stride(0), b_u8.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    ), warmup=25, rep=100)
    print(f"FP8×FP8 dot_scaled: {flops/(t*1e-3)/1e12:.1f} TFLOPS, max_err={max_err:.4f}")
except Exception as e:
    print(f"FP8×FP8 FAIL: {type(e).__name__}: {str(e)[:200]}")

# Test FP16 × FP8 mixed
try:
    matmul_dot_scaled_mixed[grid](
        a_fp16, b_u8, b_scale_t, c,
        M, N, K,
        a_fp16.stride(0), a_fp16.stride(1),
        b_u8.stride(0), b_u8.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    )
    torch.cuda.synchronize()
    max_err = (c - ref).abs().max().item()
    t = do_bench(lambda: matmul_dot_scaled_mixed[grid](
        a_fp16, b_u8, b_scale_t, c,
        M, N, K,
        a_fp16.stride(0), a_fp16.stride(1),
        b_u8.stride(0), b_u8.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    ), warmup=25, rep=100)
    print(f"FP16×FP8 dot_scaled: {flops/(t*1e-3)/1e12:.1f} TFLOPS, max_err={max_err:.4f}")
except Exception as e:
    print(f"FP16×FP8 FAIL: {type(e).__name__}: {str(e)[:200]}")
