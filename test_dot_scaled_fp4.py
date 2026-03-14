"""Test tl.dot_scaled with E2M1 (NVFP4) format on SM120."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import numpy as np

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

@triton.jit
def matmul_dot_scaled_fp4(
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

    a_ptr = tl.make_block_ptr(A, (M, K), (stride_am, stride_ak), (pid_m*BM, 0), (BM, BK), (1, 0))
    b_ptr = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn), (0, pid_n*BN), (BK, BN), (1, 0))

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

        acc = tl.dot_scaled(a_data, a_sc, "e2m1",
                           b_data, b_sc, "e2m1",
                           acc=acc)

        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
        a_scale_ptr = tl.advance(a_scale_ptr, (0, BK // GROUP_SIZE))
        b_scale_ptr = tl.advance(b_scale_ptr, (0, BK // GROUP_SIZE))

    c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# Helper: convert to FP4 E2M1 with E8M0 block scaling
def to_fp4_scaled(tensor, group_size=32):
    """Convert FP16/FP32 tensor to simulated FP4 E2M1 with E8M0 block scaling."""
    rows, cols = tensor.shape
    assert cols % group_size == 0

    grouped = tensor.reshape(rows, cols // group_size, group_size)
    group_max = grouped.abs().amax(dim=-1)

    # E2M1 max value = 6.0
    fp4_max = 6.0
    log2_max = torch.log2(group_max.clamp(min=1e-12))
    exponent = torch.ceil(log2_max - np.log2(fp4_max)).clamp(-127, 127).to(torch.int32)
    scale_float = (2.0 ** exponent.float())

    # Scale and quantize to FP4 range (stored as uint8)
    scaled_vals = grouped / scale_float.unsqueeze(-1)
    # Clamp to FP4 E2M1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    quantized = scaled_vals.clamp(-fp4_max, fp4_max).to(torch.uint8).reshape(rows, cols)

    e8m0_scale = (exponent + 127).clamp(0, 254).to(torch.uint8)

    return quantized, e8m0_scale


# Prepare data
a_fp16 = torch.randn(M, K, device='cuda', dtype=torch.float16)
b_fp16 = torch.randn(K, N, device='cuda', dtype=torch.float16)
ref = torch.mm(a_fp16, b_fp16)

GROUP_SIZE = 32
BM, BN, BK, G = 128, 128, 128, 8

a_fp4, a_scale = to_fp4_scaled(a_fp16.float(), group_size=GROUP_SIZE)
b_fp4, b_scale = to_fp4_scaled(b_fp16.float(), group_size=GROUP_SIZE)
b_scale_t = b_scale.reshape(K // GROUP_SIZE, N).t().contiguous()

print(f"a_fp4: {a_fp4.shape} {a_fp4.dtype}")
print(f"a_scale: {a_scale.shape} {a_scale.dtype}")

grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)

try:
    matmul_dot_scaled_fp4[grid](
        a_fp4, a_scale, b_fp4, b_scale_t, c,
        M, N, K,
        a_fp4.stride(0), a_fp4.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_fp4.stride(0), b_fp4.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    )
    torch.cuda.synchronize()
    max_err = (c - ref).abs().max().item()
    t = do_bench(lambda: matmul_dot_scaled_fp4[grid](
        a_fp4, a_scale, b_fp4, b_scale_t, c,
        M, N, K,
        a_fp4.stride(0), a_fp4.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_fp4.stride(0), b_fp4.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    ), warmup=25, rep=100)
    print(f"FP4×FP4 dot_scaled: {flops/(t*1e-3)/1e12:.1f} TFLOPS, max_err={max_err:.4f}")
except Exception as e:
    print(f"FP4×FP4 FAIL: {type(e).__name__}: {str(e)[:300]}")


# Also test: what about FP16 × FP4 mixed?
try:
    @triton.jit
    def matmul_dot_scaled_fp16_fp4(
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

            acc = tl.dot_scaled(a_data, None, "fp16",
                               b_data, b_sc, "e2m1",
                               acc=acc)

            a_ptr = tl.advance(a_ptr, (0, BK))
            b_ptr = tl.advance(b_ptr, (BK, 0))
            b_scale_ptr = tl.advance(b_scale_ptr, (0, BK // GROUP_SIZE))

        c_ptr = tl.make_block_ptr(C, (M, N), (stride_cm, stride_cn), (pid_m*BM, pid_n*BN), (BM, BN), (1, 0))
        tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

    matmul_dot_scaled_fp16_fp4[grid](
        a_fp16, b_fp4, b_scale_t, c,
        M, N, K,
        a_fp16.stride(0), a_fp16.stride(1),
        b_fp4.stride(0), b_fp4.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    )
    torch.cuda.synchronize()
    max_err = (c - ref).abs().max().item()
    t = do_bench(lambda: matmul_dot_scaled_fp16_fp4[grid](
        a_fp16, b_fp4, b_scale_t, c,
        M, N, K,
        a_fp16.stride(0), a_fp16.stride(1),
        b_fp4.stride(0), b_fp4.stride(1),
        b_scale_t.stride(0), b_scale_t.stride(1),
        c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, G=G, GROUP_SIZE=GROUP_SIZE,
        num_warps=8, num_stages=3,
    ), warmup=25, rep=100)
    print(f"FP16×FP4 dot_scaled: {flops/(t*1e-3)/1e12:.1f} TFLOPS, max_err={max_err:.4f}")
except Exception as e:
    print(f"FP16×FP4 FAIL: {type(e).__name__}: {str(e)[:300]}")
