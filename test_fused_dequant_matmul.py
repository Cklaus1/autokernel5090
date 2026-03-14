"""Test fused dequant+matmul kernel - dequant on the fly in shared memory."""
import torch, sys
sys.path.insert(0, '/root/projects/autokernel')
import triton, triton.language as tl
from triton.testing import do_bench

torch.manual_seed(42)
M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K

def _pack_int4_weights(K, N, device):
    w = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    K_packed = K // 8
    packed = torch.zeros(K_packed, N, device=device, dtype=torch.int32)
    for i in range(8):
        packed |= (w[i::8] & 0xF) << (i * 4)
    return packed

activation = torch.randn(M, K, device='cuda', dtype=torch.float16)
packed_weights = _pack_int4_weights(K, N, 'cuda')
scales = torch.randn(K//128, N, device='cuda', dtype=torch.float16).abs() * 0.01 + 0.001
zeros = torch.randint(0, 16, (K//128, N), device='cuda').to(torch.float16)

# Reference via separate dequant
import kernel
kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
W = kernel._dequant_cache[list(kernel._dequant_cache.keys())[0]]
ref = torch.mm(activation, W)


@triton.autotune(
    configs=[
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_stages=2, num_warps=8),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 64}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 32}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 64}, num_stages=2, num_warps=4),
    ],
    key=['M_key', 'N_key', 'K_key'],
)
@triton.jit
def fused_dequant_matmul(
    A_ptr, QW_ptr, S_ptr, Z_ptr, O_ptr,
    M_key, N_key, K_key,
    stride_am, stride_ak,
    stride_qwk, stride_qwn,
    stride_skg, stride_sn,
    stride_zkg, stride_zn,
    stride_om, stride_on,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
):
    """Fused dequant + matmul: dequantize W blocks on-the-fly."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M_key, BM)
    num_n = tl.cdiv(N_key, BN)
    GRP: tl.constexpr = 32
    group_id = pid // (num_m * GRP)
    first_n = group_id * GRP
    gsn = min(num_n - first_n, GRP)
    pid_m = (pid % (num_m * gsn)) // gsn
    pid_n = first_n + (pid % gsn)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    m_mask = offs_m < M_key
    n_mask = offs_n < N_key

    a_ptr = tl.make_block_ptr(
        base=A_ptr, shape=(M_key, K_key), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K_key, BK)):
        # Load activation block
        aa = tl.load(a_ptr, boundary_check=(0, 1))

        # Dequantize weight block on the fly
        offs_k = k_start * BK + tl.arange(0, BK)
        k_mask = offs_k < K_key

        # Load packed int4 weights
        packed_k_idx = offs_k // 8
        bit_shift = ((offs_k & 7) * 4).to(tl.int32)
        qw_ptrs = QW_ptr + packed_k_idx[:, None] * stride_qwk + offs_n[None, :] * stride_qwn
        qw_packed = tl.load(qw_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
        int4_vals = (qw_packed >> bit_shift[:, None]) & 0xF

        # Load scales and zeros for this K block
        g = offs_k // QUANT_GROUP_SIZE
        s_ptrs = S_ptr + g[:, None] * stride_skg + offs_n[None, :] * stride_sn
        z_ptrs = Z_ptr + g[:, None] * stride_zkg + offs_n[None, :] * stride_zn
        scales = tl.load(s_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=1.0)
        zeros = tl.load(z_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Dequantize
        w_block = (int4_vals.to(tl.float16) - zeros) * scales

        # Matmul
        partial = tl.dot(aa, w_block, out_dtype=tl.float16)
        acc += partial.to(tl.float32)

        a_ptr = tl.advance(a_ptr, (0, BK))

    o_ptr = tl.make_block_ptr(
        base=O_ptr, shape=(M_key, N_key), strides=(stride_om, stride_on),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(o_ptr, acc.to(tl.float16), boundary_check=(0, 1))


o = torch.empty(M, N, device='cuda', dtype=torch.float16)
def grid(META):
    return (triton.cdiv(M, META['BM']) * triton.cdiv(N, META['BN']),)

def fn():
    fused_dequant_matmul[grid](
        activation, packed_weights, scales, zeros, o,
        M, N, K,
        activation.stride(0), activation.stride(1),
        packed_weights.stride(0), packed_weights.stride(1),
        scales.stride(0), scales.stride(1),
        zeros.stride(0), zeros.stride(1),
        o.stride(0), o.stride(1),
        QUANT_GROUP_SIZE=128,
    )

fn(); torch.cuda.synchronize()
err = (o - ref).abs().max().item()
ms = do_bench(fn, warmup=25, rep=100)
tflops = flops / ms / 1e9
print(f"Fused dequant+matmul: {ms*1000:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}")
print(f"Best config: {fused_dequant_matmul.best_config}")

# Compare with split approach
kernel._dequant_cache.clear()
def fn_split():
    kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)

# Warm up cache
fn_split(); torch.cuda.synchronize()
kernel._dequant_cache.clear()
ms_split = do_bench(fn_split, warmup=5, rep=50)
tflops_split = flops / ms_split / 1e9
print(f"Split dequant+matmul (uncached): {ms_split*1000:.0f} us = {tflops_split:.1f} TFLOPS")

# Cached
fn_split(); torch.cuda.synchronize()
ms_cached = do_bench(fn_split, warmup=25, rep=100)
tflops_cached = flops / ms_cached / 1e9
print(f"Split dequant+matmul (cached W): {ms_cached*1000:.0f} us = {tflops_cached:.1f} TFLOPS")
