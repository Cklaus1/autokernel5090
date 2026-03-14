import torch, sys
sys.path.insert(0, '/root/projects/autokernel')
import triton, triton.language as tl
from triton.testing import do_bench

torch.manual_seed(42)
M, N, K = 2048, 5120, 5120
flops = 2 * M * N * K

def _pack_int4_weights(K, N, device):
    w = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    packed = torch.zeros(K//8, N, device=device, dtype=torch.int32)
    for i in range(8):
        packed |= (w[i::8] & 0xF) << (i * 4)
    return packed

activation = torch.randn(M, K, device='cuda', dtype=torch.float16)
packed_weights = _pack_int4_weights(K, N, 'cuda')
scales = torch.randn(K//128, N, device='cuda', dtype=torch.float16).abs() * 0.01 + 0.001
zeros = torch.randint(0, 16, (K//128, N), device='cuda').to(torch.float16)

import kernel
kernel.kernel_fn(activation, packed_weights, scales, zeros, 128)
W = kernel._dequant_cache[list(kernel._dequant_cache.keys())[0]]
ref = torch.mm(activation, W)

@triton.jit
def matmul_ieee(A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM); num_n = tl.cdiv(N, BN)
    group_id = pid // (num_m * G); first_n = group_id * G
    gsn = min(num_n - first_n, G)
    pid_m = (pid % (num_m * gsn)) // gsn; pid_n = first_n + (pid % gsn)
    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m*BM, 0), block_shape=(BM, BK), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n*BN), block_shape=(BK, BN), order=(1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(aa, bb, out_dtype=tl.float16, input_precision="ieee")
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK)); b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m*BM, pid_n*BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
o = torch.empty(M, N, device='cuda', dtype=torch.float16)
def fn():
    matmul_ieee[grid](activation, W, o, M, N, K,
        activation.stride(0), activation.stride(1), W.stride(0), W.stride(1),
        o.stride(0), o.stride(1), BM=128, BN=256, BK=64, G=32, num_stages=3, num_warps=8)
fn(); torch.cuda.synchronize()
err = (o - ref).abs().max().item()
correct = torch.allclose(o.float(), ref.float(), atol=0.05, rtol=0.05)

# Run 5 times
for i in range(5):
    ms = do_bench(fn, warmup=25, rep=100)
    tflops = flops / ms / 1e9
    print(f"Run {i}: {ms*1000:.0f} us = {tflops:.1f} TFLOPS  err={err:.4f}  correct={correct}")
