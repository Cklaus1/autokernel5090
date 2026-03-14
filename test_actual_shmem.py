"""Check compiled kernel shared memory via Triton compile API."""
import torch
import triton, triton.language as tl
import triton.compiler as tc

@triton.jit
def matmul_test(A, B, C, M, N, K,
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
    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BM, 0), block_shape=(BM, BK), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BN), block_shape=(BK, BN), order=(1, 0))
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptr, boundary_check=(0, 1))
        bb = tl.load(b_ptr, boundary_check=(0, 1))
        partial = tl.dot(aa, bb, out_dtype=tl.float16)
        acc += partial.to(tl.float32)
        a_ptr = tl.advance(a_ptr, (0, BK))
        b_ptr = tl.advance(b_ptr, (BK, 0))
    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BM, pid_n * BN), block_shape=(BM, BN), order=(1, 0))
    tl.store(c_ptr, acc.to(tl.float16), boundary_check=(0, 1))

target = triton.runtime.driver.active.get_current_target()
print(f"Target: {target}")

for stages in [1, 2, 3]:
    try:
        sig = {
            0: '*fp16', 1: '*fp16', 2: '*fp16',
            3: 'i32', 4: 'i32', 5: 'i32',
            6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32',
        }
        constants = {12: 128, 13: 256, 14: 64, 15: 32}
        compiled = tc.compile(matmul_test, signature=sig, constants=constants,
                              target=target, num_stages=stages, num_warps=8)
        md = compiled.metadata
        print(f"stages={stages}: shared={md.shared} bytes, num_stages={md.num_stages}")
    except Exception as e:
        print(f"stages={stages}: FAILED ({str(e)[:100]})")
