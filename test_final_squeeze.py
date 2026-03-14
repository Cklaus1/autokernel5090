"""Final squeeze: try more autotune configs to push past 293 TFLOPS."""

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import sys
sys.path.insert(0, '.')
from bench import gen_quantized_matmul_w4a16_inputs, _ref_quantized_matmul_w4a16

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

# Generate inputs once
inputs = gen_quantized_matmul_w4a16_inputs(
    {"M": M, "N": N, "K": K, "group_size": 128},
    torch.float16, "cuda", seed=42
)

# Import kernel components
from kernel import _dequant_cache, matmul_fp16_dot

# Warm up dequant cache
from kernel import kernel_fn
for _ in range(3):
    kernel_fn(**inputs)
torch.cuda.synchronize()

activation = inputs['activation']
cache_key = (id(inputs['packed_weights']), id(inputs['scales']), id(inputs['zeros']), K, N, torch.float16)
W = _dequant_cache[cache_key]

ref = _ref_quantized_matmul_w4a16(inputs)

# Test many configs manually
configs = [
    # (BM, BN, BK, G, stages, warps)
    (256, 128, 32, 8, 3, 8),   # current best
    (256, 128, 32, 8, 4, 8),
    (256, 128, 32, 8, 2, 8),
    (256, 128, 32, 8, 5, 8),
    (256, 128, 64, 8, 3, 8),
    (256, 128, 64, 8, 4, 8),
    (256, 128, 64, 8, 2, 8),
    (128, 256, 32, 8, 3, 8),
    (128, 256, 32, 8, 4, 8),
    (128, 256, 64, 8, 3, 8),
    (128, 128, 64, 8, 3, 8),
    (128, 128, 64, 8, 4, 8),
    (128, 128, 32, 8, 4, 8),
    (256, 128, 32, 4, 3, 8),   # G=4
    (256, 128, 32, 16, 3, 8),  # G=16
    (256, 128, 32, 8, 3, 4),   # 4 warps
    (256, 128, 32, 8, 3, 16),  # 16 warps
]

best_tflops = 0
best_cfg = None

for BM, BN, BK, G, stages, warps in configs:
    try:
        output = torch.empty(M, N, device='cuda', dtype=torch.float16)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

        # Warmup - use all positional args to avoid name collision with 'B' param
        sa0, sa1 = activation.stride(0), activation.stride(1)
        sw0, sw1 = W.stride(0), W.stride(1)
        so0, so1 = output.stride(0), output.stride(1)

        def run():
            matmul_fp16_dot[grid](
                activation, W, output, M, N, K,
                sa0, sa1, sw0, sw1, so0, so1,
                BM, BN, BK, G, True,
                num_warps=warps, num_stages=stages,
            )

        for _ in range(3):
            run()
        torch.cuda.synchronize()

        err = (output - ref).abs().max().item()
        if err > 0.05:
            print(f"BM={BM:3d} BN={BN:3d} BK={BK:2d} G={G:2d} s={stages} w={warps:2d}: FAIL err={err:.4f}")
            continue

        t = do_bench(run, warmup=50, rep=200)

        tflops = flops / (t * 1e-3) / 1e12
        marker = " ***" if tflops > best_tflops else ""
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:2d} G={G:2d} s={stages} w={warps:2d}: {tflops:.1f} TFLOPS err={err:.4f}{marker}")

        if tflops > best_tflops:
            best_tflops = tflops
            best_cfg = (BM, BN, BK, G, stages, warps)
    except Exception as e:
        print(f"BM={BM:3d} BN={BN:3d} BK={BK:2d} G={G:2d} s={stages} w={warps:2d}: FAIL {str(e)[:80]}")

print(f"\nBest: {best_tflops:.1f} TFLOPS with {best_cfg}")
