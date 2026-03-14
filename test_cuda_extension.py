"""Test custom CUDA extension with hand-tuned HMMA FP16 matmul.
Uses two-stage accumulation: FP16 MMA with periodic FP32 flush."""

import torch
from torch.utils.cpp_extension import load_inline
from triton.testing import do_bench

M, K, N = 2048, 5120, 5120
flops = 2 * M * N * K

cuda_source = """
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Tile sizes matching our Triton kernel
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 32
#define WARP_M 32
#define WARP_N 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_fp16_accum_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K)
{
    // Use WMMA API for FP16 matrix multiply
    // Each warp handles a 16x16 output tile

    int warpId = threadIdx.x / 32;
    int numWarps = blockDim.x / 32;

    // Block-level tile
    int blockRow = blockIdx.x * BLOCK_M;
    int blockCol = blockIdx.y * BLOCK_N;

    // Warp-level tile within block
    int warpsPerRow = BLOCK_N / WMMA_N;
    int warpRow = (warpId / warpsPerRow) * WMMA_M;
    int warpCol = (warpId % warpsPerRow) * WMMA_N;

    int globalRow = blockRow + warpRow;
    int globalCol = blockCol + warpCol;

    if (globalRow >= M || globalCol >= N) return;

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;

    // FP16 accumulator for fast accumulation
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_fp16;
    // FP32 accumulator for periodic flush
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_fp32;

    wmma::fill_fragment(acc_fp32, 0.0f);

    // Two-stage accumulation: accumulate FLUSH_INTERVAL steps in FP16, then flush to FP32
    const int FLUSH_INTERVAL = 4;  // Flush every 4 iterations
    int num_k_tiles = K / WMMA_K;

    for (int k_outer = 0; k_outer < num_k_tiles; k_outer += FLUSH_INTERVAL) {
        wmma::fill_fragment(acc_fp16, __float2half(0.0f));

        int k_end = min(k_outer + FLUSH_INTERVAL, num_k_tiles);
        for (int k = k_outer; k < k_end; k++) {
            int kk = k * WMMA_K;

            // Load A tile (row-major)
            wmma::load_matrix_sync(a_frag, A + globalRow * K + kk, K);
            // Load B tile (row-major)
            wmma::load_matrix_sync(b_frag, B + kk * N + globalCol, N);

            // FP16 MMA with FP16 accumulation (full speed!)
            wmma::mma_sync(acc_fp16, a_frag, b_frag, acc_fp16);
        }

        // Flush FP16 accumulator to FP32
        for (int i = 0; i < acc_fp16.num_elements; i++) {
            acc_fp32.x[i] += __half2float(acc_fp16.x[i]);
        }
    }

    // Store result as FP16
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> out_frag;
    for (int i = 0; i < out_frag.num_elements; i++) {
        out_frag.x[i] = __float2half(acc_fp32.x[i]);
    }

    wmma::store_matrix_sync(C + globalRow * N + globalCol, out_frag, N, wmma::mem_row_major);
}

torch::Tensor matmul_fp16_twostage(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    // Launch config: each block handles BLOCK_M x BLOCK_N output
    // Each warp handles WMMA_M x WMMA_N = 16x16
    // Warps per block = (BLOCK_M/WMMA_M) * (BLOCK_N/WMMA_N) = 8 * 8 = 64
    // But max warps = 32 (1024 threads). Use smaller block.
    int warpsM = BLOCK_M / WMMA_M;  // 8
    int warpsN = BLOCK_N / WMMA_N;  // 8
    int totalWarps = warpsM * warpsN;  // 64 -- too many!

    // Reduce to 4 warps per block, smaller tiles
    dim3 block(128);  // 4 warps
    int tileMPerBlock = 2 * WMMA_M;  // 32
    int tileNPerBlock = 2 * WMMA_N;  // 32
    dim3 grid((M + tileMPerBlock - 1) / tileMPerBlock,
              (N + tileNPerBlock - 1) / tileNPerBlock);

    matmul_fp16_accum_kernel<<<grid, block>>>(
        (const half*)A.data_ptr(),
        (const half*)B.data_ptr(),
        (half*)C.data_ptr(),
        M, N, K
    );

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_fp16_twostage(torch::Tensor A, torch::Tensor B);
"""

print("Compiling CUDA extension...")
try:
    module = load_inline(
        name='matmul_fp16',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['matmul_fp16_twostage'],
        extra_cuda_cflags=['-arch=sm_120', '-O3'],
        verbose=False,
    )

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(3):
        c = module.matmul_fp16_twostage(a, b)
    torch.cuda.synchronize()

    # Benchmark
    t = do_bench(lambda: module.matmul_fp16_twostage(a, b), warmup=25, rep=100)
    tflops = flops / (t * 1e-3) / 1e12

    # Accuracy
    ref = torch.mm(a, b)
    max_err = (c - ref).abs().max().item()

    print(f"CUDA two-stage FP16: {tflops:.1f} TFLOPS, max_err={max_err:.4f}")

except Exception as e:
    print(f"CUDA extension failed: {type(e).__name__}: {e}")
