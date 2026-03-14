#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// Fused FP16 → NVFP4 quantization with padded FP8 scale output
// Each thread processes 16 elements (one scale block)
// Outputs: packed FP4 codes + padded FP8 scales ready for cuBLASLt
__global__ void quantize_fp16_to_nvfp4_v2_kernel(
    const half* __restrict__ input,         // [M, K]
    uint8_t* __restrict__ output,            // [M, K/2] packed
    __nv_fp8_e4m3* __restrict__ scales_out,  // [padded_M, padded_nb] flat
    int M, int K,
    int n_blocks, int padded_nb, int padded_M
) {
    int row = blockIdx.x;
    int block_idx = threadIdx.x;

    if (row >= M || block_idx >= n_blocks) return;

    const half* in_ptr = input + row * K + block_idx * 16;

    // Load 16 FP16 values and find max absolute
    float vals[16];
    float max_abs = 0.0f;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        vals[i] = __half2float(in_ptr[i]);
        float av = fabsf(vals[i]);
        if (av > max_abs) max_abs = av;
    }

    float scale = max_abs / 6.0f;
    if (scale < 1e-12f) scale = 1e-12f;
    float inv_scale = 1.0f / scale;

    // Store scale directly in padded FP8 layout
    scales_out[row * padded_nb + block_idx] = __nv_fp8_e4m3(scale);

    // Quantize and pack pairs
    uint8_t* out_ptr = output + row * (K / 2) + block_idx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v0 = vals[i * 2] * inv_scale;
        float v1 = vals[i * 2 + 1] * inv_scale;

        float a0 = fabsf(v0);
        uint8_t c0 = 0;
        if (a0 >= 0.25f) c0 = 1;
        if (a0 >= 0.75f) c0 = 2;
        if (a0 >= 1.25f) c0 = 3;
        if (a0 >= 1.75f) c0 = 4;
        if (a0 >= 2.50f) c0 = 5;
        if (a0 >= 3.50f) c0 = 6;
        if (a0 >= 5.00f) c0 = 7;
        if (v0 < 0) c0 |= 8;

        float a1 = fabsf(v1);
        uint8_t c1 = 0;
        if (a1 >= 0.25f) c1 = 1;
        if (a1 >= 0.75f) c1 = 2;
        if (a1 >= 1.25f) c1 = 3;
        if (a1 >= 1.75f) c1 = 4;
        if (a1 >= 2.50f) c1 = 5;
        if (a1 >= 3.50f) c1 = 6;
        if (a1 >= 5.00f) c1 = 7;
        if (v1 < 0) c1 |= 8;

        out_ptr[i] = c0 | (c1 << 4);
    }
}

// Python-callable wrapper - outputs pre-padded FP8 scales
std::vector<torch::Tensor> quantize_nvfp4_v2(torch::Tensor input, int padded_M, int padded_nb) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be FP16");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");

    auto output = torch::empty({M, K / 2}, input.options().dtype(torch::kUInt8));
    // Pre-allocate padded scale buffer as uint8 (will be viewed as float8)
    auto scales = torch::zeros({padded_M * padded_nb}, input.options().dtype(torch::kUInt8));

    int n_blocks = K / 16;
    dim3 grid(M);
    dim3 block(min(n_blocks, 1024));

    quantize_fp16_to_nvfp4_v2_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(scales.data_ptr<uint8_t>()),
        M, K, n_blocks, padded_nb, padded_M
    );

    return {output, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_nvfp4", &quantize_nvfp4_v2, "Fast NVFP4 quantization v2 with padded FP8 scales");
}
