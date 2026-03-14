#include <torch/extension.h>
#include <cuda_fp16.h>

// v3: Vectorized FP16 loads (half2) + pre-allocated output support
// Each thread processes one 16-element block using 8x half2 loads
__global__ void quantize_fp16_to_nvfp4_v3_kernel(
    const half2* __restrict__ input,    // [M, K/2] as half2
    uint8_t* __restrict__ output,        // [M, K/2] packed
    uint8_t* __restrict__ scales_out,    // [padded_M * padded_nb] as fp8
    int M, int K,
    int n_blocks, int padded_nb
) {
    int row = blockIdx.x;
    int block_idx = threadIdx.x;

    if (row >= M || block_idx >= n_blocks) return;

    // Load 8x half2 = 16 FP16 values via vectorized loads
    const half2* in_ptr = input + row * (K / 2) + block_idx * 8;
    float vals[16];
    float max_abs = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        half2 h2 = in_ptr[i];
        vals[i * 2]     = __half2float(h2.x);
        vals[i * 2 + 1] = __half2float(h2.y);
        float a0 = fabsf(vals[i * 2]);
        float a1 = fabsf(vals[i * 2 + 1]);
        if (a0 > max_abs) max_abs = a0;
        if (a1 > max_abs) max_abs = a1;
    }

    float scale = fmaxf(max_abs / 6.0f, 1e-12f);
    float inv_scale = 1.0f / scale;

    // Store scale as FP8 e4m3 via manual bit conversion
    // FP32 -> FP16 -> FP8 e4m3 (s1 e4 m3, bias=7)
    __half sh = __float2half(scale);
    unsigned short hbits = *reinterpret_cast<unsigned short*>(&sh);
    // FP16: s(1) e(5) m(10), bias=15
    // FP8 e4m3: s(1) e(4) m(3), bias=7
    int exp16 = (hbits >> 10) & 0x1F;
    int exp8 = exp16 - 15 + 7;  // rebias
    uint8_t fp8;
    if (exp8 <= 0) {
        fp8 = 0;  // underflow to zero
    } else if (exp8 >= 15) {
        fp8 = 0x7E;  // max normal (0 1111 110 = 448.0)
    } else {
        fp8 = ((uint8_t)exp8 << 3) | ((hbits >> 7) & 0x7);
    }
    // Scale is always positive, no sign bit needed
    scales_out[row * padded_nb + block_idx] = fp8;

    // Quantize and pack
    uint8_t* out_ptr = output + row * (K / 2) + block_idx * 8;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v0 = vals[i * 2] * inv_scale;
        float v1 = vals[i * 2 + 1] * inv_scale;

        float a0 = fabsf(v0);
        uint8_t c0 = (a0 >= 0.25f) + (a0 >= 0.75f) + (a0 >= 1.25f) + (a0 >= 1.75f) + (a0 >= 2.50f) + (a0 >= 3.50f) + (a0 >= 5.00f);
        if (v0 < 0) c0 |= 8;

        float a1 = fabsf(v1);
        uint8_t c1 = (a1 >= 0.25f) + (a1 >= 0.75f) + (a1 >= 1.25f) + (a1 >= 1.75f) + (a1 >= 2.50f) + (a1 >= 3.50f) + (a1 >= 5.00f);
        if (v1 < 0) c1 |= 8;

        out_ptr[i] = c0 | (c1 << 4);
    }
}

std::vector<torch::Tensor> quantize_nvfp4_v3(torch::Tensor input, int padded_M, int padded_nb) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be FP16");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");

    int M = input.size(0);
    int K = input.size(1);
    TORCH_CHECK(K % 16 == 0, "K must be divisible by 16");

    auto output = torch::empty({M, K / 2}, input.options().dtype(torch::kUInt8));
    auto scales = torch::zeros({padded_M * padded_nb}, input.options().dtype(torch::kUInt8));

    int n_blocks = K / 16;
    dim3 grid(M);
    dim3 block(min(n_blocks, 1024));

    quantize_fp16_to_nvfp4_v3_kernel<<<grid, block>>>(
        reinterpret_cast<const half2*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(),
        M, K, n_blocks, padded_nb
    );

    return {output, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_nvfp4", &quantize_nvfp4_v3, "Fast NVFP4 quantization v3 (vectorized half2 + FP8 scales)");
}
