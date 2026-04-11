// SPDX-License-Identifier: Apache-2.0
//
// FusenCache Store KV — CUDA C++ kernel for CUDA graph capture.
//
// Translates the Triton _universal_store_kernel into a CUDA kernel that
// can be captured in CUDA graphs, bypassing Triton codegen issues on SM120.
//
// Algorithm (per token, per KV head):
//   Pass 1: Compute per-block scales for K and V (absmax / offset)
//   Pass 2: Quantize K and V using scales, pack sub-byte values, scatter to cache
//
// Supports: 4-bit K, 4-bit V (k4v4), 8-bit, and 2-bit quantization.
// The kernel is launched with grid = (num_tokens, num_kv_heads).

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace fusencache {

// ============================================================
// Helper: load BF16 or FP16 as float
// ============================================================

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr, int idx);

template <>
__device__ __forceinline__ float load_as_float(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_float(const __half* ptr, int idx) {
    return __half2float(ptr[idx]);
}

// ============================================================
// Store kernel: quantize + pack + scatter into paged KV cache
// ============================================================
//
// Grid: (num_tokens, num_kv_heads)
// Block: BLOCK_THREADS threads
//
// Each thread block processes one (token, kv_head) pair.
// Uses cooperative threads to compute per-block scales (max-reduction)
// and then quantize + pack.
//
// Template params:
//   SRC_T: __nv_bfloat16 or __half (input type)
//   K_BITS: 2, 4, or 8
//   V_BITS: 2, 4, or 8
//   BLOCK_THREADS: threads per block

template <typename SRC_T, int K_BITS, int V_BITS, int BLOCK_THREADS = 256>
__global__ void store_kv_kernel(
    const SRC_T* __restrict__ key,        // [N, Hk, D]
    const SRC_T* __restrict__ value,      // [N, Hk, D]
    uint8_t* __restrict__ kv_cache,       // [num_blocks, block_size, Hk, slot_bytes]
    __half* __restrict__ scales,          // [max_slots, Hk, num_scale_blocks, 2]
    const int32_t* __restrict__ slot_mapping, // [N]
    // Strides (in elements)
    int64_t stride_kv_n, int64_t stride_kv_h,
    int64_t stride_cache_block, int64_t stride_cache_pos, int64_t stride_cache_head,
    int64_t stride_sc_slot, int64_t stride_sc_head, int64_t stride_sc_block, int64_t stride_sc_kv,
    // Params
    int head_dim,
    int page_size,
    int k_scale_block,
    int v_scale_block,
    int scale_block_store,  // = min(k_scale_block, v_scale_block)
    float k_offset,         // e.g. 7.5 for 4-bit
    float v_offset,
    int k_region_bytes,     // = head_dim * K_BITS / 8
    int v_region_start      // = k_region_bytes
) {
    const int token_id = blockIdx.x;
    const int head_id = blockIdx.y;
    const int tid = threadIdx.x;

    // Check slot mapping -- skip padding tokens
    const int32_t slot = slot_mapping[token_id];
    if (slot < 0) return;

    const int blk_idx = slot / page_size;
    const int blk_off = slot % page_size;

    const int64_t cache_base = (int64_t)blk_idx * stride_cache_block
                             + (int64_t)blk_off * stride_cache_pos
                             + (int64_t)head_id * stride_cache_head;
    const int64_t sc_base = (int64_t)slot * stride_sc_slot
                          + (int64_t)head_id * stride_sc_head;

    const int64_t k_src_base = (int64_t)token_id * stride_kv_n + (int64_t)head_id * stride_kv_h;
    const int64_t v_src_base = (int64_t)token_id * stride_kv_n + (int64_t)head_id * stride_kv_h;

    // Shared memory for per-block absmax values
    // Max number of scale blocks: head_dim / min_scale_block. Head_dim up to 512, min_block >= 16 => max 32
    constexpr int MAX_SCALE_BLOCKS = 64;
    __shared__ float s_absmax[MAX_SCALE_BLOCKS];

    // ================================================================
    // Pass 1: Compute per-block scales for K
    // ================================================================
    {
        const int num_k_blocks = head_dim / k_scale_block;
        // Initialize shared memory
        for (int sb = tid; sb < num_k_blocks; sb += BLOCK_THREADS) {
            s_absmax[sb] = 0.0f;
        }
        __syncthreads();

        // Each thread processes multiple elements, tracking per-block absmax
        for (int d = tid; d < head_dim; d += BLOCK_THREADS) {
            float val = load_as_float(key, k_src_base + d);
            float abs_val = fabsf(val);
            int sb = d / k_scale_block;
            atomicMax(reinterpret_cast<int*>(&s_absmax[sb]),
                      __float_as_int(abs_val));
        }
        __syncthreads();

        // Store K scales
        for (int sb = tid; sb < num_k_blocks; sb += BLOCK_THREADS) {
            float absmax = s_absmax[sb];
            float scale = absmax / k_offset;
            int repeat = k_scale_block / scale_block_store;
            for (int r = 0; r < repeat; r++) {
                int sc_idx = sb * repeat + r;
                scales[sc_base + sc_idx * stride_sc_block + 0 * stride_sc_kv] = __float2half(scale);
            }
        }
        __syncthreads();
    }

    // ================================================================
    // Pass 1b: Compute per-block scales for V
    // ================================================================
    {
        const int num_v_blocks = head_dim / v_scale_block;
        for (int sb = tid; sb < num_v_blocks; sb += BLOCK_THREADS) {
            s_absmax[sb] = 0.0f;
        }
        __syncthreads();

        for (int d = tid; d < head_dim; d += BLOCK_THREADS) {
            float val = load_as_float(value, v_src_base + d);
            float abs_val = fabsf(val);
            int sb = d / v_scale_block;
            atomicMax(reinterpret_cast<int*>(&s_absmax[sb]),
                      __float_as_int(abs_val));
        }
        __syncthreads();

        // Store V scales
        for (int sb = tid; sb < num_v_blocks; sb += BLOCK_THREADS) {
            float absmax = s_absmax[sb];
            float scale = absmax / v_offset;
            int repeat = v_scale_block / scale_block_store;
            for (int r = 0; r < repeat; r++) {
                int sc_idx = sb * repeat + r;
                scales[sc_base + sc_idx * stride_sc_block + 1 * stride_sc_kv] = __float2half(scale);
            }
        }
        __syncthreads();
    }

    // ================================================================
    // Pass 2: Quantize + pack + store K
    // ================================================================
    if constexpr (K_BITS == 8) {
        // 8-bit: 1 byte per value, no packing needed
        for (int d = tid; d < head_dim; d += BLOCK_THREADS) {
            float val = load_as_float(key, k_src_base + d);
            int sc_idx = (d / k_scale_block) * (k_scale_block / scale_block_store);
            float sc = __half2float(scales[sc_base + sc_idx * stride_sc_block + 0 * stride_sc_kv]);
            float safe_sc = (sc > 1e-8f) ? sc : 1e-8f;
            float code_f = rintf(fminf(fmaxf(val / safe_sc + k_offset, 0.0f), (float)(255)));
            uint8_t code = (uint8_t)code_f;
            kv_cache[cache_base + 0 + d] = code;
        }
    } else if constexpr (K_BITS == 4) {
        // 4-bit: 2 values per byte (even in low nibble, odd in high nibble)
        const int half_d = head_dim / 2;
        for (int i = tid; i < half_d; i += BLOCK_THREADS) {
            int even_idx = i * 2;
            int odd_idx = i * 2 + 1;

            float val_even = load_as_float(key, k_src_base + even_idx);
            float val_odd = load_as_float(key, k_src_base + odd_idx);

            // Scale lookup (even and odd are in the same scale block since scale_block >= 16)
            int sc_idx = (even_idx / k_scale_block) * (k_scale_block / scale_block_store);
            float sc = __half2float(scales[sc_base + sc_idx * stride_sc_block + 0 * stride_sc_kv]);
            float safe_sc = (sc > 1e-8f) ? sc : 1e-8f;

            int code_even = (int)rintf(fminf(fmaxf(val_even / safe_sc + k_offset, 0.0f), 15.0f));
            int code_odd = (int)rintf(fminf(fmaxf(val_odd / safe_sc + k_offset, 0.0f), 15.0f));

            uint8_t packed = (uint8_t)(code_even | (code_odd << 4));
            kv_cache[cache_base + 0 + i] = packed;
        }
    } else if constexpr (K_BITS == 2) {
        // 2-bit: 4 values per byte
        const int quarter_d = head_dim / 4;
        for (int i = tid; i < quarter_d; i += BLOCK_THREADS) {
            int idx0 = i * 4;
            float v0 = load_as_float(key, k_src_base + idx0);
            float v1 = load_as_float(key, k_src_base + idx0 + 1);
            float v2 = load_as_float(key, k_src_base + idx0 + 2);
            float v3 = load_as_float(key, k_src_base + idx0 + 3);

            int sc_idx = (idx0 / k_scale_block) * (k_scale_block / scale_block_store);
            float sc = __half2float(scales[sc_base + sc_idx * stride_sc_block + 0 * stride_sc_kv]);
            float safe_sc = (sc > 1e-8f) ? sc : 1e-8f;

            int c0 = (int)rintf(fminf(fmaxf(v0 / safe_sc + k_offset, 0.0f), 3.0f));
            int c1 = (int)rintf(fminf(fmaxf(v1 / safe_sc + k_offset, 0.0f), 3.0f));
            int c2 = (int)rintf(fminf(fmaxf(v2 / safe_sc + k_offset, 0.0f), 3.0f));
            int c3 = (int)rintf(fminf(fmaxf(v3 / safe_sc + k_offset, 0.0f), 3.0f));

            uint8_t packed = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
            kv_cache[cache_base + 0 + i] = packed;
        }
    }

    // ================================================================
    // Pass 2b: Quantize + pack + store V
    // ================================================================
    if constexpr (V_BITS == 8) {
        for (int d = tid; d < head_dim; d += BLOCK_THREADS) {
            float val = load_as_float(value, v_src_base + d);
            int sc_idx = (d / v_scale_block) * (v_scale_block / scale_block_store);
            float sc = __half2float(scales[sc_base + sc_idx * stride_sc_block + 1 * stride_sc_kv]);
            float safe_sc = (sc > 1e-8f) ? sc : 1e-8f;
            float code_f = rintf(fminf(fmaxf(val / safe_sc + v_offset, 0.0f), 255.0f));
            uint8_t code = (uint8_t)code_f;
            kv_cache[cache_base + v_region_start + d] = code;
        }
    } else if constexpr (V_BITS == 4) {
        const int half_d = head_dim / 2;
        for (int i = tid; i < half_d; i += BLOCK_THREADS) {
            int even_idx = i * 2;
            int odd_idx = i * 2 + 1;

            float val_even = load_as_float(value, v_src_base + even_idx);
            float val_odd = load_as_float(value, v_src_base + odd_idx);

            int sc_idx = (even_idx / v_scale_block) * (v_scale_block / scale_block_store);
            float sc = __half2float(scales[sc_base + sc_idx * stride_sc_block + 1 * stride_sc_kv]);
            float safe_sc = (sc > 1e-8f) ? sc : 1e-8f;

            int code_even = (int)rintf(fminf(fmaxf(val_even / safe_sc + v_offset, 0.0f), 15.0f));
            int code_odd = (int)rintf(fminf(fmaxf(val_odd / safe_sc + v_offset, 0.0f), 15.0f));

            uint8_t packed = (uint8_t)(code_even | (code_odd << 4));
            kv_cache[cache_base + v_region_start + i] = packed;
        }
    } else if constexpr (V_BITS == 2) {
        const int quarter_d = head_dim / 4;
        for (int i = tid; i < quarter_d; i += BLOCK_THREADS) {
            int idx0 = i * 4;
            float v0 = load_as_float(value, v_src_base + idx0);
            float v1 = load_as_float(value, v_src_base + idx0 + 1);
            float v2 = load_as_float(value, v_src_base + idx0 + 2);
            float v3 = load_as_float(value, v_src_base + idx0 + 3);

            int sc_idx = (idx0 / v_scale_block) * (v_scale_block / scale_block_store);
            float sc = __half2float(scales[sc_base + sc_idx * stride_sc_block + 1 * stride_sc_kv]);
            float safe_sc = (sc > 1e-8f) ? sc : 1e-8f;

            int c0 = (int)rintf(fminf(fmaxf(v0 / safe_sc + v_offset, 0.0f), 3.0f));
            int c1 = (int)rintf(fminf(fmaxf(v1 / safe_sc + v_offset, 0.0f), 3.0f));
            int c2 = (int)rintf(fminf(fmaxf(v2 / safe_sc + v_offset, 0.0f), 3.0f));
            int c3 = (int)rintf(fminf(fmaxf(v3 / safe_sc + v_offset, 0.0f), 3.0f));

            uint8_t packed = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
            kv_cache[cache_base + v_region_start + i] = packed;
        }
    }
}

}  // namespace fusencache


// ============================================================
// C++ entry point — dispatches to the correct template instantiation
// ============================================================

void fusencache_store_kv(
    torch::Tensor const& key,           // [N, Hk, D] bf16 or fp16
    torch::Tensor const& value,         // [N, Hk, D] bf16 or fp16
    torch::Tensor& kv_cache,            // [num_blocks, block_size, Hk, slot_bytes] uint8
    torch::Tensor& scales,              // [max_slots, Hk, num_scale_blocks, 2] fp16
    torch::Tensor const& slot_mapping,  // [N] int32
    int64_t head_dim,
    int64_t page_size,
    int64_t k_bits,
    int64_t v_bits,
    int64_t k_scale_block,
    int64_t v_scale_block,
    double k_offset,
    double v_offset
) {
    const int N = key.size(0);
    const int Hk = key.size(1);

    if (N == 0) return;

    TORCH_CHECK(k_bits == 2 || k_bits == 4 || k_bits == 8,
                "k_bits must be 2, 4, or 8, got ", k_bits);
    TORCH_CHECK(v_bits == 2 || v_bits == 4 || v_bits == 8,
                "v_bits must be 2, 4, or 8, got ", v_bits);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int k_region_bytes = (int)(head_dim * k_bits / 8);
    int v_region_start = k_region_bytes;
    int scale_block_store = std::min((int)k_scale_block, (int)v_scale_block);

    constexpr int BLOCK_THREADS = 256;
    dim3 grid(N, Hk);

    // Dispatch on dtype and bit-width combination
    // We use a macro to reduce boilerplate for all combinations.

    #define LAUNCH_STORE_KERNEL(SRC_T, KB, VB) \
        fusencache::store_kv_kernel<SRC_T, KB, VB, BLOCK_THREADS> \
            <<<grid, BLOCK_THREADS, 0, stream>>>( \
                reinterpret_cast<const SRC_T*>(key.data_ptr()), \
                reinterpret_cast<const SRC_T*>(value.data_ptr()), \
                kv_cache.data_ptr<uint8_t>(), \
                reinterpret_cast<__half*>(scales.data_ptr()), \
                slot_mapping.data_ptr<int32_t>(), \
                key.stride(0), key.stride(1), \
                kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), \
                scales.stride(0), scales.stride(1), scales.stride(2), scales.stride(3), \
                (int)head_dim, (int)page_size, \
                (int)k_scale_block, (int)v_scale_block, scale_block_store, \
                (float)k_offset, (float)v_offset, \
                k_region_bytes, v_region_start \
            )

    bool is_bf16 = (key.scalar_type() == at::ScalarType::BFloat16);

    // Dispatch on dtype x k_bits x v_bits
    if (is_bf16) {
        if (k_bits == 4 && v_bits == 4) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 4, 4); }
        else if (k_bits == 8 && v_bits == 8) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 8, 8); }
        else if (k_bits == 8 && v_bits == 4) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 8, 4); }
        else if (k_bits == 4 && v_bits == 8) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 4, 8); }
        else if (k_bits == 2 && v_bits == 2) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 2, 2); }
        else if (k_bits == 4 && v_bits == 2) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 4, 2); }
        else if (k_bits == 2 && v_bits == 4) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 2, 4); }
        else if (k_bits == 8 && v_bits == 2) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 8, 2); }
        else if (k_bits == 2 && v_bits == 8) { LAUNCH_STORE_KERNEL(__nv_bfloat16, 2, 8); }
        else { TORCH_CHECK(false, "Unsupported k_bits/v_bits combination"); }
    } else {
        // FP16
        if (k_bits == 4 && v_bits == 4) { LAUNCH_STORE_KERNEL(__half, 4, 4); }
        else if (k_bits == 8 && v_bits == 8) { LAUNCH_STORE_KERNEL(__half, 8, 8); }
        else if (k_bits == 8 && v_bits == 4) { LAUNCH_STORE_KERNEL(__half, 8, 4); }
        else if (k_bits == 4 && v_bits == 8) { LAUNCH_STORE_KERNEL(__half, 4, 8); }
        else if (k_bits == 2 && v_bits == 2) { LAUNCH_STORE_KERNEL(__half, 2, 2); }
        else if (k_bits == 4 && v_bits == 2) { LAUNCH_STORE_KERNEL(__half, 4, 2); }
        else if (k_bits == 2 && v_bits == 4) { LAUNCH_STORE_KERNEL(__half, 2, 4); }
        else if (k_bits == 8 && v_bits == 2) { LAUNCH_STORE_KERNEL(__half, 8, 2); }
        else if (k_bits == 2 && v_bits == 8) { LAUNCH_STORE_KERNEL(__half, 2, 8); }
        else { TORCH_CHECK(false, "Unsupported k_bits/v_bits combination"); }
    }

    #undef LAUNCH_STORE_KERNEL
}
