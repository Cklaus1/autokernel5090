// expert_prefetch_test.cu
//
// Standalone CUDA test for L2-resident expert prefetch.
//
// Simulates 2 MoE layers (Gemma4 26B NVFP4 shape):
//   - E=128 experts, top_k=8 activated per layer.
//   - Per-expert weight block: 2.0 MiB (NVFP4 packed).
//
// Three scenarios timed on GPU 1:
//   A) BASELINE:   cold run, no prefetch
//   B) PREFETCH:   warmup kernel issues L2-landing reads over predicted
//                  top-16 experts on a second CUDA stream, overlapped with
//                  layer-N compute, just before layer-N+1 consumes its experts.
//   C) ORACLE:     prefetch exactly the top-8 that will be used (upper bound).

#include <cuda_runtime.h>
#include <cuda.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <cstdint>
#include <cstdio>

// ---- Multi-expert warmup ("prefetch") kernel ----
// Single-launch warmup: reads a LIST of experts and writes their XOR to a
// sink buffer. Brings weight bytes into L2. grid.y indexes the expert.
__global__ void warmup_list_kernel(const int8_t* __restrict__ W,
                                    int64_t bytes_per_expert,
                                    int num_experts,
                                    const int32_t* __restrict__ ids,
                                    int32_t* __restrict__ sink) {
    const int eidx = blockIdx.y;
    if (eidx >= num_experts) return;
    const int expert = ids[eidx];
    const int8_t* p = W + (int64_t)expert * bytes_per_expert;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    int32_t acc = 0;
    // 16B per thread per step (int4 load), which keeps us BW-bound.
    for (int64_t i = (int64_t)tid * 16; i < bytes_per_expert;
         i += (int64_t)stride * 16) {
        int4 v = *reinterpret_cast<const int4*>(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // one non-predicated store per block to prevent DCE
    if (threadIdx.x == 0) sink[blockIdx.x + eidx * gridDim.x] = acc;
}

// Fake "MoE GEMM" kernel: bandwidth-bound copy with XOR to simulate the
// streaming weight load behavior of a grouped GEMM on activated experts.
__global__ void fake_moe_gemm_kernel(const int8_t* __restrict__ W,
                                      int8_t* __restrict__ out,
                                      int64_t bytes_per_expert,
                                      int num_active,
                                      const int32_t* __restrict__ active_idx) {
    const int eid = blockIdx.y;
    if (eid >= num_active) return;
    const int expert = active_idx[eid];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int8_t* Wp = W + (int64_t)expert * bytes_per_expert;
    int8_t* op = out + (int64_t)eid * bytes_per_expert;
    for (int64_t i = (int64_t)tid * 16; i < bytes_per_expert;
         i += (int64_t)stride * 16) {
        int4 v = *reinterpret_cast<const int4*>(Wp + i);
        v.x ^= 0x5a5a5a5a;
        *reinterpret_cast<int4*>(op + i) = v;
    }
}

// ---- Host wrappers ----

void launch_warmup_list(const int8_t* W, int64_t bytes_per_expert,
                        int num_experts, const int32_t* ids,
                        int32_t* sink, cudaStream_t stream) {
    dim3 grid(64, num_experts), block(256);
    warmup_list_kernel<<<grid, block, 0, stream>>>(
        W, bytes_per_expert, num_experts, ids, sink);
}

void launch_fake_moe(const int8_t* W, int8_t* out,
                     int64_t bytes_per_expert, int num_active,
                     const int32_t* active_idx, cudaStream_t stream) {
    dim3 grid(64, num_active), block(256);
    fake_moe_gemm_kernel<<<grid, block, 0, stream>>>(
        W, out, bytes_per_expert, num_active, active_idx);
}

// ---- PyBind interface ----

// Run one scenario and return latency in microseconds.
// mode: 0=baseline, 1=prefetch(predicted), 2=oracle(layer1 exact)
double run_scenario(torch::Tensor W,
                    torch::Tensor out,
                    int64_t bytes_per_expert,
                    torch::Tensor active_layer0,   // int32 [top_k0]
                    torch::Tensor active_layer1,   // int32 [top_k1]
                    torch::Tensor predicted,       // int32 [top_p]
                    torch::Tensor flush_ids,       // int32 [n_flush] cold experts for L2 flush
                    torch::Tensor sink,
                    int64_t mode,
                    int64_t iters) {
    TORCH_CHECK(W.is_cuda() && W.dtype() == torch::kInt8, "W int8 cuda");
    TORCH_CHECK(active_layer0.dtype() == torch::kInt32 && active_layer0.is_cuda(),
                "active_layer0 int32 cuda");
    TORCH_CHECK(active_layer1.dtype() == torch::kInt32 && active_layer1.is_cuda(),
                "active_layer1 int32 cuda");

    const int8_t* Wp = W.data_ptr<int8_t>();
    int8_t* outp = out.data_ptr<int8_t>();
    const int32_t* act0 = active_layer0.data_ptr<int32_t>();
    const int32_t* act1 = active_layer1.data_ptr<int32_t>();
    const int32_t* pred = predicted.data_ptr<int32_t>();
    const int32_t* flushp = flush_ids.data_ptr<int32_t>();
    int32_t* sinkp = sink.data_ptr<int32_t>();

    const int topk0 = active_layer0.size(0);
    const int topk1 = active_layer1.size(0);
    const int topp = predicted.size(0);
    const int nflush = flush_ids.size(0);

    cudaStream_t compute_stream = at::cuda::getCurrentCUDAStream();
    // Persistent prefetch stream, created once. Re-created across calls but
    // the destroy is deferred by wrapping in a static-local.
    static cudaStream_t prefetch_stream = nullptr;
    if (prefetch_stream == nullptr) {
        cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking);
    }

    cudaEvent_t start, stop, pfdone;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreateWithFlags(&pfdone, cudaEventDisableTiming);

    // Warmup JIT paths + stream pipeline. Run the EXACT sequence several times
    // so the graph/launch caches settle before timed iters.
    for (int w = 0; w < 5; ++w) {
        launch_warmup_list(Wp, bytes_per_expert, nflush, flushp,
                           sinkp, compute_stream);
        cudaStreamSynchronize(compute_stream);
        launch_fake_moe(Wp, outp, bytes_per_expert, topk0, act0, compute_stream);
        if (mode == 1 && topp > 0) {
            launch_warmup_list(Wp, bytes_per_expert, topp, pred, sinkp,
                               prefetch_stream);
        } else if (mode == 2) {
            launch_warmup_list(Wp, bytes_per_expert, topk1, act1, sinkp,
                               prefetch_stream);
        }
        if (mode != 0) {
            cudaEventRecord(pfdone, prefetch_stream);
            cudaStreamWaitEvent(compute_stream, pfdone, 0);
        }
        launch_fake_moe(Wp, outp, bytes_per_expert, topk1, act1, compute_stream);
        cudaStreamSynchronize(compute_stream);
        cudaStreamSynchronize(prefetch_stream);
    }

    double total_ms = 0.0;
    for (int64_t it = 0; it < iters; ++it) {
        // Flush L2: read N cold experts (N * 2 MB > L2 = 128 MB).
        // flush_ids point to experts NOT in layer0/layer1/predicted.
        launch_warmup_list(Wp, bytes_per_expert, nflush, flushp,
                           sinkp, compute_stream);
        cudaStreamSynchronize(compute_stream);

        cudaEventRecord(start, compute_stream);

        // Layer 0 compute
        launch_fake_moe(Wp, outp, bytes_per_expert, topk0, act0, compute_stream);

        // Overlapped prefetch on prefetch_stream (for mode 1/2)
        if (mode == 1 && topp > 0) {
            launch_warmup_list(Wp, bytes_per_expert, topp, pred, sinkp,
                               prefetch_stream);
        } else if (mode == 2) {
            launch_warmup_list(Wp, bytes_per_expert, topk1, act1, sinkp,
                               prefetch_stream);
        }

        if (mode != 0) {
            cudaEventRecord(pfdone, prefetch_stream);
            cudaStreamWaitEvent(compute_stream, pfdone, 0);
        }

        // Layer 1 compute
        launch_fake_moe(Wp, outp, bytes_per_expert, topk1, act1, compute_stream);

        cudaEventRecord(stop, compute_stream);
        cudaEventSynchronize(stop);

        float ms = 0.f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(pfdone);
    // prefetch_stream retained across calls

    return (total_ms / (double)iters) * 1e3;   // μs
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_scenario", &run_scenario,
          "Run prefetch-test scenario; returns avg μs",
          py::arg("W"),
          py::arg("out"),
          py::arg("bytes_per_expert"),
          py::arg("active_layer0"),
          py::arg("active_layer1"),
          py::arg("predicted"),
          py::arg("flush_ids"),
          py::arg("sink"),
          py::arg("mode"),
          py::arg("iters"));
}
