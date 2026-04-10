# CUDA SM120 (RTX 5090 Blackwell) Feature Audit

**Date:** 2026-04-09
**Hardware:** NVIDIA GeForce RTX 5090 (SM120, 170 SMs, 32 GB VRAM)
**Software:** CUDA 12.8 (nvcc) / PyTorch 2.11+cu130 / CUTLASS 4.4.2 / Triton 3.6.0
**Container:** vLLM 0.19.1rc1 with Gemma 4 26B NVFP4

## Hardware Specs (SM120)

| Property | Value |
|----------|-------|
| L2 Cache | 96 MB |
| Max Persisting L2 | 60 MB (set to 48 MB currently) |
| Shared Memory per Block | 48 KB |
| Shared Memory per SM (opt-in) | 99 KB |
| SMs | 170 |
| Max Threads per SM | 1536 |
| Warp Size | 32 |
| cuDNN | 9.19.0 |
| Stream Priority Range | -5 to 0 |

## Feature-by-Feature Audit

### 1. TMA (Tensor Memory Accelerator) -- PARTIALLY USED

**Status:** Used by CUTLASS kernels internally (NVFP4 GEMM, FP8 GEMM, MoE), NOT used by custom Triton kernels.

**What's available:**
- Triton 3.6 exposes `tl.make_tensor_descriptor`, `tl.load_tensor_descriptor`, `tl.store_tensor_descriptor`
- CUTLASS 4.4.2 uses SM90_TMA_LOAD / SM90_TMA_LOAD_MULTICAST internally for its mainloop
- The vLLM Machete kernel (quantized GEMM) uses TMA extensively

**What's NOT being exploited:**
- Custom Triton kernels (e.g., our autokernel experiments) do not use tensor descriptors
- FlashInfer's SM120 MLA kernel (`mla_sm120.cu`) uses TMA via CUtensorMap for KV cache loading -- this IS active
- Flash attention on SM120 uses CpAsync (SM80-era), NOT TMA

**Expected impact:** Moderate for custom kernels. TMA eliminates address generation overhead and enables hardware-managed async copies. For attention kernels with large KV cache loads, 10-20% improvement possible.

**Effort:** Medium. Requires rewriting Triton kernels to use tensor descriptors, or using CUTLASS collective builders.

---

### 2. Cluster Launch (Multi-SM Cooperative Kernels) -- PARTIALLY USED

**Status:** Used by CUTLASS MoE kernels (MXFP8 grouped GEMM with cluster shapes like (1,4,1)). NOT used by attention or custom kernels.

**What's available:**
- CUTLASS MoE grouped MM uses `hw_info.cluster_shape = (1, 4, 1)` with fallback `(1, 2, 1)`
- SM120 NVFP4 GEMM uses ClusterShape `(1, 1, 1)` -- single-SM only, NOT exploiting clusters

**What's NOT being exploited:**
- NVFP4 GEMM `sm120_fp4_config_default` and `sm120_fp4_config_M256` both use ClusterShape `Shape<_1, _1, _1>` -- no multi-SM cooperation
- Flash attention SM120 does not use clusters (uses SM80 code path)
- No Triton kernel uses cluster launch on SM120

**Expected impact:** 
- For NVFP4 GEMM: potentially 15-30% for large M shapes by enabling multicast TMA across cluster
- For attention: would require major rewrite, likely best left to FlashInfer's XQA kernel

**Effort:** For NVFP4 GEMM, change ClusterShape in CUTLASS config (requires vLLM source mod). For Triton, not yet supported in Triton 3.6 for SM120.

---

### 3. FP4 Native Instructions -- FULLY USED (for GEMM)

**Status:** SM120 block-scaled FP4 tensor ops are fully used via CUTLASS `OpClassBlockScaledTensorOp` with `cutlass::arch::Sm120`.

**What's available:**
- `nv_float4_t<float_e2m1_t>` native FP4 type
- Block-scaled tensor core ops (128x128x128 tile)
- Scale factors in `float_ue4m3_t` format
- Both dense GEMM and grouped MoE GEMM supported

**What's used:**
- `nvfp4_scaled_mm_sm120_kernels.cu`: Dense FP4 GEMM with 2 tile configs (M256 and default)
- `nvfp4_blockwise_moe_kernel.cu`: MoE FP4 grouped GEMM
- FlashInfer `fp4_gemm_cutlass_sm120.cu`: Alternative FP4 GEMM runner

**Gaps:**
- Only 2 tile configs for dense GEMM (128x128x128 and 256x128x128). Missing configs for small M (decode, M=1-16)
- ClusterShape is (1,1,1) everywhere -- not exploiting multi-SM FP4

---

### 4. Grid Dependency Control / PDL (Programmatic Dependent Launch) -- PARTIALLY USED

**Status:** Used in `grouped_topk_kernels.cu` for MoE routing. NOT widely used elsewhere.

**What's available:**
- `cudaGridDependencySynchronize()` / `cudaTriggerProgrammaticLaunchCompletion()` (SM >= 900)
- `cudaLaunchAttributeProgrammaticStreamSerialization` for launch config
- Enables overlap of dependent kernel launches -- producer kernel signals completion before fully finishing

**What's used:**
- MoE top-k routing kernel uses PDL with `enable_pdl` parameter
- Flash attention backward (SM90) uses PDL
- LoRA ops have PDL support

**What's NOT being exploited:**
- Flash attention forward on SM120 does NOT use PDL
- NVFP4 GEMM kernels do NOT use PDL
- No overlap between attention decode and MoE dispatch

**Expected impact:** 5-15% reduction in kernel launch latency for dependent kernel chains. Critical for decode where many small kernels run sequentially.

**Effort:** Low for adding PDL to existing CUTLASS kernel launches. Medium for Triton kernels.

---

### 5. DeepGemm -- NOT AVAILABLE ON SM120

**Status:** NOT supported. `support_deep_gemm()` returns False for SM120.

**Root cause:** `vllm/platforms/cuda.py` line 543:
```python
def support_deep_gemm(cls) -> bool:
    return cls.is_device_capability(90) or cls.is_device_capability_family(100)
```
SM120 is NOT in the SM100 family (`is_device_capability_family(100)` returns False).

**Impact:** DeepGemm provides highly optimized FP8 block-scaled GEMM kernels used for MoE. Without it, SM120 falls back to CUTLASS or Triton FP8 kernels.

**Effort:** Low -- likely just needs `or cls.is_device_capability_family(120)` added to the platform check, BUT deep_gemm itself is not installed in this container. Would need to verify deep_gemm has SM120 support in its JIT compilation.

---

### 6. Flash Attention -- USING SM80 CODE PATH

**Status:** SM120 flash attention uses SM80-era MMA instructions (`mma.sync.aligned.m16n8k16`), NOT the newer SM90 warp-group MMA or SM100/SM120 features.

**What's happening:**
- `FlashAttentionForwardSm120` subclasses `FlashAttentionForwardSm80` with `arch = 80`
- Uses CpAsync (not TMA) for memory operations
- Has REDUCED shared memory: 99 KB vs 163 KB (SM80) vs 227 KB (SM90)
- Tile sizes are constrained: D>64 uses 128x64 (not 128x128) due to SMEM pressure
- Does NOT support paged KV, block sparsity, or split-KV on SM120

**Why:** SM120 (GeForce) lacks the warp-group execution model of SM90/SM100 (data center GPUs). It's a "Blackwell lite" -- same generation but reduced feature set.

**Impact:** Attention throughput is significantly lower than it could be. The SM120 flash attention kernel cannot compete with SM90's 2-CTA warpgroup kernels.

**Mitigation:** FlashInfer's XQA MLA kernel (`mla_sm120.cu`) IS available and uses SM120-native features for MLA decode. This is the preferred attention path for Gemma 4.

---

### 7. L2 Cache Persistence API -- NOT USED

**Status:** API works but is not used by any vLLM component.

**Verified working:**
- `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 48MB)` -- succeeds
- Max persisting L2: 60 MB out of 96 MB total
- No `cudaAccessPolicyWindow` or `cuCtxSetLimit` calls found in vLLM source

**Opportunity:**
- Pin KV cache pages in L2 for decode (hot data stays in 96 MB L2)
- Pin MoE routing tables / expert weights in L2
- Pin attention softmax intermediate results

**Expected impact:** 5-20% for decode-bound workloads where L2 hit rate is critical. RTX 5090's 96 MB L2 can hold significant working sets.

**Effort:** Medium. Requires C++ extension or ctypes wrapper to set access policies per-stream. Need to identify optimal allocation of persisting lines between KV cache and weights.

---

### 8. FlashInfer XQA MLA SM120 Kernel -- AVAILABLE, MAY NOT BE ACTIVE

**Status:** FlashInfer 0.6.6 has a dedicated SM120 MLA kernel (`mla_sm120.cu`) but it requires JIT compilation on first use.

**Features of XQA MLA SM120:**
- Uses TMA for KV cache tile loading
- FP8 tensor core math (`float_e4m3` precision)
- Multi-block support with CGA (Cooperative Grid Array) -- `nbProducerCtasPerCga = 2`
- 232 registers per math warp, 32 registers per IO warp (aggressive register allocation)
- K-head partitioning: 64-element K parts, 128-element V parts
- Supports paged KV cache

**Constraints:**
- Only supports `q_len_per_request == 1` (decode only)
- Only FP8 operation (no FP16/BF16 compute)
- No attention sinks support
- `cvt_rs` (register-to-shared conversion) NOT supported on SM120

**Expected impact:** This kernel should be significantly faster than the SM80-fallback attention for MLA decode. If not already active for Gemma 4 serving, enabling it is the single highest-impact change.

**Effort:** Low if JIT compilation works. Check if `FLASHINFER_ENABLE_XQA=1` or similar env var is needed.

---

### 9. CUTLASS SM120 Blockwise FP8 GEMM -- AVAILABLE AND USED

**Status:** `scaled_mm_c3x_sm120.cu` provides FP8 GEMM with blockwise scaling for SM120. Compiled and active.

**Details:**
- Supports FP8 (e4m3/e5m2) with per-tensor and per-block scaling
- INT8 NOT supported on SM120
- Uses CUTLASS 3.x collective builder with `cutlass::arch::Sm120`

**Gap:** Only FP8, no INT8. This matches hardware -- SM120 doesn't have INT8 tensor core fast-path like SM89.

---

### 10. Triton 3.6 SM120 Support -- BASIC

**Status:** Triton 3.6 compiles for SM120 (`sm_120` in arch list) but without SM120-specific optimizations.

**Available:**
- `tl.tensor_descriptor` / `tl.make_tensor_descriptor` -- TMA descriptor support
- `tl.load_tensor_descriptor` / `tl.store_tensor_descriptor` -- TMA load/store
- Standard Triton kernels compile and run

**NOT available in Triton 3.6 for SM120:**
- No cluster launch support (no `num_ctas` grid dimension)
- No warp-group scheduling
- No native FP4 Triton instructions (must use CUTLASS for FP4)
- No programmatic dependent launch from Triton

**Expected impact:** Triton kernels on SM120 miss out on ~20-30% of the potential performance vs hand-written CUDA.

---

## Summary: Priority Actions

### HIGH PRIORITY (Likely 10-30% impact each)

1. **Verify FlashInfer XQA MLA SM120 is active for decode.** If Gemma 4 is using the SM80-fallback flash attention instead of XQA MLA, switching would be the single biggest win. Check `VLLM_ATTENTION_BACKEND` and FlashInfer's MLA decode path.

2. **L2 Cache Persistence for KV Cache.** Set `cudaLimitPersistingL2CacheSize` to 48-60 MB and configure access policies for KV cache pages. 96 MB L2 on RTX 5090 is massive -- use it.

3. **NVFP4 GEMM cluster shapes.** The SM120 NVFP4 GEMM uses ClusterShape (1,1,1). Testing (1,2,1) or (1,4,1) could enable TMA multicast and improve throughput for prefill.

### MEDIUM PRIORITY (5-15% impact)

4. **PDL for kernel chains.** Enable programmatic dependent launch between attention -> MoE -> linear chains in decode. Already proven in grouped_topk.

5. **Deep GEMM for SM120.** Add SM120 to `support_deep_gemm()` platform check. Install deep_gemm package. Could provide better FP8 MoE performance if the JIT supports SM120.

6. **NVFP4 GEMM tile configs for small M.** Current configs target M >= 128. Decode has M=1-8. Need small-M tile configs (e.g., 16x128x128 or 64x128x128).

### LOW PRIORITY (Marginal or high effort)

7. **TMA in custom Triton kernels.** Rewrite kernels to use tensor descriptors. High effort, moderate gain.

8. **Flash attention SM120 improvements.** Waiting for upstream (FlashAttention / vLLM) to add TMA/warpgroup support for SM120. Currently blocked by hardware limitations (SM120 lacks SM90 warp-group execution model).

## Key Architectural Insight

SM120 (RTX 5090) is NOT a full Blackwell chip. It lacks:
- Warp-group execution model (SM90a/SM100a feature)
- `cvt.rs` instruction (register-to-shared fast path)
- Full SM100 MMA capabilities

It IS a powerful chip with:
- 96 MB L2 cache (3x RTX 4090)
- Native FP4 tensor cores (block-scaled)
- TMA support
- Cluster launch (limited configurations)
- 170 SMs (vs 128 on RTX 4090)
- 99 KB shared memory per SM opt-in

The optimal strategy is to maximize CUTLASS SM120-native paths (NVFP4 GEMM, FP8 blockwise GEMM) and FlashInfer XQA MLA, while using the massive L2 cache for decode-bound workloads.
