# T2-N Polish — Findings

Date: 2026-04-17
Scope: Production-harden the T2-N fused shuffle+quant vLLM integration. Remove the two workarounds (scale buffer shape + `VLLM_USE_FLASHINFER_MOE_FP4=0`) that block deployment.

## 1. Scale-buffer reshape (implemented)

### Problem
`fused_shuffle_quant_sm120a.so` allocates a scale buffer shaped
`[num_experts * 320, padded_k_int32]` int32 (int32 view of fp8_e4m3fn). vLLM's
`cutlass_fp4_moe_mm` expects `[VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE * topk, padded_k_int32]`
(≈ `[163,840 * topk, padded_k_int32]`). The prior code relied on the fact that
`blockscale_offsets[e] + (te-ts)` happened to stay inside the small buffer, which is
fragile for large batches / large topk / future MAX_TPE tuning.

### Fix
In `fused_shuffle_and_quant_moe()` (`patches/fused_shuffle_quant_wrapper.py`) after the
`.so` call we now:

1. Read `VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE` from `vllm.envs`.
2. Allocate `full_scales = torch.zeros([MAX_TPE * topk, padded_k_int32], dtype=int32)`.
3. Bulk-copy the small buffer: `full_scales[:small_rows].copy_(scales_i32)`.
4. Return the fp8_e4m3fn view of `full_scales` as `rep_a_blockscale`.

### Why a flat copy works
The CUTLASS 128×4 swizzle maps every `(row_in_buf, block_idx)` to a fixed flat byte offset
via `((mTileIdx * numKTiles + kTileIdx) << 9) | (outerMIdx << 4) | ...`. That offset is
independent of total buffer size — `numKTiles` depends only on `K`, which is identical
between the small and large buffers. So a byte-accurate copy of the first
`small_rows * padded_k_int32 * 4` bytes into the start of the large buffer places every
written scale at the identical offset the consumer will read from, with padding rows
(init-zero) occupying the unused tail.

### Cost
- Allocation of `[MAX_TPE * topk, padded_k_int32]` int32 buffer per MoE call. At
  MAX_TPE=163,840, topk=8, K=2048 (Qwen3-30B-A3B → padded_k_int32=32) the buffer is
  163,840 * 8 * 32 * 4 B ≈ **163 MB**. After warmup this is served from the CUDA caching
  allocator as a same-size slab reuse — negligible alloc overhead but **memset of 163 MB**
  on each call at ~3 TB/s HBM is ~54 µs.
- Copy of the small buffer (~5 MB for `num_experts=128, 320 rows, padded_k=32`) adds ~2 µs.

**Total reshape overhead per MoE forward: ~55 µs/call** (memset-dominated).
Qwen3-30B-A3B has 48 MoE layers × 2 GEMMs = 96 MoE GEMMs per forward. But the reshape is
only on the `rep_a_blockscale` produced by the fused kernel (GEMM1 entry) — 48× per token.
For decode (1 token) that is ~2.6 ms of pure memset per forward → non-trivial.

**Mitigation option (not implemented in this pass):** promote the zero-init target to a
module-level persistent tensor and `torch.cuda.empty_cache()`-resistant allocator slab;
then on each call only memcpy the small-buffer rows and **not** zero-init the tail (the
tail rows are never read because `blockscale_offsets[e] + tokens_in_expert` never exceeds
small_rows). That removes the 54 µs memset; only the 2 µs row copy remains. See
"Follow-up" below.

## 2. FLASHINFER_CUTLASS MoE path — can it be patched?

### Short answer: no, not via the same monkey-patch.

### Analysis
`vllm/model_executor/layers/fused_moe/flashinfer_cutlass_moe.py` `FlashInferExperts.apply()`
calls `flashinfer.fused_moe.cutlass_fused_moe(...)` (lazy-imported wrapper in
`vllm/utils/flashinfer.py`).

`flashinfer.fused_moe.core.cutlass_fused_moe()` signature (truncated):

    def cutlass_fused_moe(
        input: torch.Tensor,              # [num_tokens, hidden] bf16 / nvfp4 / fp8
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,  # as torch.long for nvfp4
        fc2_expert_weights: torch.Tensor,
        output_dtype: torch.dtype,
        quant_scales: list[torch.Tensor],  # [a1_gscale, w1_scale, g1_alphas,
                                           #  a2_gscale, w2_scale, g2_alphas] for nvfp4
        input_sf: Optional[torch.Tensor] = None,
        ...
    ):

This is a **monolithic CUTLASS-scheduled MoE** that ingests *unquantized bf16* hidden
states and performs, internally in CUDA:
permute → quantize (→ nvfp4 + per-block scales) → GEMM1 → activation → quantize →
GEMM2 → unpermute → reduce.

There is **no exposed shuffle_rows / scaled_fp4_experts_quant boundary** to hook into.
The `input_sf` argument exists, but for the default nvfp4 path `a1q_scale` is set up-stream
by the modular kernel's prepare step (not the shuffle/quant pipeline our kernel replaces).
Replacing the whole pipeline would require re-implementing the full
CUTLASS-NVFP4-MoE-GEMM — out of scope and almost certainly slower than the tuned
vendor kernel.

### Consequence
As long as vLLM routes Qwen3 NVFP4 through the FLASHINFER_CUTLASS backend
(`_supports_current_device` returns true for SM120 & has_flashinfer_cutlass_fused_moe()),
our `run_cutlass_moe_fp4` monkey-patch is **inert**. The `VLLM_USE_FLASHINFER_MOE_FP4=0`
env-var is required to force vLLM onto the VLLM_CUTLASS backend (which still has the
exposed shuffle+quant pair) where our patch operates.

### Recommendation
- Keep `VLLM_USE_FLASHINFER_MOE_FP4=0` in the launch script as a **required production
  flag** when deploying the T2-N fused kernel on SM120 for models routed through
  FLASHINFER_CUTLASS by default.
- File an upstream proposal (FlashInfer repo): expose a callback or allow users to supply
  pre-quantized `(fp4, blockscale)` directly to `cutlass_fused_moe` so external fused
  kernels can compete with the monolithic path.

## 3. End-to-end rebench (pending — GPU 1)

See `results.tsv` row `T2N_polish_noenv`.

## Follow-up work (not in budget for this pass)

1. **Persistent large-buffer slab.** Allocate `full_scales` once at layer-init (or lazily
   cached per shape signature) and reuse across forwards — eliminates the 54 µs memset per
   call. Requires ensuring `torch.compile`/cudagraph capture doesn't bind the pointer.
2. **.so-level large-buffer allocation.** Pass `MAX_TPE` from Python into the kernel
   launcher (via `torch.ops` or a C++ export) and have the CUDA side allocate the
   vLLM-standard shape directly — removes the Python copy entirely.
3. **Upstream FlashInfer fused-quant callback.** Biggest potential payoff — would let the
   fused kernel participate in the default NVFP4 MoE path without the env-var.
